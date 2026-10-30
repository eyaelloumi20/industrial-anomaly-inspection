"""
retrain_thin.py
================
OPTION 2 — Retrain a thinner Student + AutoEncoder (narrower hidden
channel widths, controlled by --width_mult) against the frozen, full-size
Teacher, to speed up inference.

This mirrors retrain_efficientad.py / retrain_lowres.py exactly (frozen
teacher, quantile calibration on the "test" split, AUROC monitoring on
"validation"), but swaps in Student_Thin / AutoEncoder_Thin from
models_thin.py instead of the full-width Student / AutoEncoder.

Only the Student and AutoEncoder get thinner — the Teacher keeps its
original architecture and weights unchanged, since the loss compares
against its exact 384-channel output.

Usage
-----
    # Baseline sanity check: width_mult=1.0 should behave ~identically to
    # the original retrain_efficientad.py (same architecture, different
    # class names)
    python retrain_thin.py \
        --dataset_path /path/to/cropped_dataset \
        --ckpt_dir     ./ckpt_thin_100 \
        --teacher_path ./ckptSmall/best_teacher.pth \
        --category     chewinggum \
        --width_mult   1.0

    # Try a thinner network
    python retrain_thin.py \
        --dataset_path /path/to/cropped_dataset \
        --ckpt_dir     ./ckpt_thin_050 \
        --teacher_path ./ckptSmall/best_teacher.pth \
        --category     chewinggum \
        --width_mult   0.5

Compare AUROC across width_mult in {1.0, 0.75, 0.5, 0.25} and pick the
smallest one that still meets your accuracy bar.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.models import Teacher
from models_thin import Student_Thin, AutoEncoder_Thin, count_params
from utils.data_loader import get_AD_dataset, load_infinite, ImageNetDataset

IMAGE_SIZE   = 256
OUT_CHANNELS = 384

torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path',  required=True)
    p.add_argument('--ckpt_dir',      required=True)
    p.add_argument('--teacher_path',  required=True,
                   help='Path to best_teacher.pth (kept frozen, full-size architecture).')
    p.add_argument('--imagenet_dir',  default="/home/ahmed/Downloads/imagenet/dtd/images")
    p.add_argument('--no_imagenet',   action='store_true')
    p.add_argument('--category',      default='chewinggum')
    p.add_argument('--model_size',    default='S', choices=['S', 'M'])
    p.add_argument('--width_mult',    type=float, default=0.5,
                   help='Scales hidden channel widths of Student/AE. 1.0 = original '
                        'width, 0.5 = half, etc. Output channel counts vs. Teacher '
                        'are always preserved.')
    p.add_argument('--iterations',    type=int, default=70000)
    p.add_argument('--batch_size',    type=int, default=1)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-5)
    p.add_argument('--print_freq',    type=int, default=200)
    p.add_argument('--device',        default='cuda')
    p.add_argument('--seed',          type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ThinTrainer:
    def __init__(self, args):
        self.args       = args
        self.device     = args.device
        self.category   = args.category
        self.ckpt_dir   = args.ckpt_dir
        self.width_mult = args.width_mult
        os.makedirs(self.ckpt_dir, exist_ok=True)
        set_seed(args.seed)

        self.data_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.gt_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.imagenet_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomGrayscale(p=0.3),
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

        print(f'[Thin] Building PDN-{args.model_size} teacher (full width, frozen) '
              f'and width_mult={self.width_mult} Student/AutoEncoder ...')
        self.teacher = Teacher(args.model_size)
        self.student = Student_Thin(args.model_size, width_mult=self.width_mult)
        self.ae      = AutoEncoder_Thin(width_mult=self.width_mult)

        print(f'[Thin] Student params : {count_params(self.student):,}')
        print(f'[Thin] AE params      : {count_params(self.ae):,}')

        print(f'[Thin] Loading frozen teacher: {args.teacher_path}')
        self.teacher.load_state_dict(
            torch.load(args.teacher_path, map_location=self.device, weights_only=False))
        self.teacher.eval().to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student.to(self.device)
        self.ae.to(self.device)

        self.channel_mean = None
        self.channel_std  = None

    def compute_channel_stats(self, dataloader, n_samples=500):
        print('[Thin] Computing teacher channel mean/std on crops ...')
        self.teacher.eval()
        feats, n = [], 0
        for sample in tqdm(dataloader, desc='  Channel stats'):
            if n >= n_samples:
                break
            img = sample['image'].to(self.device)
            with torch.no_grad():
                f = self.teacher(img).detach().cpu()
            feats.append(f)
            n += f.shape[0]
        feats = torch.cat(feats, dim=0)[:n_samples]
        mean = feats.mean(dim=(0, 2, 3), keepdim=True).to(self.device)
        std  = feats.std(dim=(0, 2, 3),  keepdim=True).to(self.device)
        self.channel_mean = mean
        self.channel_std  = std

    def loss_st(self, image, imagenet_iter):
        with torch.no_grad():
            t_out  = self.teacher(image)
            norm_t = (t_out - self.channel_mean) / (self.channel_std + 1e-8)

        s_out = self.student(image)
        y_st  = s_out[:, :OUT_CHANNELS, :, :]
        dist  = torch.pow(norm_t - y_st, 2)

        dhard  = torch.quantile(dist[:8], 0.999)
        hard   = dist[dist >= dhard]
        L_hard = torch.mean(hard)

        if imagenet_iter is not None:
            img_p = next(imagenet_iter)
            if isinstance(img_p, (list, tuple)):
                img_p = img_p[0]
            s_inet = self.student(img_p.to(self.device))
            N = torch.mean(torch.pow(s_inet[:, :OUT_CHANNELS, :, :], 2))
        else:
            N = torch.tensor(0.0, device=self.device)

        return L_hard + N

    def loss_ae(self, image):
        aug_idx = random.choice([1, 2, 3])
        coeff   = random.uniform(0.8, 1.2)
        if aug_idx == 1:
            aug = transforms.functional.adjust_brightness(image, coeff)
        elif aug_idx == 2:
            aug = transforms.functional.adjust_contrast(image, coeff)
        else:
            aug = transforms.functional.adjust_saturation(image, coeff)

        with torch.no_grad():
            t_out  = self.teacher(aug)
            norm_t = (t_out - self.channel_mean) / (self.channel_std + 1e-8)

        ae_out = self.ae(aug)
        s_out  = self.student(aug)
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]

        L_ae   = torch.mean(torch.pow(norm_t - ae_out,  2))
        L_stae = torch.mean(torch.pow(ae_out  - y_stae, 2))
        return L_ae, L_stae

    def compute_quantiles(self, dataloader):
        self.teacher.eval(); self.student.eval(); self.ae.eval()
        xst, xae = [], []
        for sample in dataloader:
            img = sample['image'].to(self.device)
            with torch.no_grad():
                t_out = self.teacher(img)
                s_out = self.student(img)
                a_out = self.ae(img)

            y_st   = s_out[:, :OUT_CHANNELS,  :, :]
            y_stae = s_out[:, -OUT_CHANNELS:, :, :]
            norm_t = (t_out - self.channel_mean) / (self.channel_std + 1e-8)
            d_st   = torch.pow(norm_t - y_st,   2)
            d_stae = torch.pow(a_out  - y_stae, 2)

            fm_st   = torch.mean(d_st,   dim=1, keepdim=True)
            fm_stae = torch.mean(d_stae, dim=1, keepdim=True)
            fm_st   = F.interpolate(fm_st,   size=(IMAGE_SIZE, IMAGE_SIZE),
                                    mode='bilinear', align_corners=False)
            fm_stae = F.interpolate(fm_stae, size=(IMAGE_SIZE, IMAGE_SIZE),
                                    mode='bilinear', align_corners=False)

            xst.append(fm_st.detach().cpu().numpy())
            xae.append(fm_stae.detach().cpu().numpy())

        qa_st = float(np.percentile(np.concatenate(xst), 90))
        qb_st = float(np.percentile(np.concatenate(xst), 99.5))
        qa_ae = float(np.percentile(np.concatenate(xae), 90))
        qb_ae = float(np.percentile(np.concatenate(xae), 99.5))
        return qa_st, qb_st, qa_ae, qb_ae

    def eval_auroc(self, dataloader, qa_st, qb_st, qa_ae, qb_ae):
        self.teacher.eval(); self.student.eval(); self.ae.eval()
        scores, labels = [], []
        for sample in dataloader:
            img   = sample['image'].to(self.device)
            label = sample['label'].item()
            with torch.no_grad():
                t_out = self.teacher(img)
                s_out = self.student(img)
                a_out = self.ae(img)

            y_st   = s_out[:, :OUT_CHANNELS,  :, :]
            y_stae = s_out[:, -OUT_CHANNELS:, :, :]
            norm_t = (t_out - self.channel_mean) / (self.channel_std + 1e-8)
            d_st   = torch.pow(norm_t - y_st,   2)
            d_stae = torch.pow(a_out  - y_stae, 2)

            fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                                    size=(IMAGE_SIZE, IMAGE_SIZE),
                                    mode='bilinear', align_corners=False)
            fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                                    size=(IMAGE_SIZE, IMAGE_SIZE),
                                    mode='bilinear', align_corners=False)

            norm_mst = (0.1 * (fm_st   - qa_st)) / (qb_st - qa_st + 1e-8)
            norm_mae = (0.1 * (fm_stae - qa_ae)) / (qb_ae - qa_ae + 1e-8)
            combined = 0.5 * norm_mst + 0.5 * norm_mae

            scores.append(float(combined.max().cpu()))
            labels.append(label)

        labels = np.array(labels)
        scores = np.array(scores)
        if labels.sum() == 0 or labels.sum() == len(labels):
            return 0.0
        return float(roc_auc_score(labels, scores))

    def save_checkpoint(self, qa_st, qb_st, qa_ae, qb_ae, suffix='best'):
        wm_tag = f'wm{int(self.width_mult * 100)}'
        torch.save(self.student.state_dict(),
                   os.path.join(self.ckpt_dir, f'{self.category}_student_{wm_tag}_{suffix}.pth'))
        torch.save(self.ae.state_dict(),
                   os.path.join(self.ckpt_dir, f'{self.category}_autoencoder_{wm_tag}_{suffix}.pth'))
        quantiles = {
            'qa_st': qa_st, 'qb_st': qb_st, 'qa_ae': qa_ae, 'qb_ae': qb_ae,
            'mean':  self.channel_mean.cpu().numpy(),
            'std':   self.channel_std.cpu().numpy(),
            'width_mult': self.width_mult,
        }
        np.save(os.path.join(self.ckpt_dir, f'{self.category}_quantiles_{wm_tag}_{suffix}.npy'),
                quantiles)
        if suffix == 'best':
            # NOTE: the canonical filenames (no wm tag) will only load correctly
            # with a loader that knows this width_mult — see load_efficientad_thin
            # in this same file's companion, or pass width_mult explicitly when
            # reconstructing Student_Thin/AutoEncoder_Thin at inference time.
            torch.save(self.student.state_dict(),
                       os.path.join(self.ckpt_dir, f'{self.category}_student.pth'))
            torch.save(self.ae.state_dict(),
                       os.path.join(self.ckpt_dir, f'{self.category}_autoencoder.pth'))
            np.save(os.path.join(self.ckpt_dir, f'{self.category}_quantiles.npy'), quantiles)

    def train(self):
        args = self.args
        print(f'[Thin] Loading dataset from {args.dataset_path}')

        train_ds = get_AD_dataset(type='VisA', root=args.dataset_path,
                                   transform=self.data_tf, gt_transform=self.gt_tf,
                                   phase='train', category=args.category)
        test_ds  = get_AD_dataset(type='VisA', root=args.dataset_path,
                                   transform=self.data_tf, gt_transform=self.gt_tf,
                                   phase='test', category=args.category)
        val_ds   = get_AD_dataset(type='VisA', root=args.dataset_path,
                                   transform=self.data_tf, gt_transform=self.gt_tf,
                                   phase='validation', category=args.category)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4)
        train_iter   = load_infinite(train_loader)

        n_train, n_test, n_val = len(train_ds), len(test_ds), len(val_ds)
        n_normal  = sum(1 for i in range(n_val) if val_ds[i]['label'] == 0)
        n_anomaly = n_val - n_normal
        print(f'[Thin] Train: {n_train}  Test(quantile): {n_test}  '
              f'Val(AUROC): {n_val} (normal={n_normal}, anomaly={n_anomaly})')

        imagenet_iter = None
        if not args.no_imagenet:
            if args.imagenet_dir and os.path.isdir(args.imagenet_dir):
                inet_ds     = ImageNetDataset(args.imagenet_dir, self.imagenet_tf)
                inet_loader = DataLoader(inet_ds, batch_size=1, shuffle=True, num_workers=4)
                imagenet_iter = load_infinite(inet_loader)
                print('[Thin] ImageNet regularisation: ON')
            else:
                print('[Thin] WARNING: --imagenet_dir not found. Running without it.')
        else:
            print('[Thin] ImageNet regularisation: OFF (--no_imagenet)')

        stats_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
        self.compute_channel_stats(stats_loader, n_samples=500)

        qa_st, qb_st, qa_ae, qb_ae = self.compute_quantiles(test_loader)

        optimizer = optim.Adam(list(self.student.parameters()) + list(self.ae.parameters()),
                                lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(0.95 * args.iterations), gamma=0.1)

        best_auroc = 0.0
        print(f'\n[Thin] Training width_mult={self.width_mult} for {args.iterations} iterations ...\n')

        for i in range(args.iterations):
            self.student.train(); self.ae.train()
            sample = next(train_iter)
            image  = sample['image'].to(self.device)

            L_st         = self.loss_st(image, imagenet_iter)
            L_ae, L_stae = self.loss_ae(image)
            loss         = L_st + L_ae + L_stae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % args.print_freq == 0:
                qa_st, qb_st, qa_ae, qb_ae = self.compute_quantiles(test_loader)
                auroc = self.eval_auroc(val_loader, qa_st, qb_st, qa_ae, qb_ae)

                print(f'  iter {i:6d}/{args.iterations}  loss={loss.item():.4f}  '
                      f'L_st={L_st.item():.4f}  L_ae={L_ae.item():.4f}  '
                      f'AUROC(val)={auroc:.4f}  [width_mult={self.width_mult}]')

                self.save_checkpoint(qa_st, qb_st, qa_ae, qb_ae, suffix='last')
                if auroc > best_auroc:
                    best_auroc = auroc
                    self.save_checkpoint(qa_st, qb_st, qa_ae, qb_ae, suffix='best')
                    print(f'  *** New best AUROC(val) @ width_mult={self.width_mult}: '
                          f'{best_auroc:.4f} → saved')

        print(f'\n[Thin] Training complete. width_mult={self.width_mult}')
        print(f'  Best AUROC (validation split) : {best_auroc:.4f}')
        print(f'  Student params : {count_params(self.student):,}')
        print(f'  AE params      : {count_params(self.ae):,}')
        print(f'  Checkpoints: {args.ckpt_dir}/')
        print(f'\nCopy best_teacher.pth into this ckpt_dir (unchanged, full-size teacher):')
        print(f'  cp {args.teacher_path} {args.ckpt_dir}/best_teacher.pth')
        print(f'\nRemember: at inference time you must reconstruct Student_Thin/'
              f'AutoEncoder_Thin with the SAME width_mult ({self.width_mult}) before '
              f'load_state_dict, since the layer shapes depend on it.')


if __name__ == '__main__':
    args = parse_args()
    trainer = ThinTrainer(args)
    trainer.train()