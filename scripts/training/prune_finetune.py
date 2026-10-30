"""
prune_finetune.py
==================
OPTION 4 — Structured (channel) pruning of Student + AutoEncoder, followed
by fine-tune recovery, to speed up inference.

Why structured pruning needs a dependency-aware library
---------------------------------------------------------
Unstructured pruning (zeroing individual weights with torch.nn.utils.prune)
does NOT speed up dense inference — the zeros are still multiplied. To get
an actual latency win you have to remove whole output channels, which means
every downstream layer that consumes those channels (the next conv's input
channels, any parallel BatchNorm, etc.) must be resized to match. Doing
that resizing by hand is fiddly and error-prone across the Student's dual
output heads and the AutoEncoder's encoder/decoder skip structure, so this
script uses `torch-pruning` (dependency-graph-aware structured pruning),
which traces the model and prunes+resizes coupled layers automatically.

Install:  pip install torch-pruning --break-system-packages

Pipeline
--------
1. Load an already fp32-trained Student/AutoEncoder checkpoint.
2. Build an importance criterion (L2 norm of each output channel's
   weights) and use MagnitudePruner (or GroupNormPruner) from
   torch-pruning to remove `--prune_ratio` of channels from every prunable
   conv layer, respecting the dependency graph so the model stays valid.
3. Fine-tune the now-smaller network against the frozen Teacher for
   `--finetune_iterations` to recover accuracy lost during pruning —
   this is the standard "prune, then fine-tune" recipe; skipping it
   typically costs several AUROC points.
4. Re-calibrate quantile thresholds (post-pruning feature statistics
   differ) and report AUROC before/after.

Caveats
-------
- The Teacher is NOT pruned (frozen, full-size, used only to produce
  training targets) — same rationale as train_qat.py.
- Because channels are physically removed, the resulting checkpoint has a
  DIFFERENT layer shape than the original Student/AutoEncoder classes in
  models.py — you cannot `load_state_dict` it into a fresh `Student(...)`.
  This script saves the pruned model as a full pickled `nn.Module` (not
  just a state_dict) via `torch.save(model, ...)` so the architecture
  travels with the weights; load it back with `torch.load(path,
  weights_only=False)`.
- Start with a conservative --prune_ratio (0.2-0.3) and check the AUROC
  delta before going more aggressive — recoverability drops off sharply
  past ~50% channel removal for a network this small.

Usage
-----
    python prune_finetune.py \
        --dataset_path        /path/to/cropped_dataset \
        --ckpt_dir            ./ckptSmall \
        --out_dir             ./ckpt_pruned \
        --teacher_path        ./ckptSmall/best_teacher.pth \
        --category            chewinggum \
        --model_size          S \
        --prune_ratio         0.3 \
        --finetune_iterations 10000
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
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.models import Teacher, Student, AutoEncoder
from utils.data_loader import get_AD_dataset, load_infinite

IMAGE_SIZE   = 256
OUT_CHANNELS = 384

torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path',        required=True)
    p.add_argument('--ckpt_dir',            required=True,
                   help='Directory with an already fp32-trained '
                        '{category}_student.pth / {category}_autoencoder.pth '
                        '/ {category}_quantiles.npy')
    p.add_argument('--out_dir',             required=True)
    p.add_argument('--teacher_path',        required=True)
    p.add_argument('--category',            default='chewinggum')
    p.add_argument('--model_size',          default='S', choices=['S', 'M'])
    p.add_argument('--prune_ratio',         type=float, default=0.3,
                   help='Fraction of channels to remove from each prunable conv layer.')
    p.add_argument('--finetune_iterations', type=int, default=10000,
                   help='Recovery fine-tuning after pruning. Shorter than the '
                        'original 70k-iteration training run.')
    p.add_argument('--batch_size',          type=int, default=1)
    p.add_argument('--lr',                  type=float, default=3e-5,
                   help='Small LR for recovery fine-tuning.')
    p.add_argument('--print_freq',          type=int, default=200)
    p.add_argument('--device',              default='cuda')
    p.add_argument('--seed',                type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_pruners(student, ae, prune_ratio, example_input, example_input_ae):
    """
    Build torch-pruning MagnitudePruner objects for Student and AutoEncoder.
    Imported lazily so the rest of the script (and --help) still works if
    torch-pruning isn't installed yet.
    """
    try:
        import torch_pruning as tp
    except ImportError as e:
        raise ImportError(
            "This script requires torch-pruning for dependency-aware structured "
            "pruning. Install it with:\n"
            "    pip install torch-pruning --break-system-packages"
        ) from e

    # Never prune the final output conv of each network — those channel
    # counts (384 for AE decoder, 768 for Student) must stay fixed because
    # the loss functions compare directly against the frozen Teacher's
    # 384-channel output.
    # PDN_M (model_size='M') has both conv4 (hidden 512-ch) and conv6 (final,
    # 768-ch output for Student) — only the final output conv must be
    # protected. PDN_S's final output conv is conv4. Prefer conv6 when it
    # exists (PDN_M case); otherwise fall back to conv4 (PDN_S case).
    if hasattr(student.pdn, 'conv6'):
        student_ignored = [student.pdn.conv6]
    else:
        student_ignored = [student.pdn.conv4]
    ae_ignored = [ae.decoder.deconv8]

    imp = tp.importance.MagnitudeImportance(p=2)

    student_pruner = tp.pruner.MagnitudePruner(
        student, example_input, importance=imp,
        pruning_ratio=prune_ratio, ignored_layers=student_ignored,
    )
    ae_pruner = tp.pruner.MagnitudePruner(
        ae, example_input_ae, importance=imp,
        pruning_ratio=prune_ratio, ignored_layers=ae_ignored,
    )
    return student_pruner, ae_pruner


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def compute_quantiles(teacher, student, ae, dataloader, mean, std, device):
    teacher.eval(); student.eval(); ae.eval()
    xst, xae = [], []
    for sample in dataloader:
        img = sample['image'].to(device)
        with torch.no_grad():
            t_out = teacher(img)
            s_out = student(img)
            a_out = ae(img)

        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]
        norm_t = (t_out - mean) / (std + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)

        fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                                size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                                size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        xst.append(fm_st.detach().cpu().numpy())
        xae.append(fm_stae.detach().cpu().numpy())

    qa_st = float(np.percentile(np.concatenate(xst), 90))
    qb_st = float(np.percentile(np.concatenate(xst), 99.5))
    qa_ae = float(np.percentile(np.concatenate(xae), 90))
    qb_ae = float(np.percentile(np.concatenate(xae), 99.5))
    return qa_st, qb_st, qa_ae, qb_ae


def eval_auroc(teacher, student, ae, dataloader, mean, std, qa_st, qb_st, qa_ae, qb_ae, device):
    teacher.eval(); student.eval(); ae.eval()
    scores, labels = [], []
    for sample in dataloader:
        img   = sample['image'].to(device)
        label = sample['label'].item()
        with torch.no_grad():
            t_out = teacher(img)
            s_out = student(img)
            a_out = ae(img)

        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]
        norm_t = (t_out - mean) / (std + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)

        fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                                size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                                size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

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


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    data_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    train_ds = get_AD_dataset(type='VisA', root=args.dataset_path, transform=data_tf,
                               gt_transform=data_tf, phase='train', category=args.category)
    test_ds  = get_AD_dataset(type='VisA', root=args.dataset_path, transform=data_tf,
                               gt_transform=data_tf, phase='test', category=args.category)
    val_ds   = get_AD_dataset(type='VisA', root=args.dataset_path, transform=data_tf,
                               gt_transform=data_tf, phase='validation', category=args.category)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=4)
    train_iter   = load_infinite(train_loader)

    print(f'[Prune] Loading frozen teacher: {args.teacher_path}')
    teacher = Teacher(args.model_size)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device, weights_only=False))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    print(f'[Prune] Loading fp32 student/autoencoder from {args.ckpt_dir}')
    student = Student(args.model_size)
    ae      = AutoEncoder()
    student.load_state_dict(torch.load(
        os.path.join(args.ckpt_dir, f'{args.category}_student.pth'),
        map_location=device, weights_only=False))
    ae.load_state_dict(torch.load(
        os.path.join(args.ckpt_dir, f'{args.category}_autoencoder.pth'),
        map_location=device, weights_only=False))
    quantiles = np.load(
        os.path.join(args.ckpt_dir, f'{args.category}_quantiles.npy'),
        allow_pickle=True).item()
    student.to(device); ae.to(device)

    mean = torch.tensor(np.array(quantiles['mean']), dtype=torch.float32, device=device)
    std  = torch.tensor(np.array(quantiles['std']),  dtype=torch.float32, device=device)
    qa_st, qb_st, qa_ae, qb_ae = (quantiles['qa_st'], quantiles['qb_st'],
                                   quantiles['qa_ae'], quantiles['qb_ae'])

    baseline_auroc = eval_auroc(teacher, student, ae, val_loader,
                                mean, std, qa_st, qb_st, qa_ae, qb_ae, device)
    baseline_student_params = count_params(student)
    baseline_ae_params      = count_params(ae)
    print(f'[Prune] Baseline AUROC(val)     : {baseline_auroc:.4f}')
    print(f'[Prune] Baseline Student params : {baseline_student_params:,}')
    print(f'[Prune] Baseline AE params      : {baseline_ae_params:,}')

    # ── structured pruning ───────────────────────────────────────────────
    example_input    = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    example_input_ae = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    print(f'\n[Prune] Building dependency-aware pruners (ratio={args.prune_ratio}) ...')
    student_pruner, ae_pruner = build_pruners(
        student, ae, args.prune_ratio, example_input, example_input_ae)

    print('[Prune] Pruning Student ...')
    student_pruner.step()
    print('[Prune] Pruning AutoEncoder ...')
    ae_pruner.step()

    pruned_student_params = count_params(student)
    pruned_ae_params      = count_params(ae)
    print(f'[Prune] Student params : {baseline_student_params:,} -> {pruned_student_params:,} '
          f'({100 * (1 - pruned_student_params / baseline_student_params):.1f}% removed)')
    print(f'[Prune] AE params      : {baseline_ae_params:,} -> {pruned_ae_params:,} '
          f'({100 * (1 - pruned_ae_params / baseline_ae_params):.1f}% removed)')

    pre_finetune_auroc = eval_auroc(teacher, student, ae, val_loader,
                                    mean, std, qa_st, qb_st, qa_ae, qb_ae, device)
    print(f'[Prune] AUROC(val) immediately after pruning (pre fine-tune): '
          f'{pre_finetune_auroc:.4f}  (delta = {pre_finetune_auroc - baseline_auroc:+.4f})')

    # ── fine-tune recovery ───────────────────────────────────────────────
    optimizer = optim.Adam(list(student.parameters()) + list(ae.parameters()), lr=args.lr)

    print(f'\n[Prune] Fine-tuning for {args.finetune_iterations} iterations to recover accuracy ...\n')
    best_auroc = pre_finetune_auroc

    for i in range(args.finetune_iterations):
        student.train(); ae.train()
        sample = next(train_iter)
        image  = sample['image'].to(device)

        with torch.no_grad():
            t_out  = teacher(image)
            norm_t = (t_out - mean) / (std + 1e-8)

        s_out = student(image)
        y_st  = s_out[:, :OUT_CHANNELS, :, :]
        dist  = torch.pow(norm_t - y_st, 2)
        dhard = torch.quantile(dist[:8], 0.999)
        L_st  = torch.mean(dist[dist >= dhard])

        aug_idx = random.choice([1, 2, 3])
        coeff   = random.uniform(0.8, 1.2)
        if aug_idx == 1:
            aug = transforms.functional.adjust_brightness(image, coeff)
        elif aug_idx == 2:
            aug = transforms.functional.adjust_contrast(image, coeff)
        else:
            aug = transforms.functional.adjust_saturation(image, coeff)

        with torch.no_grad():
            t_aug      = teacher(aug)
            norm_t_aug = (t_aug - mean) / (std + 1e-8)

        ae_out = ae(aug)
        s_aug  = student(aug)
        y_stae = s_aug[:, -OUT_CHANNELS:, :, :]

        L_ae   = torch.mean(torch.pow(norm_t_aug - ae_out,  2))
        L_stae = torch.mean(torch.pow(ae_out     - y_stae, 2))

        loss = L_st + L_ae + L_stae
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            qa_st, qb_st, qa_ae, qb_ae = compute_quantiles(
                teacher, student, ae, test_loader, mean, std, device)
            auroc = eval_auroc(teacher, student, ae, val_loader,
                               mean, std, qa_st, qb_st, qa_ae, qb_ae, device)
            print(f'  iter {i:6d}/{args.finetune_iterations}  loss={loss.item():.4f}  '
                  f'AUROC(val)={auroc:.4f}')

            if auroc > best_auroc:
                best_auroc = auroc
                torch.save(student, os.path.join(args.out_dir, f'{args.category}_student_pruned.pt'))
                torch.save(ae,      os.path.join(args.out_dir, f'{args.category}_autoencoder_pruned.pt'))
                np.save(os.path.join(args.out_dir, f'{args.category}_quantiles.npy'), {
                    'qa_st': qa_st, 'qb_st': qb_st, 'qa_ae': qa_ae, 'qb_ae': qb_ae,
                    'mean': mean.cpu().numpy(), 'std': std.cpu().numpy(),
                    'prune_ratio': args.prune_ratio,
                })
                print(f'  *** New best AUROC(val): {best_auroc:.4f} → saved')

    print(f'\n[Prune] Done.')
    print(f'  Baseline AUROC          : {baseline_auroc:.4f}')
    print(f'  Post-prune, pre-finetune: {pre_finetune_auroc:.4f}')
    print(f'  Best after fine-tune    : {best_auroc:.4f}')
    print(f'  Student params: {baseline_student_params:,} -> {pruned_student_params:,}')
    print(f'  AE params     : {baseline_ae_params:,} -> {pruned_ae_params:,}')
    print(f'  Checkpoints (full pickled nn.Module, NOT plain state_dicts) in {args.out_dir}/')
    print(f'\nLoad with:')
    print(f"  student = torch.load('{args.out_dir}/{args.category}_student_pruned.pt', weights_only=False)")
    print(f"  ae      = torch.load('{args.out_dir}/{args.category}_autoencoder_pruned.pt', weights_only=False)")


if __name__ == '__main__':
    main()