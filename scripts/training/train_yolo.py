"""
prepare_and_train.py
=====================
Takes the raw Label Studio YOLO export and:
  1. Splits images + labels into train / val (80/20)
  2. Writes a chewinggum.yaml dataset config
  3. Fine-tunes YOLOv8 on the annotated data
  4. Validates and prints final mAP


"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare Label Studio export + train YOLOv8")
    p.add_argument("--dataset_dir", required=True,
                   help="Path to extracted Label Studio export folder "
                        "(contains images/, labels/, classes.txt)")
    p.add_argument("--out_dir",     default=None,
                   help="Where to write the split dataset. "
                        "Defaults to <dataset_dir>_split")
    p.add_argument("--val_ratio",   type=float, default=0.2,
                   help="Fraction of images for validation (default: 0.2)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--model",       default="yolov8n.pt",
                   help="Base YOLOv8 weights (default: yolov8n.pt)")
    p.add_argument("--save_dir",    default="./yolo_runs")
    p.add_argument("--name",        default="chewinggum_finetune")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--imgsz",       type=int,   default=640)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--device",      default=None,
                   help="CUDA device index (e.g. 0) or 'cpu'")
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--skip_train",  action="store_true",
                   help="Only prepare the dataset, skip training")
    return p.parse_args()


# ── Step 1: validate export folder ───────────────────────────────────────────

def validate_export(dataset_dir: str):
    images_dir  = os.path.join(dataset_dir, "images")
    labels_dir  = os.path.join(dataset_dir, "labels")
    classes_txt = os.path.join(dataset_dir, "classes.txt")

    assert os.path.isdir(images_dir),  f"Missing images/ in {dataset_dir}"
    assert os.path.isdir(labels_dir),  f"Missing labels/ in {dataset_dir}"
    assert os.path.isfile(classes_txt), f"Missing classes.txt in {dataset_dir}"

    # collect all image files
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [f for f in os.listdir(images_dir)
              if os.path.splitext(f)[1] in exts]

    # check every image has a matching label
    missing_labels = []
    for img in images:
        stem = os.path.splitext(img)[0]
        lbl  = os.path.join(labels_dir, stem + ".txt")
        if not os.path.isfile(lbl):
            missing_labels.append(img)

    if missing_labels:
        print(f"  [WARN] {len(missing_labels)} images have no label file "
              f"(unannotated) — they will be SKIPPED.")
        print(f"         First few: {missing_labels[:5]}")

    # only keep images that have a label
    annotated = [img for img in images
                 if os.path.isfile(os.path.join(
                     labels_dir, os.path.splitext(img)[0] + ".txt"))]

    print(f"[Prepare] Found {len(annotated)} annotated images "
          f"(skipped {len(missing_labels)} without labels)")

    # read class names
    with open(classes_txt) as f:
        classes = [l.strip() for l in f if l.strip()]
    print(f"[Prepare] Classes: {classes}")

    return annotated, classes, images_dir, labels_dir


# ── Step 2: train / val split ─────────────────────────────────────────────────

def split_dataset(annotated, images_dir, labels_dir, out_dir,
                  val_ratio=0.2, seed=42):

    random.seed(seed)
    random.shuffle(annotated)

    n_val   = max(1, int(len(annotated) * val_ratio))
    val_set = set(annotated[:n_val])
    trn_set = set(annotated[n_val:])

    print(f"[Prepare] Split → train: {len(trn_set)}  val: {len(val_set)}")

    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    def copy_split(file_set, split_name):
        for img_file in file_set:
            stem = os.path.splitext(img_file)[0]
            src_img = os.path.join(images_dir, img_file)
            src_lbl = os.path.join(labels_dir, stem + ".txt")
            dst_img = os.path.join(out_dir, "images", split_name, img_file)
            dst_lbl = os.path.join(out_dir, "labels", split_name, stem + ".txt")
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

    copy_split(trn_set, "train")
    copy_split(val_set, "val")
    print(f"[Prepare] Files copied to {out_dir}")


# ── Step 3: write YAML ────────────────────────────────────────────────────────

def write_yaml(out_dir: str, classes: list) -> str:
    yaml_path = os.path.join(out_dir, "chewinggum.yaml")
    abs_out   = os.path.abspath(out_dir)
    with open(yaml_path, "w") as f:
        f.write(f"# YOLOv8 dataset — manually annotated chewinggum\n")
        f.write(f"path:  {abs_out}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")
    print(f"[Prepare] YAML written → {yaml_path}")
    return yaml_path


# ── Step 4: train ─────────────────────────────────────────────────────────────

def train(args, yaml_path: str) -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed: pip install ultralytics")
        sys.exit(1)

    import torch
    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"\n[Train] Starting YOLOv8 fine-tuning")
    print(f"  Base model  : {args.model}")
    print(f"  Dataset     : {yaml_path}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch       : {args.batch}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Device      : {device}\n")

    model = YOLO(args.model)
    model.train(
        data      = os.path.abspath(yaml_path),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = device,
        project   = args.save_dir,
        name      = args.name,
        patience  = args.patience,
        workers   = args.workers,
        lr0       = 0.01,
        lrf       = 0.01,
        # augmentation — mild for controlled product photography
        hsv_h     = 0.015,
        hsv_s     = 0.4,
        hsv_v     = 0.3,
        flipud    = 0.0,   # gum has a fixed orientation
        fliplr    = 0.5,
        mosaic    = 0.5,
        mixup     = 0.0,
        verbose   = True,
        save      = True,
        plots     = True,
    )

    best = os.path.join(args.save_dir, args.name, "weights", "best.pt")
    print(f"\n[Train] Done. Best weights → {best}")
    return best


# ── Step 5: validate ──────────────────────────────────────────────────────────

def validate(args, yaml_path: str, best: str):
    from ultralytics import YOLO
    import torch
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    print(f"\n[Val] Running validation on val split ...")
    model   = YOLO(best)
    metrics = model.val(
        data    = os.path.abspath(yaml_path),
        imgsz   = args.imgsz,
        conf    = 0.25,
        device  = device,
        verbose = True,
    )
    print(f"\n[Val] mAP50    : {metrics.box.map50:.4f}")
    print(f"[Val] mAP50-95 : {metrics.box.map:.4f}")

    print(f"\n{'='*60}")
    print(f"Next step — build the cropped EfficientAD dataset:")
    print(f"  python build_crop_dataset.py \\")
    print(f"      --visa_root    /home/ahmed/Downloads/VisA_20220922 \\")
    print(f"      --out_dir      ./cropped_dataset \\")
    print(f"      --yolo_weights {best} \\")
    print(f"      --target_class 0 \\")
    print(f"      --device       cuda")
    print(f"{'='*60}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    dataset_dir = args.dataset_dir
    out_dir     = args.out_dir or (dataset_dir.rstrip("/") + "_split")

    print(f"\n{'='*60}")
    print(f"  Label Studio Export → YOLO Training Pipeline")
    print(f"{'='*60}\n")

    # 1. validate
    annotated, classes, images_dir, labels_dir = validate_export(dataset_dir)

    # 2. split
    split_dataset(annotated, images_dir, labels_dir, out_dir,
                  val_ratio=args.val_ratio, seed=args.seed)

    # 3. yaml
    yaml_path = write_yaml(out_dir, classes)

    if args.skip_train:
        print(f"\n[Prepare] --skip_train set. Dataset ready at: {out_dir}")
        print(f"  To train manually:\n"
              f"    python train_yolo.py --data {yaml_path}")
        return

    # 4. train
    best = train(args, yaml_path)

    # 5. validate
    validate(args, yaml_path, best)


if __name__ == "__main__":
    main()
