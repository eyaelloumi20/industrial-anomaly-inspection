#!/usr/bin/env python3
"""Split a flat YOLO export (images/ + labels/ + classes.txt) into train/val.

Usage:
    python split_dataset.py [--src SRC] [--dst DST] [--val-frac 0.2] [--seed 42]
"""
import argparse
import os
import random
import shutil


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="/home/ahmed/two-stage-efficientad-inspection/project_mycategory",
        help="Source dir containing images/, labels/, classes.txt",
    )
    p.add_argument(
        "--dst",
        default="/home/ahmed/two-stage-efficientad-inspection/project_mycategory_split",
        help="Output dir for the split dataset",
    )
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of images for val")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    return p.parse_args()


def main():
    args = parse_args()

    img_dir = os.path.join(args.src, "images")
    lbl_dir = os.path.join(args.src, "labels")
    classes_file = os.path.join(args.src, "classes.txt")

    if not os.path.isdir(img_dir):
        raise SystemExit(f"Images dir not found: {img_dir}")
    if not os.path.isdir(lbl_dir):
        raise SystemExit(f"Labels dir not found: {lbl_dir}")
    if not os.path.isfile(classes_file):
        raise SystemExit(f"classes.txt not found: {classes_file}")

    with open(classes_file) as f:
        classes = [c.strip() for c in f if c.strip()]

    images = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    random.seed(args.seed)
    random.shuffle(images)

    n_val = max(1, int(len(images) * args.val_frac))
    val_set = set(images[:n_val])
    train_set = set(images[n_val:])

    missing_labels = []

    for split, files in [("train", train_set), ("val", val_set)]:
        os.makedirs(os.path.join(args.dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.dst, "labels", split), exist_ok=True)
        for img_name in files:
            stem = os.path.splitext(img_name)[0]
            lbl_name = stem + ".txt"
            shutil.copy(
                os.path.join(img_dir, img_name),
                os.path.join(args.dst, "images", split, img_name),
            )
            lbl_src = os.path.join(lbl_dir, lbl_name)
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(args.dst, "labels", split, lbl_name))
            else:
                # Create empty label file so YOLO treats it as a negative/background image
                open(os.path.join(args.dst, "labels", split, lbl_name), "w").close()
                missing_labels.append(img_name)

    yaml_content = f"path: {args.dst}\ntrain: images/train\nval: images/val\n\nnames:\n"
    for i, c in enumerate(classes):
        yaml_content += f"  {i}: {c}\n"

    yaml_path = os.path.join(args.dst, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Done. {len(train_set)} train / {len(val_set)} val images.")
    print(f"data.yaml written to {yaml_path}")
    if missing_labels:
        print(f"\nWarning: {len(missing_labels)} image(s) had no matching label file "
              f"(empty label written instead):")
        for name in missing_labels[:20]:
            print(f"  - {name}")
        if len(missing_labels) > 20:
            print(f"  ... and {len(missing_labels) - 20} more")


if __name__ == "__main__":
    main()
