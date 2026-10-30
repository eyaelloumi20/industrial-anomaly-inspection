"""
filter_quality_crops.py
=======================
Filter crop images based on image quality metrics.

Metrics used:
    1. Laplacian variance (blur detection) – lower = blurrier.
    2. Mean brightness – too dark or too bright.
    3. Contrast (standard deviation of pixel values) – low = washed out.
    4. Otsu coverage – fraction of the crop that is foreground (non‑black).
    5. Edge density (optional) – to detect over‑smooth or textureless crops.

For each image, the script computes these metrics and compares them against
configurable thresholds. If any metric falls outside the acceptable range,
the crop is marked as BAD_QUALITY.

Usage:
    python filter_quality_crops.py \
        --input_dir ./crops_dataset_normal/crops \
        --output_dir ./crops_filtered \
        --min_blur 100 \
        --min_brightness 50 \
        --max_brightness 200 \
        --min_contrast 30 \
        --min_coverage 0.30 \
        --min_edge_density 0.02 \
        --move (or --copy)
"""

import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import csv

def compute_quality_metrics(img_path):
    """Compute image quality metrics for a single crop."""
    # Load with OpenCV (BGR)
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Blur detection: Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Brightness: mean pixel intensity
    brightness = np.mean(gray)

    # 3. Contrast: standard deviation of pixel intensities
    contrast = np.std(gray)

    # 4. Otsu coverage: fraction of the crop that is foreground (non‑background)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coverage = np.mean(mask > 0)  # fraction of pixels > 0

    # 5. Edge density: fraction of pixels that are edges (Canny)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)

    return {
        'laplacian_var': laplacian_var,
        'brightness': brightness,
        'contrast': contrast,
        'coverage': coverage,
        'edge_density': edge_density,
    }

def is_good_quality(metrics, args):
    """Return (bool, reason) where bool indicates if crop is good quality."""
    reasons = []

    if metrics['laplacian_var'] < args.min_blur:
        reasons.append(f"blur={metrics['laplacian_var']:.1f} < {args.min_blur}")
    if metrics['brightness'] < args.min_brightness:
        reasons.append(f"too_dark={metrics['brightness']:.1f} < {args.min_brightness}")
    if metrics['brightness'] > args.max_brightness:
        reasons.append(f"too_bright={metrics['brightness']:.1f} > {args.max_brightness}")
    if metrics['contrast'] < args.min_contrast:
        reasons.append(f"low_contrast={metrics['contrast']:.1f} < {args.min_contrast}")
    if metrics['coverage'] < args.min_coverage:
        reasons.append(f"coverage={metrics['coverage']:.3f} < {args.min_coverage}")
    if metrics['edge_density'] < args.min_edge_density:
        reasons.append(f"low_edges={metrics['edge_density']:.3f} < {args.min_edge_density}")

    if reasons:
        return False, "; ".join(reasons)
    return True, "good"

def main():
    parser = argparse.ArgumentParser(
        description="Filter crop images based on quality metrics."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Folder containing crop PNGs.")
    parser.add_argument("--output_dir", required=True,
                        help="Output folder where GOOD_QUALITY/BAD_QUALITY subfolders will be created.")
    parser.add_argument("--min_blur", type=float, default=100,
                        help="Minimum Laplacian variance (higher = sharper). Default 100.")
    parser.add_argument("--min_brightness", type=float, default=50,
                        help="Minimum mean brightness (0–255). Default 50.")
    parser.add_argument("--max_brightness", type=float, default=200,
                        help="Maximum mean brightness (0–255). Default 200.")
    parser.add_argument("--min_contrast", type=float, default=30,
                        help="Minimum contrast (std dev of pixel values). Default 30.")
    parser.add_argument("--min_coverage", type=float, default=0.30,
                        help="Minimum Otsu foreground coverage (0–1). Default 0.30.")
    parser.add_argument("--min_edge_density", type=float, default=0.02,
                        help="Minimum edge density (fraction of pixels on edges). Default 0.02.")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying (default: copy).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    good_dir = output_dir / "GOOD_QUALITY"
    bad_dir = output_dir / "BAD_QUALITY"
    for d in [good_dir, bad_dir]:
        d.mkdir(parents=True, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    print(f"Found {len(image_paths)} images in {input_dir}")

    if not image_paths:
        print("No images found. Exiting.")
        return

    csv_path = output_dir / "quality_report.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "laplacian_var", "brightness", "contrast", "coverage", "edge_density",
            "verdict", "reason"
        ])

    move_func = shutil.move if args.move else shutil.copy2

    good_count = 0
    bad_count = 0

    for i, img_path in enumerate(image_paths):
        metrics = compute_quality_metrics(img_path)
        if metrics is None:
            print(f"  [{i+1}/{len(image_paths)}] Could not read {img_path.name}")
            continue

        is_good, reason = is_good_quality(metrics, args)

        dest_dir = good_dir if is_good else bad_dir
        dest_path = dest_dir / img_path.name
        move_func(str(img_path), str(dest_path))

        if is_good:
            good_count += 1
        else:
            bad_count += 1

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                img_path.name,
                f"{metrics['laplacian_var']:.2f}",
                f"{metrics['brightness']:.2f}",
                f"{metrics['contrast']:.2f}",
                f"{metrics['coverage']:.4f}",
                f"{metrics['edge_density']:.4f}",
                "GOOD" if is_good else "BAD",
                reason
            ])

        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(image_paths)} images...")

    print(f"\nDone. Processed {len(image_paths)} images.")
    print(f"  GOOD_QUALITY : {good_count}")
    print(f"  BAD_QUALITY  : {bad_count}")
    print(f"Report saved to: {csv_path}")

if __name__ == "__main__":
    main()