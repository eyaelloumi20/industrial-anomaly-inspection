#!/usr/bin/env python3
"""
calibrate_ema_thresholds.py

Calibrate per-model EMA thresholds for the current simple real-time detector
using NORMAL-ONLY calibration videos.

The detector is run with a temporary threshold only to produce score CSVs.
The final threshold is the requested upper quantile of EMA scores from normal
data (default: 99th percentile, alpha=0.01).

This does NOT use video_A / video_A3 / video_A4, so held-out anomaly videos
remain available for evaluation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")

VARIANTS = {
    "lowres192": {
        "method": "lowres",
        "ckpt_dir": "ckpt_lowres_192",
    },
    "lowres224": {
        "method": "lowres",
        "ckpt_dir": "ckpt_lowres_224",
    },
    "thin075": {
        "method": "thin",
        "ckpt_dir": "ckpt_thin_075",
    },
    "thin050": {
        "method": "thin",
        "ckpt_dir": "ckpt_thin_050",
    },
    "pruned030": {
        "method": "pruned",
        "ckpt_dir": "ckpt_pruned_030",
    },
}

NORMAL_VIDEOS = [
    ROOT / "videos_N/videoN1.mp4",
    ROOT / "videos_N/video_N.mp4",
    ROOT / "videos_N/video_N2.mp4",
]


def run_detector(
    detector: Path,
    variant_name: str,
    cfg: dict,
    video: Path,
    out_dir: Path,
    python_exe: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scores.csv"

    # Reuse an already completed run.
    if csv_path.is_file():
        print(f"[reuse] {variant_name} / {video.name}: {csv_path}")
        return csv_path

    cmd = [
        python_exe,
        str(detector),
        "--video", str(video),
        "--yolo_det_weights",
        str(ROOT / "runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt"),
        "--method", cfg["method"],
        "--ckpt_dir", str(ROOT / cfg["ckpt_dir"]),
        # Temporary only. It does not affect raw/EMA scores.
        "--threshold", "0.45",
        "--category", "chewinggum",
        "--model_size", "S",
        "--device", "cuda:0",
        "--fp16",
        "--threshold_basis", "ema",
        "--ema_alpha", "0.3",
        "--clip_margin", "5",
        "--min_box_frac", "0.5",
        "--batch_frames", "16",
        "--batch_crops", "96",
        "--gpu_batch_crops", "8",
        "--crop_workers", "12",
        "--no_detect_unknown",
        "--no_video",
        "--save_dir", str(out_dir),
    ]

    print("\n" + "=" * 78)
    print(f"CALIBRATION RUN: {variant_name} / {video.name}")
    print("=" * 78)
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=ROOT, check=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Detector finished but scores.csv is missing: {csv_path}")
    return csv_path


def load_scores(csv_path: Path):
    df = pd.read_csv(csv_path)
    needed = {"raw_score", "ema_score"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

    raw = pd.to_numeric(df["raw_score"], errors="coerce").dropna().to_numpy(float)
    ema = pd.to_numeric(df["ema_score"], errors="coerce").dropna().to_numpy(float)

    if len(ema) == 0:
        raise ValueError(f"{csv_path}: no valid EMA scores")
    return raw, ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Normal-score quantile used as anomaly threshold (default: 0.99).",
    )
    ap.add_argument(
        "--detector",
        default=str(ROOT / "video_streaming_detector_simple_optimized_v2.py"),
    )
    ap.add_argument(
        "--output_root",
        default=str(ROOT / "calibration_ema_variants"),
    )
    ap.add_argument(
        "--threshold_dir",
        default=str(ROOT / "video_calibration_new"),
    )
    ap.add_argument(
        "--variants",
        nargs="*",
        default=list(VARIANTS),
        choices=list(VARIANTS),
    )
    args = ap.parse_args()

    if not (0.5 < args.quantile < 1.0):
        raise ValueError("--quantile must be between 0.5 and 1.0")

    detector = Path(args.detector)
    if not detector.is_file():
        raise FileNotFoundError(
            f"Detector not found: {detector}\n"
            "Copy video_streaming_detector_simple_optimized_v2.py into the project root."
        )

    missing_videos = [str(v) for v in NORMAL_VIDEOS if not v.is_file()]
    if missing_videos:
        raise FileNotFoundError(
            "Missing normal calibration video(s):\n  " + "\n  ".join(missing_videos)
        )

    out_root = Path(args.output_root)
    threshold_dir = Path(args.threshold_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    threshold_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for variant_name in args.variants:
        cfg = VARIANTS[variant_name]
        pooled_raw = []
        pooled_ema = []
        per_video = {}

        for video in NORMAL_VIDEOS:
            run_dir = out_root / variant_name / video.stem
            csv_path = run_detector(
                detector=detector,
                variant_name=variant_name,
                cfg=cfg,
                video=video,
                out_dir=run_dir,
                python_exe=sys.executable,
            )

            raw, ema = load_scores(csv_path)
            pooled_raw.append(raw)
            pooled_ema.append(ema)

            per_video[video.stem] = {
                "video": str(video),
                "n_scores": int(len(ema)),
                "raw_mean": float(np.mean(raw)),
                "raw_p99": float(np.quantile(raw, 0.99)),
                "ema_mean": float(np.mean(ema)),
                "ema_p95": float(np.quantile(ema, 0.95)),
                "ema_p99": float(np.quantile(ema, 0.99)),
                "ema_max": float(np.max(ema)),
            }

        pooled_raw = np.concatenate(pooled_raw)
        pooled_ema = np.concatenate(pooled_ema)

        tau_ema = float(np.quantile(pooled_ema, args.quantile))
        tau_raw = float(np.quantile(pooled_raw, args.quantile))

        result = {
            "method": cfg["method"],
            "variant": variant_name,
            "category": "chewinggum",
            "model_size": "S",
            "ckpt_dir": str(ROOT / cfg["ckpt_dir"]),
            "calibration_type": "normal_only_quantile",
            "calibration_basis": "ema",
            "recommended_basis": "ema",
            "ema_alpha": 0.3,
            "quantile": float(args.quantile),
            "alpha_p": float(1.0 - args.quantile),
            # Compatible with video_streaming_detector_simple_optimized_v2.py
            "tau_p": tau_ema,
            "tau_f": tau_ema,
            "raw_quantile_same_level": tau_raw,
            "n_videos": len(NORMAL_VIDEOS),
            "n_scores_total": int(len(pooled_ema)),
            "ema_mean": float(np.mean(pooled_ema)),
            "ema_std": float(np.std(pooled_ema)),
            "ema_min": float(np.min(pooled_ema)),
            "ema_max": float(np.max(pooled_ema)),
            "per_video": per_video,
            "settings": {
                "fp16": True,
                "gpu_batch_crops": 8,
                "batch_frames": 16,
                "batch_crops": 96,
                "crop_workers": 12,
                "clip_margin": 5,
                "min_box_frac": 0.5,
                "threshold_basis": "ema",
                "ema_alpha": 0.3,
            },
            "note": (
                "Threshold calibrated only on normal videos. "
                "Held-out anomaly videos must be used only for final evaluation."
            ),
        }

        out_json = threshold_dir / f"video_thresholds_{variant_name}_ema.json"
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results[variant_name] = result

        print("\n" + "-" * 78)
        print(f"{variant_name}: calibrated EMA threshold = {tau_ema:.6f}")
        print(f"saved -> {out_json}")
        print("-" * 78)

    summary_path = threshold_dir / "variant_ema_thresholds_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("CALIBRATION COMPLETE")
    print("=" * 78)
    for name, r in results.items():
        print(f"{name:10s}  tau_ema={r['tau_p']:.6f}  n={r['n_scores_total']}")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
