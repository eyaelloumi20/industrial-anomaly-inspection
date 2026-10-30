#!/usr/bin/env python3
"""
benchmark_video_fps.py

Runs a set of YOLO model files (.pt / .onnx / .engine) on a video and
reports FPS for each, both detect-only and detect+track (ByteTrack).

Usage:
    python benchmark_video_fps.py --video /path/to/video.mp4

    # Or point it at specific weights explicitly:
    python benchmark_video_fps.py --video clip.mp4 \
        --weights runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.pt \
                  runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.onnx \
                  runs/detect/yolo_runs/chewinggum_yolov8n_v2/weights/best.engine

If --weights is not given, the script auto-discovers best.pt / best.onnx /
best.engine under --runs-dir for the three architectures (yolov8n_v2,
yolo11n, yolo26n).

Notes:
- .engine files are hardware-specific — only benchmark them on the machine
  that built them.
- warmup runs (default 10 frames) are excluded from timing so CUDA/TensorRT
  context init doesn't skew the numbers.
- --no-track skips the ByteTrack pass if you only care about raw detection speed.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. pip install ultralytics", file=sys.stderr)
    sys.exit(1)


DEFAULT_ARCHS = ["chewinggum_yolov8n_v2", "chewinggum_yolo11n", "chewinggum_yolo26n"]
DEFAULT_FORMATS = ["best.pt", "best.onnx", "best.engine"]


def discover_weights(runs_dir: Path):
    """Find best.pt/onnx/engine under runs_dir/<arch>/weights/."""
    found = []
    for arch in DEFAULT_ARCHS:
        weights_dir = runs_dir / arch / "weights"
        if not weights_dir.exists():
            print(f"  (skip) {weights_dir} not found")
            continue
        for fname in DEFAULT_FORMATS:
            fpath = weights_dir / fname
            if fpath.exists():
                found.append(fpath)
            else:
                print(f"  (skip) {fpath} not found")
    return found


def benchmark_one(weights_path: Path, video_path: Path, imgsz: int, device: str,
                   warmup: int, do_track: bool, conf: float):
    """Runs a single weights file over the video, returns a dict of results."""
    label = f"{weights_path.parent.parent.name}/{weights_path.name}"
    print(f"\n=== {label} ===")

    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return {"model": label, "error": str(e)}

    result_row = {"model": label, "format": weights_path.suffix.lstrip(".")}

    # ---- Detect-only pass ----
    try:
        n_frames = 0
        t_start = None
        stream = model.predict(
            source=str(video_path), imgsz=imgsz, device=device,
            conf=conf, stream=True, verbose=False,
        )
        for i, _ in enumerate(stream):
            if i == warmup:
                t_start = time.perf_counter()  # start timing after warmup
            if t_start is not None:
                n_frames += 1
        t_elapsed = time.perf_counter() - t_start if t_start else 0
        fps_detect = n_frames / t_elapsed if t_elapsed > 0 else float("nan")
        result_row["detect_fps"] = round(fps_detect, 2)
        result_row["detect_frames_timed"] = n_frames
        print(f"  detect-only:      {fps_detect:.2f} FPS  ({n_frames} frames timed)")
    except Exception as e:
        print(f"  detect-only FAILED: {e}")
        result_row["detect_fps"] = None
        result_row["detect_error"] = str(e)

    # ---- Detect+track pass ----
    if do_track:
        try:
            n_frames = 0
            t_start = None
            stream = model.track(
                source=str(video_path), imgsz=imgsz, device=device,
                conf=conf, tracker="bytetrack.yaml", stream=True,
                persist=True, verbose=False,
            )
            for i, _ in enumerate(stream):
                if i == warmup:
                    t_start = time.perf_counter()
                if t_start is not None:
                    n_frames += 1
            t_elapsed = time.perf_counter() - t_start if t_start else 0
            fps_track = n_frames / t_elapsed if t_elapsed > 0 else float("nan")
            result_row["track_fps"] = round(fps_track, 2)
            result_row["track_frames_timed"] = n_frames
            print(f"  detect+track:      {fps_track:.2f} FPS  ({n_frames} frames timed)")
        except Exception as e:
            print(f"  detect+track FAILED: {e}")
            result_row["track_fps"] = None
            result_row["track_error"] = str(e)

    return result_row


def main():
    ap = argparse.ArgumentParser(description="Benchmark FPS of multiple YOLO model formats on a video")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--weights", nargs="+", default=None,
                     help="Explicit list of weight files (.pt/.onnx/.engine) to benchmark. "
                          "If omitted, auto-discovers under --runs-dir.")
    ap.add_argument("--runs-dir", default="runs/detect/yolo_runs",
                     help="Base dir to auto-discover weights from (default: runs/detect/yolo_runs)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="'0' for GPU 0, 'cpu' for CPU")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=10, help="Frames to discard before timing starts")
    ap.add_argument("--no-track", action="store_true", help="Skip the ByteTrack (detect+track) pass")
    ap.add_argument("--out-csv", default="fps_benchmark_results.csv")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if args.weights:
        weights_list = [Path(w) for w in args.weights]
        missing = [w for w in weights_list if not w.exists()]
        if missing:
            print("ERROR: missing weight files:", file=sys.stderr)
            for m in missing:
                print(f"  {m}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Auto-discovering weights under {args.runs_dir} ...")
        weights_list = discover_weights(Path(args.runs_dir))
        if not weights_list:
            print("ERROR: no weight files found. Pass --weights explicitly.", file=sys.stderr)
            sys.exit(1)

    print(f"\nVideo: {video_path}")
    print(f"Formats to benchmark: {len(weights_list)}")
    print(f"imgsz={args.imgsz} device={args.device} conf={args.conf} warmup={args.warmup} track={not args.no_track}")

    results = []
    for w in weights_list:
        row = benchmark_one(
            weights_path=w,
            video_path=video_path,
            imgsz=args.imgsz,
            device=args.device,
            warmup=args.warmup,
            do_track=not args.no_track,
            conf=args.conf,
        )
        results.append(row)

    # ---- Write CSV ----
    fieldnames = ["model", "format", "detect_fps", "detect_frames_timed",
                  "track_fps", "track_frames_timed", "detect_error", "track_error", "error"]
    out_path = Path(args.out_csv)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # ---- Print summary table ----
    print("\n" + "=" * 70)
    print(f"{'MODEL':<45}{'DETECT FPS':>12}{'TRACK FPS':>12}")
    print("-" * 70)
    for row in results:
        det = row.get("detect_fps")
        trk = row.get("track_fps")
        det_str = f"{det:.2f}" if isinstance(det, (int, float)) else "FAIL"
        trk_str = f"{trk:.2f}" if isinstance(trk, (int, float)) else ("FAIL" if not args.no_track else "-")
        print(f"{row['model']:<45}{det_str:>12}{trk_str:>12}")
    print("=" * 70)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
