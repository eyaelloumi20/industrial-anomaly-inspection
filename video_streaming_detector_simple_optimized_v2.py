#!/usr/bin/env python3
"""
video_streaming_detector_simple_optimized.py
============================================

Simple optimized YOLO + EfficientAD video detector.

Decision logic
--------------
There is NO alarm state, NO FIRED state, NO tau_track, NO inactive_frames,
NO min_samples, and NO track-end decision.

For each visible YOLO/ByteTrack detection:
    raw EfficientAD score
          ↓
    EMA per track
          ↓
    compare with threshold
          ↓
    EMA >= threshold  -> ANOMALY (red)
    EMA <  threshold  -> NORMAL  (green)
    clipped box       -> CLIPPED (grey)

This matches the simple per-frame behaviour of compare_pipeline_methods_batched.py.

Performance design
------------------
1) YOLO + ByteTrack runs once and caches detections + frames.
2) Crops are prepared in chunks across multiple frames.
3) EfficientAD scores crops in GPU microbatches (default 32).
4) Only compact score metadata is saved during inference.
5) The annotated video is rendered afterwards WITHOUT rerunning YOLO/EfficientAD.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import json
import math
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import utils.models as models_module
from compare_pipeline_methods_batched import (
    OUT_CHANNELS,
    load_bundle,
    get_crop,
    run_yolo_once,
    reencode_h264,
)

RED = (0, 0, 255)
GREEN = (0, 200, 0)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)

_NORM_CACHE = {}


# ---------------------------------------------------------------------------
# FP16-safe EfficientAD
# ---------------------------------------------------------------------------

def _dtype_preserving_norm(x):
    key = (x.device.type, x.device.index, x.dtype)
    if key not in _NORM_CACHE:
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        _NORM_CACHE[key] = (mean, std + x.new_tensor(1e-11))
    mean, denom = _NORM_CACHE[key]
    return (x - mean) / denom


models_module.imagenet_norm_batch = _dtype_preserving_norm


def patch_model_forward_dtype_safe(model, kind):
    if kind in ("teacher", "student"):
        if not hasattr(model, "pdn"):
            raise TypeError(f"{kind} model does not expose .pdn")

        def forward(self, x):
            return self.pdn(_dtype_preserving_norm(x))

    elif kind == "ae":
        if not (hasattr(model, "encoder") and hasattr(model, "decoder")):
            raise TypeError("autoencoder does not expose encoder/decoder")

        def forward(self, x):
            return self.decoder(self.encoder(_dtype_preserving_norm(x)))
    else:
        raise ValueError(kind)

    model.forward = types.MethodType(forward, model)


class EfficientADScorer:
    def __init__(self, args):
        self.models, self.image_size, actual_device = load_bundle(
            args.method,
            args.ckpt_dir,
            args.category,
            args.model_size,
            args.device,
        )
        self.image_size = int(self.image_size)
        self.device = torch.device(actual_device)
        self.ratio = float(args.ratio)

        self.dtype = (
            torch.float16
            if args.fp16 and self.device.type == "cuda"
            else torch.float32
        )

        for key in ("teacher", "student", "ae"):
            model = self.models[key]
            model.eval().requires_grad_(False)

            if self.dtype == torch.float16:
                model = model.half()
                patch_model_forward_dtype_safe(model, key)
            else:
                model = model.float()

            model = model.to(memory_format=torch.channels_last)
            self.models[key] = model

        for key in ("mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae"):
            self.models[key] = torch.as_tensor(
                self.models[key],
                device=self.device,
                dtype=self.dtype,
            )

        self.mean = self.models["mean"]
        self.std_denom = self.models["std"] + 1e-8

        self.qa_st = self.models["qa_st"].float()
        self.qb_st = self.models["qb_st"].float()
        self.qa_ae = self.models["qa_ae"].float()
        self.qb_ae = self.models["qb_ae"].float()

        self.st_denom = self.qb_st - self.qa_st + 1e-8
        self.ae_denom = self.qb_ae - self.qa_ae + 1e-8

    @torch.inference_mode()
    def __call__(self, crops_pil):
        if not crops_pil:
            return []

        # One contiguous batch conversion.
        pixels = np.stack(
            [np.asarray(c, dtype=np.uint8) for c in crops_pil],
            axis=0,
        )
        x = torch.from_numpy(
            np.ascontiguousarray(pixels)
        ).permute(0, 3, 1, 2)

        x = x.to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        x = x.div_(255.0).contiguous(
            memory_format=torch.channels_last
        )

        teacher = self.models["teacher"](x)
        student = self.models["student"](x)
        ae = self.models["ae"](x)

        norm_teacher = (teacher - self.mean) / self.std_denom

        d_st = (
            norm_teacher - student[:, :OUT_CHANNELS]
        ).pow(2).mean(dim=1, keepdim=True)

        d_ae = (
            ae - student[:, -OUT_CHANNELS:]
        ).pow(2).mean(dim=1, keepdim=True)

        # Normalize and combine at native feature resolution.
        nm_st = (
            self.ratio * (d_st.float() - self.qa_st)
        ) / self.st_denom

        nm_ae = (
            self.ratio * (d_ae.float() - self.qa_ae)
        ) / self.ae_denom

        combined = 0.5 * nm_st + 0.5 * nm_ae

        # Only ONE interpolation.
        amap = F.interpolate(
            combined,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return amap.flatten(1).amax(1).cpu().tolist()


# ---------------------------------------------------------------------------
# Box filtering
# ---------------------------------------------------------------------------

def classify_boxes(dets, width, height, margin, min_frac):
    if not dets:
        return []

    areas = np.asarray(
        [
            max(0, x2 - x1) * max(0, y2 - y1)
            for _, x1, y1, x2, y2 in dets
        ],
        dtype=np.float64,
    )

    use_size = min_frac > 0 and len(dets) >= 3
    median_area = float(np.median(areas)) if use_size else 0.0

    flags = []
    for i, (_tid, x1, y1, x2, y2) in enumerate(dets):
        edge = (
            x1 <= margin
            or y1 <= margin
            or x2 >= width - margin
            or y2 >= height - margin
        )
        too_small = (
            use_size
            and areas[i] < min_frac * median_area
        )
        flags.append(bool(edge or too_small))

    return flags


# ---------------------------------------------------------------------------
# Chunking + scoring
# ---------------------------------------------------------------------------

def make_chunks(
    detections,
    n_frames,
    max_frames,
    max_crops,
    width,
    height,
    clip_margin,
    min_box_frac,
):
    start = 0

    while start < n_frames:
        end = start
        crop_count = 0

        while end < n_frames and end - start < max_frames:
            dets = detections.get(end, [])
            clipped = classify_boxes(
                dets,
                width,
                height,
                clip_margin,
                min_box_frac,
            )
            n_this = sum(not x for x in clipped)

            if end > start and crop_count + n_this > max_crops:
                break

            crop_count += n_this
            end += 1

        if end == start:
            end = start + 1

        yield start, end
        start = end


def run_scoring(
    args,
    scorer,
    frame_cache,
    detections,
    width,
    height,
    n_frames,
):
    ema_by_track = {}
    frame_records = [None] * n_frames

    crop_pool = (
        ThreadPoolExecutor(max_workers=args.crop_workers)
        if args.crop_workers > 0
        else None
    )

    total_crop_seconds = 0.0
    total_gpu_seconds = 0.0
    total_state_seconds = 0.0
    total_crops = 0

    t_all = time.perf_counter()

    chunks = list(
        make_chunks(
            detections,
            n_frames,
            args.batch_frames,
            args.batch_crops,
            width,
            height,
            args.clip_margin,
            args.min_box_frac,
        )
    )

    try:
        for ci, (start, end) in enumerate(chunks):
            tasks = []
            spans = []
            clipped_by_frame = []

            # Build one task list across multiple frames.
            for fi in range(start, end):
                dets = detections.get(fi, [])
                clipped = classify_boxes(
                    dets,
                    width,
                    height,
                    args.clip_margin,
                    args.min_box_frac,
                )
                clipped_by_frame.append(clipped)

                task_start = len(tasks)

                pil_frame = frame_cache.pil_frames[fi]

                for (_tid, x1, y1, x2, y2), is_clip in zip(
                    dets,
                    clipped,
                ):
                    if is_clip:
                        continue

                    tasks.append(
                        (
                            pil_frame,
                            (x1, y1, x2, y2),
                        )
                    )

                spans.append((task_start, len(tasks)))

            # Crop preprocessing.
            t0 = time.perf_counter()

            def build_crop(task):
                pil, box = task
                return get_crop(
                    pil,
                    box,
                    scorer.image_size,
                    args.pad_ratio,
                    args.denoise,
                )

            if crop_pool is not None:
                crops = list(crop_pool.map(build_crop, tasks))
            else:
                crops = [build_crop(t) for t in tasks]

            total_crop_seconds += time.perf_counter() - t0
            total_crops += len(crops)

            # GPU microbatches.
            t0 = time.perf_counter()
            scores = []

            for mb_start in range(
                0,
                len(crops),
                args.gpu_batch_crops,
            ):
                microbatch = crops[
                    mb_start:mb_start + args.gpu_batch_crops
                ]
                scores.extend(scorer(microbatch))

            total_gpu_seconds += time.perf_counter() - t0

            # Sequential EMA + verdict.
            t0 = time.perf_counter()

            for local_index, fi in enumerate(range(start, end)):
                dets = detections.get(fi, [])
                clipped = clipped_by_frame[local_index]
                score_start, score_end = spans[local_index]
                frame_scores = scores[score_start:score_end]

                raw_out = [None] * len(dets)
                ema_out = [None] * len(dets)
                verdicts = ["CLIPPED" if c else "" for c in clipped]

                score_i = 0

                for j, ((tid, *_box), is_clip) in enumerate(
                    zip(dets, clipped)
                ):
                    if is_clip:
                        continue

                    raw = float(frame_scores[score_i])
                    score_i += 1

                    previous = ema_by_track.get(tid, raw)
                    ema = (
                        args.ema_alpha * raw
                        + (1.0 - args.ema_alpha) * previous
                    )
                    ema_by_track[tid] = ema

                    raw_out[j] = raw
                    ema_out[j] = ema

                    decision_score = (
                        raw if args.threshold_basis == "raw" else ema
                    )

                    verdicts[j] = (
                        "ANOMALY"
                        if decision_score >= args.threshold
                        else "NORMAL"
                    )

                frame_records[fi] = {
                    "detections": [
                        list(map(int, d)) for d in dets
                    ],
                    "clipped": [bool(v) for v in clipped],
                    "raw": raw_out,
                    "ema": ema_out,
                    "verdict": verdicts,
                }

            total_state_seconds += time.perf_counter() - t0

            done = end

            if (
                args.log_every > 0
                and (
                    done % args.log_every == 0
                    or ci == len(chunks) - 1
                )
            ):
                elapsed = time.perf_counter() - t_all
                print(
                    f"[Score] {done}/{n_frames} frames | "
                    f"{done / max(elapsed, 1e-9):.2f} fps | "
                    f"crops={total_crops}"
                )

    finally:
        if crop_pool is not None:
            crop_pool.shutdown(wait=True)

    total_seconds = time.perf_counter() - t_all

    return {
        "records": frame_records,
        "seconds": total_seconds,
        "crop_seconds": total_crop_seconds,
        "gpu_seconds": total_gpu_seconds,
        "state_seconds": total_state_seconds,
        "n_crops": total_crops,
    }


# ---------------------------------------------------------------------------
# Video + CSV
# ---------------------------------------------------------------------------

def draw_box(frame, box, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        1,
    )

    tx = max(0, x1 + 3)
    ty = max(th + 6, y1 + 3)

    cv2.rectangle(
        frame,
        (tx - 2, ty - th - 4),
        (tx + tw + 4, ty + 2),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def render_video(
    args,
    frame_cache,
    records,
    src_fps,
    width,
    height,
):
    if args.no_video:
        return None, 0.0

    raw_path = os.path.join(
        args.save_dir,
        "annotated_raw.mp4",
    )
    final_path = os.path.join(
        args.save_dir,
        "annotated.mp4",
    )

    writer = cv2.VideoWriter(
        raw_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create output video: {raw_path}"
        )

    t0 = time.perf_counter()

    for fi, src_frame in enumerate(frame_cache.bgr_frames):
        frame = src_frame.copy()
        rec = records[fi]

        anomaly_count = 0

        for detection, raw, ema, verdict in zip(
            rec["detections"],
            rec["raw"],
            rec["ema"],
            rec["verdict"],
        ):
            tid, x1, y1, x2, y2 = detection

            if verdict == "ANOMALY":
                color = RED
                thickness = 3
                anomaly_count += 1
            elif verdict == "CLIPPED":
                color = GREY
                thickness = 2
            else:
                color = GREEN
                thickness = 2

            if verdict == "CLIPPED":
                label = f"ID{tid} CLIPPED"
            else:
                shown_score = (
                    raw
                    if args.threshold_basis == "raw"
                    else ema
                )
                label = (
                    f"ID{tid} {verdict} "
                    f"{0.0 if shown_score is None else shown_score:.2f}"
                )

            draw_box(
                frame,
                (x1, y1, x2, y2),
                label,
                color,
                thickness,
            )

        banner_color = RED if anomaly_count else GREEN

        cv2.putText(
            frame,
            f"{args.method} | frame {fi} | "
            f"anomalies={anomaly_count}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            banner_color,
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()

    if reencode_h264(raw_path, final_path):
        os.remove(raw_path)
        output_path = final_path
    else:
        output_path = raw_path

    seconds = time.perf_counter() - t0

    print(f"[Video] Saved: {output_path}")

    return output_path, seconds


def save_csv(args, records):
    path = os.path.join(
        args.save_dir,
        "scores.csv",
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "track_id",
                "raw_score",
                "ema_score",
                "verdict",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )

        for fi, rec in enumerate(records):
            for detection, raw, ema, verdict in zip(
                rec["detections"],
                rec["raw"],
                rec["ema"],
                rec["verdict"],
            ):
                tid, x1, y1, x2, y2 = detection

                writer.writerow(
                    [
                        fi,
                        tid,
                        "" if raw is None else f"{raw:.6f}",
                        "" if ema is None else f"{ema:.6f}",
                        verdict,
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )

    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Simple optimized per-frame YOLO + EfficientAD detector"
    )

    p.add_argument("--video", required=True)
    p.add_argument("--yolo_det_weights", required=True)

    p.add_argument(
        "--method",
        default="baseline",
        choices=["baseline", "lowres", "thin", "qat", "pruned"],
    )

    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--thresholds_json", default=None,
                   help="Optional threshold JSON. Not required when --threshold is supplied.")

    p.add_argument("--category", default="chewinggum")
    p.add_argument(
        "--model_size",
        default="S",
        choices=["S", "M"],
    )
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--fp16", action="store_true")

    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise", action="store_true")

    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold JSON. Default uses tau_p.",
    )

    p.add_argument(
        "--threshold_basis",
        choices=["raw", "ema"],
        default="ema",
        help="Compare threshold against current raw or EMA score.",
    )

    p.add_argument("--ema_alpha", type=float, default=0.3)

    p.add_argument("--batch_frames", type=int, default=16)
    p.add_argument(
        "--batch_crops",
        type=int,
        default=96,
        help="Maximum crops gathered across several frames.",
    )
    p.add_argument(
        "--gpu_batch_crops",
        type=int,
        default=32,
        help="EfficientAD GPU microbatch size.",
    )
    p.add_argument("--crop_workers", type=int, default=12)

    p.add_argument("--clip_margin", type=int, default=5)
    p.add_argument("--min_box_frac", type=float, default=0.5)

    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--log_every", type=int, default=60)

    p.add_argument(
        "--no_detect_unknown",
        action="store_true",
        help="Disable Otsu foreign-object computation.",
    )

    p.add_argument(
        "--no_video",
        action="store_true",
        help="Skip video rendering for benchmark.",
    )

    p.add_argument(
        "--save_dir",
        default="./simple_optimized_outputs",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    if args.gpu_batch_crops < 1:
        raise ValueError("--gpu_batch_crops must be >= 1")

    if args.batch_frames < 1 or args.batch_crops < 1:
        raise ValueError("batch sizes must be >= 1")

    os.makedirs(args.save_dir, exist_ok=True)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    threshold_data = {}

    if args.thresholds_json:
        if not os.path.isfile(args.thresholds_json):
            raise FileNotFoundError(
                f"Threshold JSON not found: {args.thresholds_json}\n"
                "Either provide the correct file or pass --threshold VALUE "
                "without --thresholds_json for speed benchmarking."
            )

        with open(
            args.thresholds_json,
            encoding="utf-8",
        ) as f:
            threshold_data = json.load(f)

    if args.threshold is None:
        if not threshold_data:
            raise ValueError(
                "No threshold provided. Use either "
                "--thresholds_json <file.json> or --threshold <value>."
            )
        args.threshold = float(threshold_data["tau_p"])

    print("=" * 72)
    print("SIMPLE OPTIMIZED DETECTOR")
    print("=" * 72)
    print("Decision logic : current score / EMA only")
    print("Alarm logic    : REMOVED")
    print("FIRED logic    : REMOVED")
    print("Track-end wait : REMOVED")
    print(
        f"threshold      : {args.threshold:.4f} "
        f"({args.threshold_basis})"
    )
    print(
        f"batch          : frames={args.batch_frames}, "
        f"chunk_crops={args.batch_crops}, "
        f"gpu_crops={args.gpu_batch_crops}"
    )

    # ------------------------------------------------------------------
    # Phase 1: trusted YOLO/ByteTrack path from comparison script.
    # ------------------------------------------------------------------
    yolo_t0 = time.perf_counter()

    (
        detections,
        width,
        height,
        src_fps,
        n_frames,
        frame_cache,
    ) = run_yolo_once(
        args.video,
        args.yolo_det_weights,
        args.conf,
        args.iou,
        device=args.device,
        max_frames=(args.max_frames or None),
        detect_unknown=not args.no_detect_unknown,
        otsu_max_area_frac=0.25,
        cache_frames=True,
    )

    yolo_seconds = time.perf_counter() - yolo_t0

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------
    scorer = EfficientADScorer(args)

    # Warm-up at benchmark-friendly batch size.
    warm_n = min(
        args.gpu_batch_crops,
        16,
    )
    dummy = Image.new(
        "RGB",
        (scorer.image_size, scorer.image_size),
    )
    _ = scorer([dummy] * warm_n)

    print(
        f"[EfficientAD] Warm-up complete "
        f"(image_size={scorer.image_size}, "
        f"dtype={scorer.dtype})"
    )

    result = run_scoring(
        args,
        scorer,
        frame_cache,
        detections,
        width,
        height,
        n_frames,
    )

    # ------------------------------------------------------------------
    # Output artifacts
    # ------------------------------------------------------------------
    csv_path = save_csv(
        args,
        result["records"],
    )

    video_path, render_seconds = render_video(
        args,
        frame_cache,
        result["records"],
        src_fps,
        width,
        height,
    )

    yolo_fps = n_frames / max(yolo_seconds, 1e-9)
    scoring_fps = n_frames / max(result["seconds"], 1e-9)

    sequential_seconds = (
        yolo_seconds + result["seconds"]
    )
    sequential_fps = (
        n_frames / max(sequential_seconds, 1e-9)
    )

    summary = {
        "n_frames": n_frames,
        "n_crops": result["n_crops"],
        "threshold": args.threshold,
        "threshold_basis": args.threshold_basis,
        "yolo_seconds": yolo_seconds,
        "yolo_fps": yolo_fps,
        "scoring_seconds": result["seconds"],
        "scoring_fps": scoring_fps,
        "crop_seconds": result["crop_seconds"],
        "efficientad_seconds": result["gpu_seconds"],
        "state_seconds": result["state_seconds"],
        "sequential_seconds": sequential_seconds,
        "sequential_fps": sequential_fps,
        "render_seconds": render_seconds,
        "video": video_path,
        "csv": csv_path,
        "settings": vars(args),
    }

    summary_path = os.path.join(
        args.save_dir,
        "summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=False,
        )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(
        f"YOLO + ByteTrack      : {yolo_seconds:8.2f}s | "
        f"{yolo_fps:7.2f} FPS"
    )
    print(
        f"Scoring phase         : {result['seconds']:8.2f}s | "
        f"{scoring_fps:7.2f} FPS"
    )
    print(
        f"  crop preprocessing  : {result['crop_seconds']:8.2f}s"
    )
    print(
        f"  EfficientAD         : {result['gpu_seconds']:8.2f}s | "
        f"{result['n_crops'] / max(result['gpu_seconds'], 1e-9):7.1f} crops/s"
    )
    print(
        f"  EMA/verdict         : {result['state_seconds']:8.2f}s"
    )
    print(
        f"Sequential compute    : {sequential_seconds:8.2f}s | "
        f"{sequential_fps:7.2f} FPS"
    )
    print(
        f"Threshold             : {args.threshold:.4f} "
        f"({args.threshold_basis})"
    )
    print(f"Video                 : {video_path}")
    print(f"CSV                   : {csv_path}")
    print(f"Summary               : {summary_path}")


if __name__ == "__main__":
    main()
