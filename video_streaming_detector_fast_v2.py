#!/usr/bin/env python3
"""
video_streaming_detector_fast_v2.py
===================================

Goal
----
High-throughput YOLO + EfficientAD inspection with BOTH:
  1) an EARLY review video showing provisional red alerts from the exact frame
     where the alert became available, and
  2) a FINAL retrospective video where every confirmed anomalous track is red
     from its first appearance.

Performance architecture
------------------------
Phase 1: YOLO + ByteTrack ONCE; cache BGR frames + track boxes.
Phase 2: Build crops from BGR frames with NumPy/OpenCV, batch crops ACROSS
         multiple frames, score EfficientAD in FP16, update EMA/track state.
         NO annotation and NO video writing occur in this inference timing.
Phase 3: Render videos from cached frames + tiny metadata. Models are NOT rerun.

This script reports separate timing for:
  - YOLO/ByteTrack
  - crop preprocessing
  - EfficientAD scoring
  - state bookkeeping
  - Phase-2 wall time
  - honest sequential compute FPS = YOLO + Phase 2
  - video rendering (excluded from AI compute FPS)

Notes
-----
- Default crop backend is OpenCV for speed. OpenCV interpolation is very close
  to PIL Lanczos but not bit-identical. Revalidate/calibrate thresholds before
  production if the score distribution changes materially.
- --crop_backend pil uses the project's original get_crop() path for maximum
  preprocessing compatibility, but it is slower.
- The script processes every frame. It does not use vid_stride.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import collections
import importlib
import json
import math
import os
import subprocess
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

import utils.models as models_module
from compare_pipeline_methods_batched import (
    OUT_CHANNELS,
    get_crop as pil_get_crop,
)

RED = (0, 0, 255)
GREEN = (0, 200, 0)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)

_NORM_CACHE = {}


# ---------------------------------------------------------------------------
# EfficientAD model helpers
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
            raise TypeError("Expected teacher/student model with .pdn")

        def forward(self, x):
            return self.pdn(_dtype_preserving_norm(x))

    elif kind == "ae":
        if not (hasattr(model, "encoder") and hasattr(model, "decoder")):
            raise TypeError("Expected autoencoder with .encoder/.decoder")

        def forward(self, x):
            return self.decoder(self.encoder(_dtype_preserving_norm(x)))

    else:
        raise ValueError(kind)

    model.forward = types.MethodType(forward, model)


def load_any_bundle(ckpt_dir, category, model_size, device):
    """Auto-detect baseline / lowres / thin / pruned / QAT checkpoints."""
    ckpt = Path(ckpt_dir)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    qfile = ckpt / f"{category}_quantiles.npy"
    q = None
    if qfile.is_file():
        try:
            q = np.load(qfile, allow_pickle=True).item()
        except Exception as exc:
            print(f"[Load] Could not inspect quantiles: {exc}")

    # QAT / int8
    if (ckpt / f"{category}_student_int8.jit").is_file() or \
       (ckpt / f"{category}_student_int8.pth").is_file():
        print("[Load] QAT/int8 checkpoint detected")
        mod = importlib.import_module("utils.load_efficientad_qat")
        models = mod.load_efficientad_model(
            str(ckpt), category, model_size, "cpu"
        )
        return models, int(models.get("image_size", 256)), "cpu"

    # Pruned
    if (ckpt / f"{category}_student_pruned.pt").is_file():
        print("[Load] Pruned checkpoint detected")
        mod = importlib.import_module("utils.load_efficientad_pruned")
        models = mod.load_efficientad_model(
            str(ckpt), category, model_size, device
        )
        return models, int(models.get("image_size", 256)), device

    # Thin
    if q is not None and "width_mult" in q and float(q["width_mult"]) != 1.0:
        print(f"[Load] Thin checkpoint detected (width_mult={q['width_mult']})")
        mod = importlib.import_module("utils.load_efficientad_thin")
        models = mod.load_efficientad_model(
            str(ckpt), category, model_size, device
        )
        return models, int(models.get("image_size", 256)), device

    # Low-res adaptive
    if q is not None and "image_size" in q and int(q["image_size"]) != 256:
        print(f"[Load] Low-res checkpoint detected (image_size={q['image_size']})")
        mod = importlib.import_module("utils.load_efficientad_adaptive")
        models = mod.load_efficientad_model(
            str(ckpt), category, model_size, device
        )
        return models, int(models.get("image_size", q["image_size"])), device

    # Baseline: adaptive loader also supports normal 256-px checkpoints in
    # this project and keeps one loading path.
    print("[Load] Baseline checkpoint detected")
    mod = importlib.import_module("utils.load_efficientad_adaptive")
    models = mod.load_efficientad_model(
        str(ckpt), category, model_size, device
    )
    return models, int(models.get("image_size", 256)), device


class FastScorer:
    def __init__(self, args):
        models, image_size, actual_device = load_any_bundle(
            args.ckpt_dir, args.category, args.model_size, args.device
        )
        self.models = models
        self.image_size = int(image_size)
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
        self.qa_ae = self.models["qa_ae"].float()
        self.st_denom = (
            self.models["qb_st"].float() - self.qa_st + 1e-8
        )
        self.ae_denom = (
            self.models["qb_ae"].float() - self.qa_ae + 1e-8
        )

    @torch.inference_mode()
    def score_rgb_uint8(self, rgb_batch):
        """rgb_batch: contiguous uint8 [N,H,W,3]."""
        if len(rgb_batch) == 0:
            return []

        arr = np.ascontiguousarray(rgb_batch, dtype=np.uint8)
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)
        x = x.to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        ).div_(255.0)
        x = x.contiguous(memory_format=torch.channels_last)

        teacher = self.models["teacher"](x)
        student = self.models["student"](x)
        ae = self.models["ae"](x)

        norm_teacher = (teacher - self.mean) / self.std_denom
        d_st = (
            norm_teacher - student[:, :OUT_CHANNELS]
        ).pow(2).mean(1, keepdim=True)
        d_ae = (
            ae - student[:, -OUT_CHANNELS:]
        ).pow(2).mean(1, keepdim=True)

        nm_st = (
            self.ratio * (d_st.float() - self.qa_st)
        ) / self.st_denom
        nm_ae = (
            self.ratio * (d_ae.float() - self.qa_ae)
        ) / self.ae_denom

        combined = 0.5 * nm_st + 0.5 * nm_ae

        # ONE interpolation rather than separate ST + AE upsamplings.
        amap = F.interpolate(
            combined,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return amap.flatten(1).amax(1).cpu().tolist()


# ---------------------------------------------------------------------------
# Video / YOLO caching
# ---------------------------------------------------------------------------

def run_yolo_cache(args):
    print(f"[YOLO] Loading {args.yolo_det_weights}")
    model = YOLO(args.yolo_det_weights)
    model.to(args.device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(src_fps) or src_fps <= 0:
        src_fps = 25.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    detections = {}
    frame_idx = 0
    fallback_id = -1

    t0 = time.perf_counter()
    precision_note_printed = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        kwargs = dict(
            source=frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.yolo_imgsz,
            max_det=args.yolo_max_det,
            verbose=False,
        )

        # Newer Ultralytics uses quantize=16; older installs commonly use
        # half=True. Try the current API first and transparently fall back.
        if args.yolo_fp16 and str(args.device).startswith("cuda"):
            try:
                results = model.track(**kwargs, quantize=16)
                if not precision_note_printed:
                    print("[YOLO] 16-bit inference: quantize=16")
                    precision_note_printed = True
            except (TypeError, ValueError, SyntaxError):
                results = model.track(**kwargs, half=True)
                if not precision_note_printed:
                    print("[YOLO] 16-bit inference: half=True compatibility path")
                    precision_note_printed = True
        else:
            results = model.track(**kwargs)

        dets = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.int32)
            ids = (
                result.boxes.id.detach().cpu().numpy().astype(np.int64)
                if result.boxes.id is not None
                else None
            )

            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(width, int(x2))
                y2 = min(height, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                if ids is None:
                    tid = fallback_id
                    fallback_id -= 1
                else:
                    tid = int(ids[i])
                dets.append((tid, x1, y1, x2, y2))

        frames.append(frame)
        detections[frame_idx] = dets
        frame_idx += 1

        if frame_idx % args.log_every == 0:
            dt = time.perf_counter() - t0
            print(
                f"[YOLO] frame {frame_idx}/{total} | "
                f"{frame_idx / max(dt, 1e-9):.2f} fps"
            )

    cap.release()
    seconds = time.perf_counter() - t0
    fps = frame_idx / max(seconds, 1e-9)

    print(
        f"[YOLO] Done: {frame_idx} frames in {seconds:.2f}s "
        f"({fps:.2f} fps), imgsz={args.yolo_imgsz}, "
        f"max_det={args.yolo_max_det}"
    )
    return frames, detections, width, height, src_fps, seconds


# ---------------------------------------------------------------------------
# Crop preparation
# ---------------------------------------------------------------------------

def classify_boxes(dets, width, height, margin, min_frac):
    if not dets:
        return []

    areas = np.asarray(
        [(x2 - x1) * (y2 - y1) for _, x1, y1, x2, y2 in dets],
        dtype=np.float64,
    )
    use_size = min_frac > 0 and len(dets) >= 3
    median_area = float(np.median(areas)) if use_size else 0.0

    out = []
    for i, (_tid, x1, y1, x2, y2) in enumerate(dets):
        edge = (
            x1 <= margin
            or y1 <= margin
            or x2 >= width - margin
            or y2 >= height - margin
        )
        too_small = use_size and areas[i] < min_frac * median_area
        out.append(bool(edge or too_small))
    return out


def crop_opencv_rgb(frame_bgr, box, out_size, pad_ratio, denoise):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    ch, cw = crop.shape[:2]
    scale = out_size / max(cw, ch)
    nw = max(1, int(cw * scale))
    nh = max(1, int(ch * scale))

    # Fast but high-quality resize. Downsample normally uses AREA; otherwise
    # LANCZOS4 is a close OpenCV analogue to PIL's Lanczos.
    interpolation = (
        cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    )
    resized = cv2.resize(crop, (nw, nh), interpolation=interpolation)

    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    ox = (out_size - nw) // 2
    oy = (out_size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized

    if denoise:
        canvas = cv2.GaussianBlur(canvas, (3, 3), sigmaX=0.8)
        canvas = cv2.bilateralFilter(
            canvas, d=5, sigmaColor=15, sigmaSpace=15
        )

    # Model expects RGB.
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def crop_pil_rgb(frame_bgr, box, out_size, pad_ratio, denoise):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img = pil_get_crop(pil, box, out_size, pad_ratio, denoise)
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------

class TrackManager:
    def __init__(self, args):
        self.args = args
        self.alpha = float(args.ema_alpha)
        self.buffers = {}
        self.ema = {}
        self.last_seen = {}
        self.decided = set()
        self.final_scores = {}
        self.fired = set()

        self.early_from = {}
        self.early_scores = {}
        self.early_streak = {}

    def _score(self, tid):
        buf = self.buffers.get(tid)
        if not buf or len(buf) < self.args.min_samples:
            return None
        if len(buf) < 25:
            return float(max(buf))
        return float(np.median(buf))

    def _finalize_tid(self, tid, frame_idx, reason):
        if tid in self.decided:
            return
        score = self._score(tid)
        self.decided.add(tid)
        self.final_scores[tid] = score

        if score is not None and score >= self.args.tau_track:
            self.fired.add(int(tid))
            print(
                f"[Track] FIRED frame {frame_idx}: ID{tid} "
                f"score={score:.3f} ({reason})"
            )

    def step_frame(self, frame_idx, dets, clipped, scores):
        if len(scores) != sum(not c for c in clipped):
            raise RuntimeError("Score/detection alignment error")

        emas = [None] * len(dets)
        si = 0

        for i, ((tid, *_box), is_clip) in enumerate(zip(dets, clipped)):
            if is_clip:
                continue

            raw = float(scores[si])
            si += 1

            prev = self.ema.get(tid, raw)
            ema = self.alpha * raw + (1.0 - self.alpha) * prev
            self.ema[tid] = ema
            self.last_seen[tid] = frame_idx
            self.buffers.setdefault(
                tid, collections.deque(maxlen=200)
            ).append(ema)
            emas[i] = ema

            if (
                self.args.early_fire
                and tid not in self.decided
                and tid not in self.early_from
            ):
                candidate = self._score(tid)
                qualifies = (
                    candidate is not None
                    and candidate >= self.args.tau_track
                    and ema >= self.args.tau_track
                )
                streak = self.early_streak.get(tid, 0)
                streak = streak + 1 if qualifies else 0
                self.early_streak[tid] = streak

                if streak >= self.args.early_consecutive:
                    self.early_from[int(tid)] = int(frame_idx)
                    self.early_scores[int(tid)] = float(candidate)
                    print(
                        f"[Track] EARLY frame {frame_idx}: ID{tid} "
                        f"score={candidate:.3f} (provisional)"
                    )

        # Tracks that have been absent for long enough get final verdicts.
        for tid, last in list(self.last_seen.items()):
            if tid in self.decided:
                continue
            if frame_idx - last >= self.args.inactive_frames:
                self._finalize_tid(tid, frame_idx, "inactive")

        return emas

    def finish_all(self, last_frame):
        for tid in list(self.last_seen):
            self._finalize_tid(tid, last_frame, "end_of_video")


# ---------------------------------------------------------------------------
# Batched Phase 2
# ---------------------------------------------------------------------------

def make_chunks(frames, detections, width, height, args):
    start = 0
    n_frames = len(frames)

    while start < n_frames:
        end = start
        crops = 0

        while (
            end < n_frames
            and (end - start) < args.batch_frames
        ):
            dets = detections.get(end, [])
            clipped = classify_boxes(
                dets,
                width,
                height,
                args.clip_margin,
                args.min_box_frac,
            )
            n_this = sum(not c for c in clipped)

            if (
                end > start
                and crops + n_this > args.batch_crops
            ):
                break

            crops += n_this
            end += 1

        if end == start:
            end += 1

        yield start, end
        start = end


def run_efficientad_phase(
    frames, detections, width, height, scorer, args
):
    manager = TrackManager(args)
    n_frames = len(frames)

    # Small metadata only; rendering is deferred.
    frame_meta = [None] * n_frames

    crop_seconds = 0.0
    score_seconds = 0.0
    state_seconds = 0.0
    n_crops_total = 0

    pool = (
        ThreadPoolExecutor(max_workers=args.crop_workers)
        if args.crop_workers > 0
        else None
    )

    wall_t0 = time.perf_counter()

    try:
        chunks = list(
            make_chunks(
                frames, detections, width, height, args
            )
        )

        for chunk_idx, (a, b) in enumerate(chunks):
            tasks = []
            spans = []
            clipped_per_frame = []

            for fi in range(a, b):
                dets = detections.get(fi, [])
                clipped = classify_boxes(
                    dets,
                    width,
                    height,
                    args.clip_margin,
                    args.min_box_frac,
                )
                clipped_per_frame.append(clipped)

                s = len(tasks)
                for (_tid, x1, y1, x2, y2), is_clip in zip(
                    dets, clipped
                ):
                    if not is_clip:
                        tasks.append(
                            (fi, (x1, y1, x2, y2))
                        )
                spans.append((s, len(tasks)))

            # -------- crop preprocessing --------
            t_crop = time.perf_counter()

            def do_crop(task):
                fi, box = task
                if args.crop_backend == "opencv":
                    return crop_opencv_rgb(
                        frames[fi],
                        box,
                        scorer.image_size,
                        args.pad_ratio,
                        args.denoise,
                    )
                return crop_pil_rgb(
                    frames[fi],
                    box,
                    scorer.image_size,
                    args.pad_ratio,
                    args.denoise,
                )

            if tasks:
                if pool is not None:
                    crop_list = list(pool.map(do_crop, tasks))
                else:
                    crop_list = [do_crop(t) for t in tasks]

                crop_batch = np.stack(
                    crop_list, axis=0
                ).astype(np.uint8, copy=False)
            else:
                crop_batch = np.empty(
                    (0, scorer.image_size, scorer.image_size, 3),
                    dtype=np.uint8,
                )

            crop_seconds += time.perf_counter() - t_crop
            n_crops_total += len(tasks)

            # -------- EfficientAD GPU inference --------
            t_score = time.perf_counter()
            scores = scorer.score_rgb_uint8(crop_batch)
            score_seconds += time.perf_counter() - t_score

            # -------- sequential track state --------
            t_state = time.perf_counter()

            for local_i, fi in enumerate(range(a, b)):
                s, e = spans[local_i]
                dets = detections.get(fi, [])
                clipped = clipped_per_frame[local_i]

                emas = manager.step_frame(
                    fi,
                    dets,
                    clipped,
                    scores[s:e],
                )

                frame_meta[fi] = {
                    "clipped": clipped,
                    "emas": emas,
                    "early_ids": sorted(
                        tid
                        for tid, from_frame in manager.early_from.items()
                        if from_frame <= fi
                    ),
                }

            state_seconds += time.perf_counter() - t_state

            done = b
            if done % args.log_every == 0 or chunk_idx == len(chunks) - 1:
                wall = time.perf_counter() - wall_t0
                print(
                    f"[EfficientAD] {done}/{n_frames} frames | "
                    f"{done / max(wall, 1e-9):.2f} phase2 fps | "
                    f"crops={n_crops_total}"
                )

        manager.finish_all(n_frames - 1)

    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    wall_seconds = time.perf_counter() - wall_t0

    return (
        manager,
        frame_meta,
        wall_seconds,
        crop_seconds,
        score_seconds,
        state_seconds,
        n_crops_total,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_box(frame, box, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(
        frame, (x1, y1), (x2, y2), color, thickness
    )

    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        1,
    )
    tx = max(0, min(x1 + 3, frame.shape[1] - tw - 5))
    ty = max(th + 6, min(y1 + 3, frame.shape[0] - 3))

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


def reencode_h264(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-loglevel",
        "error",
        output_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return (
            proc.returncode == 0
            and os.path.isfile(output_path)
            and os.path.getsize(output_path) > 1024
        )
    except FileNotFoundError:
        print("[Video] ffmpeg not installed; keeping mp4v output.")
        return False


def render_review(
    kind,
    frames,
    detections,
    frame_meta,
    manager,
    fps,
    save_dir,
):
    assert kind in ("early", "final")

    raw = os.path.join(
        save_dir, f"annotated_{kind}_raw.mp4"
    )
    final = os.path.join(
        save_dir, f"annotated_{kind}_h264.mp4"
    )

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        raw,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {raw}")

    t0 = time.perf_counter()

    for fi, src in enumerate(frames):
        frame = src.copy()
        dets = detections.get(fi, [])
        meta = frame_meta[fi] or {
            "clipped": [False] * len(dets),
            "emas": [None] * len(dets),
            "early_ids": [],
        }

        clipped = meta["clipped"]
        emas = meta["emas"]

        for idx, ((tid, x1, y1, x2, y2), is_clip) in enumerate(
            zip(dets, clipped)
        ):
            tid = int(tid)

            if kind == "final" and tid in manager.fired:
                score = manager.final_scores.get(tid)
                label = (
                    f"ID{tid} ANOMALY "
                    f"{0.0 if score is None else score:.2f}"
                )
                color = RED
                thickness = 4

            elif (
                kind == "early"
                and tid in manager.early_from
                and fi >= manager.early_from[tid]
            ):
                score = manager.early_scores.get(tid, 0.0)
                label = f"ID{tid} EARLY {score:.2f}"
                color = RED
                thickness = 3

            elif is_clip:
                label = f"ID{tid} CLIP"
                color = GREY
                thickness = 2

            else:
                ema = emas[idx] if idx < len(emas) else None
                label = (
                    f"ID{tid}"
                    if ema is None
                    else f"ID{tid} {ema:.2f}"
                )
                color = GREEN
                thickness = 2

            draw_box(
                frame,
                (x1, y1, x2, y2),
                label,
                color,
                thickness,
            )

        if kind == "final":
            visible_red = sorted(
                int(tid)
                for tid, *_ in dets
                if int(tid) in manager.fired
            )
            banner = (
                f"FINAL | frame {fi} | "
                f"anomaly IDs={visible_red if visible_red else 'none'}"
            )
        else:
            visible_red = sorted(
                int(tid)
                for tid, *_ in dets
                if int(tid) in manager.early_from
                and fi >= manager.early_from[int(tid)]
            )
            banner = (
                f"EARLY REVIEW | frame {fi} | "
                f"early IDs={visible_red if visible_red else 'none'}"
            )

        cv2.putText(
            frame,
            banner,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            RED if visible_red else GREEN,
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()

    if reencode_h264(raw, final):
        os.remove(raw)
        out = final
    else:
        out = raw

    seconds = time.perf_counter() - t0
    print(
        f"[Video] {kind} review: {out} "
        f"({seconds:.2f}s)"
    )
    return out, seconds


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--video", required=True)
    p.add_argument("--yolo_det_weights", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--thresholds_json", required=True)

    p.add_argument("--category", default="chewinggum")
    p.add_argument("--model_size", default="S", choices=["S", "M"])
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--yolo_fp16", action="store_true")
    p.add_argument("--yolo_imgsz", type=int, default=512)
    p.add_argument("--yolo_max_det", type=int, default=32)

    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--pad_ratio", type=float, default=0.02)
    p.add_argument("--denoise", action="store_true")

    p.add_argument(
        "--crop_backend",
        choices=["opencv", "pil"],
        default="opencv",
        help="opencv=faster; pil=original preprocessing compatibility",
    )
    p.add_argument("--crop_workers", type=int, default=8)
    p.add_argument("--batch_frames", type=int, default=12)
    p.add_argument("--batch_crops", type=int, default=96)

    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--tau_track", type=float, default=0.45)
    p.add_argument("--inactive_frames", type=int, default=30)
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--clip_margin", type=int, default=5)
    p.add_argument("--min_box_frac", type=float, default=0.5)

    p.add_argument("--early_fire", action="store_true")
    p.add_argument("--early_consecutive", type=int, default=3)

    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--log_every", type=int, default=60)
    p.add_argument("--save_dir", default="./fast_v2_outputs")
    p.add_argument(
        "--no_video",
        action="store_true",
        help="Benchmark compute only; skip both review videos.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if args.batch_frames < 1 or args.batch_crops < 1:
        raise ValueError("batch sizes must be >= 1")

    if args.early_consecutive < 1:
        raise ValueError("--early_consecutive must be >= 1")

    os.makedirs(args.save_dir, exist_ok=True)

    # Fixed shapes benefit from autotuning.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    with open(args.thresholds_json, encoding="utf-8") as f:
        thresholds = json.load(f)

    print("=" * 72)
    print("FAST V2 INSPECTION")
    print("=" * 72)
    print(
        f"crop_backend={args.crop_backend} | "
        f"batch_frames={args.batch_frames} | "
        f"batch_crops={args.batch_crops} | "
        f"crop_workers={args.crop_workers}"
    )
    print(
        f"YOLO imgsz={args.yolo_imgsz} | "
        f"max_det={args.yolo_max_det} | "
        f"16-bit={args.yolo_fp16}"
    )
    print(
        f"tau_p={thresholds.get('tau_p')} | "
        f"tau_f={thresholds.get('tau_f')} | "
        f"tau_track={args.tau_track}"
    )

    # PHASE 1 ---------------------------------------------------------------
    frames, detections, width, height, src_fps, yolo_seconds = (
        run_yolo_cache(args)
    )

    # PHASE 2 ---------------------------------------------------------------
    scorer = FastScorer(args)

    # Warm-up
    dummy = np.zeros(
        (
            min(16, args.batch_crops),
            scorer.image_size,
            scorer.image_size,
            3,
        ),
        dtype=np.uint8,
    )
    _ = scorer.score_rgb_uint8(dummy)
    print(
        f"[EfficientAD] Warm-up complete "
        f"(image_size={scorer.image_size}, dtype={scorer.dtype})"
    )

    (
        manager,
        frame_meta,
        phase2_seconds,
        crop_seconds,
        score_seconds,
        state_seconds,
        n_crops,
    ) = run_efficientad_phase(
        frames,
        detections,
        width,
        height,
        scorer,
        args,
    )

    n_frames = len(frames)
    yolo_fps = n_frames / max(yolo_seconds, 1e-9)
    phase2_fps = n_frames / max(phase2_seconds, 1e-9)

    sequential_compute = yolo_seconds + phase2_seconds
    e2e_fps = n_frames / max(sequential_compute, 1e-9)

    # PHASE 3 ---------------------------------------------------------------
    early_video = None
    final_video = None
    early_render_seconds = 0.0
    final_render_seconds = 0.0

    if not args.no_video and n_frames:
        early_video, early_render_seconds = render_review(
            "early",
            frames,
            detections,
            frame_meta,
            manager,
            src_fps,
            args.save_dir,
        )
        final_video, final_render_seconds = render_review(
            "final",
            frames,
            detections,
            frame_meta,
            manager,
            src_fps,
            args.save_dir,
        )

    summary = {
        "n_frames": n_frames,
        "n_crops": n_crops,
        "yolo_seconds": yolo_seconds,
        "yolo_fps": yolo_fps,
        "phase2_seconds": phase2_seconds,
        "phase2_fps": phase2_fps,
        "crop_seconds": crop_seconds,
        "efficientad_score_seconds": score_seconds,
        "state_seconds": state_seconds,
        "sequential_compute_seconds": sequential_compute,
        "sequential_compute_fps": e2e_fps,
        "early_render_seconds": early_render_seconds,
        "final_render_seconds": final_render_seconds,
        "fired_track_ids": sorted(manager.fired),
        "early_from_frame": manager.early_from,
        "early_video": early_video,
        "final_video": final_video,
        "settings": vars(args),
        "thresholds": thresholds,
    }

    summary_path = os.path.join(args.save_dir, "summary_fast_v2.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(
        f"YOLO + ByteTrack      : {yolo_seconds:8.2f}s | "
        f"{yolo_fps:7.2f} FPS"
    )
    print(
        f"Phase 2 total         : {phase2_seconds:8.2f}s | "
        f"{phase2_fps:7.2f} FPS"
    )
    print(
        f"  crop preprocessing  : {crop_seconds:8.2f}s"
    )
    print(
        f"  EfficientAD scoring : {score_seconds:8.2f}s | "
        f"{n_crops / max(score_seconds, 1e-9):7.1f} crops/s"
    )
    print(
        f"  track bookkeeping   : {state_seconds:8.2f}s"
    )
    print(
        f"Sequential AI compute : {sequential_compute:8.2f}s | "
        f"{e2e_fps:7.2f} FPS"
    )
    print(
        f"Final anomaly IDs     : {sorted(manager.fired)}"
    )
    print(
        f"Early alert frames    : {manager.early_from}"
    )

    if early_video:
        print(f"EARLY video           : {early_video}")
    if final_video:
        print(f"FINAL video           : {final_video}")
    print(f"Summary               : {summary_path}")

    if e2e_fps >= 20:
        print("[TARGET] >= 20 FPS sequential compute: ACHIEVED")
    else:
        print(
            "[TARGET] < 20 FPS sequential compute. "
            "Use the stage timings above to choose the next optimization."
        )


if __name__ == "__main__":
    main()
