"""
combined_gum_inspection_pipeline_compare_all.py
==================================================
Runs ONE video through ALL FIVE checkpoint types from this project
(baseline / lowres / thin / qat / pruned) and produces a side-by-side
comparison: one annotated video per method, one CSV per method, and a
single HTML report + timing chart comparing them all.

CHANGELOG vs. the original combined_gum_inspection_pipeline_compare_all.py
------------------------------------------------------------------------------
1. Compatibility checking against calibration_meta.json is now tiered:
   - HARD fields (category, model_size, image_size, ratio, yolo detector
     weights): mismatches here mean the threshold is almost certainly
     meaningless (different model/normalization/detector). The threshold is
     refused (that method falls back to score-only mode) unless
     --force_threshold_mismatch is passed.
   - SOFT fields (conf, iou, pad_ratio, denoise, ema_alpha): mismatches shift
     scores somewhat but don't invalidate the threshold outright -- still
     just a loud warning, as before.
   - A checkpoint-directory fingerprint is also compared, to catch "this
     checkpoint was retrained/replaced since it was calibrated" even when
     the --*_dir path is unchanged.
2. The verdict now reads `recommended_basis` ('ema' or 'raw') from
   calibration_meta.json and compares against the matching score, instead of
   always assuming EMA. Previously a raw-basis threshold (produced whenever
   the calibration script's --sample_stride was >1) was silently compared
   against the EMA score, which is a systematic mismatch, not just "may run
   a bit hot/cold".
3. Added --video_dir and --image_dir support for processing multiple videos or image sequences.
4. Added vertical-clipping filter during inference: detections touching top/bottom frame edge are marked as "CLIPPED" and excluded from anomaly count.
5. Fixed --video_dir output to save each video in its own subfolder.

Fairness design
----------------
YOLO detection + ByteTrack tracking runs EXACTLY ONCE for the whole video
(Phase 1), cached per frame. Every method then scores the SAME cached
crops (Phase 2) -- so timing differences reflect ONLY the EfficientAD
forward pass + preprocessing for that method, not YOLO variance, and
accuracy-relevant score differences are being computed on identical
regions across methods. This also means YOLO's cost (often the dominant
cost in earlier full-pipeline runs) is paid once instead of five times.

Each method uses the correct loader for its checkpoint type -- see
benchmark_efficientad.py, which this script's dispatch logic mirrors:
    baseline / lowres -> utils.load_efficientad_adaptive  (GPU, any image_size)
    thin              -> utils.load_efficientad_thin       (GPU, 256px)
    qat               -> utils.load_efficientad_qat         (CPU forced, 256px)
    pruned            -> utils.load_efficientad_pruned      (GPU, 256px)

Any --*_dir that doesn't exist is skipped, same as benchmark_efficientad.py
--compare_all, so you don't need every checkpoint trained to use this.

Thresholding is optional per-method (--baseline_threshold, etc., or
--*_threshold_file pointing at a calibrate_threshold_all.py output). Without
one, that method runs in score-only mode (matches
combined_gum_inspection_pipeline.py's existing behaviour).

Foreign-object ("UNKNOWN") overlay
-----------------------------------
Same Otsu-based logic as combined_gum_inspection_pipeline.py: a global Otsu
foreground mask, minus every YOLO-known box for that frame, drawn in red
with an "UNKNOWN" label. The known boxes come from the SAME cached Phase 1
detections every method already scores against, so the overlay is IDENTICAL
across methods -- differences between methods' videos are only ever the
EfficientAD score/verdict, never the foreign-object overlay. On by default;
disable with --no_detect_unknown.

Video naming
------------
Annotated videos are named to match combined_gum_inspection_pipeline.py's
convention (same "YOLO-Detect + Track + EfficientAD (Raw Crop)" method --
this script's five checkpoints are ALL raw-crop preprocessing, just
different checkpoint types), with the checkpoint name appended so the five
outputs stay distinguishable:
    annotated_yolo-detect_track_efficientad_raw_crop_baseline_h264.mp4
    annotated_yolo-detect_track_efficientad_raw_crop_lowres_h264.mp4
    annotated_yolo-detect_track_efficientad_raw_crop_thin_h264.mp4
    annotated_yolo-detect_track_efficientad_raw_crop_qat_h264.mp4
    annotated_yolo-detect_track_efficientad_raw_crop_pruned_h264.mp4

Usage
-----
    python combined_gum_inspection_pipeline_compare_all.py \
        --video /path/to/video.mp4 \
        --yolo_det_weights /path/to/best.pt \
        --baseline_dir /home/ahmed/framework/ckpt_cropped_large \
        --lowres_dir ./ckpt_192 \
        --thin_dir ./ckpt_thin_025 \
        --qat_dir ./ckpt_qat \
        --pruned_dir ./ckpt_pruned_030 \
        --thin_threshold_file ./calibration_thin/threshold.txt \
        --pruned_threshold_file ./calibration_pruned/threshold.txt \
        --save_dir ./results_compare_all \
        --category chewinggum --model_size S

    Or with --video_dir to process all videos in a folder:
    python combined_gum_inspection_pipeline_compare_all.py \
        --video_dir /path/to/videos \
        --yolo_det_weights /path/to/best.pt \
        --baseline_dir /home/ahmed/framework/ckpt_cropped_large \
        --skip_methods lowres thin qat pruned \
        --category chewinggum --model_size S \
        --fp16 \
        --save_dir ./results
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUT_CHANNELS = 384
IMAGE_SIZE_FALLBACK = 256

METHODS = ['baseline', 'lowres', 'thin', 'qat', 'pruned']

METHOD_COLOURS = {   # BGR, for cv2 drawing
    'baseline': (200, 150, 0),
    'lowres':   (0, 200, 200),
    'thin':     (0, 200, 0),
    'qat':      (200, 0, 200),
    'pruned':   (0, 100, 255),
}

# Fields whose mismatch means the threshold is almost certainly meaningless
# (different model, different normalization, different detector).
HARD_COMPAT_FIELDS = ["category", "model_size", "ratio", "yolo_weights"]
# image_size is also HARD but can only be checked after load_bundle() runs
# (it depends on which checkpoint actually loads), so it's handled separately
# inside run_method().

# Fields whose mismatch shifts scores somewhat but doesn't invalidate the
# threshold outright.
SOFT_COMPAT_FIELDS = ["conf", "iou", "pad_ratio", "denoise", "ema_alpha"]


# ── loader dispatch (mirrors benchmark_efficientad.py) ────────────────────────

def load_bundle(method: str, ckpt_dir: str, category: str, model_size: str, device: str):
    if method in ('baseline', 'lowres'):
        from utils.load_efficientad_adaptive import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, models.get('image_size', IMAGE_SIZE_FALLBACK), device
    elif method == 'thin':
        from utils.load_efficientad_thin import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, IMAGE_SIZE_FALLBACK, device
    elif method == 'qat':
        from utils.load_efficientad_qat import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, IMAGE_SIZE_FALLBACK, 'cpu'
    elif method == 'pruned':
        from utils.load_efficientad_pruned import load_efficientad_model
        models = load_efficientad_model(ckpt_dir, category, model_size, device)
        return models, IMAGE_SIZE_FALLBACK, device
    else:
        raise ValueError(f"Unknown method '{method}'")


def fingerprint_ckpt_dir(ckpt_dir: str) -> str:
    """Same fingerprint logic as calibrate_threshold_all.py -- lets us detect
    'this checkpoint was retrained/replaced since it was calibrated'."""
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return ""
    h = hashlib.sha1()
    for root, _dirs, files in os.walk(ckpt_dir):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, ckpt_dir)
            h.update(f"{rel}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def reencode_h264(input_path: str, output_path: str) -> bool:
    """Re-encode mp4v output to H.264 so it plays back everywhere."""
    if not os.path.exists(input_path) or os.path.getsize(input_path) < 1024:
        print(f"  [WARN] reencode_h264: input missing or too small ({input_path})")
        return False
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-loglevel", "error", output_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not os.path.exists(output_path):
        print(f"  [WARN] ffmpeg re-encode failed (code {proc.returncode}): "
              f"{proc.stderr.strip()[:300]}")
        return False
    return True


# ── Otsu foreground / foreign-object ("UNKNOWN") detection ────────────────────
# Ported from combined_gum_inspection_pipeline.py. Draws whatever Otsu calls
# foreground that ISN'T inside a known YOLO box -- e.g. debris, a hand, a
# foreign object on the belt.

def otsu_mask_global(gray: np.ndarray, max_area_frac: float) -> np.ndarray:
    """Single global Otsu threshold. Flips polarity if it flags the majority of the frame as foreground."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (mask.mean() / 255.0) > max_area_frac:
        mask = cv2.bitwise_not(mask)
    return mask


def clean_mask(mask: np.ndarray, kernel_size: int = 3, close_kernel_size: int = 7) -> np.ndarray:
    """Open (remove speckle noise) then close (fill small holes / bridge gaps)."""
    open_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def interior_roi_mask(h: int, w: int,
                       margin_x_frac: float = 0.0,
                       margin_y_frac: float = 0.0) -> np.ndarray:
    """White interior rectangle, black margin band around all four sides.
    ANDing this with the unknown mask drops any foreign-object contour that
    lives in the excluded border (e.g. the conveyor's side rails)."""
    roi = np.zeros((h, w), np.uint8)
    mx = int(w * margin_x_frac)
    my = int(h * margin_y_frac)
    roi[my:h - my if my > 0 else h, mx:w - mx if mx > 0 else w] = 255
    return roi


def draw_unknown_contours(frame: np.ndarray, unknown_mask: np.ndarray, min_area: int) -> None:
    """Draws foreign-object ('UNKNOWN') contours in-place."""
    contours, _ = cv2.findContours(unknown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(frame, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def compute_unknown_mask(frame: np.ndarray,
                         dets: List[Tuple[int, int, int, int, int]],
                         otsu_max_area_frac: float) -> np.ndarray:
    """Computes the cleaned UNKNOWN foreign-object mask for one frame, using
    the raw `frame` for Otsu and `dets` (cached, shared Phase 1 boxes:
    (tid, x1, y1, x2, y2)) as the known-region mask. Identical across every
    method since both inputs come from the single shared Phase 1 pass -- this
    is why it's safe (and free performance) to compute it ONCE per frame in
    Phase 1 and reuse the cached mask for every method's video in Phase 2,
    instead of recomputing the Otsu threshold + morphology from scratch once
    per method."""
    src_h, src_w = frame.shape[:2]
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    non_bg_mask = otsu_mask_global(gray_full, otsu_max_area_frac)
    known_mask = np.zeros_like(non_bg_mask)
    for _tid, x1, y1, x2, y2 in dets:
        cv2.rectangle(known_mask, (x1, y1), (x2, y2), 255, -1)
    unknown_mask = cv2.bitwise_and(non_bg_mask, cv2.bitwise_not(known_mask))
    unknown_mask = clean_mask(unknown_mask)
    unknown_mask = cv2.bitwise_and(unknown_mask, interior_roi_mask(src_h, src_w))
    return unknown_mask


def draw_unknown_overlay(annotated: np.ndarray, frame: np.ndarray,
                         dets: List[Tuple[int, int, int, int, int]],
                         otsu_max_area_frac: float, otsu_min_unknown_area: int) -> None:
    """Computes + draws the UNKNOWN foreign-object overlay in-place on `annotated`.
    Kept for any caller that doesn't have a cached mask; the main pipeline now
    uses compute_unknown_mask() once in Phase 1 + draw_unknown_contours() per
    method instead of calling this (which would recompute the mask 5x)."""
    unknown_mask = compute_unknown_mask(frame, dets, otsu_max_area_frac)
    draw_unknown_contours(annotated, unknown_mask, otsu_min_unknown_area)


def resize_with_padding(crop: Image.Image, size: int) -> Image.Image:
    ow, oh = crop.size
    scale  = size / max(ow, oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas  = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def get_crop(pil_img: Image.Image, box: tuple, size: int,
            pad_ratio: float, denoise: bool) -> Image.Image:
    W, H = pil_img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2 = min(W, x2 + px);  y2 = min(H, y2 + py)
    crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
    crop = resize_with_padding(crop, size)
    if denoise:
        arr = np.array(crop, dtype=np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.8, sigmaY=0.8)
        arr = cv2.bilateralFilter(arr, d=5, sigmaColor=15, sigmaSpace=15)
        crop = Image.fromarray(arr)
    return crop


@torch.no_grad()
def score_crop(models: dict, crop_pil: Image.Image, device: str,
               image_size: int, ratio: float = 0.1, use_amp: bool = False) -> float:
    t = transforms.ToTensor()(crop_pil).unsqueeze(0).to(device)

    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
              if use_amp and device.startswith("cuda") else nullcontext())
    with amp_ctx:
        t_out = models['teacher'](t)
        s_out = models['student'](t)
        a_out = models['ae'](t)

        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]
        norm_t = (t_out - models['mean']) / (models['std'] + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)

        fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                                size=(image_size, image_size), mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                                size=(image_size, image_size), mode='bilinear', align_corners=False)

    # Normalization + max score always in fp32 -- this is a threshold-facing
    # scalar, so keep it numerically stable regardless of use_amp.
    fm_st, fm_stae = fm_st.float(), fm_stae.float()
    nm_st = (ratio * (fm_st   - models['qa_st'])) / (models['qb_st'] - models['qa_st'] + 1e-8)
    nm_ae = (ratio * (fm_stae - models['qa_ae'])) / (models['qb_ae'] - models['qa_ae'] + 1e-8)
    amap  = (0.5 * nm_st + 0.5 * nm_ae)[0, 0].cpu().numpy()
    return float(np.max(amap))


@torch.no_grad()
def score_crop_batch(models: dict, crops_pil: List[Image.Image], device: str,
                     image_size: int, ratio: float = 0.1, use_amp: bool = False) -> List[float]:
    """Batched version of score_crop: stacks ALL crops for a frame into ONE
    [N, 3, H, W] tensor and runs teacher/student/ae ONCE for the whole frame,
    instead of once per crop with N separate [1, 3, H, W] forward passes.
    Returns one max-score float per crop, in the same order as crops_pil.
    Empty input returns [] without touching the GPU."""
    if not crops_pil:
        return []

    tensors = [transforms.ToTensor()(c) for c in crops_pil]
    t = torch.stack(tensors, dim=0).to(device)   # [N, 3, H, W] -- ONE transfer, not N

    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
              if use_amp and device.startswith("cuda") else nullcontext())
    with amp_ctx:
        t_out = models['teacher'](t)
        s_out = models['student'](t)
        a_out = models['ae'](t)

        y_st   = s_out[:, :OUT_CHANNELS,  :, :]
        y_stae = s_out[:, -OUT_CHANNELS:, :, :]
        norm_t = (t_out - models['mean']) / (models['std'] + 1e-8)
        d_st   = torch.pow(norm_t - y_st,   2)
        d_stae = torch.pow(a_out  - y_stae, 2)

        fm_st   = F.interpolate(torch.mean(d_st,   dim=1, keepdim=True),
                                size=(image_size, image_size), mode='bilinear', align_corners=False)
        fm_stae = F.interpolate(torch.mean(d_stae, dim=1, keepdim=True),
                                size=(image_size, image_size), mode='bilinear', align_corners=False)

    fm_st, fm_stae = fm_st.float(), fm_stae.float()
    nm_st = (ratio * (fm_st   - models['qa_st'])) / (models['qb_st'] - models['qa_st'] + 1e-8)
    nm_ae = (ratio * (fm_stae - models['qa_ae'])) / (models['qb_ae'] - models['qa_ae'] + 1e-8)
    amap  = (0.5 * nm_st + 0.5 * nm_ae)[:, 0].cpu().numpy()   # [N, H, W] -- ONE sync, not N
    return [float(np.max(amap[i])) for i in range(amap.shape[0])]


# ── Frame source abstraction: video, image_dir, or video_dir ─────────────────

def iter_video_frames(video_path: str, max_frames: Optional[int]) -> Iterator[Tuple[np.ndarray, Image.Image, int]]:
    """Yields (bgr_frame, pil_frame, frame_idx) from a single video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        yield frame, pil_frame, frame_idx
        frame_idx += 1
    cap.release()


def iter_image_dir(image_dir: str, max_frames: Optional[int], fps: float) -> Iterator[Tuple[np.ndarray, Image.Image, int]]:
    """Yields (bgr_frame, pil_frame, frame_idx) from a folder of images (sorted by filename)."""
    paths = sorted(Path(image_dir).glob('*'))
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    paths = [p for p in paths if p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"No image files found in {image_dir}")
    for idx, p in enumerate(paths[:max_frames]):
        img = cv2.imread(str(p))
        if img is None:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        yield img, pil_img, idx


def frame_source(args, video_path: str) -> Iterator[Tuple[np.ndarray, Image.Image, int]]:
    """Choose the appropriate iterator based on args."""
    if args.video_dir:
        return iter_video_frames(video_path, args.max_frames)
    elif args.video:
        return iter_video_frames(video_path, args.max_frames)
    elif args.image_dir:
        return iter_image_dir(args.image_dir, args.max_frames, args.image_seq_fps)
    else:
        raise ValueError("No video or image source specified")


# ── Phase 1: shared YOLO detection + tracking, run once ───────────────────────

@dataclass
class FrameCache:
    """Everything Phase 2 needs per frame, decoded/computed exactly ONCE in
    Phase 1 and reused by every method -- instead of each method reopening
    the video and redoing the BGR->PIL conversion + UNKNOWN-mask computation
    from scratch. For a 214-frame 1080x1920 clip this runs ~1.2-3GB in RAM
    depending on whether unknown masks are cached too; for much longer videos
    you'd want to stream instead of fully caching."""
    bgr_frames:    List[np.ndarray]
    pil_frames:    List[Image.Image]
    unknown_masks: List[Optional[np.ndarray]]  # None per-frame if detect_unknown is off


def run_yolo_once(video_path: str, yolo_weights: str, conf: float, iou: float,
                  device: str, max_frames: Optional[int],
                  detect_unknown: bool = False, otsu_max_area_frac: float = 0.25,
                  cache_frames: bool = True):
    """Detect+track every frame ONCE, cache boxes per frame. If cache_frames,
    ALSO decodes each frame exactly once here (rather than once per method in
    Phase 2) and, if detect_unknown, computes the UNKNOWN foreign-object mask
    here too (identical across every method, so recomputing it per method is
    pure waste). Returns
    (detections: {frame_idx: [(track_id, x1,y1,x2,y2), ...]},
     src_w, src_h, src_fps, n_frames, frame_cache_or_None)."""
    from ultralytics import YOLO
    print(f"[Phase 1] Loading YOLO: {yolo_weights}")
    yolo = YOLO(yolo_weights)
    yolo.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Phase 1] Source: {src_w}x{src_h} @ {src_fps:.1f}fps  (~{total} frames)")
    if cache_frames:
        est_mb = (src_w * src_h * 3 * 2 + (src_w * src_h if detect_unknown else 0)) \
                 * (max_frames or total) / 1e6
        print(f"[Phase 1] Caching decoded frames in RAM for reuse across all methods "
              f"(~{est_mb:.0f}MB estimated)")

    detections: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    bgr_frames: List[np.ndarray] = []
    pil_frames: List[Image.Image] = []
    unknown_masks: List[Optional[np.ndarray]] = []
    frame_idx = 0
    n_tracker_resets = 0
    force_tracker_reset = False
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        used_fallback = False
        try:
            results = yolo.track(source=frame, conf=conf, iou=iou, classes=[0],
                                 persist=not force_tracker_reset,
                                 tracker="bytetrack.yaml", verbose=False)
            force_tracker_reset = False
        except np.linalg.LinAlgError:
            n_tracker_resets += 1
            used_fallback = True
            force_tracker_reset = True
            print(f"  [Phase 1] WARNING: degenerate detection box broke the tracker "
                  f"at frame {frame_idx} -- using untracked detections for this "
                  f"frame only, then reinitializing tracker "
                  f"[reset #{n_tracker_resets}]")
            results = yolo.predict(source=frame, conf=conf, iou=iou, classes=[0],
                                   verbose=False)

        boxes_this_frame = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            if used_fallback or r.boxes.id is None:
                ids = [100_000 + frame_idx * 1000 + i for i in range(len(xyxy))]
            else:
                ids = r.boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(xyxy, ids):
                x1 = max(0, box[0]); y1 = max(0, box[1])
                x2 = min(src_w, box[2]); y2 = min(src_h, box[3])
                if x2 > x1 and y2 > y1:
                    boxes_this_frame.append((int(tid), x1, y1, x2, y2))
        detections[frame_idx] = boxes_this_frame

        if cache_frames:
            bgr_frames.append(frame)
            pil_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            unknown_masks.append(
                compute_unknown_mask(frame, boxes_this_frame, otsu_max_area_frac)
                if detect_unknown else None)

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"  [Phase 1] frame {frame_idx}/{total}  "
                  f"({frame_idx / (time.perf_counter() - t0):.1f} fps)")

    cap.release()
    n_frames = frame_idx
    print(f"[Phase 1] Done: {n_frames} frames, "
          f"{sum(len(v) for v in detections.values())} total detections, "
          f"{time.perf_counter() - t0:.1f}s"
          + (f", {n_tracker_resets} tracker reset(s)" if n_tracker_resets else ""))
    frame_cache = FrameCache(bgr_frames, pil_frames, unknown_masks) if cache_frames else None
    return detections, src_w, src_h, src_fps, n_frames, frame_cache


# ── Phase 2: score cached crops with one method's EfficientAD ────────────────

@dataclass
class MethodResult:
    method:        str
    n_frames:      int      = 0
    n_detections:  int      = 0
    mean_ms:       float    = 0.0      # ms/crop (batch time spread evenly across that frame's crops -- informational)
    fps:           float    = 0.0      # crop-equivalent scores/sec, derived from mean_ms
    mean_ms_frame: float    = 0.0      # ms per FRAME's batched forward pass (the real cost unit now)
    fps_frame:     float    = 0.0      # frames/sec the EfficientAD scoring stage can sustain
    score_mean:    float    = 0.0
    score_max:     float    = 0.0
    n_anomalous:   int      = 0
    n_clipped:     int      = 0
    threshold:     Optional[float] = None
    threshold_basis: str    = "ema"
    video_path:    Optional[str]   = None
    csv_path:      Optional[str]   = None
    error:         Optional[str]   = None


def run_method(method: str, ckpt_dir: str, video_path: str,
               detections: Dict[int, List[Tuple[int, int, int, int, int]]],
               src_w: int, src_h: int, src_fps: float, n_frames: int,
               args, threshold: Optional[float], basis: str,
               cal_meta: Optional[dict],
               frame_cache: Optional[FrameCache] = None,
               base_save_dir: Optional[str] = None) -> MethodResult:

    print(f"\n{'='*70}\n  Method: {method}  (ckpt={ckpt_dir})\n{'='*70}")
    result = MethodResult(method=method, threshold=threshold, threshold_basis=basis)

    try:
        models, image_size, device = load_bundle(
            method, ckpt_dir, args.category, args.model_size, args.device)
    except Exception as e:
        print(f"  [{method}] FAILED to load: {e}")
        result.error = str(e)
        return result

    if threshold is not None and cal_meta is not None:
        meta_image_size = cal_meta.get("image_size")
        if meta_image_size is not None and meta_image_size != image_size:
            print(f"  [{method}]   [ERROR] HARD compatibility mismatch: calibrated at "
                  f"image_size={meta_image_size}, this run's checkpoint loads at "
                  f"image_size={image_size} -- crops are resized differently, so the "
                  f"threshold does not apply.")
            if not args.force_threshold_mismatch:
                print(f"  [{method}]   -> refusing threshold (pass --force_threshold_mismatch "
                      f"to override). Falling back to SCORE-ONLY mode.")
                threshold = None
                result.threshold = None
            else:
                print(f"  [{method}]   -> --force_threshold_mismatch set: using threshold anyway.")

    ema_hist: Dict[int, float] = {}
    alpha = args.ema_alpha

    save_root = base_save_dir if base_save_dir is not None else args.save_dir
    save_dir = os.path.join(save_root, method)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"scores_{method}.csv")
    video_out_raw = os.path.join(
        save_dir, f"annotated_yolo-detect_track_efficientad_raw_crop_{method}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_raw, fourcc, src_fps, (src_w, src_h)) if not args.no_video else None

    cap = None if frame_cache is not None else cv2.VideoCapture(video_path)
    csv_file = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(["frame", "track_id", "raw_score", "ema_score", "verdict",
                    "x1", "y1", "x2", "y2", "score_ms"])

    all_scores = []
    all_ms = []
    all_frame_ms = []
    n_detections = 0
    n_anomalous = 0
    n_clipped = 0
    colour = METHOD_COLOURS.get(method, (255, 255, 255))
    use_amp = args.fp16 and device.startswith("cuda")

    for frame_idx in range(n_frames):
        if frame_cache is not None:
            frame = frame_cache.bgr_frames[frame_idx]
            pil_frame = frame_cache.pil_frames[frame_idx]
        else:
            ret, frame = cap.read()
            if not ret:
                break
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated = frame.copy()
        frame_dets = detections.get(frame_idx, [])

        if writer is not None and args.detect_unknown:
            if frame_cache is not None and frame_cache.unknown_masks[frame_idx] is not None:
                draw_unknown_contours(annotated, frame_cache.unknown_masks[frame_idx],
                                      args.otsu_min_unknown_area)
            else:
                draw_unknown_overlay(annotated, frame, frame_dets,
                                     args.otsu_max_area_frac, args.otsu_min_unknown_area)

        crops = [get_crop(pil_frame, (x1, y1, x2, y2), image_size,
                          args.pad_ratio, args.denoise)
                 for _tid, x1, y1, x2, y2 in frame_dets]

        t0 = time.perf_counter()
        scores = score_crop_batch(models, crops, device, image_size,
                                  ratio=args.ratio, use_amp=use_amp)
        batch_ms = (time.perf_counter() - t0) * 1000.0
        all_frame_ms.append(batch_ms)
        per_crop_ms = (batch_ms / len(crops)) if crops else 0.0

        for (tid, x1, y1, x2, y2), score in zip(frame_dets, scores):
            prev = ema_hist.get(tid, score)
            ema = alpha * score + (1 - alpha) * prev
            ema_hist[tid] = ema

            all_scores.append(score)
            all_ms.append(per_crop_ms)
            n_detections += 1

            touches_top    = (y1 <= 0)
            touches_bottom = (y2 >= src_h)
            vertical_clipped = touches_top or touches_bottom

            verdict = ""
            if threshold is not None:
                if vertical_clipped:
                    verdict = "CLIPPED"
                    n_clipped += 1
                else:
                    verdict_score = score if basis == "raw" else ema
                    verdict = "ANOMALY" if verdict_score >= threshold else "NORMAL"
                    if verdict == "ANOMALY":
                        n_anomalous += 1

            csv_w.writerow([frame_idx, tid, f"{score:.6f}", f"{ema:.6f}", verdict,
                            x1, y1, x2, y2, f"{per_crop_ms:.2f}"])

            if writer is not None:
                box_colour = colour
                if threshold is not None:
                    if vertical_clipped:
                        box_colour = (100, 100, 100)
                    else:
                        box_colour = (0, 0, 255) if verdict == "ANOMALY" else (0, 200, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_colour, 2)
                label = f"ID{tid} {verdict + ' ' if verdict else ''}{ema:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                tx, ty = max(0, x1 + 3), max(th + 6, y1 + 3)
                cv2.rectangle(annotated, (tx-2, ty-th-4), (tx+tw+4, ty+2), box_colour, -1)
                cv2.putText(annotated, label, (tx, ty),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is not None:
            tag = f"{method}  |  frame {frame_idx}  |  {len(frame_dets)} dets"
            cv2.putText(annotated, tag, (8, 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
            writer.write(annotated)

    if cap is not None:
        cap.release()
    csv_file.close()
    if writer is not None:
        writer.release()
        h264_path = video_out_raw.replace(".mp4", "_h264.mp4")
        if reencode_h264(video_out_raw, h264_path):
            os.remove(video_out_raw)
            result.video_path = h264_path
        else:
            result.video_path = video_out_raw

    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result.n_frames        = n_frames
    result.n_detections    = n_detections
    result.mean_ms          = float(np.mean(all_ms)) if all_ms else 0.0
    result.fps               = float(1000.0 / result.mean_ms) if result.mean_ms > 0 else 0.0
    result.mean_ms_frame    = float(np.mean(all_frame_ms)) if all_frame_ms else 0.0
    result.fps_frame          = float(1000.0 / result.mean_ms_frame) if result.mean_ms_frame > 0 else 0.0
    result.score_mean        = float(np.mean(all_scores)) if all_scores else 0.0
    result.score_max          = float(np.max(all_scores)) if all_scores else 0.0
    result.n_anomalous       = n_anomalous
    result.n_clipped         = n_clipped
    result.csv_path          = csv_path
    result.threshold         = threshold

    print(f"  [{method}] {n_detections} detections over {n_frames} frames  "
          f"batch(frame) mean={result.mean_ms_frame:.2f}ms -> {result.fps_frame:.1f} FPS  "
          f"(crop-equivalent: {result.mean_ms:.2f}ms/crop, {result.fps:.1f} scores/sec)  "
          f"score_mean={result.score_mean:.4f}"
          + (f"  anomalous={n_anomalous}, clipped={n_clipped}  (basis={basis})" if threshold is not None else ""))
    return result


# ── HTML report + timing chart ─────────────────────────────────────────────────

def save_timing_chart(results: List[MethodResult], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in results if r.error is None]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    names = [r.method for r in ok]
    fps   = [r.fps_frame for r in ok]
    bars = ax.bar(names, fps, color="#e94560")
    ax.set_ylabel("Frames / sec (batched EfficientAD forward pass)", color="#eaeaea")
    ax.set_title("Method comparison — batched scoring throughput", color="#eaeaea")
    ax.tick_params(colors="#eaeaea")
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    for bar, v in zip(bars, fps):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.1f}",
               ha='center', va='bottom', color="#eaeaea")
    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#1a1a2e")
    plt.close()


def save_html_report(results: List[MethodResult], save_dir: str, video_path: str,
                     chart_path: str):
    rows = ""
    for r in results:
        if r.error:
            rows += (f"<tr><td>{r.method}</td><td colspan='9' style='color:#f87171'>"
                     f"FAILED: {r.error}</td></tr>")
            continue
        thr = f"{r.threshold:.4f} ({r.threshold_basis})" if r.threshold is not None else "n/a (score-only)"
        anomalous = f"{r.n_anomalous}" if r.threshold is not None else "n/a"
        clipped = f"{r.n_clipped}" if r.threshold is not None else "n/a"
        video_link = (f"<a href='{os.path.relpath(r.video_path, save_dir)}' "
                      f"style='color:#60a5fa'>video</a>") if r.video_path else "n/a"
        csv_link = (f"<a href='{os.path.relpath(r.csv_path, save_dir)}' "
                    f"style='color:#60a5fa'>csv</a>") if r.csv_path else "n/a"
        rows += (f"<tr><td><b>{r.method}</b></td><td>{r.n_detections}</td>"
                 f"<td>{r.mean_ms_frame:.2f} ms</td><td>{r.fps_frame:.1f}</td>"
                 f"<td>{r.mean_ms:.2f} ms</td><td>{r.fps:.1f}</td>"
                 f"<td>{r.score_mean:.4f}</td><td>{thr}</td><td>{anomalous}</td><td>{clipped}</td>"
                 f"<td>{video_link}</td><td>{csv_link}</td></tr>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Method Comparison Report</title>
<style>
body {{ background:#1a1a2e; color:#eaeaea; font-family: -apple-system, sans-serif; padding:24px; }}
table {{ border-collapse: collapse; width:100%; margin-top:16px; }}
th, td {{ padding:8px 12px; border-bottom:1px solid #333; text-align:left; }}
th {{ color:#9ca3af; font-weight:600; }}
img {{ max-width:100%; border-radius:8px; margin-top:16px; }}
.muted {{ color:#9ca3af; }}
</style></head>
<body>
<h1>Method Comparison — {os.path.basename(video_path)}</h1>
<p class="muted">YOLO detection + tracking ran once and was shared across every method below,
so timing and score differences reflect only the EfficientAD forward pass + preprocessing
for each checkpoint type — not YOLO variance. Crops are batched: all detections in a frame
are scored in ONE forward pass, so "FPS" (frames/sec) is the real throughput number; "scores/sec"
is the crop-equivalent rate, kept for comparison against the old crop-by-crop numbers.</p>
<p class="muted"><b>Note:</b> Detections clipped at the top or bottom frame edge are marked as "CLIPPED"
and excluded from the anomaly count. This prevents artificially inflated false positives from
partial crops near the frame boundary.</p>
<table>
<tr><th>Method</th><th>Detections</th><th>Mean ms/frame</th><th>FPS</th>
<th>Mean ms/crop</th><th>Scores/sec</th>
<th>Mean score</th><th>Threshold (basis)</th><th># Anomalous</th><th># Clipped</th><th>Video</th><th>CSV</th></tr>
{rows}
</table>
<img src="{os.path.basename(chart_path)}" alt="timing chart">
</body></html>"""

    path = os.path.join(save_dir, "report.html")
    with open(path, "w") as f:
        f.write(html)
    return path


# ── args + main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare all optimization methods (baseline/lowres/thin/qat/pruned) on one video or a folder of videos.")
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Single video file")
    input_group.add_argument("--video_dir", help="Folder of video files")
    input_group.add_argument("--image_dir", help="Folder of image files (treated as a sequence)")
    p.add_argument("--image_seq_fps", type=float, default=25.0,
                   help="Synthetic FPS for annotated video output when using --image_dir (default 25)")
    p.add_argument("--yolo_det_weights",  required=True)
    p.add_argument("--baseline_dir", default="/home/ahmed/framework/ckpt_cropped_large")
    p.add_argument("--lowres_dir",   default="./ckpt_192")
    p.add_argument("--thin_dir",     default="./ckpt_thin_025")
    p.add_argument("--qat_dir",      default="./ckpt_qat")
    p.add_argument("--pruned_dir",   default="./ckpt_pruned_030")
    p.add_argument("--skip_methods", nargs="*", default=[],
                   help="Method names to skip even if their --*_dir exists, e.g. --skip_methods qat")
    p.add_argument("--baseline_threshold", type=float, default=None)
    p.add_argument("--lowres_threshold",   type=float, default=None)
    p.add_argument("--thin_threshold",     type=float, default=None)
    p.add_argument("--qat_threshold",      type=float, default=None)
    p.add_argument("--pruned_threshold",   type=float, default=None)
    p.add_argument("--baseline_threshold_file", default=None)
    p.add_argument("--lowres_threshold_file",   default=None)
    p.add_argument("--thin_threshold_file",     default=None)
    p.add_argument("--qat_threshold_file",      default=None)
    p.add_argument("--pruned_threshold_file",   default=None)
    p.add_argument("--force_threshold_mismatch", action="store_true",
                   help="Use a calibrated threshold even when a HARD compatibility mismatch is detected.")
    p.add_argument("--category",     default="chewinggum")
    p.add_argument("--model_size",   default="S", choices=["S", "M"])
    p.add_argument("--device",       default="cuda")
    p.add_argument("--ratio",        type=float, default=0.1)
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--pad_ratio",    type=float, default=0.02)
    p.add_argument("--denoise",      action="store_true")
    p.add_argument("--ema_alpha",    type=float, default=0.3)
    p.add_argument("--max_frames",   type=int, default=0)
    p.add_argument("--no_video",     action="store_true", help="Skip saving annotated videos (faster)")
    p.add_argument("--otsu_max_area_frac",    type=float, default=0.25)
    p.add_argument("--otsu_min_unknown_area", type=int, default=20)
    p.add_argument("--no_detect_unknown",     action="store_true")
    p.add_argument("--save_dir",     default="./results_compare_all")
    p.add_argument("--fp16",         action="store_true")
    p.add_argument("--no_frame_cache", action="store_true")
    return p.parse_args()


def check_calibration_compatibility(args, method: str, file_path: str) -> Tuple[bool, str]:
    meta_path = os.path.join(os.path.dirname(file_path), "calibration_meta.json")
    if not os.path.isfile(meta_path):
        print(f"  [{method:<10s}]   [WARN] no calibration_meta.json found next to threshold "
              f"-- cannot verify compatibility. Assuming basis='ema'.")
        return True, "ema"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        print(f"  [{method:<10s}]   [WARN] calibration_meta.json found but unreadable "
              f"({type(e).__name__}: {e}) -- skipping compatibility check")
        return True, "ema"

    run_values = dict(
        category=args.category, model_size=args.model_size, ratio=args.ratio,
        yolo_weights=os.path.basename(args.yolo_det_weights),
        conf=args.conf, iou=args.iou, pad_ratio=args.pad_ratio,
        denoise=args.denoise, ema_alpha=args.ema_alpha,
    )
    meta_yolo = meta.get("yolo_weights")
    meta_values = dict(meta)
    meta_values["yolo_weights"] = os.path.basename(meta_yolo) if meta_yolo else None

    hard_mismatches = [(f, meta_values.get(f), run_values.get(f))
                       for f in HARD_COMPAT_FIELDS
                       if meta_values.get(f) is not None and meta_values.get(f) != run_values.get(f)]
    soft_mismatches = [(f, meta_values.get(f), run_values.get(f))
                       for f in SOFT_COMPAT_FIELDS
                       if meta_values.get(f) is not None and meta_values.get(f) != run_values.get(f)]

    if soft_mismatches:
        print(f"  [{method:<10s}]   [WARN] parameter drift (threshold still usable, but scores will shift):")
        for name, cal_v, run_v in soft_mismatches:
            print(f"  [{method:<10s}]     --{name}: calibrated with {cal_v!r}, running with {run_v!r}")

    ok = True
    if hard_mismatches:
        print(f"  [{method:<10s}]   [ERROR] HARD compatibility mismatch -- this threshold is "
              f"almost certainly invalid for this run:")
        for name, cal_v, run_v in hard_mismatches:
            print(f"  [{method:<10s}]     --{name}: calibrated with {cal_v!r}, running with {run_v!r}")
        if not args.force_threshold_mismatch:
            print(f"  [{method:<10s}]   -> refusing to use this threshold (pass "
                  f"--force_threshold_mismatch to override). Falling back to SCORE-ONLY mode.")
            ok = False
        else:
            print(f"  [{method:<10s}]   -> --force_threshold_mismatch set: using threshold anyway.")

    ckpt_dir = getattr(args, f"{method}_dir", None)
    fp_now = fingerprint_ckpt_dir(ckpt_dir)
    fp_then = meta.get("checkpoint_fingerprint")
    if fp_then and fp_now and fp_then != fp_now:
        print(f"  [{method:<10s}]   [WARN] checkpoint directory contents changed since "
              f"calibration (fingerprint {fp_then} -> {fp_now}).")

    basis = meta.get("recommended_basis") or ("ema" if meta.get("ema_reliable", True) else "raw")
    if basis == "raw":
        print(f"  [{method:<10s}]   [NOTE] calibrated on raw scores (ema_reliable=False) -- "
              f"this run will verdict {method} against the RAW per-crop score, not the "
              f"EMA-smoothed one.")
    return ok, basis


def resolve_threshold(args, method: str) -> Tuple[Optional[float], str, Optional[dict]]:
    direct = getattr(args, f"{method}_threshold")
    if direct is not None:
        print(f"  [{method:<10s}] threshold = {direct:.6f}   "
              f"(source: --{method}_threshold, basis=ema)")
        return direct, "ema", None

    file_path = getattr(args, f"{method}_threshold_file")
    if not file_path:
        print(f"  [{method:<10s}] threshold = NONE          "
              f"(no --{method}_threshold / --{method}_threshold_file given "
              f"-> SCORE-ONLY MODE)")
        return None, "ema", None

    if not os.path.isfile(file_path):
        print(f"  [{method:<10s}] threshold = NONE          "
              f"(--{method}_threshold_file='{file_path}' NOT FOUND on disk "
              f"-> SCORE-ONLY MODE)")
        return None, "ema", None

    try:
        with open(file_path) as f:
            raw = f.read().strip()
        value = float(raw)
    except Exception as e:
        print(f"  [{method:<10s}] threshold = NONE          "
              f"(--{method}_threshold_file='{file_path}' failed to parse "
              f"({type(e).__name__}: {e}) -> SCORE-ONLY MODE)")
        return None, "ema", None

    ok, basis = check_calibration_compatibility(args, method, file_path)
    meta_path = os.path.join(os.path.dirname(file_path), "calibration_meta.json")
    meta = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = None

    if not ok:
        return None, basis, meta

    print(f"  [{method:<10s}] threshold = {value:.6f}   (source: {file_path}, basis={basis})")
    return value, basis, meta


def run_single_source(source_path: str, args, base_save_dir: str, source_type: str, fps_override: Optional[float] = None):
    """Process a single video or image sequence."""
    detections, src_w, src_h, src_fps, n_frames, frame_cache = run_yolo_once(
        source_path, args.yolo_det_weights, args.conf, args.iou,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_frames=(args.max_frames or None),
        detect_unknown=args.detect_unknown, otsu_max_area_frac=args.otsu_max_area_frac,
        cache_frames=not args.no_frame_cache)

    results = []
    for method in active_methods:
        r = run_method(method, ckpt_dirs[method], source_path, detections,
                       src_w, src_h, src_fps, n_frames, args,
                       thresholds[method], bases[method], metas[method],
                       frame_cache=frame_cache, base_save_dir=base_save_dir)
        results.append(r)

    chart_path = os.path.join(base_save_dir, "timing_comparison.png")
    save_timing_chart(results, chart_path)
    report_path = save_html_report(results, base_save_dir, source_path, chart_path)

    print(f"\n{'='*70}\n  Summary  (batched: one forward pass per frame, not per crop)\n{'='*70}")
    print(f"  {'method':<10s} {'dets':>6s} {'ms/frame':>9s} {'FPS':>7s} {'ms/crop':>9s} "
          f"{'scores/s':>9s} {'mean_score':>11s} {'anomalous':>10s} {'clipped':>10s}")
    for r in results:
        if r.error:
            print(f"  {r.method:<10s}  FAILED: {r.error}")
            continue
        anomalous = str(r.n_anomalous) if r.threshold is not None else "n/a"
        clipped = str(r.n_clipped) if r.threshold is not None else "n/a"
        print(f"  {r.method:<10s} {r.n_detections:>6d} {r.mean_ms_frame:>9.2f} "
              f"{r.fps_frame:>7.1f} {r.mean_ms:>9.2f} {r.fps:>9.1f} "
              f"{r.score_mean:>11.4f} {anomalous:>10s} {clipped:>10s}")

    print(f"\nReport      → {report_path}")
    print(f"Timing chart → {chart_path}")
    for r in results:
        if r.video_path:
            print(f"{r.method:<10s} video → {r.video_path}")
    return results


def main():
    global active_methods, ckpt_dirs, thresholds, bases, metas
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    args.detect_unknown = not args.no_detect_unknown
    if args.detect_unknown:
        print("UNKNOWN foreign-object overlay : ON  "
              f"(otsu_max_area_frac={args.otsu_max_area_frac}, "
              f"min_area={args.otsu_min_unknown_area}px)")
    else:
        print("UNKNOWN foreign-object overlay : OFF")

    ckpt_dirs = {
        'baseline': args.baseline_dir, 'lowres': args.lowres_dir,
        'thin':     args.thin_dir,     'qat':    args.qat_dir,
        'pruned':   args.pruned_dir,
    }

    active_methods = [m for m in METHODS
                      if m not in args.skip_methods and os.path.isdir(ckpt_dirs[m])]
    skipped = [m for m in METHODS if m not in active_methods]
    print(f"Active methods : {active_methods}")
    if skipped:
        print(f"Skipped        : {skipped}  (missing --*_dir or in --skip_methods)")
    if not active_methods:
        raise SystemExit("No checkpoint directories found. Nothing to compare.")

    print(f"\n{'='*70}\n  Threshold check\n{'='*70}")
    resolved = {m: resolve_threshold(args, m) for m in active_methods}
    thresholds = {m: v[0] for m, v in resolved.items()}
    bases      = {m: v[1] for m, v in resolved.items()}
    metas      = {m: v[2] for m, v in resolved.items()}
    n_missing = sum(1 for t in thresholds.values() if t is None)
    if n_missing:
        missing_methods = [m for m, t in thresholds.items() if t is None]
        print(f"\n  WARNING: {n_missing} method(s) have NO threshold -> "
              f"{missing_methods} will run in score-only mode.")
    print()

    if args.video:
        run_single_source(args.video, args, args.save_dir, 'video')
    elif args.video_dir:
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv'}
        video_files = [p for p in Path(args.video_dir).iterdir() if p.suffix.lower() in video_exts]
        if not video_files:
            raise SystemExit(f"No video files found in {args.video_dir}")
        for video_path in sorted(video_files):
            video_name = video_path.stem
            base_save_dir = os.path.join(args.save_dir, video_name)
            os.makedirs(base_save_dir, exist_ok=True)
            print(f"\n{'='*70}\n  Processing video: {video_path.name}\n{'='*70}")
            run_single_source(str(video_path), args, base_save_dir, 'video')
    elif args.image_dir:
        run_single_source(args.image_dir, args, args.save_dir, 'image', fps_override=args.image_seq_fps)
    else:
        raise SystemExit("No input source specified. Use --video, --video_dir, or --image_dir.")


if __name__ == '__main__':
    main()