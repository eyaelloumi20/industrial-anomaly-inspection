#!/usr/bin/env python3
"""
Fast multi-background dataset builder for LowRes192.

Fast strategy:
- KEEP the existing dataset_multi unchanged.
- Create dataset_multi_bg_fast as a hard-link clone (little extra disk usage).
- Reuse the already-generated calibration_lowres192/*/scores.csv files.
- Use ONLY the user-approved NORMAL calibration tracks.
- Extract ONLY user-approved NORMAL tracks from the 4 new backgrounds.
- Use the exact ORIGINAL/current crop geometry (get_crop), NOT K78.
- Cap at 30 evenly spaced crops per normal track.
- Split the NEW normal tracks 80/20 train/test at TRACK level.
- Keep the existing validation split unchanged.

This is a FAST adaptation experiment, not the pristine final evaluation.
"""

from pathlib import Path
import csv
import os
import random
import shutil
import re
from collections import defaultdict, Counter

import cv2
from PIL import Image

ROOT = Path("/home/ahmed/two-stage-efficientad-inspection")
SRC_DATASET = ROOT / "dataset_multi"
DST_DATASET = ROOT / "dataset_multi_bg_fast"

MATCHED = ROOT / "calibration_lowres192_threshold" / "matched_tracks.csv"
GT = ROOT / "gt_reserved_newyolo" / "tracks_to_label.csv"
APPROVAL_CSV = ROOT / "normal_tracks_review_approved_all.csv"

MAX_CROPS_PER_TRACK = 30
TRAIN_RATIO = 0.80
SEED = 42
IMAGE_SIZE = 192
PAD_RATIO = 0.02
DENOISE = False

VIDEO_INFO = {
    "black": {
        "video": ROOT / "videos_background/black/WhatsApp Video 2026-09-18 at 15.15.353.mp4",
        "scores": ROOT / "calibration_lowres192/black/scores.csv",
        "tokens": ["black", "15.15.353"],
    },
    "metallic": {
        "video": ROOT / "videos_background/metallic/WhatsApp Video 2026-09-18 at 15.49.53.mp4",
        "scores": ROOT / "calibration_lowres192/metallic/scores.csv",
        "tokens": ["metallic", "15.49.53"],
    },
    "white": {
        "video": ROOT / "videos_background/white/WhatsApp Video 2026-09-18 at 15.55.11.mp4",
        "scores": ROOT / "calibration_lowres192/white/scores.csv",
        "tokens": ["white", "15.55.11"],
    },
    "wooden": {
        "video": ROOT / "videos_background/wooden/WhatsApp Video 2026-09-18 at 15.31.552.mp4",
        "scores": ROOT / "calibration_lowres192/wooden/scores.csv",
        "tokens": ["wooden", "15.31.552"],
    },
}

def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(fieldnames, candidates):
    m = {norm_name(x): x for x in fieldnames}
    for c in candidates:
        if norm_name(c) in m:
            return m[norm_name(c)]
    return None

def detect_bg_from_row(row):
    text = " ".join(str(v) for v in row.values()).lower()
    for bg, info in VIDEO_INFO.items():
        for token in info["tokens"]:
            if token.lower() in text:
                return bg
    return None

def load_normal_track_ids():
    """
    Load ONLY tracks explicitly approved by the user.

    Expected CSV columns:
      background, track_id, manual_label, APPROVE
    """
    src = APPROVAL_CSV
    if not src.exists():
        raise SystemExit(
            "ERROR: approval CSV not found:\n"
            f"  {src}\n"
            "Copy normal_tracks_review_approved_all.csv into the project root."
        )

    with src.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"ERROR: empty approval CSV: {src}")

    required = {"background", "track_id", "manual_label", "APPROVE"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(
            f"ERROR: approval CSV missing columns: {sorted(missing)}\n"
            f"Columns found: {list(rows[0].keys())}"
        )

    out = defaultdict(set)
    rejected = 0
    non_normal = 0

    for r in rows:
        approve = str(r.get("APPROVE", "")).strip().lower()
        label = str(r.get("manual_label", "")).strip().lower()
        bg = str(r.get("background", "")).strip().lower()

        if approve not in {"yes", "y", "1", "true", "approved"}:
            rejected += 1
            continue

        if label not in {"normal", "0", "good", "ok"}:
            non_normal += 1
            continue

        if bg not in VIDEO_INFO:
            continue

        try:
            tid = int(float(str(r.get("track_id", "")).strip()))
        except Exception:
            continue

        out[bg].add(tid)

    print(f"Approval source: {src}")
    total = 0
    for bg in VIDEO_INFO:
        n = len(out[bg])
        total += n
        print(f"  {bg:<9s}: {n} APPROVED NORMAL tracks")

    print(f"  TOTAL    : {total} APPROVED NORMAL tracks")
    print(f"  rejected : {rejected}")
    print(f"  nonnormal: {non_normal}")

    if total != 69:
        print(
            f"WARNING: expected 69 approved tracks from the reviewed file, found {total}."
        )

    missing_bg = [bg for bg in VIDEO_INFO if not out[bg]]
    if missing_bg:
        raise SystemExit(
            "ERROR: no approved normal tracks for: "
            + ", ".join(missing_bg)
        )

    return out

def hardlink_clone_dataset():
    if not SRC_DATASET.exists():
        raise SystemExit(f"ERROR: missing source dataset: {SRC_DATASET}")
    csv_src = SRC_DATASET / "split_csv/1cls.csv"
    if not csv_src.exists():
        raise SystemExit(f"ERROR: missing source CSV: {csv_src}")

    if DST_DATASET.exists():
        print(f"Removing previous fast dataset: {DST_DATASET}")
        shutil.rmtree(DST_DATASET)

    print(f"Hard-link cloning {SRC_DATASET.name} -> {DST_DATASET.name} ...")
    shutil.copytree(SRC_DATASET, DST_DATASET, copy_function=os.link)

    # Important: break the hard link for the CSV before replacing it.
    csv_dst = DST_DATASET / "split_csv/1cls.csv"
    original_text = csv_dst.read_text(encoding="utf-8")
    csv_dst.unlink()
    csv_dst.write_text(original_text, encoding="utf-8")
    return csv_dst

def evenly_spaced(rows, n):
    if len(rows) <= n:
        return rows
    if n <= 1:
        return [rows[len(rows)//2]]
    idxs = [round(i * (len(rows)-1) / (n-1)) for i in range(n)]
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            out.append(rows[i])
            seen.add(i)
    return out

def load_score_rows(scores_path, allowed_tracks):
    with scores_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise SystemExit(f"ERROR: {scores_path} missing columns: {sorted(missing)}")

    by_track = defaultdict(list)
    for r in rows:
        try:
            tid = int(float(r["track_id"]))
            if tid not in allowed_tracks:
                continue

            verdict = str(r.get("verdict", "")).upper()
            if verdict == "CLIPPED":
                continue

            # A scored row should have a raw score. This also excludes non-scored rows.
            if "raw_score" in r and not str(r["raw_score"]).strip():
                continue

            rr = {
                "frame": int(float(r["frame"])),
                "track_id": tid,
                "x1": float(r["x1"]),
                "y1": float(r["y1"]),
                "x2": float(r["x2"]),
                "y2": float(r["y2"]),
            }
            by_track[tid].append(rr)
        except Exception:
            continue

    for tid in by_track:
        by_track[tid].sort(key=lambda z: z["frame"])

    return by_track

def split_tracks(normal_tracks):
    rng = random.Random(SEED)
    result = {}
    for bg, tids in normal_tracks.items():
        ids = sorted(tids)
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * (1.0 - TRAIN_RATIO)))
        if len(ids) > 1:
            n_test = min(n_test, len(ids)-1)
        test = set(ids[:n_test])
        train = set(ids[n_test:])
        result[bg] = {"train": train, "test": test}
        print(f"{bg:<9s}: train tracks={len(train):2d}, test tracks={len(test):2d}")
    return result

def extract_background(bg, info, allowed_tracks, split_map, extra_root):
    from compare_pipeline_methods_batched import get_crop

    video_path = info["video"]
    scores_path = info["scores"]

    if not video_path.exists():
        raise SystemExit(f"ERROR: video missing: {video_path}")
    if not scores_path.exists():
        raise SystemExit(f"ERROR: score CSV missing: {scores_path}")

    rows_by_track = load_score_rows(scores_path, allowed_tracks)
    usable_tracks = sorted(set(rows_by_track) & set(allowed_tracks))

    if not usable_tracks:
        raise SystemExit(f"ERROR: no scored NORMAL tracks found for {bg}")

    selected = []
    for tid in usable_tracks:
        chosen = evenly_spaced(rows_by_track[tid], MAX_CROPS_PER_TRACK)
        split = "train" if tid in split_map["train"] else "test"
        for r in chosen:
            r = dict(r)
            r["split"] = split
            selected.append(r)

    by_frame = defaultdict(list)
    for r in selected:
        by_frame[r["frame"]].append(r)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video: {video_path}")

    saved = []
    frame_idx = 0
    wanted_frames = set(by_frame)
    max_frame = max(wanted_frames)

    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in wanted_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            for r in by_frame[frame_idx]:
                crop = get_crop(
                    pil,
                    (r["x1"], r["y1"], r["x2"], r["y2"]),
                    IMAGE_SIZE,
                    PAD_RATIO,
                    DENOISE,
                )

                fname = (
                    f"bgfast_{bg}_track{r['track_id']:04d}_"
                    f"frame{frame_idx:06d}.png"
                )
                out_path = extra_root / bg / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out_path)

                saved.append({
                    "split": r["split"],
                    "background": bg,
                    "track_id": r["track_id"],
                    "frame": frame_idx,
                    "path": out_path,
                })

        frame_idx += 1

    cap.release()

    print(
        f"{bg:<9s}: saved {len(saved):4d} crops "
        f"from {len(set(x['track_id'] for x in saved))} normal tracks"
    )
    return saved

def append_csv(csv_path, extracted):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        old_rows = list(reader)

    expected = ["object", "split", "label", "image", "mask"]
    if fields != expected:
        raise SystemExit(
            f"ERROR: expected VisA CSV header {expected}, got {fields}"
        )

    new_rows = []
    for x in extracted:
        rel = x["path"].relative_to(DST_DATASET)
        new_rows.append({
            "object": "chewinggum",
            "split": x["split"],
            "label": "normal",
            "image": str(rel),
            "mask": "",
        })

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=expected)
        w.writeheader()
        w.writerows(old_rows)
        w.writerows(new_rows)

    return old_rows, new_rows

def print_csv_counts(csv_path):
    counts = defaultdict(Counter)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            counts[r["split"]][r["label"]] += 1

    print("\nFINAL CSV COUNTS")
    print("-" * 64)
    for split in ["train", "test", "validation"]:
        c = counts[split]
        print(f"{split:<12s}: {sum(c.values()):5d} -> {dict(c)}")

def main():
    print("=" * 88)
    print("FAST MULTI-BACKGROUND NORMAL DATASET")
    print("=" * 88)
    print("Current/original crop preprocessing only. K78 is NOT used.\n")

    normal_tracks = load_normal_track_ids()
    track_splits = split_tracks(normal_tracks)
    csv_dst = hardlink_clone_dataset()

    extra_root = DST_DATASET / "chewinggum/Data/Images/NormalBGFast"
    all_saved = []

    for bg, info in VIDEO_INFO.items():
        all_saved.extend(
            extract_background(
                bg,
                info,
                normal_tracks[bg],
                track_splits[bg],
                extra_root,
            )
        )

    old_rows, new_rows = append_csv(csv_dst, all_saved)

    print("\n" + "=" * 88)
    print("DATASET READY")
    print("=" * 88)
    print(f"Original rows preserved : {len(old_rows)}")
    print(f"New normal crops added  : {len(new_rows)}")
    print(f"New dataset             : {DST_DATASET}")
    print(f"CSV                     : {csv_dst}")

    by_bg = Counter(x["background"] for x in all_saved)
    by_split = Counter(x["split"] for x in all_saved)
    print(f"Added by background     : {dict(by_bg)}")
    print(f"Added by split          : {dict(by_split)}")
    print_csv_counts(csv_dst)

    print("\nNEXT:")
    print("  ./train_lowres192_multibg_fast.sh")

if __name__ == "__main__":
    main()
