#!/usr/bin/env python3
"""
Final held-out evaluation.

Frozen configuration:
- improved multi-background YOLO
- ByteTrack
- LowRes192 EfficientAD
- EMA alpha = 0.3
- threshold = 0.8333 by default

No threshold search is performed.

Track prediction:
ANOMALY if any valid EMA sample for that physical track >= threshold.
NORMAL otherwise.

Only manually labeled valid TEST tracks are evaluated.
'not_a_piece' and 'unsure' are excluded.
"""

from pathlib import Path
import argparse, csv, json
from collections import defaultdict
import numpy as np

TEST_MAP = {
    ("black", "WhatsApp Video 2026-09-18 at 15.14.22.mp4"): "black_01",
    ("black", "WhatsApp Video 2026-09-18 at 15.15.35.mp4"): "black_02",
    ("black", "WhatsApp Video 2026-09-18 at 15.15.352.mp4"): "black_03",

    ("metallic", "WhatsApp Video 2026-09-18 at 15.47.23.mp4"): "metallic_01",
    ("metallic", "WhatsApp Video 2026-09-18 at 15.49.36.mp4"): "metallic_02",

    ("white", "WhatsApp Video 2026-09-18 at 15.55.45.mp4"): "white_01",
    ("white", "WhatsApp Video 2026-09-18 at 15.55.46.mp4"): "white_02",
    ("white", "WhatsApp Video 2026-09-18 at 15.55.461.mp4"): "white_03",

    ("wooden", "WhatsApp Video 2026-09-18 at 15.31.55.mp4"): "wooden_01",
    ("wooden", "WhatsApp Video 2026-09-18 at 15.31.551.mp4"): "wooden_02",
    ("wooden", "WhatsApp Video 2026-09-18 at 15.31.56.mp4"): "wooden_03",
    ("wooden", "WhatsApp Video 2026-09-18 at 15.31.561.mp4"): "wooden_04",
}

def norm_label(s):
    s = (s or "").strip().lower()
    if s in {"normal", "n", "0"}:
        return 0
    if s in {"anomaly", "anomalous", "a", "1"}:
        return 1
    return None

def metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "specificity": specificity,
        "accuracy": acc,
    }

def auc_roc(y, scores):
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--threshold", type=float, default=0.8333)
    ap.add_argument("--out", default="./final_unseen_lowres192_evaluation")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Ground truth TEST tracks only.
    gt = {}
    excluded_unsure = 0
    excluded_not_piece = 0

    with open(args.gt, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("role") or "").strip().lower() != "test":
                continue

            bg = (r.get("background") or "").strip().lower()
            video = (r.get("video") or "").strip()
            key_video = (bg, video)

            if key_video not in TEST_MAP:
                continue

            label_txt = (r.get("gt_label") or "").strip().lower()
            valid_txt = (r.get("gt_valid_piece") or "").strip().lower()

            if label_txt == "unsure":
                excluded_unsure += 1
                continue

            if label_txt == "not_a_piece" or valid_txt not in {"1", "true", "yes"}:
                excluded_not_piece += 1
                continue

            label = norm_label(label_txt)
            if label is None:
                continue

            tid = int(float(r["track_id"]))
            gt[(bg, video, tid)] = {
                "background": bg,
                "video": video,
                "track_id": tid,
                "label": label,
                "label_name": "anomaly" if label else "normal",
            }

    # Load score sequences per test video/track.
    score_seqs = {}

    for (bg, video), tag in TEST_MAP.items():
        path = Path(args.scores) / bg / tag / "scores.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

        tmp = defaultdict(list)

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                ema = (r.get("ema_score") or "").strip()
                if not ema:
                    continue

                tid = int(float(r["track_id"]))
                frame = int(float(r["frame"]))
                tmp[tid].append((frame, float(ema)))

        for tid, vals in tmp.items():
            vals.sort(key=lambda z: z[0])
            score_seqs[(bg, video, tid)] = [v for _, v in vals]

    matched = []
    missing = []

    for key, g in gt.items():
        seq = score_seqs.get(key)
        if not seq:
            missing.append(g)
            continue

        max_ema = max(seq)
        pred = int(max_ema >= args.threshold)

        matched.append({
            **g,
            "n_scored_frames": len(seq),
            "max_ema": max_ema,
            "prediction": pred,
            "prediction_name": "anomaly" if pred else "normal",
            "correct": int(pred == g["label"]),
        })

    fields = [
        "background","video","track_id","label","label_name",
        "n_scored_frames","max_ema","prediction","prediction_name","correct"
    ]

    with (out / "final_track_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(matched)

    missing_fields = ["background","video","track_id","label","label_name"]
    with (out / "missing_test_tracks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=missing_fields)
        w.writeheader()
        w.writerows(missing)

    if not matched:
        raise RuntimeError("No test tracks matched.")

    y = np.asarray([r["label"] for r in matched], dtype=int)
    p = np.asarray([r["prediction"] for r in matched], dtype=int)
    s = np.asarray([r["max_ema"] for r in matched], dtype=float)

    overall = metrics(y, p)
    overall["auroc_max_ema"] = auc_roc(y, s)

    # Background metrics.
    per_bg = []
    for bg in ["black","metallic","white","wooden"]:
        rr = [r for r in matched if r["background"] == bg]
        if not rr:
            continue
        yy = np.asarray([r["label"] for r in rr], int)
        pp = np.asarray([r["prediction"] for r in rr], int)
        ss = np.asarray([r["max_ema"] for r in rr], float)

        m = metrics(yy, pp)
        m["background"] = bg
        m["n"] = len(rr)
        m["normal"] = int(np.sum(yy == 0))
        m["anomaly"] = int(np.sum(yy == 1))
        m["auroc_max_ema"] = auc_roc(yy, ss)
        per_bg.append(m)

    # Video metrics.
    per_video = []
    for (bg, video), tag in TEST_MAP.items():
        rr = [r for r in matched if r["background"] == bg and r["video"] == video]
        if not rr:
            continue
        yy = np.asarray([r["label"] for r in rr], int)
        pp = np.asarray([r["prediction"] for r in rr], int)
        ss = np.asarray([r["max_ema"] for r in rr], float)

        m = metrics(yy, pp)
        m["background"] = bg
        m["video"] = video
        m["tag"] = tag
        m["n"] = len(rr)
        m["normal"] = int(np.sum(yy == 0))
        m["anomaly"] = int(np.sum(yy == 1))
        m["auroc_max_ema"] = auc_roc(yy, ss)
        per_video.append(m)

    bg_fields = [
        "background","n","normal","anomaly","TP","TN","FP","FN",
        "precision","recall","f1","fpr","specificity","accuracy","auroc_max_ema"
    ]
    with (out / "per_background_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=bg_fields)
        w.writeheader()
        w.writerows(per_bg)

    vid_fields = [
        "background","video","tag","n","normal","anomaly","TP","TN","FP","FN",
        "precision","recall","f1","fpr","specificity","accuracy","auroc_max_ema"
    ]
    with (out / "per_video_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=vid_fields)
        w.writeheader()
        w.writerows(per_video)

    summary = {
        "configuration": {
            "detector": "YOLOv8n multi-background fine-tuned",
            "anomaly_model": "EfficientAD LowRes192",
            "ema_alpha": 0.3,
            "threshold": args.threshold,
            "track_decision": "anomaly if max valid EMA >= threshold",
            "threshold_frozen_before_final_test": True,
        },
        "ground_truth": {
            "test_valid_labeled_tracks": len(gt),
            "matched_scored_tracks": len(matched),
            "missing_tracks": len(missing),
            "excluded_unsure": excluded_unsure,
            "excluded_not_a_piece": excluded_not_piece,
            "normal_matched": int(np.sum(y == 0)),
            "anomaly_matched": int(np.sum(y == 1)),
        },
        "overall": overall,
        "per_background": per_bg,
        "per_video": per_video,
    }

    (out / "final_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("=" * 78)
    print("FINAL UNSEEN TEST — LOWRES192")
    print("=" * 78)
    print(f"Threshold                  : {args.threshold:.4f}")
    print(f"GT valid test tracks       : {len(gt)}")
    print(f"Matched scored tracks      : {len(matched)}")
    print(f"Missing scored tracks      : {len(missing)}")
    print(f"Excluded unsure            : {excluded_unsure}")
    print(f"Excluded not-a-piece       : {excluded_not_piece}")
    print(f"Matched normal / anomaly   : {np.sum(y==0)} / {np.sum(y==1)}")
    print()
    print("OVERALL")
    print(
        f"TP={overall['TP']} TN={overall['TN']} "
        f"FP={overall['FP']} FN={overall['FN']} | "
        f"P={overall['precision']:.4f} "
        f"R={overall['recall']:.4f} "
        f"F1={overall['f1']:.4f} "
        f"FPR={overall['fpr']:.4f} "
        f"Acc={overall['accuracy']:.4f} "
        f"AUC={overall['auroc_max_ema'] if overall['auroc_max_ema'] is not None else 'N/A'}"
    )
    print()
    print("PER BACKGROUND")
    for m in per_bg:
        auc = m["auroc_max_ema"]
        aucs = f"{auc:.3f}" if auc is not None else "N/A"
        print(
            f"{m['background']:<8s} n={m['n']:>3d} | "
            f"TP={m['TP']:>2d} TN={m['TN']:>3d} "
            f"FP={m['FP']:>3d} FN={m['FN']:>2d} | "
            f"P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} FPR={m['fpr']:.3f} "
            f"Acc={m['accuracy']:.3f} AUC={aucs}"
        )

    if missing:
        print()
        print("WARNING: inspect missing_test_tracks.csv before reporting final metrics.")

    print()
    print("Artifacts:", out.resolve())

if __name__ == "__main__":
    main()
