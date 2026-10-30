#!/usr/bin/env python3
"""
Fair old-vs-new YOLO comparison for the final 12 unseen videos.

Reference labels are attached to NEW-YOLO track IDs. Therefore OLD-YOLO
tracks are first mapped to those physical pieces using temporal + spatial
trajectory overlap (frame-by-frame bounding-box IoU).

The script reports:
1) detection/track coverage,
2) old-YOLO classification metrics on matched physical pieces,
3) new-YOLO classification metrics,
4) a COMMON-SUBSET comparison using exactly the same physical pieces,
5) mapping-quality / ambiguity information for manual audit.

No threshold tuning is performed here.
"""

from pathlib import Path
from collections import defaultdict
import argparse, csv, json, math
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

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    aa = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    bb = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    den = aa + bb - inter
    return inter/den if den > 0 else 0.0

def auc_roc(y, scores):
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5*np.sum(p == neg)
    return float(wins / (len(pos)*len(neg)))

def metrics(rows):
    if not rows:
        return {
            "n":0,"normal":0,"anomaly":0,"TP":0,"TN":0,"FP":0,"FN":0,
            "precision":0.0,"recall":0.0,"f1":0.0,"fpr":0.0,
            "specificity":0.0,"accuracy":0.0,"auroc":None
        }
    y = np.array([r["label"] for r in rows], dtype=int)
    p = np.array([r["prediction"] for r in rows], dtype=int)
    s = np.array([r["score"] for r in rows], dtype=float)

    tp = int(np.sum((y==1)&(p==1)))
    tn = int(np.sum((y==0)&(p==0)))
    fp = int(np.sum((y==0)&(p==1)))
    fn = int(np.sum((y==1)&(p==0)))

    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    fpr = fp/(fp+tn) if fp+tn else 0.0
    specificity = tn/(tn+fp) if tn+fp else 0.0
    accuracy = (tp+tn)/len(rows) if rows else 0.0

    return {
        "n": len(rows),
        "normal": int(np.sum(y==0)),
        "anomaly": int(np.sum(y==1)),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fpr, "specificity": specificity, "accuracy": accuracy,
        "auroc": auc_roc(y, s),
    }

def load_scores(path):
    tracks = defaultdict(lambda: {"boxes":{}, "ema":[]})
    n_rows = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            tid = int(float(r["track_id"]))
            fr = int(float(r["frame"]))
            box = tuple(float(r[k]) for k in ("x1","y1","x2","y2"))
            tracks[tid]["boxes"][fr] = box
            e = (r.get("ema_score") or "").strip()
            if e:
                tracks[tid]["ema"].append(float(e))
    return tracks, n_rows

def similarity(old_t, ref_t):
    of = set(old_t["boxes"])
    rf = set(ref_t["boxes"])
    common = sorted(of & rf)
    if not common:
        return None

    vals = [iou(old_t["boxes"][fr], ref_t["boxes"][fr]) for fr in common]
    mean_iou = float(np.mean(vals))
    median_iou = float(np.median(vals))
    hit03 = float(np.mean(np.asarray(vals) >= 0.30))
    overlap_shorter = len(common) / max(1, min(len(of), len(rf)))
    overlap_old = len(common) / max(1, len(of))
    overlap_ref = len(common) / max(1, len(rf))

    # Fragment-friendly: a short OLD track may correspond to part of one
    # physical NEW-YOLO reference track.
    spatial = 0.5*median_iou + 0.5*mean_iou
    quality = spatial * math.sqrt(overlap_shorter)

    return {
        "quality": quality,
        "n_common": len(common),
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "hit_fraction_iou03": hit03,
        "overlap_shorter": overlap_shorter,
        "overlap_old": overlap_old,
        "overlap_ref": overlap_ref,
    }

def accepted_match(sim, old_len):
    if sim is None:
        return False

    # Normal tracks: require at least 3 shared frames and reasonable spatial
    # agreement. Very short fragments are accepted only with strong IoU.
    if sim["n_common"] >= 3:
        return (
            sim["quality"] >= 0.20 and
            (sim["median_iou"] >= 0.20 or sim["hit_fraction_iou03"] >= 0.50)
        )

    if old_len <= 2 and sim["n_common"] >= 1:
        return sim["mean_iou"] >= 0.55

    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="./gt_reserved_newyolo/tracks_to_label.csv")
    ap.add_argument("--old-scores", default="./final_unseen_lowres192_oldyolo")
    ap.add_argument("--new-scores", default="./final_unseen_lowres192_newyolo_recalibrated")
    ap.add_argument("--old-threshold", type=float, default=0.833292)
    ap.add_argument("--new-threshold", type=float, default=0.83485001)
    ap.add_argument("--out", default="./old_vs_new_yolo_fair_comparison")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load valid TEST GT
    # -------------------------
    gt = {}
    excluded_unsure = 0
    excluded_not_piece = 0

    with open(args.gt, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("role") or "").strip().lower() != "test":
                continue

            bg = (r.get("background") or "").strip().lower()
            video = (r.get("video") or "").strip()

            if (bg, video) not in TEST_MAP:
                continue

            label_txt = (r.get("gt_label") or "").strip().lower()
            valid_txt = (r.get("gt_valid_piece") or "").strip().lower()

            if label_txt == "unsure":
                excluded_unsure += 1
                continue
            if label_txt == "not_a_piece" or valid_txt not in {"1","true","yes"}:
                excluded_not_piece += 1
                continue

            label = norm_label(label_txt)
            if label is None:
                continue

            tid = int(float(r["track_id"]))
            gt[(bg, video, tid)] = {
                "background": bg, "video": video,
                "ref_track_id": tid, "label": label,
                "label_name": "anomaly" if label else "normal",
            }

    # -------------------------
    # Load trajectories
    # -------------------------
    new_tracks_by_video = {}
    old_tracks_by_video = {}
    old_detection_rows = 0
    new_detection_rows = 0

    for (bg, video), tag in TEST_MAP.items():
        npth = Path(args.new_scores) / bg / tag / "scores.csv"
        opth = Path(args.old_scores) / bg / tag / "scores.csv"

        if not npth.exists():
            raise FileNotFoundError(f"Missing NEW score file: {npth}")
        if not opth.exists():
            raise FileNotFoundError(f"Missing OLD score file: {opth}")

        nt, nr = load_scores(npth)
        ot, orows = load_scores(opth)
        new_tracks_by_video[(bg,video)] = nt
        old_tracks_by_video[(bg,video)] = ot
        new_detection_rows += nr
        old_detection_rows += orows

    # -------------------------
    # Build NEW reference pieces
    # -------------------------
    refs = {}
    reference_missing = []

    for key, g in gt.items():
        bg, video, tid = key
        tr = new_tracks_by_video[(bg,video)].get(tid)
        if tr is None or not tr["boxes"]:
            reference_missing.append(g)
            continue

        refs[key] = {
            **g,
            "track": tr,
        }

    # -------------------------
    # Map each OLD track to the most likely physical reference piece.
    # Many OLD fragments may map to one physical piece.
    # -------------------------
    mapping_rows = []
    assignments = defaultdict(list)

    for (bg, video), old_tracks in old_tracks_by_video.items():
        candidates = [r for k,r in refs.items()
                      if r["background"] == bg and r["video"] == video]

        for old_tid, old_t in old_tracks.items():
            ranked = []
            for ref in candidates:
                sim = similarity(old_t, ref["track"])
                if sim is not None:
                    ranked.append((sim["quality"], ref, sim))

            ranked.sort(key=lambda x: x[0], reverse=True)

            best = ranked[0] if ranked else None
            second = ranked[1] if len(ranked) > 1 else None

            accepted = False
            ambiguous = False
            ref_tid = ""
            sim = None
            second_quality = ""

            if best is not None:
                _, ref, sim = best
                accepted = accepted_match(sim, len(old_t["boxes"]))
                ref_tid = ref["ref_track_id"] if accepted else ""
                second_quality = second[0] if second else ""

                if accepted and second is not None:
                    # Flag close alternatives for manual audit; keep assignment
                    # so the automatic result remains complete.
                    ambiguous = (best[0] - second[0]) < 0.03

                if accepted:
                    ref_key = (bg, video, ref["ref_track_id"])
                    assignments[ref_key].append(old_tid)

            mapping_rows.append({
                "background": bg,
                "video": video,
                "old_track_id": old_tid,
                "old_track_frames": len(old_t["boxes"]),
                "matched_ref_track_id": ref_tid,
                "accepted": int(accepted),
                "ambiguous": int(ambiguous),
                "quality": "" if sim is None else sim["quality"],
                "second_best_quality": second_quality,
                "n_common": "" if sim is None else sim["n_common"],
                "mean_iou": "" if sim is None else sim["mean_iou"],
                "median_iou": "" if sim is None else sim["median_iou"],
                "hit_fraction_iou03": "" if sim is None else sim["hit_fraction_iou03"],
                "overlap_shorter": "" if sim is None else sim["overlap_shorter"],
                "overlap_old": "" if sim is None else sim["overlap_old"],
                "overlap_ref": "" if sim is None else sim["overlap_ref"],
            })

    # -------------------------
    # Piece-level predictions
    # -------------------------
    old_rows = []
    new_rows = []
    piece_rows = []

    for key, ref in refs.items():
        bg, video, ref_tid = key
        label = ref["label"]

        # NEW prediction
        new_ema = ref["track"]["ema"]
        new_score = max(new_ema) if new_ema else None
        new_pred = int(new_score >= args.new_threshold) if new_score is not None else None

        if new_score is not None:
            new_rows.append({
                "key": key, "label": label,
                "score": new_score, "prediction": new_pred
            })

        # OLD prediction, aggregating all old-track fragments mapped to this piece.
        old_tids = assignments.get(key, [])
        old_ema = []
        for tid in old_tids:
            old_ema.extend(old_tracks_by_video[(bg,video)][tid]["ema"])

        old_score = max(old_ema) if old_ema else None
        old_pred = int(old_score >= args.old_threshold) if old_score is not None else None

        if old_score is not None:
            old_rows.append({
                "key": key, "label": label,
                "score": old_score, "prediction": old_pred
            })

        piece_rows.append({
            "background": bg,
            "video": video,
            "ref_track_id": ref_tid,
            "label": label,
            "label_name": ref["label_name"],
            "new_score": "" if new_score is None else new_score,
            "new_prediction": "" if new_pred is None else new_pred,
            "old_mapped_track_ids": ";".join(map(str, old_tids)),
            "old_n_mapped_tracks": len(old_tids),
            "old_score": "" if old_score is None else old_score,
            "old_prediction": "" if old_pred is None else old_pred,
        })

    old_by_key = {r["key"]:r for r in old_rows}
    new_by_key = {r["key"]:r for r in new_rows}
    common_keys = sorted(set(old_by_key) & set(new_by_key))

    old_common = [old_by_key[k] for k in common_keys]
    new_common = [new_by_key[k] for k in common_keys]

    old_m = metrics(old_rows)
    new_m = metrics(new_rows)
    old_common_m = metrics(old_common)
    new_common_m = metrics(new_common)

    # Per-background common subset.
    per_bg = []
    for bg in ["black","metallic","white","wooden"]:
        keys_bg = [k for k in common_keys if k[0] == bg]
        om = metrics([old_by_key[k] for k in keys_bg])
        nm = metrics([new_by_key[k] for k in keys_bg])
        per_bg.append({
            "background": bg,
            "n_common": len(keys_bg),
            "old": om,
            "new": nm,
        })

    # Coverage / misses
    old_matched_keys = set(old_by_key)
    valid_ref_keys = set(refs)
    old_missing = sorted(valid_ref_keys - old_matched_keys)

    old_missing_anomaly = sum(refs[k]["label"] == 1 for k in old_missing)
    old_missing_normal = sum(refs[k]["label"] == 0 for k in old_missing)

    # End-to-end anomaly recall counts missing anomalous pieces as FN.
    old_e2e_tp = old_m["TP"]
    old_e2e_fn = old_m["FN"] + old_missing_anomaly
    old_e2e_recall = (
        old_e2e_tp/(old_e2e_tp+old_e2e_fn)
        if old_e2e_tp+old_e2e_fn else 0.0
    )

    new_missing_anomaly = sum(g["label"] == 1 for g in reference_missing)
    new_e2e_tp = new_m["TP"]
    new_e2e_fn = new_m["FN"] + new_missing_anomaly
    new_e2e_recall = (
        new_e2e_tp/(new_e2e_tp+new_e2e_fn)
        if new_e2e_tp+new_e2e_fn else 0.0
    )

    ambiguous_count = sum(int(r["ambiguous"]) for r in mapping_rows)
    accepted_old_tracks = sum(int(r["accepted"]) for r in mapping_rows)

    summary = {
        "thresholds": {
            "old_yolo_lowres192": args.old_threshold,
            "new_yolo_lowres192": args.new_threshold,
        },
        "ground_truth": {
            "valid_test_tracks_total": len(gt),
            "reference_tracks_available_from_new_yolo": len(refs),
            "reference_tracks_missing_from_new_yolo": len(reference_missing),
            "excluded_unsure": excluded_unsure,
            "excluded_not_a_piece": excluded_not_piece,
        },
        "detection_rows": {
            "old_yolo": old_detection_rows,
            "new_yolo": new_detection_rows,
            "new_vs_old_percent": (
                100.0*(new_detection_rows-old_detection_rows)/old_detection_rows
                if old_detection_rows else None
            ),
        },
        "old_mapping": {
            "accepted_old_tracks": accepted_old_tracks,
            "ambiguous_old_tracks": ambiguous_count,
            "physical_reference_pieces_with_old_prediction": len(old_rows),
            "physical_reference_pieces_missing_old_prediction": len(old_missing),
            "missing_old_normal": old_missing_normal,
            "missing_old_anomaly": old_missing_anomaly,
            "coverage_vs_reference_percent": 100.0*len(old_rows)/len(refs) if refs else 0.0,
        },
        "classification_on_available_pieces": {
            "old_yolo": old_m,
            "new_yolo": new_m,
        },
        "common_subset_same_physical_pieces": {
            "n": len(common_keys),
            "old_yolo": old_common_m,
            "new_yolo": new_common_m,
            "per_background": per_bg,
        },
        "end_to_end_anomaly_recall_including_detection_misses": {
            "old_yolo": old_e2e_recall,
            "new_yolo": new_e2e_recall,
        },
    }

    # -------------------------
    # Save artifacts
    # -------------------------
    map_fields = [
        "background","video","old_track_id","old_track_frames",
        "matched_ref_track_id","accepted","ambiguous","quality",
        "second_best_quality","n_common","mean_iou","median_iou",
        "hit_fraction_iou03","overlap_shorter","overlap_old","overlap_ref"
    ]
    with (out/"old_track_to_physical_piece_mapping.csv").open("w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=map_fields)
        w.writeheader(); w.writerows(mapping_rows)

    piece_fields = [
        "background","video","ref_track_id","label","label_name",
        "new_score","new_prediction","old_mapped_track_ids",
        "old_n_mapped_tracks","old_score","old_prediction"
    ]
    with (out/"physical_piece_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=piece_fields)
        w.writeheader(); w.writerows(piece_rows)

    with (out/"reference_missing_new_yolo.csv").open("w",newline="",encoding="utf-8") as f:
        fields = ["background","video","ref_track_id","label","label_name"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(reference_missing)

    with (out/"old_yolo_missing_physical_pieces.csv").open("w",newline="",encoding="utf-8") as f:
        fields = ["background","video","ref_track_id","label","label_name"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in old_missing:
            r = refs[k]
            w.writerow({x:r[x] for x in fields})

    (out/"fair_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Flat comparison CSV for report.
    rows_flat = []
    for scope, om, nm in [
        ("available_per_pipeline", old_m, new_m),
        ("common_subset", old_common_m, new_common_m),
    ]:
        for name, m in [("old_yolo",om),("new_yolo",nm)]:
            rows_flat.append({
                "scope": scope, "pipeline": name,
                "n":m["n"],"TP":m["TP"],"TN":m["TN"],"FP":m["FP"],"FN":m["FN"],
                "precision":m["precision"],"recall":m["recall"],"f1":m["f1"],
                "fpr":m["fpr"],"specificity":m["specificity"],
                "accuracy":m["accuracy"],"auroc":m["auroc"],
            })
    fields = ["scope","pipeline","n","TP","TN","FP","FN","precision","recall","f1","fpr","specificity","accuracy","auroc"]
    with (out/"old_vs_new_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows_flat)

    # -------------------------
    # Terminal summary
    # -------------------------
    def fmt(m):
        auc = "N/A" if m["auroc"] is None else f"{m['auroc']:.4f}"
        return (
            f"n={m['n']} | TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']} | "
            f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
            f"FPR={m['fpr']:.4f} Acc={m['accuracy']:.4f} AUC={auc}"
        )

    print("="*90)
    print("FAIR OLD-vs-NEW YOLO COMPARISON — LOWRES192")
    print("="*90)
    print(f"Old threshold : {args.old_threshold:.6f}")
    print(f"New threshold : {args.new_threshold:.8f}")
    print(f"GT valid pieces                         : {len(gt)}")
    print(f"NEW reference trajectories available    : {len(refs)}")
    print(f"NEW reference trajectories missing      : {len(reference_missing)}")
    print()
    print("DETECTION ROWS")
    print(f"Old YOLO : {old_detection_rows}")
    print(f"New YOLO : {new_detection_rows}")
    if old_detection_rows:
        pct = 100.0*(new_detection_rows-old_detection_rows)/old_detection_rows
        print(f"Change   : {pct:+.2f}%")
    print()
    print("OLD->PHYSICAL-PIECE MAPPING")
    print(f"Accepted old tracks                     : {accepted_old_tracks}")
    print(f"Ambiguous accepted old tracks           : {ambiguous_count}")
    print(f"Physical pieces with OLD prediction     : {len(old_rows)}/{len(refs)}")
    print(f"OLD coverage                            : {100.0*len(old_rows)/len(refs):.2f}%")
    print(f"OLD missing normal / anomaly            : {old_missing_normal} / {old_missing_anomaly}")
    print()
    print("CLASSIFICATION — EACH PIPELINE'S AVAILABLE PIECES")
    print("OLD :", fmt(old_m))
    print("NEW :", fmt(new_m))
    print()
    print(f"COMMON SUBSET — EXACT SAME {len(common_keys)} PHYSICAL PIECES")
    print("OLD :", fmt(old_common_m))
    print("NEW :", fmt(new_common_m))
    print()
    print("END-TO-END ANOMALY RECALL INCLUDING DETECTOR MISSES")
    print(f"OLD : {old_e2e_recall:.4f}")
    print(f"NEW : {new_e2e_recall:.4f}")
    print()
    print("PER BACKGROUND — COMMON SUBSET")
    for x in per_bg:
        print(f"{x['background']:<8s} n={x['n_common']}")
        print("  OLD:", fmt(x["old"]))
        print("  NEW:", fmt(x["new"]))
    print()
    if ambiguous_count:
        print("NOTE: ambiguous mappings are flagged in old_track_to_physical_piece_mapping.csv")
        print("      Review them before using the result as the final reported table.")
    print(f"Artifacts: {out.resolve()}")

if __name__ == "__main__":
    main()
