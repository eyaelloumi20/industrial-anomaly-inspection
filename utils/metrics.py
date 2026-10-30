"""
metrics.py
==========
Evaluation metrics for the two-stage inspection pipeline.

Computes image-level:
    AUROC         – Area Under the ROC Curve
    AUPRO         – Area Under the Per-Region Overlap Curve (FPR ≤ 0.3)
    AP            – Average Precision (area under P-R curve)
    Confusion     – TP, TN, FP, FN at the Youden-J optimal threshold
    Inference stats – mean / std / total time

All functions accept plain numpy arrays and are dataset-agnostic.

Exported symbols
----------------
    compute_auroc(y_true, y_score)
    compute_aupro(y_true, y_score, fpr_limit=0.3)
    compute_ap(y_true, y_score)
    optimal_threshold(y_true, y_score)
    confusion_at_threshold(y_true, y_score, threshold)
    bootstrap_std(y_true, y_score, n_boot=500, seed=42)
    compute_all_metrics(y_true, y_score, times_s)
    print_results(metrics_dict, category)
    save_quantitative_txt(results_dict, txt_path)
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
)

# np.trapz was removed in NumPy 2.0; use np.trapezoid when available
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ── individual metrics ────────────────────────────────────────────────────────

def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def compute_aupro(
    y_true:         np.ndarray,
    y_score:        np.ndarray,
    num_thresholds: int   = 100,
    fpr_limit:      float = 0.3,
) -> float:
    """
    Image-level proxy for AUPRO: integrate the Per-Region Overlap (TPR)
    over FPR ∈ [0, fpr_limit] and normalise by fpr_limit.
    """
    thresholds = np.linspace(y_score.min(), y_score.max(), num_thresholds)
    fprs, pros = [], []

    for thresh in thresholds:
        preds = (y_score >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tn = np.sum((preds == 0) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        fprs.append(fp / (fp + tn + 1e-8))
        pros.append(tp / (tp + fn + 1e-8))

    fprs = np.array(fprs);  pros = np.array(pros)
    idx  = np.argsort(fprs);  fprs, pros = fprs[idx], pros[idx]
    mask = fprs <= fpr_limit

    if mask.sum() < 2:
        return float(_trapz(pros[mask], fprs[mask]) / fpr_limit) \
               if mask.sum() else 0.0
    return float(_trapz(pros[mask], fprs[mask]) / fpr_limit)


def optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Youden J statistic: threshold that maximises TPR − FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[np.argmax(tpr - fpr)])


def confusion_at_threshold(
    y_true:    np.ndarray,
    y_score:   np.ndarray,
    threshold: float,
) -> dict:
    preds = (y_score >= threshold).astype(int)
    return {
        "tp": int(np.sum((preds == 1) & (y_true == 1))),
        "tn": int(np.sum((preds == 0) & (y_true == 0))),
        "fp": int(np.sum((preds == 1) & (y_true == 0))),
        "fn": int(np.sum((preds == 0) & (y_true == 1))),
        "preds": preds,
    }


def bootstrap_std(
    y_true:  np.ndarray,
    y_score: np.ndarray,
    n_boot:  int = 500,
    seed:    int = 42,
) -> dict:
    """Bootstrap standard deviations for AUROC, AUPRO, AP."""
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    boot_auroc, boot_aupro, boot_ap = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_auroc.append(roc_auc_score(yt, ys))
        boot_aupro.append(compute_aupro(yt, ys))
        boot_ap.append(average_precision_score(yt, ys))

    return {
        "std_auroc": float(np.std(boot_auroc)) if boot_auroc else float("nan"),
        "std_aupro": float(np.std(boot_aupro)) if boot_aupro else float("nan"),
        "std_ap":    float(np.std(boot_ap))    if boot_ap    else float("nan"),
    }


# ── combined entry-point ──────────────────────────────────────────────────────

def compute_all_metrics(
    y_true:   np.ndarray,
    y_score:  np.ndarray,
    times_s:  np.ndarray,   # per-image times in seconds
    n_boot:   int = 500,
) -> dict:
    """
    Compute the full metric suite and return a flat dict.
    """
    auroc = compute_auroc(y_true, y_score)
    aupro = compute_aupro(y_true, y_score)
    ap    = compute_ap(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    thresh       = optimal_threshold(y_true, y_score)
    conf         = confusion_at_threshold(y_true, y_score, thresh)

    std_dict = bootstrap_std(y_true, y_score, n_boot=n_boot)

    scores_n = y_score[y_true == 0]
    scores_a = y_score[y_true == 1]

    return {
        # image-level AUCs
        "auroc":       auroc,
        "aupro":       aupro,
        "ap":          ap,
        "std_auroc":   std_dict["std_auroc"],
        "std_aupro":   std_dict["std_aupro"],
        "std_ap":      std_dict["std_ap"],
        # threshold & confusion
        "threshold":   thresh,
        "tp":          conf["tp"],
        "tn":          conf["tn"],
        "fp":          conf["fp"],
        "fn":          conf["fn"],
        "preds":       conf["preds"],
        # ROC arrays (for plotting)
        "fpr":         fpr,
        "tpr":         tpr,
        # score distributions
        "score_normal_mean":   float(scores_n.mean()) if len(scores_n) else float("nan"),
        "score_normal_std":    float(scores_n.std())  if len(scores_n) else float("nan"),
        "score_anomaly_mean":  float(scores_a.mean()) if len(scores_a) else float("nan"),
        "score_anomaly_std":   float(scores_a.std())  if len(scores_a) else float("nan"),
        # counts
        "n_total":   len(y_true),
        "n_normal":  int(np.sum(y_true == 0)),
        "n_anomaly": int(np.sum(y_true == 1)),
        # timing
        "inf_mean_ms":  float(times_s.mean() * 1000),
        "inf_std_ms":   float(times_s.std()  * 1000),
        "inf_total_s":  float(times_s.sum()),
    }


# ── display helpers ───────────────────────────────────────────────────────────

def print_results(m: dict, category: str = "") -> None:
    """Pretty-print a metrics dict to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Results  —  {category.upper()}")
    print(sep)
    print(f"  AUROC : {m['auroc']:.5f}  ± {m['std_auroc']:.5f}  (bootstrap n=500)")
    print(f"  AUPRO : {m['aupro']:.5f}  ± {m['std_aupro']:.5f}")
    print(f"  AP    : {m['ap']:.5f}  ± {m['std_ap']:.5f}")
    print(f"  Threshold  : {m['threshold']:.5f}")
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  Scores — Normal  : {m['score_normal_mean']:.4f} ± {m['score_normal_std']:.4f}")
    print(f"  Scores — Anomaly : {m['score_anomaly_mean']:.4f} ± {m['score_anomaly_std']:.4f}")
    print(f"  Inf. time : {m['inf_mean_ms']:.2f} ± {m['inf_std_ms']:.2f} ms/img  "
          f"(total {m['inf_total_s']:.2f} s)")
    print(f"  N total={m['n_total']}  normal={m['n_normal']}  anomaly={m['n_anomaly']}")
    print(sep)


def save_quantitative_txt(results_dict: dict, txt_path: str) -> None:
    """
    Write a multi-category summary table to a text file.

    Parameters
    ----------
    results_dict : {category_name: metrics_dict, ...}
    txt_path     : output file path
    """
    header = (f"{'Category':<20} {'AUROC':>8} {'±':>7} {'AUPRO':>8} {'±':>7} "
              f"{'AP':>8} {'±':>7} "
              f"{'Score_N':>9} {'±':>7} {'Score_A':>9} {'±':>7} "
              f"{'Thr':>8} "
              f"{'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} "
              f"{'Inf_mean_ms':>13} {'Inf_std_ms':>11} {'Total_s':>10}\n")
    divider = "-" * 165 + "\n"

    with open(txt_path, "w") as f:
        f.write(header)
        f.write(divider)
        for cat, m in results_dict.items():
            f.write(
                f"{cat:<20} "
                f"{m['auroc']:>8.5f} {m['std_auroc']:>7.5f} "
                f"{m['aupro']:>8.5f} {m['std_aupro']:>7.5f} "
                f"{m['ap']:>8.5f} {m['std_ap']:>7.5f} "
                f"{m['score_normal_mean']:>9.4f} {m['score_normal_std']:>7.4f} "
                f"{m['score_anomaly_mean']:>9.4f} {m['score_anomaly_std']:>7.4f} "
                f"{m['threshold']:>8.5f} "
                f"{m['tp']:>4} {m['tn']:>4} {m['fp']:>4} {m['fn']:>4} "
                f"{m['inf_mean_ms']:>13.3f} {m['inf_std_ms']:>11.3f} "
                f"{m['inf_total_s']:>10.3f}\n"
            )
        f.write(divider)
        if len(results_dict) > 1:
            avg_auroc = np.mean([m["auroc"] for m in results_dict.values()])
            avg_aupro = np.mean([m["aupro"] for m in results_dict.values()])
            avg_ap    = np.mean([m["ap"]    for m in results_dict.values()])
            f.write(f"{'Average':<20} {avg_auroc:>8.5f} {'':>7} "
                    f"{avg_aupro:>8.5f} {'':>7} {avg_ap:>8.5f}\n")