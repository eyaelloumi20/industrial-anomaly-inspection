"""
load_efficientad.py
====================
Loads a pre-trained EfficientAD model (Teacher + Student + AutoEncoder)
from a checkpoint directory.  No training is performed here.

Expected checkpoint layout
--------------------------
<ckpt_dir>/
    best_teacher.pth                  # Teacher PDN weights (shared)
    chewinggum_student.pth            # Student PDN weights
    chewinggum_autoencoder.pth        # AutoEncoder weights
    chewinggum_quantiles.npy          # Normalisation statistics dict:
                                      #   keys: mean, std, qa_st, qb_st,
                                      #          qa_ae, qb_ae

Usage
-----
    from load_efficientad import load_efficientad_model

    models = load_efficientad_model(
        ckpt_dir   = "ckptSmall",
        category   = "chewinggum",
        model_size = "S",          # "S" or "M"
        device     = "cuda",
    )

    # models is a plain dict with keys:
    #   teacher, student, ae          – nn.Module (eval mode, on device)
    #   mean, std                     – torch.Tensor  channel normalisation
    #   qa_st, qb_st, qa_ae, qb_ae   – torch.Tensor  score normalisation
"""

import os
import sys
import numpy as np
import torch

# ── allow running from any working directory ─────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models import Teacher, Student, AutoEncoder


# ── public API ────────────────────────────────────────────────────────────────

def load_efficientad_model(
    ckpt_dir:   str,
    category:   str  = "chewinggum",
    model_size: str  = "S",
    device:     str  = "cpu",
) -> dict:
    """
    Load and return a fully initialised EfficientAD model bundle.

    Parameters
    ----------
    ckpt_dir   : path to the folder that contains the .pth and .npy files.
    category   : dataset category name used as file prefix (e.g. "chewinggum").
    model_size : "S" (PDN-Small) or "M" (PDN-Medium).
    device     : "cuda" or "cpu".

    Returns
    -------
    dict with keys:
        "teacher"  : Teacher  (nn.Module, eval, on device)
        "student"  : Student  (nn.Module, eval, on device)
        "ae"       : AutoEncoder (nn.Module, eval, on device)
        "mean"     : torch.Tensor  [384,1,1]  channel mean
        "std"      : torch.Tensor  [384,1,1]  channel std
        "qa_st"    : torch.Tensor  scalar     student-teacher low quantile
        "qb_st"    : torch.Tensor  scalar     student-teacher high quantile
        "qa_ae"    : torch.Tensor  scalar     autoencoder low quantile
        "qb_ae"    : torch.Tensor  scalar     autoencoder high quantile

    Raises
    ------
    FileNotFoundError  if any required checkpoint file is missing.
    ValueError         if model_size is not "S" or "M".
    """

    # ── validate inputs ───────────────────────────────────────────────────────
    if model_size not in ("S", "M"):
        raise ValueError(f"model_size must be 'S' or 'M', got '{model_size}'")

    required = {
        "teacher":   os.path.join(ckpt_dir, "best_teacher.pth"),
        "student":   os.path.join(ckpt_dir, f"{category}_student.pth"),
        "ae":        os.path.join(ckpt_dir, f"{category}_autoencoder.pth"),
        "quantiles": os.path.join(ckpt_dir, f"{category}_quantiles.npy"),
    }

    missing = [name for name, path in required.items()
               if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint files for '{category}' in '{ckpt_dir}':\n"
            + "\n".join(f"  [{k}] {required[k]}" for k in missing)
        )

    # ── build model instances ─────────────────────────────────────────────────
    print(f"[EfficientAD Loader] Building PDN-{model_size} architecture ...")
    teacher = Teacher(model_size)
    student = Student(model_size)
    ae      = AutoEncoder()

    # ── load weights ──────────────────────────────────────────────────────────
    print(f"[EfficientAD Loader] Loading teacher   : {required['teacher']}")
    teacher.load_state_dict(
        torch.load(required["teacher"],
                   map_location=device, weights_only=False))

    print(f"[EfficientAD Loader] Loading student   : {required['student']}")
    student.load_state_dict(
        torch.load(required["student"],
                   map_location=device, weights_only=False))

    print(f"[EfficientAD Loader] Loading autoencoder: {required['ae']}")
    ae.load_state_dict(
        torch.load(required["ae"],
                   map_location=device, weights_only=False))

    # ── move to device and set eval mode ─────────────────────────────────────
    for model in (teacher, student, ae):
        model.eval()
        model.to(device)

    # ── load normalisation quantiles ─────────────────────────────────────────
    print(f"[EfficientAD Loader] Loading quantiles : {required['quantiles']}")
    quantiles = np.load(
        required["quantiles"], allow_pickle=True).item()

    def _to_tensor(value):
        arr = np.array(value)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    mean   = _to_tensor(quantiles["mean"])
    std    = _to_tensor(quantiles["std"])
    qa_st  = _to_tensor(quantiles["qa_st"])
    qb_st  = _to_tensor(quantiles["qb_st"])
    qa_ae  = _to_tensor(quantiles["qa_ae"])
    qb_ae  = _to_tensor(quantiles["qb_ae"])

    print(f"[EfficientAD Loader] ✓ Model loaded successfully "
          f"(category='{category}', size='{model_size}', device='{device}')")

    return {
        "teacher": teacher,
        "student": student,
        "ae":      ae,
        "mean":    mean,
        "std":     std,
        "qa_st":   qa_st,
        "qb_st":   qb_st,
        "qa_ae":   qa_ae,
        "qb_ae":   qb_ae,
    }


def verify_model_bundle(models: dict, device: str = "cpu") -> None:
    """
    Run a quick sanity-check forward pass on the loaded model bundle.
    Prints tensor shapes and confirms the pipeline is working end-to-end.

    Parameters
    ----------
    models : dict returned by load_efficientad_model()
    device : same device as used when loading
    """
    import torch.nn.functional as F

    print("\n[EfficientAD Loader] Running verification forward pass ...")
    dummy = torch.randn(1, 3, 256, 256).to(device)

    with torch.no_grad():
        t_out = models["teacher"](dummy)
        s_out = models["student"](dummy)
        a_out = models["ae"](dummy)

    y_st   = s_out[:, :384, :, :]
    y_stae = s_out[:, -384:, :, :]

    norm_t  = (t_out - models["mean"]) / (models["std"] + 1e-8)
    d_st    = torch.pow(norm_t - y_st,   2)
    d_stae  = torch.pow(a_out  - y_stae, 2)

    fm_st   = torch.mean(d_st,   dim=1, keepdim=True)
    fm_stae = torch.mean(d_stae, dim=1, keepdim=True)

    fm_st   = F.interpolate(fm_st,   size=(256, 256), mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(fm_stae, size=(256, 256), mode="bilinear", align_corners=False)

    norm_mst = (0.1 * (fm_st   - models["qa_st"])) / (models["qb_st"] - models["qa_st"] + 1e-8)
    norm_mae = (0.1 * (fm_stae - models["qa_ae"])) / (models["qb_ae"] - models["qa_ae"] + 1e-8)

    combined = 0.5 * norm_mst + 0.5 * norm_mae
    score    = float(combined.max())

    print(f"  Teacher output   : {tuple(t_out.shape)}")
    print(f"  Student output   : {tuple(s_out.shape)}")
    print(f"  AE output        : {tuple(a_out.shape)}")
    print(f"  Anomaly map      : {tuple(combined.shape)}")
    print(f"  Image score      : {score:.6f}")
    print("[EfficientAD Loader] ✓ Verification passed.\n")


# ── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and verify a pre-trained EfficientAD model bundle.")
    parser.add_argument("--ckpt_dir",   required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--category",   default="chewinggum",
                        help="Category name (default: chewinggum)")
    parser.add_argument("--model_size", default="S", choices=["S", "M"],
                        help="PDN size: S or M (default: S)")
    parser.add_argument("--device",     default="cpu",
                        help="Device: cuda or cpu (default: cpu)")
    parser.add_argument("--verify",     action="store_true",
                        help="Run a sanity-check forward pass after loading")
    args = parser.parse_args()

    models = load_efficientad_model(
        ckpt_dir   = args.ckpt_dir,
        category   = args.category,
        model_size = args.model_size,
        device     = args.device,
    )

    if args.verify:
        verify_model_bundle(models, device=args.device)
