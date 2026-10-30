"""
load_efficientad_thin.py
=========================
Thin-aware sibling of utils/load_efficientad.py, for checkpoints produced by
retrain_thin.py (models_thin.py's Student_Thin / AutoEncoder_Thin).

The original load_efficientad_model() always builds a full-width Student()
and AutoEncoder(), which will fail to load (shape mismatch) against a
checkpoint whose hidden channel widths were scaled by --width_mult during
training. This version:

  1. Reads 'width_mult' out of the checkpoint's quantiles.npy (defaults to
     1.0 if absent, e.g. a full-width checkpoint).
  2. Builds Student_Thin(model_size, width_mult) and
     AutoEncoder_Thin(width_mult) instead of the full-width classes.
  3. Builds the Teacher exactly as before — UNCHANGED, full-width, frozen
     — since retrain_thin.py never touches the teacher's architecture.
  4. Returns everything the original load_efficientad_model() returns, PLUS
     a 'width_mult' key, so callers know what width this checkpoint expects.

At width_mult=1.0 this is architecturally identical to the original
Student/AutoEncoder (same channel counts throughout), so it also works
unmodified on your original full-width checkpoints.

Usage
-----
    from load_efficientad_thin import load_efficientad_model

    models = load_efficientad_model(
        ckpt_dir   = "ckpt_thin_025",
        category   = "chewinggum",
        model_size = "S",
        device     = "cuda",
    )
    # models["width_mult"] tells you what width this checkpoint expects
"""

import os
import sys
import numpy as np
import torch

# ── make both this package (utils/) and the project root importable ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import Teacher
from models_thin import Student_Thin, AutoEncoder_Thin, count_params


def load_efficientad_model(
    ckpt_dir:   str,
    category:   str  = "chewinggum",
    model_size: str  = "S",
    device:     str  = "cpu",
) -> dict:
    """
    Load and return a fully initialised, width-aware EfficientAD model
    bundle.

    Returns
    -------
    dict with keys:
        "teacher", "student", "ae"  : nn.Module (eval, on device)
        "mean", "std"               : torch.Tensor  channel normalisation
        "qa_st", "qb_st"            : torch.Tensor  student-teacher quantiles
        "qa_ae", "qb_ae"            : torch.Tensor  autoencoder quantiles
        "width_mult"                : float -- the channel-width multiplier
                                       this checkpoint was trained at (from
                                       quantiles.npy, defaults to 1.0 if absent)

    Raises
    ------
    FileNotFoundError  if any required checkpoint file is missing.
    ValueError         if model_size is not "S" or "M".
    """
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

    print(f"[EfficientAD Loader] Loading quantiles : {required['quantiles']}")
    quantiles  = np.load(required["quantiles"], allow_pickle=True).item()
    width_mult = float(quantiles.get("width_mult", 1.0))
    print(f"[EfficientAD Loader] Checkpoint width_mult = {width_mult}")

    print(f"[EfficientAD Loader] Building PDN-{model_size} teacher (full width, frozen) ...")
    teacher = Teacher(model_size)
    print(f"[EfficientAD Loader] Loading teacher   : {required['teacher']}")
    teacher.load_state_dict(
        torch.load(required["teacher"], map_location=device, weights_only=False))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    student = Student_Thin(model_size, width_mult=width_mult)
    print(f"[EfficientAD Loader] Loading student   : {required['student']}  "
          f"({count_params(student):,} params)")
    student.load_state_dict(
        torch.load(required["student"], map_location=device, weights_only=False))
    student.eval().to(device)

    ae = AutoEncoder_Thin(width_mult=width_mult)
    print(f"[EfficientAD Loader] Loading autoencoder: {required['ae']}  "
          f"({count_params(ae):,} params)")
    ae.load_state_dict(
        torch.load(required["ae"], map_location=device, weights_only=False))
    ae.eval().to(device)

    def _to_tensor(value):
        arr = np.array(value)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    mean  = _to_tensor(quantiles["mean"])
    std   = _to_tensor(quantiles["std"])
    qa_st = _to_tensor(quantiles["qa_st"])
    qb_st = _to_tensor(quantiles["qb_st"])
    qa_ae = _to_tensor(quantiles["qa_ae"])
    qb_ae = _to_tensor(quantiles["qb_ae"])

    print(f"[EfficientAD Loader] \u2713 Model loaded successfully "
          f"(category='{category}', size='{model_size}', device='{device}', "
          f"width_mult={width_mult})")

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
        "width_mult": width_mult,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and verify a width-aware EfficientAD model bundle.")
    parser.add_argument("--ckpt_dir",   required=True)
    parser.add_argument("--category",   default="chewinggum")
    parser.add_argument("--model_size", default="S", choices=["S", "M"])
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    models = load_efficientad_model(
        ckpt_dir   = args.ckpt_dir,
        category   = args.category,
        model_size = args.model_size,
        device     = args.device,
    )
    print(f"\nStudent params : {count_params(models['student']):,}")
    print(f"AE params      : {count_params(models['ae']):,}")