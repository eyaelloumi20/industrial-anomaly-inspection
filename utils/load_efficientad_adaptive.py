"""
load_efficientad_adaptive.py
=============================
Resolution-aware sibling of utils/load_efficientad.py.

The original load_efficientad_model() always builds a plain AutoEncoder(),
which only works for checkpoints trained at 256x256 (see models_adaptive.py
for why). This version:

  1. Reads 'image_size' out of the checkpoint's quantiles.npy (defaults to
     256 if absent, which covers checkpoints trained before retrain_lowres.py
     existed).
  2. Builds AutoEncoderAdaptive sized to match the Teacher/Student's actual
     output spatial size at that image_size (measured with a real forward
     pass via probe_output_size, not computed by hand).
  3. Returns everything the original load_efficientad_model() returns, PLUS
     an 'image_size' key, so callers (e.g. InspectionPipelineAdaptive) know
     what resolution to resize crops to.

At 256px this is a strict drop-in: AutoEncoderAdaptive's state_dict keys and
parameter shapes are identical to the original AutoEncoder's regardless of
target_size, so this loader also works unmodified on your original 256px
checkpoints.

Usage
-----
    from load_efficientad_adaptive import load_efficientad_model

    models = load_efficientad_model(
        ckpt_dir   = "ckpt_192",
        category   = "chewinggum",
        model_size = "S",
        device     = "cuda",
    )
    # models["image_size"] tells you what resolution this checkpoint expects
"""

import os
import sys
import numpy as np
import torch

# ── make both this package (utils/) and the project root importable ──────────
# utils/ itself needs to be on sys.path so "from models import ..." resolves
# to utils/models.py (matches the trick used by the original load_efficientad.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The project root (one level up from utils/) needs to be on sys.path so
# "from models_adaptive import ..." resolves to the top-level models_adaptive.py.
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import Teacher, Student
from models_adaptive import AutoEncoderAdaptive, probe_output_size


def load_efficientad_model(
    ckpt_dir:   str,
    category:   str  = "chewinggum",
    model_size: str  = "S",
    device:     str  = "cpu",
) -> dict:
    """
    Load and return a fully initialised, resolution-aware EfficientAD model
    bundle.

    Returns
    -------
    dict with keys:
        "teacher", "student", "ae"  : nn.Module (eval, on device)
        "mean", "std"               : torch.Tensor  channel normalisation
        "qa_st", "qb_st"            : torch.Tensor  student-teacher quantiles
        "qa_ae", "qb_ae"            : torch.Tensor  autoencoder quantiles
        "image_size"                : int  -- the resolution this checkpoint
                                       was trained at (from quantiles.npy,
                                       defaults to 256 if absent)

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
    image_size = int(quantiles.get("image_size", 256))
    print(f"[EfficientAD Loader] Checkpoint image_size = {image_size}")

    print(f"[EfficientAD Loader] Building PDN-{model_size} architecture ...")
    teacher = Teacher(model_size)

    print(f"[EfficientAD Loader] Loading teacher   : {required['teacher']}")
    teacher.load_state_dict(
        torch.load(required["teacher"], map_location=device, weights_only=False))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # Measure the Teacher's actual output spatial size at this resolution
    # (rather than computing it by hand) so the AutoEncoder's decoder
    # schedule lines up exactly.
    pdn_out_size = probe_output_size(teacher, image_size, device)
    print(f"[EfficientAD Loader] Teacher/Student PDN output size: "
          f"{pdn_out_size}x{pdn_out_size}")

    student = Student(model_size)
    print(f"[EfficientAD Loader] Loading student   : {required['student']}")
    student.load_state_dict(
        torch.load(required["student"], map_location=device, weights_only=False))
    student.eval().to(device)

    ae = AutoEncoderAdaptive(target_size=pdn_out_size)
    print(f"[EfficientAD Loader] Loading autoencoder: {required['ae']}")
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
          f"image_size={image_size})")

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
        "image_size": image_size,
    }


def verify_model_bundle(models: dict, device: str = "cpu") -> None:
    """Quick sanity-check forward pass, sized to the bundle's own image_size."""
    import torch.nn.functional as F

    image_size = models.get("image_size", 256)
    print(f"\n[EfficientAD Loader] Running verification forward pass "
          f"at {image_size}px ...")
    dummy = torch.randn(1, 3, image_size, image_size).to(device)

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
    fm_st   = F.interpolate(fm_st,   size=(image_size, image_size), mode="bilinear", align_corners=False)
    fm_stae = F.interpolate(fm_stae, size=(image_size, image_size), mode="bilinear", align_corners=False)

    norm_mst = (0.1 * (fm_st   - models["qa_st"])) / (models["qb_st"] - models["qa_st"] + 1e-8)
    norm_mae = (0.1 * (fm_stae - models["qa_ae"])) / (models["qb_ae"] - models["qa_ae"] + 1e-8)

    combined = 0.5 * norm_mst + 0.5 * norm_mae
    score    = float(combined.max())

    print(f"  Teacher output   : {tuple(t_out.shape)}")
    print(f"  Student output   : {tuple(s_out.shape)}")
    print(f"  AE output        : {tuple(a_out.shape)}")
    print(f"  Anomaly map      : {tuple(combined.shape)}")
    print(f"  Image score      : {score:.6f}")
    print("[EfficientAD Loader] \u2713 Verification passed.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and verify a resolution-aware EfficientAD model bundle.")
    parser.add_argument("--ckpt_dir",   required=True)
    parser.add_argument("--category",   default="chewinggum")
    parser.add_argument("--model_size", default="S", choices=["S", "M"])
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--verify",     action="store_true")
    args = parser.parse_args()

    models = load_efficientad_model(
        ckpt_dir   = args.ckpt_dir,
        category   = args.category,
        model_size = args.model_size,
        device     = args.device,
    )

    if args.verify:
        verify_model_bundle(models, device=args.device)
