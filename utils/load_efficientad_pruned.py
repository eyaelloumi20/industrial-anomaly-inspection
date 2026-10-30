"""
load_efficientad_pruned.py
=============================
Loader for checkpoints produced by prune_finetune.py (structurally pruned
Student + AutoEncoder, fp32 Teacher).

Unlike every other loader in this project, the pruned Student/AutoEncoder
are NOT loaded via load_state_dict into a freshly-constructed Student()/
AutoEncoder() -- pruning physically removes channels, so the layer shapes
no longer match those classes. prune_finetune.py saves them as full
pickled nn.Module objects instead (torch.save(model, ...), not
torch.save(model.state_dict(), ...)), so the architecture travels with
the weights. This loader just torch.load()s them directly.

Pruned checkpoints always train at the ORIGINAL 256px resolution and stay
in fp32 -- pruning changes channel counts, not spatial size or precision --
so this runs on GPU exactly like your original checkpoints, no CPU
constraint like the QAT loader has.

Expected checkpoint layout (prune_finetune.py's --out_dir)
--------------------------------------------------------------
<out_dir>/
    best_teacher.pth                     # NOT written by prune_finetune.py --
                                          # copy it in yourself, see below
    <category>_student_pruned.pt         # full pickled nn.Module
    <category>_autoencoder_pruned.pt     # full pickled nn.Module
    <category>_quantiles.npy             # mean/std/qa/qb + 'prune_ratio' key

prune_finetune.py does not copy best_teacher.pth into --out_dir
automatically. Copy it yourself, same as every other checkpoint in this
project:
    cp <ckpt_dir>/best_teacher.pth <pruned_out_dir>/best_teacher.pth

Usage
-----
    from load_efficientad_pruned import load_efficientad_model

    models = load_efficientad_model(
        ckpt_dir   = "ckpt_pruned_030",
        category   = "chewinggum",
        model_size = "S",
        device     = "cuda",
    )
"""

import os
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import Teacher


def load_efficientad_model(
    ckpt_dir:   str,
    category:   str  = "chewinggum",
    model_size: str  = "S",
    device:     str  = "cuda",
) -> dict:
    """
    Load and return a pruned EfficientAD model bundle.

    Returns
    -------
    dict with keys:
        "teacher"     : Teacher (nn.Module, fp32, eval, on device)
        "student"     : pruned Student (nn.Module, structurally smaller, eval, on device)
        "ae"          : pruned AutoEncoder (nn.Module, structurally smaller, eval, on device)
        "mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae" : torch.Tensor
        "prune_ratio" : float -- the --prune_ratio this checkpoint was pruned at

    Raises
    ------
    FileNotFoundError  if any required checkpoint file is missing.
    """
    teacher_path   = os.path.join(ckpt_dir, "best_teacher.pth")
    student_path   = os.path.join(ckpt_dir, f"{category}_student_pruned.pt")
    ae_path        = os.path.join(ckpt_dir, f"{category}_autoencoder_pruned.pt")
    quantiles_path = os.path.join(ckpt_dir, f"{category}_quantiles.npy")

    required = {
        "teacher":   teacher_path,
        "student":   student_path,
        "ae":        ae_path,
        "quantiles": quantiles_path,
    }
    missing = [name for name, path in required.items() if not os.path.isfile(path)]
    if missing:
        extra = ""
        if "teacher" in missing:
            extra = (f"\nprune_finetune.py does not copy best_teacher.pth into --out_dir "
                     f"automatically -- copy it in yourself:\n"
                     f"  cp <ckpt_dir>/best_teacher.pth {ckpt_dir}/best_teacher.pth")
        raise FileNotFoundError(
            f"Missing checkpoint files for '{category}' in '{ckpt_dir}':\n"
            + "\n".join(f"  [{k}] {required[k]}" for k in missing) + extra
        )

    print(f"[EfficientAD Loader] Loading quantiles : {quantiles_path}")
    quantiles    = np.load(quantiles_path, allow_pickle=True).item()
    prune_ratio  = quantiles.get("prune_ratio")
    print(f"[EfficientAD Loader] Checkpoint prune_ratio = {prune_ratio}")

    print(f"[EfficientAD Loader] Building PDN-{model_size} teacher (full width, frozen) ...")
    teacher = Teacher(model_size)
    teacher.load_state_dict(
        torch.load(teacher_path, map_location=device, weights_only=False))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[EfficientAD Loader] Loading pruned student   (full module): {student_path}")
    student = torch.load(student_path, map_location=device, weights_only=False)
    student.eval().to(device)

    print(f"[EfficientAD Loader] Loading pruned autoencoder (full module): {ae_path}")
    ae = torch.load(ae_path, map_location=device, weights_only=False)
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

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    print(f"[EfficientAD Loader] Pruned student params : {count_params(student):,}")
    print(f"[EfficientAD Loader] Pruned AE params      : {count_params(ae):,}")
    print(f"[EfficientAD Loader] \u2713 Model loaded successfully "
          f"(category='{category}', size='{model_size}', device='{device}', "
          f"prune_ratio={prune_ratio})")

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
        "prune_ratio": prune_ratio,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and verify a pruned EfficientAD model bundle.")
    parser.add_argument("--ckpt_dir",   required=True)
    parser.add_argument("--category",   default="chewinggum")
    parser.add_argument("--model_size", default="S", choices=["S", "M"])
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    models = load_efficientad_model(
        ckpt_dir=args.ckpt_dir, category=args.category,
        model_size=args.model_size, device=args.device)

    import torch.nn.functional as F
    dummy = torch.randn(1, 3, 256, 256, device=args.device)
    with torch.no_grad():
        t_out = models["teacher"](dummy)
        s_out = models["student"](dummy)
        a_out = models["ae"](dummy)
    print(f"\nTeacher output: {tuple(t_out.shape)}")
    print(f"Student output: {tuple(s_out.shape)}")
    print(f"AE output     : {tuple(a_out.shape)}")
    print("Verification passed.")
