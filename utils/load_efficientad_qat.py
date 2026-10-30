"""
load_efficientad_qat.py
=========================
Loader for checkpoints produced by train_qat.py (int8 quantized Student +
AutoEncoder, fp32 Teacher).

Two hard constraints this loader enforces, both consequences of how
PyTorch's eager-mode quantization works:

  1. int8 kernels are CPU-only. If you pass device="cuda" this loader
     overrides it to "cpu" and prints a warning rather than silently doing
     the wrong thing or crashing deep in a kernel dispatch.
  2. torch.backends.quantized.engine must be set to the SAME backend the
     checkpoint was quantized with (fbgemm for x86, qnnpack for ARM) BEFORE
     any forward pass, or you'll get either a crash or silently wrong
     results. This is read automatically from quantiles.npy['backend'],
     which train_qat.py already saves.

Loading strategy: prefer the TorchScript .jit artifacts (architecture
travels with the weights, most robust) and fall back to reconstructing the
QuantStudentWrapper/QuantAEWrapper classes + prepare_qat/convert steps from
train_qat.py and loading the raw int8 state_dicts, in case scripting failed
when the checkpoint was produced.

The Teacher is loaded fp32, exactly as in every other loader in this
project -- it's never quantized (only needed during training/QAT
fine-tuning to produce targets, not part of the deployed inference graph).
IMAGE_SIZE is always 256 here: train_qat.py fine-tunes an existing fp32
checkpoint in place and never changes resolution.

Expected checkpoint layout (train_qat.py's --out_dir)
-------------------------------------------------------
<out_dir>/
    best_teacher.pth                        # NOT written by train_qat.py --
                                             # copy it in yourself, see below
    <category>_student_int8.jit             # preferred (TorchScript)
    <category>_autoencoder_int8.jit         # preferred (TorchScript)
    <category>_student_int8.pth             # fallback (raw state_dict)
    <category>_autoencoder_int8.pth         # fallback (raw state_dict)
    <category>_quantiles.npy                # mean/std/qa/qb + 'backend' key

train_qat.py does not copy best_teacher.pth into --out_dir automatically.
Copy it yourself, same as every other checkpoint in this project:
    cp <ckpt_dir>/best_teacher.pth <qat_out_dir>/best_teacher.pth

Usage
-----
    from load_efficientad_qat import load_efficientad_model

    models = load_efficientad_model(
        ckpt_dir   = "ckpt_qat",
        category   = "chewinggum",
        model_size = "S",
        device     = "cuda",   # will be overridden to "cpu" with a warning
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

from models_fixed import Teacher, Student, AutoEncoder


def load_efficientad_model(
    ckpt_dir:   str,
    category:   str  = "chewinggum",
    model_size: str  = "S",
    device:     str  = "cpu",
) -> dict:
    """
    Load and return an int8-aware EfficientAD model bundle.

    Returns
    -------
    dict with keys:
        "teacher"  : Teacher (nn.Module, fp32, eval, on CPU)
        "student"  : quantized Student (int8, eval, on CPU)
        "ae"       : quantized AutoEncoder (int8, eval, on CPU)
        "mean", "std", "qa_st", "qb_st", "qa_ae", "qb_ae" : torch.Tensor
        "backend"  : str -- the quantized engine this checkpoint expects
        "is_int8"  : True

    Raises
    ------
    FileNotFoundError  if best_teacher.pth or the quantiles file is missing.
    """
    if device != "cpu":
        print(f"[EfficientAD Loader] WARNING: int8 quantized models only run on CPU "
              f"(PyTorch eager-mode int8 kernels are CPU-only). Overriding "
              f"device='{device}' -> 'cpu'.")
    device = "cpu"

    # PyTorch's default CPU thread count is sometimes 1 depending on how the
    # process was launched (e.g. under certain conda/MKL environment configs),
    # which makes every conv op run on a single core -- a 494ms Teacher forward
    # pass is consistent with single-threaded execution on a multi-core desktop
    # CPU, not with genuine int8-vs-fp32 overhead. Explicitly use all available
    # cores; this is always safe and never slower than the default.
    n_threads = os.cpu_count() or 1
    torch.set_num_threads(n_threads)
    print(f"[EfficientAD Loader] CPU threads: {torch.get_num_threads()} "
          f"(of {n_threads} available cores)")

    teacher_path   = os.path.join(ckpt_dir, "best_teacher.pth")
    quantiles_path = os.path.join(ckpt_dir, f"{category}_quantiles.npy")

    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(
            f"Missing {teacher_path}. train_qat.py does not copy best_teacher.pth "
            f"into --out_dir automatically -- copy it in yourself:\n"
            f"  cp <ckpt_dir>/best_teacher.pth {ckpt_dir}/best_teacher.pth")
    if not os.path.isfile(quantiles_path):
        raise FileNotFoundError(f"Missing {quantiles_path}")

    print(f"[EfficientAD Loader] Loading quantiles : {quantiles_path}")
    quantiles = np.load(quantiles_path, allow_pickle=True).item()
    backend   = quantiles.get("backend", "fbgemm")
    print(f"[EfficientAD Loader] Quantized backend  : {backend}")
    torch.backends.quantized.engine = backend

    print(f"[EfficientAD Loader] Building PDN-{model_size} teacher (fp32, frozen) ...")
    teacher = Teacher(model_size)
    teacher.load_state_dict(
        torch.load(teacher_path, map_location=device, weights_only=False))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # ── student / ae: prefer TorchScript, fall back to state_dict + rebuild ──
    student_jit = os.path.join(ckpt_dir, f"{category}_student_int8.jit")
    ae_jit      = os.path.join(ckpt_dir, f"{category}_autoencoder_int8.jit")
    student_pth = os.path.join(ckpt_dir, f"{category}_student_int8.pth")
    ae_pth      = os.path.join(ckpt_dir, f"{category}_autoencoder_int8.pth")

    if os.path.isfile(student_jit) and os.path.isfile(ae_jit):
        print(f"[EfficientAD Loader] Loading student   (TorchScript): {student_jit}")
        student = torch.jit.load(student_jit, map_location=device)
        print(f"[EfficientAD Loader] Loading autoencoder (TorchScript): {ae_jit}")
        ae = torch.jit.load(ae_jit, map_location=device)
        student.eval()
        ae.eval()
    elif os.path.isfile(student_pth) and os.path.isfile(ae_pth):
        print(f"[EfficientAD Loader] TorchScript artifacts not found, "
              f"reconstructing from state_dict (train_qat.py's scripting step "
              f"must have failed when this checkpoint was produced) ...")
        # Must mirror train_qat.py's QuantStudentWrapper/QuantAEWrapper +
        # prepare_qat/convert exactly, or the state_dict keys/shapes won't line up.
        _HERE_ROOT = os.path.dirname(os.path.abspath(__file__))
        _PROJ_ROOT = os.path.dirname(_HERE_ROOT)
        if _PROJ_ROOT not in sys.path:
            sys.path.insert(0, _PROJ_ROOT)
        from train_qat import QuantStudentWrapper, QuantAEWrapper

        student_fp32 = Student(model_size)
        ae_fp32      = AutoEncoder()
        qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)

        student = QuantStudentWrapper(student_fp32)
        ae      = QuantAEWrapper(ae_fp32)
        student.qconfig = qconfig
        ae.qconfig      = qconfig
        student.train(); ae.train()
        torch.ao.quantization.prepare_qat(student, inplace=True)
        torch.ao.quantization.prepare_qat(ae, inplace=True)
        student.eval(); ae.eval()
        student = torch.ao.quantization.convert(student, inplace=False)
        ae      = torch.ao.quantization.convert(ae, inplace=False)

        print(f"[EfficientAD Loader] Loading student   : {student_pth}")
        student.load_state_dict(torch.load(student_pth, map_location=device, weights_only=False))
        print(f"[EfficientAD Loader] Loading autoencoder: {ae_pth}")
        ae.load_state_dict(torch.load(ae_pth, map_location=device, weights_only=False))
        student.eval()
        ae.eval()
    else:
        raise FileNotFoundError(
            f"No int8 student/autoencoder artifacts found in '{ckpt_dir}' "
            f"(looked for {category}_student_int8.jit / .pth). "
            f"Did train_qat.py finish and write to this --out_dir?")

    def _to_tensor(value):
        arr = np.array(value)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    mean  = _to_tensor(quantiles["mean"])
    std   = _to_tensor(quantiles["std"])
    qa_st = _to_tensor(quantiles["qa_st"])
    qb_st = _to_tensor(quantiles["qb_st"])
    qa_ae = _to_tensor(quantiles["qa_ae"])
    qb_ae = _to_tensor(quantiles["qb_ae"])

    print(f"[EfficientAD Loader] \u2713 int8 model loaded successfully "
          f"(category='{category}', size='{model_size}', device='cpu', backend='{backend}')")

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
        "backend": backend,
        "is_int8": True,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and verify an int8 quantized EfficientAD model bundle.")
    parser.add_argument("--ckpt_dir",   required=True)
    parser.add_argument("--category",   default="chewinggum")
    parser.add_argument("--model_size", default="S", choices=["S", "M"])
    args = parser.parse_args()

    models = load_efficientad_model(
        ckpt_dir=args.ckpt_dir, category=args.category, model_size=args.model_size)

    import torch.nn.functional as F
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        t_out = models["teacher"](dummy)
        s_out = models["student"](dummy)
        a_out = models["ae"](dummy)
    print(f"\nTeacher output: {tuple(t_out.shape)}")
    print(f"Student output: {tuple(s_out.shape)}")
    print(f"AE output     : {tuple(a_out.shape)}")
    print("Verification passed.")