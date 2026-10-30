# Development setup and validation

This repository contains research source, **not** bundled pretrained/trained models or datasets. Runtime execution is not yet verified against a pinned environment.

## Set up a local Python environment

Use Python 3.11 as an initial verification target. Select a PyTorch + torchvision installation compatible with the machine's CPU/CUDA driver, following the official PyTorch installer. Then install other dependencies from `requirements.txt`. Exact compatible versions remain to be established by a successful full run.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

## Checks that do not require weights or videos

```bash
python -m compileall -q -f .
python tests/check_structure.py
```

The following commands also require successful installation of Python ML dependencies, and may fail if the research code needs further compatibility fixes:

```bash
python video_streaming_detector_simple_optimized_v2.py --help
python video_streaming_detector_fast_v2.py --help
```

## For a real inference test

Provide locally: input video, YOLO detector weights, compatible EfficientAD teacher/student/AE checkpoint bundle, and calibrated threshold metadata. Never publish private videos or proprietary weights without permission. Match the crop preprocessing and threshold configuration used during model calibration.

## Reproducibility caveat

Passing static checks does **not** establish that the model loads, tracks correctly, or reproduces reported FPS/quality metrics. Those require a GPU/CPU execution test using the actual model assets.
