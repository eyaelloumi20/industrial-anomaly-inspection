# Workspace setup

This is a clean research workspace. **The existing Python scripts are intentionally unchanged.** Exact package compatibility, model loading, and end-to-end video inference remain unverified.

## Clone and create environment

```bash
git clone https://github.com/eyaelloumi20/industrial-anomaly-inspection.git
cd industrial-anomaly-inspection
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

Choose a PyTorch/torchvision installation compatible with the available CPU or CUDA environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
python -m compileall -q .
```

The dependency list is provisional and may need additions or version constraints after testing.

## Suggested local workspace directories

The directories below are **local, untracked folders**. Do not commit raw datasets or private videos.

```text
data/
  raw/
  interim/
  processed/
  splits/
checkpoints/
  detector/
  anomaly/
runs/
outputs/
```

## Smoke checks

```bash
python video_streaming_detector_simple_optimized_v2.py --help
python video_streaming_detector_fast_v2.py --help
```

These checks require dependencies and only validate imports/argument parsing. Successful inference requires the corresponding YOLO detector, EfficientAD teacher/student/autoencoder bundle, appropriate thresholds, and representative videos.

## Beginning a new dataset

Record dataset origin, license, normal/defective class meanings, annotation format, physical object unit, split logic, resolution and background variation before training. Never tune thresholds on the final evaluation split.
