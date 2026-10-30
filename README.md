# Industrial Anomaly Inspection

**YOLOv8 · ByteTrack · EfficientAD | Video-based industrial quality inspection**

A research-oriented computer vision project for inspecting moving objects in industrial video. The original application concerns chewing-gum quality inspection; this repository also serves as the starting point for a **new-dataset experimental study**.

## System overview

```text
Video / camera frames
    ↓
YOLOv8 — object localization
    ↓
ByteTrack — persistent object identities
    ↓
Crop extraction and preprocessing
    ↓
EfficientAD — visual anomaly scores
    ↓
Temporal EMA / calibrated decision rules
    ↓
Annotated video and evaluation
```

The initial project investigated Baseline, LowRes224, LowRes192, Thin and pruned model configurations, including the accuracy–throughput trade-off. The future study will validate and improve the approach on a different dataset, without assuming the original thresholds or trained weights will transfer.

## Start here

- [Workspace setup](docs/SETUP.md)
- [Architecture](docs/ARCHITECTURE.md)
- [New dataset research plan](docs/NEW_DATASET_STUDY.md)
- [Source and reproduction status](docs/MIGRATION.md)

## Repository principles

1. **Preserve the original Python research scripts.** They are intentionally not refactored or rewritten at this stage.
2. Separate local datasets, model weights and generated outputs from Git-tracked source.
3. Track dataset versions, preprocessing, model settings, split definitions and experimental outcomes.
4. Keep calibration and final evaluation separate to prevent information leakage.
5. Distinguish original-PFE baselines from newly trained models and new-dataset results.

## Main source files

- `video_streaming_detector_simple_optimized_v2.py`: candidate primary inference pipeline.
- `video_streaming_detector_fast_v2.py`: alternative high-throughput track-level inference.
- `compare_pipeline_methods_batched.py`: shared inference/crop/model utilities.
- `utils/`: EfficientAD model definitions and variant loaders.
- `scripts/`: selected training, preprocessing, calibration, evaluation and benchmarking scripts.

The scripts retain their original names and semantics. Existing Python code has not been established as portable or executable end-to-end in a fresh environment.

## Local quick start

```bash
git clone https://github.com/eyaelloumi20/industrial-anomaly-inspection.git
cd industrial-anomaly-inspection
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m compileall -q .
```

The requirements file is a preliminary dependency list, not a locked environment. Select a compatible PyTorch and torchvision installation for your hardware. Real inference requires independently supplied YOLOv8 weights, matching EfficientAD checkpoint bundles, calibrated thresholds, and video inputs.

## Data and models

No private datasets, source videos, trained student/autoencoder checkpoints or calibrated threshold files are distributed here. Refer to [the original PFE research archive](https://github.com/eyaelloumi20/industrial-anomaly-detection-yolo-efficientad) for historical experimental code and methodological context.

## Status

**Research workspace prepared; new-dataset experiments not started.** No improved performance is claimed until results are measured on an independently evaluated dataset. License and checkpoint redistribution permissions still require review.
