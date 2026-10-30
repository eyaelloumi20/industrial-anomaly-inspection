# Migration and reproducibility status

## Provenance

This is a curated derivative of [the complete research archive](https://github.com/eyaelloumi20/industrial-anomaly-detection-yolo-efficientad). The original main branch was left unchanged.

## Migrated core

- `video_streaming_detector_simple_optimized_v2.py`: candidate primary demonstration (per-track EMA threshold logic)
- `video_streaming_detector_fast_v2.py`: alternative track-level retrospective inference
- `compare_pipeline_methods_batched.py`: shared crop, detector, and model loading helpers
- `utils/models.py`: core EfficientAD model definitions
- `utils/load_efficientad*.py`: loaders for baseline/adaptive/thin/pruned/QAT research variants

These are copied with original code and names to avoid breaking imports. They have **not** been end-to-end tested in the new repository. Other local dependencies, settings, and training/evaluation scripts are still under review.

## External assets required

- Input video or camera
- YOLOv8 detector weights trained for chewing-gum pieces
- Compatible EfficientAD student/autoencoder and teacher checkpoints
- Calibration thresholds and associated metadata for the selected model/configuration
- Compatible CUDA/PyTorch/Ultralytics installation for the target computer

Do not report a working one-command demo until the missing weights and calibration configuration are provided and an execution test succeeds. A standalone pretrained teacher checkpoint cannot reproduce trained anomaly decisions.

## Next verification steps

1. Review import graph and model-loader paths; resolve any missing local modules.
2. Review and remove original absolute workstation paths and confidential data references.
3. Set up a pinned, tested environment and run syntax/import checks.
4. Run representative video inference and compare output with original implementation.
5. Migrate only the training, calibration, evaluation, and benchmarking scripts needed to substantiate the report.
6. Add small synthetic smoke tests and vetted visual examples.
7. Verify code and model checkpoint licensing before adding a repository license or distributing weights.

## Example entry point

```bash
python video_streaming_detector_simple_optimized_v2.py --help
```

The command above is only a proposed import/CLI smoke check; its success is not yet verified.
