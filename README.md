# Industrial Anomaly Inspection

**Video-based industrial quality inspection using YOLOv8, ByteTrack, and EfficientAD.**

This portfolio repository is a curated presentation of the engineering project **Automatic Anomaly Detection for Industrial Quality Inspection**, developed for inspection of moving chewing-gum pieces under different backgrounds.

## Pipeline

```text
Video frames
   → YOLOv8: locate pieces
   → ByteTrack: associate detections and maintain track IDs
   → Crop and preprocess each detected piece
   → EfficientAD: appearance anomaly score
   → Per-track EMA and calibrated threshold
   → Inspection annotation and evaluation
```

The optimization study evaluated configurations including **LowRes192**, **LowRes224**, Thin and pruned EfficientAD variants, with an emphasis on the speed–accuracy trade-off.

## Repository status

**Curated source migration in progress.** Code is selected from the [original research repository](https://github.com/eyaelloumi20/industrial-anomaly-detection-yolo-efficientad), which remains the full experiment archive. The scripts have workstation-specific assumptions and depend on external pretrained/trained weights, data, and calibration files. **An end-to-end reproduction has not yet been verified from this clean repository.**

See [Migration and verification](docs/MIGRATION.md) before attempting to run the pipeline.

## What the stages do

- **YOLOv8** detects and localizes chewing-gum pieces; detection does not classify defects.
- **ByteTrack** maintains piece identities across frames.
- **EfficientAD** assigns anomaly scores to extracted crops.
- **Temporal EMA** helps aggregate observations of the same tracked piece. Crops clipped at image boundaries must be handled consistently with the selected pipeline.
- **Calibration and evaluation** require the appropriate held-out labels, model configuration, preprocessing, and thresholds.

## Reproducibility

The research dataset, recorded videos, detector checkpoint, trained EfficientAD student/autoencoder, and calibrated threshold files are not shipped here. Do not assume that a standalone teacher checkpoint is sufficient for inference. See the documentation for the required assets.

## Attribution and license

Source originates from the research repository linked above. The license and redistribution rights for external code and model checkpoints must be verified before a license is added to this curated repository.
