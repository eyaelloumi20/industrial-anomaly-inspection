# New dataset study — research protocol

**Status:** Not started. This is a plan, not an experimental result.

## Objective

Adapt the original YOLOv8 → ByteTrack → EfficientAD inspection pipeline to a new dataset, measure its baseline performance, and systematically investigate improvements in anomaly detection, robustness and inference speed.

## Research workflow

1. **Dataset audit.** Record source, permissions, number of samples/frames, normal/anomaly prevalence, object types, annotations, frame rate, image resolution, acquisition conditions and potential domain shift.
2. **Leakage-safe split.** Separate by physical piece, original video, acquisition session or production batch (as applicable), rather than randomly splitting correlated frames. Reserve final test data.
3. **Reference baseline.** Evaluate the original architecture with appropriate new-data preprocessing and calibration; distinguish zero-shot transfer from new-data training.
4. **Detection/tracking evaluation.** Record detector precision/recall/mAP and identity continuity separately from anomaly classification.
5. **Anomaly evaluation.** Report anomaly precision, recall, F1, AUROC where valid, and a confusion matrix at object/track level; document threshold selection on validation/calibration data only.
6. **Ablation studies.** Compare resolution, crop preparation, temporal EMA, clipping policy, model variants, and backgrounds under matched evaluation protocols.
7. **Runtime analysis.** Report hardware, frame size, object count, detection+tracking time, anomaly scoring time, end-to-end compute FPS and rendering overhead separately.
8. **Final held-out assessment.** Freeze the selected configuration and evaluate exactly once on the reserved test split; record failure cases and limitations.

## Study records

Use [experiment log template](../experiments/EXPERIMENT_TEMPLATE.md) for each configuration. Keep all private assets and large artifacts out of Git.

## Research safeguards

- Do not reuse calibration thresholds from the old dataset without validation.
- Do not conflate image, crop, frame, and track-level metrics.
- Do not report improvements unless compared with a clearly specified baseline on equivalent test data.
- Preserve source script versions and random seeds for reproducibility.
