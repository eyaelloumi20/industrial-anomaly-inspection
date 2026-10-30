# System architecture

```text
Input video frames
    ↓
YOLOv8 — detect chewing-gum bounding boxes
    ↓
ByteTrack — maintain piece IDs across frames
    ↓
Crop extraction and preprocessing
    ↓
EfficientAD — anomaly scores for each crop
    ↓
Per-track EMA (reference configuration α = 0.3)
    ↓
Calibrated threshold — visual normal/anomaly labels
    ↓
Annotated output and track-level evaluation
```

YOLOv8 localizes but does not classify anomalies in this system. ByteTrack supports temporal association. EfficientAD performs the appearance-based anomaly estimation. The simple optimized and fast-v2 scripts use different final decision semantics; verify the exact script and threshold configuration against the reported research results.
