"""
detection.py
=============
Stage 1 of the two-stage inspection pipeline: object detection with YOLOv8.

YOLOv8 (via ultralytics) is the PRIMARY detection path.
A full-image bounding box fallback is available ONLY for:
  - offline development when ultralytics is not installed
  - passing --use_full_image explicitly in evaluate_pipeline.py

The detector should be fine-tuned on the VisA chewinggum category
using prepare_yolo_dataset.py + train_yolo.py before running evaluation.

Exported symbols
----------------
    Detector          – main class
    full_image_box()  – utility: returns the whole image as one BBox
    BBox              – type alias (x1, y1, x2, y2) in pixel coordinates
"""

from __future__ import annotations
import sys
import numpy as np
from PIL import Image
from typing import List, Tuple

# BoundingBox type: (x1, y1, x2, y2) pixel coordinates
BBox = Tuple[int, int, int, int]


# ── helpers ───────────────────────────────────────────────────────────────────

def full_image_box(pil_image: Image.Image) -> List[BBox]:
    """Return a single bounding box equal to the full image dimensions."""
    w, h = pil_image.size
    return [(0, 0, w, h)]


# ── detector ─────────────────────────────────────────────────────────────────

class Detector:
    """
    YOLOv8-based object detector for VisA chewing-gum localisation.

    Parameters
    ----------
    model_path      : Path to fine-tuned YOLOv8 .pt weights file.
                      Typically: ./yolo_runs/chewinggum_finetune/weights/best.pt
                      Pass None or "yolov8n.pt" to use the generic pretrained model.
    conf_threshold  : Minimum confidence score to keep a box (default 0.25).
    iou_threshold   : NMS IoU threshold (default 0.45).
    target_class    : YOLO class index to keep (default 0 = chewinggum).
                      Boxes from any other class are silently discarded.
                      Set to None to keep all classes (not recommended).
    use_full_image  : If True, skip YOLO entirely and return the whole image
                      as a single bounding box. Use only for debugging or
                      ablation -- NOT for the real pipeline run.
    device          : "cuda", "cuda:0", "cpu", etc.
    verbose         : Print per-image YOLO logs (default False).

    Notes
    -----
    Fine-tune the detector first:
        python prepare_yolo_dataset.py --visa_root /path/to/VisA
        python train_yolo.py --data ./yolo_dataset/chewinggum.yaml
    """

    def __init__(
        self,
        model_path:     str   = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold:  float = 0.45,
        target_class:   int   = 0,
        use_full_image: bool  = False,
        device:         str   = "cpu",
        verbose:        bool  = False,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.target_class   = target_class   # ← only keep boxes of this class
        self.use_full_image = use_full_image
        self.device         = device
        self.verbose        = verbose
        self._model         = None

        if not use_full_image:
            self._load_yolo(model_path)

    # ── private ───────────────────────────────────────────────────────────────

    def _load_yolo(self, model_path: str) -> None:
        """Import ultralytics and load YOLOv8 weights."""
        try:
            from ultralytics import YOLO
        except ImportError:
            print(
                "[Detector] ERROR: 'ultralytics' is not installed.\n"
                "           Install it with:  pip install ultralytics\n"
                "           Or run with --use_full_image for a no-YOLO fallback.",
                file=sys.stderr,
            )
            raise

        path = model_path if model_path else "yolov8n.pt"
        print(f"[Detector] Loading YOLOv8 weights from '{path}' ...")
        self._model = YOLO(path)
        self._model.to(self.device)

        n_params = sum(p.numel() for p in self._model.model.parameters())
        print(f"[Detector] YOLOv8 ready  "
              f"(params={n_params/1e6:.2f}M, device={self.device})")

    # ── public ────────────────────────────────────────────────────────────────

    def detect(self, pil_image: Image.Image) -> List[BBox]:
        """
        Run YOLOv8 detection on one PIL image.

        Returns
        -------
        List of (x1, y1, x2, y2) bounding boxes in pixel coordinates.
        If no box exceeds conf_threshold, falls back to the full-image box
        so the EfficientAD scoring stage always receives at least one crop.
        """
        if self.use_full_image or self._model is None:
            return full_image_box(pil_image)

        img_np  = np.array(pil_image.convert("RGB"))
        results = self._model.predict(
            source  = img_np,
            conf    = self.conf_threshold,
            iou     = self.iou_threshold,
            device  = self.device,
            verbose = self.verbose,
        )

        boxes: List[BBox] = []
        for r in results:
            if r.boxes is not None and len(r.boxes):
                xyxy    = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                for box, cls in zip(xyxy, classes):
                    # ── class filter: skip background / wrong-class boxes ──────
                    if self.target_class is not None and cls != self.target_class:
                        continue
                    x1, y1, x2, y2 = (int(box[0]), int(box[1]),
                                       int(box[2]), int(box[3]))
                    boxes.append((x1, y1, x2, y2))

        # No detections above threshold: fall back to full-image
        # (product is present but confidence is low due to domain shift)
        if not boxes:
            boxes = full_image_box(pil_image)

        return boxes

    def detect_with_scores(
        self, pil_image: Image.Image
    ) -> List[tuple]:
        """
        Like detect(), but also returns the confidence score per box.

        Returns
        -------
        List of (x1, y1, x2, y2, conf) tuples.
        """
        if self.use_full_image or self._model is None:
            w, h = pil_image.size
            return [(0, 0, w, h, 1.0)]

        img_np  = np.array(pil_image.convert("RGB"))
        results = self._model.predict(
            source  = img_np,
            conf    = self.conf_threshold,
            iou     = self.iou_threshold,
            device  = self.device,
            verbose = self.verbose,
        )

        boxes = []
        for r in results:
            if r.boxes is not None and len(r.boxes):
                xyxy    = r.boxes.xyxy.cpu().numpy()
                confs   = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls in zip(xyxy, confs, classes):
                    # ── class filter: skip background / wrong-class boxes ──────
                    if self.target_class is not None and cls != self.target_class:
                        continue
                    x1, y1, x2, y2 = (int(box[0]), int(box[1]),
                                       int(box[2]), int(box[3]))
                    boxes.append((x1, y1, x2, y2, float(conf)))

        if not boxes:
            w, h = pil_image.size
            boxes = [(0, 0, w, h, 1.0)]

        return boxes

    def crop_rois(
        self,
        pil_image:   Image.Image,
        boxes:       List[BBox],
        target_size: int = 256,
    ) -> List[Image.Image]:
        """
        Crop and resize each detected bounding box region.

        Parameters
        ----------
        pil_image   : source PIL image (full resolution)
        boxes       : list of (x1, y1, x2, y2) from detect()
        target_size : output size for EfficientAD input (default 256)

        Returns
        -------
        List of PIL images, one per box, each resized to target_size x target_size.
        """
        W, H = pil_image.size
        crops = []
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, min(x1, W));  x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H));  y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                crop = pil_image.convert("RGB")
            else:
                crop = pil_image.crop((x1, y1, x2, y2)).convert("RGB")
            crops.append(crop.resize((target_size, target_size), Image.BILINEAR))
        return crops

    @property
    def is_yolo_active(self) -> bool:
        """True if YOLOv8 is loaded and will be used for detection."""
        return (not self.use_full_image) and (self._model is not None)