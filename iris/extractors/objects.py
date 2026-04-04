"""Object detection feature extractor — detected objects and categories."""

import numpy as np
from typing import Any, Dict
from extractors.base import FeatureExtractor


class ObjectExtractor(FeatureExtractor):
    """Detect objects using YOLO v8 and return categories + counts."""

    def __init__(self, confidence_threshold: float = 0.3):
        self._model = None
        self.confidence_threshold = confidence_threshold

    def is_available(self) -> bool:
        try:
            from ultralytics import YOLO  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        self._load_model()
        results = self._model(image, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return self.defaults()

        detections = results[0]
        objects = []
        categories = set()

        for box in detections.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            cls_id = int(box.cls[0])
            cls_name = detections.names[cls_id]
            categories.add(cls_name)
            objects.append({
                "category": cls_name,
                "confidence": conf,
                "bbox": box.xyxy[0].tolist(),
            })

        return {
            "detected_objects": objects,
            "categories": " ".join(sorted(categories)),
            "object_count": len(objects),
        }

    def defaults(self) -> Dict[str, Any]:
        return {
            "detected_objects": [],
            "categories": "",
            "object_count": 0,
        }
