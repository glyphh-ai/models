"""OCR feature extractor — text content, role, and position."""

import numpy as np
from typing import Any, Dict, List
from extractors.base import FeatureExtractor


class OCRExtractor(FeatureExtractor):
    """Extract text regions with content, bounding boxes, and structural role."""

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        try:
            import os
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            import paddle  # noqa: F401
            from paddleocr import PaddleOCR  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            import os
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR
            # New PaddleOCR API (>= 3.x) — no show_log, lang, or use_angle_cls
            try:
                self._model = PaddleOCR(lang="en")
            except TypeError:
                # Fallback for even newer API that may not accept lang
                self._model = PaddleOCR()

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        self._load_model()

        # New API uses predict() instead of ocr()
        try:
            result = self._model.predict(image)
        except AttributeError:
            # Legacy API fallback
            result = self._model.ocr(image, cls=True)
            return self._parse_legacy_result(result, image)

        return self._parse_predict_result(result, image)

    def _parse_predict_result(self, result, image: np.ndarray) -> Dict[str, Any]:
        """Parse result from new PaddleOCR predict() API."""
        if not result:
            return self.defaults()

        regions = []
        all_text = []
        h, w = image.shape[:2]

        for page in result:
            if not hasattr(page, "rec_texts") or not page.rec_texts:
                continue
            for i, text in enumerate(page.rec_texts):
                score = page.rec_scores[i] if i < len(page.rec_scores) else 0.0
                if score < 0.5:
                    continue

                # Get bounding box
                if hasattr(page, "dt_polys") and i < len(page.dt_polys):
                    bbox = page.dt_polys[i]
                    cx = float(np.mean(bbox[:, 0])) / w
                    cy = float(np.mean(bbox[:, 1])) / h
                    text_h = float(np.max(bbox[:, 1]) - np.min(bbox[:, 1])) / h
                    bbox_list = [int(np.min(bbox[:, 0])), int(np.min(bbox[:, 1])),
                                 int(np.max(bbox[:, 0])), int(np.max(bbox[:, 1]))]
                else:
                    cx, cy, text_h = 0.5, 0.5, 0.02
                    bbox_list = [0, 0, 0, 0]

                role = _infer_text_role(text, text_h, cy)
                position = _infer_text_position(cx, cy)

                regions.append({
                    "text": text,
                    "confidence": float(score),
                    "role": role,
                    "position": position,
                    "bbox": bbox_list,
                })
                all_text.append(text)

        if not regions:
            return self.defaults()

        primary_role = regions[0]["role"]
        primary_position = regions[0]["position"]

        return {
            "text_content": " ".join(all_text),
            "text_role": primary_role,
            "text_position": primary_position,
            "text_regions": regions,
        }

    def _parse_legacy_result(self, result, image: np.ndarray) -> Dict[str, Any]:
        """Parse result from legacy PaddleOCR ocr() API."""
        if not result or not result[0]:
            return self.defaults()

        regions = []
        all_text = []
        h, w = image.shape[:2]

        for line in result[0]:
            bbox, (text, conf) = line[0], line[1]
            if conf < 0.5:
                continue

            cx = sum(p[0] for p in bbox) / 4 / w
            cy = sum(p[1] for p in bbox) / 4 / h
            text_h = (max(p[1] for p in bbox) - min(p[1] for p in bbox)) / h
            role = _infer_text_role(text, text_h, cy)
            position = _infer_text_position(cx, cy)

            regions.append({
                "text": text,
                "confidence": float(conf),
                "role": role,
                "position": position,
                "bbox": [int(bbox[0][0]), int(bbox[0][1]),
                         int(bbox[2][0]), int(bbox[2][1])],
            })
            all_text.append(text)

        if not regions:
            return self.defaults()

        primary_role = regions[0]["role"]
        primary_position = regions[0]["position"]

        return {
            "text_content": " ".join(all_text),
            "text_role": primary_role,
            "text_position": primary_position,
            "text_regions": regions,
        }

    def defaults(self) -> Dict[str, Any]:
        return {
            "text_content": "",
            "text_role": "none",
            "text_position": "none",
            "text_regions": [],
        }


def _infer_text_role(text: str, relative_height: float, cy: float) -> str:
    """Infer text role from size and content."""
    if relative_height > 0.08:
        return "title"
    elif relative_height > 0.04:
        return "heading"
    elif len(text) < 15 and relative_height < 0.02:
        return "label"
    elif relative_height < 0.015:
        if cy > 0.9:
            return "watermark"
        return "caption"
    return "body"


def _infer_text_position(cx: float, cy: float) -> str:
    """Infer text position from center coordinates (normalized 0-1)."""
    if cy < 0.2:
        return "top"
    elif cy > 0.8:
        return "bottom"
    elif cx < 0.2:
        return "left"
    elif cx > 0.8:
        return "right"
    return "center"
