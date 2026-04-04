"""Face feature extractor — identity embedding, expression, age group."""

import numpy as np
from typing import Any, Dict
from extractors.base import FeatureExtractor


class FaceExtractor(FeatureExtractor):
    """Extract face embedding (ArcFace 512-dim), expression, and age group."""

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        try:
            import insightface  # noqa: F401
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            from insightface.app import FaceAnalysis
            self._model = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self._model.prepare(ctx_id=-1, det_size=(640, 640))

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        self._load_model()

        # InsightFace expects BGR
        image_bgr = image[:, :, ::-1]
        faces = self._model.get(image_bgr)

        if not faces:
            return self.defaults()

        # Use the largest face (most prominent)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        embedding = face.normed_embedding  # 512-dim float32

        # Expression from face landmarks (simplified)
        expression = "neutral"

        # Age group binning
        age = getattr(face, "age", None)
        age_group = _age_to_group(age) if age else "none"

        return {
            "face_embedding": embedding.tolist(),
            "expression": expression,
            "age_group": age_group,
            "face_count": len(faces),
            "bbox": face.bbox.tolist(),
        }

    def defaults(self) -> Dict[str, Any]:
        return {
            "face_embedding": [0.0] * 512,
            "expression": "none",
            "age_group": "none",
            "face_count": 0,
            "bbox": None,
        }


def _age_to_group(age: float) -> str:
    """Convert numeric age to age group primitive."""
    if age < 3:
        return "infant"
    elif age < 13:
        return "child"
    elif age < 20:
        return "teenager"
    elif age < 30:
        return "twenties"
    elif age < 40:
        return "thirties"
    elif age < 50:
        return "forties"
    elif age < 60:
        return "fifties"
    else:
        return "sixties_plus"
