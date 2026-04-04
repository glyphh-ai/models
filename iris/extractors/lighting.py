"""Lighting feature extractor — direction, quality, and contrast from histogram analysis."""

import numpy as np
from typing import Any, Dict
from extractors.base import FeatureExtractor


class LightingExtractor(FeatureExtractor):
    """Analyze lighting direction, quality, and contrast from image histograms.

    No external dependencies — uses pure numpy histogram analysis.
    """

    def is_available(self) -> bool:
        return True  # Pure numpy, always available

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        gray = _to_grayscale(image)
        h, w = gray.shape

        # Contrast from histogram spread
        contrast = _analyze_contrast(gray)

        # Direction from quadrant brightness
        direction = _analyze_direction(gray)

        # Quality from gradient smoothness
        quality = _analyze_quality(gray)

        return {
            "direction": direction,
            "quality": quality,
            "contrast": contrast,
        }

    def defaults(self) -> Dict[str, Any]:
        return {
            "direction": "none",
            "quality": "none",
            "contrast": "none",
        }


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale."""
    if image.ndim == 2:
        return image.astype(np.float32)
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)


def _analyze_contrast(gray: np.ndarray) -> str:
    """Classify contrast as high/medium/low from std dev."""
    std = gray.std()
    if std > 70:
        return "high"
    elif std > 40:
        return "medium"
    else:
        return "low"


def _analyze_direction(gray: np.ndarray) -> str:
    """Estimate primary light direction from quadrant brightness."""
    h, w = gray.shape
    mid_h, mid_w = h // 2, w // 2

    quadrants = {
        "top": gray[:mid_h, :].mean(),
        "bottom": gray[mid_h:, :].mean(),
        "side_left": gray[:, :mid_w].mean(),
        "side_right": gray[:, mid_w:].mean(),
    }

    # Check for backlight (bright edges, dark center)
    center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
    edge = gray.mean()
    if center < edge * 0.7:
        return "back"

    # Check for rim light (bright edges)
    if center < edge * 0.85:
        return "rim"

    brightest = max(quadrants, key=quadrants.get)
    darkest = min(quadrants, key=quadrants.get)

    # Significant difference needed
    spread = quadrants[brightest] - quadrants[darkest]
    if spread < 15:
        return "ambient"
    if spread < 30:
        return "front"

    return brightest


def _analyze_quality(gray: np.ndarray) -> str:
    """Classify light quality from gradient smoothness."""
    # Compute gradient magnitude
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    # Use overlapping region
    min_h = min(gy.shape[0], gx.shape[0])
    min_w = min(gy.shape[1], gx.shape[1])
    grad_mag = np.sqrt(gy[:min_h, :min_w] ** 2 + gx[:min_h, :min_w] ** 2)

    avg_gradient = grad_mag.mean()
    max_gradient = np.percentile(grad_mag, 95)

    # Hard light = strong sharp gradients, soft = gentle gradients
    if max_gradient > 80 and avg_gradient > 20:
        return "hard"
    elif avg_gradient < 10:
        return "soft"
    else:
        return "natural"
