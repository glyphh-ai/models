"""Composition feature extractor — framing, scene category, layout analysis."""

import numpy as np
from typing import Any, Dict
from extractors.base import FeatureExtractor


class CompositionExtractor(FeatureExtractor):
    """Analyze image composition and scene category.

    No external dependencies — uses numpy-based heuristics.
    """

    def is_available(self) -> bool:
        return True  # Pure numpy

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) if image.ndim == 3 else image.astype(float)
        h, w = gray.shape

        composition = _analyze_composition(gray, h, w)
        scene_category = _analyze_scene_category(image)

        return {
            "composition": composition,
            "scene_category": scene_category,
        }

    def defaults(self) -> Dict[str, Any]:
        return {
            "composition": "none",
            "scene_category": "none",
        }


def _analyze_composition(gray: np.ndarray, h: int, w: int) -> str:
    """Classify composition from subject placement and symmetry."""
    # Find salient region (brightest/highest-contrast area)
    # Simple approach: threshold to find "subject" pixels
    mean_val = gray.mean()
    std_val = gray.std()
    threshold = mean_val + std_val * 0.5

    mask = gray > threshold
    if mask.sum() < 0.01 * h * w:
        mask = gray > mean_val

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return "none"

    # Centroid of salient region (normalized 0-1)
    cy = ys.mean() / h
    cx = xs.mean() / w

    # Check symmetry (left-right)
    left_energy = gray[:, : w // 2].std()
    right_energy = gray[:, w // 2 :].std()
    symmetry_ratio = min(left_energy, right_energy) / max(left_energy, right_energy) if max(left_energy, right_energy) > 0 else 1.0

    # Check aspect ratio for panoramic
    aspect = w / h
    if aspect > 2.5:
        return "panoramic"

    # Check if subject fills most of frame (close-up)
    subject_area = mask.sum() / (h * w)
    if subject_area > 0.5:
        return "close_up"

    # Check symmetry
    if symmetry_ratio > 0.9 and abs(cx - 0.5) < 0.1:
        return "symmetric"

    # Check rule of thirds
    third_x = abs(cx - 1 / 3) < 0.1 or abs(cx - 2 / 3) < 0.1
    third_y = abs(cy - 1 / 3) < 0.1 or abs(cy - 2 / 3) < 0.1
    if third_x or third_y:
        return "rule_of_thirds"

    # Check center composition
    if abs(cx - 0.5) < 0.15 and abs(cy - 0.5) < 0.15:
        return "center"

    # Check diagonal
    diag_dist = abs(cx - cy)
    anti_diag_dist = abs(cx - (1 - cy))
    if diag_dist < 0.15 or anti_diag_dist < 0.15:
        return "diagonal"

    # Check isolated subject
    if subject_area < 0.1:
        return "isolated"

    return "center"


def _analyze_scene_category(image: np.ndarray) -> str:
    """Rough scene category from color distribution heuristics."""
    if image.ndim < 3:
        return "none"

    h, w = image.shape[:2]
    mean_color = image.mean(axis=(0, 1))

    # Check top third for sky (blue-ish)
    top_third = image[: h // 3, :, :]
    top_mean = top_third.mean(axis=(0, 1))
    top_blue_ratio = top_mean[2] / (top_mean.sum() + 1e-6)

    # Check bottom third for ground/greenery
    bottom_third = image[2 * h // 3 :, :, :]
    bottom_mean = bottom_third.mean(axis=(0, 1))
    bottom_green_ratio = bottom_mean[1] / (bottom_mean.sum() + 1e-6)

    # Overall brightness
    brightness = mean_color.mean()

    if top_blue_ratio > 0.4 and bottom_green_ratio > 0.38:
        return "nature"
    elif top_blue_ratio > 0.4:
        return "outdoor"
    elif brightness < 60:
        return "studio"
    elif brightness > 200:
        return "studio"

    # Default based on color variance
    color_std = image.std()
    if color_std < 30:
        return "studio"
    elif color_std > 60:
        return "outdoor"

    return "indoor"
