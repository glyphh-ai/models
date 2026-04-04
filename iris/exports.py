"""
Iris export functions — ControlNet, prompt generation, structured JSON, diff.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def to_json(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert extracted features to a structured JSON specification.

    Args:
        features: Dict of extracted features from encode_query or extractors

    Returns:
        Structured JSON spec organized by visual dimension
    """
    spec = {
        "identity": {
            "expression": features.get("expression", "none"),
            "age_group": features.get("age_group", "none"),
            "face_detected": features.get("face_embedding", [0.0] * 512) != [0.0] * 512,
        },
        "pose": {
            "body_detected": features.get("body_pose", [0.0] * 99) != [0.0] * 99,
        },
        "appearance": {
            "color_palette": features.get("color_palette", "").split() if features.get("color_palette") else [],
            "clothing": features.get("clothing", "").split() if features.get("clothing") else [],
        },
        "scene": {
            "composition": features.get("composition", "none"),
            "category": features.get("scene_category", "none"),
        },
        "lighting": {
            "direction": features.get("direction", "none"),
            "quality": features.get("quality", "none"),
            "contrast": features.get("contrast", "none"),
        },
        "text": {
            "content": features.get("text_content", ""),
            "role": features.get("text_role", "none"),
            "position": features.get("text_position", "none"),
            "regions": features.get("text_regions", []),
        },
        "objects": {
            "detected": features.get("detected_objects", []),
            "categories": features.get("categories", "").split() if features.get("categories") else [],
            "count": features.get("object_count", 0),
        },
    }
    return spec


def to_prompt(features: Dict[str, Any]) -> str:
    """
    Generate a natural language prompt fragment from extracted features.

    Useful for feeding to image generation models (Flux, SDXL, Midjourney).

    Args:
        features: Dict of extracted features

    Returns:
        Natural language description string
    """
    parts = []

    # Identity
    expression = features.get("expression", "none")
    age_group = features.get("age_group", "none")
    if age_group != "none" and expression != "none":
        parts.append(f"{age_group} person, {expression} expression")
    elif age_group != "none":
        parts.append(f"{age_group} person")
    elif expression != "none":
        parts.append(f"{expression} expression")

    # Clothing
    clothing = features.get("clothing", "")
    if clothing:
        parts.append(clothing)

    # Color palette
    palette = features.get("color_palette", "")
    if palette:
        parts.append(f"{palette} color palette")

    # Lighting
    direction = features.get("direction", "none")
    quality = features.get("quality", "none")
    contrast = features.get("contrast", "none")
    lighting_parts = []
    if quality != "none":
        lighting_parts.append(quality)
    if direction != "none":
        lighting_parts.append(f"{direction} lighting")
    if contrast != "none":
        lighting_parts.append(f"{contrast} contrast")
    if lighting_parts:
        parts.append(", ".join(lighting_parts))

    # Composition
    composition = features.get("composition", "none")
    if composition != "none":
        parts.append(f"{composition} composition")

    # Scene
    scene = features.get("scene_category", "none")
    if scene != "none":
        parts.append(f"{scene} setting")

    return ", ".join(parts) if parts else "unspecified image"


def to_controlnet(features: Dict[str, Any], image: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Generate ControlNet-ready signals from extracted features.

    Args:
        features: Dict of extracted features
        image: Optional original image for edge detection

    Returns:
        Dict with keys: 'openpose' (keypoints), 'depth' (map), 'canny' (edges)
    """
    controls = {}

    # Pose keypoints (for OpenPose ControlNet)
    body_pose = features.get("body_pose", [0.0] * 99)
    if body_pose != [0.0] * 99:
        # Reshape to 33 keypoints × 3 (x, y, z)
        keypoints = np.array(body_pose).reshape(33, 3)
        controls["openpose"] = keypoints.tolist()

    # Depth map (for Depth ControlNet)
    depth_map = features.get("depth_map")
    if depth_map and depth_map != [0.0] * 64:
        controls["depth"] = depth_map

    # Canny edges (from original image if provided)
    if image is not None:
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            controls["canny"] = edges.tolist()
        except ImportError:
            pass

    return controls


def diff(features_a: Dict[str, Any], features_b: Dict[str, Any]) -> Dict[str, float]:
    """
    Compare two feature sets by role, returning per-role similarity scores.

    Args:
        features_a: First image features
        features_b: Second image features

    Returns:
        Dict mapping role names to similarity scores (0.0 = different, 1.0 = same)
    """
    scores = {}

    # Categorical roles — exact match = 1.0, else 0.0
    categorical_roles = [
        "expression", "age_group", "composition", "scene_category",
        "direction", "quality", "contrast", "text_role", "text_position",
    ]
    for role in categorical_roles:
        a = features_a.get(role, "none")
        b = features_b.get(role, "none")
        scores[role] = 1.0 if a == b else 0.0

    # BoW roles — Jaccard similarity
    bow_roles = ["color_palette", "clothing", "text_content"]
    for role in bow_roles:
        a_words = set(str(features_a.get(role, "")).split())
        b_words = set(str(features_b.get(role, "")).split())
        a_words.discard("")
        b_words.discard("")
        if not a_words and not b_words:
            scores[role] = 1.0
        elif not a_words or not b_words:
            scores[role] = 0.0
        else:
            scores[role] = len(a_words & b_words) / len(a_words | b_words)

    # Continuous roles — cosine similarity on raw vectors
    continuous_roles = {
        "face_embedding": 512,
        "body_pose": 99,
        "head_pose": 3,
        "depth_map": 64,
    }
    for role, expected_dim in continuous_roles.items():
        a = np.array(features_a.get(role, [0.0] * expected_dim), dtype=np.float32)
        b = np.array(features_b.get(role, [0.0] * expected_dim), dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0 and norm_b > 0:
            scores[role] = float(np.dot(a, b) / (norm_a * norm_b))
        else:
            scores[role] = 0.0

    return scores
