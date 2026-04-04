"""
Iris encoder — ENCODER_CONFIG, encode_query, entry_to_record.

This is the standard model interface consumed by the Glyphh runtime.
Defines the 6-layer encoder config and the two encode_query modes:
1. Image path → full extraction pipeline → concept dict
2. Text query → intent extraction → concept dict
"""

import hashlib
import os
from typing import Dict, Any, Optional

import numpy as np

# Model-local imports (must be top-level — runtime removes model dir
# from sys.path after module load, so lazy imports would fail)
from intent import extract_intent
from extractors.base import ExtractorRegistry
from extractors.face import FaceExtractor
from extractors.pose import PoseExtractor
from extractors.depth import DepthExtractor
from extractors.ocr import OCRExtractor
from extractors.objects import ObjectExtractor
from extractors.color import ColorExtractor
from extractors.lighting import LightingExtractor
from extractors.composition import CompositionExtractor

# SDK imports
from glyphh.core.config import (
    EncoderConfig,
    Layer,
    Segment,
    Role,
    ContinuousConfig,
)

# ============================================================================
# ENCODER_CONFIG — 6 layers, weighted
# ============================================================================

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        # --- Identity (face) ---
        Layer(
            name="identity",
            similarity_weight=0.25,
            segments=[
                Segment(
                    name="face",
                    similarity_weight=1.0,
                    roles=[
                        Role(
                            name="face_embedding",
                            similarity_weight=1.0,
                            continuous_config=ContinuousConfig(
                                source_dim=512,
                                projection_seed=100,
                            ),
                        ),
                        Role(
                            name="expression",
                            similarity_weight=0.6,
                        ),
                        Role(
                            name="age_group",
                            similarity_weight=0.4,
                        ),
                    ],
                ),
            ],
        ),
        # --- Pose ---
        Layer(
            name="pose",
            similarity_weight=0.20,
            segments=[
                Segment(
                    name="body",
                    similarity_weight=1.0,
                    roles=[
                        Role(
                            name="body_pose",
                            similarity_weight=1.0,
                            continuous_config=ContinuousConfig(
                                source_dim=99,
                                projection_seed=200,
                            ),
                        ),
                        Role(
                            name="head_pose",
                            similarity_weight=0.6,
                            continuous_config=ContinuousConfig(
                                source_dim=3,
                                projection_seed=201,
                            ),
                        ),
                    ],
                ),
            ],
        ),
        # --- Appearance ---
        Layer(
            name="appearance",
            similarity_weight=0.20,
            segments=[
                Segment(
                    name="visual",
                    similarity_weight=1.0,
                    roles=[
                        Role(
                            name="color_palette",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="clothing",
                            similarity_weight=0.6,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        # --- Scene ---
        Layer(
            name="scene",
            similarity_weight=0.15,
            segments=[
                Segment(
                    name="environment",
                    similarity_weight=1.0,
                    roles=[
                        Role(
                            name="depth_map",
                            similarity_weight=0.8,
                            continuous_config=ContinuousConfig(
                                source_dim=64,
                                projection_seed=300,
                            ),
                        ),
                        Role(
                            name="composition",
                            similarity_weight=0.7,
                        ),
                        Role(
                            name="scene_category",
                            similarity_weight=0.6,
                        ),
                    ],
                ),
            ],
        ),
        # --- Lighting ---
        Layer(
            name="lighting",
            similarity_weight=0.10,
            segments=[
                Segment(
                    name="light",
                    similarity_weight=1.0,
                    roles=[
                        Role(name="direction", similarity_weight=0.8),
                        Role(name="quality", similarity_weight=0.7),
                        Role(name="contrast", similarity_weight=0.6),
                    ],
                ),
            ],
        ),
        # --- Text / OCR ---
        Layer(
            name="text",
            similarity_weight=0.10,
            segments=[
                Segment(
                    name="ocr",
                    similarity_weight=1.0,
                    roles=[
                        Role(
                            name="text_content",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                        Role(name="text_role", similarity_weight=0.5),
                        Role(name="text_position", similarity_weight=0.4),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================================
# encode_query — Dual mode: image path or text query
# ============================================================================

def encode_query(query: str) -> Dict[str, Any]:
    """
    Convert a query (image path or text) to a concept dict for similarity search.

    Args:
        query: Either a path to an image file, or a natural language query

    Returns:
        Dict with 'name' and 'attributes' keys, ready for Encoder.encode()
    """
    if _is_image_path(query):
        return _encode_image_query(query)
    else:
        return _encode_text_query(query)


def _is_image_path(query: str) -> bool:
    """Check if query looks like an image file path."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    _, ext = os.path.splitext(query.lower())
    return ext in image_extensions and os.path.isfile(query)


def _encode_image_query(image_path: str) -> Dict[str, Any]:
    """Extract features from an image and build concept dict."""
    features = _extract_all_features(image_path)
    stable_id = hashlib.md5(image_path.encode()).hexdigest()[:8]

    return {
        "name": f"query_{stable_id}",
        "attributes": features,
    }


def _encode_text_query(query: str) -> Dict[str, Any]:
    """Parse a text query about images and build concept dict."""
    intent = extract_intent(query)
    stable_id = hashlib.md5(query.encode()).hexdigest()[:8]

    # Map intent to concept attributes
    attributes = {
        "color_palette": intent["keywords"],
        "clothing": "",
        "text_content": "",
        "text_role": "none",
        "text_position": "none",
        "composition": "none",
        "scene_category": "none",
        "direction": "none",
        "quality": "none",
        "contrast": "none",
        "expression": "none",
        "age_group": "none",
    }

    # Populate from target and keywords
    target = intent["target"]
    kw = intent["keywords"]

    if target == "lighting":
        # Parse lighting keywords
        for word in kw.split():
            if word in ("soft", "hard", "natural", "artificial"):
                attributes["quality"] = word
            elif word in ("high", "low", "medium"):
                attributes["contrast"] = word
            elif word in ("front", "side_left", "side_right", "back", "top",
                          "rim", "ambient"):
                attributes["direction"] = word
    elif target == "face":
        attributes["expression"] = kw if kw else "none"
    elif target == "color":
        attributes["color_palette"] = kw
    elif target == "composition":
        for word in kw.split():
            if word in ("center", "rule_of_thirds", "symmetric", "diagonal",
                        "close_up", "panoramic", "isolated"):
                attributes["composition"] = word
                break
    elif target == "scene":
        for word in kw.split():
            if word in ("indoor", "outdoor", "studio", "urban", "nature",
                        "underwater", "aerial", "abstract"):
                attributes["scene_category"] = word
                break
    elif target == "text":
        attributes["text_content"] = kw

    return {
        "name": f"query_{stable_id}",
        "attributes": attributes,
    }


def _extract_all_features(image_path: str) -> Dict[str, Any]:
    """Run all feature extractors on an image."""
    image = _load_image(image_path)
    features = {}

    # Run extractors with graceful degradation
    registry = _get_registry()
    all_features = registry.extract_all(image)

    # Flatten into concept attributes
    for extractor_name, extracted in all_features.items():
        features.update(extracted)

    # Ensure all required attributes exist
    _fill_defaults(features)

    return features


def _load_image(path: str) -> np.ndarray:
    """Load an image as RGB numpy array."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(path).convert("RGB")
    return np.array(img)


_registry_instance = None


def _get_registry():
    """Get or create the global extractor registry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ExtractorRegistry()
        _registry_instance.register("face", FaceExtractor())
        _registry_instance.register("pose", PoseExtractor())
        _registry_instance.register("depth", DepthExtractor())
        _registry_instance.register("ocr", OCRExtractor())
        _registry_instance.register("objects", ObjectExtractor())
        _registry_instance.register("color", ColorExtractor())
        _registry_instance.register("lighting", LightingExtractor())
        _registry_instance.register("composition", CompositionExtractor())
    return _registry_instance


def _fill_defaults(features: Dict[str, Any]):
    """Ensure all required concept attributes exist."""
    defaults = {
        "face_embedding": [0.0] * 512,
        "expression": "none",
        "age_group": "none",
        "body_pose": [0.0] * 99,
        "head_pose": [0.0, 0.0, 0.0],
        "color_palette": "",
        "clothing": "",
        "depth_map": [0.0] * 64,
        "composition": "none",
        "scene_category": "none",
        "direction": "none",
        "quality": "none",
        "contrast": "none",
        "text_content": "",
        "text_role": "none",
        "text_position": "none",
    }
    for key, default in defaults.items():
        if key not in features:
            features[key] = default


# ============================================================================
# entry_to_record — Convert JSONL exemplar to concept record
# ============================================================================

def entry_to_record(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a JSONL exemplar entry to a concept record for encoding.

    Expected entry format:
    {
        "image_id": "img_001",
        "features": {
            "face_embedding": [...],
            "expression": "happy",
            ...
        },
        "metadata": {...}
    }
    """
    features = entry.get("features", {})
    _fill_defaults(features)

    return {
        "concept_text": entry.get("image_id", entry.get("name", "unknown")),
        "attributes": features,
        "metadata": entry.get("metadata", {}),
    }
