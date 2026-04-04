"""Shared test fixtures for Iris model tests."""

import os
import sys

import numpy as np
import pytest

# Add SDK and model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "glyphh-runtime"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def sample_image():
    """Generate a synthetic test image (RGB, 480x640)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_features():
    """Pre-extracted feature dict for testing without CV dependencies."""
    rng = np.random.RandomState(42)
    return {
        "face_embedding": rng.randn(512).tolist(),
        "expression": "happy",
        "age_group": "thirties",
        "body_pose": rng.randn(99).tolist(),
        "head_pose": [0.1, -0.05, 0.02],
        "color_palette": "blue white gray",
        "clothing": "suit jacket tie",
        "depth_map": rng.randn(64).tolist(),
        "composition": "rule_of_thirds",
        "scene_category": "indoor",
        "direction": "side_left",
        "quality": "soft",
        "contrast": "medium",
        "text_content": "",
        "text_role": "none",
        "text_position": "none",
    }


@pytest.fixture
def similar_features(sample_features):
    """Features similar to sample_features (small perturbation)."""
    rng = np.random.RandomState(99)
    features = dict(sample_features)
    # Perturb continuous values slightly
    base_face = np.array(features["face_embedding"])
    features["face_embedding"] = (base_face + rng.randn(512) * 0.1).tolist()
    base_pose = np.array(features["body_pose"])
    features["body_pose"] = (base_pose + rng.randn(99) * 0.1).tolist()
    return features


@pytest.fixture
def different_features():
    """Features completely different from sample_features."""
    rng = np.random.RandomState(123)
    return {
        "face_embedding": rng.randn(512).tolist(),
        "expression": "angry",
        "age_group": "child",
        "body_pose": rng.randn(99).tolist(),
        "head_pose": [-0.5, 0.3, -0.1],
        "color_palette": "red orange yellow",
        "clothing": "t-shirt shorts",
        "depth_map": rng.randn(64).tolist(),
        "composition": "close_up",
        "scene_category": "outdoor",
        "direction": "back",
        "quality": "hard",
        "contrast": "high",
        "text_content": "STOP",
        "text_role": "title",
        "text_position": "center",
    }
