"""Tests for Iris export functions — to_json, to_prompt, diff."""

import numpy as np
import pytest

from exports import to_json, to_prompt, diff, to_controlnet


class TestToJson:
    """Test structured JSON export."""

    def test_basic_structure(self, sample_features):
        spec = to_json(sample_features)
        assert "identity" in spec
        assert "pose" in spec
        assert "appearance" in spec
        assert "scene" in spec
        assert "lighting" in spec
        assert "text" in spec
        assert "objects" in spec

    def test_identity_fields(self, sample_features):
        spec = to_json(sample_features)
        assert spec["identity"]["expression"] == "happy"
        assert spec["identity"]["age_group"] == "thirties"

    def test_lighting_fields(self, sample_features):
        spec = to_json(sample_features)
        assert spec["lighting"]["direction"] == "side_left"
        assert spec["lighting"]["quality"] == "soft"
        assert spec["lighting"]["contrast"] == "medium"

    def test_color_palette_list(self, sample_features):
        spec = to_json(sample_features)
        assert spec["appearance"]["color_palette"] == ["blue", "white", "gray"]

    def test_empty_features(self):
        spec = to_json({})
        assert spec["identity"]["expression"] == "none"
        assert spec["lighting"]["direction"] == "none"


class TestToPrompt:
    """Test natural language prompt generation."""

    def test_basic_prompt(self, sample_features):
        prompt = to_prompt(sample_features)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_expression(self, sample_features):
        prompt = to_prompt(sample_features)
        assert "happy" in prompt or "thirties" in prompt

    def test_includes_lighting(self, sample_features):
        prompt = to_prompt(sample_features)
        assert "soft" in prompt or "lighting" in prompt

    def test_empty_features(self):
        prompt = to_prompt({})
        assert prompt == "unspecified image"


class TestDiff:
    """Test per-role similarity comparison."""

    def test_identical_features(self, sample_features):
        scores = diff(sample_features, sample_features)
        for role, score in scores.items():
            assert score == pytest.approx(1.0, abs=0.01), f"{role} should be 1.0, got {score}"

    def test_different_features(self, sample_features, different_features):
        scores = diff(sample_features, different_features)
        # Categorical roles should mostly be 0.0
        assert scores["expression"] == 0.0
        assert scores["scene_category"] == 0.0
        assert scores["direction"] == 0.0
        # Continuous roles should be low (random vectors)
        assert scores["face_embedding"] < 0.5

    def test_partial_match(self, sample_features, different_features):
        """Same lighting, different everything else."""
        modified = dict(different_features)
        modified["direction"] = sample_features["direction"]
        modified["quality"] = sample_features["quality"]
        modified["contrast"] = sample_features["contrast"]
        scores = diff(sample_features, modified)
        assert scores["direction"] == 1.0
        assert scores["quality"] == 1.0
        assert scores["contrast"] == 1.0
        assert scores["expression"] == 0.0  # Still different

    def test_bow_jaccard(self, sample_features):
        """BoW roles use Jaccard similarity."""
        modified = dict(sample_features)
        modified["color_palette"] = "blue red"  # Shares "blue"
        scores = diff(sample_features, modified)
        # Jaccard("blue white gray", "blue red") = 1/4 = 0.25
        assert 0.2 < scores["color_palette"] < 0.3


class TestToControlnet:
    """Test ControlNet signal export."""

    def test_pose_keypoints(self, sample_features):
        controls = to_controlnet(sample_features)
        assert "openpose" in controls
        assert len(controls["openpose"]) == 33  # 33 keypoints
        assert len(controls["openpose"][0]) == 3  # x, y, z

    def test_depth_map(self, sample_features):
        controls = to_controlnet(sample_features)
        assert "depth" in controls

    def test_no_pose_when_default(self):
        features = {"body_pose": [0.0] * 99, "depth_map": [0.0] * 64}
        controls = to_controlnet(features)
        assert "openpose" not in controls
