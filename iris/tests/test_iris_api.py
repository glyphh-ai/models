"""Tests for Iris high-level API — encode_features, search, swap, combine, diff."""

import numpy as np
import pytest

from iris import Iris, IrisGlyph


class TestIrisAPI:
    """Test the high-level Iris class."""

    def setup_method(self):
        self.iris = Iris()

    def test_encode_features(self, sample_features):
        glyph = self.iris.encode_features(sample_features, name="test")
        assert isinstance(glyph, IrisGlyph)
        assert glyph.glyph is not None
        assert "identity" in glyph.glyph.layers

    def test_decode(self, sample_features):
        glyph = self.iris.encode_features(sample_features)
        spec = self.iris.decode(glyph)
        assert spec["identity"]["expression"] == "happy"
        assert spec["lighting"]["quality"] == "soft"

    def test_to_json(self, sample_features):
        glyph = self.iris.encode_features(sample_features)
        spec = glyph.to_json()
        assert "identity" in spec
        assert "lighting" in spec

    def test_to_prompt(self, sample_features):
        glyph = self.iris.encode_features(sample_features)
        prompt = glyph.to_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_all_text(self, sample_features):
        features = dict(sample_features)
        features["text_content"] = "Hello World"
        glyph = self.iris.encode_features(features)
        assert glyph.all_text == "Hello World"


class TestIrisSearch:
    """Test search functionality."""

    def setup_method(self):
        self.iris = Iris()

    def test_search_returns_sorted(self, sample_features, similar_features, different_features):
        g1 = self.iris.encode_features(sample_features, name="base")
        g2 = self.iris.encode_features(similar_features, name="similar")
        g3 = self.iris.encode_features(different_features, name="different")

        self.iris.add_to_index(g2)
        self.iris.add_to_index(g3)

        results = self.iris.search(g1, top_k=10)
        assert len(results) == 2
        # Similar should rank higher
        assert results[0][1] > results[1][1]

    def test_search_empty_index(self, sample_features):
        glyph = self.iris.encode_features(sample_features)
        results = self.iris.search(glyph)
        assert results == []

    def test_search_by_role(self, sample_features, different_features):
        g1 = self.iris.encode_features(sample_features, name="base")

        # Same lighting as base
        same_light = dict(different_features)
        same_light["direction"] = sample_features["direction"]
        same_light["quality"] = sample_features["quality"]
        same_light["contrast"] = sample_features["contrast"]
        g2 = self.iris.encode_features(same_light, name="same_light")

        # Different lighting
        g3 = self.iris.encode_features(different_features, name="diff_light")

        self.iris.add_to_index(g2)
        self.iris.add_to_index(g3)

        results = self.iris.search_by_role(g1, layer_name="lighting", top_k=10)
        assert len(results) == 2
        # Same lighting should rank higher
        assert results[0][1] > results[1][1]


class TestIrisManipulation:
    """Test swap and combine operations."""

    def setup_method(self):
        self.iris = Iris()

    def test_swap(self, sample_features):
        glyph = self.iris.encode_features(sample_features)
        swapped = self.iris.swap(glyph, "expression", "angry")
        assert swapped.features["expression"] == "angry"
        assert swapped.features["direction"] == sample_features["direction"]  # Unchanged

    def test_combine(self, sample_features, different_features):
        g1 = self.iris.encode_features(sample_features, name="a")
        g2 = self.iris.encode_features(different_features, name="b")

        combined = self.iris.combine(
            g1, ["face_embedding", "expression", "age_group"],
            g2, ["direction", "quality", "contrast"],
        )
        assert combined.features["expression"] == sample_features["expression"]
        assert combined.features["direction"] == different_features["direction"]

    def test_diff(self, sample_features, different_features):
        g1 = self.iris.encode_features(sample_features)
        g2 = self.iris.encode_features(different_features)

        scores = self.iris.diff(g1, g2)
        assert isinstance(scores, dict)
        assert scores["expression"] == 0.0  # Different
        assert "face_embedding" in scores
