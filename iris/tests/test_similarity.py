"""Tests for Iris similarity — encoding produces meaningful similarity scores."""

import numpy as np
import pytest

from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity
from glyphh.encoder.base import Encoder

from encoder import ENCODER_CONFIG


class TestSimilarityProperties:
    """Validate that the Iris encoder preserves meaningful similarity."""

    def setup_method(self):
        self.encoder = Encoder(ENCODER_CONFIG)

    def _encode(self, features):
        return self.encoder.encode(Concept(name="test", attributes=features))

    def test_identical_images_similarity_one(self, sample_features):
        g1 = self._encode(sample_features)
        g2 = self._encode(sample_features)
        sim = cosine_similarity(g1.global_cortex.data, g2.global_cortex.data)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_perturbed_continuous_stays_similar(self, sample_features):
        """Small perturbation to continuous features → still similar."""
        rng = np.random.RandomState(0)
        perturbed = dict(sample_features)
        base = np.array(perturbed["face_embedding"])
        perturbed["face_embedding"] = (base + rng.randn(512) * 0.05).tolist()

        g1 = self._encode(sample_features)
        g2 = self._encode(perturbed)
        sim = cosine_similarity(g1.global_cortex.data, g2.global_cortex.data)
        assert sim > 0.5, f"Small perturbation should stay similar: {sim}"

    def test_same_lighting_different_face(self, sample_features, different_features):
        """Same lighting but different face → per-layer similarity differs."""
        # Copy lighting from sample to different
        mixed = dict(different_features)
        mixed["direction"] = sample_features["direction"]
        mixed["quality"] = sample_features["quality"]
        mixed["contrast"] = sample_features["contrast"]

        g_base = self._encode(sample_features)
        g_mixed = self._encode(mixed)

        # Lighting layer should be very similar
        if "lighting" in g_base.layers and "lighting" in g_mixed.layers:
            light_sim = cosine_similarity(
                g_base.layers["lighting"].cortex.data,
                g_mixed.layers["lighting"].cortex.data,
            )
            assert light_sim > 0.5, f"Same lighting should match: {light_sim}"

        # Identity layer should be different
        if "identity" in g_base.layers and "identity" in g_mixed.layers:
            id_sim = cosine_similarity(
                g_base.layers["identity"].cortex.data,
                g_mixed.layers["identity"].cortex.data,
            )
            assert id_sim < light_sim, "Lighting should be more similar than identity"

    def test_categorical_role_exact_match(self, sample_features):
        """Same categorical values → high similarity on that layer."""
        g1 = self._encode(sample_features)

        # Change only continuous features, keep categoricals
        modified = dict(sample_features)
        rng = np.random.RandomState(77)
        modified["face_embedding"] = rng.randn(512).tolist()
        modified["body_pose"] = rng.randn(99).tolist()
        g2 = self._encode(modified)

        # Lighting layer (all categorical) should be identical
        if "lighting" in g1.layers and "lighting" in g2.layers:
            sim = cosine_similarity(
                g1.layers["lighting"].cortex.data,
                g2.layers["lighting"].cortex.data,
            )
            assert sim == pytest.approx(1.0, abs=0.01), f"Same categoricals should be 1.0: {sim}"

    def test_bow_shared_words_produce_similarity(self, sample_features):
        """Shared words in BoW roles produce similarity signal."""
        g1 = self._encode(sample_features)

        modified = dict(sample_features)
        # Share some words: "blue" is shared, add "red"
        modified["color_palette"] = "blue red"
        g2 = self._encode(modified)

        # Appearance layer should have some similarity (shared "blue")
        if "appearance" in g1.layers and "appearance" in g2.layers:
            sim = cosine_similarity(
                g1.layers["appearance"].cortex.data,
                g2.layers["appearance"].cortex.data,
            )
            assert sim > 0.0, "Shared BoW words should produce positive similarity"
