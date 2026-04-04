"""Tests for Iris encoding — ENCODER_CONFIG, encode_query, entry_to_record."""

import numpy as np
import pytest

from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity
from glyphh.encoder.base import Encoder

from encoder import ENCODER_CONFIG, encode_query, entry_to_record, _fill_defaults


class TestEncoderConfig:
    """Validate the Iris ENCODER_CONFIG structure."""

    def test_config_has_six_layers(self):
        assert len(ENCODER_CONFIG.layers) == 6

    def test_layer_names(self):
        names = [l.name for l in ENCODER_CONFIG.layers]
        assert names == ["identity", "pose", "appearance", "scene", "lighting", "text"]

    def test_weights_sum_to_one(self):
        total = sum(l.similarity_weight for l in ENCODER_CONFIG.layers)
        assert abs(total - 1.0) < 0.01

    def test_continuous_roles_exist(self):
        """Verify continuous_config is set on the right roles."""
        identity_layer = ENCODER_CONFIG.layers[0]
        face_segment = identity_layer.segments[0]
        face_emb_role = face_segment.roles[0]
        assert face_emb_role.name == "face_embedding"
        assert face_emb_role.continuous_config is not None
        assert face_emb_role.continuous_config.source_dim == 512

    def test_bow_roles_exist(self):
        """Verify bag_of_words encoding on the right roles."""
        appearance_layer = ENCODER_CONFIG.layers[2]
        visual_segment = appearance_layer.segments[0]
        palette_role = visual_segment.roles[0]
        assert palette_role.name == "color_palette"
        assert palette_role.text_encoding == "bag_of_words"

    def test_encoder_creates_successfully(self):
        encoder = Encoder(ENCODER_CONFIG)
        assert encoder.dimension == 2000


class TestEncodeWithFeatures:
    """Test encoding pre-extracted features through the full pipeline."""

    def setup_method(self):
        self.encoder = Encoder(ENCODER_CONFIG)

    def test_encode_full_features(self, sample_features):
        concept = Concept(name="test", attributes=sample_features)
        glyph = self.encoder.encode(concept)

        assert glyph is not None
        assert "identity" in glyph.layers
        assert "pose" in glyph.layers
        assert "appearance" in glyph.layers
        assert "scene" in glyph.layers
        assert "lighting" in glyph.layers
        assert "text" in glyph.layers

    def test_similar_features_produce_similar_glyphs(self, sample_features, similar_features):
        g1 = self.encoder.encode(Concept(name="a", attributes=sample_features))
        g2 = self.encoder.encode(Concept(name="b", attributes=similar_features))

        sim = cosine_similarity(g1.global_cortex.data, g2.global_cortex.data)
        assert sim > 0.3, f"Similar features should produce similar glyphs, got {sim}"

    def test_different_features_produce_different_glyphs(self, sample_features, different_features):
        g1 = self.encoder.encode(Concept(name="a", attributes=sample_features))
        g2 = self.encoder.encode(Concept(name="b", attributes=different_features))

        sim = cosine_similarity(g1.global_cortex.data, g2.global_cortex.data)
        assert sim < 0.5, f"Different features should produce dissimilar glyphs, got {sim}"

    def test_similar_more_similar_than_different(self, sample_features, similar_features, different_features):
        g_base = self.encoder.encode(Concept(name="base", attributes=sample_features))
        g_sim = self.encoder.encode(Concept(name="sim", attributes=similar_features))
        g_diff = self.encoder.encode(Concept(name="diff", attributes=different_features))

        sim_close = cosine_similarity(g_base.global_cortex.data, g_sim.global_cortex.data)
        sim_far = cosine_similarity(g_base.global_cortex.data, g_diff.global_cortex.data)

        assert sim_close > sim_far, (
            f"Similar should be more similar: close={sim_close:.3f}, far={sim_far:.3f}"
        )

    def test_determinism(self, sample_features):
        g1 = self.encoder.encode(Concept(name="x", attributes=sample_features))
        g2 = self.encoder.encode(Concept(name="x", attributes=sample_features))

        np.testing.assert_array_equal(g1.global_cortex.data, g2.global_cortex.data)

    def test_layer_level_similarity(self, sample_features, similar_features, different_features):
        """Per-layer similarity should be meaningful."""
        g_base = self.encoder.encode(Concept(name="base", attributes=sample_features))

        # Same lighting, different everything else
        lighting_match = dict(different_features)
        lighting_match["direction"] = sample_features["direction"]
        lighting_match["quality"] = sample_features["quality"]
        lighting_match["contrast"] = sample_features["contrast"]
        g_light = self.encoder.encode(Concept(name="light", attributes=lighting_match))

        # Lighting layer should be similar
        if "lighting" in g_base.layers and "lighting" in g_light.layers:
            light_sim = cosine_similarity(
                g_base.layers["lighting"].cortex.data,
                g_light.layers["lighting"].cortex.data,
            )
            assert light_sim > 0.5, f"Same lighting should match: {light_sim}"


class TestEncodeQuery:
    """Test the dual-mode encode_query function."""

    def test_text_query_returns_dict(self):
        result = encode_query("find photos with soft lighting")
        assert "name" in result
        assert "attributes" in result
        assert isinstance(result["attributes"], dict)

    def test_text_query_lighting(self):
        result = encode_query("find images with soft lighting")
        attrs = result["attributes"]
        assert attrs["quality"] == "soft"

    def test_text_query_scene(self):
        result = encode_query("search for outdoor nature photos")
        attrs = result["attributes"]
        assert attrs["scene_category"] == "outdoor" or attrs["scene_category"] == "nature"

    def test_nonexistent_image_treated_as_text(self):
        result = encode_query("nonexistent_file.txt")
        assert "name" in result
        assert "attributes" in result


class TestEntryToRecord:
    """Test entry_to_record for JSONL exemplar conversion."""

    def test_basic_entry(self, sample_features):
        entry = {
            "image_id": "img_001",
            "features": sample_features,
            "metadata": {"source": "test"},
        }
        record = entry_to_record(entry)
        assert record["concept_text"] == "img_001"
        assert "attributes" in record
        assert record["metadata"]["source"] == "test"

    def test_missing_features_filled(self):
        entry = {"image_id": "img_002", "features": {}}
        record = entry_to_record(entry)
        attrs = record["attributes"]
        assert attrs["expression"] == "none"
        assert attrs["face_embedding"] == [0.0] * 512
        assert attrs["direction"] == "none"

    def test_fill_defaults(self):
        features = {"expression": "happy"}
        _fill_defaults(features)
        assert features["expression"] == "happy"  # Not overwritten
        assert features["face_embedding"] == [0.0] * 512  # Filled
        assert features["direction"] == "none"  # Filled
