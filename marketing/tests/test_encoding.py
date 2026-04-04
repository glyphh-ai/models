"""
Test encoding pipeline — config validation, encode_query, entry_to_record.
"""

import numpy as np
import pytest

from glyphh import Encoder
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity

from encoder import ENCODER_CONFIG, encode_query, entry_to_record


class TestEncoderConfig:
    """Validate ENCODER_CONFIG structure."""

    def test_dimension(self):
        assert ENCODER_CONFIG.dimension == 2000

    def test_seed(self):
        assert ENCODER_CONFIG.seed == 42

    def test_temporal_enabled(self):
        assert ENCODER_CONFIG.include_temporal is True

    def test_layer_count(self):
        assert len(ENCODER_CONFIG.layers) == 4

    def test_layer_names(self):
        names = [l.name for l in ENCODER_CONFIG.layers]
        assert names == ["identity", "strategy", "content", "performance"]

    def test_layer_weights_sum_to_one(self):
        total = sum(l.similarity_weight for l in ENCODER_CONFIG.layers)
        assert abs(total - 1.0) < 0.01

    def test_identity_layer_has_key_part(self):
        identity = ENCODER_CONFIG.layers[0]
        entity_seg = identity.segments[0]
        entity_id_role = [r for r in entity_seg.roles if r.name == "entity_id"][0]
        assert entity_id_role.key_part is True

    def test_performance_layer_has_continuous_roles(self):
        perf = ENCODER_CONFIG.layers[3]
        metrics_seg = perf.segments[0]
        for role in metrics_seg.roles:
            assert role.continuous_config is not None, f"{role.name} missing continuous_config"

    def test_content_layer_has_bow_roles(self):
        content = ENCODER_CONFIG.layers[2]
        messaging_seg = content.segments[0]
        bow_roles = [r for r in messaging_seg.roles if r.text_encoding == "bag_of_words"]
        assert len(bow_roles) == 2  # themes and keywords

    def test_encoder_creates_successfully(self, encoder):
        assert encoder is not None


class TestEncodeQuery:
    """Test NL query encoding."""

    def test_returns_dict_with_name_and_attributes(self):
        result = encode_query("find campaigns similar to product launch")
        assert "name" in result
        assert "attributes" in result
        assert result["name"].startswith("query_")

    def test_campaign_target_detection(self):
        result = encode_query("find campaigns with high engagement")
        assert result["attributes"]["entity_type"] == "campaign"

    def test_post_target_detection(self):
        result = encode_query("show me posts about developer tools")
        assert result["attributes"]["entity_type"] == "post"

    def test_audience_target_detection(self):
        result = encode_query("find audience segments for enterprise")
        assert result["attributes"]["entity_type"] == "audience"

    def test_goal_extraction(self):
        result = encode_query("find conversion campaigns")
        assert result["attributes"]["goal"] == "conversion"

    def test_platform_extraction(self):
        result = encode_query("show linkedin campaigns")
        assert "linkedin" in result["attributes"]["platform"]

    def test_tone_extraction(self):
        result = encode_query("find technical content about apis")
        assert result["attributes"]["tone"] == "technical"

    def test_keywords_populated(self):
        result = encode_query("find campaigns about product launch")
        assert len(result["attributes"]["keywords"]) > 0

    def test_performance_metrics_omitted_from_queries(self):
        """NL queries should not include performance metrics — encoder skips
        the performance layer entirely, avoiding zero-vector projection noise."""
        result = encode_query("find high performing campaigns")
        assert "engagement_metrics" not in result["attributes"]
        assert "conversion_metrics" not in result["attributes"]
        assert "reach_metrics" not in result["attributes"]

    def test_undetected_roles_omitted(self):
        """Roles with no detected value should be omitted, not set to 'none'.
        This prevents false matches with exemplars that also have 'none'."""
        result = encode_query("find campaigns")
        # tone/goal/content_type not detected → should be absent
        assert "tone" not in result["attributes"]
        assert "goal" not in result["attributes"]

    def test_deterministic(self):
        q = "find campaigns similar to our product launch"
        r1 = encode_query(q)
        r2 = encode_query(q)
        assert r1 == r2

    def test_query_encodes_to_glyph(self, encoder):
        result = encode_query("find campaigns with high engagement on linkedin")
        concept = Concept(name=result["name"], attributes=result["attributes"])
        glyph = encoder.encode(concept)
        assert glyph is not None
        assert glyph.global_cortex.data.shape[0] == 2000


class TestEntryToRecord:
    """Test exemplar conversion."""

    def test_campaign_record(self, sample_campaign):
        record = entry_to_record(sample_campaign)
        assert record["concept_text"] == "test_campaign_1"
        assert record["attributes"]["entity_type"] == "campaign"
        assert record["attributes"]["goal"] == "conversion"
        assert len(record["attributes"]["engagement_metrics"]) == 5
        assert len(record["attributes"]["conversion_metrics"]) == 5
        assert len(record["attributes"]["reach_metrics"]) == 4

    def test_post_record(self, sample_post):
        record = entry_to_record(sample_post)
        assert record["attributes"]["entity_type"] == "post"
        assert record["attributes"]["content_type"] == "social_post"

    def test_audience_record(self, sample_audience):
        record = entry_to_record(sample_audience)
        assert record["attributes"]["entity_type"] == "audience"
        assert "developer" in record["attributes"]["audience_segment"]

    def test_metadata_preserved(self, sample_campaign):
        record = entry_to_record(sample_campaign)
        assert "entity_type" in record["metadata"]
        assert "entity_id" in record["metadata"]

    def test_keywords_fallback_to_themes(self):
        entry = {
            "entity_type": "campaign",
            "entity_id": "no_keywords",
            "themes": "some themes here",
        }
        record = entry_to_record(entry)
        assert record["attributes"]["keywords"] == "some themes here"

    def test_all_exemplars_encode(self, encoder, exemplars):
        for entry in exemplars:
            record = entry_to_record(entry)
            concept = Concept(name=record["concept_text"], attributes=record["attributes"])
            glyph = encoder.encode(concept)
            assert glyph is not None
            assert glyph.global_cortex.data.shape[0] == 2000


class TestEncodingDeterminism:
    """Encoding must be deterministic."""

    def test_same_input_same_glyph(self, encoder, sample_campaign):
        """Same input should produce highly similar glyphs.

        Not exactly 1.0 because include_temporal=True adds auto-timestamps,
        so each encode gets a slightly different temporal layer.
        Non-temporal layers should be identical.
        """
        record = entry_to_record(sample_campaign)
        concept = Concept(name=record["concept_text"], attributes=record["attributes"])
        g1 = encoder.encode(concept)
        g2 = encoder.encode(concept)

        # Non-temporal layers should be identical
        for layer_name in ["identity", "strategy", "content", "performance"]:
            layer_sim = cosine_similarity(
                g1.layers[layer_name].cortex.data,
                g2.layers[layer_name].cortex.data,
            )
            assert layer_sim == 1.0, f"{layer_name} layer should be deterministic"

    def test_different_inputs_different_glyphs(self, encoder, sample_campaign, sample_post):
        r1 = entry_to_record(sample_campaign)
        r2 = entry_to_record(sample_post)
        g1 = encoder.encode(Concept(name=r1["concept_text"], attributes=r1["attributes"]))
        g2 = encoder.encode(Concept(name=r2["concept_text"], attributes=r2["attributes"]))
        sim = cosine_similarity(g1.global_cortex.data, g2.global_cortex.data)
        assert sim < 0.9  # Different entities should not be identical
