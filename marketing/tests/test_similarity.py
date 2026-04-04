"""
Test similarity — cross-type matching, performance proximity, layer-level assertions.
"""

import numpy as np
import pytest

from glyphh import Encoder
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity

from encoder import ENCODER_CONFIG, encode_query, entry_to_record


class TestCrossTypeMatching:
    """Campaigns, posts, and audiences should match on shared attributes."""

    def test_same_theme_higher_similarity(self, encoder):
        """Two campaigns with same themes > two campaigns with different themes."""
        c1 = entry_to_record({
            "entity_type": "campaign", "entity_id": "a",
            "themes": "product launch api developer",
            "keywords": "product launch api developer",
            "goal": "conversion", "tone": "professional",
            "engagement_metrics": [100, 50, 10, 5, 20],
            "conversion_metrics": [200, 30, 5, 10, 2],
            "reach_metrics": [5000, 2000, 2.0, 0.1],
        })
        c2 = entry_to_record({
            "entity_type": "campaign", "entity_id": "b",
            "themes": "product launch api developer",
            "keywords": "product launch api developer integration",
            "goal": "conversion", "tone": "professional",
            "engagement_metrics": [120, 60, 15, 8, 25],
            "conversion_metrics": [250, 40, 8, 15, 3],
            "reach_metrics": [6000, 2500, 2.2, 0.12],
        })
        c3 = entry_to_record({
            "entity_type": "campaign", "entity_id": "c",
            "themes": "holiday discount promotion seasonal",
            "keywords": "holiday discount promotion seasonal sale",
            "goal": "conversion", "tone": "urgent",
            "engagement_metrics": [5000, 1500, 300, 800, 2000],
            "conversion_metrics": [8000, 2000, 600, 100, 40],
            "reach_metrics": [150000, 60000, 5.0, 0.2],
        })

        g1 = encoder.encode(Concept(name=c1["concept_text"], attributes=c1["attributes"]))
        g2 = encoder.encode(Concept(name=c2["concept_text"], attributes=c2["attributes"]))
        g3 = encoder.encode(Concept(name=c3["concept_text"], attributes=c3["attributes"]))

        sim_same = cosine_similarity(g1.global_cortex.data, g2.global_cortex.data)
        sim_diff = cosine_similarity(g1.global_cortex.data, g3.global_cortex.data)
        assert sim_same > sim_diff, f"Same theme sim {sim_same:.4f} should > diff theme {sim_diff:.4f}"

    def test_campaign_post_with_same_campaign_id_similar(self, encoder):
        """A post belonging to a campaign should share themes/keywords similarity."""
        campaign = entry_to_record({
            "entity_type": "campaign", "entity_id": "launch",
            "themes": "product launch api developer",
            "keywords": "product launch api developer integration",
            "goal": "conversion", "audience_segment": "technical",
            "platform": "twitter linkedin",
            "engagement_metrics": [1000, 300, 80, 40, 150],
            "conversion_metrics": [2000, 150, 40, 90, 10],
            "reach_metrics": [40000, 10000, 3.0, 0.15],
        })
        related_post = entry_to_record({
            "entity_type": "post", "entity_id": "launch_tweet",
            "themes": "product launch api developer announcement",
            "keywords": "product launch api developer announcement new",
            "goal": "awareness", "audience_segment": "technical",
            "platform": "twitter", "content_type": "social_post",
            "engagement_metrics": [400, 150, 35, 20, 80],
            "conversion_metrics": [300, 40, 8, 10, 2],
            "reach_metrics": [15000, 8000, 1.0, 0.0],
        })
        unrelated_post = entry_to_record({
            "entity_type": "post", "entity_id": "holiday_email",
            "themes": "holiday discount promotion limited time",
            "keywords": "holiday discount promotion limited time sale",
            "goal": "conversion", "audience_segment": "consumer",
            "platform": "email", "content_type": "email",
            "engagement_metrics": [0, 0, 0, 0, 0],
            "conversion_metrics": [4000, 1000, 350, 0, 0],
            "reach_metrics": [40000, 38000, 1.0, 0.0],
        })

        g_c = encoder.encode(Concept(name=campaign["concept_text"], attributes=campaign["attributes"]))
        g_r = encoder.encode(Concept(name=related_post["concept_text"], attributes=related_post["attributes"]))
        g_u = encoder.encode(Concept(name=unrelated_post["concept_text"], attributes=unrelated_post["attributes"]))

        sim_related = cosine_similarity(g_c.global_cortex.data, g_r.global_cortex.data)
        sim_unrelated = cosine_similarity(g_c.global_cortex.data, g_u.global_cortex.data)
        assert sim_related > sim_unrelated


class TestPerformanceSimilarity:
    """Performance metrics should drive similarity for similar-performance queries."""

    def test_similar_performance_profiles(self, encoder):
        """Two campaigns with similar metric *patterns* > two with different patterns.

        ContinuousProjector is scale-invariant (sign quantization), so we test
        pattern similarity: same proportional distribution vs a different one.
        Compare at the performance layer level to isolate the signal.
        """
        # Pattern A: engagement-heavy (high engagement, low conversion)
        pattern_a1 = entry_to_record({
            "entity_type": "campaign", "entity_id": "pat_a1",
            "themes": "generic", "keywords": "generic",
            "engagement_metrics": [5000, 1500, 300, 200, 800],
            "conversion_metrics": [100, 20, 5, 10, 2],
            "reach_metrics": [100000, 40000, 4.0, 0.2],
        })
        pattern_a2 = entry_to_record({
            "entity_type": "campaign", "entity_id": "pat_a2",
            "themes": "generic", "keywords": "generic",
            "engagement_metrics": [4800, 1400, 280, 190, 750],
            "conversion_metrics": [95, 18, 4, 8, 1],
            "reach_metrics": [95000, 38000, 3.8, 0.19],
        })
        # Pattern B: conversion-heavy (low engagement, high conversion)
        pattern_b = entry_to_record({
            "entity_type": "campaign", "entity_id": "pat_b",
            "themes": "generic", "keywords": "generic",
            "engagement_metrics": [50, 10, 2, 1, 5],
            "conversion_metrics": [8000, 2000, 600, 100, 40],
            "reach_metrics": [1000, 400, 1.0, 0.01],
        })

        g_a1 = encoder.encode(Concept(name=pattern_a1["concept_text"], attributes=pattern_a1["attributes"]))
        g_a2 = encoder.encode(Concept(name=pattern_a2["concept_text"], attributes=pattern_a2["attributes"]))
        g_b = encoder.encode(Concept(name=pattern_b["concept_text"], attributes=pattern_b["attributes"]))

        # Compare at performance layer level
        sim_same = cosine_similarity(
            g_a1.layers["performance"].cortex.data,
            g_a2.layers["performance"].cortex.data,
        )
        sim_diff = cosine_similarity(
            g_a1.layers["performance"].cortex.data,
            g_b.layers["performance"].cortex.data,
        )
        assert sim_same > sim_diff, f"Same pattern {sim_same:.4f} should > diff pattern {sim_diff:.4f}"


class TestLayerLevelSimilarity:
    """Test similarity at individual layer levels."""

    def test_content_layer_matches_on_themes(self, encoder):
        """Content layer should show high similarity for shared themes."""
        r1 = entry_to_record({
            "entity_type": "campaign", "entity_id": "a",
            "themes": "ai machine learning innovation",
            "keywords": "ai machine learning innovation research",
            "tone": "technical",
            "engagement_metrics": [100, 50, 10, 5, 20],
            "conversion_metrics": [200, 30, 5, 10, 2],
            "reach_metrics": [5000, 2000, 2.0, 0.1],
        })
        r2 = entry_to_record({
            "entity_type": "post", "entity_id": "b",
            "themes": "ai machine learning innovation trends",
            "keywords": "ai machine learning innovation trends prediction",
            "tone": "professional",
            "engagement_metrics": [300, 120, 40, 80, 30],
            "conversion_metrics": [150, 35, 3, 15, 2],
            "reach_metrics": [20000, 10000, 1.0, 0.0],
        })

        g1 = encoder.encode(Concept(name=r1["concept_text"], attributes=r1["attributes"]))
        g2 = encoder.encode(Concept(name=r2["concept_text"], attributes=r2["attributes"]))

        content_sim = cosine_similarity(
            g1.layers["content"].cortex.data,
            g2.layers["content"].cortex.data,
        )
        assert content_sim > 0.15, f"Content layer sim {content_sim:.4f} should be positive (shared themes)"

    def test_strategy_layer_matches_on_goal(self, encoder):
        """Strategy layer should show high similarity for same goal + audience."""
        r1 = entry_to_record({
            "entity_type": "campaign", "entity_id": "a",
            "goal": "conversion", "audience_segment": "technical",
            "platform": "linkedin", "content_type": "none",
            "themes": "a", "keywords": "a",
            "engagement_metrics": [0] * 5,
            "conversion_metrics": [0] * 5,
            "reach_metrics": [0] * 4,
        })
        r2 = entry_to_record({
            "entity_type": "campaign", "entity_id": "b",
            "goal": "conversion", "audience_segment": "technical",
            "platform": "linkedin", "content_type": "none",
            "themes": "b", "keywords": "b",
            "engagement_metrics": [0] * 5,
            "conversion_metrics": [0] * 5,
            "reach_metrics": [0] * 4,
        })
        r3 = entry_to_record({
            "entity_type": "campaign", "entity_id": "c",
            "goal": "awareness", "audience_segment": "consumer",
            "platform": "instagram", "content_type": "none",
            "themes": "c", "keywords": "c",
            "engagement_metrics": [0] * 5,
            "conversion_metrics": [0] * 5,
            "reach_metrics": [0] * 4,
        })

        g1 = encoder.encode(Concept(name=r1["concept_text"], attributes=r1["attributes"]))
        g2 = encoder.encode(Concept(name=r2["concept_text"], attributes=r2["attributes"]))
        g3 = encoder.encode(Concept(name=r3["concept_text"], attributes=r3["attributes"]))

        sim_same = cosine_similarity(
            g1.layers["strategy"].cortex.data,
            g2.layers["strategy"].cortex.data,
        )
        sim_diff = cosine_similarity(
            g1.layers["strategy"].cortex.data,
            g3.layers["strategy"].cortex.data,
        )
        assert sim_same > sim_diff


class TestQueryToExemplarMatching:
    """NL queries should match relevant exemplars."""

    def test_product_launch_query(self, encoder, all_glyphs):
        """'find campaigns similar to product launch' should rank product_launch_q1 high.

        NL queries have no performance layer, so we use layer-weighted scoring
        across shared layers (identity + strategy + content) rather than raw
        cortex comparison. This mirrors how the runtime would score with
        apply_weights_during_encoding=False.
        """
        query_dict = encode_query("find campaigns similar to our product launch")
        q_concept = Concept(name=query_dict["name"], attributes=query_dict["attributes"])
        q_glyph = encoder.encode(q_concept)

        # Layer-weighted scoring across shared layers only
        layer_weights = {"identity": 0.10, "strategy": 0.25, "content": 0.30}

        scores = []
        for entry, glyph in all_glyphs:
            weighted_sum = 0.0
            weight_total = 0.0
            for lname, weight in layer_weights.items():
                if lname in q_glyph.layers and lname in glyph.layers:
                    sim = cosine_similarity(
                        q_glyph.layers[lname].cortex.data,
                        glyph.layers[lname].cortex.data,
                    )
                    weighted_sum += sim * weight
                    weight_total += weight
            score = weighted_sum / weight_total if weight_total > 0 else 0.0
            scores.append((entry["entity_id"], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = [s[0] for s in scores[:5]]
        assert "product_launch_q1" in top_ids, f"Expected product_launch_q1 in top 5, got {top_ids}"

    def test_technical_audience_query(self, encoder, all_glyphs):
        """'what content works for technical audiences' should surface technical content."""
        query_dict = encode_query("what content works for technical audiences")
        q_concept = Concept(name=query_dict["name"], attributes=query_dict["attributes"])
        q_glyph = encoder.encode(q_concept)

        scores = []
        for entry, glyph in all_glyphs:
            sim = cosine_similarity(q_glyph.global_cortex.data, glyph.global_cortex.data)
            scores.append((entry["entity_id"], entry.get("audience_segment", ""), sim))

        scores.sort(key=lambda x: x[2], reverse=True)
        # At least one of the top 5 should target technical audiences
        top_segments = [s[1] for s in scores[:5]]
        has_technical = any("technical" in seg for seg in top_segments)
        assert has_technical, f"Expected technical audience in top 5, got segments: {top_segments}"

    def test_email_platform_query(self, encoder, all_glyphs):
        """'compare email vs social performance' should surface email content."""
        query_dict = encode_query("show me email campaign performance")
        q_concept = Concept(name=query_dict["name"], attributes=query_dict["attributes"])
        q_glyph = encoder.encode(q_concept)

        scores = []
        for entry, glyph in all_glyphs:
            sim = cosine_similarity(q_glyph.global_cortex.data, glyph.global_cortex.data)
            scores.append((entry["entity_id"], entry.get("platform", ""), sim))

        scores.sort(key=lambda x: x[2], reverse=True)
        top_platforms = [s[1] for s in scores[:5]]
        has_email = any("email" in plat for plat in top_platforms)
        assert has_email, f"Expected email platform in top 5, got: {top_platforms}"
