"""Test temporal encoding for deal velocity tracking."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG

glyphh = pytest.importorskip("glyphh")

from glyphh import Encoder, Concept
from glyphh.core.ops import cosine_similarity


@pytest.fixture(scope="module")
def encoder():
    return Encoder(ENCODER_CONFIG)


def test_same_deal_different_metrics_different_glyphs(encoder):
    """Same deal_id with different metrics should produce different glyphs."""
    g1 = encoder.encode(Concept(
        name="deal-001",
        attributes={
            "deal_id": "deal-001",
            "account_name": "Test Corp",
            "industry": "technology",
            "company_size": "enterprise",
            "region": "us-west",
            "source": "inbound",
            "description": "enterprise deal",
            "keywords": "enterprise technology",
            "stage": "discovery",
            "deal_type": "new_business",
            "deal_value": 50000,
            "days_in_stage": 5,
            "activity_count": 3,
            "email_response_rate": 0,
            "meetings_held": 1,
        },
    ))

    g2 = encoder.encode(Concept(
        name="deal-001",
        attributes={
            "deal_id": "deal-001",
            "account_name": "Test Corp",
            "industry": "technology",
            "company_size": "enterprise",
            "region": "us-west",
            "source": "inbound",
            "description": "enterprise deal",
            "keywords": "enterprise technology",
            "stage": "negotiation",
            "deal_type": "new_business",
            "deal_value": 50000,
            "days_in_stage": 10,
            "activity_count": 20,
            "email_response_rate": 75,
            "meetings_held": 8,
        },
    ))

    # Global cortex should differ (metrics changed)
    cortex_sim = float(cosine_similarity(
        g1.global_cortex.data, g2.global_cortex.data
    ))
    assert cortex_sim < 1.0, "Same deal with different metrics should not be identical"

    # Metrics layer should be quite different
    metrics_sim = float(cosine_similarity(
        g1.layers["metrics"].cortex.data,
        g2.layers["metrics"].cortex.data,
    ))
    assert metrics_sim < 0.9, f"Metrics should differ: {metrics_sim:.4f}"


def test_deal_id_is_key_part(encoder):
    """deal_id should be marked as key_part for temporal tracking."""
    identity = next(l for l in ENCODER_CONFIG.layers if l.name == "identity")
    deal_seg = next(s for s in identity.segments if s.name == "deal")
    did_role = next(r for r in deal_seg.roles if r.name == "deal_id")
    assert did_role.key_part is True


def test_temporal_source_is_auto():
    """Temporal source should be auto."""
    assert ENCODER_CONFIG.temporal_source == "auto"
    assert ENCODER_CONFIG.temporal_config.signal_type == "auto"
