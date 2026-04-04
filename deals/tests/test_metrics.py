"""Test metric encoding edge cases and similarity patterns."""

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


def _encode_deal_metrics(encoder, deal_value=0, days_in_stage=0,
                          activity_count=0, email_response_rate=0,
                          meetings_held=0):
    """Encode a deal with only metric values."""
    return encoder.encode(Concept(
        name="test_deal",
        attributes={
            "deal_id": "test",
            "account_name": "",
            "industry": "",
            "company_size": "",
            "region": "",
            "source": "",
            "description": "",
            "keywords": "",
            "stage": "",
            "deal_type": "",
            "deal_value": deal_value,
            "days_in_stage": days_in_stage,
            "activity_count": activity_count,
            "email_response_rate": email_response_rate,
            "meetings_held": meetings_held,
        },
    ))


def _metrics_sim(g1, g2):
    return float(cosine_similarity(
        g1.layers["metrics"].cortex.data,
        g2.layers["metrics"].cortex.data,
    ))


class TestDealValueEncoding:
    """Deal value metric encoding tests."""

    def test_zero_value_encodes(self, encoder):
        g = _encode_deal_metrics(encoder, deal_value=0)
        assert g.global_cortex is not None

    def test_max_value_encodes(self, encoder):
        g = _encode_deal_metrics(encoder, deal_value=500000)
        assert g.global_cortex is not None

    def test_over_max_encodes(self, encoder):
        g = _encode_deal_metrics(encoder, deal_value=1000000)
        assert g.global_cortex is not None

    def test_nearby_values_more_similar(self, encoder):
        g10k = _encode_deal_metrics(encoder, deal_value=10000)
        g30k = _encode_deal_metrics(encoder, deal_value=30000)
        g200k = _encode_deal_metrics(encoder, deal_value=200000)

        sim_close = _metrics_sim(g10k, g30k)
        sim_far = _metrics_sim(g10k, g200k)
        assert sim_close > sim_far, f"close={sim_close:.4f} far={sim_far:.4f}"


class TestDaysInStageEncoding:
    """Days in stage metric encoding tests."""

    def test_fresh_deal_encodes(self, encoder):
        g = _encode_deal_metrics(encoder, days_in_stage=0)
        assert g.global_cortex is not None

    def test_stale_deal_encodes(self, encoder):
        g = _encode_deal_metrics(encoder, days_in_stage=120)
        assert g.global_cortex is not None

    def test_nearby_days_more_similar(self, encoder):
        g5 = _encode_deal_metrics(encoder, days_in_stage=5)
        g10 = _encode_deal_metrics(encoder, days_in_stage=10)
        g90 = _encode_deal_metrics(encoder, days_in_stage=90)

        sim_close = _metrics_sim(g5, g10)
        sim_far = _metrics_sim(g5, g90)
        assert sim_close > sim_far, f"close={sim_close:.4f} far={sim_far:.4f}"


class TestActivityEncoding:
    """Activity count metric encoding tests."""

    def test_zero_activity(self, encoder):
        g = _encode_deal_metrics(encoder, activity_count=0)
        assert g.global_cortex is not None

    def test_high_activity(self, encoder):
        g = _encode_deal_metrics(encoder, activity_count=50)
        assert g.global_cortex is not None


class TestEmailResponseEncoding:
    """Email response rate encoding tests."""

    def test_zero_response(self, encoder):
        g = _encode_deal_metrics(encoder, email_response_rate=0)
        assert g.global_cortex is not None

    def test_full_response(self, encoder):
        g = _encode_deal_metrics(encoder, email_response_rate=100)
        assert g.global_cortex is not None

    def test_boundary_encoding(self, encoder):
        g0 = _encode_deal_metrics(encoder, email_response_rate=0)
        g100 = _encode_deal_metrics(encoder, email_response_rate=100)
        sim = _metrics_sim(g0, g100)
        # Extremes should be quite different
        assert sim < 0.8, f"0% vs 100% response rate too similar: {sim:.4f}"
