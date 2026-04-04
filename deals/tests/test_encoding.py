"""Test that the encoder config is valid and roles encode correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG


def test_config_has_required_fields(encoder_config):
    """EncoderConfig must have dimension, seed, and layers."""
    assert encoder_config.dimension > 0
    assert encoder_config.seed >= 0
    assert len(encoder_config.layers) >= 1


def test_identity_layer_exists(encoder_config):
    """Must have an identity layer with deal segment."""
    layer_names = [l.name for l in encoder_config.layers]
    assert "identity" in layer_names

    identity = next(l for l in encoder_config.layers if l.name == "identity")
    seg_names = [s.name for s in identity.segments]
    assert "deal" in seg_names


def test_profile_layer_exists(encoder_config):
    """Must have a profile layer with firmographic segment."""
    layer_names = [l.name for l in encoder_config.layers]
    assert "profile" in layer_names

    profile = next(l for l in encoder_config.layers if l.name == "profile")
    seg_names = [s.name for s in profile.segments]
    assert "firmographic" in seg_names

    roles = profile.segments[0].roles
    role_names = [r.name for r in roles]
    assert "industry" in role_names
    assert "company_size" in role_names
    assert "region" in role_names
    assert "source" in role_names


def test_strategy_layer_exists(encoder_config):
    """Must have a strategy layer with context segment."""
    layer_names = [l.name for l in encoder_config.layers]
    assert "strategy" in layer_names

    strategy = next(l for l in encoder_config.layers if l.name == "strategy")
    seg_names = [s.name for s in strategy.segments]
    assert "context" in seg_names

    roles = strategy.segments[0].roles
    role_names = [r.name for r in roles]
    assert "description" in role_names
    assert "keywords" in role_names
    assert "stage" in role_names
    assert "deal_type" in role_names


def test_metrics_layer_exists(encoder_config):
    """Must have a metrics layer with numeric roles."""
    layer_names = [l.name for l in encoder_config.layers]
    assert "metrics" in layer_names

    metrics = next(l for l in encoder_config.layers if l.name == "metrics")
    roles = metrics.segments[0].roles
    role_names = [r.name for r in roles]
    assert "deal_value" in role_names
    assert "days_in_stage" in role_names
    assert "activity_count" in role_names
    assert "email_response_rate" in role_names
    assert "meetings_held" in role_names


def test_deal_id_is_key_part(encoder_config):
    """deal_id must be the key_part role for temporal identity."""
    identity = next(l for l in encoder_config.layers if l.name == "identity")
    deal_seg = next(s for s in identity.segments if s.name == "deal")
    did_role = next(r for r in deal_seg.roles if r.name == "deal_id")
    assert did_role.key_part is True


def test_numeric_roles_have_config(encoder_config):
    """All metric roles must have NumericConfig with THERMOMETER encoding."""
    metrics = next(l for l in encoder_config.layers if l.name == "metrics")
    for role in metrics.segments[0].roles:
        assert role.numeric_config is not None, f"{role.name} missing numeric_config"
        assert role.numeric_config.bin_width > 0
        assert role.numeric_config.encoding_strategy.value == "thermometer"


def test_temporal_config(encoder_config):
    """Temporal source should be auto with auto signal type."""
    assert encoder_config.temporal_source == "auto"
    assert encoder_config.temporal_config is not None
    assert encoder_config.temporal_config.signal_type == "auto"


def test_test_deals_have_raw_metrics_only(test_deals):
    """Test data should only contain raw metrics — no outcome or risk_level in attributes."""
    for d in test_deals:
        assert "outcome" not in d, f"{d['deal_id']} has outcome — test data should be raw"
        assert "risk_level" not in d, f"{d['deal_id']} has risk_level — test data should be raw"
        assert "deal_id" in d
        assert "deal_value" in d
        assert "days_in_stage" in d
        assert "activity_count" in d
        assert "email_response_rate" in d
        assert "meetings_held" in d


def test_bow_roles_have_text_encoding(encoder_config):
    """Description and keywords roles must have bag_of_words encoding."""
    strategy = next(l for l in encoder_config.layers if l.name == "strategy")
    context = next(s for s in strategy.segments if s.name == "context")

    desc_role = next(r for r in context.roles if r.name == "description")
    assert desc_role.text_encoding == "bag_of_words"

    kw_role = next(r for r in context.roles if r.name == "keywords")
    assert kw_role.text_encoding == "bag_of_words"
