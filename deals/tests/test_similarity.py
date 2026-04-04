"""Test that raw deal data matches the correct outcome patterns.

These tests encode raw deal metrics (no outcome labels) and compare
against the training exemplars to verify the model correctly identifies
deal outcomes from metrics alone.

Deal-to-exemplar matching uses the metrics layer cortex since
deals have metrics but no text. The strategy layer drives NL query
matching; the metrics layer drives deal data matching.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG, entry_to_record

# These tests require the glyphh SDK
glyphh = pytest.importorskip("glyphh")

from glyphh import Encoder, Concept
from glyphh.core.ops import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def encoder():
    return Encoder(ENCODER_CONFIG)


@pytest.fixture(scope="module")
def pattern_glyphs(encoder):
    """Encode all training exemplars into glyphs with their metadata."""
    exemplars_path = DATA_DIR / "exemplars.jsonl"
    glyphs = []
    with open(exemplars_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Skip catch-all exemplars (no meaningful metrics)
            if entry.get("outcome") == "all":
                continue
            record = entry_to_record(entry)
            concept = Concept(
                name=record["concept_text"],
                attributes=record["attributes"],
            )
            glyph = encoder.encode(concept)
            glyphs.append((glyph, record["metadata"]))
    return glyphs


def _encode_deal(deal, encoder):
    """Encode a raw deal record (metrics only) into a glyph."""
    return encoder.encode(Concept(
        name=deal["deal_id"],
        attributes={
            "deal_id": deal["deal_id"],
            "account_name": deal.get("account_name", ""),
            "industry": deal.get("industry", ""),
            "company_size": deal.get("company_size", ""),
            "region": deal.get("region", ""),
            "source": deal.get("source", ""),
            "description": "",
            "keywords": "",
            "stage": deal.get("stage", ""),
            "deal_type": deal.get("deal_type", ""),
            "deal_value": deal["deal_value"],
            "days_in_stage": deal["days_in_stage"],
            "activity_count": deal["activity_count"],
            "email_response_rate": deal["email_response_rate"],
            "meetings_held": deal["meetings_held"],
        },
    ))


def _metrics_score(glyph1, glyph2):
    """Cosine similarity on metrics layer cortex vectors."""
    v1 = glyph1.layers["metrics"].cortex.data
    v2 = glyph2.layers["metrics"].cortex.data
    return float(cosine_similarity(v1, v2))


def _find_best_match(deal_glyph, pattern_glyphs):
    """Find the training exemplar most similar to a deal glyph."""
    best_score = -1
    best_meta = None
    for pattern_glyph, meta in pattern_glyphs:
        score = _metrics_score(deal_glyph, pattern_glyph)
        if score > best_score:
            best_score = score
            best_meta = meta
    return best_meta, best_score


def test_strong_enterprise_deal_matches_won(
    encoder, pattern_glyphs, expected_won
):
    """acme-enterprise (high activity, high response) should match won exemplars."""
    acme = next(d for d in expected_won if d["deal_id"] == "acme-enterprise")
    glyph = _encode_deal(acme, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["outcome"] == "won", (
        f"acme-enterprise matched {meta['outcome']} (score={score:.4f}), expected won"
    )


def test_ghosted_deal_matches_lost(
    encoder, pattern_glyphs, expected_lost
):
    """beta-ghost (no activity, no response) should match lost exemplars."""
    beta = next(d for d in expected_lost if d["deal_id"] == "beta-ghost")
    glyph = _encode_deal(beta, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["outcome"] in ("lost", "stalled"), (
        f"beta-ghost matched {meta['outcome']} (score={score:.4f}), expected lost/stalled"
    )


def test_stalled_deal_matches_stalled(
    encoder, pattern_glyphs, expected_stalled
):
    """gamma-stalled (weak engagement, long stall) should match stalled exemplars."""
    gamma = next(d for d in expected_stalled if d["deal_id"] == "gamma-stalled")
    glyph = _encode_deal(gamma, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["outcome"] in ("stalled", "lost"), (
        f"gamma-stalled matched {meta['outcome']} (score={score:.4f}), expected stalled/lost"
    )


def test_expansion_deal_matches_won(
    encoder, pattern_glyphs, expected_won
):
    """epsilon-expansion (high engagement, existing customer) should match won."""
    epsilon = next(d for d in expected_won if d["deal_id"] == "epsilon-expansion")
    glyph = _encode_deal(epsilon, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["outcome"] == "won", (
        f"epsilon-expansion matched {meta['outcome']} (score={score:.4f}), expected won"
    )


def test_fast_closing_deal_matches_won(
    encoder, pattern_glyphs, expected_won
):
    """iota-fast (high velocity, executive sponsor) should match won."""
    iota = next(d for d in expected_won if d["deal_id"] == "iota-fast")
    glyph = _encode_deal(iota, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["outcome"] == "won", (
        f"iota-fast matched {meta['outcome']} (score={score:.4f}), expected won"
    )


def test_cold_outbound_matches_lost(
    encoder, pattern_glyphs, expected_lost
):
    """kappa-cold (zero engagement, tiny deal) should match lost."""
    kappa = next(d for d in expected_lost if d["deal_id"] == "kappa-cold")
    glyph = _encode_deal(kappa, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["outcome"] in ("lost", "stalled"), (
        f"kappa-cold matched {meta['outcome']} (score={score:.4f}), expected lost/stalled"
    )


def test_won_scores_higher_than_lost(
    encoder, pattern_glyphs, expected_won, expected_lost
):
    """Won deals should have stronger matches to won exemplars
    than lost deals do."""
    acme = next(d for d in expected_won if d["deal_id"] == "acme-enterprise")
    beta = next(d for d in expected_lost if d["deal_id"] == "beta-ghost")

    acme_glyph = _encode_deal(acme, encoder)
    beta_glyph = _encode_deal(beta, encoder)

    won_exemplars = [(g, m) for g, m in pattern_glyphs if m["outcome"] == "won"]

    acme_best = max(_metrics_score(acme_glyph, pg) for pg, _ in won_exemplars)
    beta_best = max(_metrics_score(beta_glyph, pg) for pg, _ in won_exemplars)

    assert acme_best > beta_best, (
        f"acme-enterprise ({acme_best:.4f}) should score higher against won "
        f"exemplars than beta-ghost ({beta_best:.4f})"
    )
