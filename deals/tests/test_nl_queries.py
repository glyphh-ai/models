"""Test that NL queries match the correct exemplar outcome categories.

These tests encode NL text queries (via the strategy layer) and compare
against training exemplars to verify broad domain queries like "stalled"
and "closing" match the expected outcome categories.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG, encode_query, entry_to_record

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
            # Skip catch-all exemplar (too generic for NL matching)
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


def _strategy_similarity(glyph1, glyph2):
    """Strategy layer cortex similarity only."""
    v1 = glyph1.layers["strategy"].cortex.data
    v2 = glyph2.layers["strategy"].cortex.data
    return float(cosine_similarity(v1, v2))


def _encode_nl_query(query_text, encoder):
    """Encode an NL query using the model's encode_query function."""
    concept_dict = encode_query(query_text)
    concept = Concept(
        name=concept_dict["name"],
        attributes=concept_dict["attributes"],
    )
    return encoder.encode(concept)


def _find_best_match(query_glyph, pattern_glyphs, sim_fn=_strategy_similarity):
    """Find the training exemplar most similar to a query glyph."""
    best_score = -1
    best_meta = None
    for pattern_glyph, meta in pattern_glyphs:
        score = sim_fn(query_glyph, pattern_glyph)
        if score > best_score:
            best_score = score
            best_meta = meta
    return best_meta, best_score


def _best_score_for_outcome(query_glyph, pattern_glyphs, outcome):
    """Best strategy similarity score against exemplars of a given outcome."""
    return max(
        _strategy_similarity(query_glyph, pg)
        for pg, m in pattern_glyphs
        if m["outcome"] == outcome
    )


# ---------------------------------------------------------------------------
# Won / closing queries
# ---------------------------------------------------------------------------

class TestWonQueries:
    """Strong deal and closing queries -> won patterns."""

    def test_deals_about_to_close(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals about to close", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] == "won", f"matched {meta['outcome']} ({score:.4f})"

    def test_strong_deals_with_champion(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("strong deals with a champion", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("won", "pending"), f"matched {meta['outcome']} ({score:.4f})"

    def test_high_velocity_deals(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("high velocity deals closing fast", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] == "won", f"matched {meta['outcome']} ({score:.4f})"

    def test_deals_with_executive_sponsor(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals with executive sponsor", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("won", "pending"), f"matched {meta['outcome']} ({score:.4f})"

    def test_expansion_opportunities(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("expansion and upsell opportunities", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] == "won", f"matched {meta['outcome']} ({score:.4f})"


# ---------------------------------------------------------------------------
# Lost / dead queries
# ---------------------------------------------------------------------------

class TestLostQueries:
    """Dead deal and lost queries -> lost patterns."""

    def test_deals_gone_dark(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals that have gone dark", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("lost", "stalled"), f"matched {meta['outcome']} ({score:.4f})"

    def test_ghosted_prospects(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("ghosted prospects with no response", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("lost", "stalled"), f"matched {meta['outcome']} ({score:.4f})"

    def test_lost_to_competitor(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals lost to a competitor", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] == "lost", f"matched {meta['outcome']} ({score:.4f})"

    def test_dead_deals_no_activity(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("dead deals with no activity", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("lost", "stalled"), f"matched {meta['outcome']} ({score:.4f})"


# ---------------------------------------------------------------------------
# Stalled / at-risk queries
# ---------------------------------------------------------------------------

class TestStalledQueries:
    """Stalled and at-risk deal queries -> stalled patterns."""

    def test_stalled_deals_in_pipeline(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("stalled deals in the pipeline", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("stalled", "lost"), f"matched {meta['outcome']} ({score:.4f})"

    def test_deals_with_budget_objection(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals with budget objections", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("stalled", "lost"), f"matched {meta['outcome']} ({score:.4f})"

    def test_deals_slipping_past_close_date(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals slipping past their close date", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("stalled", "lost"), f"matched {meta['outcome']} ({score:.4f})"

    def test_at_risk_deals(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals at risk of being lost", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("stalled", "lost"), f"matched {meta['outcome']} ({score:.4f})"

    def test_deals_with_no_decision_maker(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals without a clear decision maker", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        # "decision maker" is ambiguous — maps to stakeholder/champion which
        # can match either stalled (no DM) or won (has champion) exemplars
        assert meta["outcome"] in ("stalled", "pending", "won"), f"matched {meta['outcome']} ({score:.4f})"


# ---------------------------------------------------------------------------
# Pipeline / stage queries
# ---------------------------------------------------------------------------

class TestPipelineQueries:
    """Pipeline and stage queries -> appropriate patterns."""

    def test_deals_in_discovery(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals in discovery stage", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("stalled", "pending"), f"matched {meta['outcome']} ({score:.4f})"

    def test_deals_in_negotiation(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals in negotiation", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        # Negotiation has both won and stalled patterns
        assert meta["outcome"] in ("won", "stalled"), f"matched {meta['outcome']} ({score:.4f})"

    def test_early_stage_prospecting(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("early stage prospecting deals", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("pending", "stalled", "lost"), f"matched {meta['outcome']} ({score:.4f})"


# ---------------------------------------------------------------------------
# Score ordering — directional correctness
# ---------------------------------------------------------------------------

class TestScoreOrdering:
    """Verify won queries score higher against won exemplars and
    lost queries score higher against lost exemplars."""

    def test_closing_won_over_lost(self, encoder, pattern_glyphs):
        """'closing' query: best won score > best lost score."""
        glyph = _encode_nl_query("deals with momentum about to close", encoder)
        best_won = _best_score_for_outcome(glyph, pattern_glyphs, "won")
        best_lost = _best_score_for_outcome(glyph, pattern_glyphs, "lost")
        assert best_won > best_lost, f"won={best_won:.4f} lost={best_lost:.4f}"

    def test_ghosted_lost_over_won(self, encoder, pattern_glyphs):
        """'ghosted' query: best lost score > best won score."""
        glyph = _encode_nl_query("ghosted unresponsive dead deals", encoder)
        best_won = _best_score_for_outcome(glyph, pattern_glyphs, "won")
        best_lost = _best_score_for_outcome(glyph, pattern_glyphs, "lost")
        assert best_lost > best_won, f"lost={best_lost:.4f} won={best_won:.4f}"

    def test_stalled_higher_than_won(self, encoder, pattern_glyphs):
        """'stalled stuck' query should prefer stalled over won."""
        glyph = _encode_nl_query("stuck stalled deals not moving forward", encoder)
        best_won = _best_score_for_outcome(glyph, pattern_glyphs, "won")
        best_stalled = _best_score_for_outcome(glyph, pattern_glyphs, "stalled")
        assert best_stalled > best_won, f"stalled={best_stalled:.4f} won={best_won:.4f}"

    def test_champion_won_over_lost(self, encoder, pattern_glyphs):
        """'champion engaged' query should prefer won over lost."""
        glyph = _encode_nl_query("deals with strong champion engagement and momentum", encoder)
        best_won = _best_score_for_outcome(glyph, pattern_glyphs, "won")
        best_lost = _best_score_for_outcome(glyph, pattern_glyphs, "lost")
        assert best_won > best_lost, f"won={best_won:.4f} lost={best_lost:.4f}"

    def test_competitor_lost_over_won(self, encoder, pattern_glyphs):
        """'competitor' query should prefer lost over won."""
        glyph = _encode_nl_query("deals being evaluated against competitors", encoder)
        best_won = _best_score_for_outcome(glyph, pattern_glyphs, "won")
        best_lost = _best_score_for_outcome(glyph, pattern_glyphs, "lost")
        assert best_lost > best_won, f"lost={best_lost:.4f} won={best_won:.4f}"


# ---------------------------------------------------------------------------
# Stemming — inflected forms should match the same tier
# ---------------------------------------------------------------------------

class TestStemming:
    """Stemmed variants of key terms should match the same outcome."""

    def test_closing_matches_won(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals that are closing", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] == "won", f"matched {meta['outcome']} ({score:.4f})"

    def test_stalling_matches_stalled(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals that are stalling out", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("stalled", "lost"), f"matched {meta['outcome']} ({score:.4f})"

    def test_slipping_matches_at_risk(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("deals slipping behind schedule", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["outcome"] in ("stalled", "lost"), f"matched {meta['outcome']} ({score:.4f})"
