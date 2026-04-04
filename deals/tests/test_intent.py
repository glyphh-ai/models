"""Test intent extraction for deal intelligence queries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent import extract_keywords, preprocess, _stem


class TestPreprocess:
    """Text preprocessing tests."""

    def test_lowercases(self):
        assert preprocess("ENTERPRISE Deal") == "enterprise deal"

    def test_strips_punctuation(self):
        assert "?" not in preprocess("which deals are closing?")

    def test_strips_whitespace(self):
        result = preprocess("  hello  ")
        assert result == "hello"


class TestStemming:
    """Light stemming tests."""

    def test_closing_stems(self):
        assert _stem("closing") == "clos"

    def test_stalled_stems(self):
        assert _stem("stalled") == "stall"

    def test_ghosted_stems(self):
        assert _stem("ghosted") == "ghost"

    def test_short_word_not_stemmed(self):
        assert _stem("won") == "won"

    def test_exception_not_stemmed(self):
        assert _stem("sales") == "sales"


class TestExtractKeywords:
    """Keyword extraction tests."""

    def test_removes_stop_words(self):
        kw = extract_keywords("show me the deals in the pipeline")
        assert "show" not in kw.split()
        assert "the" not in kw.split()

    def test_preserves_domain_terms(self):
        kw = extract_keywords("stalled deals with budget objection")
        assert "stalled" in kw
        assert "budget" in kw

    def test_expands_synonyms(self):
        kw = extract_keywords("ghosted prospects")
        assert "gone_dark" in kw or "no_response" in kw

    def test_phrase_normalization(self):
        kw = extract_keywords("deals that have gone dark")
        assert "gone_dark" in kw

    def test_competitor_expansion(self):
        kw = extract_keywords("lost to competitor")
        assert "competitor" in kw
        assert "competitive" in kw or "alternative" in kw or "evaluation" in kw

    def test_champion_expansion(self):
        kw = extract_keywords("deals with a champion")
        assert "champion" in kw
        assert "strong" in kw or "engaged" in kw

    def test_pipeline_expansion(self):
        kw = extract_keywords("pipeline health check")
        assert "pipeline" in kw

    def test_renewal_expansion(self):
        kw = extract_keywords("renewal at risk")
        assert "renewal" in kw
        assert "at_risk" in kw or "renew" in kw

    def test_empty_after_stop_words(self):
        kw = extract_keywords("show me the")
        # Should still return something (even if minimal)
        assert isinstance(kw, str)

    def test_budget_expands_to_pricing(self):
        kw = extract_keywords("budget concerns")
        assert "budget" in kw
        assert "pricing" in kw or "cost" in kw

    def test_fast_deal_expansion(self):
        kw = extract_keywords("fast closing deals")
        assert "fast" in kw
        assert "quick" in kw or "accelerated" in kw or "momentum" in kw

    def test_stalled_expansion(self):
        kw = extract_keywords("stalled deals")
        assert "stalled" in kw
        assert "stuck" in kw or "no_activity" in kw
