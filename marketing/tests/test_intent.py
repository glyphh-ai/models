"""
Test intent extraction — NL query parsing for marketing queries.
"""

import pytest

from intent import extract_intent


class TestActionExtraction:
    """Test verb -> action mapping."""

    def test_search_verbs(self):
        for verb in ["find", "search", "show", "get", "discover"]:
            result = extract_intent(f"{verb} campaigns with high engagement")
            assert result["action"] == "search", f"{verb} should map to search"

    def test_compare_verbs(self):
        for verb in ["compare", "benchmark", "versus", "rank"]:
            result = extract_intent(f"{verb} email and social campaigns")
            assert result["action"] == "compare", f"{verb} should map to compare"

    def test_analyze_verbs(self):
        for verb in ["analyze", "report", "measure", "track", "evaluate"]:
            result = extract_intent(f"{verb} campaign performance")
            assert result["action"] == "analyze", f"{verb} should map to analyze"

    def test_optimize_verbs(self):
        for verb in ["optimize", "improve", "boost", "maximize"]:
            result = extract_intent(f"{verb} conversion rates")
            assert result["action"] == "optimize", f"{verb} should map to optimize"

    def test_create_verbs(self):
        for verb in ["create", "launch", "plan", "schedule", "draft"]:
            result = extract_intent(f"{verb} a new campaign")
            assert result["action"] == "create", f"{verb} should map to create"

    def test_target_verbs(self):
        for verb in ["target", "reach", "segment"]:
            result = extract_intent(f"{verb} enterprise users")
            assert result["action"] == "target", f"{verb} should map to target"

    def test_default_action_is_search(self):
        result = extract_intent("campaigns with high ROI")
        assert result["action"] == "search"


class TestTargetExtraction:
    """Test noun -> target mapping."""

    def test_campaign_targets(self):
        for word in ["campaign", "campaigns", "initiative"]:
            result = extract_intent(f"find {word} about product launches")
            assert result["target"] == "campaign", f"{word} should map to campaign"

    def test_post_targets(self):
        for word in ["post", "posts", "content", "article", "video"]:
            result = extract_intent(f"show {word} about developer tools")
            assert result["target"] == "post", f"{word} should map to post"

    def test_audience_targets(self):
        for word in ["audience", "segment", "cohort", "subscribers"]:
            result = extract_intent(f"find {word} for enterprise")
            assert result["target"] == "audience", f"{word} should map to audience"

    def test_performance_targets(self):
        for word in ["performance", "metrics", "analytics", "roi"]:
            result = extract_intent(f"analyze {word} for Q1")
            assert result["target"] == "performance", f"{word} should map to performance"


class TestDomainExtraction:
    """Test domain signal detection."""

    def test_social_domain(self):
        result = extract_intent("find twitter campaigns with high engagement")
        assert result["domain"] == "social"

    def test_email_domain(self):
        result = extract_intent("analyze newsletter open rates")
        assert result["domain"] == "email"

    def test_content_domain(self):
        result = extract_intent("find blog articles about SEO")
        assert result["domain"] == "content"

    def test_paid_domain(self):
        result = extract_intent("optimize google ads spend")
        assert result["domain"] == "paid"

    def test_brand_domain(self):
        result = extract_intent("track brand awareness campaign")
        assert result["domain"] == "brand"

    def test_analytics_domain(self):
        result = extract_intent("build attribution dashboard")
        assert result["domain"] == "analytics"

    def test_no_domain_defaults_to_none(self):
        result = extract_intent("find things")
        assert result["domain"] == "none"


class TestKeywordExtraction:
    """Test keyword extraction (stopword removal)."""

    def test_stopwords_removed(self):
        result = extract_intent("find the campaigns with high engagement")
        kw = result["keywords"]
        assert "the" not in kw
        assert "with" not in kw

    def test_meaningful_words_kept(self):
        result = extract_intent("find campaigns with high engagement on linkedin")
        kw = result["keywords"]
        assert "campaigns" in kw
        assert "high" in kw
        assert "engagement" in kw
        assert "linkedin" in kw

    def test_single_char_words_removed(self):
        result = extract_intent("a b c campaigns")
        words = result["keywords"].split()
        assert all(len(w) > 1 for w in words)


class TestModifiers:
    """Test modifier extraction."""

    def test_comparison_modifiers(self):
        result = extract_intent("find similar campaigns")
        assert "similar" in result["modifiers"]

    def test_superlative_modifiers(self):
        result = extract_intent("show best performing campaigns")
        assert "best" in result["modifiers"]

    def test_temporal_modifiers(self):
        result = extract_intent("show recent campaign results")
        assert "recent" in result["modifiers"]


class TestComplexQueries:
    """Test realistic multi-signal queries."""

    def test_full_query(self):
        result = extract_intent("find campaigns similar to our product launch on linkedin")
        assert result["action"] == "search"
        assert result["target"] == "campaign"
        assert "product" in result["keywords"]
        assert "launch" in result["keywords"]
        assert "linkedin" in result["keywords"]

    def test_what_works_query(self):
        result = extract_intent("what content works for technical audiences")
        assert result["target"] == "post"
        assert "technical" in result["keywords"]

    def test_comparison_query(self):
        result = extract_intent("compare email vs social campaign performance")
        assert result["action"] == "compare"
        assert result["target"] in ("post", "campaign", "performance")
