"""Unit tests for encode_query function."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import encode_query


class TestEncodeQuery:
    """Test that encode_query returns properly structured concept dicts."""

    def test_returns_name_and_attributes(self):
        result = encode_query("stalled deals in the pipeline")
        assert "name" in result
        assert "attributes" in result
        assert result["name"].startswith("query_")

    def test_description_has_keywords(self):
        result = encode_query("deals with budget objection")
        attrs = result["attributes"]
        assert len(attrs["description"]) > 0
        assert "budget" in attrs["description"]

    def test_keywords_populated(self):
        result = encode_query("champion engagement on enterprise deals")
        attrs = result["attributes"]
        assert len(attrs["keywords"]) > 0
        assert "champion" in attrs["keywords"]

    def test_no_numeric_values_in_query(self):
        result = encode_query("which deals are closing this quarter")
        attrs = result["attributes"]
        # Numeric fields should stay at default (empty string or will be
        # set by the caller — encode_query doesn't set them)
        assert attrs["deal_id"] == ""

    def test_stable_name_for_same_query(self):
        r1 = encode_query("deals gone dark")
        r2 = encode_query("deals gone dark")
        assert r1["name"] == r2["name"]

    def test_different_name_for_different_queries(self):
        r1 = encode_query("deals gone dark")
        r2 = encode_query("strong closing deals")
        assert r1["name"] != r2["name"]

    def test_stop_words_filtered(self):
        result = encode_query("show me the deals in the pipeline")
        attrs = result["attributes"]
        words = attrs["keywords"].split()
        assert "show" not in words
        assert "the" not in words
        assert "me" not in words


class TestEncodeQueryEdgeCases:
    """Edge case tests for encode_query."""

    def test_empty_query(self):
        result = encode_query("")
        assert "name" in result
        assert "attributes" in result

    def test_single_word_query(self):
        result = encode_query("stalled")
        attrs = result["attributes"]
        assert "stalled" in attrs["keywords"]

    def test_long_query(self):
        result = encode_query(
            "show me all the enterprise deals that have been stalled "
            "for more than 30 days with budget objections and no clear "
            "decision maker identified in the pipeline"
        )
        attrs = result["attributes"]
        assert len(attrs["keywords"]) > 0
