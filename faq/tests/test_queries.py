"""Test that encode_query produces correct attributes from NL text."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import encode_query


def test_getting_started_query():
    result = encode_query("how do I install Glyphh")
    assert result["attributes"]["category"] == "getting_started"


def test_sdk_query():
    result = encode_query("what is a Glyph and how does encoding work")
    assert result["attributes"]["category"] == "sdk"


def test_runtime_query():
    result = encode_query("how do I start the runtime server")
    assert result["attributes"]["category"] == "runtime"


def test_cli_query():
    result = encode_query("how do I use glyphh dev and glyphh serve from the command line terminal")
    assert result["attributes"]["category"] == "cli"


def test_models_query():
    result = encode_query("how do I build a custom model with config.yaml manifest.yaml and training data jsonl")
    assert result["attributes"]["category"] == "models"


def test_architecture_query():
    result = encode_query("how does Glyphh compare to RAG and why is there no hallucination")
    assert result["attributes"]["category"] == "architecture"


def test_deployment_query():
    result = encode_query("how do I deploy to Heroku with Docker")
    assert result["attributes"]["category"] == "deployment"


def test_pricing_query():
    result = encode_query("what are the pricing tiers and plans")
    assert result["attributes"]["category"] == "pricing"


def test_security_query():
    result = encode_query("how does API token authentication work")
    assert result["attributes"]["category"] == "security"


def test_troubleshooting_query():
    result = encode_query("I keep getting an error and low similarity scores")
    assert result["attributes"]["category"] == "troubleshooting"


def test_general_query():
    result = encode_query("how do I contact support or get help")
    assert result["attributes"]["category"] == "general"


def test_query_has_empty_question_id():
    """Queries are not stored FAQs — question_id must be empty."""
    result = encode_query("how do I get started")
    assert result["attributes"]["question_id"] == ""


def test_query_has_empty_answer():
    """Queries don't have answers — the model finds them."""
    result = encode_query("what dimensions should I use")
    assert result["attributes"]["answer"] == ""


def test_query_has_stable_name():
    """Same query text must produce the same concept name."""
    q  = "how do I install Glyphh"
    r1 = encode_query(q)
    r2 = encode_query(q)
    assert r1["name"] == r2["name"]


def test_question_is_lowercased():
    """Question attribute must be lowercase for consistent BoW encoding."""
    result = encode_query("How Do I Reset My PASSWORD?")
    assert result["attributes"]["question"] == result["attributes"]["question"].lower()


def test_keywords_are_non_empty():
    """extract_keywords should produce at least some keywords."""
    result = encode_query("how do I deploy Glyphh to production with Docker")
    assert result["attributes"]["keywords"].strip() != ""


def test_keywords_exclude_common_stop_words():
    """Common stop words should not dominate the keywords field."""
    result = encode_query("how do I deploy Glyphh to production with Docker")
    kw = result["attributes"]["keywords"].split()
    assert any(w in kw for w in ("deploy", "glyphh", "production", "docker")), (
        f"Expected meaningful keywords in: {kw}"
    )
