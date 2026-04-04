"""Test that the encoder config is valid and roles encode correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG, entry_to_record


# ---------------------------------------------------------------------------
# Config structure
# ---------------------------------------------------------------------------

def test_config_has_required_fields(encoder_config):
    assert encoder_config.dimension == 2000
    assert encoder_config.seed == 42
    assert encoder_config.include_temporal is False
    assert len(encoder_config.layers) >= 1


def test_semantic_layer_exists(encoder_config):
    layer_names = [l.name for l in encoder_config.layers]
    assert "semantic" in layer_names

    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    seg_names = [s.name for s in semantic.segments]
    assert "identity" in seg_names
    assert "content" in seg_names


def test_context_layer_exists(encoder_config):
    layer_names = [l.name for l in encoder_config.layers]
    assert "context" in layer_names

    context  = next(l for l in encoder_config.layers if l.name == "context")
    all_roles = [r.name for s in context.segments for r in s.roles]
    assert "keywords" in all_roles


def test_question_id_is_key_part(encoder_config):
    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    identity = next(s for s in semantic.segments if s.name == "identity")
    qid_role = next(r for r in identity.roles if r.name == "question_id")
    assert qid_role.key_part is True


def test_question_role_has_highest_weight(encoder_config):
    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    content  = next(s for s in semantic.segments if s.name == "content")
    q_role   = next(r for r in content.roles if r.name == "question")
    a_role   = next(r for r in content.roles if r.name == "answer")
    assert q_role.similarity_weight > a_role.similarity_weight


def test_question_uses_bag_of_words(encoder_config):
    """Primary matching signal must use BoW so partial-word overlap scores."""
    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    content  = next(s for s in semantic.segments if s.name == "content")
    q_role   = next(r for r in content.roles if r.name == "question")
    assert q_role.text_encoding == "bag_of_words"


def test_answer_uses_bag_of_words(encoder_config):
    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    content  = next(s for s in semantic.segments if s.name == "content")
    a_role   = next(r for r in content.roles if r.name == "answer")
    assert a_role.text_encoding == "bag_of_words"


def test_keywords_uses_bag_of_words(encoder_config):
    context  = next(l for l in encoder_config.layers if l.name == "context")
    kw_role  = next(
        r for s in context.segments for r in s.roles if r.name == "keywords"
    )
    assert kw_role.text_encoding == "bag_of_words"


def test_category_has_valid_lexicons(encoder_config):
    """Category lexicons must be the 11 Glyphh AI FAQ categories."""
    semantic = next(l for l in encoder_config.layers if l.name == "semantic")
    identity = next(s for s in semantic.segments if s.name == "identity")
    cat_role = next(r for r in identity.roles if r.name == "category")
    expected = {
        "getting_started", "sdk", "runtime", "cli", "models",
        "architecture", "deployment", "pricing", "security",
        "troubleshooting", "general",
    }
    assert expected.issubset(set(cat_role.lexicons or []))


# ---------------------------------------------------------------------------
# entry_to_record
# ---------------------------------------------------------------------------

def test_entry_to_record_produces_all_attributes():
    entry = {
        "question_id": "test_q",
        "category":    "sdk",
        "question":    "what is a Glyph",
        "answer":      "A Glyph is a high-dimensional binary vector.",
        "keywords":    ["glyph", "vector", "encoding"],
    }
    record = entry_to_record(entry)
    attrs  = record["attributes"]
    for key in ["question_id", "category", "question", "answer", "keywords"]:
        assert key in attrs, f"Missing attribute: {key}"


def test_entry_to_record_auto_generates_id():
    entry = {
        "question": "how do I deploy to Heroku",
        "answer":   "Use the Heroku deploy button or git push.",
        "category": "deployment",
    }
    record = entry_to_record(entry)
    assert record["attributes"]["question_id"].startswith("faq_")


def test_entry_to_record_auto_infers_category():
    entry = {
        "question": "how do I install the Glyphh SDK",
        "answer":   "pip install glyphh",
    }
    record = entry_to_record(entry)
    assert record["attributes"]["category"] == "getting_started"


def test_entry_to_record_lowercases_question():
    """Questions must be lowercase for consistent BoW encoding."""
    entry = {
        "question_id": "q1",
        "question":    "What Is An EncoderConfig?",
        "answer":      "An EncoderConfig defines layers and roles.",
        "category":    "sdk",
    }
    record = entry_to_record(entry)
    assert record["attributes"]["question"] == record["attributes"]["question"].lower()


def test_test_queries_have_no_answers(test_queries):
    for q in test_queries:
        assert "answer" not in q, "Test data should be raw questions only"
        assert "question" in q
