"""Test that raw user questions match the correct FAQ entries.

Encodes raw questions (no category, no answer) and compares against the
training FAQ data using Pattern A role-level weighted scoring. Verifies
both category-level and question_id-level accuracy.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import encode_query

glyphh = pytest.importorskip("glyphh")

from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity


# Role weights — Pattern A (flat weighted average)
ROLE_WEIGHTS = {
    "question": 1.0,
    "category": 0.6,
    "keywords": 0.8,
    "answer":   0.4,
}

THRESHOLD = 0.35


def _find_best_match(query_text, faq_glyphs, encoder):
    """Encode a raw question and find the best-scoring FAQ entry."""
    record    = encode_query(query_text)
    q_concept = Concept(name=record["name"], attributes=record["attributes"])
    q_glyph   = encoder.encode(q_concept)

    # Flatten roles from query glyph
    q_roles: dict = {}
    for layer in q_glyph.layers.values():
        if layer.name.startswith("_"):
            continue
        for seg in layer.segments.values():
            q_roles.update(seg.roles)

    best_score, best_meta = -1.0, None
    for faq_glyph, meta in faq_glyphs:
        e_roles: dict = {}
        for layer in faq_glyph.layers.values():
            if layer.name.startswith("_"):
                continue
            for seg in layer.segments.values():
                e_roles.update(seg.roles)

        weighted_sum = 0.0
        weight_total = 0.0
        for rname, w in ROLE_WEIGHTS.items():
            if rname in q_roles and rname in e_roles:
                sim = float(cosine_similarity(q_roles[rname].data, e_roles[rname].data))
                weighted_sum += sim * w
                weight_total += w

        score = weighted_sum / weight_total if weight_total > 0 else 0.0
        if score > best_score:
            best_score = score
            best_meta  = meta

    return best_meta, best_score


# ---------------------------------------------------------------------------
# Per-query match tests
# ---------------------------------------------------------------------------

def test_install_question_matches_getting_started(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "install" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )
    assert meta["question_id"] == q["_expected_match"], (
        f"Expected {q['_expected_match']}, got {meta['question_id']} (score={score:.4f})"
    )


def test_glyph_question_matches_sdk(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "glyph" in q["question"].lower() and "encoding" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )


def test_runtime_server_matches_runtime(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "runtime server" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )
    assert meta["question_id"] == q["_expected_match"], (
        f"Expected {q['_expected_match']}, got {meta['question_id']} (score={score:.4f})"
    )


def test_cli_question_matches_cli(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "cli" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )
    assert meta["question_id"] == q["_expected_match"], (
        f"Expected {q['_expected_match']}, got {meta['question_id']} (score={score:.4f})"
    )


def test_custom_model_matches_models(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "custom model" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )
    assert meta["question_id"] == q["_expected_match"], (
        f"Expected {q['_expected_match']}, got {meta['question_id']} (score={score:.4f})"
    )


def test_hdc_question_matches_architecture(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "hyperdimensional" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )
    assert meta["question_id"] == q["_expected_match"], (
        f"Expected {q['_expected_match']}, got {meta['question_id']} (score={score:.4f})"
    )


def test_deploy_heroku_matches_deployment(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "heroku" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )
    assert meta["question_id"] == q["_expected_match"], (
        f"Expected {q['_expected_match']}, got {meta['question_id']} (score={score:.4f})"
    )


def test_pricing_question_matches_pricing(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "pricing" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )
    assert meta["question_id"] == q["_expected_match"], (
        f"Expected {q['_expected_match']}, got {meta['question_id']} (score={score:.4f})"
    )


def test_low_scores_matches_troubleshooting(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "similarity scores" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )
    assert meta["question_id"] == q["_expected_match"], (
        f"Expected {q['_expected_match']}, got {meta['question_id']} (score={score:.4f})"
    )


def test_support_question_matches_general(encoder, faq_glyphs, test_queries):
    q = next(q for q in test_queries if "support" in q["question"].lower())
    meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
    assert meta["category"] == q["_expected_category"], (
        f"Expected {q['_expected_category']}, got {meta['category']} (score={score:.4f})"
    )
    assert meta["question_id"] == q["_expected_match"], (
        f"Expected {q['_expected_match']}, got {meta['question_id']} (score={score:.4f})"
    )


# ---------------------------------------------------------------------------
# Bulk quality checks
# ---------------------------------------------------------------------------

def test_all_queries_match_correct_category(encoder, faq_glyphs, test_queries):
    """All test queries should match the expected category."""
    failures = []
    for q in test_queries:
        meta, score = _find_best_match(q["question"], faq_glyphs, encoder)
        if meta["category"] != q["_expected_category"]:
            failures.append(
                f"  '{q['question']}' → {meta['category']} "
                f"(expected {q['_expected_category']}, score={score:.4f})"
            )
    assert not failures, "Category mismatches:\n" + "\n".join(failures)
