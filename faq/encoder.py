"""
Custom encoder for the Glyphh AI FAQ knowledge base model.

Exports:
  ENCODER_CONFIG — EncoderConfig with semantic + context layers
  encode_query(query) — converts NL question to a Concept for similarity search
  entry_to_record(entry) — converts a JSONL entry to an encodable record

Primary matching signal: bag-of-words on the question field. Shared words
between a user question ("how do I install glyphh") and an FAQ entry
("how do I install the Glyphh SDK") drive similarity.

Category inference provides a secondary boost via lexicon encoding.
"""

import hashlib
import re

from glyphh.core.config import (
    EncoderConfig,
    Layer,
    Role,
    Segment,
)

from intent import infer_category, extract_keywords, _preprocess


# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    include_temporal=False,
    layers=[
        Layer(
            name="semantic",
            similarity_weight=0.7,
            segments=[
                Segment(
                    name="identity",
                    roles=[
                        Role(
                            name="question_id",
                            similarity_weight=0.1,
                            key_part=True,
                        ),
                        Role(
                            name="category",
                            similarity_weight=0.6,
                            lexicons=[
                                "getting_started", "sdk", "runtime", "cli",
                                "models", "architecture", "deployment",
                                "pricing", "security", "troubleshooting",
                                "general",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="content",
                    roles=[
                        Role(
                            name="question",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="answer",
                            similarity_weight=0.4,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.3,
            segments=[
                Segment(
                    name="tags",
                    roles=[
                        Role(
                            name="keywords",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# encode_query — NL question → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL question into a Concept-compatible dict."""
    cleaned = _preprocess(query)
    keywords = extract_keywords(cleaned)

    # Use keyword-filtered text as the question signal so stopword-only
    # or single-character queries don't produce spurious matches.
    question_text = keywords if keywords else cleaned
    category = infer_category(cleaned)

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "question_id": "",
            "category": category,
            "question": question_text,
            "answer": "",
            "keywords": keywords,
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL entry → encodable record + metadata
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL entry to an encodable record with metadata."""
    question = entry.get("question", "")
    question_id = entry.get("question_id", "")
    category = entry.get("category", "")
    answer = entry.get("answer", "")
    kw_list = entry.get("keywords", [])
    kw_str = " ".join(kw_list) if isinstance(kw_list, list) else str(kw_list)

    # Auto-generate question_id if not provided
    if not question_id:
        slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:40]
        question_id = f"faq_{slug}"

    # Auto-infer category if not provided
    if not category:
        category = infer_category(question)

    # Preprocess text for consistent BoW encoding
    question_clean = _preprocess(question)
    answer_clean = _preprocess(answer)

    return {
        "concept_text": question,
        "attributes": {
            "question_id": question_id,
            "category": category,
            "question": question_clean,
            "answer": answer_clean,
            "keywords": kw_str,
        },
        "metadata": {
            "answer": answer,
            "category": category,
            "question_id": question_id,
            "original_question": question,
        },
    }
