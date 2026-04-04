"""
Custom encoder for the deal intelligence model.

Exports:
  ENCODER_CONFIG — EncoderConfig with 4 layers: identity, profile, strategy, metrics
  encode_query(query) — converts NL text to a Concept for similarity search
  entry_to_record(entry) — converts a JSONL exemplar to an encodable record

Architecture:
  Identity layer (0.10): deal_id (key_part) + account_name
    -> Temporal tracking keyed by deal_id
  Profile layer (0.20): industry, company_size, region, source
    -> Firmographic matching for deal context
  Strategy layer (0.30): description (BoW), keywords (BoW), stage, deal_type
    -> Text-based matching for NL queries
  Metrics layer (0.40): deal_value, days_in_stage, activity_count,
                         email_response_rate, meetings_held (THERMOMETER)
    -> Numeric matching for raw deal data

  Two matching paths, one config:
    - Deal records (metrics + profile) -> match on metrics + profile layers
    - NL queries (text, no metrics) -> match on strategy layer
    - Exemplars have BOTH -> matchable from either direction

  Outcome (won/lost/stalled) is metadata, never an encoded attribute.

  Temporal: deal_id is key_part with auto timestamps for pipeline velocity tracking.
"""

import hashlib
import re

from glyphh.core.config import (
    EncoderConfig,
    EncodingStrategy,
    Layer,
    NumericConfig,
    Role,
    Segment,
    TemporalConfig,
)

from intent import extract_keywords, preprocess

# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    temporal_source="auto",
    temporal_config=TemporalConfig(signal_type="auto"),
    layers=[
        # --- Identity layer: deal tracking ---
        Layer(
            name="identity",
            similarity_weight=0.10,
            segments=[
                Segment(
                    name="deal",
                    roles=[
                        Role(
                            name="deal_id",
                            similarity_weight=0.1,
                            key_part=True,
                        ),
                        Role(
                            name="account_name",
                            similarity_weight=0.3,
                        ),
                    ],
                ),
            ],
        ),
        # --- Profile layer: firmographic context ---
        Layer(
            name="profile",
            similarity_weight=0.20,
            segments=[
                Segment(
                    name="firmographic",
                    roles=[
                        Role(
                            name="industry",
                            similarity_weight=1.0,
                        ),
                        Role(
                            name="company_size",
                            similarity_weight=0.8,
                        ),
                        Role(
                            name="region",
                            similarity_weight=0.6,
                        ),
                        Role(
                            name="source",
                            similarity_weight=0.7,
                        ),
                    ],
                ),
            ],
        ),
        # --- Strategy layer: text-based matching for NL queries ---
        Layer(
            name="strategy",
            similarity_weight=0.30,
            segments=[
                Segment(
                    name="context",
                    roles=[
                        Role(
                            name="description",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="keywords",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="stage",
                            similarity_weight=0.9,
                        ),
                        Role(
                            name="deal_type",
                            similarity_weight=0.7,
                        ),
                    ],
                ),
            ],
        ),
        # --- Metrics layer: numeric matching for deal data ---
        Layer(
            name="metrics",
            similarity_weight=0.40,
            segments=[
                Segment(
                    name="pipeline",
                    roles=[
                        Role(
                            name="deal_value",
                            similarity_weight=1.0,
                            numeric_config=NumericConfig(
                                bin_width=25000.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=500000.0,
                            ),
                        ),
                        Role(
                            name="days_in_stage",
                            similarity_weight=0.9,
                            numeric_config=NumericConfig(
                                bin_width=10.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=120.0,
                            ),
                        ),
                        Role(
                            name="activity_count",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=5.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=50.0,
                            ),
                        ),
                        Role(
                            name="email_response_rate",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=15.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=100.0,
                            ),
                        ),
                        Role(
                            name="meetings_held",
                            similarity_weight=0.8,
                            numeric_config=NumericConfig(
                                bin_width=3.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=20.0,
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# encode_query — NL text -> Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL query about deals into a Concept-compatible dict.

    Encodes text into the strategy layer (description + keywords).
    No numeric values — query matching is text-driven.
    Outcome labels are results of similarity, not inputs.
    """
    keywords = extract_keywords(query)

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "deal_id": "",
            "account_name": "",
            "industry": "",
            "company_size": "",
            "region": "",
            "source": "",
            "description": keywords,
            "keywords": keywords,
            "stage": "",
            "deal_type": "",
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL exemplar -> encodable record + metadata
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL exemplar to an encodable record with metadata.

    Attributes encode the deal profile (text + metrics).
    Outcome/risk labels go to metadata only — returned after matching.
    """
    question = entry.get("question", "")
    deal_id = entry.get("deal_id", "")
    kw_list = entry.get("keywords", [])
    kw_str = " ".join(kw_list) if isinstance(kw_list, list) else str(kw_list)

    return {
        "concept_text": question,
        "attributes": {
            "deal_id": deal_id,
            "account_name": entry.get("account_name", ""),
            "industry": entry.get("industry", ""),
            "company_size": entry.get("company_size", ""),
            "region": entry.get("region", ""),
            "source": entry.get("source", ""),
            "description": preprocess(question),
            "keywords": kw_str,
            "stage": entry.get("stage", ""),
            "deal_type": entry.get("deal_type", ""),
            "deal_value": entry.get("deal_value", 0),
            "days_in_stage": entry.get("days_in_stage", 0),
            "activity_count": entry.get("activity_count", 0),
            "email_response_rate": entry.get("email_response_rate", 0),
            "meetings_held": entry.get("meetings_held", 0),
        },
        "metadata": {
            "record_type": entry.get("record_type", "pattern"),
            "outcome": entry.get("outcome", ""),
            "risk_level": entry.get("risk_level", ""),
            "deal_driver": entry.get("deal_driver", ""),
            "recommended_action": entry.get("recommended_action", ""),
            "response": entry.get("response", ""),
            "original_question": question,
            "deal_value": entry.get("deal_value", 0),
            "days_in_stage": entry.get("days_in_stage", 0),
            "activity_count": entry.get("activity_count", 0),
            "email_response_rate": entry.get("email_response_rate", 0),
            "meetings_held": entry.get("meetings_held", 0),
        },
    }
