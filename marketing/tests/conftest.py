"""
Shared fixtures for marketing model tests.

No CV dependencies needed — everything is synthetic data + HDC.
"""

import json
import os
import sys

import pytest

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "glyphh-runtime"))
# Add model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glyphh import Encoder
from glyphh.core.types import Concept
from encoder import ENCODER_CONFIG, encode_query, entry_to_record


@pytest.fixture
def encoder():
    """Create an Encoder from the marketing ENCODER_CONFIG."""
    return Encoder(ENCODER_CONFIG)


@pytest.fixture
def sample_campaign():
    """A sample campaign exemplar."""
    return {
        "entity_type": "campaign",
        "entity_id": "test_campaign_1",
        "goal": "conversion",
        "audience_segment": "technical enterprise",
        "platform": "twitter linkedin blog",
        "content_type": "none",
        "themes": "product launch api integration developer",
        "tone": "professional",
        "keywords": "product launch enterprise api integration developer platform",
        "engagement_metrics": [1250, 340, 89, 45, 156],
        "conversion_metrics": [2100, 180, 42, 95, 12],
        "reach_metrics": [45000, 12000, 3.2, 0.15],
    }


@pytest.fixture
def sample_post():
    """A sample post exemplar."""
    return {
        "entity_type": "post",
        "entity_id": "test_post_1",
        "goal": "awareness",
        "audience_segment": "technical",
        "platform": "twitter",
        "content_type": "social_post",
        "themes": "api launch developer platform announcement",
        "tone": "casual",
        "keywords": "api launch developer platform new feature announcement",
        "engagement_metrics": [450, 180, 42, 22, 85],
        "conversion_metrics": [320, 45, 8, 12, 2],
        "reach_metrics": [18000, 9500, 1.0, 0.0],
    }


@pytest.fixture
def sample_audience():
    """A sample audience exemplar."""
    return {
        "entity_type": "audience",
        "entity_id": "test_audience_1",
        "goal": "none",
        "audience_segment": "technical developer engineering",
        "platform": "twitter linkedin blog",
        "content_type": "none",
        "themes": "developer api code integration sdk",
        "tone": "none",
        "keywords": "developer api code integration sdk documentation engineering",
        "engagement_metrics": [850, 380, 120, 210, 95],
        "conversion_metrics": [620, 180, 35, 85, 28],
        "reach_metrics": [45000, 22000, 3.5, 0.18],
    }


@pytest.fixture
def campaign_glyph(encoder, sample_campaign):
    """Encoded campaign glyph."""
    record = entry_to_record(sample_campaign)
    concept = Concept(name=record["concept_text"], attributes=record["attributes"])
    return encoder.encode(concept)


@pytest.fixture
def post_glyph(encoder, sample_post):
    """Encoded post glyph."""
    record = entry_to_record(sample_post)
    concept = Concept(name=record["concept_text"], attributes=record["attributes"])
    return encoder.encode(concept)


@pytest.fixture
def audience_glyph(encoder, sample_audience):
    """Encoded audience glyph."""
    record = entry_to_record(sample_audience)
    concept = Concept(name=record["concept_text"], attributes=record["attributes"])
    return encoder.encode(concept)


@pytest.fixture
def exemplars():
    """Load all exemplars from data/exemplars.jsonl."""
    exemplar_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "exemplars.jsonl"
    )
    entries = []
    with open(exemplar_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


@pytest.fixture
def all_glyphs(encoder, exemplars):
    """Encode all exemplars into glyphs. Returns list of (entry, glyph) tuples."""
    results = []
    for entry in exemplars:
        record = entry_to_record(entry)
        concept = Concept(name=record["concept_text"], attributes=record["attributes"])
        glyph = encoder.encode(concept)
        results.append((entry, glyph))
    return results
