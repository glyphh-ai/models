"""Shared fixtures for FAQ model tests."""

import json
import sys
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

TESTS_DIR  = Path(__file__).resolve().parent
DATA_DIR   = MODEL_DIR / "data"
CONCEPTS_PATH = TESTS_DIR / "test-concepts.json"


@pytest.fixture(scope="session")
def test_queries():
    """Load raw test queries from test-concepts.json."""
    with open(CONCEPTS_PATH) as f:
        return json.load(f)["queries"]


@pytest.fixture(scope="session")
def encoder_config():
    """Import and return the model's ENCODER_CONFIG."""
    from encoder import ENCODER_CONFIG
    return ENCODER_CONFIG


@pytest.fixture(scope="session")
def encoder(encoder_config):
    """Build a session-scoped Encoder instance."""
    from glyphh import Encoder
    return Encoder(encoder_config)


@pytest.fixture(scope="session")
def faq_glyphs(encoder):
    """Encode all FAQ entries into (glyph, metadata) pairs."""
    from glyphh.core.types import Concept
    from encoder import entry_to_record

    faq_path = DATA_DIR / "exemplars.jsonl"
    glyphs = []
    with open(faq_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json as _json
            entry  = _json.loads(line)
            record = entry_to_record(entry)
            concept = Concept(
                name=record["attributes"]["question_id"],
                attributes=record["attributes"],
            )
            glyph = encoder.encode(concept)
            glyphs.append((glyph, record["metadata"]))
    return glyphs
