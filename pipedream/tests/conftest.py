"""Shared test fixtures for the Pipedream Action Router model."""

import json
import sys
from pathlib import Path

import pytest

# Ensure model root is importable
MODEL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(MODEL_DIR))

# Add glyphh-runtime to path
RUNTIME_DIR = MODEL_DIR.parent.parent / "glyphh-runtime"
if RUNTIME_DIR.exists():
    sys.path.insert(0, str(RUNTIME_DIR))


@pytest.fixture
def encoder():
    """Create an Encoder from the model's ENCODER_CONFIG."""
    from glyphh.encoder import Encoder
    from encoder import ENCODER_CONFIG
    return Encoder(ENCODER_CONFIG)


@pytest.fixture
def exemplar_glyphs(encoder):
    """Load and encode all exemplars from data/exemplars.jsonl."""
    from glyphh.core.types import Concept
    from encoder import entry_to_record

    exemplar_path = MODEL_DIR / "data" / "exemplars.jsonl"
    if not exemplar_path.exists():
        pytest.skip("No exemplars.jsonl — run discover.py first")

    glyphs = []
    with open(exemplar_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            record = entry_to_record(entry)
            glyph = encoder.encode(Concept(
                name=record["concept_text"],
                attributes=record["attributes"],
            ))
            glyphs.append((glyph, record))

    if not glyphs:
        pytest.skip("No exemplars loaded")

    return glyphs
