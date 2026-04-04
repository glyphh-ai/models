"""Shared fixtures for deal intelligence model tests."""

import json
import sys
from pathlib import Path

import pytest

# Ensure the model directory is importable
MODEL_DIR = Path(__file__).resolve().parent.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

TESTS_DIR = Path(__file__).resolve().parent
CONCEPTS_PATH = TESTS_DIR / "test-concepts.json"


@pytest.fixture(scope="session")
def test_deals():
    """Load raw deal data from test-concepts.json."""
    with open(CONCEPTS_PATH) as f:
        return json.load(f)["deals"]


@pytest.fixture(scope="session")
def encoder_config():
    """Import and return the model's ENCODER_CONFIG."""
    from encoder import ENCODER_CONFIG
    return ENCODER_CONFIG


@pytest.fixture(scope="session")
def expected_won(test_deals):
    """Deals we expect the model to match as won."""
    return [d for d in test_deals if d["_expected_outcome"] == "won"]


@pytest.fixture(scope="session")
def expected_lost(test_deals):
    """Deals we expect the model to match as lost."""
    return [d for d in test_deals if d["_expected_outcome"] == "lost"]


@pytest.fixture(scope="session")
def expected_stalled(test_deals):
    """Deals we expect the model to match as stalled."""
    return [d for d in test_deals if d["_expected_outcome"] == "stalled"]


@pytest.fixture(scope="session")
def expected_pending(test_deals):
    """Deals we expect the model to match as pending."""
    return [d for d in test_deals if d["_expected_outcome"] == "pending"]
