"""
Encoder for the Pipedream Action Router model.

Exports:
  ENCODER_CONFIG — EncoderConfig with intent + semantics layers
  encode_query(query) — converts NL text to a Concept dict for similarity search
  entry_to_record(entry) — converts a JSONL exemplar entry to a build record
  assess_query(query) — slot completeness check for disambiguation
  extract_app(query) — extracts app slug from query text (for pre-filtering)

Architecture:
  Two-stage routing:
  1. App extraction: string-match app name from query → filter to that app
  2. HDC similarity: intent + semantics layers match within app

  Intent layer (0.4): action (lexicon) + target (lexicon) + domain (lexicon)
  Semantics layer (0.6): description (BoW) + keywords (BoW)
"""

import hashlib
import json
import re
from pathlib import Path

from glyphh.core.config import (
    EncoderConfig,
    Layer,
    Role,
    Segment,
)

from intent import extract_intent, assess_query, set_extract_app  # noqa: F401 — re-export

# ---------------------------------------------------------------------------
# Build app name lookup from exemplars (loaded once at import time)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"

# Build reverse lookup: lowercase name/alias → app_slug
# Strategy: when multiple slugs claim the same key, prefer the shortest slug
# (the "base" app, not _oauth/_sandbox/_api_key variants).
_APP_LOOKUP: dict[str, str] = {}

def _set_lookup(key: str, slug: str) -> None:
    """Set lookup entry, preferring shorter (base) slugs over longer (variant) slugs."""
    key = key.lower()
    if not key:
        return
    existing = _APP_LOOKUP.get(key)
    if existing is None:
        _APP_LOOKUP[key] = slug
    elif len(slug) < len(existing):
        # Shorter slug wins (base app over variant)
        _APP_LOOKUP[key] = slug
    # If same length or new is longer, keep existing

for _f in _DATA_DIR.glob("exemplars*.jsonl"):
    with open(_f) as _fh:
        for _line in _fh:
            _line = _line.strip()
            if not _line:
                continue
            _entry = json.loads(_line)
            _slug = _entry.get("app_slug", "")
            _name = _entry.get("app_name", "")
            if _slug:
                _set_lookup(_slug, _slug)
                # Register space-separated version for multi-word matching
                _set_lookup(_slug.replace("_", " "), _slug)
                if _slug.startswith("_"):
                    _set_lookup(_slug[1:], _slug)
                    _set_lookup(_slug[1:].replace("_", " "), _slug)
            if _name:
                _set_lookup(_name, _slug)
                # Also register with parens stripped: "Linear (API key)" → "linear api key"
                _clean = _name.replace("(", "").replace(")", "").strip()
                _clean = re.sub(r"\s+", " ", _clean)
                if _clean.lower() != _name.lower():
                    _set_lookup(_clean, _slug)

# Also load APP_ALIASES from per-app intent files
_APPS_DIR = Path(__file__).parent / "apps"
if _APPS_DIR.exists():
    for _app_dir in sorted(_APPS_DIR.iterdir()):
        _intent_file = _app_dir / "intent.py"
        if not _intent_file.exists():
            continue
        try:
            _content = _intent_file.read_text()
            _m = re.search(r'APP_ALIASES\s*=\s*\[([^\]]+)\]', _content)
            if _m:
                _aliases = [a.strip().strip('"\'') for a in _m.group(1).split(",")]
                for _alias in _aliases:
                    if _alias:
                        _set_lookup(_alias, _app_dir.name)
        except Exception:
            pass

# Sort by length (longest first) for greedy matching
_APP_KEYS_BY_LENGTH = sorted(_APP_LOOKUP.keys(), key=len, reverse=True)

# Common words that happen to be app names — skip unless after preposition
_COMMON_WORDS = {
    "message", "messages", "contact", "contacts", "task", "tasks",
    "deal", "deals", "note", "notes", "form", "forms", "post",
    "file", "files", "event", "events", "page", "pages", "track",
    "search", "reply", "share", "comment", "record", "table",
    "order", "product", "report", "channel",
    "send", "create", "get", "list", "update", "delete", "add",
    "remove", "set", "new", "make", "find", "view", "open",
    # Generic suffixes in variant slugs — prevent false matches
    "app", "api", "bot", "admin", "oauth", "rest", "sandbox",
    "staging", "key", "keys", "dev", "developer", "pro",
}

# ---------------------------------------------------------------------------
# App family expansion: base slug → all variant slugs
# ---------------------------------------------------------------------------

_ALL_APP_SLUGS = set(_APP_LOOKUP.values())


def expand_app_family(slug: str) -> list[str]:
    """Expand slug to include all variant slugs sharing a common base prefix.

    Forward:  "shopify" → ["shopify", "shopify_developer_app", "shopify_partner"]
    Reverse:  "slack_v2" → ["slack_bot", "slack_v2"]  (common prefix "slack")
    """
    family = {slug}
    # Forward: slug is base → include variants
    for s in _ALL_APP_SLUGS:
        if s != slug and s.startswith(slug + "_"):
            family.add(s)
    # Reverse: slug is variant → find common-prefix peers
    if "_" in slug:
        parts = slug.split("_")
        for prefix_len in range(len(parts) - 1, 0, -1):
            prefix = "_".join(parts[:prefix_len])
            if len(prefix) < 3:
                continue
            peers = [s for s in _ALL_APP_SLUGS
                     if s == prefix or s.startswith(prefix + "_")]
            if len(peers) > 1:
                family.update(peers)
                break
    return sorted(family)


def extract_app(query: str) -> str | None:
    """Extract app slug from query by matching known app names/aliases.

    Returns the matched app_slug, or None if no app found.

    Strategy:
    1. Look for app name after prepositions (on/in/with/via/using)
    2. Look for app name at start of query (app-first pattern)
    3. Fall back to longest non-common-word match
    """
    q = query.lower()
    tokens = q.split()

    # 1. Preposition-based: "send message on Slack" → "slack"
    _PREPS = {"on", "in", "with", "via", "using"}
    for i, tok in enumerate(tokens):
        if tok in _PREPS and i + 1 < len(tokens):
            for width in range(min(5, len(tokens) - i - 1), 0, -1):
                candidate = " ".join(tokens[i + 1 : i + 1 + width])
                # Skip single-word common words (prevents "on rest" → shorten_rest)
                if width == 1 and candidate in _COMMON_WORDS:
                    continue
                if candidate in _APP_LOOKUP:
                    return _APP_LOOKUP[candidate]

    # 2. App-first: "Slack send message"
    for width in range(min(5, len(tokens)), 0, -1):
        candidate = " ".join(tokens[:width])
        if candidate in _APP_LOOKUP and candidate not in _COMMON_WORDS:
            return _APP_LOOKUP[candidate]

    # 3. Fallback: longest non-common-word match
    for key in _APP_KEYS_BY_LENGTH:
        if len(key) < 3 or key in _COMMON_WORDS:
            continue
        idx = q.find(key)
        if idx >= 0:
            before = q[idx - 1] if idx > 0 else " "
            after = q[idx + len(key)] if idx + len(key) < len(q) else " "
            if not before.isalnum() and not after.isalnum():
                return _APP_LOOKUP[key]

    return None


# Register extract_app with intent.py so assess_query() can use it
# without a circular import (intent → encoder).
set_extract_app(extract_app)


# ---------------------------------------------------------------------------
# ENCODER_CONFIG — Two-layer: intent + semantics
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="intent",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="action",
                    roles=[
                        Role(
                            name="action",
                            similarity_weight=1.0,
                            lexicons=[
                                "send", "search", "create", "get", "list",
                                "update", "delete", "reply", "share", "upload",
                                "track", "add", "remove", "set", "subscribe",
                                "unsubscribe", "charge", "refund", "cancel",
                                "invite", "assign", "comment", "export",
                                "import", "sync", "trigger", "schedule",
                                "archive", "none",
                            ],
                        ),
                        Role(
                            name="target",
                            similarity_weight=0.7,
                            lexicons=[
                                "message", "channel", "thread", "email",
                                "contact", "customer", "deal", "ticket",
                                "task", "file", "folder", "spreadsheet",
                                "event", "invoice", "payment", "subscription",
                                "campaign", "audience", "order", "product",
                                "repo", "pull_request", "commit", "deploy",
                                "record", "table", "post", "comment",
                                "employee", "applicant", "user", "page",
                                "note", "form", "report", "metric",
                                "workflow", "pipeline",
                                # Crypto / trading
                                "trade", "ticker", "candle", "orderbook",
                                "depth", "rate", "fee", "fill", "contract",
                                "position", "balance", "price", "symbol",
                                # Discord / community
                                "role", "invite", "reaction", "emoji",
                                "nickname", "guild", "member",
                                # General
                                "tag", "label", "template", "draft",
                                "attachment", "status", "config",
                                "none",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="scope",
                    roles=[
                        Role(
                            name="domain",
                            similarity_weight=0.8,
                            lexicons=[
                                "messaging", "email", "crm", "payments",
                                "calendar", "files", "tickets", "analytics",
                                "developer", "social", "marketing",
                                "database", "ai", "productivity",
                                "hr", "ecommerce", "crypto", "none",
                            ],
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="semantics",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="text",
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
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# encode_query — NL text → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL query into a Concept-compatible dict.

    Uses local intent.py for action/target/domain/keyword extraction.
    Returns a dict with 'name', 'attributes', and '_filters' keys.
    The '_filters' dict is used by the runtime to narrow pgvector search
    to the matched app's exemplars (e.g. app_slug metadata filter).
    """
    extracted = extract_intent(query)

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    # Extract app slug for metadata filtering (Stage 1 pre-filter)
    matched_app = extract_app(query)
    filters = {}
    if matched_app:
        filters["app_slug"] = matched_app

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "action": extracted["action"],
            "target": extracted["target"],
            "domain": extracted["domain"],
            "description": extracted["keywords"],
            "keywords": extracted["keywords"],
        },
        "_filters": filters,
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL exemplar → build record
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL exemplar entry into a record for building/encoding.

    Expects entry keys from auto-generated exemplars:
      action_key, action, target, domain, app_slug, keywords, description
    Also supports metadata fields: app_name, version, configured_props

    Returns: {"concept_text": str, "attributes": dict, "metadata": dict}
    """
    keywords = entry.get("keywords", [])
    if isinstance(keywords, list):
        keywords = " ".join(keywords)

    description = entry.get("description", keywords)
    action_key = entry.get("action_key", entry.get("tool_id", "unknown"))

    return {
        "concept_text": action_key,
        "attributes": {
            "action": entry.get("action", "none"),
            "target": entry.get("target", "none"),
            "domain": entry.get("domain", "none"),
            "description": description,
            "keywords": keywords,
        },
        "metadata": {
            "action_key": action_key,
            "action_name": entry.get("action_name", ""),
            "app_slug": entry.get("app_slug", ""),
            "app_name": entry.get("app_name", ""),
            "version": entry.get("version", ""),
            "configured_props": entry.get("configured_props", {}),
        },
    }
