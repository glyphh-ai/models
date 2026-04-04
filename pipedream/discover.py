#!/usr/bin/env python3
"""
Pipedream Registry Discovery — Per-App Architecture

Pulls the app + action catalog from Pipedream Connect API and generates
per-app exemplars, tests, and intent scaffolds under apps/{slug}/.

Usage:
    python discover.py                          # full discovery
    python discover.py --apps slack_bot,github  # specific apps only
    python discover.py --limit 100              # limit apps
    python discover.py --dry-run                # preview without writing

Requires env vars:
    PIPEDREAM_CLIENT_ID
    PIPEDREAM_CLIENT_SECRET
    PIPEDREAM_PROJECT_ID
    PIPEDREAM_ENVIRONMENT (default: development)

Output per app:
    apps/{slug}/exemplars.jsonl    — 2-3 varied exemplars per action
    apps/{slug}/tests.jsonl        — 10-20+ exhaustive NL test queries per action
    apps/{slug}/intent.py          — auto-scaffolded (only if not already present)

Aggregated output:
    data/exemplars.jsonl           — all per-app exemplars concatenated
    data/test_queries.jsonl        — all per-app tests concatenated
    data/registry_cache.json       — raw registry data
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

MODEL_DIR = Path(__file__).parent
APPS_DIR = MODEL_DIR / "apps"
DATA_DIR = MODEL_DIR / "data"
OUTPUT_EXEMPLARS = DATA_DIR / "exemplars.jsonl"
OUTPUT_CACHE = DATA_DIR / "registry_cache.json"
OUTPUT_TEST_QUERIES = DATA_DIR / "test_queries.jsonl"

# ---------------------------------------------------------------------------
# Pipedream API client
# ---------------------------------------------------------------------------

BASE_URL = "https://api.pipedream.com/v1"


class PipedreamClient:
    """Minimal Pipedream Connect API client for registry discovery."""

    def __init__(self):
        self.client_id = os.environ.get("PIPEDREAM_CLIENT_ID")
        self.client_secret = os.environ.get("PIPEDREAM_CLIENT_SECRET")
        self.project_id = os.environ.get("PIPEDREAM_PROJECT_ID")
        self.environment = os.environ.get("PIPEDREAM_ENVIRONMENT", "development")

        if not all([self.client_id, self.client_secret, self.project_id]):
            print("Error: Missing Pipedream env vars.")
            print("  Required: PIPEDREAM_CLIENT_ID, PIPEDREAM_CLIENT_SECRET, PIPEDREAM_PROJECT_ID")
            sys.exit(1)

        self._access_token: str | None = None

    def _get_token(self) -> str:
        """Obtain OAuth access token via client credentials."""
        if self._access_token:
            return self._access_token

        resp = requests.post(
            f"{BASE_URL}/oauth/token",
            json={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]
        return self._access_token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "X-PD-Environment": self.environment,
        }

    def list_apps(self, query: str = "", limit: int = 200) -> list[dict]:
        """List available apps from the registry."""
        apps = []
        params: dict[str, Any] = {"limit": min(limit, 200)}
        if query:
            params["q"] = query

        url = f"{BASE_URL}/connect/apps"
        while url and len(apps) < limit:
            resp = requests.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            data = resp.json()
            apps.extend(data.get("data", []))

            page_info = data.get("page_info", {})
            end_cursor = page_info.get("end_cursor")
            if end_cursor and len(data.get("data", [])) > 0:
                params["after"] = end_cursor
            else:
                break

        return apps[:limit]

    def list_actions(self, app_slug: str) -> list[dict]:
        """List available actions for a specific app."""
        actions = []
        params: dict[str, Any] = {"app": app_slug}
        url = f"{BASE_URL}/connect/{self.project_id}/actions"

        while url:
            resp = requests.get(url, headers=self._headers(), params=params)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            actions.extend(data.get("data", []))

            page_info = data.get("page_info", {})
            end_cursor = page_info.get("end_cursor")
            if end_cursor and len(data.get("data", [])) > 0:
                params["after"] = end_cursor
            else:
                break

        return actions

    def get_action_props(self, action_key: str) -> list[dict]:
        """Get configurable properties for a single action.

        Returns a list of prop dicts, each with at least 'name' and optionally
        'type', 'required', 'description', 'options', etc.

        Returns empty list on error (non-fatal — props are best-effort).
        """
        try:
            resp = requests.get(
                f"{BASE_URL}/connect/{self.project_id}/actions/{action_key}",
                headers=self._headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("configurable_props", [])
            return []
        except requests.RequestException:
            return []


# ---------------------------------------------------------------------------
# Props cache — avoids re-fetching 10K+ action prop schemas on every run
# ---------------------------------------------------------------------------

PROPS_CACHE_PATH = DATA_DIR / "props_cache.json"


def _load_props_cache() -> dict[str, list[dict]]:
    """Load cached action props from disk. Returns {action_key: [prop, ...]}."""
    if PROPS_CACHE_PATH.exists():
        try:
            return json.loads(PROPS_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_props_cache(cache: dict[str, list[dict]]) -> None:
    """Persist props cache to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROPS_CACHE_PATH.write_text(json.dumps(cache, separators=(",", ":")))


def _simplify_props(raw_props: list[dict]) -> dict[str, str]:
    """Convert raw prop list to simple {name: type} dict for exemplar storage.

    Extracts name and type from each prop. Skips internal/app-auth props.
    """
    result = {}
    for prop in raw_props:
        name = prop.get("name", "")
        if not name or name.startswith("$") or name == "app":
            continue
        # Determine type — Pipedream uses various formats
        ptype = prop.get("type", "string")
        if isinstance(ptype, dict):
            ptype = ptype.get("type", "string")
        # Normalize to simple type names
        if ptype in ("$.app", "app"):
            continue
        result[name] = str(ptype).lower()
    return result


def fetch_action_props(
    client: "PipedreamClient",
    action_keys: list[str],
    cache: dict[str, list[dict]],
    batch_size: int = 50,
    delay: float = 1.0,
) -> dict[str, list[dict]]:
    """Fetch props for actions not already cached, with throttling.

    Args:
        client: Pipedream API client
        action_keys: List of action keys to fetch
        cache: Existing props cache (mutated in place)
        batch_size: Number of requests between saves/sleeps
        delay: Seconds to sleep between batches

    Returns:
        Updated cache dict
    """
    missing = [k for k in action_keys if k not in cache]
    if not missing:
        print(f"  Props cache: all {len(action_keys)} actions cached")
        return cache

    print(f"  Fetching props for {len(missing)} actions "
          f"({len(action_keys) - len(missing)} cached)...")

    errors = 0
    for i, key in enumerate(missing):
        raw_props = client.get_action_props(key)
        cache[key] = raw_props

        if not raw_props:
            errors += 1

        # Progress + throttling
        if (i + 1) % batch_size == 0:
            _save_props_cache(cache)
            pct = (i + 1) / len(missing) * 100
            print(f"    {i + 1}/{len(missing)} ({pct:.0f}%) — "
                  f"{errors} errors, saving cache...")
            time.sleep(delay)

    # Final save
    _save_props_cache(cache)
    print(f"  Props fetched: {len(missing) - errors} ok, {errors} errors")
    return cache


# ---------------------------------------------------------------------------
# Per-app intent loading
# ---------------------------------------------------------------------------

def _load_app_intent(app_slug: str) -> dict:
    """Load per-app intent extensions from apps/{slug}/intent.py.

    Returns dict with keys: aliases, domain, target_overrides, action_synonyms.
    Falls back to defaults if no per-app intent exists.
    """
    intent_path = APPS_DIR / app_slug / "intent.py"
    defaults = {
        "aliases": [],
        "domain": "",
        "target_overrides": {},
        "action_synonyms": {},
    }

    if not intent_path.exists():
        return defaults

    spec = importlib.util.spec_from_file_location(f"apps.{app_slug}.intent", intent_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return defaults

    return {
        "aliases": getattr(mod, "APP_ALIASES", []),
        "domain": getattr(mod, "APP_DOMAIN", ""),
        "target_overrides": getattr(mod, "TARGET_OVERRIDES", {}),
        "action_synonyms": getattr(mod, "ACTION_SYNONYMS", {}),
    }


# ---------------------------------------------------------------------------
# Exemplar generation — multiple per action
# ---------------------------------------------------------------------------

# Maps Pipedream action name patterns → canonical actions
_ACTION_NAME_TO_VERB: list[tuple[str, str]] = [
    ("send", "send"), ("post", "send"), ("notify", "send"),
    ("create", "create"), ("add", "add"), ("new", "create"),
    ("insert", "create"), ("register", "create"),
    ("get", "get"), ("fetch", "get"), ("retrieve", "get"),
    ("read", "get"), ("find", "search"), ("search", "search"),
    ("list", "list"), ("query", "get"),
    ("update", "update"), ("edit", "update"), ("modify", "update"),
    ("set", "set"), ("patch", "update"),
    ("delete", "delete"), ("remove", "remove"),
    ("upload", "upload"), ("share", "share"),
    ("export", "export"), ("download", "export"),
    ("import", "import"),
    ("subscribe", "subscribe"), ("unsubscribe", "unsubscribe"),
    ("charge", "charge"), ("refund", "refund"),
    ("cancel", "cancel"),
    ("invite", "invite"), ("assign", "assign"),
    ("comment", "comment"),
    ("trigger", "trigger"), ("run", "trigger"), ("execute", "trigger"),
    ("schedule", "schedule"), ("book", "schedule"),
    ("sync", "sync"), ("connect", "sync"),
    ("archive", "archive"), ("close", "archive"),
    ("track", "track"), ("log", "track"),
    ("place", "create"), ("rename", "update"), ("change", "update"),
    ("replace", "update"), ("lookup", "get"),
]

# Maps Pipedream app categories → domains
_CATEGORY_TO_DOMAIN: dict[str, str] = {
    "communication": "messaging",
    "messaging": "messaging",
    "email": "email",
    "crm": "crm",
    "customer relationship management": "crm",
    "payment": "payments",
    "payments": "payments",
    "finance": "payments",
    "calendar": "calendar",
    "scheduling": "calendar",
    "file storage": "files",
    "storage": "files",
    "cloud storage": "files",
    "project management": "tickets",
    "issue tracking": "tickets",
    "task management": "tickets",
    "analytics": "analytics",
    "developer tools": "developer",
    "developer": "developer",
    "devops": "developer",
    "social media": "social",
    "social": "social",
    "marketing": "marketing",
    "email marketing": "marketing",
    "database": "database",
    "databases": "database",
    "artificial intelligence": "ai",
    "ai": "ai",
    "machine learning": "ai",
    "productivity": "productivity",
    "human resources": "hr",
    "hr": "hr",
    "e-commerce": "ecommerce",
    "ecommerce": "ecommerce",
    "commerce": "ecommerce",
    "cryptocurrency": "crypto",
    "crypto": "crypto",
    "blockchain": "crypto",
    "trading": "crypto",
}

# Well-known app slug → domain overrides
_APP_DOMAIN_OVERRIDES: dict[str, str] = {
    "slack": "messaging", "slack_bot": "messaging",
    "discord": "messaging", "microsoft_teams": "messaging",
    "telegram_bot_api": "messaging", "whatsapp_business": "messaging",
    "gmail": "email", "outlook": "email", "sendgrid": "email",
    "mailchimp": "marketing", "constant_contact": "marketing",
    "salesforce_rest_api": "crm", "hubspot": "crm", "pipedrive": "crm",
    "zoho_crm": "crm",
    "stripe": "payments", "paypal": "payments", "square": "payments",
    "google_calendar": "calendar", "calendly": "calendar",
    "google_drive": "files", "dropbox": "files", "box": "files",
    "onedrive": "files",
    "jira": "tickets", "linear_app": "tickets", "asana": "tickets",
    "trello": "tickets", "clickup": "tickets",
    "google_analytics": "analytics", "segment": "analytics",
    "amplitude": "analytics", "mixpanel": "analytics",
    "github": "developer", "gitlab": "developer", "bitbucket": "developer",
    "twitter": "social", "facebook_pages": "social",
    "linkedin": "social", "instagram_business": "social",
    "airtable": "database", "supabase": "database",
    "notion": "productivity", "google_sheets": "productivity",
    "openai": "ai",
    "shopify": "ecommerce", "woocommerce": "ecommerce",
    "bamboohr": "hr", "gusto": "hr",
    "twilio": "messaging",
    "bitget": "crypto", "binance": "crypto", "bybit": "crypto",
    "coinbase": "crypto", "kraken": "crypto",
}


def _extract_action_verb(action_name: str) -> str:
    """Extract canonical action verb from a Pipedream action name."""
    name_lower = action_name.lower().replace("-", " ").replace("_", " ")
    words = name_lower.split()
    for word in words:
        for pattern, verb in _ACTION_NAME_TO_VERB:
            if word == pattern or word.startswith(pattern):
                return verb
    return "none"


def _extract_target(action_name: str, action_key: str,
                    target_overrides: dict[str, str] | None = None) -> str:
    """Extract target noun from action name/key, with optional per-app overrides.

    Scans forward (after the verb) to find the primary target.
    "List Members in Channel" → "user" (members), not "channel".
    """
    from intent import _TARGET_MAP

    # Merge base map with per-app overrides
    target_map = dict(_TARGET_MAP)
    if target_overrides:
        target_map.update(target_overrides)

    # Try action name first (forward scan, skip the verb word)
    name_words = action_name.lower().split()
    for word in name_words[1:]:  # Skip first word (verb)
        target = target_map.get(word)
        if target:
            return target

    # Fall back to action key parts (forward scan, skip app prefix)
    parts = action_key.lower().replace("_", "-").split("-")
    # Find the verb position and scan after it
    verb_idx = -1
    for idx, part in enumerate(parts):
        for pattern, _ in _ACTION_NAME_TO_VERB:
            if part == pattern or part.startswith(pattern):
                verb_idx = idx
                break
        if verb_idx >= 0:
            break
    search_parts = parts[verb_idx + 1:] if verb_idx >= 0 else parts
    for part in search_parts:
        target = target_map.get(part)
        if target:
            return target

    return "none"


def _infer_domain(app: dict, app_intent: dict | None = None) -> str:
    """Infer domain from app metadata, with per-app intent override."""
    # Per-app intent overrides everything
    if app_intent and app_intent.get("domain"):
        return app_intent["domain"]

    slug = app.get("name_slug", "")
    if slug in _APP_DOMAIN_OVERRIDES:
        return _APP_DOMAIN_OVERRIDES[slug]

    categories = app.get("categories", [])
    for cat in categories:
        domain = _CATEGORY_TO_DOMAIN.get(cat.lower())
        if domain:
            return domain

    return "none"


def _get_app_aliases(app: dict, app_intent: dict | None = None) -> list[str]:
    """Get all name aliases for an app (for keywords and test generation).

    Only includes full slug and full name variants — NOT individual word tokens,
    which can collide with other app slugs (e.g., 'discord' from 'discord_bot'
    would incorrectly match the separate 'discord' app).
    """
    aliases = set()

    slug = app.get("name_slug", "")
    name = app.get("name", slug)

    # Full slug (exact)
    aliases.add(slug.lower())
    # Slug with underscores→spaces (for NL matching)
    aliases.add(slug.lower().replace("_", " "))

    # Derive names from registry name
    full_name = name.replace("(", "").replace(")", "").strip()
    full_name = re.sub(r"\s+", " ", full_name)
    clean_name = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
    base_slug_candidate = clean_name.lower().replace(" ", "_")

    # Check if this is a variant app (slug extends a base that's also an app)
    # e.g., "shopify_developer_app" → base "shopify" exists → variant
    # Also handle mismatched underscores: "launch_darkly_oauth" base "launchdarkly"
    slug_no_sep = slug.replace("_", "")
    base_no_sep = base_slug_candidate.replace("_", "")
    is_variant = ("_" in slug and base_slug_candidate != slug
                  and (slug.startswith(base_slug_candidate + "_")
                       or (slug_no_sep.startswith(base_no_sep) and len(slug_no_sep) > len(base_no_sep))))

    if is_variant:
        # Variant app: only add full_name if it has EXTRA tokens beyond clean_name
        # "Linear (API key)" → full="Linear API key", clean="Linear" → add "linear api key" (distinctive)
        # "Shopify" → full="Shopify", clean="Shopify" → DON'T add (ambiguous)
        if full_name.lower() != clean_name.lower():
            aliases.add(full_name.lower())
    else:
        # Non-variant: add both full and clean names
        aliases.add(full_name.lower())
        if clean_name.lower() != full_name.lower():
            aliases.add(clean_name.lower())

    # Per-app intent aliases — filter out generic and ambiguous entries
    # For variant apps, also filter aliases that match the base app's name/slug
    base_names = set()
    if is_variant:
        base_names.add(base_slug_candidate)
        base_names.add(clean_name.lower())
        base_names.add(base_no_sep)  # "launchdarkly" for launch_darkly_oauth

    if app_intent and app_intent.get("aliases"):
        for a in app_intent["aliases"]:
            if not a:
                continue
            a_lower = a.lower().strip()
            # Skip single-word generic aliases
            if " " not in a_lower and a_lower in _GENERIC_ALIAS_WORDS:
                continue
            # Skip aliases that match the base app name (ambiguous for variants)
            if is_variant and a_lower in base_names:
                continue
            aliases.add(a_lower)

    return sorted(aliases)


# Single words that should NEVER be standalone app aliases
_GENERIC_ALIAS_WORDS = {
    # Verbs / actions
    "send", "create", "get", "list", "update", "delete", "add",
    "remove", "set", "new", "make", "find", "view", "open", "search",
    "reply", "share", "comment", "track", "post",
    # Common object nouns
    "message", "messages", "contact", "contacts", "task", "tasks",
    "deal", "deals", "note", "notes", "form", "forms", "file", "files",
    "event", "events", "page", "pages", "record", "table", "order",
    "product", "report", "pipeline", "channel",
    # Generic slug suffixes
    "app", "api", "bot", "admin", "oauth", "rest", "sandbox",
    "staging", "key", "keys", "dev", "developer", "pro",
    "v2", "v3", "v4", "beta", "test", "demo", "free", "plus",
    "premium", "enterprise", "cloud", "server", "center",
    "data", "labs", "studio", "hub", "connect", "platform",
    "manager", "service", "tool", "tools", "kit", "web",
    "online", "digital", "smart", "global", "standard",
}


def _generate_keywords(action_name: str, action_key: str,
                       app_aliases: list[str]) -> list[str]:
    """Generate search keywords from action metadata + app aliases."""
    keywords = set()

    # App aliases are critical for app-level disambiguation
    keywords.update(app_aliases)

    # Action name tokens (skip articles/prepositions)
    skip = {"a", "an", "the", "to", "for", "in", "on", "at", "by", "with", "from", "or", "and"}
    for token in re.split(r"[\s_-]+", action_name.lower()):
        if len(token) > 1 and token not in skip:
            keywords.add(token)

    # Action key tokens (skip generic slug parts like "app", "api", "bot")
    for token in re.split(r"[\s_-]+", action_key.lower()):
        if len(token) > 1 and token not in skip and token not in _GENERIC_ALIAS_WORDS:
            keywords.add(token)

    # Remove "none" if it somehow got in
    keywords.discard("none")

    return sorted(keywords)


# Patterns that indicate a variant action
_VARIANT_PATTERNS = re.compile(
    r"\(advanced\)|\(beta\)|\badvanced\b|\bbatch\b|\bbulk\b|\bmultiple\b"
    r"|\bblock kit\b|\blarge\b|\bv2\b|\bor update\b|\bor create\b"
    r"|\bcustom\b|\braw\b|\blegacy\b",
    re.IGNORECASE,
)


def _is_variant(action_name: str) -> bool:
    """Detect if an action is a variant of a primary action."""
    return bool(_VARIANT_PATTERNS.search(action_name))


def generate_exemplars(app: dict, action: dict,
                       app_intent: dict | None = None,
                       props_cache: dict[str, list[dict]] | None = None) -> list[dict]:
    """Generate 2-3 varied exemplars for a single action.

    Each exemplar has a different description/keyword mix so BoW encoding
    creates distinct vectors that cover different user phrasings.

    Args:
        app: App dict from registry
        action: Action dict from registry
        app_intent: Per-app intent extensions
        props_cache: Cached props {action_key: [raw_prop, ...]}

    Returns list of exemplar dicts, or empty list for variants.
    """
    app_slug = app.get("name_slug", "unknown")
    app_name = app.get("name", app_slug)
    action_key = action.get("key", "unknown")
    action_name = action.get("name", action_key)
    action_version = action.get("version", "0.0.1")

    if _is_variant(action_name):
        return []  # Skip variants entirely

    target_overrides = app_intent.get("target_overrides") if app_intent else None
    verb = _extract_action_verb(action_name)
    target = _extract_target(action_name, action_key, target_overrides)
    domain = _infer_domain(app, app_intent)
    app_aliases = _get_app_aliases(app, app_intent)
    base_keywords = _generate_keywords(action_name, action_key, app_aliases)

    # Derive clean_name from SLUG (not registry name) — slug always has differentiating tokens
    # e.g., "shopify_developer_app" → "shopify developer app" (not registry "Shopify" → "shopify")
    # This ensures BoW descriptions for variant apps are distinct from base apps
    clean_name = app_slug.lstrip("_").replace("_", " ")

    # Get configured props from cache if available, else from action dict
    if props_cache and action_key in props_cache:
        configured_props = _simplify_props(props_cache[action_key])
    else:
        configured_props = action.get("configuredProps", {})

    # Build the base dict shared across all exemplar variations
    base = {
        "action_key": action_key,
        "app_slug": app_slug,
        "app_name": app_name,
        "action_name": action_name,
        "version": action_version,
        "action": verb,
        "target": target,
        "domain": domain,
        "variant": False,
        "configured_props": configured_props,
    }

    exemplars = []

    # Extract all significant words from the action name for BoW differentiation
    # "List Members in Channel" → ["members", "channel"]
    skip = {"a", "an", "the", "to", "for", "in", "on", "at", "by", "with", "from", "or", "and"}
    action_name_words = [w for w in action_name.lower().split()
                         if len(w) > 1 and w not in skip]
    action_name_lc = action_name.lower()

    # Extract action-key parts (after app prefix and verb) for extra BoW signal
    # "discord_bot-send-message-with-file" → ["message", "file"]
    key_parts = action_key.lower().replace("_", "-").split("-")
    # Find verb position, take everything after it
    verb_idx = -1
    for idx, part in enumerate(key_parts):
        for pattern, _ in _ACTION_NAME_TO_VERB:
            if part == pattern:
                verb_idx = idx
                break
        if verb_idx >= 0:
            break
    action_key_nouns = [p for p in key_parts[verb_idx + 1:] if len(p) > 1 and p not in skip] if verb_idx >= 0 else []

    # Separate CATEGORY words (shared across actions, appear in key prefix before verb)
    # from DIFFERENTIATING words (unique to this action, appear after verb)
    # "bitget-spot-market-get-candle-data" → category=["spot","market"], diff=["candle","data"]
    category_words = set()
    if verb_idx > 0:
        # Words between app prefix and verb are category words
        app_prefix_parts = app_slug.lower().replace("_", "-").split("-")
        prefix_len = len(app_prefix_parts)
        category_words = {p for p in key_parts[prefix_len:verb_idx] if len(p) > 1 and p not in skip}

    # Differentiating words: action-specific minus category minus verb
    all_action_words = set(action_name_words + action_key_nouns)
    diff_words = list(all_action_words - category_words - {verb} - {v for _, v in _ACTION_NAME_TO_VERB})
    category_list = list(category_words)

    # Combine for full action-specific set
    action_specific = list(all_action_words)

    # --- Descriptions with moderate frequency weighting ---
    # Now that BoW preserves word frequency, repetition creates real emphasis.
    # Keep it moderate (2-3x) to avoid over-specializing vs flat-weighted queries.
    diff_str = " ".join(diff_words)
    cat_str = " ".join(category_list)
    all_specific_str = " ".join(action_specific)

    # Exemplar 1: Action name + diff words 3x + category + app name
    desc1 = f"{action_name_lc} {diff_str} {diff_str} {diff_str} {cat_str} {clean_name}"
    exemplars.append({
        **base,
        "keywords": list(set(base_keywords + diff_words + [target])),
        "description": desc1,
    })

    # Exemplar 2: Verb + action-specific words 2x + app name
    desc2 = f"{verb} {all_specific_str} {all_specific_str} {clean_name}"
    exemplars.append({
        **base,
        "keywords": list(set(base_keywords + action_specific + app_aliases[:2])),
        "description": desc2,
    })

    # Exemplar 3: Domain + verb + diff words 2x + action name + app name
    desc3 = f"{domain} {verb} {diff_str} {diff_str} {action_name_lc} {clean_name}"
    exemplars.append({
        **base,
        "keywords": list(set(base_keywords + [domain, verb, target] + diff_words)),
        "description": desc3,
    })

    return exemplars


# ---------------------------------------------------------------------------
# Exhaustive test query generation
# ---------------------------------------------------------------------------

# Verb → all natural language phrasings
_VERB_PHRASINGS: dict[str, list[str]] = {
    "send": ["send", "send a", "post a", "send out", "fire off"],
    "create": ["create", "create a", "make a", "new", "open a", "start a"],
    "get": ["get", "get the", "retrieve", "fetch", "show me the", "pull", "check"],
    "list": ["list", "list all", "show all", "get all", "show me all"],
    "update": ["update", "update the", "edit the", "modify the", "change the"],
    "delete": ["delete", "delete the", "remove the", "destroy the"],
    "search": ["search for", "find", "look for", "search", "locate"],
    "add": ["add", "add a", "add new"],
    "remove": ["remove", "remove the", "take out"],
    "upload": ["upload", "upload a", "upload the"],
    "share": ["share", "share the", "share a"],
    "export": ["export", "export the", "download", "download the"],
    "import": ["import", "import the", "import a"],
    "set": ["set", "set the", "configure the", "configure"],
    "track": ["track", "track a", "log", "record"],
    "subscribe": ["subscribe to", "subscribe"],
    "unsubscribe": ["unsubscribe from", "unsubscribe"],
    "charge": ["charge", "charge the"],
    "refund": ["refund", "refund the"],
    "cancel": ["cancel", "cancel the"],
    "invite": ["invite", "invite a", "invite the"],
    "assign": ["assign", "assign the", "assign a"],
    "comment": ["comment on", "add a comment to", "leave a comment on"],
    "trigger": ["trigger", "run", "execute", "start", "launch"],
    "schedule": ["schedule", "schedule a", "book a"],
    "sync": ["sync", "sync the", "synchronize"],
    "archive": ["archive", "archive the", "close the"],
    "reply": ["reply to", "respond to"],
    "none": [],
}

# Target → all natural language noun forms
_TARGET_PHRASINGS: dict[str, list[str]] = {
    "message": ["message", "msg", "chat message"],
    "channel": ["channel"],
    "thread": ["thread"],
    "email": ["email", "mail"],
    "contact": ["contact", "contact record"],
    "customer": ["customer", "account"],
    "deal": ["deal", "opportunity"],
    "ticket": ["ticket", "issue"],
    "task": ["task", "story"],
    "file": ["file", "document"],
    "folder": ["folder", "directory"],
    "spreadsheet": ["spreadsheet", "sheet"],
    "event": ["event", "meeting", "calendar event"],
    "invoice": ["invoice", "bill"],
    "payment": ["payment", "charge"],
    "subscription": ["subscription", "plan"],
    "record": ["record", "row", "entry"],
    "table": ["table", "database table"],
    "post": ["post", "status update"],
    "comment": ["comment", "note"],
    "page": ["page", "document"],
    "note": ["note"],
    "user": ["user", "member", "person"],
    "repo": ["repo", "repository"],
    "pull_request": ["pull request", "PR", "merge request"],
    "commit": ["commit"],
    "deploy": ["deployment", "deploy"],
    "workflow": ["workflow"],
    "pipeline": ["pipeline"],
    "order": ["order", "purchase"],
    "product": ["product", "item"],
    "campaign": ["campaign"],
    "audience": ["audience", "list"],
    "employee": ["employee", "staff member"],
    "applicant": ["applicant", "candidate"],
    "sprint": ["sprint"],
    "board": ["board"],
    "slide": ["slide", "presentation"],
    "refund": ["refund"],
    "payout": ["payout"],
    "form": ["form"],
    "report": ["report"],
    "metric": ["metric", "stat"],
    "database": ["database", "db"],
    "release": ["release"],
    "branch": ["branch"],
    "none": [],
}


def generate_test_queries(exemplar: dict, app_aliases: list[str],
                          app_intent: dict | None = None) -> list[dict]:
    """Generate exhaustive test queries for a single exemplar.

    Produces 10-20+ NL query variants covering every reasonable phrasing.
    Each test query includes the expected action_key and app_slug.
    """
    action_key = exemplar["action_key"]
    app_slug = exemplar["app_slug"]
    app_name = exemplar["app_name"]
    verb = exemplar["action"]
    target = exemplar["target"]

    # Derive app_display from slug or registry name — whichever has MORE distinctive tokens
    # Slug: "shopify_developer_app" → "Shopify Developer App" (3 tokens, distinctive)
    # Registry: "Shopify" (1 token, ambiguous) → slug wins
    # Registry: "Bot for Slack" (3 tokens) vs slug "slack_bot" → "Slack Bot" (2 tokens) → registry wins
    slug_display = app_slug.lstrip("_").replace("_", " ").title()
    registry_display = app_name.replace("(", "").replace(")", "").strip()
    registry_display = re.sub(r"\s+", " ", registry_display)
    app_display = slug_display if len(slug_display.split()) > len(registry_display.split()) else registry_display

    verb_phrases = _VERB_PHRASINGS.get(verb, [verb] if verb != "none" else [])
    target_phrases = _TARGET_PHRASINGS.get(target, [target] if target != "none" else [])

    # Add per-app target overrides to test phrasings
    if app_intent and app_intent.get("target_overrides"):
        extra_targets = []
        for synonym, canonical in app_intent["target_overrides"].items():
            if canonical == target and synonym not in target_phrases:
                extra_targets.append(synonym)
        target_phrases = target_phrases + extra_targets

    # Collect all unique app name forms for test generation
    app_names = [app_display]
    # Add slug-derived form if different from primary display
    if slug_display.lower() != app_display.lower():
        app_names.append(slug_display)
    # Add registry form ONLY if it's distinctive (not ambiguous with another app)
    # Skip if slug_display won (= slug has more tokens than registry name = variant app)
    # because registry name alone is ambiguous (e.g., "Shopify" for shopify_developer_app)
    is_variant = len(slug_display.split()) > len(registry_display.split())
    if not is_variant and registry_display.lower() != app_display.lower() and registry_display.lower() != slug_display.lower():
        app_names.append(registry_display)
    # Add significant aliases (not just single-letter tokens)
    for alias in app_aliases:
        if len(alias) > 2 and alias.lower() != app_display.lower():
            app_names.append(alias)
    # Deduplicate preserving order
    seen = set()
    unique_app_names = []
    for n in app_names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            unique_app_names.append(n)
    app_names = unique_app_names[:4]  # Cap at 4 to avoid combinatorial explosion

    test_queries = []
    seen_queries = set()

    def _add(query: str, test_type: str):
        q_lower = query.lower().strip()
        if q_lower not in seen_queries:
            seen_queries.add(q_lower)
            test_queries.append({
                "query": query,
                "expected_action_key": action_key,
                "expected_app_slug": app_slug,
                "app_name": app_name,
                "test_type": test_type,
            })

    # Use the full action name (stripped of parenthetical) for compound-action queries
    action_name_clean = re.sub(r"\s*\(.*?\)\s*", "", exemplar["action_name"]).strip()
    action_name_lc = action_name_clean.lower()

    # Detect compound actions: action name has more than just "{verb} {target}"
    # e.g., "List Members in Channel" (4 words), "Set Channel Topic" (3 words)
    # vs simple "Send Message" (2 words), "List Users" (2 words)
    action_words = action_name_clean.split()
    skip_words = {"a", "an", "the", "to", "for", "in", "on", "at", "by", "with", "from"}
    significant_words = [w for w in action_words if w.lower() not in skip_words]
    is_compound = len(significant_words) > 2

    # Always generate action-name-based queries (most specific, unambiguous)
    _add(f"{action_name_clean} on {app_display}", "action_name")
    _add(f"{action_name_clean} in {app_display}", "action_name")
    _add(f"{app_display} {action_name_lc}", "action_name")
    for an in app_names[1:3]:
        _add(f"{action_name_clean} on {an}", "action_name")

    # Extract action-specific nouns from the action name (after the verb)
    # "Get Backlinks Summary" → ["backlinks", "summary"]
    action_nouns = [w.lower() for w in action_words[1:]
                    if w.lower() not in skip_words and len(w) > 1]
    action_noun_str = " ".join(action_nouns) if action_nouns else ""

    if target_phrases and verb_phrases and not is_compound:
        # Simple actions with a recognized target
        # Use action_noun_str (from action name) as the primary target phrasing
        # to avoid cross-action collisions within the same app
        primary_target = action_noun_str if action_noun_str else target_phrases[0]

        # Pattern 1: "{verb} {action_nouns} on {app}"
        for vp in verb_phrases:
            _add(f"{vp} {primary_target} on {app_display}", "canonical")

        # Pattern 2: "{verb} {action_nouns} in/via/using {app}"
        for prep in ["in", "via", "using", "with"]:
            _add(f"{verb_phrases[0]} {primary_target} {prep} {app_display}", "alt_prep")

        # Pattern 3: "{app} {verb} {action_nouns}"
        for vp in verb_phrases[:3]:
            _add(f"{app_display} {vp} {primary_target}", "app_first")

        # Pattern 4: alt target nouns (only from _TARGET_PHRASINGS, NOT cross-action)
        for tp in target_phrases[1:]:
            _add(f"{verb_phrases[0]} {tp} on {app_display}", "alt_target")
            _add(f"{app_display} {verb_phrases[0]} {tp}", "alt_target")

        # Pattern 5: alt app names
        for an in app_names[1:]:
            _add(f"{verb_phrases[0]} {primary_target} on {an}", "alt_app_name")
            _add(f"{an} {verb_phrases[0]} {primary_target}", "alt_app_name")

        # Pattern 6: with context
        _add(f"{verb_phrases[0]} a {app_display.lower()} {primary_target}", "contextual")

    elif verb_phrases:
        # Compound action OR no recognized target — ALWAYS use action-specific
        # nouns from the action name. NEVER generate bare "verb on app" queries.

        if action_noun_str:
            # Use action nouns: "Get Backlinks Summary" → "get backlinks summary on DataForSEO"
            for vp in verb_phrases[:3]:
                _add(f"{vp} {action_noun_str} on {app_display}", "canonical")
                _add(f"{app_display} {vp} {action_noun_str}", "app_first")

            for prep in ["in", "via", "using"]:
                _add(f"{verb_phrases[0]} {action_noun_str} {prep} {app_display}", "alt_prep")

            for an in app_names[1:]:
                _add(f"{verb_phrases[0]} {action_noun_str} on {an}", "alt_app_name")
                _add(f"{an} {verb_phrases[0]} {action_noun_str}", "alt_app_name")
        else:
            # Last resort: use the full action name (already added above as action_name patterns)
            for an in app_names[1:]:
                _add(f"{action_name_clean} on {an}", "alt_app_name")

    return test_queries


# ---------------------------------------------------------------------------
# Per-app intent scaffold generation
# ---------------------------------------------------------------------------

def _scaffold_app_intent(app: dict, actions: list[dict]) -> str:
    """Generate a default apps/{slug}/intent.py scaffold from registry metadata."""
    slug = app.get("name_slug", "unknown")
    name = app.get("name", slug)
    clean_name = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
    domain = _infer_domain(app)

    # Build aliases from slug and name — only full forms, not generic tokens
    aliases = set()
    aliases.add(slug.lower())
    # Full name with parenthetical content (remove parens only)
    full_name = name.replace("(", "").replace(")", "").strip()
    full_name = re.sub(r"\s+", " ", full_name)
    if full_name.lower() != slug.lower():
        aliases.add(full_name.lower())
    # Only add stripped name if not a variant of a base app
    if clean_name.lower().replace(" ", "_") == slug:
        aliases.add(clean_name.lower())
    # Filter out any generic single-word aliases and slug fragments
    slug_tokens = set(slug.lstrip("_").split("_"))
    is_multi_word = len(slug_tokens) > 1
    aliases = {a for a in aliases
               if " " in a or "_" in a  # multi-word aliases always OK
               or (a not in _GENERIC_ALIAS_WORDS
                   and not (is_multi_word and a in slug_tokens))}

    # Build target overrides from action names (detect app-specific nouns)
    target_overrides = {}
    for action in actions:
        action_name = action.get("name", "").lower()
        # Look for nouns in action names that aren't in the base target map
        # This is just a scaffold — human curation expected
        pass

    aliases_str = json.dumps(sorted(aliases), indent=4)

    return f'''"""Per-app intent extensions for {clean_name}."""

# Aliases users might say when referring to this app
APP_ALIASES = {aliases_str}

# Correct domain for this app (overrides domain inference from query keywords)
APP_DOMAIN = "{domain}"

# App-specific target noun remappings (override base _TARGET_MAP)
# e.g., {{"ticket": "issue"}} if this app calls issues "tickets"
TARGET_OVERRIDES: dict[str, str] = {{}}

# App-specific verb synonyms (extend base _VERB_MAP)
ACTION_SYNONYMS: dict[str, str] = {{}}
'''


# ---------------------------------------------------------------------------
# Main discovery flow
# ---------------------------------------------------------------------------

def discover(
    app_filter: list[str] | None = None,
    limit: int = 0,
    dry_run: bool = False,
    fetch_props: bool = False,
    props_batch_size: int = 50,
    props_delay: float = 1.0,
) -> dict:
    """Pull the Pipedream registry and generate per-app exemplars + tests.

    Creates apps/{slug}/ directories with exemplars.jsonl, tests.jsonl,
    and intent.py (scaffold only if not already present).

    Then aggregates into data/exemplars.jsonl and data/test_queries.jsonl.
    """
    print("Connecting to Pipedream API...")
    client = PipedreamClient()

    # Step 1: List apps
    if app_filter:
        apps = []
        for slug in app_filter:
            found = client.list_apps(query=slug, limit=10)
            exact = [a for a in found if a.get("name_slug") == slug]
            if exact:
                apps.extend(exact)
            elif found:
                apps.append(found[0])
            else:
                print(f"  Warning: app '{slug}' not found")
        print(f"Found {len(apps)} apps (filtered)")
    else:
        max_apps = limit if limit > 0 else 10000
        apps = client.list_apps(limit=max_apps)
        print(f"Found {len(apps)} apps")

    # Load props cache (persists across runs)
    props_cache = _load_props_cache() if fetch_props else None

    # Step 2: Discover actions per app, generate per-app output
    all_exemplars = []
    all_test_queries = []
    registry_cache = {"apps": [], "discovered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    app_stats = []

    for i, app in enumerate(apps):
        slug = app.get("name_slug", "?")
        name = app.get("name", slug)
        print(f"  [{i+1}/{len(apps)}] {name} ({slug})...", end=" ", flush=True)

        try:
            actions = client.list_actions(slug)
        except Exception as e:
            print(f"error: {e}")
            continue

        print(f"{len(actions)} actions")

        registry_cache["apps"].append({
            "slug": slug,
            "name": name,
            "categories": app.get("categories", []),
            "actions": [{"key": a.get("key"), "name": a.get("name"), "version": a.get("version")}
                        for a in actions],
        })

        # Fetch props for this app's actions (if enabled)
        if fetch_props and props_cache is not None:
            action_keys = [a.get("key") for a in actions if a.get("key")]
            props_cache = fetch_action_props(
                client, action_keys, props_cache,
                batch_size=props_batch_size, delay=props_delay,
            )

        # Load per-app intent extensions
        app_intent = _load_app_intent(slug)
        app_aliases = _get_app_aliases(app, app_intent)

        # Generate exemplars (2-3 per primary action, 0 for variants)
        app_exemplars = []
        skipped = 0
        for action in actions:
            exs = generate_exemplars(app, action, app_intent, props_cache=props_cache)
            if not exs:
                skipped += 1
                continue
            app_exemplars.extend(exs)

        if skipped:
            print(f"    ({skipped} variants skipped)")

        # Generate test queries for each primary action (use first exemplar as reference)
        app_tests = []
        action_keys_seen = set()
        for ex in app_exemplars:
            ak = ex["action_key"]
            if ak in action_keys_seen:
                continue  # Only generate tests once per action, not per exemplar
            action_keys_seen.add(ak)
            tests = generate_test_queries(ex, app_aliases, app_intent)
            app_tests.extend(tests)

        all_exemplars.extend(app_exemplars)
        all_test_queries.extend(app_tests)

        n_actions = len(action_keys_seen)
        app_stats.append((slug, n_actions, len(app_exemplars), len(app_tests)))

        # Write per-app files
        if not dry_run:
            app_dir = APPS_DIR / slug
            app_dir.mkdir(parents=True, exist_ok=True)

            # Write exemplars
            with open(app_dir / "exemplars.jsonl", "w") as f:
                for ex in app_exemplars:
                    f.write(json.dumps(ex, separators=(",", ":")) + "\n")

            # Write tests
            with open(app_dir / "tests.jsonl", "w") as f:
                for tq in app_tests:
                    f.write(json.dumps(tq, separators=(",", ":")) + "\n")

            # Write intent scaffold (only if not already present)
            intent_path = app_dir / "intent.py"
            if not intent_path.exists():
                intent_path.write_text(_scaffold_app_intent(app, actions))

        # Rate limiting
        if i % 10 == 9:
            time.sleep(0.5)

    # Step 3: Print stats
    print(f"\nDiscovery complete:")
    print(f"  Apps: {len(apps)}")

    total_actions = sum(s[1] for s in app_stats)
    print(f"  Actions (primary): {total_actions}")
    print(f"  Exemplars: {len(all_exemplars)} ({len(all_exemplars)/max(total_actions,1):.1f} per action)")
    print(f"  Test queries: {len(all_test_queries)} ({len(all_test_queries)/max(total_actions,1):.1f} per action)")

    # Domain distribution
    domain_counts: dict[str, int] = {}
    for e in all_exemplars:
        d = e["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    print(f"\nDomain distribution:")
    for d, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d}: {count}")

    # Per-app summary
    print(f"\nPer-app stats:")
    for slug, n_actions, n_exemplars, n_tests in sorted(app_stats, key=lambda x: -x[2]):
        print(f"  {slug}: {n_actions} actions, {n_exemplars} exemplars, {n_tests} tests")

    if dry_run:
        print("\n(Dry run — no files written)")
        return {"apps": len(apps), "actions": total_actions,
                "exemplars": len(all_exemplars),
                "test_queries": len(all_test_queries), "output": ""}

    # Step 4: Write aggregated data files
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_EXEMPLARS, "w") as f:
        for ex in all_exemplars:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")
    print(f"\nWrote {len(all_exemplars)} exemplars to {OUTPUT_EXEMPLARS}")

    with open(OUTPUT_TEST_QUERIES, "w") as f:
        for tq in all_test_queries:
            f.write(json.dumps(tq, separators=(",", ":")) + "\n")
    print(f"Wrote {len(all_test_queries)} test queries to {OUTPUT_TEST_QUERIES}")

    with open(OUTPUT_CACHE, "w") as f:
        json.dump(registry_cache, f, indent=2)
    print(f"Wrote registry cache to {OUTPUT_CACHE}")

    return {
        "apps": len(apps),
        "actions": total_actions,
        "exemplars": len(all_exemplars),
        "test_queries": len(all_test_queries),
        "output": str(OUTPUT_EXEMPLARS),
    }


def regenerate_exemplars(apps_dir: Path | None = None) -> int:
    """Regenerate exemplar descriptions and keywords from existing exemplar metadata.

    Reads apps/*/exemplars.jsonl, regenerates descriptions using the improved
    action-focused logic, and writes back. Does NOT change action_key, metadata, etc.
    Uses props_cache.json if available to populate configured_props.
    """
    apps_dir = apps_dir or APPS_DIR
    props_cache = _load_props_cache()  # Use cached props if available
    total_updated = 0

    for app_dir in sorted(apps_dir.iterdir()):
        if not app_dir.is_dir():
            continue

        ex_path = app_dir / "exemplars.jsonl"
        if not ex_path.exists():
            continue

        slug = app_dir.name
        app_intent = _load_app_intent(slug)

        # Read existing exemplars
        entries = []
        for line in ex_path.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line.strip()))

        if not entries:
            continue

        # Group by action_key (3 exemplars per action)
        from collections import defaultdict
        by_action = defaultdict(list)
        for e in entries:
            by_action[e["action_key"]].append(e)

        # Regenerate
        updated = []
        for action_key, exemplars in by_action.items():
            ref = exemplars[0]  # Use first as reference for metadata

            # Build a fake action dict for generate_exemplars()
            fake_app = {
                "name_slug": ref.get("app_slug", slug),
                "name": ref.get("app_name", slug),
                "categories": [],
            }
            fake_action = {
                "key": action_key,
                "name": ref.get("action_name", action_key),
                "version": ref.get("version", "0.0.1"),
                "configuredProps": ref.get("configured_props", {}),
            }

            new_exemplars = generate_exemplars(fake_app, fake_action, app_intent,
                                                props_cache=props_cache or None)
            if new_exemplars:
                updated.extend(new_exemplars)
            else:
                # Keep original if generate_exemplars returns empty (variant)
                updated.extend(exemplars)

        # Write back
        with open(ex_path, "w") as f:
            for ex in updated:
                f.write(json.dumps(ex, separators=(",", ":")) + "\n")

        total_updated += len(updated)

    print(f"Regenerated {total_updated} exemplars across {sum(1 for d in apps_dir.iterdir() if d.is_dir())} apps")
    return total_updated


def regenerate_tests(apps_dir: Path | None = None) -> dict:
    """Regenerate test queries from existing per-app exemplars (no API needed).

    Reads apps/*/exemplars.jsonl, generates new tests using the fixed
    generate_test_queries() logic, writes apps/*/tests.jsonl, then aggregates.
    """
    apps_dir = apps_dir or APPS_DIR
    total_tests = 0
    total_exemplars = 0
    app_count = 0

    for app_dir in sorted(apps_dir.iterdir()):
        if not app_dir.is_dir():
            continue

        ex_path = app_dir / "exemplars.jsonl"
        if not ex_path.exists():
            continue

        slug = app_dir.name
        app_intent = _load_app_intent(slug)

        # Load exemplars
        exemplars = []
        for line in ex_path.read_text().splitlines():
            if line.strip():
                exemplars.append(json.loads(line.strip()))

        if not exemplars:
            continue

        # Get app aliases from first exemplar
        app_aliases = _get_app_aliases(
            {"name_slug": slug, "name": exemplars[0].get("app_name", slug)},
            app_intent,
        )

        # Generate tests (one per action, using first exemplar for each)
        app_tests = []
        action_keys_seen = set()
        for ex in exemplars:
            ak = ex["action_key"]
            if ak in action_keys_seen:
                continue
            action_keys_seen.add(ak)
            tests = generate_test_queries(ex, app_aliases, app_intent)
            app_tests.extend(tests)

        # Write tests
        with open(app_dir / "tests.jsonl", "w") as f:
            for tq in app_tests:
                f.write(json.dumps(tq, separators=(",", ":")) + "\n")

        total_tests += len(app_tests)
        total_exemplars += len(exemplars)
        app_count += 1

    print(f"Regenerated tests for {app_count} apps: {total_tests} tests from {total_exemplars} exemplars")

    # Now aggregate
    return aggregate(apps_dir)


def aggregate(apps_dir: Path | None = None) -> dict:
    """Aggregate per-app data into data/ without re-running discovery.

    Useful after manually editing per-app exemplars or tests.
    """
    apps_dir = apps_dir or APPS_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_exemplars = 0
    total_tests = 0

    with open(OUTPUT_EXEMPLARS, "w") as ef, open(OUTPUT_TEST_QUERIES, "w") as tf:
        for app_dir in sorted(apps_dir.iterdir()):
            if not app_dir.is_dir():
                continue

            ex_path = app_dir / "exemplars.jsonl"
            if ex_path.exists():
                for line in ex_path.read_text().splitlines():
                    if line.strip():
                        ef.write(line.strip() + "\n")
                        total_exemplars += 1

            test_path = app_dir / "tests.jsonl"
            if test_path.exists():
                for line in test_path.read_text().splitlines():
                    if line.strip():
                        tf.write(line.strip() + "\n")
                        total_tests += 1

    print(f"Aggregated: {total_exemplars} exemplars, {total_tests} tests")
    return {"exemplars": total_exemplars, "test_queries": total_tests}


def fetch_props_only(
    app_filter: list[str] | None = None,
    batch_size: int = 50,
    delay: float = 1.0,
) -> None:
    """Fetch props for all known actions without re-running full discovery.

    Uses registry_cache.json or exemplars.jsonl to get action keys, then
    fetches props for any actions not already in the props cache.
    """
    all_keys = []

    # Try registry cache first
    if OUTPUT_CACHE.exists():
        print("Loading action keys from registry cache...")
        cache_data = json.loads(OUTPUT_CACHE.read_text())
        for app in cache_data.get("apps", []):
            if app_filter and app["slug"] not in app_filter:
                continue
            for action in app.get("actions", []):
                key = action.get("key")
                if key:
                    all_keys.append(key)
    else:
        # Fall back to exemplars — extract unique action keys
        print("No registry cache found, extracting action keys from exemplars...")
        seen = set()
        for source in [OUTPUT_EXEMPLARS, *(APPS_DIR.glob("*/exemplars.jsonl"))]:
            if not source.exists():
                continue
            for line in source.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line.strip())
                key = entry.get("action_key")
                slug = entry.get("app_slug", "")
                if key and key not in seen:
                    if app_filter and slug not in app_filter:
                        continue
                    all_keys.append(key)
                    seen.add(key)

    if not all_keys:
        print("Error: No action keys found. Run discovery first.")
        sys.exit(1)

    print(f"Found {len(all_keys)} action keys")

    client = PipedreamClient()
    props_cache = _load_props_cache()
    fetch_action_props(client, all_keys, props_cache,
                       batch_size=batch_size, delay=delay)

    # Show summary
    populated = sum(1 for k in all_keys if props_cache.get(k))
    print(f"\nProps cache: {populated}/{len(all_keys)} actions have props")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover Pipedream registry and generate per-app exemplars")
    parser.add_argument("--apps", type=str, default="",
                        help="Comma-separated app slugs to discover (default: all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of apps to process (default: unlimited)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Just re-aggregate existing per-app data (no API calls)")
    parser.add_argument("--regenerate-tests", action="store_true",
                        help="Regenerate test queries from existing exemplars (no API calls)")
    parser.add_argument("--regenerate-all", action="store_true",
                        help="Regenerate both exemplars and tests (no API calls)")
    parser.add_argument("--fetch-props", action="store_true",
                        help="Fetch action props from API (uses cache, incremental)")
    parser.add_argument("--props-batch-size", type=int, default=50,
                        help="Number of prop requests between saves/sleeps (default: 50)")
    parser.add_argument("--props-delay", type=float, default=1.0,
                        help="Seconds to sleep between prop batches (default: 1.0)")
    args = parser.parse_args()

    if args.fetch_props and not any([args.regenerate_all, args.regenerate_tests,
                                     args.aggregate_only]):
        # Standalone prop fetching mode
        app_filter = [a.strip() for a in args.apps.split(",") if a.strip()] if args.apps else None
        fetch_props_only(app_filter=app_filter,
                         batch_size=args.props_batch_size,
                         delay=args.props_delay)
    elif args.regenerate_all:
        regenerate_exemplars()
        regenerate_tests()
    elif args.regenerate_tests:
        regenerate_tests()
    elif args.aggregate_only:
        aggregate()
    else:
        app_filter = [a.strip() for a in args.apps.split(",") if a.strip()] if args.apps else None
        discover(app_filter=app_filter, limit=args.limit, dry_run=args.dry_run,
                 fetch_props=args.fetch_props,
                 props_batch_size=args.props_batch_size,
                 props_delay=args.props_delay)
