"""Per-app intent extensions for Metatext.AI Pre-build AI models API."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "metatext.ai pre-build ai models api",
    "metatext_ai_pre_build_ai_models_api"
]

# Correct domain for this app (overrides domain inference from query keywords)
APP_DOMAIN = "none"

# App-specific target noun remappings (override base _TARGET_MAP)
# e.g., {"ticket": "issue"} if this app calls issues "tickets"
TARGET_OVERRIDES: dict[str, str] = {}

# App-specific verb synonyms (extend base _VERB_MAP)
ACTION_SYNONYMS: dict[str, str] = {}
