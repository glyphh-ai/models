"""Per-app intent extensions for CDC - National Environmental Public Health Tracking."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "cdc - national environmental public health tracking",
    "cdc_national_environmental_public_health_tracking"
]

# Correct domain for this app (overrides domain inference from query keywords)
APP_DOMAIN = "none"

# App-specific target noun remappings (override base _TARGET_MAP)
# e.g., {"ticket": "issue"} if this app calls issues "tickets"
TARGET_OVERRIDES: dict[str, str] = {}

# App-specific verb synonyms (extend base _VERB_MAP)
ACTION_SYNONYMS: dict[str, str] = {}
