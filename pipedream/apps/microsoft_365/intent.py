"""Per-app intent extensions for Microsoft 365."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "microsoft 365",
    "microsoft_365"
]

# Correct domain for this app (overrides domain inference from query keywords)
APP_DOMAIN = "productivity"

# App-specific target noun remappings (override base _TARGET_MAP)
# e.g., {"ticket": "issue"} if this app calls issues "tickets"
TARGET_OVERRIDES: dict[str, str] = {}

# App-specific verb synonyms (extend base _VERB_MAP)
ACTION_SYNONYMS: dict[str, str] = {}
