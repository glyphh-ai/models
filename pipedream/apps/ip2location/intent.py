"""Per-app intent extensions for IP2Location."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "ip2location"
]

# Correct domain for this app (overrides domain inference from query keywords)
APP_DOMAIN = "developer"

# App-specific target noun remappings (override base _TARGET_MAP)
# e.g., {"ticket": "issue"} if this app calls issues "tickets"
TARGET_OVERRIDES: dict[str, str] = {}

# App-specific verb synonyms (extend base _VERB_MAP)
ACTION_SYNONYMS: dict[str, str] = {}
