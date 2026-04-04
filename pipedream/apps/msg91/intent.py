"""Per-app intent extensions for MSG91."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "msg91"
]

# Correct domain for this app (overrides domain inference from query keywords)
APP_DOMAIN = "messaging"

# App-specific target noun remappings (override base _TARGET_MAP)
# e.g., {"ticket": "issue"} if this app calls issues "tickets"
TARGET_OVERRIDES: dict[str, str] = {}

# App-specific verb synonyms (extend base _VERB_MAP)
ACTION_SYNONYMS: dict[str, str] = {}
