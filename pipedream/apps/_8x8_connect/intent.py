"""Per-app intent extensions for 8x8 Connect."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "8x8 connect",
    "_8x8_connect"
]

# Correct domain for this app (overrides domain inference from query keywords)
APP_DOMAIN = "messaging"

# App-specific target noun remappings (override base _TARGET_MAP)
# e.g., {"ticket": "issue"} if this app calls issues "tickets"
TARGET_OVERRIDES: dict[str, str] = {}

# App-specific verb synonyms (extend base _VERB_MAP)
ACTION_SYNONYMS: dict[str, str] = {}
