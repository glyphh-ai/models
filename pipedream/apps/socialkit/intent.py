"""Per-app intent extensions for SocialKit."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "socialkit"
]

# Correct domain for this app (overrides domain inference from query keywords)
APP_DOMAIN = "social"

# App-specific target noun remappings (override base _TARGET_MAP)
# e.g., {"ticket": "issue"} if this app calls issues "tickets"
TARGET_OVERRIDES: dict[str, str] = {}

# App-specific verb synonyms (extend base _VERB_MAP)
ACTION_SYNONYMS: dict[str, str] = {}
