"""Per-app intent extensions for Claris FileMaker Server - Admin API."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "claris filemaker server - admin api",
    "claris_filemaker_server_admin_api"
]

# Correct domain for this app (overrides domain inference from query keywords)
APP_DOMAIN = "none"

# App-specific target noun remappings (override base _TARGET_MAP)
# e.g., {"ticket": "issue"} if this app calls issues "tickets"
TARGET_OVERRIDES: dict[str, str] = {}

# App-specific verb synonyms (extend base _VERB_MAP)
ACTION_SYNONYMS: dict[str, str] = {}
