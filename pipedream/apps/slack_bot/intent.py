"""Per-app intent extensions for Slack (Bot for Slack)."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "slack_bot",
    "slackbot",
    "slack bot",
    "bot for slack"
]

# Correct domain for this app
APP_DOMAIN = "messaging"

# App-specific target noun remappings
# Note: "group", "topic", "description" are NOT mapped to "channel" —
# they're qualifiers/keywords, not the actual target. Keeping them unmapped
# lets the forward scan find the true target (e.g., "members" → user).
TARGET_OVERRIDES: dict[str, str] = {
    "reaction": "emoji",
    "emoji": "emoji",
    "members": "user",
    "replies": "thread",
    "conversations": "channel",
}

# App-specific verb synonyms
ACTION_SYNONYMS: dict[str, str] = {
    "react": "add",
    "pin": "add",
    "dm": "send",
    "notify": "send",
    "announce": "send",
    "post": "send",
    "ping": "send",
}
