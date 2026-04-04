"""Per-app intent extensions for GitHub."""

# Aliases users might say when referring to this app
APP_ALIASES = [
    "github", "gh", "git hub", "github.com",
]

# Correct domain for this app (NOT "tickets" even though it has issues)
APP_DOMAIN = "developer"

# App-specific target noun remappings
# Critical: "ticket" and "bug" should map to "ticket" (the canonical target)
# but the domain override ensures GitHub wins over Jira for developer context
TARGET_OVERRIDES: dict[str, str] = {
    "pr": "pull_request",
    "merge request": "pull_request",
    "MR": "pull_request",
    "star": "repo",
    "fork": "repo",
    "gist": "file",
    "assignee": "user",
    "reviewer": "user",
    "collaborator": "user",
    "run": "workflow",
    "action": "workflow",
    "actions": "workflow",
    "dispatch": "workflow",
}

# App-specific verb synonyms
ACTION_SYNONYMS: dict[str, str] = {
    "merge": "update",
    "fork": "create",
    "clone": "get",
    "star": "add",
    "unstar": "remove",
    "enable": "trigger",
    "disable": "archive",
}
