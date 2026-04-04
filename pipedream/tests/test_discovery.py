"""Tests for exemplar generation logic (no API calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from discover import (
    generate_exemplar,
    _extract_action_verb,
    _extract_target,
    _infer_domain,
    _generate_keywords,
)


# ---------------------------------------------------------------------------
# Action verb extraction
# ---------------------------------------------------------------------------

def test_verb_send_message():
    assert _extract_action_verb("Send Message") == "send"


def test_verb_create_issue():
    assert _extract_action_verb("Create Issue") == "create"


def test_verb_list_commits():
    assert _extract_action_verb("List Commits") == "list"


def test_verb_delete_record():
    assert _extract_action_verb("Delete Record") == "delete"


def test_verb_update_contact():
    assert _extract_action_verb("Update Contact") == "update"


def test_verb_search_files():
    assert _extract_action_verb("Search Files") == "search"


def test_verb_upload_file():
    assert _extract_action_verb("Upload File") == "upload"


def test_verb_get_user():
    assert _extract_action_verb("Get User") == "get"


def test_verb_add_comment():
    assert _extract_action_verb("Add Comment") == "add"


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

def test_target_from_action_key():
    assert _extract_target("Send Message", "slack-send-message") == "message"


def test_target_channel():
    assert _extract_target("Send Message to Channel", "slack-send-message-to-channel") == "channel"


def test_target_ticket():
    assert _extract_target("Create Ticket", "jira-create-ticket") == "ticket"


def test_target_file():
    assert _extract_target("Upload File", "gdrive-upload-file") == "file"


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

def test_domain_slack():
    app = {"name_slug": "slack", "categories": ["Communication"]}
    assert _infer_domain(app) == "messaging"


def test_domain_stripe():
    app = {"name_slug": "stripe", "categories": ["Payment"]}
    assert _infer_domain(app) == "payments"


def test_domain_github():
    app = {"name_slug": "github", "categories": ["Developer Tools"]}
    assert _infer_domain(app) == "developer"


def test_domain_from_category():
    app = {"name_slug": "some_unknown_app", "categories": ["E-Commerce"]}
    assert _infer_domain(app) == "ecommerce"


def test_domain_none_unknown():
    app = {"name_slug": "totally_unknown", "categories": ["Misc"]}
    assert _infer_domain(app) == "none"


# ---------------------------------------------------------------------------
# Full exemplar generation
# ---------------------------------------------------------------------------

def test_generate_exemplar_slack():
    app = {"name_slug": "slack", "name": "Slack", "categories": ["Communication"]}
    action = {"key": "slack-send-message", "name": "Send Message", "version": "0.0.3"}
    exemplar = generate_exemplar(app, action)

    assert exemplar["action_key"] == "slack-send-message"
    assert exemplar["app_slug"] == "slack"
    assert exemplar["action"] == "send"
    assert exemplar["domain"] == "messaging"
    assert "slack" in exemplar["keywords"]
    assert "send" in exemplar["keywords"]
    assert "message" in exemplar["keywords"]


def test_generate_exemplar_github():
    app = {"name_slug": "github", "name": "GitHub", "categories": ["Developer Tools"]}
    action = {"key": "github-create-issue", "name": "Create Issue", "version": "0.0.1"}
    exemplar = generate_exemplar(app, action)

    assert exemplar["action"] == "create"
    assert exemplar["target"] == "ticket"  # "issue" maps to "ticket"
    assert exemplar["domain"] == "developer"


def test_generate_exemplar_stripe():
    app = {"name_slug": "stripe", "name": "Stripe", "categories": ["Payment"]}
    action = {"key": "stripe-create-refund", "name": "Create Refund", "version": "0.0.1"}
    exemplar = generate_exemplar(app, action)

    assert exemplar["action"] == "create"
    assert exemplar["domain"] == "payments"


def test_generate_exemplar_preserves_metadata():
    app = {"name_slug": "airtable", "name": "Airtable", "categories": ["Database"]}
    action = {
        "key": "airtable-create-record",
        "name": "Create Record",
        "version": "0.2.0",
        "configuredProps": {"baseId": None, "tableId": None},
    }
    exemplar = generate_exemplar(app, action)

    assert exemplar["version"] == "0.2.0"
    assert exemplar["configured_props"] == {"baseId": None, "tableId": None}
