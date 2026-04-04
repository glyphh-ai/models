"""Tests for encoder config, intent extraction, and record generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from encoder import ENCODER_CONFIG, encode_query, entry_to_record
from intent import extract_intent, assess_query, preprocess


# ---------------------------------------------------------------------------
# Config structure
# ---------------------------------------------------------------------------

def test_config_has_two_layers():
    assert len(ENCODER_CONFIG.layers) == 2
    names = {l.name for l in ENCODER_CONFIG.layers}
    assert names == {"intent", "semantics"}


def test_intent_layer_has_action_and_scope():
    intent_layer = next(l for l in ENCODER_CONFIG.layers if l.name == "intent")
    seg_names = {s.name for s in intent_layer.segments}
    assert "action" in seg_names
    assert "scope" in seg_names


def test_semantics_layer_has_bow_roles():
    sem_layer = next(l for l in ENCODER_CONFIG.layers if l.name == "semantics")
    roles = []
    for seg in sem_layer.segments:
        for role in seg.roles:
            roles.append(role.name)
    assert "description" in roles
    assert "keywords" in roles


# ---------------------------------------------------------------------------
# Intent extraction
# ---------------------------------------------------------------------------

def test_extract_intent_send_slack():
    result = extract_intent("send a message to #general on Slack")
    assert result["action"] == "send"
    assert result["domain"] == "messaging"


def test_extract_intent_create_jira():
    result = extract_intent("create a Jira ticket for the login bug")
    assert result["action"] == "create"
    assert result["domain"] == "tickets"


def test_extract_intent_refund_stripe():
    result = extract_intent("refund the customer's last charge on Stripe")
    assert result["action"] == "refund"
    assert result["domain"] == "payments"


def test_extract_intent_search_github():
    result = extract_intent("search for open pull requests on GitHub")
    assert result["action"] == "search"
    assert result["domain"] == "developer"


def test_extract_intent_schedule_calendar():
    result = extract_intent("schedule a meeting on Google Calendar for tomorrow")
    assert result["action"] == "schedule"
    assert result["domain"] == "calendar"


def test_extract_intent_upload_drive():
    result = extract_intent("upload the report to Google Drive")
    assert result["action"] == "upload"
    assert result["domain"] == "files"


def test_extract_intent_track_segment():
    result = extract_intent("track a page view event in Segment")
    assert result["action"] == "track"
    assert result["domain"] == "analytics"


def test_extract_intent_suppresses_questions():
    result = extract_intent("what does this Slack integration do?")
    assert result["action"] == "none"


def test_extract_intent_suppresses_greetings():
    result = extract_intent("hello how are you today")
    assert result["action"] == "none"


def test_extract_intent_email_domain():
    result = extract_intent("send an email to john@example.com via Gmail")
    assert result["action"] == "send"
    assert result["domain"] == "email"


def test_extract_intent_crm_domain():
    result = extract_intent("update the contact in Salesforce")
    assert result["action"] == "update"
    assert result["domain"] == "crm"


def test_extract_intent_ecommerce():
    result = extract_intent("list all orders from Shopify")
    assert result["action"] == "list"
    assert result["domain"] == "ecommerce"


def test_extract_intent_ai_domain():
    result = extract_intent("generate an image with OpenAI DALL-E")
    assert result["action"] == "create"
    assert result["domain"] == "ai"


def test_extract_intent_social():
    result = extract_intent("post a tweet on Twitter about the launch")
    assert result["action"] == "send"
    assert result["domain"] == "social"


# ---------------------------------------------------------------------------
# assess_query
# ---------------------------------------------------------------------------

def test_assess_complete_query():
    result = assess_query("send a Slack message to #general")
    assert result["complete"] is True
    assert result["missing"] == []


def test_assess_missing_action():
    result = assess_query("something with Slack")
    assert result["complete"] is False
    assert "action" in result["missing"]


def test_assess_missing_domain():
    result = assess_query("send a message")
    # "send" found but no domain signal
    assert result["action"] == "send"


# ---------------------------------------------------------------------------
# encode_query
# ---------------------------------------------------------------------------

def test_encode_query_returns_dict():
    result = encode_query("send a Slack message")
    assert "name" in result
    assert "attributes" in result
    assert result["attributes"]["action"] == "send"


def test_encode_query_has_all_attributes():
    result = encode_query("create a Jira ticket")
    attrs = result["attributes"]
    assert all(k in attrs for k in ["action", "target", "domain", "description", "keywords"])


# ---------------------------------------------------------------------------
# entry_to_record
# ---------------------------------------------------------------------------

def test_entry_to_record_structure():
    entry = {
        "action_key": "slack-send-message",
        "app_slug": "slack",
        "app_name": "Slack",
        "action_name": "Send Message",
        "action": "send",
        "target": "message",
        "domain": "messaging",
        "keywords": ["send", "message", "slack", "channel"],
        "description": "send message slack channel",
    }
    record = entry_to_record(entry)
    assert record["concept_text"] == "slack-send-message"
    assert record["attributes"]["action"] == "send"
    assert record["attributes"]["domain"] == "messaging"
    assert record["metadata"]["action_key"] == "slack-send-message"
    assert record["metadata"]["app_slug"] == "slack"


def test_entry_to_record_joins_keyword_list():
    entry = {
        "action_key": "test-action",
        "action": "get",
        "target": "none",
        "domain": "none",
        "keywords": ["foo", "bar", "baz"],
    }
    record = entry_to_record(entry)
    assert record["attributes"]["keywords"] == "foo bar baz"


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def test_preprocess_strips_punctuation():
    assert preprocess("Hello, World!") == "hello  world"


def test_preprocess_lowercase():
    assert preprocess("SEND MESSAGE") == "send message"
