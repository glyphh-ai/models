"""
Intent extraction for the Pipedream Action Router.

Unlike the toolrouter (which has a fixed 8-domain lexicon), this model
covers 17+ domains across 3,000+ apps.  Intent extraction is broad:
we extract action/target/domain from the NL query, then let HDC similarity
do the heavy lifting against auto-generated exemplars.

Exports:
  extract_intent(query) → {action, target, domain, keywords}
  preprocess(text) → str
  assess_query(query) → {complete, missing, reason, action, domain}
"""

import re

# ---------------------------------------------------------------------------
# Verb → canonical action mapping
# ---------------------------------------------------------------------------
# Broad coverage — Pipedream spans messaging, CRM, payments, dev tools,
# marketing, HR, ecommerce, databases, AI, and more.

_VERB_MAP: dict[str, str] = {
    # send / notify
    "send": "send", "email": "send", "mail": "send",
    "notify": "send", "announce": "send", "post": "send",
    "broadcast": "send", "ping": "send", "alert": "send",
    "message": "send", "dm": "send", "text": "send",
    # reply
    "reply": "reply", "respond": "reply",
    # create
    "create": "create", "make": "create", "open": "create",
    "file": "create", "add": "add", "new": "create",
    "insert": "create", "register": "create", "setup": "create",
    "build": "create", "compose": "create", "draft": "create",
    "generate": "create", "provision": "create", "spawn": "create",
    # get / retrieve
    "get": "get", "fetch": "get", "retrieve": "get",
    "give": "get", "show": "get", "display": "get",
    "pull": "get", "check": "get", "describe": "get",
    "summarize": "get", "summarise": "get", "tell": "get",
    "read": "get", "view": "get", "look": "get", "lookup": "get",
    "inspect": "get", "query": "get", "find": "search",
    # list
    "list": "list", "enumerate": "list",
    # search
    "search": "search", "find": "search", "locate": "search",
    "discover": "search", "browse": "search", "filter": "search",
    # update / edit
    "update": "update", "edit": "update", "change": "update",
    "modify": "update", "patch": "update", "rename": "update",
    "set": "set", "configure": "set", "adjust": "set",
    # delete
    "delete": "delete", "remove": "remove", "destroy": "delete",
    "drop": "delete", "purge": "delete", "clear": "delete",
    "unsubscribe": "unsubscribe",
    # share / upload / export
    "share": "share", "upload": "upload", "attach": "upload",
    "export": "export", "download": "export",
    "import": "import",
    # payments
    "charge": "charge", "refund": "refund", "pay": "charge",
    "bill": "charge", "invoice": "charge",
    "cancel": "cancel", "subscribe": "subscribe",
    # tracking / analytics
    "track": "track", "record": "track", "log": "track",
    "identify": "track", "monitor": "track",
    # calendar / scheduling
    "schedule": "schedule", "book": "schedule",
    "invite": "invite", "rsvp": "invite",
    # assignment / workflow
    "assign": "assign", "delegate": "assign",
    "comment": "comment", "annotate": "comment",
    "approve": "trigger", "reject": "trigger",
    "trigger": "trigger", "run": "trigger", "execute": "trigger",
    "start": "trigger", "launch": "trigger",
    "sync": "sync", "connect": "sync", "integrate": "sync",
    "archive": "archive", "close": "archive",
    "forward": "send",
}

# ---------------------------------------------------------------------------
# Target noun mapping
# ---------------------------------------------------------------------------
_TARGET_MAP: dict[str, str] = {
    # messaging
    "message": "message", "messages": "message", "msg": "message", "chat": "message",
    "notification": "message", "notifications": "message", "alert": "message",
    "channel": "channel", "channels": "channel", "room": "channel",
    "thread": "thread", "threads": "thread",
    # email
    "email": "email", "emails": "email", "mail": "email", "inbox": "email",
    # contacts / CRM
    "contact": "contact", "contacts": "contact", "lead": "contact",
    "leads": "contact", "prospect": "contact",
    "customer": "customer", "customers": "customer",
    "account": "customer", "client": "customer",
    "deal": "deal", "deals": "deal", "opportunity": "deal", "pipeline": "deal",
    # tickets / issues
    "ticket": "ticket", "tickets": "ticket", "issue": "ticket",
    "issues": "ticket", "bug": "ticket", "bugs": "ticket",
    "task": "task", "tasks": "task", "story": "task", "epic": "task",
    "sprint": "sprint", "sprints": "sprint", "board": "board",
    # files
    "file": "file", "files": "file", "document": "file", "doc": "file",
    "folder": "folder", "folders": "folder", "directory": "folder",
    "spreadsheet": "spreadsheet", "sheet": "spreadsheet",
    "slide": "slide", "presentation": "slide",
    # calendar
    "event": "event", "events": "event", "meeting": "event",
    "meetings": "event", "appointment": "event", "calendar": "event",
    # payments
    "invoice": "invoice", "invoices": "invoice",
    "payment": "payment", "payments": "payment", "charge": "payment",
    "subscription": "subscription", "subscriptions": "subscription",
    "plan": "subscription",
    "refund": "refund", "payout": "payout",
    # marketing
    "campaign": "campaign", "audience": "audience",
    "list": "list", "segment": "segment",
    # ecommerce
    "order": "order", "product": "product", "item": "product",
    "cart": "cart", "checkout": "cart",
    "review": "review", "rating": "review",
    # crypto / trading
    "trade": "trade", "trades": "trade", "trading": "trade",
    "ticker": "ticker", "tickers": "ticker",
    "candle": "candle", "candles": "candle", "candlestick": "candle",
    "orderbook": "orderbook", "order book": "orderbook",
    "depth": "depth",
    "rate": "rate", "rates": "rate",
    "fee": "fee", "fees": "fee",
    "fill": "fill", "fills": "fill",
    "contract": "contract", "contracts": "contract",
    "position": "position", "positions": "position",
    "balance": "balance", "balances": "balance",
    "price": "price", "prices": "price",
    "symbol": "symbol", "symbols": "symbol",
    "auction": "auction",
    "coin": "coin", "coins": "coin", "token": "coin",
    # developer
    "repo": "repo", "repos": "repo", "repository": "repo",
    "branch": "branch", "branches": "branch",
    "pr": "pull_request", "pull request": "pull_request",
    "commit": "commit", "commits": "commit",
    "deploy": "deploy", "release": "release", "releases": "release",
    "workflow": "workflow", "workflows": "workflow",
    "pipeline": "pipeline",
    # database
    "record": "record", "records": "record", "row": "record", "entry": "record",
    "table": "table", "tables": "table",
    "database": "database", "collection": "table",
    # social
    "post": "post", "posts": "post", "tweet": "post", "status": "post",
    "comment": "comment", "comments": "comment", "reply": "comment",
    # HR
    "employee": "employee", "employees": "employee",
    "applicant": "applicant", "applicants": "applicant",
    "candidate": "applicant", "timeoff": "timeoff",
    # discord / community
    "role": "role", "roles": "role",
    "invite": "invite", "invites": "invite", "invitation": "invite",
    "reaction": "reaction", "reactions": "reaction",
    "emoji": "emoji", "emojis": "emoji",
    "nickname": "nickname", "nick": "nickname",
    "guild": "guild", "server": "guild",
    "member": "member", "members": "member",
    # generic
    "user": "user", "users": "user",
    "person": "user",
    "page": "page", "pages": "page",
    "note": "note", "notes": "note",
    "form": "form", "report": "report", "metric": "metric",
    "tag": "tag", "tags": "tag",
    "label": "label", "labels": "label",
    "template": "template", "templates": "template",
    "draft": "draft", "drafts": "draft",
    "attachment": "attachment", "attachments": "attachment",
    "status": "status",
    "config": "config", "configuration": "config", "setting": "config",
    "settings": "config",
}

# ---------------------------------------------------------------------------
# Domain inference signals
# ---------------------------------------------------------------------------
_DOMAIN_SIGNALS: list[tuple[str, list[str]]] = [
    ("messaging", ["slack", "discord", "teams", "whatsapp", "telegram",
                    "channel", "dm", "direct message", "chat"]),
    ("email", ["email", "gmail", "outlook", "mailchimp", "sendgrid",
               "smtp", "inbox", " cc ", " bcc "]),
    ("crm", ["salesforce", "hubspot", "zoho", "pipedrive", "crm",
             "contact record", "deal", "pipeline", "lead"]),
    ("payments", ["stripe", "paypal", "square", "braintree",
                  "invoice", "subscription", "charge", "refund", "billing"]),
    ("calendar", ["google calendar", "gcal", "calendly", "cal.com",
                  "calendar event", "meeting", "schedule", "standup"]),
    ("files", ["google drive", "gdrive", "dropbox", "box", "onedrive",
               "sharepoint", "s3", "file", "folder", "upload"]),
    ("tickets", ["jira", "linear", "asana", "trello", "monday",
                 "clickup", "notion", "ticket", "issue", "task", "sprint"]),
    ("analytics", ["segment", "amplitude", "mixpanel", "google analytics",
                   "posthog", "heap", "metric", "funnel", "pageview", "track event",
                   "page view", "track a page", "event in segment"]),
    ("developer", ["github", "gitlab", "bitbucket", "vercel", "netlify",
                   "heroku", "aws", "azure", "gcp", "docker", "kubernetes",
                   "repo", "pull request", "deploy", "ci", "cd"]),
    ("social", ["twitter", "facebook", "instagram", "linkedin", "reddit",
                "tiktok", "youtube", "pinterest", "tweet", "post"]),
    ("marketing", ["mailchimp", "sendgrid", "constant contact", "klaviyo",
                   "campaign", "newsletter", "audience", "drip"]),
    ("database", ["airtable", "supabase", "firebase", "mongodb", "postgres",
                  "mysql", "notion database", "dynamodb", "redis"]),
    ("ai", ["openai", "anthropic", "claude", "chatgpt", "gpt",
            "dall-e", "stable diffusion", "whisper", "replicate"]),
    ("productivity", ["notion", "google sheets", "excel", "google docs",
                      "confluence", "wiki", "evernote", "todoist"]),
    ("hr", ["bamboohr", "gusto", "rippling", "workday", "deel",
            "employee", "applicant", "timeoff", "payroll"]),
    ("ecommerce", ["shopify", "woocommerce", "magento", "bigcommerce",
                   "order", "product", "cart", "inventory"]),
    ("crypto", ["bitcoin", "ethereum", "crypto", "blockchain", "defi",
                "binance", "coinbase", "bitget", "bybit", "kraken",
                "futures", "spot trading", "candlestick", "orderbook"]),
]

# ---------------------------------------------------------------------------
# Stop words + suppression
# ---------------------------------------------------------------------------
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "am", "do", "does", "did", "has", "have", "had", "will", "would",
    "shall", "should", "may", "might", "can", "could", "must",
    "to", "of", "in", "for", "on", "at", "by", "with", "from",
    "as", "into", "about", "between", "through", "after", "before",
    "and", "but", "or", "nor", "not", "so", "yet", "both",
    "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "this", "that", "these", "those", "there", "here",
    "please", "just", "also", "very", "really", "then",
    "up", "out", "if", "when", "all", "each", "every",
})

_QUESTION_PATTERNS = [
    "what does", "what is", "what are", "what's the", "whats the",
    "how does", "how do", "how many", "how much",
    "why does", "why is", "can you explain", "tell me about",
]

_GREETING_PATTERNS = ["hello", "hi there", "hey there", "how are you"]


def preprocess(text: str) -> str:
    """Lowercase and strip punctuation for consistent BoW encoding."""
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


def _extract_action(words: list[str]) -> str:
    """Extract the canonical action from word list."""
    for w in words:
        action = _VERB_MAP.get(w)
        if action is not None:
            return action
    return "none"


def _extract_target(words: list[str]) -> str:
    """Extract the canonical target from word list."""
    for w in words:
        target = _TARGET_MAP.get(w)
        if target is not None:
            return target
    return "none"


def _infer_domain(text: str) -> str:
    """Infer domain from keyword signals in the raw query text."""
    text_lower = text.lower()
    for domain, signals in _DOMAIN_SIGNALS:
        if any(sig in text_lower for sig in signals):
            return domain
    return "none"


def _extract_keywords(words: list[str]) -> list[str]:
    """Extract content keywords, filtering stop words."""
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _is_suppressed(text: str) -> bool:
    """Detect queries that should not route to any tool."""
    text_lower = text.lower()
    return (
        any(p in text_lower for p in _QUESTION_PATTERNS)
        or any(text_lower.startswith(p) for p in _GREETING_PATTERNS)
    )


def extract_intent(query: str) -> dict:
    """Extract action, target, domain, and keywords from an NL query.

    Returns:
        {action, target, domain, keywords: str (space-separated)}
    """
    clean = preprocess(query)
    words = clean.split()

    if _is_suppressed(query):
        return {
            "action": "none",
            "target": "none",
            "domain": "none",
            "keywords": " ".join(_extract_keywords(words)) or clean,
        }

    action = _extract_action(words)
    target = _extract_target(words)
    domain = _infer_domain(query)
    keywords = _extract_keywords(words)

    return {
        "action": action,
        "target": target,
        "domain": domain,
        "keywords": " ".join(keywords) or clean,
    }


_extract_app_fn = None  # set by encoder.py after import


def set_extract_app(fn):
    """Register extract_app callback from encoder.py (avoids circular import)."""
    global _extract_app_fn
    _extract_app_fn = fn


def assess_query(query: str) -> dict:
    """Assess whether a query has enough signal to route with confidence.

    For Pipedream, a query is complete if:
    - action + domain are both detected, OR
    - an app name is detected (extract_app finds it) — domain not required
      because thousands of niche apps won't have domain signals.

    Returns:
        {complete, missing, reason, action, domain}
    """
    intent = extract_intent(query)
    action = intent["action"]
    domain = intent["domain"]

    missing: list[str] = []
    if action == "none":
        missing.append("action")
    if domain == "none":
        missing.append("domain")

    complete = len(missing) == 0

    # If app name is detected, that's sufficient signal — skip domain requirement
    if not complete and "domain" in missing and _extract_app_fn and _extract_app_fn(query):
        missing.remove("domain")
        complete = len(missing) == 0

    if not complete:
        if "action" in missing and "domain" in missing:
            reason = "Cannot determine what to do or which service to use"
        elif "action" in missing:
            reason = "Cannot determine what action to perform"
        else:
            reason = "Cannot determine which service or app to use"
    else:
        reason = ""

    return {
        "complete": complete,
        "missing": missing,
        "reason": reason,
        "action": action,
        "domain": domain,
    }
