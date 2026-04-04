"""
Marketing intent extraction — parse NL queries about campaigns, posts, and audiences.

Extracts marketing search intent from natural language queries like
"find campaigns similar to our product launch" or "what content works
for technical audiences".
"""

import re
from typing import Dict, List

# ============================================================================
# Action Verbs -> Canonical Actions
# ============================================================================

_ACTION_MAP = {
    # Search
    "find": "search", "search": "search", "show": "search",
    "get": "search", "look": "search", "list": "search",
    "browse": "search", "discover": "search", "explore": "search",
    "locate": "search", "display": "search", "pull": "search",
    # Compare
    "compare": "compare", "benchmark": "compare", "versus": "compare",
    "vs": "compare", "contrast": "compare", "rank": "compare",
    "stack": "compare",
    # Analyze
    "analyze": "analyze", "report": "analyze", "measure": "analyze",
    "track": "analyze", "evaluate": "analyze", "assess": "analyze",
    "audit": "analyze", "review": "analyze", "summarize": "analyze",
    "breakdown": "analyze", "inspect": "analyze",
    # Optimize
    "optimize": "optimize", "improve": "optimize", "boost": "optimize",
    "increase": "optimize", "maximize": "optimize", "grow": "optimize",
    "enhance": "optimize", "scale": "optimize", "accelerate": "optimize",
    # Create
    "create": "create", "launch": "create", "plan": "create",
    "schedule": "create", "draft": "create", "build": "create",
    "design": "create", "compose": "create", "set up": "create",
    # Target
    "target": "target", "reach": "target", "segment": "target",
    "personalize": "target", "retarget": "target",
}

# ============================================================================
# Target Keywords -> Target Categories
# ============================================================================

_TARGET_MAP = {
    # Campaign
    "campaign": "campaign", "campaigns": "campaign",
    "initiative": "campaign", "program": "campaign",
    "push": "campaign", "launch": "campaign",
    "strategy": "campaign", "effort": "campaign",
    # Post / Content
    "post": "post", "posts": "post", "content": "post",
    "creative": "post", "ad": "post", "ads": "post",
    "copy": "post", "message": "post", "tweet": "post",
    "article": "post", "video": "post", "email": "post",
    "newsletter": "post", "blog": "post", "webinar": "post",
    "podcast": "post", "whitepaper": "post", "infographic": "post",
    # Audience
    "audience": "audience", "audiences": "audience",
    "segment": "audience", "segments": "audience",
    "cohort": "audience", "persona": "audience",
    "demographic": "audience", "users": "audience",
    "subscribers": "audience", "followers": "audience",
    "customers": "audience", "prospects": "audience",
    # Performance
    "performance": "performance", "metrics": "performance",
    "results": "performance", "analytics": "performance",
    "engagement": "performance", "conversion": "performance",
    "roi": "performance", "kpi": "performance", "kpis": "performance",
    "stats": "performance", "numbers": "performance",
}

# ============================================================================
# Domain Signals -> Marketing Channels
# ============================================================================

_DOMAIN_SIGNALS = {
    "social": [
        "twitter", "linkedin", "instagram", "facebook", "tiktok",
        "social", "organic", "feed", "story", "reel", "thread",
        "hashtag", "followers", "shares", "retweet", "repost",
    ],
    "email": [
        "email", "newsletter", "drip", "sequence", "autoresponder",
        "mailchimp", "sendgrid", "subject", "open rate", "inbox",
        "unsubscribe", "subscriber",
    ],
    "content": [
        "blog", "article", "whitepaper", "case study", "webinar",
        "podcast", "seo", "organic search", "content marketing",
        "thought leadership", "editorial", "long form",
    ],
    "paid": [
        "ad", "ads", "ppc", "cpc", "cpm", "spend", "budget",
        "retarget", "google ads", "meta ads", "display", "banner",
        "impression", "bid", "cost per", "roas", "paid media",
    ],
    "brand": [
        "brand", "awareness", "positioning", "messaging", "pr",
        "press", "launch", "rebrand", "identity", "voice",
        "reputation", "sentiment",
    ],
    "analytics": [
        "analytics", "report", "dashboard", "attribution", "funnel",
        "a/b", "test", "experiment", "cohort", "segmentation",
        "forecast", "predict", "trend", "benchmark",
    ],
}

# ============================================================================
# Modifier Keywords
# ============================================================================

_MODIFIERS = {
    "similar", "same", "like", "matching", "different",
    "best", "worst", "top", "bottom", "highest", "lowest",
    "recent", "latest", "last", "previous", "current",
    "high", "low", "strong", "weak", "growing", "declining",
}


def extract_intent(query: str) -> Dict[str, str]:
    """
    Extract marketing search intent from a text query.

    Args:
        query: Natural language marketing query

    Returns:
        Dict with keys: action, target, domain, modifiers, keywords
    """
    query_lower = query.lower().strip()
    words = re.findall(r'\w+', query_lower)

    action = "search"
    target = "none"
    domain = "none"
    modifiers = []
    keywords = []

    # Extract action from first matching verb
    for word in words:
        if word in _ACTION_MAP:
            action = _ACTION_MAP[word]
            break

    # Extract targets and modifiers
    targets_found = []
    for word in words:
        if word in _TARGET_MAP:
            targets_found.append(_TARGET_MAP[word])
        if word in _MODIFIERS:
            modifiers.append(word)

    # Primary target (first found)
    if targets_found:
        target = targets_found[0]

    # Infer domain from weighted signal scoring
    domain_scores = {}
    for dom, signals in _DOMAIN_SIGNALS.items():
        score = 0
        for signal in signals:
            if signal in query_lower:
                score += 1
        if score > 0:
            domain_scores[dom] = score

    if domain_scores:
        domain = max(domain_scores, key=domain_scores.get)

    # Keywords = all non-stopword tokens
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "with",
        "in", "on", "at", "to", "for", "of", "and", "or",
        "me", "my", "i", "you", "it", "that", "this", "our",
        "find", "show", "get", "search", "look", "what", "how",
        "which", "do", "does", "did", "has", "have", "can",
        "will", "would", "should", "could", "about", "from",
        "than", "then", "but", "not", "no", "so", "if",
    }
    keywords = [w for w in words if w not in stopwords and len(w) > 1]

    return {
        "action": action,
        "target": target,
        "domain": domain,
        "modifiers": " ".join(modifiers) if modifiers else "",
        "keywords": " ".join(keywords) if keywords else "",
        "all_targets": " ".join(dict.fromkeys(targets_found)) if targets_found else "",
    }
