"""
Intent extraction for the deal intelligence model.

Provides keyword extraction and text preprocessing for NL queries
about sales deals, pipeline health, and forecasting. No outcome
inference — won/lost/stalled are outcomes derived from HDC similarity.

Exports:
  extract_keywords(text) -> str  — space-separated keyword tokens
  preprocess(text) -> str        — cleaned text for BoW encoding
"""

import re


# ---------------------------------------------------------------------------
# Stop words — filtered from keyword extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "to", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "how", "what", "which", "who", "whom", "when", "where", "why",
    "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "he", "she", "they", "them", "their", "this", "that", "these",
    "in", "on", "at", "for", "with", "about", "of", "from", "by",
    "can", "will", "would", "could", "should", "may", "might",
    "and", "or", "but", "not", "if", "so", "than", "then", "too", "very",
    "show", "find", "get", "tell", "give", "list", "see", "look",
    "us", "me", "there", "here", "just", "also", "been", "being",
    "some", "any", "all", "most", "many", "much", "more", "less",
    "deal", "deals", "opportunity", "opportunities",
})


def preprocess(text: str) -> str:
    """Lowercase and strip punctuation for consistent BoW encoding."""
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# Light stemming — normalize common suffixes before synonym lookup
# ---------------------------------------------------------------------------

_STEM_SUFFIXES = [
    "ation", "tion", "ment", "ness", "able", "ible",
    "ing", "ers", "ies", "ous", "ive",
    "ed", "es", "er", "ly",
    "s",
]

_STEM_EXCEPTIONS = frozenset({
    "this", "has", "was", "does", "goes", "is", "us", "yes", "no",
    "less", "plus", "bus", "gas", "his", "its",
    "cases", "issues", "series", "sales", "process",
})


def _stem(word: str) -> str:
    """Strip the longest common suffix if the remaining stem is >= 3 chars."""
    if word in _STEM_EXCEPTIONS or len(word) <= 3:
        return word
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# ---------------------------------------------------------------------------
# Phrase normalization — compound terms -> single tokens
# ---------------------------------------------------------------------------

_PHRASE_MAP = {
    "at risk":           "at_risk",
    "high value":        "high_value",
    "low value":         "low_value",
    "deal size":         "deal_size",
    "deal value":        "deal_value",
    "close rate":        "close_rate",
    "win rate":          "win_rate",
    "sales cycle":       "sales_cycle",
    "pipeline health":   "pipeline_health",
    "next step":         "next_step",
    "next steps":        "next_step",
    "decision maker":    "decision_maker",
    "buying signal":     "buying_signal",
    "buying signals":    "buying_signal",
    "no activity":       "no_activity",
    "gone dark":         "gone_dark",
    "went dark":         "gone_dark",
    "no response":       "no_response",
    "follow up":         "follow_up",
    "closed won":        "closed_won",
    "closed lost":       "closed_lost",
    "moved forward":     "progressing",
    "moving forward":    "progressing",
    "pushed back":       "slipping",
    "pushed out":        "slipping",
    "high velocity":     "high_velocity",
    "slip date":         "slipping",
    "slipped date":      "slipping",
    "close date":        "close_date",
    "expected close":    "close_date",
    "proof of concept":  "poc",
    "technical evaluation": "technical_eval",
    "price negotiation": "negotiation",
    "contract negotiation": "negotiation",
}


def _apply_phrases(text: str) -> str:
    """Replace known multi-word phrases with underscore-joined tokens."""
    for phrase, token in sorted(_PHRASE_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(phrase, token)
    return text


# ---------------------------------------------------------------------------
# Domain synonym expansion — maps broad NL terms to exemplar vocabulary
# ---------------------------------------------------------------------------

_DOMAIN_SYNONYMS: dict[str, list[str]] = {
    # -- Won / closing signals --
    "won": ["won", "closed_won", "success", "converted", "signed"],
    "winning": ["won", "closed_won", "success", "progressing", "strong"],
    "close": ["closing", "won", "closed_won", "negotiation", "final", "contract", "strong", "champion", "momentum"],
    "closed": ["closed_won", "closed_lost", "won", "lost", "finished"],
    "signed": ["won", "closed_won", "signed", "committed", "converted"],
    "converted": ["won", "closed_won", "converted", "success"],
    "success": ["won", "closed_won", "success", "strong", "champion"],
    "champion": ["champion", "strong", "engaged", "progressing", "won"],

    # -- Lost / failed signals --
    "lost": ["lost", "closed_lost", "failed", "dead", "no_response"],
    "failed": ["lost", "closed_lost", "failed", "dead", "stalled"],
    "dead": ["lost", "closed_lost", "dead", "no_activity", "abandoned"],
    "abandoned": ["lost", "dead", "no_activity", "gone_dark", "no_response"],
    "killed": ["lost", "closed_lost", "dead", "competitor"],

    # -- Stalled / stuck signals --
    "stalled": ["stalled", "stuck", "no_activity", "at_risk", "slipping"],
    "stuck": ["stalled", "stuck", "no_activity", "at_risk", "slipping"],
    "frozen": ["stalled", "stuck", "no_activity", "at_risk"],
    "blocked": ["stalled", "stuck", "objection", "at_risk", "blocker"],
    "ghosted": ["gone_dark", "no_response", "no_activity", "stalled", "at_risk"],
    "dark": ["gone_dark", "no_response", "no_activity", "stalled"],
    "silent": ["gone_dark", "no_response", "no_activity", "stalled"],
    "unresponsive": ["no_response", "gone_dark", "no_activity", "stalled"],

    # -- Risk signals --
    "risk": ["at_risk", "stalled", "slipping", "competitor", "objection"],
    "risky": ["at_risk", "stalled", "slipping", "competitor"],
    "slipping": ["slipping", "at_risk", "delayed", "stalled", "pushed"],
    "delayed": ["slipping", "delayed", "stalled", "at_risk"],
    "overdue": ["slipping", "delayed", "stalled", "at_risk", "past_due"],
    "behind": ["slipping", "delayed", "stalled", "at_risk"],

    # -- Pipeline / stage signals --
    "pipeline": ["pipeline", "funnel", "forecast", "stage"],
    "funnel": ["pipeline", "funnel", "stage", "conversion"],
    "forecast": ["forecast", "pipeline", "prediction", "quota"],
    "quota": ["forecast", "quota", "target", "pipeline"],
    "target": ["forecast", "quota", "target", "goal"],

    # -- Engagement signals --
    "engaged": ["engaged", "active", "progressing", "responsive", "champion"],
    "active": ["engaged", "active", "progressing", "responsive"],
    "responsive": ["engaged", "active", "responsive", "progressing"],
    "progressing": ["progressing", "advancing", "engaged", "active", "momentum"],
    "momentum": ["progressing", "momentum", "advancing", "strong", "engaged"],
    "advancing": ["progressing", "advancing", "momentum", "engaged"],
    "moving": ["progressing", "advancing", "momentum"],

    # -- Competitor signals --
    "competitor": ["competitor", "competitive", "alternative", "at_risk", "evaluation"],
    "competitive": ["competitor", "competitive", "evaluation", "at_risk"],
    "alternative": ["competitor", "alternative", "evaluation", "at_risk"],
    "evaluation": ["evaluation", "competitor", "poc", "technical_eval"],
    "poc": ["poc", "evaluation", "technical_eval", "trial"],
    "trial": ["poc", "trial", "evaluation", "testing"],

    # -- Deal size / value --
    "enterprise": ["enterprise", "large", "high_value", "strategic"],
    "strategic": ["enterprise", "strategic", "high_value", "large"],
    "large": ["enterprise", "large", "high_value", "strategic"],
    "small": ["small", "smb", "low_value", "transactional"],
    "smb": ["small", "smb", "low_value", "quick"],
    "mid": ["mid_market", "medium", "growth"],
    "midmarket": ["mid_market", "medium", "growth"],

    # -- Stage names --
    "prospecting": ["prospecting", "outbound", "cold", "initial", "early"],
    "discovery": ["discovery", "qualification", "needs", "early"],
    "qualification": ["qualification", "discovery", "needs", "fit"],
    "demo": ["demo", "presentation", "evaluation", "showing"],
    "proposal": ["proposal", "quote", "pricing", "negotiation"],
    "negotiation": ["negotiation", "pricing", "contract", "terms", "closing"],
    "closing": ["closing", "negotiation", "final", "contract", "won", "strong", "champion", "momentum"],
    "onboarding": ["onboarding", "implementation", "setup", "post_sale"],

    # -- Source / channel --
    "inbound": ["inbound", "marketing", "organic", "website"],
    "outbound": ["outbound", "cold", "prospecting", "sdr"],
    "referral": ["referral", "partner", "warm", "introduction"],
    "partner": ["partner", "referral", "channel", "reseller"],
    "marketing": ["marketing", "inbound", "campaign", "lead"],
    "event": ["event", "conference", "trade_show", "meetup"],

    # -- Objection signals --
    "objection": ["objection", "pushback", "concern", "blocker", "at_risk"],
    "pushback": ["objection", "pushback", "concern", "at_risk"],
    "concern": ["objection", "concern", "at_risk", "hesitation"],
    "budget": ["budget", "pricing", "cost", "objection", "at_risk"],
    "pricing": ["pricing", "budget", "cost", "negotiation", "objection"],
    "cost": ["pricing", "budget", "cost", "objection"],
    "expensive": ["pricing", "budget", "objection", "at_risk"],
    "cheap": ["pricing", "discount", "low_value"],
    "discount": ["pricing", "discount", "negotiation", "objection"],
    "timing": ["timing", "delay", "not_now", "objection", "slipping"],

    # -- Velocity / speed --
    "fast": ["fast", "quick", "accelerated", "short_cycle", "momentum"],
    "quick": ["fast", "quick", "accelerated", "short_cycle"],
    "slow": ["slow", "long_cycle", "stalled", "delayed"],
    "velocity": ["velocity", "speed", "momentum", "cycle_time"],
    "speed": ["velocity", "speed", "momentum", "fast"],

    # -- Renewal / expansion --
    "renewal": ["renewal", "renew", "expansion", "upsell", "retention"],
    "renew": ["renewal", "renew", "expansion", "retention"],
    "upsell": ["upsell", "expansion", "cross_sell", "growth"],
    "expansion": ["expansion", "upsell", "cross_sell", "growth", "add_on"],
    "cross": ["cross_sell", "expansion", "upsell", "add_on"],

    # -- Healthy / strong --
    "healthy": ["healthy", "strong", "progressing", "engaged", "on_track"],
    "strong": ["strong", "healthy", "champion", "engaged", "progressing"],
    "hot": ["hot", "strong", "closing", "urgent", "high_priority"],
    "warm": ["warm", "engaged", "progressing", "interested"],
    "cold": ["cold", "no_response", "stalled", "gone_dark", "at_risk"],

    # -- Compound phrase tokens --
    "at_risk": ["at_risk", "stalled", "slipping", "competitor", "objection", "lost"],
    "high_value": ["high_value", "enterprise", "strategic", "large"],
    "low_value": ["low_value", "small", "smb", "transactional"],
    "gone_dark": ["gone_dark", "no_response", "no_activity", "stalled", "ghosted"],
    "no_activity": ["no_activity", "stalled", "gone_dark", "no_response", "inactive"],
    "no_response": ["no_response", "gone_dark", "no_activity", "stalled", "ghosted"],
    "closed_won": ["closed_won", "won", "success", "converted", "signed"],
    "closed_lost": ["closed_lost", "lost", "failed", "dead"],
    "pipeline_health": ["pipeline", "forecast", "health", "funnel"],
    "buying_signal": ["buying_signal", "champion", "engaged", "progressing", "strong"],
    "decision_maker": ["decision_maker", "executive", "champion", "stakeholder"],
    "high_velocity": ["high_velocity", "fast", "velocity", "momentum", "closing", "champion", "strong"],
    "sales_cycle": ["sales_cycle", "velocity", "cycle_time", "duration"],
    "close_date": ["close_date", "deadline", "forecast", "timeline"],
    "next_step": ["next_step", "action", "follow_up", "plan"],
}


def extract_keywords(text: str) -> str:
    """Extract content keywords from NL text, filtering stop words.

    Applies phrase normalization, light stemming, and domain synonym
    expansion to produce a keyword string that overlaps with exemplar
    keywords for BoW matching.

    Returns space-separated keyword tokens.
    """
    cleaned = preprocess(text)

    # Phase 1: normalize compound phrases
    cleaned = _apply_phrases(cleaned)

    # Phase 2: tokenize and filter
    tokens = [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]

    # Phase 3: expand domain synonyms (try raw token, then stemmed)
    expanded: list[str] = list(tokens)
    for token in tokens:
        synonyms = _DOMAIN_SYNONYMS.get(token)
        if not synonyms:
            stemmed = _stem(token)
            synonyms = _DOMAIN_SYNONYMS.get(stemmed)
        if synonyms:
            for syn in synonyms:
                for word in syn.split():
                    if word not in _STOP_WORDS and word not in expanded:
                        expanded.append(word)

    return " ".join(expanded)
