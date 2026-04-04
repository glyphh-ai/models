"""
Marketing encoder — ENCODER_CONFIG, encode_query, entry_to_record.

Encodes three glyph types — Campaigns, Posts, and Audiences — through a
single 4-layer config. Performance metrics are continuous vectors updated
via the listener endpoint as Pipedream webhooks fire.

Architecture:
  Identity layer (0.10): entity_type + entity_id (key_part for temporal)
  Strategy layer (0.25): goal, audience_segment, platform, content_type
  Content layer (0.30): themes (BoW), tone, keywords (BoW)
  Performance layer (0.35): engagement/conversion/reach as continuous vectors

  Two matching paths, one config:
    - NL queries (text, no metrics) -> match on content + strategy layers
    - Data records (metrics + text) -> match on all layers including performance
    - Listener updates (metrics only) -> performance layer drives similarity
"""

import hashlib

from intent import extract_intent

from glyphh.core.config import (
    EncoderConfig,
    Layer,
    Segment,
    Role,
    ContinuousConfig,
    TemporalConfig,
)

# ============================================================================
# ENCODER_CONFIG — 4 layers, weighted
# ============================================================================

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=True,
    temporal_source="auto",
    temporal_config=TemporalConfig(signal_type="auto"),
    layers=[
        # --- Identity ---
        Layer(
            name="identity",
            similarity_weight=0.10,
            segments=[
                Segment(
                    name="entity",
                    similarity_weight=1.0,
                    roles=[
                        Role(
                            name="entity_type",
                            similarity_weight=1.0,
                        ),
                        Role(
                            name="entity_id",
                            similarity_weight=0.1,
                            key_part=True,
                        ),
                    ],
                ),
            ],
        ),
        # --- Strategy ---
        Layer(
            name="strategy",
            similarity_weight=0.25,
            segments=[
                Segment(
                    name="targeting",
                    similarity_weight=1.0,
                    roles=[
                        Role(
                            name="goal",
                            similarity_weight=1.0,
                        ),
                        Role(
                            name="audience_segment",
                            similarity_weight=0.9,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="platform",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="content_type",
                            similarity_weight=0.6,
                        ),
                    ],
                ),
            ],
        ),
        # --- Content ---
        Layer(
            name="content",
            similarity_weight=0.30,
            segments=[
                Segment(
                    name="messaging",
                    similarity_weight=1.0,
                    roles=[
                        Role(
                            name="themes",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="tone",
                            similarity_weight=0.6,
                        ),
                        Role(
                            name="keywords",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        # --- Performance ---
        Layer(
            name="performance",
            similarity_weight=0.35,
            segments=[
                Segment(
                    name="metrics",
                    similarity_weight=1.0,
                    roles=[
                        Role(
                            name="engagement_metrics",
                            similarity_weight=1.0,
                            continuous_config=ContinuousConfig(
                                source_dim=5,
                                projection_seed=400,
                            ),
                        ),
                        Role(
                            name="conversion_metrics",
                            similarity_weight=1.0,
                            continuous_config=ContinuousConfig(
                                source_dim=5,
                                projection_seed=401,
                            ),
                        ),
                        Role(
                            name="reach_metrics",
                            similarity_weight=0.8,
                            continuous_config=ContinuousConfig(
                                source_dim=4,
                                projection_seed=402,
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================================
# encode_query — NL text -> Concept dict for similarity search
# ============================================================================

def encode_query(query: str) -> dict:
    """
    Convert a marketing NL query to a concept dict.

    Text queries match on content + strategy layers.
    Performance roles get zero vectors (no metrics in NL queries).
    """
    intent = extract_intent(query)
    stable_id = hashlib.md5(query.encode()).hexdigest()[:8]

    # Map intent to concept attributes.
    # Only include roles where we have actual signal — omitting a role causes
    # the encoder to skip it entirely, which is better than defaulting to "none"
    # (since "none" is a real symbol that would match other "none" values).
    # Performance metrics are also omitted — NL queries match on content +
    # strategy layers only.
    attributes = {
        "entity_id": "",
        "keywords": intent["keywords"],
    }

    target = intent["target"]
    kw = intent["keywords"]
    domain = intent["domain"]

    # Set entity_type from target (only if detected)
    if target in ("campaign", "post", "audience"):
        attributes["entity_type"] = target

    # Populate strategy from keywords and domain
    _populate_goal(attributes, kw)
    _populate_audience_segment(attributes, kw)
    _populate_platform(attributes, kw, domain)
    _populate_content_type(attributes, kw)
    _populate_tone(attributes, kw)

    # Themes = keywords (let BoW handle overlap)
    if kw:
        attributes["themes"] = kw

    return {
        "name": f"query_{stable_id}",
        "attributes": attributes,
    }


def _populate_goal(attrs: dict, kw: str):
    """Extract goal from keywords."""
    goals = {
        "awareness": ["awareness", "brand", "reach", "visibility", "impressions"],
        "conversion": ["conversion", "convert", "signup", "purchase", "sale", "revenue", "roi"],
        "engagement": ["engagement", "engage", "likes", "shares", "comments", "interaction"],
        "retention": ["retention", "retain", "churn", "loyalty", "renewal", "lifetime"],
        "acquisition": ["acquisition", "acquire", "new users", "growth", "onboard"],
        "branding": ["branding", "positioning", "identity", "voice", "messaging"],
    }
    for goal, signals in goals.items():
        for signal in signals:
            if signal in kw:
                attrs["goal"] = goal
                return


def _populate_audience_segment(attrs: dict, kw: str):
    """Extract audience segment from keywords."""
    segments = {
        "technical": ["technical", "developer", "engineering", "devops", "api"],
        "business": ["business", "executive", "manager", "leadership", "c-suite"],
        "enterprise": ["enterprise", "large", "fortune"],
        "startup": ["startup", "founder", "early stage", "seed"],
        "consumer": ["consumer", "b2c", "retail", "individual"],
        "smb": ["smb", "small business", "mid market"],
    }
    found = []
    for seg, signals in segments.items():
        for signal in signals:
            if signal in kw:
                found.append(seg)
                break
    if found:
        attrs["audience_segment"] = " ".join(found)


def _populate_platform(attrs: dict, kw: str, domain: str):
    """Extract platform from keywords and domain."""
    platforms = {
        "twitter": ["twitter", "tweet", "x.com"],
        "linkedin": ["linkedin"],
        "instagram": ["instagram", "insta"],
        "facebook": ["facebook", "meta"],
        "tiktok": ["tiktok"],
        "email": ["email", "newsletter"],
        "blog": ["blog", "seo"],
        "youtube": ["youtube"],
    }
    found = []
    for plat, signals in platforms.items():
        for signal in signals:
            if signal in kw:
                found.append(plat)
                break
    if found:
        attrs["platform"] = " ".join(found)
    elif domain == "social":
        attrs["platform"] = "social"
    elif domain == "email":
        attrs["platform"] = "email"
    elif domain == "content":
        attrs["platform"] = "blog"


def _populate_content_type(attrs: dict, kw: str):
    """Extract content type from keywords."""
    types = {
        "article": ["article", "blog post", "long form"],
        "video": ["video", "youtube", "reel"],
        "social_post": ["social post", "tweet", "post"],
        "email": ["email", "newsletter"],
        "webinar": ["webinar", "live session"],
        "case_study": ["case study"],
        "ad": ["ad", "advertisement", "banner"],
        "podcast": ["podcast", "episode"],
        "whitepaper": ["whitepaper", "white paper"],
        "infographic": ["infographic"],
    }
    for ct, signals in types.items():
        for signal in signals:
            if signal in kw:
                attrs["content_type"] = ct
                return


def _populate_tone(attrs: dict, kw: str):
    """Extract tone from keywords."""
    tones = {
        "professional": ["professional", "formal", "corporate"],
        "casual": ["casual", "friendly", "informal", "fun"],
        "technical": ["technical", "detailed", "deep dive"],
        "inspirational": ["inspirational", "motivational", "uplifting"],
        "educational": ["educational", "tutorial", "how to", "guide"],
        "urgent": ["urgent", "limited", "now", "deadline", "flash"],
        "conversational": ["conversational", "chatty", "personal"],
    }
    for tone, signals in tones.items():
        for signal in signals:
            if signal in kw:
                attrs["tone"] = tone
                return


# ============================================================================
# entry_to_record — JSONL exemplar -> concept record
# ============================================================================

def entry_to_record(entry: dict) -> dict:
    """
    Convert a JSONL exemplar entry to a concept record for encoding.

    Expected entry format:
    {
        "entity_type": "campaign",
        "entity_id": "product_launch_q1",
        "goal": "conversion",
        "audience_segment": "technical enterprise",
        "platform": "twitter linkedin",
        "content_type": "article",
        "themes": "product launch api integration",
        "tone": "professional",
        "keywords": "product launch enterprise api integration developer",
        "engagement_metrics": [1250, 340, 89, 45, 156],
        "conversion_metrics": [2100, 180, 42, 95, 12],
        "reach_metrics": [45000, 12000, 3.2, 0.15],
        "metadata": {...}
    }
    """
    entity_type = entry.get("entity_type", "none")
    entity_id = entry.get("entity_id", "")
    concept_text = entity_id or f"{entity_type}_{hashlib.md5(str(entry).encode()).hexdigest()[:8]}"

    # Build keywords from themes + explicit keywords
    themes = entry.get("themes", "")
    keywords = entry.get("keywords", "")
    if isinstance(keywords, list):
        keywords = " ".join(keywords)

    return {
        "concept_text": concept_text,
        "attributes": {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "goal": entry.get("goal", "none"),
            "audience_segment": entry.get("audience_segment", ""),
            "platform": entry.get("platform", ""),
            "content_type": entry.get("content_type", "none"),
            "themes": themes,
            "tone": entry.get("tone", "none"),
            "keywords": keywords if keywords else themes,
            "engagement_metrics": entry.get("engagement_metrics", [0.0] * 5),
            "conversion_metrics": entry.get("conversion_metrics", [0.0] * 5),
            "reach_metrics": entry.get("reach_metrics", [0.0] * 4),
        },
        "metadata": {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "campaign_id": entry.get("campaign_id", ""),
            **(entry.get("metadata", {})),
        },
    }
