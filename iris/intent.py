"""
Iris intent extraction — parse text queries about images.

Extracts visual search intent from natural language queries like
"find photos with soft lighting" or "show me portraits in outdoor settings".
"""

import re
from typing import Dict, List, Optional

# ============================================================================
# Action Verbs → Canonical Actions
# ============================================================================

_ACTION_MAP = {
    # Search
    "find": "search", "search": "search", "show": "search",
    "get": "search", "look": "search", "locate": "search",
    "browse": "search", "explore": "search", "discover": "search",
    # Compare
    "compare": "compare", "diff": "compare", "difference": "compare",
    "versus": "compare", "vs": "compare",
    # Match
    "match": "match", "similar": "match", "like": "match",
    "resembling": "match", "same": "match",
    # Analyze
    "analyze": "analyze", "describe": "analyze", "what": "analyze",
    "identify": "analyze", "detect": "analyze", "recognize": "analyze",
    # Generate
    "generate": "generate", "create": "generate", "make": "generate",
    "swap": "generate", "change": "generate", "replace": "generate",
}

# ============================================================================
# Visual Target Keywords → Target Categories
# ============================================================================

_TARGET_MAP = {
    # Lighting
    "lighting": "lighting", "light": "lighting", "lit": "lighting",
    "shadow": "lighting", "shadows": "lighting", "bright": "lighting",
    "dark": "lighting", "backlit": "lighting", "silhouette": "lighting",
    "exposure": "lighting",
    # Pose
    "pose": "pose", "posture": "pose", "position": "pose",
    "standing": "pose", "sitting": "pose", "walking": "pose",
    "running": "pose", "gesture": "pose", "body": "pose",
    # Face / Identity
    "face": "face", "portrait": "face", "person": "face",
    "expression": "face", "emotion": "face", "smile": "face",
    "people": "face", "headshot": "face",
    # Color
    "color": "color", "palette": "color", "hue": "color",
    "red": "color", "blue": "color", "green": "color",
    "warm": "color", "cool": "color", "vibrant": "color",
    "monochrome": "color", "saturated": "color",
    # Composition
    "composition": "composition", "framing": "composition",
    "layout": "composition", "symmetric": "composition",
    "centered": "composition", "rule_of_thirds": "composition",
    "crop": "composition", "angle": "composition",
    # Text / OCR
    "text": "text", "words": "text", "writing": "text",
    "sign": "text", "label": "text", "title": "text",
    "watermark": "text", "caption": "text",
    # Scene
    "scene": "scene", "background": "scene", "setting": "scene",
    "indoor": "scene", "outdoor": "scene", "studio": "scene",
    "nature": "scene", "urban": "scene", "aerial": "scene",
    # Depth
    "depth": "depth", "distance": "depth", "foreground": "depth",
    "background": "depth", "shallow": "depth", "deep": "depth",
    # Objects
    "object": "objects", "objects": "objects", "thing": "objects",
    "car": "objects", "animal": "objects", "building": "objects",
    "furniture": "objects", "vehicle": "objects", "food": "objects",
}

# ============================================================================
# Modifier Keywords
# ============================================================================

_MODIFIERS = {
    "similar", "same", "like", "matching", "different",
    "opposite", "contrasting", "more", "less", "very",
    "slightly", "extremely", "subtle", "dramatic",
    "soft", "hard", "natural", "artificial",
    "high", "low", "medium", "close", "wide",
}


def extract_intent(query: str) -> Dict[str, str]:
    """
    Extract visual search intent from a text query.

    Args:
        query: Natural language image query

    Returns:
        Dict with keys: action, target, modifiers, keywords
    """
    query_lower = query.lower().strip()
    words = re.findall(r'\w+', query_lower)

    action = "search"  # Default action
    target = "none"
    modifiers = []
    keywords = []

    # Extract action from first verb
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

    # Keywords = all non-stopword tokens
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "with",
                 "in", "on", "at", "to", "for", "of", "and", "or",
                 "me", "my", "i", "you", "it", "that", "this",
                 "find", "show", "get", "search", "look"}
    keywords = [w for w in words if w not in stopwords and len(w) > 1]

    return {
        "action": action,
        "target": target,
        "modifiers": " ".join(modifiers) if modifiers else "",
        "keywords": " ".join(keywords) if keywords else "",
        "all_targets": " ".join(dict.fromkeys(targets_found)) if targets_found else "",
    }
