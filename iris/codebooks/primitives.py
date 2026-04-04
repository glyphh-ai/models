"""
Iris visual primitives — finite composable vocabulary for image encoding.

~400 primitives organized by visual dimension. These are the building
blocks for all Iris glyph encoding — no infinite dictionaries.
"""

# ============================================================================
# Spatial Primitives
# ============================================================================

SPATIAL = [
    "center", "left", "right", "top", "bottom",
    "top_left", "top_right", "bottom_left", "bottom_right",
    "foreground", "midground", "background",
    "close", "medium", "wide", "extreme_wide",
]

# ============================================================================
# Composition Primitives
# ============================================================================

COMPOSITION = [
    "center", "rule_of_thirds", "symmetric", "diagonal",
    "frame_within_frame", "leading_lines", "isolated",
    "panoramic", "close_up", "none",
]

# ============================================================================
# Lighting Primitives
# ============================================================================

LIGHTING_DIRECTION = [
    "front", "side_left", "side_right", "back", "top",
    "bottom", "rim", "ambient", "none",
]

LIGHTING_QUALITY = [
    "soft", "hard", "natural", "artificial", "mixed", "none",
]

LIGHTING_CONTRAST = [
    "high", "medium", "low", "none",
]

# ============================================================================
# Mood / Atmosphere Primitives
# ============================================================================

MOOD = [
    "calm", "tense", "joyful", "somber", "energetic",
    "peaceful", "chaotic", "mysterious", "clinical",
    "dramatic", "romantic", "nostalgic", "futuristic",
    "warm", "cool", "neutral",
]

# ============================================================================
# Depth Primitives
# ============================================================================

DEPTH = [
    "flat", "shallow", "deep", "layered", "compressed",
]

# ============================================================================
# Time of Day Primitives
# ============================================================================

TIME_OF_DAY = [
    "dawn", "day", "dusk", "night", "golden_hour", "blue_hour",
]

# ============================================================================
# Expression Primitives
# ============================================================================

EXPRESSIONS = [
    "neutral", "happy", "sad", "angry", "surprised",
    "fearful", "disgusted", "contempt", "calm", "confused",
    "focused", "determined", "pensive", "amused", "excited", "bored",
    "none",
]

# ============================================================================
# Age Group Primitives
# ============================================================================

AGE_GROUPS = [
    "infant", "child", "teenager", "twenties", "thirties",
    "forties", "fifties", "sixties_plus", "none",
]

# ============================================================================
# Scene Category Primitives
# ============================================================================

SCENE_CATEGORIES = [
    "indoor", "outdoor", "studio", "urban", "nature",
    "underwater", "aerial", "abstract", "none",
]

# ============================================================================
# Text Primitives
# ============================================================================

TEXT_ROLES = [
    "title", "heading", "body", "label", "button",
    "watermark", "caption", "logo", "none",
]

TEXT_POSITIONS = [
    "top", "center", "bottom", "left", "right", "overlay", "none",
]

TEXT_STYLES = [
    "serif", "sans", "mono", "script", "handwritten",
]

TEXT_WEIGHTS = [
    "light", "regular", "bold", "heavy",
]

TEXT_SCALES = [
    "dominant", "prominent", "balanced", "subtle", "fine_print",
]

# ============================================================================
# Aggregate
# ============================================================================

ALL_PRIMITIVES = (
    SPATIAL + COMPOSITION + LIGHTING_DIRECTION + LIGHTING_QUALITY +
    LIGHTING_CONTRAST + MOOD + DEPTH + TIME_OF_DAY + EXPRESSIONS +
    AGE_GROUPS + SCENE_CATEGORIES + TEXT_ROLES + TEXT_POSITIONS +
    TEXT_STYLES + TEXT_WEIGHTS + TEXT_SCALES
)
