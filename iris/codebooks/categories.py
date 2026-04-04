"""
Object categories, color names, clothing, and relationship primitives.
"""

# ============================================================================
# Object Categories (~60)
# ============================================================================

OBJECT_CATEGORIES = {
    # Beings
    "beings": [
        "human", "animal", "bird", "fish", "insect", "creature",
    ],
    # Nature
    "nature": [
        "landscape", "sky", "water", "ocean", "river", "lake",
        "mountain", "forest", "tree", "flower", "vegetation", "terrain",
        "rock", "sand", "snow", "cloud",
    ],
    # Built environment
    "built": [
        "building", "house", "skyscraper", "bridge", "road", "street",
        "interior", "room", "wall", "floor", "ceiling",
        "vehicle", "car", "truck", "bicycle", "boat", "airplane",
        "furniture", "chair", "table", "bed", "shelf",
    ],
    # Objects
    "objects": [
        "tool", "device", "phone", "computer", "screen", "camera",
        "container", "bottle", "cup", "box", "bag",
        "food", "fruit", "plate", "utensil",
        "book", "pen", "paper", "document",
    ],
    # Abstract / UI
    "abstract": [
        "text_element", "symbol", "icon", "logo",
        "shape", "pattern", "diagram", "chart", "graph",
        "interface", "button", "form", "menu",
    ],
}

# Flat list of all categories
ALL_OBJECT_CATEGORIES = []
for group in OBJECT_CATEGORIES.values():
    ALL_OBJECT_CATEGORIES.extend(group)

# ============================================================================
# Color Names (~30)
# ============================================================================

COLOR_NAMES = [
    # Primary / secondary
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    # Neutrals
    "black", "white", "gray", "brown", "beige", "cream", "ivory",
    # Extended
    "navy", "teal", "cyan", "magenta", "coral", "gold", "silver",
    "olive", "maroon", "burgundy", "turquoise", "lavender", "peach",
    # Descriptors (used in palette)
    "warm_tones", "cool_tones", "muted", "vibrant", "pastel", "earth_tones",
    "monochrome", "high_saturation", "desaturated",
]

# ============================================================================
# Clothing Items (~30)
# ============================================================================

CLOTHING_ITEMS = [
    "shirt", "blouse", "sweater", "hoodie", "jacket", "coat",
    "suit", "blazer", "vest", "dress", "skirt", "pants", "jeans",
    "shorts", "uniform", "scrubs", "lab_coat", "athletic_wear",
    "swimwear", "formal_wear", "casual_wear", "business_casual",
    "hat", "glasses", "sunglasses", "scarf", "tie", "watch",
    "jewelry", "mask", "helmet",
]

# ============================================================================
# Spatial Relationships (~20)
# ============================================================================

RELATIONSHIPS = [
    "contains", "beside", "above", "below", "overlapping",
    "facing", "interacting", "grouped", "isolated",
    "in_front_of", "behind", "surrounding", "between",
    "touching", "holding", "wearing", "riding", "sitting_on",
    "standing_on", "leaning_against",
]
