"""Color palette feature extractor — dominant colors via K-means clustering."""

import numpy as np
from typing import Any, Dict, List, Tuple
from extractors.base import FeatureExtractor

# CSS3 named colors → RGB for nearest-name lookup
_NAMED_COLORS = {
    "red": (255, 0, 0), "orange": (255, 165, 0), "yellow": (255, 255, 0),
    "green": (0, 128, 0), "blue": (0, 0, 255), "purple": (128, 0, 128),
    "pink": (255, 192, 203), "black": (0, 0, 0), "white": (255, 255, 255),
    "gray": (128, 128, 128), "brown": (139, 69, 19), "beige": (245, 245, 220),
    "navy": (0, 0, 128), "teal": (0, 128, 128), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "coral": (255, 127, 80), "gold": (255, 215, 0),
    "silver": (192, 192, 192), "olive": (128, 128, 0), "maroon": (128, 0, 0),
    "turquoise": (64, 224, 208), "lavender": (230, 230, 250),
}


class ColorExtractor(FeatureExtractor):
    """Extract dominant color palette via K-means clustering."""

    def __init__(self, n_colors: int = 5):
        self.n_colors = n_colors

    def is_available(self) -> bool:
        try:
            from sklearn.cluster import KMeans  # noqa: F401
            return True
        except ImportError:
            return False

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        from sklearn.cluster import KMeans

        # Subsample for speed
        h, w = image.shape[:2]
        step = max(1, int(np.sqrt(h * w / 10000)))
        pixels = image[::step, ::step].reshape(-1, 3).astype(np.float32)

        n_clusters = min(self.n_colors, len(pixels))
        kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=42)
        kmeans.fit(pixels)

        # Sort by cluster size (most dominant first)
        counts = np.bincount(kmeans.labels_, minlength=n_clusters)
        order = np.argsort(-counts)
        centers = kmeans.cluster_centers_[order]

        # Map to named colors
        color_names = [_nearest_color_name(c) for c in centers]
        # Deduplicate while preserving order
        seen = set()
        unique_names = []
        for name in color_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        return {
            "color_palette": " ".join(unique_names),
            "dominant_color": unique_names[0] if unique_names else "none",
            "color_count": len(unique_names),
        }

    def defaults(self) -> Dict[str, Any]:
        return {
            "color_palette": "",
            "dominant_color": "none",
            "color_count": 0,
        }


def _nearest_color_name(rgb: np.ndarray) -> str:
    """Find the nearest named color to an RGB value."""
    min_dist = float("inf")
    best = "gray"
    for name, ref in _NAMED_COLORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if dist < min_dist:
            min_dist = dist
            best = name
    return best
