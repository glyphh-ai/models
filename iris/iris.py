"""
Iris — High-level API for structured visual encoding.

Usage:
    from iris import Iris

    iris = Iris()
    glyph = iris.encode("photo.jpg")
    spec = iris.to_json(glyph)
    prompt = iris.to_prompt(glyph)
    similar = iris.search(glyph, top_k=10)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# SDK imports
from glyphh.core.types import Glyph, Concept
from glyphh.core.ops import cosine_similarity
from glyphh.encoder.base import Encoder

from encoder import ENCODER_CONFIG, encode_query, _extract_all_features, _load_image, _fill_defaults
import exports


@dataclass
class IrisGlyph:
    """
    An encoded image representation.

    Attributes:
        glyph: The HDC Glyph (10K-dim bipolar vector with hierarchical structure)
        features: Raw extracted features dict
        image_path: Source image path (if encoded from file)
    """
    glyph: Glyph
    features: Dict[str, Any]
    image_path: Optional[str] = None

    @property
    def all_text(self) -> str:
        """All detected text content."""
        return self.features.get("text_content", "")

    def to_json(self) -> Dict[str, Any]:
        """Structured JSON specification."""
        return exports.to_json(self.features)

    def to_prompt(self) -> str:
        """Natural language prompt fragment."""
        return exports.to_prompt(self.features)


class Iris:
    """
    Structured visual encoder — the high-level API for Iris.

    Encodes images into searchable, manipulable HDC glyphs.
    """

    def __init__(self):
        self._encoder = Encoder(ENCODER_CONFIG)
        self._index: List[IrisGlyph] = []  # In-memory index for search

    def encode(self, image_path: str) -> IrisGlyph:
        """
        Encode an image to an IrisGlyph.

        Args:
            image_path: Path to image file (jpg, png, etc.)

        Returns:
            IrisGlyph with HDC vector and extracted features
        """
        features = _extract_all_features(image_path)
        concept = Concept(
            name=os.path.basename(image_path),
            attributes=features,
        )
        glyph = self._encoder.encode(concept)

        return IrisGlyph(
            glyph=glyph,
            features=features,
            image_path=image_path,
        )

    def encode_features(self, features: Dict[str, Any], name: str = "manual") -> IrisGlyph:
        """
        Encode pre-extracted features to an IrisGlyph.

        Useful when features are extracted externally or from exemplars.
        """
        _fill_defaults(features)
        concept = Concept(name=name, attributes=features)
        glyph = self._encoder.encode(concept)
        return IrisGlyph(glyph=glyph, features=features)

    def decode(self, iris_glyph: IrisGlyph) -> Dict[str, Any]:
        """Decode an IrisGlyph to structured specification."""
        return iris_glyph.to_json()

    def add_to_index(self, iris_glyph: IrisGlyph):
        """Add a glyph to the in-memory search index."""
        self._index.append(iris_glyph)

    def search(self, query_glyph: IrisGlyph, top_k: int = 10) -> List[Tuple[IrisGlyph, float]]:
        """
        Search the index for similar images.

        Args:
            query_glyph: Query IrisGlyph
            top_k: Number of results to return

        Returns:
            List of (IrisGlyph, similarity_score) tuples, sorted by score
        """
        if not self._index:
            return []

        scores = []
        query_vec = query_glyph.glyph.global_cortex.data
        for idx_glyph in self._index:
            sim = cosine_similarity(query_vec, idx_glyph.glyph.global_cortex.data)
            scores.append((idx_glyph, float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search_by_role(
        self,
        query_glyph: IrisGlyph,
        layer_name: str,
        top_k: int = 10,
    ) -> List[Tuple[IrisGlyph, float]]:
        """
        Search the index using only a specific layer's cortex.

        Args:
            query_glyph: Query IrisGlyph
            layer_name: Layer to search by (e.g., "lighting", "identity", "pose")
            top_k: Number of results

        Returns:
            List of (IrisGlyph, similarity_score) tuples
        """
        if not self._index:
            return []
        if layer_name not in query_glyph.glyph.layers:
            return []

        query_vec = query_glyph.glyph.layers[layer_name].cortex.data
        scores = []
        for idx_glyph in self._index:
            if layer_name in idx_glyph.glyph.layers:
                idx_vec = idx_glyph.glyph.layers[layer_name].cortex.data
                sim = cosine_similarity(query_vec, idx_vec)
                scores.append((idx_glyph, float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def swap(
        self,
        iris_glyph: IrisGlyph,
        role_name: str,
        new_value: Any,
    ) -> IrisGlyph:
        """
        Create a new glyph with one role's value swapped.

        Args:
            iris_glyph: Source IrisGlyph
            role_name: Role to swap (e.g., "direction", "expression")
            new_value: New value for the role

        Returns:
            New IrisGlyph with the swapped role
        """
        new_features = dict(iris_glyph.features)
        new_features[role_name] = new_value
        return self.encode_features(new_features, name=f"swap_{role_name}")

    def combine(
        self,
        glyph_a: IrisGlyph,
        roles_a: List[str],
        glyph_b: IrisGlyph,
        roles_b: List[str],
    ) -> IrisGlyph:
        """
        Combine features from two glyphs into a new one.

        Args:
            glyph_a: First source IrisGlyph
            roles_a: Roles to take from glyph_a
            glyph_b: Second source IrisGlyph
            roles_b: Roles to take from glyph_b

        Returns:
            New IrisGlyph with combined features
        """
        combined = {}
        for role in roles_a:
            if role in glyph_a.features:
                combined[role] = glyph_a.features[role]
        for role in roles_b:
            if role in glyph_b.features:
                combined[role] = glyph_b.features[role]
        return self.encode_features(combined, name="combined")

    def diff(
        self,
        glyph_a: IrisGlyph,
        glyph_b: IrisGlyph,
    ) -> Dict[str, float]:
        """
        Compare two images by role, returning per-role similarity scores.

        Returns:
            Dict mapping role names to similarity (0.0 = different, 1.0 = same)
        """
        return exports.diff(glyph_a.features, glyph_b.features)

    def to_controlnet(
        self,
        iris_glyph: IrisGlyph,
        image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Get ControlNet-ready signals (pose, depth, canny edges)."""
        if image is None and iris_glyph.image_path:
            image = _load_image(iris_glyph.image_path)
        return exports.to_controlnet(iris_glyph.features, image)

    def to_prompt(self, iris_glyph: IrisGlyph) -> str:
        """Generate a natural language prompt fragment."""
        return iris_glyph.to_prompt()

    def to_json(self, iris_glyph: IrisGlyph) -> Dict[str, Any]:
        """Get structured JSON specification."""
        return iris_glyph.to_json()
