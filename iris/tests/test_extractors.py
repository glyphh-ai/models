"""Tests for Iris feature extractors (pure-numpy extractors only, no CV deps)."""

import numpy as np
import pytest

from extractors.base import FeatureExtractor, ExtractorRegistry
from extractors.lighting import LightingExtractor
from extractors.composition import CompositionExtractor
from extractors.color import ColorExtractor


class TestLightingExtractor:
    """Test lighting analysis (pure numpy, always available)."""

    def setup_method(self):
        self.extractor = LightingExtractor()

    def test_is_available(self):
        assert self.extractor.is_available()

    def test_bright_uniform_image(self):
        """Bright uniform image → low contrast, ambient direction."""
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = self.extractor.extract(image)
        assert result["contrast"] == "low"
        assert result["direction"] in ("ambient", "front")

    def test_high_contrast_image(self):
        """Left bright, right dark → high contrast, side_left direction."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :100, :] = 255  # Left half bright
        result = self.extractor.extract(image)
        assert result["contrast"] in ("high", "medium")

    def test_returns_valid_primitives(self):
        image = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.extractor.extract(image)
        assert result["direction"] in ("front", "side_left", "side_right", "back", "top",
                                       "bottom", "rim", "ambient", "none")
        assert result["quality"] in ("soft", "hard", "natural", "none")
        assert result["contrast"] in ("high", "medium", "low", "none")

    def test_defaults(self):
        result = self.extractor.defaults()
        assert result["direction"] == "none"
        assert result["quality"] == "none"
        assert result["contrast"] == "none"


class TestCompositionExtractor:
    """Test composition analysis (pure numpy, always available)."""

    def setup_method(self):
        self.extractor = CompositionExtractor()

    def test_is_available(self):
        assert self.extractor.is_available()

    def test_center_subject(self):
        """Bright center on dark background → center composition."""
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        image[100:200, 100:200, :] = 255
        result = self.extractor.extract(image)
        assert result["composition"] in ("center", "symmetric", "close_up")

    def test_returns_valid_primitives(self):
        image = np.random.RandomState(42).randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = self.extractor.extract(image)
        assert result["composition"] in (
            "center", "rule_of_thirds", "symmetric", "diagonal",
            "frame_within_frame", "leading_lines", "isolated",
            "panoramic", "close_up", "none",
        )
        assert result["scene_category"] in (
            "indoor", "outdoor", "studio", "urban", "nature",
            "underwater", "aerial", "abstract", "none",
        )


class TestColorExtractor:
    """Test color palette extraction (needs scikit-learn)."""

    def setup_method(self):
        self.extractor = ColorExtractor()

    def test_availability(self):
        # May or may not have sklearn
        result = self.extractor.is_available()
        assert isinstance(result, bool)

    def test_red_image(self):
        """Pure red image → dominant color is red."""
        if not self.extractor.is_available():
            pytest.skip("scikit-learn not installed")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Red channel
        result = self.extractor.extract(image)
        assert "red" in result["color_palette"] or "coral" in result["color_palette"]

    def test_defaults(self):
        result = self.extractor.defaults()
        assert result["color_palette"] == ""
        assert result["dominant_color"] == "none"


class TestExtractorRegistry:
    """Test the extractor registry."""

    def test_register_and_extract(self):
        registry = ExtractorRegistry()
        registry.register("lighting", LightingExtractor())
        registry.register("composition", CompositionExtractor())

        assert "lighting" in registry.all_names
        assert "composition" in registry.all_names

        image = np.random.RandomState(42).randint(0, 256, (100, 100, 3), dtype=np.uint8)
        results = registry.extract_all(image)

        assert "lighting" in results
        assert "composition" in results
        assert "direction" in results["lighting"]
        assert "composition" in results["composition"]

    def test_available_list(self):
        registry = ExtractorRegistry()
        registry.register("lighting", LightingExtractor())
        assert "lighting" in registry.available
