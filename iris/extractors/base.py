"""
Base feature extractor and registry for Iris.

Each extractor wraps a CV model and returns structured features as a dict.
Extractors degrade gracefully — if dependencies are missing, they return
default "unknown" values instead of crashing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """Base class for all Iris feature extractors."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from an image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8

        Returns:
            Dict of extracted features (keys depend on extractor type)
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the required CV dependencies are installed."""

    def safe_extract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features with graceful degradation.

        Returns default values if the extractor is unavailable or fails.
        """
        if not self.is_available():
            logger.debug(f"{self.__class__.__name__}: dependencies not available, returning defaults")
            return self.defaults()
        try:
            return self.extract(image)
        except Exception as e:
            logger.warning(f"{self.__class__.__name__} failed: {e}")
            return self.defaults()

    @abstractmethod
    def defaults(self) -> Dict[str, Any]:
        """Return default values when extraction is unavailable."""


class ExtractorRegistry:
    """Registry of available feature extractors."""

    def __init__(self):
        self._extractors: Dict[str, FeatureExtractor] = {}

    def register(self, name: str, extractor: FeatureExtractor):
        """Register a feature extractor."""
        self._extractors[name] = extractor

    def extract_all(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run all registered extractors on an image."""
        results = {}
        for name, extractor in self._extractors.items():
            results[name] = extractor.safe_extract(image)
        return results

    def get(self, name: str) -> Optional[FeatureExtractor]:
        """Get a specific extractor by name."""
        return self._extractors.get(name)

    @property
    def available(self) -> list:
        """List names of extractors with satisfied dependencies."""
        return [n for n, e in self._extractors.items() if e.is_available()]

    @property
    def all_names(self) -> list:
        """List all registered extractor names."""
        return list(self._extractors.keys())
