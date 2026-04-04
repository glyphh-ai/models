"""Tests for Iris codebooks — primitives and categories."""

import pytest

from codebooks.primitives import (
    SPATIAL, COMPOSITION, LIGHTING_DIRECTION, LIGHTING_QUALITY,
    LIGHTING_CONTRAST, MOOD, DEPTH, TIME_OF_DAY, EXPRESSIONS,
    AGE_GROUPS, SCENE_CATEGORIES, TEXT_ROLES, TEXT_POSITIONS,
    ALL_PRIMITIVES,
)
from codebooks.categories import (
    OBJECT_CATEGORIES, ALL_OBJECT_CATEGORIES, COLOR_NAMES,
    CLOTHING_ITEMS, RELATIONSHIPS,
)


class TestPrimitives:
    """Validate primitive vocabulary completeness."""

    def test_spatial_not_empty(self):
        assert len(SPATIAL) >= 15

    def test_composition_not_empty(self):
        assert len(COMPOSITION) >= 8

    def test_lighting_direction_not_empty(self):
        assert len(LIGHTING_DIRECTION) >= 8

    def test_expressions_not_empty(self):
        assert len(EXPRESSIONS) >= 15

    def test_all_primitives_count(self):
        # Should be roughly 150+ universal primitives
        assert len(ALL_PRIMITIVES) >= 100

    def test_no_duplicate_expressions(self):
        assert len(EXPRESSIONS) == len(set(EXPRESSIONS))

    def test_no_duplicate_age_groups(self):
        assert len(AGE_GROUPS) == len(set(AGE_GROUPS))

    def test_scene_categories_include_basics(self):
        for cat in ["indoor", "outdoor", "studio", "nature"]:
            assert cat in SCENE_CATEGORIES


class TestCategories:
    """Validate object categories and color names."""

    def test_object_categories_groups(self):
        assert "beings" in OBJECT_CATEGORIES
        assert "nature" in OBJECT_CATEGORIES
        assert "built" in OBJECT_CATEGORIES
        assert "objects" in OBJECT_CATEGORIES
        assert "abstract" in OBJECT_CATEGORIES

    def test_all_object_categories_flat(self):
        assert len(ALL_OBJECT_CATEGORIES) >= 50

    def test_color_names(self):
        for color in ["red", "blue", "green", "white", "black"]:
            assert color in COLOR_NAMES

    def test_clothing_items(self):
        assert len(CLOTHING_ITEMS) >= 20

    def test_relationships(self):
        assert len(RELATIONSHIPS) >= 15
        for rel in ["contains", "beside", "above", "below"]:
            assert rel in RELATIONSHIPS
