"""Tests for Iris intent extraction — text query parsing."""

from intent import extract_intent


class TestIntentExtraction:
    """Test visual query intent parsing."""

    def test_search_action(self):
        result = extract_intent("find photos with soft lighting")
        assert result["action"] == "search"

    def test_compare_action(self):
        result = extract_intent("compare these two portraits")
        assert result["action"] == "compare"

    def test_match_action(self):
        result = extract_intent("match similar images")
        assert result["action"] == "match"

    def test_analyze_action(self):
        result = extract_intent("what is in this image")
        assert result["action"] == "analyze"

    def test_lighting_target(self):
        result = extract_intent("find photos with soft lighting")
        assert result["target"] == "lighting"

    def test_face_target(self):
        result = extract_intent("show me face portraits with happy expression")
        assert result["target"] == "face"

    def test_color_target(self):
        result = extract_intent("search for blue images")
        assert result["target"] == "color"

    def test_scene_target(self):
        result = extract_intent("find outdoor nature photos")
        assert result["target"] == "scene"

    def test_composition_target(self):
        result = extract_intent("show symmetric compositions")
        assert result["target"] == "composition"

    def test_text_target(self):
        result = extract_intent("find images with text watermark")
        assert result["target"] == "text"

    def test_keywords_extracted(self):
        result = extract_intent("find dramatic portraits in studio")
        assert "dramatic" in result["keywords"]
        assert "portraits" in result["keywords"] or "portrait" in result["keywords"]
        assert "studio" in result["keywords"]

    def test_modifiers_extracted(self):
        result = extract_intent("find very similar soft lighting")
        assert "very" in result["modifiers"]
        assert "similar" in result["modifiers"]
        assert "soft" in result["modifiers"]

    def test_default_action_is_search(self):
        result = extract_intent("blue sky landscape")
        assert result["action"] == "search"

    def test_empty_query(self):
        result = extract_intent("")
        assert result["action"] == "search"
        assert result["target"] == "none"
