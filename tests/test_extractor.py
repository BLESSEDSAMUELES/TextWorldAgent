"""Unit tests for observation extractor."""

import unittest
from app.extractor.extractor import Extractor


class TestExtractor(unittest.TestCase):

    def setUp(self) -> None:
        self.extractor = Extractor()

    def test_extract_room_name(self) -> None:
        text = "== Entrance Hall ==\nA dusty hall.\nExits: north, south"
        result = self.extractor.extract(text)
        self.assertEqual(result.room_name, "Entrance Hall")

    def test_extract_exits(self) -> None:
        text = "== Kitchen ==\nExits: north, east and down"
        result = self.extractor.extract(text)
        self.assertEqual(result.exits, ["north", "east", "down"])

    def test_extract_objects(self) -> None:
        text = "== Library ==\nYou see: ancient_book, golden_key"
        result = self.extractor.extract(text)
        self.assertIn("ancient_book", result.objects)
        self.assertIn("golden_key", result.objects)

    def test_extract_npcs(self) -> None:
        text = "== Study ==\nThe ghost of blackwood is here."
        result = self.extractor.extract(text)
        self.assertTrue(any("ghost" in npc for npc in result.npcs))

    def test_extract_state_changes(self) -> None:
        text = "You pick up the iron_key. You unlock the cabinet."
        result = self.extractor.extract(text)
        self.assertEqual(result.state_changes.get("cabinet"), "unlocked")
        self.assertEqual(result.state_changes.get("iron_key"), "taken")


if __name__ == "__main__":
    unittest.main()
