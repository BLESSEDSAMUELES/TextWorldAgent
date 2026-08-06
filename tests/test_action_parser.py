"""Unit tests for action parser."""

import unittest
from app.agent.action_parser import ActionParser
from app.models.schemas import ActionType


class TestActionParser(unittest.TestCase):

    def setUp(self) -> None:
        self.parser = ActionParser()
        self.valid_actions = ["go north", "take key", "examine box", "unlock door", "look"]

    def test_exact_match(self) -> None:
        action = self.parser.parse("go north", self.valid_actions)
        self.assertEqual(action.action_type, ActionType.GO)
        self.assertEqual(action.raw_command, "go north")

    def test_messy_llm_output_first_line(self) -> None:
        output = "Action: take key.\nBecause we need the key to open the door."
        action = self.parser.parse(output, self.valid_actions)
        self.assertEqual(action.action_type, ActionType.TAKE)
        self.assertEqual(action.raw_command, "take key")

    def test_fuzzy_match(self) -> None:
        output = "go nroth"
        action = self.parser.parse(output, self.valid_actions)
        self.assertEqual(action.raw_command, "go north")

    def test_embedded_action(self) -> None:
        output = "I think the best option right now is to examine box to see what is inside."
        action = self.parser.parse(output, self.valid_actions)
        self.assertEqual(action.raw_command, "examine box")

    def test_fallback(self) -> None:
        output = "gibberish non matching text"
        action = self.parser.parse(output, self.valid_actions)
        self.assertEqual(action.raw_command, "look")


if __name__ == "__main__":
    unittest.main()
