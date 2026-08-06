"""Unit tests for CustomEnvironment."""

import unittest
from pathlib import Path
from app.environment.custom_env import CustomEnvironment


class TestCustomEnvironment(unittest.TestCase):

    def setUp(self) -> None:
        self.world_path = Path("app/environment/worlds/sample_world.json")
        self.env = CustomEnvironment(self.world_path)

    def test_reset(self) -> None:
        obs = self.env.reset()
        self.assertIn("Observatory Gates", obs)
        self.assertIn("Find the golden chalice", self.env.get_objective())

    def test_valid_actions(self) -> None:
        self.env.reset()
        valid_actions = self.env.get_valid_actions()
        self.assertIn("go north", valid_actions)
        self.assertIn("examine padlock", valid_actions)

    def test_movement(self) -> None:
        self.env.reset()
        self.env.step("examine dark_ivy")
        self.env.step("take gate_key")
        self.env.step("unlock padlock")
        obs, reward, done = self.env.step("go north")
        self.assertIn("Courtyard", obs)
        self.assertGreater(reward, 0)
        self.assertFalse(done)

    def test_take_object(self) -> None:
        self.env.reset()
        self.env.step("examine dark_ivy")
        obs, reward, done = self.env.step("take gate_key")
        self.assertIn("pick up", obs)
        self.assertGreater(reward, 0)


if __name__ == "__main__":
    unittest.main()

