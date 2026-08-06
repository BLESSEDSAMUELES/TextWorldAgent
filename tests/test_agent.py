"""Unit & Integration tests for TextWorldAgent."""

import unittest
from pathlib import Path

from app.agent.action_parser import ActionParser
from app.agent.agent import TextWorldAgent
from app.config import AppConfig
from app.database.connection import DatabaseConnection
from app.environment.custom_env import CustomEnvironment
from app.extractor.extractor import Extractor
from app.llm.llm_client import LLMClient
from app.query_engine.query_engine import QueryEngine
from app.world_model.world_model import WorldModel


class TestTextWorldAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.config = AppConfig(db_path=Path(":memory:"))
        self.db = DatabaseConnection(Path(":memory:"))
        self.db.initialize()

        self.world_model = WorldModel(self.db, self.config)
        self.query_engine = QueryEngine(self.config)
        self.llm_client = LLMClient(self.config)
        self.extractor = Extractor()
        self.action_parser = ActionParser()

        self.agent = TextWorldAgent(
            world_model=self.world_model,
            query_engine=self.query_engine,
            llm_client=self.llm_client,
            extractor=self.extractor,
            action_parser=self.action_parser,
            config=self.config,
        )

        self.env = CustomEnvironment(Path("app/environment/worlds/sample_world.json"))

    def tearDown(self) -> None:
        self.db.close()

    def test_agent_single_step(self) -> None:
        obs = self.env.reset()
        next_obs, reward, done, info = self.agent.step(self.env, current_step=1, last_obs=obs)

        self.assertIsNotNone(next_obs)
        self.assertIn("action", info)
        self.assertIn(info["source"], ("llm", "heuristic_fallback"))

    def test_agent_run_multi_step(self) -> None:
        logs = self.agent.run(self.env, max_steps=10)
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]["step"], 1)

    def test_loop_prevention(self) -> None:
        # Simulate repeating the same action
        self.agent._recent_actions = ["look", "look", "look"]
        is_loop = self.agent._is_looping("look")
        self.assertTrue(is_loop)


if __name__ == "__main__":
    unittest.main()
