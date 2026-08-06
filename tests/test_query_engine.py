"""Unit tests for QueryEngine."""

import unittest
from pathlib import Path
from app.config import AppConfig
from app.database.connection import DatabaseConnection
from app.models.schemas import ExtractionResult
from app.query_engine.query_engine import QueryEngine
from app.world_model.world_model import WorldModel


class TestQueryEngine(unittest.TestCase):

    def setUp(self) -> None:
        self.config = AppConfig(
            db_path=Path(":memory:"),
            world_slice_max_tokens=100,
        )
        self.db = DatabaseConnection(Path(":memory:"))
        self.db.initialize()
        self.world_model = WorldModel(self.db, self.config)
        self.query_engine = QueryEngine(self.config)

    def tearDown(self) -> None:
        self.db.close()

    def test_build_slice(self) -> None:
        extraction = ExtractionResult(
            room_name="Library",
            room_description="Shelves of old leather-bound books line the stone walls.",
            exits=["south"],
            objects=["ancient_book"],
        )
        self.world_model.process_observation(extraction, step=1)

        slice_obj = self.query_engine.build_slice(
            world_model=self.world_model,
            objective="Find the chalice",
            valid_actions=["go south", "examine ancient_book"],
        )

        self.assertEqual(slice_obj.current_room, "Library")
        self.assertIn("go south", slice_obj.valid_actions)

        # Check prompt text rendering
        prompt_text = slice_obj.to_prompt_text()
        self.assertIn("OBJECTIVE: Find the chalice", prompt_text)
        self.assertIn("CURRENT ROOM: Library", prompt_text)


if __name__ == "__main__":
    unittest.main()
