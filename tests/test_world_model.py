"""Unit tests for WorldModel and Reconciler."""

import unittest
from pathlib import Path
from app.config import AppConfig
from app.database.connection import DatabaseConnection
from app.models.schemas import ExtractionResult, AgentAction, ActionType
from app.world_model.world_model import WorldModel


class TestWorldModel(unittest.TestCase):

    def setUp(self) -> None:
        self.config = AppConfig(db_path=Path(":memory:"))
        self.db = DatabaseConnection(Path(":memory:"))
        self.db.initialize()
        self.world_model = WorldModel(self.db, self.config)

    def tearDown(self) -> None:
        self.db.close()

    def test_process_observation(self) -> None:
        extraction = ExtractionResult(
            room_name="Entrance Hall",
            room_description="A large hall.",
            exits=["north", "east"],
            objects=["key", "map"],
            npcs=[],
        )
        self.world_model.process_observation(extraction, step=1)

        self.assertEqual(self.world_model.current_room_name, "Entrance Hall")
        self.assertIsNotNone(self.world_model.current_room_id)

        # Verify networkx graph
        graph = self.world_model.get_room_graph()
        self.assertEqual(len(graph.nodes), 1)

    def test_inventory_updates(self) -> None:
        extraction = ExtractionResult(
            room_name="Kitchen",
            objects=["bread"],
        )
        self.world_model.process_observation(extraction, step=1)

        # Action: take bread
        action = AgentAction(action_type=ActionType.TAKE, target="bread", raw_command="take bread")
        self.world_model.update_after_action(action, "You pick up the bread.", step=2)

        inventory = self.world_model.get_inventory_names()
        self.assertIn("bread", inventory)


if __name__ == "__main__":
    unittest.main()
