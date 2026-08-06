"""
World Model — the central orchestrator for persistent world state.

Sits between the Extractor and the Database. All world state mutations
flow through this module. Provides:
- Observation processing with reconciliation
- NetworkX graph for spatial reasoning
- Current state snapshots for debugging
- Post-action updates
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx

from app.config import AppConfig
from app.database.connection import DatabaseConnection
from app.database.repository import (
    ConnectionRepository,
    FactRepository,
    InventoryRepository,
    MemoryRepository,
    NPCRepository,
    ObjectRepository,
    RoomRepository,
    RoomStateRepository,
)
from app.models.schemas import (
    AgentAction,
    AgentMemory,
    ActionType,
    Connection,
    ExtractionResult,
    GameObject,
    InventoryItem,
    MemoryType,
    ObservedFact,
)
from app.world_model.reconciler import Reconciler


class WorldModel:
    """Persistent world model backed by SQLite and NetworkX."""

    def __init__(self, db: DatabaseConnection, config: AppConfig) -> None:
        self._db = db
        self._config = config

        # Initialize repositories
        self.rooms = RoomRepository(db)
        self.connections = ConnectionRepository(db)
        self.objects = ObjectRepository(db)
        self.inventory = InventoryRepository(db)
        self.npcs = NPCRepository(db)
        self.room_states = RoomStateRepository(db)
        self.facts = FactRepository(db)
        self.memories = MemoryRepository(db)

        # Initialize reconciler
        self._reconciler = Reconciler(
            rooms=self.rooms,
            connections=self.connections,
            objects=self.objects,
            inventory=self.inventory,
            npcs=self.npcs,
            room_states=self.room_states,
            facts=self.facts,
            memories=self.memories,
            config=config,
        )

        # Current state tracking
        self._current_room_id: Optional[int] = None
        self._current_room_name: str = ""
        self._last_room_id: Optional[int] = None
        self._last_move_direction: Optional[str] = None
        self._graph: nx.DiGraph = nx.DiGraph()

    @property
    def current_room_id(self) -> Optional[int]:
        """The ID of the room the agent is currently in."""
        return self._current_room_id

    @property
    def current_room_name(self) -> str:
        """The name of the room the agent is currently in."""
        return self._current_room_name

    def process_observation(
        self,
        extraction: ExtractionResult,
        step: int,
    ) -> None:
        """
        Process an extraction result, updating all relevant tables.

        This is the main entry point called after each game step.
        """
        room_name = extraction.room_name or self._current_room_name
        if not room_name:
            return

        self._current_room_name = room_name
        self._current_room_id = self._reconciler.reconcile(
            extraction=extraction,
            current_room_name=room_name,
            step=step,
        )

        if (
            self._last_room_id is not None
            and self._last_move_direction is not None
            and self._current_room_id is not None
            and self._last_room_id != self._current_room_id
        ):
            self.connections.upsert(
                Connection(
                    from_room_id=self._last_room_id,
                    to_room_id=self._current_room_id,
                    direction=self._last_move_direction,
                )
            )
            opposite = {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
                "up": "down",
                "down": "up",
            }.get(self._last_move_direction)
            if opposite:
                self.connections.upsert(
                    Connection(
                        from_room_id=self._current_room_id,
                        to_room_id=self._last_room_id,
                        direction=opposite,
                    )
                )
            self._last_room_id = None
            self._last_move_direction = None

        self._rebuild_graph()

    def update_after_action(
        self,
        action: AgentAction,
        observation: str,
        step: int,
    ) -> None:
        """
        Update world model based on an action's result.

        Handles inventory changes, room transitions, and state updates.
        """
        if action.action_type == ActionType.TAKE and action.target:
            self._handle_take(action.target, step)

        elif action.action_type == ActionType.DROP and action.target:
            self._handle_drop(action.target, step)

        elif action.action_type == ActionType.GO and action.target:
            self._last_room_id = self._current_room_id
            self._last_move_direction = action.target.replace("go ", "").strip().lower()

        elif action.action_type == ActionType.UNLOCK and action.target:
            self._handle_unlock(action.target, observation, step)

        # Record action as memory
        self.memories.add(AgentMemory(
            memory_type=MemoryType.OBSERVATION,
            content=f"Action: {action.raw_command}",
            relevance=1.0,
            created_step=step,
        ))

    def record_failure(self, action: str, reason: str, step: int) -> None:
        """Record a failed action for anti-loop detection."""
        self.memories.add(AgentMemory(
            memory_type=MemoryType.FAILURE,
            content=f"Failed: {action} - {reason}",
            relevance=1.0,
            created_step=step,
        ))

    def get_room_graph(self) -> nx.DiGraph:
        """Return the current room connectivity graph."""
        return self._graph

    def get_inventory_names(self) -> list[str]:
        """Get names of items in the player's inventory."""
        return self.inventory.get_object_names()

    def get_visited_room_names(self) -> list[str]:
        """Get names of all visited rooms."""
        return [r.name for r in self.rooms.get_all_visited()]

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _handle_take(self, target: str, step: int) -> None:
        """Move an object from the room to inventory."""
        obj = self.objects.get_by_name(target)
        if obj and obj.id is not None:
            self.objects.move_to_room(obj.id, None)
            self.inventory.add(InventoryItem(
                object_id=obj.id,
                acquired_step=step,
            ))
            self.facts.upsert(ObservedFact(
                subject=target,
                predicate="is_in",
                object="inventory",
                source_step=step,
            ))

    def _handle_drop(self, target: str, step: int) -> None:
        """Move an object from inventory to the current room."""
        obj = self.objects.get_by_name(target)
        if obj and obj.id is not None and self._current_room_id:
            self.objects.move_to_room(obj.id, self._current_room_id)
            self.inventory.remove_by_object(obj.id)
            self.facts.upsert(ObservedFact(
                subject=target,
                predicate="is_in",
                object=self._current_room_name,
                source_step=step,
            ))

    def _handle_unlock(self, target: str, observation: str, step: int) -> None:
        """Update state after an unlock action."""
        self.facts.upsert(ObservedFact(
            subject=target,
            predicate="state_is",
            object="unlocked",
            source_step=step,
        ))

    def _rebuild_graph(self) -> None:
        """Rebuild the NetworkX graph from the connections table."""
        self._graph = nx.DiGraph()

        # Add all rooms as nodes
        for room in self.rooms.get_all():
            self._graph.add_node(room.id, name=room.name, visited=room.visited)

        # Add all connections as edges
        for conn in self.connections.get_all():
            if conn.to_room_id is not None:
                self._graph.add_edge(
                    conn.from_room_id,
                    conn.to_room_id,
                    direction=conn.direction,
                    locked=conn.locked,
                )
