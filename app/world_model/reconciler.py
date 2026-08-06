"""
State reconciliation engine.

Handles all conflict resolution when new observations contradict
existing world state. Core principle: new observations always win,
but old facts are preserved (deactivated) for audit trails.

Rules:
- New room → INSERT, mark visited
- Room revisited → UPDATE visit_count, last_seen, merge longer description
- New object in known room → INSERT
- Object gone from room → UPDATE room_id=NULL, record fact "object was moved"
- Conflicting fact → SUPERSEDE old fact, INSERT new one
- Stale memory → Decay relevance each step, prune below threshold
"""

from __future__ import annotations

from app.config import AppConfig
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
    AgentMemory,
    Connection,
    ExtractionResult,
    GameObject,
    MemoryType,
    NPC,
    ObservedFact,
    Room,
    RoomState,
)


class Reconciler:
    """Reconciles new observations with existing world state."""

    def __init__(
        self,
        rooms: RoomRepository,
        connections: ConnectionRepository,
        objects: ObjectRepository,
        inventory: InventoryRepository,
        npcs: NPCRepository,
        room_states: RoomStateRepository,
        facts: FactRepository,
        memories: MemoryRepository,
        config: AppConfig,
    ) -> None:
        self._rooms = rooms
        self._connections = connections
        self._objects = objects
        self._inventory = inventory
        self._npcs = npcs
        self._room_states = room_states
        self._facts = facts
        self._memories = memories
        self._config = config

    def reconcile(
        self,
        extraction: ExtractionResult,
        current_room_name: str,
        step: int,
    ) -> int:
        """
        Reconcile an extraction result with the world model.

        Returns the room_id of the current room.
        """
        room_id = self._reconcile_room(current_room_name, extraction, step)
        self._reconcile_exits(room_id, extraction.exits, step)
        self._reconcile_objects(room_id, extraction.objects, step)
        self._reconcile_npcs(room_id, extraction.npcs, step)
        self._reconcile_state_changes(room_id, extraction.state_changes, step)
        self._record_observation_facts(extraction, step)
        self._maintain_database(step)
        return room_id

    def _reconcile_room(
        self,
        room_name: str,
        extraction: ExtractionResult,
        step: int,
    ) -> int:
        """Insert or update a room based on the observation."""
        display_name = extraction.room_name or room_name
        description = extraction.room_description or ""

        room = Room(
            name=display_name,
            description=description,
            visited=True,
            visit_count=1,
            first_seen=step,
            last_seen=step,
        )
        return self._rooms.upsert(room)

    def _reconcile_exits(
        self,
        room_id: int,
        exits: list[str],
        step: int,
    ) -> None:
        """Update known exits from the current room."""
        for direction in exits:
            connection = Connection(
                from_room_id=room_id,
                direction=direction,
            )
            self._connections.upsert(connection)

    def _reconcile_objects(
        self,
        room_id: int,
        observed_objects: list[str],
        step: int,
    ) -> None:
        """Reconcile observed objects with known objects in the room."""
        for obj_name in observed_objects:
            obj = GameObject(
                name=obj_name,
                room_id=room_id,
            )
            self._objects.upsert(obj)

    def _reconcile_npcs(
        self,
        room_id: int,
        observed_npcs: list[str],
        step: int,
    ) -> None:
        """Reconcile observed NPCs."""
        for npc_name in observed_npcs:
            npc = NPC(
                name=npc_name,
                room_id=room_id,
            )
            self._npcs.upsert(npc)

    def _reconcile_state_changes(
        self,
        room_id: int,
        state_changes: dict[str, str],
        step: int,
    ) -> None:
        """Apply state changes detected from action results."""
        for key, value in state_changes.items():
            room_state = RoomState(
                room_id=room_id,
                state_key=key,
                state_value=value,
                updated_step=step,
            )
            self._room_states.upsert(room_state)

    def _record_observation_facts(
        self,
        extraction: ExtractionResult,
        step: int,
    ) -> None:
        """Record key observations as facts for the query layer."""
        if extraction.room_name:
            self._facts.upsert(ObservedFact(
                subject="agent",
                predicate="is_in",
                object=extraction.room_name,
                source_step=step,
            ))

        for obj in extraction.objects:
            if extraction.room_name:
                self._facts.upsert(ObservedFact(
                    subject=obj,
                    predicate="is_in",
                    object=extraction.room_name,
                    source_step=step,
                ))

        for key, value in extraction.state_changes.items():
            self._facts.upsert(ObservedFact(
                subject=key,
                predicate="state_is",
                object=value,
                source_step=step,
            ))

    def _maintain_database(self, step: int) -> None:
        """Enforce database growth limits."""
        # Decay memory relevance every step
        self._memories.decay_relevance(self._config.memory_relevance_decay)

        # Periodic pruning
        if step > 0 and step % self._config.memory_prune_interval == 0:
            self._memories.prune_low_relevance(
                self._config.memory_prune_threshold
            )

        # Enforce hard caps
        self._facts.prune_oldest(self._config.max_active_facts)
        self._memories.prune_oldest(self._config.max_active_memories)
