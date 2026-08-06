"""
Repository layer — clean data access with UPSERT semantics.

Each repository class handles one table and returns Pydantic models,
never raw tuples. All write operations use INSERT OR REPLACE to
enforce the deduplication constraints from the schema.
"""

from __future__ import annotations

import json
from typing import Optional

from app.database.connection import DatabaseConnection
from app.models.schemas import (
    AgentMemory,
    Connection,
    GameObject,
    InventoryItem,
    MemoryType,
    NPC,
    ObservedFact,
    Room,
    RoomState,
)


# =============================================================================
# Room Repository
# =============================================================================


class RoomRepository:
    """CRUD operations for the rooms table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def upsert(self, room: Room) -> int:
        """Insert or update a room. Returns the room id."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO rooms (name, description, visited, visit_count, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = CASE
                        WHEN LENGTH(excluded.description) > LENGTH(rooms.description)
                        THEN excluded.description
                        ELSE rooms.description
                    END,
                    visited = MAX(rooms.visited, excluded.visited),
                    visit_count = rooms.visit_count + 1,
                    last_seen = excluded.last_seen
                """,
                (
                    room.name, room.description, room.visited,
                    room.visit_count, room.first_seen, room.last_seen,
                ),
            )
            cur.execute("SELECT id FROM rooms WHERE name = ?", (room.name,))
            row = cur.fetchone()
            return row["id"]

    def get_by_name(self, name: str) -> Optional[Room]:
        """Fetch a room by name (case and space/underscore insensitive)."""
        clean_name = name.strip().lower()
        alt_name = clean_name.replace("_", " ")
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM rooms
                WHERE LOWER(name) = ? OR LOWER(name) = ? OR LOWER(REPLACE(name, ' ', '_')) = ?
                LIMIT 1
                """,
                (clean_name, alt_name, clean_name),
            )
            row = cur.fetchone()
            return Room(**dict(row)) if row else None

    def get_by_id(self, room_id: int) -> Optional[Room]:
        """Fetch a room by its primary key."""
        with self._db.get_cursor() as cur:
            cur.execute("SELECT * FROM rooms WHERE id = ?", (room_id,))
            row = cur.fetchone()
            return Room(**dict(row)) if row else None

    def get_all_visited(self) -> list[Room]:
        """Return all rooms the agent has visited."""
        with self._db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM rooms WHERE visited = 1 ORDER BY last_seen DESC"
            )
            return [Room(**dict(row)) for row in cur.fetchall()]

    def get_all(self) -> list[Room]:
        """Return all known rooms."""
        with self._db.get_cursor() as cur:
            cur.execute("SELECT * FROM rooms ORDER BY name")
            return [Room(**dict(row)) for row in cur.fetchall()]


# =============================================================================
# Connection Repository
# =============================================================================


class ConnectionRepository:
    """CRUD operations for the connections table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def upsert(self, connection: Connection) -> int:
        """Insert or update a connection. Returns the connection id."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO connections
                    (from_room_id, to_room_id, direction, locked, lock_key)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(from_room_id, direction) DO UPDATE SET
                    to_room_id = COALESCE(excluded.to_room_id, connections.to_room_id),
                    locked = excluded.locked,
                    lock_key = excluded.lock_key
                """,
                (
                    connection.from_room_id, connection.to_room_id,
                    connection.direction, connection.locked, connection.lock_key,
                ),
            )
            cur.execute(
                "SELECT id FROM connections WHERE from_room_id = ? AND direction = ?",
                (connection.from_room_id, connection.direction),
            )
            row = cur.fetchone()
            return row["id"]

    def get_exits(self, room_id: int) -> list[Connection]:
        """Get all exits from a room."""
        with self._db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM connections WHERE from_room_id = ?",
                (room_id,),
            )
            return [Connection(**dict(row)) for row in cur.fetchall()]

    def get_all(self) -> list[Connection]:
        """Get all connections for graph building."""
        with self._db.get_cursor() as cur:
            cur.execute("SELECT * FROM connections")
            return [Connection(**dict(row)) for row in cur.fetchall()]


# =============================================================================
# Object Repository
# =============================================================================


class ObjectRepository:
    """CRUD operations for the objects table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def upsert(self, obj: GameObject) -> int:
        """Insert or update an object. Returns the object id."""
        props_json = json.dumps(obj.properties)
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO objects (name, description, room_id, portable, state, properties)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    room_id = CASE
                        WHEN excluded.room_id IS NOT NULL THEN excluded.room_id
                        ELSE objects.room_id
                    END,
                    description = CASE
                        WHEN LENGTH(excluded.description) > LENGTH(objects.description)
                        THEN excluded.description
                        ELSE objects.description
                    END,
                    state = excluded.state,
                    properties = excluded.properties
                """,
                (
                    obj.name, obj.description, obj.room_id,
                    obj.portable, obj.state, props_json,
                ),
            )
            cur.execute(
                "SELECT id FROM objects WHERE name = ?",
                (obj.name,),
            )
            row = cur.fetchone()
            return row["id"]

    def get_by_room(self, room_id: int) -> list[GameObject]:
        """Get all objects in a room."""
        with self._db.get_cursor() as cur:
            cur.execute("SELECT * FROM objects WHERE room_id = ?", (room_id,))
            return [self._row_to_obj(row) for row in cur.fetchall()]

    def get_by_name(self, name: str) -> Optional[GameObject]:
        """Find an object by name (first match)."""
        with self._db.get_cursor() as cur:
            cur.execute("SELECT * FROM objects WHERE name = ? LIMIT 1", (name,))
            row = cur.fetchone()
            return self._row_to_obj(row) if row else None

    def move_to_room(self, object_id: int, room_id: Optional[int]) -> None:
        """Move an object to a room (or None for consumed)."""
        with self._db.get_cursor() as cur:
            cur.execute(
                "UPDATE objects SET room_id = ? WHERE id = ?",
                (room_id, object_id),
            )

    def update_state(self, object_id: int, state: str) -> None:
        """Update an object's state."""
        with self._db.get_cursor() as cur:
            cur.execute(
                "UPDATE objects SET state = ? WHERE id = ?",
                (state, object_id),
            )

    @staticmethod
    def _row_to_obj(row: dict) -> GameObject:
        """Convert a database row to a GameObject."""
        data = dict(row)
        data["properties"] = json.loads(data.get("properties", "{}"))
        return GameObject(**data)


# =============================================================================
# Inventory Repository
# =============================================================================


class InventoryRepository:
    """CRUD operations for the inventory table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def add(self, item: InventoryItem) -> int:
        """Add an item to inventory."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO inventory (object_id, acquired_step)
                VALUES (?, ?)
                """,
                (item.object_id, item.acquired_step),
            )
            cur.execute(
                "SELECT id FROM inventory WHERE object_id = ?",
                (item.object_id,),
            )
            row = cur.fetchone()
            return row["id"]

    def remove_by_object(self, object_id: int) -> None:
        """Remove an item from inventory by object id."""
        with self._db.get_cursor() as cur:
            cur.execute(
                "DELETE FROM inventory WHERE object_id = ?",
                (object_id,),
            )

    def get_all(self) -> list[InventoryItem]:
        """Get all items in inventory."""
        with self._db.get_cursor() as cur:
            cur.execute("SELECT * FROM inventory")
            return [InventoryItem(**dict(row)) for row in cur.fetchall()]

    def get_object_names(self) -> list[str]:
        """Get names of all inventory items via join."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                SELECT o.name FROM inventory i
                JOIN objects o ON i.object_id = o.id
                ORDER BY i.acquired_step
                """
            )
            return [row["name"] for row in cur.fetchall()]


# =============================================================================
# NPC Repository
# =============================================================================


class NPCRepository:
    """CRUD operations for the npcs table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def upsert(self, npc: NPC) -> int:
        """Insert or update an NPC. Returns the NPC id."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO npcs (name, description, room_id, dialogue, state)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = CASE
                        WHEN LENGTH(excluded.description) > LENGTH(npcs.description)
                        THEN excluded.description
                        ELSE npcs.description
                    END,
                    room_id = excluded.room_id,
                    dialogue = COALESCE(excluded.dialogue, npcs.dialogue),
                    state = excluded.state
                """,
                (npc.name, npc.description, npc.room_id, npc.dialogue, npc.state),
            )
            cur.execute("SELECT id FROM npcs WHERE name = ?", (npc.name,))
            row = cur.fetchone()
            return row["id"]

    def get_by_room(self, room_id: int) -> list[NPC]:
        """Get all NPCs in a room."""
        with self._db.get_cursor() as cur:
            cur.execute("SELECT * FROM npcs WHERE room_id = ?", (room_id,))
            return [NPC(**dict(row)) for row in cur.fetchall()]


# =============================================================================
# Room State Repository
# =============================================================================


class RoomStateRepository:
    """CRUD operations for the room_states table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def upsert(self, room_state: RoomState) -> None:
        """Set a room state key-value pair (replaces if exists)."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO room_states (room_id, state_key, state_value, updated_step)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(room_id, state_key) DO UPDATE SET
                    state_value = excluded.state_value,
                    updated_step = excluded.updated_step
                """,
                (
                    room_state.room_id, room_state.state_key,
                    room_state.state_value, room_state.updated_step,
                ),
            )

    def get_by_room(self, room_id: int) -> list[RoomState]:
        """Get all state entries for a room."""
        with self._db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM room_states WHERE room_id = ?",
                (room_id,),
            )
            return [RoomState(**dict(row)) for row in cur.fetchall()]


# =============================================================================
# Observed Fact Repository
# =============================================================================


class FactRepository:
    """CRUD operations for the observed_facts table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def upsert(self, fact: ObservedFact) -> int:
        """Insert a fact, superseding any existing fact with same subject+predicate."""
        with self._db.get_cursor() as cur:
            # Check for existing conflicting fact
            cur.execute(
                """
                SELECT id, object FROM observed_facts
                WHERE subject = ? AND predicate = ? AND active = 1
                """,
                (fact.subject, fact.predicate),
            )
            existing = cur.fetchone()

            if existing and existing["object"] != fact.object:
                # Supersede the old fact
                old_id = existing["id"]
                cur.execute(
                    "UPDATE observed_facts SET active = 0 WHERE id = ?",
                    (old_id,),
                )
                cur.execute(
                    """
                    INSERT INTO observed_facts
                        (subject, predicate, object, confidence, source_step, superseded_by, active)
                    VALUES (?, ?, ?, ?, ?, NULL, 1)
                    """,
                    (fact.subject, fact.predicate, fact.object,
                     fact.confidence, fact.source_step),
                )
                new_id = cur.lastrowid
                # Link old → new
                cur.execute(
                    "UPDATE observed_facts SET superseded_by = ? WHERE id = ?",
                    (new_id, old_id),
                )
                return new_id  # type: ignore[return-value]
            elif existing:
                # Same fact already exists, just update step
                cur.execute(
                    "UPDATE observed_facts SET source_step = ?, confidence = ? WHERE id = ?",
                    (fact.source_step, fact.confidence, existing["id"]),
                )
                return existing["id"]
            else:
                # Brand new fact
                cur.execute(
                    """
                    INSERT INTO observed_facts
                        (subject, predicate, object, confidence, source_step, active)
                    VALUES (?, ?, ?, ?, ?, 1)
                    """,
                    (fact.subject, fact.predicate, fact.object,
                     fact.confidence, fact.source_step),
                )
                return cur.lastrowid  # type: ignore[return-value]

    def get_active(self, limit: int = 50) -> list[ObservedFact]:
        """Get active facts ordered by recency."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM observed_facts
                WHERE active = 1
                ORDER BY source_step DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [ObservedFact(**dict(row)) for row in cur.fetchall()]

    def get_by_subject(self, subject: str) -> list[ObservedFact]:
        """Get all active facts about a subject."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM observed_facts
                WHERE subject = ? AND active = 1
                ORDER BY source_step DESC
                """,
                (subject,),
            )
            return [ObservedFact(**dict(row)) for row in cur.fetchall()]

    def count_active(self) -> int:
        """Count active facts."""
        with self._db.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM observed_facts WHERE active = 1")
            return cur.fetchone()["cnt"]

    def prune_oldest(self, keep: int) -> int:
        """Deactivate oldest active facts beyond the keep limit. Returns pruned count."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE observed_facts SET active = 0
                WHERE id IN (
                    SELECT id FROM observed_facts
                    WHERE active = 1
                    ORDER BY source_step ASC
                    LIMIT MAX(0, (SELECT COUNT(*) FROM observed_facts WHERE active = 1) - ?)
                )
                """,
                (keep,),
            )
            return cur.rowcount


# =============================================================================
# Agent Memory Repository
# =============================================================================


class MemoryRepository:
    """CRUD operations for the agent_memory table."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def add(self, memory: AgentMemory) -> int:
        """Insert a new memory entry."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_memory (memory_type, content, relevance, created_step, active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (memory.memory_type.value, memory.content,
                 memory.relevance, memory.created_step),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_active(self, limit: int = 20) -> list[AgentMemory]:
        """Get active memories ordered by relevance."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM agent_memory
                WHERE active = 1
                ORDER BY relevance DESC, created_step DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [AgentMemory(**dict(row)) for row in cur.fetchall()]

    def decay_relevance(self, decay_factor: float) -> None:
        """Apply relevance decay to all active memories."""
        with self._db.get_cursor() as cur:
            cur.execute(
                "UPDATE agent_memory SET relevance = relevance * ? WHERE active = 1",
                (decay_factor,),
            )

    def prune_low_relevance(self, threshold: float) -> int:
        """Hard-delete memories below the relevance threshold."""
        with self._db.get_cursor() as cur:
            cur.execute(
                "DELETE FROM agent_memory WHERE active = 1 AND relevance < ?",
                (threshold,),
            )
            return cur.rowcount

    def prune_oldest(self, keep: int) -> int:
        """Deactivate oldest memories beyond the keep limit."""
        with self._db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE agent_memory SET active = 0
                WHERE id IN (
                    SELECT id FROM agent_memory
                    WHERE active = 1
                    ORDER BY relevance ASC, created_step ASC
                    LIMIT MAX(0, (SELECT COUNT(*) FROM agent_memory WHERE active = 1) - ?)
                )
                """,
                (keep,),
            )
            return cur.rowcount
