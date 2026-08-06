"""
Domain models for the Text World Agent.

All domain objects are Pydantic models with strict validation.
These models are the single source of truth for data shapes across
the entire application — database, extractor, query engine, and agent
all speak in terms of these types.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class MemoryType(str, Enum):
    """Categories of agent memory entries."""
    GOAL = "goal"
    PLAN = "plan"
    OBSERVATION = "observation"
    FAILURE = "failure"
    SUCCESS = "success"


class ActionType(str, Enum):
    """Recognized action categories."""
    GO = "go"
    TAKE = "take"
    DROP = "drop"
    EXAMINE = "examine"
    USE = "use"
    TALK = "talk"
    LOOK = "look"
    INVENTORY = "inventory"
    UNLOCK = "unlock"
    UNKNOWN = "unknown"


# =============================================================================
# Database Entity Models
# =============================================================================


class Room(BaseModel):
    """A room in the game world."""
    id: Optional[int] = None
    name: str
    description: str = ""
    visited: bool = False
    visit_count: int = 0
    first_seen: int = 0
    last_seen: int = 0


class Connection(BaseModel):
    """A directed edge between two rooms."""
    id: Optional[int] = None
    from_room_id: int
    to_room_id: Optional[int] = None  # None if destination unknown
    direction: str
    locked: bool = False
    lock_key: Optional[str] = None


class GameObject(BaseModel):
    """An interactable object in the game world."""
    id: Optional[int] = None
    name: str
    description: str = ""
    room_id: Optional[int] = None  # None if in inventory or consumed
    portable: bool = True
    state: str = "default"
    properties: dict = Field(default_factory=dict)


class NPC(BaseModel):
    """A non-player character."""
    id: Optional[int] = None
    name: str
    description: str = ""
    room_id: Optional[int] = None
    dialogue: Optional[str] = None
    state: str = "idle"


class InventoryItem(BaseModel):
    """An item currently in the player's inventory."""
    id: Optional[int] = None
    object_id: int
    acquired_step: int


class RoomState(BaseModel):
    """A key-value state entry for a room."""
    id: Optional[int] = None
    room_id: int
    state_key: str
    state_value: str
    updated_step: int = 0


class ObservedFact(BaseModel):
    """An RDF-style triple representing observed knowledge."""
    id: Optional[int] = None
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_step: int = 0
    superseded_by: Optional[int] = None
    active: bool = True


class AgentMemory(BaseModel):
    """A high-level strategic memory entry."""
    id: Optional[int] = None
    memory_type: MemoryType
    content: str
    relevance: float = 1.0
    created_step: int = 0
    active: bool = True


# =============================================================================
# Operational Models (not persisted directly)
# =============================================================================


class ExtractionResult(BaseModel):
    """Output of the extractor — structured data from raw text."""
    room_name: Optional[str] = None
    room_description: Optional[str] = None
    exits: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    npcs: list[str] = Field(default_factory=list)
    state_changes: dict[str, str] = Field(default_factory=dict)
    raw_text: str = ""


class WorldSlice(BaseModel):
    """The minimal context the LLM receives — never the full DB."""
    objective: str
    current_room: str
    room_description: str = ""
    exits: list[str] = Field(default_factory=list)
    objects_here: list[str] = Field(default_factory=list)
    inventory: list[str] = Field(default_factory=list)
    nearby_npcs: list[str] = Field(default_factory=list)
    relevant_facts: list[str] = Field(default_factory=list)
    recent_failures: list[str] = Field(default_factory=list)
    valid_actions: list[str] = Field(default_factory=list)
    visited_rooms: list[str] = Field(default_factory=list)

    def to_prompt_text(self, override_actions: list[str] | None = None) -> str:
        """Render the world slice as a compact text block for the LLM."""
        lines: list[str] = []
        lines.append(f"OBJECTIVE: {self.objective}")
        lines.append(f"CURRENT ROOM: {self.current_room}")
        if self.room_description:
            lines.append(f"DESCRIPTION: {self.room_description}")
        lines.append(f"EXITS: {', '.join(self.exits) if self.exits else 'none'}")
        lines.append(
            f"OBJECTS HERE: "
            f"{', '.join(self.objects_here) if self.objects_here else 'none'}"
        )
        lines.append(
            f"INVENTORY: "
            f"{', '.join(self.inventory) if self.inventory else 'empty'}"
        )
        if self.nearby_npcs:
            lines.append(f"NPCS: {', '.join(self.nearby_npcs)}")
        if self.relevant_facts:
            lines.append("IMPORTANT FACTS:")
            for fact in self.relevant_facts:
                lines.append(f"  - {fact}")
        if self.recent_failures:
            lines.append("RECENT FAILURES (avoid repeating):")
            for fail in self.recent_failures:
                lines.append(f"  - {fail}")
        if self.visited_rooms:
            lines.append(f"VISITED ROOMS: {', '.join(self.visited_rooms)}")
        actions_to_show = override_actions if override_actions is not None else self.valid_actions
        lines.append(f"VALID ACTIONS: {', '.join(actions_to_show)}")
        return "\n".join(lines)


class AgentAction(BaseModel):
    """A parsed action from LLM output."""
    action_type: ActionType
    target: Optional[str] = None
    raw_command: str

    @classmethod
    def from_command(cls, command: str) -> "AgentAction":
        """Parse a raw command string into a structured action."""
        parts = command.strip().lower().split(maxsplit=1)
        if not parts:
            return cls(action_type=ActionType.UNKNOWN, raw_command=command)

        verb = parts[0]
        target = parts[1] if len(parts) > 1 else None

        type_map: dict[str, ActionType] = {
            "go": ActionType.GO,
            "north": ActionType.GO,
            "south": ActionType.GO,
            "east": ActionType.GO,
            "west": ActionType.GO,
            "up": ActionType.GO,
            "down": ActionType.GO,
            "take": ActionType.TAKE,
            "get": ActionType.TAKE,
            "pick": ActionType.TAKE,
            "grab": ActionType.TAKE,
            "drop": ActionType.DROP,
            "examine": ActionType.EXAMINE,
            "look": ActionType.LOOK,
            "x": ActionType.EXAMINE,
            "use": ActionType.USE,
            "talk": ActionType.TALK,
            "speak": ActionType.TALK,
            "inventory": ActionType.INVENTORY,
            "i": ActionType.INVENTORY,
            "unlock": ActionType.UNLOCK,
            "open": ActionType.UNLOCK,
        }

        action_type = type_map.get(verb, ActionType.UNKNOWN)

        # Handle direction-as-verb: "north" → GO north
        if verb in ("north", "south", "east", "west", "up", "down"):
            target = verb
            command = f"go {verb}"

        return cls(
            action_type=action_type,
            target=target,
            raw_command=command.strip().lower(),
        )
