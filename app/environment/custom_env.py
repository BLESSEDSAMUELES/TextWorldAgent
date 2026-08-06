"""
Custom text adventure environment.

Loads a world definition from JSON and runs a fully functional text adventure
game with support for: movement, object interaction, locked doors, hidden
objects, NPCs with dialogue, and win conditions.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional


class CustomEnvironment:
    """A text adventure environment loaded from a JSON world definition."""

    def __init__(self, world_path: Path) -> None:
        self._world_path = world_path
        self._world_def: dict[str, Any] = {}
        self._rooms: dict[str, dict[str, Any]] = {}
        self._current_room: str = ""
        self._inventory: list[str] = []
        self._objective: str = ""
        self._done: bool = False
        self._total_reward: float = 0.0
        self._step_count: int = 0

    def reset(self) -> str:
        """Load the world and return the initial observation."""
        with open(self._world_path, encoding="utf-8") as f:
            self._world_def = json.load(f)

        self._rooms = deepcopy(self._world_def["rooms"])
        self._current_room = self._world_def["start_room"]
        self._inventory = []
        self._objective = self._world_def["objective"]
        self._done = False
        self._total_reward = 0.0
        self._step_count = 0

        return self._describe_room()

    def step(self, action: str) -> tuple[str, float, bool]:
        """Execute an action. Returns (observation, reward, done)."""
        if self._done:
            return "The game is over.", 0.0, True

        self._step_count += 1
        action = action.strip().lower()
        parts = action.split(maxsplit=1)
        verb = parts[0] if parts else ""
        target = parts[1] if len(parts) > 1 else ""

        handlers = {
            "go": self._handle_go,
            "north": lambda _: self._handle_go("north"),
            "south": lambda _: self._handle_go("south"),
            "east": lambda _: self._handle_go("east"),
            "west": lambda _: self._handle_go("west"),
            "up": lambda _: self._handle_go("up"),
            "down": lambda _: self._handle_go("down"),
            "look": lambda _: self._handle_look(),
            "take": self._handle_take,
            "get": self._handle_take,
            "grab": self._handle_take,
            "pick": self._handle_take,
            "drop": self._handle_drop,
            "examine": self._handle_examine,
            "x": self._handle_examine,
            "use": self._handle_use,
            "unlock": self._handle_unlock,
            "open": self._handle_unlock,
            "talk": self._handle_talk,
            "speak": self._handle_talk,
            "inventory": lambda _: self._handle_inventory(),
            "i": lambda _: self._handle_inventory(),
        }

        handler = handlers.get(verb)
        if handler is None:
            return f"I don't understand '{action}'.", -0.1, False

        obs, reward = handler(target)
        self._total_reward += reward

        # Check win conditions
        win_obs, win_reward = self._check_win_conditions(action)
        if win_obs:
            self._done = True
            return win_obs, reward + win_reward, True

        return obs, reward, False

    def get_valid_actions(self) -> list[str]:
        """Return all currently valid actions."""
        room = self._rooms[self._current_room]
        actions: list[str] = ["look", "inventory"]

        # Movement
        for direction in room.get("exits", {}):
            actions.append(f"go {direction}")

        # Objects in room
        for obj_name, obj_data in room.get("objects", {}).items():
            actions.append(f"examine {obj_name}")
            if obj_data.get("portable", True):
                actions.append(f"take {obj_name}")
            if obj_data.get("locked", False):
                actions.append(f"unlock {obj_name}")

        # Inventory actions
        for item in self._inventory:
            actions.append(f"use {item}")
            actions.append(f"drop {item}")
            actions.append(f"examine {item}")

        # NPC actions
        for npc_name in room.get("npcs", {}):
            actions.append(f"talk {npc_name}")

        return sorted(set(actions))

    def get_objective(self) -> str:
        """Return the game objective."""
        return self._objective

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _handle_go(self, direction: str) -> tuple[str, float]:
        """Move to an adjacent room."""
        room = self._rooms[self._current_room]
        exits = room.get("exits", {})

        if direction not in exits:
            return f"You can't go {direction} from here.", -0.1

        # Check for locked exits via room states
        states = room.get("states", {})
        for obj_name, obj_data in room.get("objects", {}).items():
            if obj_data.get("locked", False):
                on_unlock = obj_data.get("on_unlock", {})
                added_exits = on_unlock.get("add_exit", {})
                if direction in added_exits and obj_data.get("locked", False):
                    return f"The way {direction} is blocked.", -0.1

        self._current_room = exits[direction]
        reward = 0.1  # Small reward for exploration

        # Bonus reward for first visit
        if not self._rooms[self._current_room].get("_visited", False):
            self._rooms[self._current_room]["_visited"] = True
            reward = 0.5

        return self._describe_room(), reward

    def _handle_look(self) -> tuple[str, float]:
        """Describe the current room."""
        return self._describe_room(), 0.0

    def _handle_take(self, obj_name: str) -> tuple[str, float]:
        """Pick up an object."""
        obj_name = self._resolve_object_name(obj_name)
        room = self._rooms[self._current_room]
        objects = room.get("objects", {})

        if obj_name not in objects:
            return f"There is no '{obj_name}' here.", -0.1

        obj_data = objects[obj_name]
        if not obj_data.get("portable", True):
            return f"You can't pick up the {obj_name}.", -0.1

        self._inventory.append(obj_name)
        # Store object data for later use
        if "_taken_objects" not in room:
            room["_taken_objects"] = {}
        room["_taken_objects"][obj_name] = objects.pop(obj_name)

        reward = 0.3
        if obj_data.get("is_objective_item", False):
            reward = 2.0

        return f"You pick up the {obj_name}.", reward

    def _handle_drop(self, obj_name: str) -> tuple[str, float]:
        """Drop an object from inventory."""
        obj_name = self._resolve_inventory_name(obj_name)
        if obj_name not in self._inventory:
            return f"You don't have '{obj_name}'.", -0.1

        self._inventory.remove(obj_name)
        room = self._rooms[self._current_room]
        objects = room.setdefault("objects", {})
        # Restore object data if available
        taken = room.get("_taken_objects", {})
        obj_data = taken.pop(obj_name, {"description": f"A {obj_name}.", "portable": True})
        objects[obj_name] = obj_data

        return f"You drop the {obj_name}.", 0.0

    def _handle_examine(self, obj_name: str) -> tuple[str, float]:
        """Examine an object in the room or inventory."""
        obj_name_resolved = self._resolve_object_name(obj_name)
        room = self._rooms[self._current_room]
        objects = room.get("objects", {})

        # Check room objects
        if obj_name_resolved in objects:
            obj_data = objects[obj_name_resolved]
            desc = obj_data.get("description", f"You see a {obj_name_resolved}.")
            result = desc

            # Handle reveal on examine
            on_examine = obj_data.get("on_examine", {})
            if on_examine:
                reveal = on_examine.get("reveal_object")
                msg = on_examine.get("message", "")
                if reveal and reveal not in objects and reveal not in self._inventory:
                    objects[reveal] = {
                        "description": f"A {reveal}.",
                        "portable": True,
                    }
                    result = f"{desc}\n{msg}"
                    # Remove on_examine to prevent re-triggering
                    del obj_data["on_examine"]

            return result, 0.1

        # Check inventory
        inv_name = self._resolve_inventory_name(obj_name)
        if inv_name in self._inventory:
            # Look up stored object data
            for r in self._rooms.values():
                taken = r.get("_taken_objects", {})
                if inv_name in taken:
                    desc = taken[inv_name].get("description", f"A {inv_name}.")
                    return desc, 0.0
            return f"You examine the {inv_name}. It's a {inv_name}.", 0.0

        return f"You don't see '{obj_name}' here.", -0.1

    def _handle_use(self, obj_name: str) -> tuple[str, float]:
        """Use an item from inventory."""
        obj_name = self._resolve_inventory_name(obj_name)
        if obj_name not in self._inventory:
            return f"You don't have '{obj_name}'.", -0.1

        # Generic use — check for win condition or special interaction
        return f"You hold up the {obj_name}, but nothing happens.", 0.0

    def _handle_unlock(self, target: str) -> tuple[str, float]:
        """Unlock a locked object using a key from inventory."""
        target = self._resolve_object_name(target)
        room = self._rooms[self._current_room]
        objects = room.get("objects", {})

        if target not in objects:
            return f"There is no '{target}' here.", -0.1

        obj_data = objects[target]
        if not obj_data.get("locked", False):
            return f"The {target} is not locked.", -0.1

        required_key = obj_data.get("lock_key", "")
        if required_key not in self._inventory:
            return f"You need the {required_key} to unlock the {target}.", -0.1

        # Unlock it
        obj_data["locked"] = False
        self._inventory.remove(required_key)

        # Process on_unlock effects
        on_unlock = obj_data.get("on_unlock", {})
        result_parts: list[str] = []

        if on_unlock.get("message"):
            result_parts.append(on_unlock["message"])

        if on_unlock.get("reveal_object"):
            revealed = on_unlock["reveal_object"]
            objects[revealed] = {
                "description": f"A {revealed}.",
                "portable": True,
            }
            if not result_parts:
                result_parts.append(f"You unlock the {target} and find a {revealed}!")

        if on_unlock.get("add_exit"):
            exits = room.setdefault("exits", {})
            exits.update(on_unlock["add_exit"])
            if not result_parts:
                for d in on_unlock["add_exit"]:
                    result_parts.append(f"Unlocking the {target} reveals a passage to the {d}!")

        if not result_parts:
            result_parts.append(f"You unlock the {target}.")

        return " ".join(result_parts), 1.0

    def _handle_talk(self, npc_name: str) -> tuple[str, float]:
        """Talk to an NPC."""
        npc_name = self._resolve_npc_name(npc_name)
        room = self._rooms[self._current_room]
        npcs = room.get("npcs", {})

        if npc_name not in npcs:
            return f"There is no '{npc_name}' here to talk to.", -0.1

        npc_data = npcs[npc_name]
        dialogue = npc_data.get("dialogue", "They have nothing to say.")
        desc = npc_data.get("description", "")

        return f"{desc}\n{npc_data.get('name', npc_name)} says: \"{dialogue}\"", 0.2

    def _handle_inventory(self) -> tuple[str, float]:
        """List inventory contents."""
        if not self._inventory:
            return "Your inventory is empty.", 0.0
        items = ", ".join(self._inventory)
        return f"You are carrying: {items}", 0.0

    # =========================================================================
    # Win Condition Checking
    # =========================================================================

    def _check_win_conditions(self, action: str) -> tuple[Optional[str], float]:
        """Check if any win condition is met."""
        for cond in self._world_def.get("win_conditions", []):
            if cond["type"] == "has_items_at_room":
                required_items = cond["items"]
                required_room = cond["room"]
                required_action = cond.get("action", "")

                has_items = all(item in self._inventory for item in required_items)
                in_room = self._current_room == required_room
                action_match = not required_action or action.startswith(required_action)

                if has_items and in_room and action_match:
                    return cond.get("message", "You win!"), 10.0

        return None, 0.0

    # =========================================================================
    # Helpers
    # =========================================================================

    def _describe_room(self) -> str:
        """Generate the room description text."""
        room = self._rooms[self._current_room]
        parts: list[str] = []

        name = room.get("name", self._current_room)
        parts.append(f"== {name} ==")
        parts.append(room.get("description", ""))

        # List exits
        exits = room.get("exits", {})
        if exits:
            exit_list = ", ".join(exits.keys())
            parts.append(f"Exits: {exit_list}")

        # List objects
        objects = room.get("objects", {})
        if objects:
            obj_list = ", ".join(objects.keys())
            parts.append(f"You see: {obj_list}")

        # List NPCs
        npcs = room.get("npcs", {})
        if npcs:
            for npc_id, npc_data in npcs.items():
                desc = npc_data.get("description", f"A {npc_id} is here.")
                parts.append(desc)

        return "\n".join(parts)

    def _resolve_object_name(self, name: str) -> str:
        """Fuzzy-match an object name against room objects."""
        room = self._rooms[self._current_room]
        objects = room.get("objects", {})
        name = name.strip().lower().replace(" ", "_")

        if name in objects:
            return name

        # Try partial match
        for obj_name in objects:
            if name in obj_name or obj_name in name:
                return obj_name

        return name

    def _resolve_inventory_name(self, name: str) -> str:
        """Fuzzy-match an item name against inventory."""
        name = name.strip().lower().replace(" ", "_")
        if name in self._inventory:
            return name
        for item in self._inventory:
            if name in item or item in name:
                return item
        return name

    def _resolve_npc_name(self, name: str) -> str:
        """Fuzzy-match an NPC name."""
        room = self._rooms[self._current_room]
        npcs = room.get("npcs", {})
        name = name.strip().lower().replace(" ", "_")

        if name in npcs:
            return name
        for npc_name in npcs:
            if name in npc_name or npc_name in name:
                return npc_name
        return name
