"""
Query Engine — builds minimal world slices for the LLM.

Core principle: the LLM never sees the full database.
It receives only a surgical ~250 token slice containing:
- Objective
- Current room + description
- Exits
- Objects here
- Inventory
- Nearby NPCs
- Top-K relevant facts
- Recent failures
- Valid actions

Query strategy:
- Spatial: current room + 1-hop neighbors only
- Temporal: recent facts weighted higher
- Relevance: facts mentioning current room objects ranked first
- Budget: total slice < max_tokens (enforced)
"""

from __future__ import annotations

from app.config import AppConfig
from app.models.schemas import ObservedFact, WorldSlice
from app.utils.token_counter import estimate_tokens
from app.world_model.world_model import WorldModel


class QueryEngine:
    """Builds minimal world slices from the world model."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def build_slice(
        self,
        world_model: WorldModel,
        objective: str,
        valid_actions: list[str],
    ) -> WorldSlice:
        """
        Build a WorldSlice containing only what the LLM needs.

        Args:
            world_model: The current world model state.
            objective: The game objective.
            valid_actions: Currently valid actions from the environment.

        Returns:
            A WorldSlice ready for prompt rendering.
        """
        current_room = world_model.current_room_name
        room_id = world_model.current_room_id

        # Get room description
        room_desc = ""
        if room_id is not None:
            room = world_model.rooms.get_by_id(room_id)
            if room:
                room_desc = room.description

        # Get exits
        exits: list[str] = []
        if room_id is not None:
            for conn in world_model.connections.get_exits(room_id):
                exits.append(conn.direction)

        # Get objects in current room
        objects_here: list[str] = []
        if room_id is not None:
            for obj in world_model.objects.get_by_room(room_id):
                objects_here.append(obj.name)

        # Get inventory
        inventory = world_model.get_inventory_names()

        # Get NPCs in current room
        nearby_npcs: list[str] = []
        if room_id is not None:
            for npc in world_model.npcs.get_by_room(room_id):
                nearby_npcs.append(npc.name)

        # Get relevant facts (ranked by relevance to current context)
        relevant_facts = self._get_relevant_facts(
            world_model, current_room, objects_here
        )

        # Get recent failures for anti-loop
        recent_failures = self._get_recent_failures(world_model)

        # Get visited rooms for exploration guidance
        visited_rooms = world_model.get_visited_room_names()

        # Build the slice
        world_slice = WorldSlice(
            objective=objective,
            current_room=current_room,
            room_description=room_desc,
            exits=exits,
            objects_here=objects_here,
            inventory=inventory,
            nearby_npcs=nearby_npcs,
            relevant_facts=relevant_facts,
            recent_failures=recent_failures,
            valid_actions=valid_actions,
            visited_rooms=visited_rooms,
        )

        # Enforce token budget by trimming facts if needed
        return self._enforce_budget(world_slice)

    def _get_relevant_facts(
        self,
        world_model: WorldModel,
        current_room: str,
        objects_here: list[str],
    ) -> list[str]:
        """
        Get the most relevant facts for the current context.

        Priority:
        1. Facts about the current room
        2. Facts about objects here
        3. Facts about inventory items
        4. Recent general facts
        """
        all_facts = world_model.facts.get_active(
            limit=self._config.max_active_facts
        )

        scored: list[tuple[float, str]] = []
        for fact in all_facts:
            score = self._score_fact(fact, current_room, objects_here)
            text = f"{fact.subject} {fact.predicate} {fact.object}"
            scored.append((score, text))

        # Sort by score descending, take top entries
        scored.sort(key=lambda x: x[0], reverse=True)

        # Filter out low-relevance and agent location (already shown)
        filtered: list[str] = []
        for score, text in scored:
            if score <= 0:
                continue
            if "agent is_in" in text:
                continue
            filtered.append(text)
            if len(filtered) >= 8:
                break

        return filtered

    def _score_fact(
        self,
        fact: ObservedFact,
        current_room: str,
        objects_here: list[str],
    ) -> float:
        """Score a fact by relevance to the current context."""
        score: float = fact.confidence

        # Boost facts about current room
        if current_room.lower() in fact.subject.lower():
            score += 3.0
        if current_room.lower() in fact.object.lower():
            score += 2.0

        # Boost facts about objects in the room
        for obj in objects_here:
            if obj.lower() in fact.subject.lower():
                score += 2.0

        # Boost facts about state (locked, unlocked, etc.)
        if fact.predicate == "state_is":
            score += 1.5

        # Recency bonus
        recency_window = self._config.fact_recency_window
        if fact.source_step > 0:
            score += min(1.0, fact.source_step / max(1, recency_window))

        return score

    def _get_recent_failures(self, world_model: WorldModel) -> list[str]:
        """Get recent failure memories for anti-loop."""
        memories = world_model.memories.get_active(limit=10)
        failures: list[str] = []
        for mem in memories:
            if mem.memory_type.value == "failure":
                content = mem.content.replace("Failed: ", "")
                failures.append(content)
                if len(failures) >= 3:
                    break
        return failures

    def _enforce_budget(self, world_slice: WorldSlice) -> WorldSlice:
        """Trim the world slice to stay within the token budget."""
        max_tokens = self._config.world_slice_max_tokens
        prompt_text = world_slice.to_prompt_text()
        current_tokens = estimate_tokens(prompt_text)

        if current_tokens <= max_tokens:
            return world_slice

        # Progressive trimming: facts first, then description
        while current_tokens > max_tokens and world_slice.relevant_facts:
            world_slice.relevant_facts.pop()
            prompt_text = world_slice.to_prompt_text()
            current_tokens = estimate_tokens(prompt_text)

        if current_tokens > max_tokens and world_slice.room_description:
            # Truncate description
            words = world_slice.room_description.split()
            while current_tokens > max_tokens and len(words) > 5:
                words = words[:len(words) - 5]
                world_slice.room_description = " ".join(words) + "..."
                prompt_text = world_slice.to_prompt_text()
                current_tokens = estimate_tokens(prompt_text)

        return world_slice
