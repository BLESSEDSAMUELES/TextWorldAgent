"""
Text World Agent — central agent orchestrator.

Coordinates environment interaction, observation extraction, world model updates,
query engine context construction, LLM generation, action parsing, loop detection,
and smart fallback strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import networkx as nx

from app.agent.action_parser import ActionParser
from app.config import AppConfig
from app.environment.base import GameEnvironment
from app.extractor.extractor import Extractor
from app.llm.llm_client import LLMClient
from app.models.schemas import AgentAction, ActionType, WorldSlice
from app.query_engine.query_engine import QueryEngine
from app.world_model.world_model import WorldModel

logger = logging.getLogger(__name__)


class TextWorldAgent:
    """Orchestrates LLM, World Model, and Environment for text adventure games."""

    def __init__(
        self,
        world_model: WorldModel,
        query_engine: QueryEngine,
        llm_client: LLMClient,
        extractor: Extractor,
        action_parser: ActionParser,
        config: AppConfig,
    ) -> None:
        self.world_model = world_model
        self.query_engine = query_engine
        self.llm_client = llm_client
        self.extractor = extractor
        self.action_parser = action_parser
        self.config = config

        self._recent_actions: list[str] = []
        self._action_history: list[tuple[str, str]] = []  # (room, action)
        self._step_count: int = 0
        self._examined_objects: set[str] = set()
        self._failed_uses: set[str] = set()
        self._failed_unlocks: set[tuple[str, str]] = set()
        self._spoke_to_npcs: set[str] = set()
        self._recent_rooms: list[str] = []   # rolling window for room-cycle detection
        self._start_room_name: Optional[str] = None

    def reset(self) -> None:
        """Reset the agent's internal turn state."""
        self._recent_actions.clear()
        self._action_history.clear()
        self._step_count = 0
        self._examined_objects.clear()
        self._failed_uses.clear()
        self._failed_unlocks.clear()
        self._spoke_to_npcs.clear()
        self._recent_rooms.clear()
        self._start_room_name = None

    def step(
        self,
        env: GameEnvironment,
        current_step: int,
        last_obs: str = "",
    ) -> tuple[str, float, bool, dict[str, Any]]:
        """
        Execute a single step of the agent.

        Args:
            env: The game environment.
            current_step: Current game step number.
            last_obs: Observation from previous environment reset or step.

        Returns:
            Tuple of (observation, reward, done, step_info)
        """
        self._step_count = current_step
        objective = env.get_objective()
        valid_actions = env.get_valid_actions()

        # 1. Extract structure from last observation
        if last_obs:
            extraction = self.extractor.extract(last_obs)
            self.world_model.process_observation(extraction, step=current_step)

        # 2. Build minimal world slice for context
        world_slice = self.query_engine.build_slice(
            world_model=self.world_model,
            objective=objective,
            valid_actions=valid_actions,
        )

        # 3. Select action via LLM or fallback strategy
        chosen_command, selection_source = self._select_action(
            world_slice=world_slice,
            valid_actions=valid_actions,
        )

        # 4. Parse action
        action = self.action_parser.parse(chosen_command, valid_actions)

        # 5. Loop prevention check: override if sticking in repetitive loop
        if self._is_looping(action.raw_command):
            fallback_cmd = self._get_heuristic_fallback(valid_actions)
            if fallback_cmd and fallback_cmd != action.raw_command:
                logger.info(
                    "Loop detected! Overriding '%s' with fallback '%s'",
                    action.raw_command,
                    fallback_cmd,
                )
                action = AgentAction.from_command(fallback_cmd)
                selection_source = "anti_loop_fallback"

        # Track history
        current_room = self.world_model.current_room_name
        if self._start_room_name is None:
            self._start_room_name = current_room
        self._recent_actions.append(action.raw_command)
        self._action_history.append((current_room, action.raw_command))
        self._recent_rooms.append(current_room)
        if len(self._recent_rooms) > 12:
            self._recent_rooms.pop(0)
        if action.action_type == ActionType.EXAMINE and action.target:
            self._examined_objects.add(action.target)
        if action.action_type == ActionType.TALK and action.target:
            self._spoke_to_npcs.add(action.target)

        # 6. Execute step in environment
        obs, reward, done = env.step(action.raw_command)

        # Track failed item use attempts (room-specific)
        if action.action_type == ActionType.USE and (reward <= 0 or "nothing happens" in obs.lower()):
            self._failed_uses.add((current_room, action.raw_command))

        # Track failed unlock attempts (room-specific)
        if action.action_type == ActionType.UNLOCK and (reward <= 0 or "need" in obs.lower()):
            self._failed_unlocks.add((current_room, action.raw_command))

        # 7. Post-action world model update
        self.world_model.update_after_action(action, obs, step=current_step)

        # Record negative reward as failure memory
        if reward < 0:
            self.world_model.record_failure(
                action=action.raw_command,
                reason=obs,
                step=current_step,
            )

        info = {
            "action": action.raw_command,
            "action_type": action.action_type.value,
            "source": selection_source,
            "current_room": current_room,
            "slice_prompt": world_slice.to_prompt_text(),
        }

        return obs, reward, done, info

    def run(
        self,
        env: GameEnvironment,
        max_steps: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Run the agent until completion or max_steps.

        Returns:
            List of step information logs.
        """
        self.reset()
        limit = max_steps or self.config.max_game_steps
        logs: list[dict[str, Any]] = []

        obs = env.reset()
        extraction = self.extractor.extract(obs)
        self.world_model.process_observation(extraction, step=0)

        step_num = 1
        done = False

        while not done and (limit <= 0 or step_num <= limit):
            obs, reward, done, info = self.step(env, step_num, last_obs=obs)
            info["step"] = step_num
            info["reward"] = reward
            info["observation"] = obs
            logs.append(info)
            step_num += 1

        return logs

    def _select_action(
        self,
        world_slice: WorldSlice,
        valid_actions: list[str],
    ) -> tuple[str, str]:
        """Select an action using LLM generation with heuristic fallback."""
        if self.llm_client.is_available():
            # Build a filtered action list for the LLM to reduce its chance of picking bad repeats
            filtered_actions = self._filter_actions_for_llm(valid_actions)
            # Always keep at least 2 actions so the model has a real choice
            actions_for_llm = filtered_actions if len(filtered_actions) >= 2 else valid_actions
            # Render the world slice with only the filtered actions visible to the LLM
            world_slice_text = world_slice.to_prompt_text(override_actions=actions_for_llm)
            prompt = self._build_prompt(world_slice_text, actions_for_llm)
            raw_response = self.llm_client.generate(prompt)

            if raw_response:
                # Parse against full valid_actions so fuzzy match works, but prefer filtered
                parsed = self.action_parser.parse(raw_response, actions_for_llm)
                if parsed.raw_command in valid_actions:
                    return parsed.raw_command, "llm"
                # Fallback: try parsing against full list
                parsed_full = self.action_parser.parse(raw_response, valid_actions)
                if parsed_full.raw_command in valid_actions:
                    return parsed_full.raw_command, "llm"

        # Fallback heuristic action selection
        fallback = self._get_heuristic_fallback(valid_actions)
        return fallback, "heuristic_fallback"

    def _filter_actions_for_llm(self, valid_actions: list[str]) -> list[str]:
        """Remove redundant actions before sending to LLM to reduce looping."""
        from collections import Counter
        recent_counts = Counter(self._recent_actions[-6:])
        result = []
        for act in valid_actions:
            # Skip examine actions for already-examined objects
            if act.startswith("examine ") or act.startswith("x "):
                target = act.split(maxsplit=1)[1] if " " in act else ""
                if target in self._examined_objects:
                    continue
            # Skip actions that were taken 2+ times recently (let heuristic handle them)
            if recent_counts.get(act, 0) >= 2:
                continue
            result.append(act)
        return result

    def _build_prompt(
        self,
        world_slice_text: str,
        valid_actions: list[str],
    ) -> str:
        """Construct prompt for the LLM with anti-loop context."""
        # Recent action history to prevent repetition
        recent = self._recent_actions[-5:] if self._recent_actions else []
        recent_str = ", ".join(recent) if recent else "none"

        # Objects already examined — tell LLM to skip them
        examined_str = ", ".join(sorted(self._examined_objects)) if self._examined_objects else "none"

        # Build banned action hint: anything repeated 2+ times recently
        from collections import Counter
        recent_counts = Counter(self._recent_actions[-6:])
        overused = [a for a, c in recent_counts.items() if c >= 2]
        avoid_str = ", ".join(overused) if overused else "none"

        recent_rooms = list(dict.fromkeys(self._recent_rooms[-4:])) if self._recent_rooms else []
        recent_rooms_str = ", ".join(recent_rooms) if recent_rooms else "none"

        return (
            "You are an agent playing a text adventure game.\n"
            "Choose the single best action to make progress toward the objective.\n"
            "You MUST select one action EXACTLY as written in the VALID ACTIONS list.\n"
            "Do NOT include explanations, reasoning, quotes, or markdown.\n\n"
            f"{world_slice_text}\n\n"
            f"RECENT ACTIONS (do NOT repeat these unless necessary): {recent_str}\n"
            f"RECENT ROOMS (avoid bouncing back and forth): {recent_rooms_str}\n"
            f"ALREADY EXAMINED: {examined_str}\n"
            f"AVOID (repeated too often): {avoid_str}\n\n"
            "Respond ONLY with your chosen action:"
        )

    def _get_heuristic_fallback(self, valid_actions: list[str]) -> str:
        """
        Rule-based decision tree with graph navigation for optimal game progress.

        Priority:
        1. Winning/Objective items use (e.g. use golden_chalice)
        2. Unlock doors if key is in inventory
        3. Take objective / portable items
        4. Examine unexamined objects in current room
        5. Talk to NPCs
        6. Move directly to unvisited neighbors
        7. Navigate via NetworkX shortest path towards unvisited frontiers
        8. Fallback to unvisited/recent actions
        """
        curr_room_name = self.world_model.current_room_name
        curr_room_id = self.world_model.current_room_id
        inventory = set(self.world_model.get_inventory_names())
        visited_rooms = set(self.world_model.get_visited_room_names())

        start_room = self._start_room_name or "Observatory Gates"

        # 1. Goal completion: if holding all win items, execute victory action or navigate directly to start room
        if "golden_chalice" in inventory and "silver_amulet" in inventory:
            if curr_room_name == start_room:
                if "use golden_chalice" in valid_actions and (curr_room_name, "use golden_chalice") not in self._failed_uses:
                    return "use golden_chalice"
            else:
                entrance = self.world_model.rooms.get_by_name(start_room)
                if entrance and entrance.id is not None and curr_room_id is not None:
                    graph = self.world_model.get_room_graph()
                    if graph.has_node(curr_room_id) and graph.has_node(entrance.id) and nx.has_path(graph, curr_room_id, entrance.id):
                        try:
                            p = nx.shortest_path(graph, curr_room_id, entrance.id)
                            if len(p) > 1:
                                next_node = p[1]
                                edge_data = graph.get_edge_data(curr_room_id, next_node)
                                if edge_data and edge_data.get("direction"):
                                    nav_dir = f"go {edge_data['direction']}"
                                    go_actions = [a for a in valid_actions if a.startswith("go ") or a in ("north", "south", "east", "west", "up", "down")]
                                    if nav_dir in valid_actions:
                                        return nav_dir
                        except nx.NetworkXNoPath:
                            pass

        # 2. Unlock actions
        unlock_actions = [
            a for a in valid_actions
            if a.startswith("unlock ") and (curr_room_name, a) not in self._failed_unlocks
        ]
        if unlock_actions:
            return unlock_actions[0]

        # 3. Take objective / portable items
        take_actions = [a for a in valid_actions if a.startswith("take ") or a.startswith("get ")]
        for take_act in take_actions:
            target = take_act.split(maxsplit=1)[1] if " " in take_act else ""
            if target not in inventory:
                return take_act

        # 4. Examine unexamined objects
        examine_actions = [
            a for a in valid_actions
            if a.startswith("examine ") or a.startswith("x ")
        ]
        for exam_act in examine_actions:
            target = exam_act.split(maxsplit=1)[1] if " " in exam_act else ""
            if target not in self._examined_objects and target not in inventory:
                return exam_act

        # 5. Talk to unvisited NPCs
        talk_actions = [a for a in valid_actions if a.startswith("talk ") or a.startswith("speak ")]
        for talk_act in talk_actions:
            target = talk_act.split(maxsplit=1)[1] if " " in talk_act else ""
            if target not in self._spoke_to_npcs:
                return talk_act

        # 6. Direct move to unvisited neighbor
        go_actions = [a for a in valid_actions if a.startswith("go ") or a in ("north", "south", "east", "west", "up", "down")]
        graph = self.world_model.get_room_graph()

        if curr_room_id is not None and graph.has_node(curr_room_id):
            for go_act in go_actions:
                direction = go_act.replace("go ", "").strip()
                for neighbor in graph.neighbors(curr_room_id):
                    edge_data = graph.get_edge_data(curr_room_id, neighbor)
                    if edge_data and edge_data.get("direction") == direction:
                        neighbor_name = graph.nodes[neighbor].get("name", "")
                        if neighbor_name not in visited_rooms:
                            return go_act

            # 6b. Move to an unexplored exit directly from the current room
            curr_exits = self.world_model.connections.get_exits(curr_room_id)
            unexplored_dirs = {conn.direction for conn in curr_exits if conn.to_room_id is None}
            for go_act in go_actions:
                direction = go_act.replace("go ", "").strip()
                if direction in unexplored_dirs:
                    return go_act

            # 7. Shortest-path navigation to nearest unexplored frontier or goal room
            target_rooms: set[int] = set()
            if "golden_chalice" in inventory and "silver_amulet" in inventory:
                entrance = self.world_model.rooms.get_by_name(start_room)
                if entrance and entrance.id is not None:
                    target_rooms.add(entrance.id)

            if not target_rooms:
                for r in self.world_model.rooms.get_all_visited():
                    if r.id is not None:
                        exits = self.world_model.connections.get_exits(r.id)
                        if any(conn.to_room_id is None for conn in exits):
                            target_rooms.add(r.id)

            shortest_path: Optional[list[int]] = None
            min_len = float("inf")
            for target in target_rooms:
                if target != curr_room_id and nx.has_path(graph, curr_room_id, target):
                    try:
                        p = nx.shortest_path(graph, curr_room_id, target)
                        if len(p) < min_len:
                            min_len = len(p)
                            shortest_path = p
                    except nx.NetworkXNoPath:
                        pass

            if shortest_path and len(shortest_path) > 1:
                next_node = shortest_path[1]
                edge_data = graph.get_edge_data(curr_room_id, next_node)
                if edge_data and edge_data.get("direction"):
                    nav_dir = f"go {edge_data['direction']}"
                    if nav_dir in valid_actions:
                        return nav_dir

        # 8. Any go action not recently taken, preferring ones that don't lead to a recent room
        recent_rooms_set = set(self._recent_rooms[-4:]) if self._recent_rooms else set()
        if curr_room_id is not None and graph.has_node(curr_room_id):
            for go_act in go_actions:
                if go_act not in self._recent_actions[-3:]:
                    direction = go_act.replace("go ", "").strip()
                    dest_room_name = None
                    for neighbor in graph.neighbors(curr_room_id):
                        edge_data = graph.get_edge_data(curr_room_id, neighbor)
                        if edge_data and edge_data.get("direction") == direction:
                            dest_room_name = graph.nodes[neighbor].get("name", "")
                            break
                    if dest_room_name and dest_room_name not in recent_rooms_set:
                        return go_act

        for go_act in go_actions:
            if go_act not in self._recent_actions[-3:]:
                return go_act

        # 9. Any valid action not equal to immediate last action
        for act in valid_actions:
            if act not in self._recent_actions[-2:]:
                return act

        return valid_actions[0] if valid_actions else "look"

    def _is_looping(self, candidate_action: str) -> bool:
        """Check if candidate action will create a repetitive loop.

        Fires if the candidate appears >= threshold times in the last 6 steps,
        OR if all of the last `threshold` steps are identical to the candidate,
        OR if the agent has been stuck in a cycle of the same few rooms.
        """
        threshold = self.config.loop_detection_threshold  # default 3
        window = self._recent_actions[-6:]
        if len(window) >= threshold:
            from collections import Counter
            counts = Counter(window)
            # Fire if the candidate appeared >= threshold times in the window
            if counts.get(candidate_action, 0) >= threshold:
                return True

        # Room cycle detection: if we are trying to move, check if we are bouncing
        is_move = candidate_action.startswith("go ") or candidate_action in ("north", "south", "east", "west", "up", "down")
        if is_move and len(self._recent_rooms) >= 6:
            # If we've only visited 2 or fewer distinct rooms in the last 6 steps, we are bouncing.
            unique_rooms = set(self._recent_rooms[-6:])
            if len(unique_rooms) <= 2:
                return True

        return False
