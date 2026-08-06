"""
Action parser — converts LLM output into structured AgentAction.

Handles messy LLM outputs by fuzzy-matching against valid actions.
The 2B model often adds explanations, reasoning, or formatting that
needs to be stripped before we can execute the action.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

from app.models.schemas import AgentAction


class ActionParser:
    """Parses raw LLM output into executable AgentAction."""

    def parse(
        self,
        llm_output: str,
        valid_actions: list[str],
    ) -> AgentAction:
        """
        Parse LLM output into an AgentAction.

        Strategy:
        1. Try exact match against valid actions
        2. Try extracting action from first line
        3. Fuzzy match against valid actions
        4. Fall back to 'look' as a safe default

        Args:
            llm_output: Raw text from the LLM.
            valid_actions: Currently valid actions.

        Returns:
            A parsed AgentAction.
        """
        cleaned = self._clean_output(llm_output)

        # 1. Exact match
        if cleaned in valid_actions:
            return AgentAction.from_command(cleaned)

        # 2. Try first line / first sentence
        first_line = cleaned.split("\n")[0].strip()
        first_line = re.sub(r"^(?:action:|>)\s*", "", first_line, flags=re.I)
        first_line = first_line.strip().rstrip(".")

        if first_line.lower() in valid_actions:
            return AgentAction.from_command(first_line)

        # 3. Fuzzy match
        best_match = self._fuzzy_match(first_line, valid_actions)
        if best_match:
            return AgentAction.from_command(best_match)

        # 4. Try matching any valid action found within the text
        embedded = self._find_embedded_action(cleaned, valid_actions)
        if embedded:
            return AgentAction.from_command(embedded)

        # 5. Safe fallback
        return AgentAction.from_command("look")

    def _clean_output(self, text: str) -> str:
        """Clean up LLM output for parsing."""
        text = text.strip()
        # Remove common LLM formatting artifacts
        text = re.sub(r"```\w*\n?", "", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"^\d+\.\s*", "", text)
        text = re.sub(r"^[-•]\s*", "", text)
        return text.strip().lower()

    def _fuzzy_match(
        self,
        text: str,
        valid_actions: list[str],
        threshold: float = 0.6,
    ) -> Optional[str]:
        """Find the closest valid action using SequenceMatcher."""
        best_score = 0.0
        best_action: Optional[str] = None

        text_lower = text.lower()
        for action in valid_actions:
            score = SequenceMatcher(
                None, text_lower, action.lower()
            ).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_action = action

        return best_action

    def _find_embedded_action(
        self,
        text: str,
        valid_actions: list[str],
    ) -> Optional[str]:
        """Find a valid action embedded within the text."""
        # Sort by length descending to prefer longer (more specific) matches
        sorted_actions = sorted(valid_actions, key=len, reverse=True)
        for action in sorted_actions:
            if action.lower() in text.lower():
                return action
        return None
