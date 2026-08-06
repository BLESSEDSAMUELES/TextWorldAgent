"""
Text observation extractor.

Converts raw text adventure observations into structured ExtractionResult
objects. Uses a two-tier approach:

1. Rule-based parser (primary) — fast regex patterns for standard formats
2. LLM-assisted parser (fallback) — for ambiguous text only

For a 2B model on CPU, the rule-based parser handles 90%+ of extractions.
"""

from __future__ import annotations

import re

from app.models.schemas import ExtractionResult


class Extractor:
    """Extracts structured data from raw text adventure observations."""

    # Regex patterns for common text adventure output formats
    _ROOM_NAME_PATTERN = re.compile(r"^==\s*(.+?)\s*==$", re.MULTILINE)
    _EXITS_PATTERN = re.compile(
        r"[Ee]xits?:\s*(.+?)(?:\n|$)", re.MULTILINE
    )
    _OBJECTS_PATTERN = re.compile(
        r"[Yy]ou (?:see|notice|spot|find):\s*(.+?)(?:\n|$)", re.MULTILINE
    )
    _NPC_PATTERNS = [
        re.compile(r"(?:A|An|The)\s+(.+?)\s+(?:is here|stands|hovers|sits)", re.MULTILINE),
        re.compile(r"(\w[\w\s]+?)\s+says?:", re.MULTILINE),
    ]
    _DIRECTION_WORDS = frozenset(
        {"north", "south", "east", "west", "up", "down",
         "northeast", "northwest", "southeast", "southwest"}
    )

    # Action result patterns for state change detection
    _TAKE_PATTERN = re.compile(r"[Yy]ou (?:pick up|take|grab|get)\s+(?:the\s+)?(.+?)\.", re.MULTILINE)
    _DROP_PATTERN = re.compile(r"[Yy]ou drop\s+(?:the\s+)?(.+?)\.", re.MULTILINE)
    _UNLOCK_PATTERN = re.compile(
        r"[Yy]ou unlock\s+(?:the\s+)?(.+?)(?:\.|!|,)", re.MULTILINE
    )
    _REVEAL_PATTERN = re.compile(
        r"(?:find|reveal|discover)\s+(?:a\s+)?(.+?)(?:\.|!)", re.MULTILINE
    )
    _LOCKED_PATTERN = re.compile(
        r"(?:The\s+)?(.+?)\s+is\s+(?:locked|blocked|closed)", re.MULTILINE
    )

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract structured data from a raw observation.

        Args:
            text: Raw text from the game environment.

        Returns:
            ExtractionResult with parsed room, objects, exits, etc.
        """
        return ExtractionResult(
            room_name=self._extract_room_name(text),
            room_description=self._extract_description(text),
            exits=self._extract_exits(text),
            objects=self._extract_objects(text),
            npcs=self._extract_npcs(text),
            state_changes=self._extract_state_changes(text),
            raw_text=text,
        )

    def _extract_room_name(self, text: str) -> str | None:
        """Extract room name from '== Room Name ==' format."""
        match = self._ROOM_NAME_PATTERN.search(text)
        return match.group(1).strip() if match else None

    def _extract_description(self, text: str) -> str | None:
        """Extract room description (text between name and exits/objects)."""
        lines = text.strip().split("\n")
        desc_lines: list[str] = []
        in_desc = False

        for line in lines:
            stripped = line.strip()
            # Skip room name header
            if self._ROOM_NAME_PATTERN.match(stripped):
                in_desc = True
                continue
            # Stop at exits/objects/NPC lines
            if in_desc and self._is_metadata_line(stripped):
                break
            if in_desc and stripped:
                desc_lines.append(stripped)

        return " ".join(desc_lines) if desc_lines else None

    def _extract_exits(self, text: str) -> list[str]:
        """Extract exit directions."""
        match = self._EXITS_PATTERN.search(text)
        if not match:
            return []

        raw = match.group(1).strip()
        exits: list[str] = []

        # Split by comma, 'and', or whitespace
        parts = re.split(r"[,\s]+(?:and\s+)?", raw)
        for part in parts:
            cleaned = part.strip().lower().rstrip(".")
            if cleaned in self._DIRECTION_WORDS:
                exits.append(cleaned)

        return exits

    def _extract_objects(self, text: str) -> list[str]:
        """Extract visible objects."""
        match = self._OBJECTS_PATTERN.search(text)
        if not match:
            return self._extract_objects_from_context(text)

        raw = match.group(1).strip()
        objects = [
            obj.strip().lower().rstrip(".")
            for obj in re.split(r"[,]+\s*(?:and\s+)?", raw)
            if obj.strip()
        ]
        return objects

    def _extract_objects_from_context(self, text: str) -> list[str]:
        """Fallback: extract objects mentioned in take/reveal patterns."""
        objects: list[str] = []
        for pattern in [self._TAKE_PATTERN, self._REVEAL_PATTERN]:
            for match in pattern.finditer(text):
                name = match.group(1).strip().lower().replace(" ", "_")
                if name and name not in objects:
                    objects.append(name)
        return objects

    def _extract_npcs(self, text: str) -> list[str]:
        """Extract NPC names from description text."""
        npcs: list[str] = []
        for pattern in self._NPC_PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1).strip()
                # Filter out common false positives
                if len(name) > 2 and name.lower() not in ("you", "the", "a", "an"):
                    normalized = name.lower().replace(" ", "_")
                    if normalized not in npcs:
                        npcs.append(normalized)
        return npcs

    def _extract_state_changes(self, text: str) -> dict[str, str]:
        """Extract state changes from action results."""
        changes: dict[str, str] = {}

        # Detect unlock events
        for match in self._UNLOCK_PATTERN.finditer(text):
            target = match.group(1).strip().lower().replace(" ", "_")
            changes[target] = "unlocked"

        # Detect locked items
        for match in self._LOCKED_PATTERN.finditer(text):
            target = match.group(1).strip().lower().replace(" ", "_")
            if target not in changes:
                changes[target] = "locked"

        # Detect take events
        for match in self._TAKE_PATTERN.finditer(text):
            obj = match.group(1).strip().lower().replace(" ", "_")
            changes[obj] = "taken"

        # Detect drop events
        for match in self._DROP_PATTERN.finditer(text):
            obj = match.group(1).strip().lower().replace(" ", "_")
            changes[obj] = "dropped"

        return changes

    @staticmethod
    def _is_metadata_line(line: str) -> bool:
        """Check if a line is a metadata line (exits, objects, etc.)."""
        lower = line.lower()
        return any(lower.startswith(prefix) for prefix in (
            "exit", "you see", "you notice", "you spot",
            "you find", "there is", "you are carrying",
        ))
