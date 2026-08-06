"""
Abstract game environment protocol.

Any text adventure environment (custom, TextWorld, etc.) must implement
this interface. This is the only contract the rest of the system depends on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GameEnvironment(Protocol):
    """Protocol for text adventure environments."""

    def reset(self) -> str:
        """Reset the environment and return the initial observation."""
        ...

    def step(self, action: str) -> tuple[str, float, bool]:
        """
        Execute an action in the environment.

        Returns:
            observation: Text description of the result
            reward: Numeric reward signal
            done: Whether the game is over
        """
        ...

    def get_valid_actions(self) -> list[str]:
        """Return the list of currently valid actions."""
        ...

    def get_objective(self) -> str:
        """Return the current game objective."""
        ...
