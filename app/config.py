"""
Application configuration using Pydantic BaseSettings.

All configuration is centralized here — zero hardcoded values in business logic.
Environment variables override defaults (prefix: ENVORA_).
"""

from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings  # type: ignore[assignment]


class AppConfig(BaseSettings):
    """Central configuration for ENVORA (Environment-aware Reasoning Agent)."""

    model_config = {"env_prefix": "ENVORA_"}

    # --- LLM Settings ---
    llm_model: str = "gemma2:2b"
    llm_temperature: float = 0.2     # lower = more deterministic = faster decode
    llm_max_tokens: int = 30         # actions are short; 30 tokens is plenty
    llm_max_retries: int = 1         # fail fast — heuristic fallback handles misses
    ollama_host: str = "http://localhost:11434"

    # --- Database Settings ---
    db_path: Path = Path("world.db")

    # --- World Model Settings ---
    max_active_facts: int = 50
    max_active_memories: int = 20
    memory_relevance_decay: float = 0.95
    memory_prune_threshold: float = 0.1
    memory_prune_interval: int = 20

    # --- Query Engine Settings ---
    world_slice_max_tokens: int = 250
    fact_recency_window: int = 10
    spatial_hop_distance: int = 1

    # --- Agent Settings ---
    max_action_history: int = 5
    loop_detection_threshold: int = 2   # trigger fallback after 2 repeats in last 6 steps
    max_game_steps: int = 100

    # --- Logging Settings ---
    log_level: str = "INFO"
    rich_logging: bool = True


def get_config() -> AppConfig:
    """Factory function for configuration. Enables dependency injection."""
    return AppConfig()
