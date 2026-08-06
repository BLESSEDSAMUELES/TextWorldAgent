"""
Main CLI entry point for Text World Agent.

Runs the agent in a text adventure game environment, displaying step-by-step
observations, selected actions, world state updates, and game results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.agent.action_parser import ActionParser
from app.agent.agent import TextWorldAgent
from app.config import get_config
from app.database.connection import DatabaseConnection
from app.environment.custom_env import CustomEnvironment
from app.extractor.extractor import Extractor
from app.llm.llm_client import LLMClient
from app.query_engine.query_engine import QueryEngine
from app.world_model.world_model import WorldModel

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def print_step(step: int, obs: str, action: str, reward: float, source: str, room: str) -> None:
    """Print step details with rich or standard formatting."""
    if HAS_RICH and console is not None:
        source_color = "green" if source == "llm" else "yellow"
        header = f"[bold cyan]Step {step}[/bold cyan] | Room: [bold white]{room}[/bold white] | Action: [bold magenta]{action}[/bold magenta] ({f'[{source_color}]{source}[/{source_color}]'})"
        content = f"{obs}\n\n[dim]Reward: {reward:+.1f}[/dim]"
        console.print(Panel(content, title=header, expand=False))
    else:
        print(f"\n--- Step {step} ---")
        print(f"Room: {room}")
        print(f"Action: {action} (source: {source})")
        print(f"Observation:\n{obs}")
        print(f"Reward: {reward:+.1f}")


def main() -> int:
    """CLI execution entry point."""
    parser = argparse.ArgumentParser(description="Text World Agent Runner")
    parser.add_argument(
        "--world",
        type=Path,
        default=Path("app/environment/worlds/sample_world.json"),
        help="Path to JSON world definition",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Maximum game steps to run (default: 100, use 0 for unlimited)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=":memory:",
        help="Database file path (default: :memory: — use --db world_run.db for persistence)",
    )
    args = parser.parse_args()

    world_path: Path = args.world
    if not world_path.exists():
        print(f"Error: World file '{world_path}' does not exist.")
        return 1

    # Load configuration
    config = get_config()
    # Keep :memory: as-is; convert real file paths to Path objects
    config.db_path = args.db if args.db == ":memory:" else Path(args.db)
    config.max_game_steps = args.steps

    # Initialize environment
    env = CustomEnvironment(world_path)
    env.reset()

    if HAS_RICH and console is not None:
        console.print(Panel.fit(
            f"[bold gold1]Text World Agent[/bold gold1]\n"
            f"World: [cyan]{world_path.name}[/cyan]\n"
            f"Objective: {env.get_objective()}\n"
            f"Mode: [magenta]AI Agent[/magenta]",
            title="Initialization",
        ))
    else:
        print("=== Text World Agent ===")
        print(f"World: {world_path.name}")
        print(f"Objective: {env.get_objective()}")

    # Run the AI agent
    db = DatabaseConnection(config.db_path)
    db.reset()

    world_model = WorldModel(db, config)
    query_engine = QueryEngine(config)
    llm_client = LLMClient(config)
    extractor = Extractor()
    action_parser = ActionParser()

    agent = TextWorldAgent(
        world_model=world_model,
        query_engine=query_engine,
        llm_client=llm_client,
        extractor=extractor,
        action_parser=action_parser,
        config=config,
    )
    logs = agent.run(env, max_steps=args.steps)
    db.close()

    # Print step logs
    for log in logs:
        print_step(
            step=log["step"],
            obs=log["observation"],
            action=log["action"],
            reward=log["reward"],
            source=log["source"],
            room=log["current_room"],
        )

    # Final Summary
    total_reward = sum(log["reward"] for log in logs)
    won = any("YOU WIN" in log["observation"] for log in logs)

    if HAS_RICH and console is not None:
        status_text = "[bold green]VICTORY! Game Completed![/bold green]" if won else "[bold yellow]Game Ended (Max Steps Reached)[/bold yellow]"
        console.print(Panel(
            f"{status_text}\nTotal Steps: {len(logs)}\nTotal Reward: {total_reward:.1f}",
            title="Game Results",
        ))
    else:
        print("\n=== Game Results ===")
        print("Outcome: VICTORY!" if won else "Outcome: Game Ended")
        print(f"Total Steps: {len(logs)}")
        print(f"Total Reward: {total_reward:.1f}")

    return 0 if won else 1


if __name__ == "__main__":
    sys.exit(main())
