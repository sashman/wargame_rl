#!/usr/bin/env python3
"""Replay and inspect recorded match event logs.

Usage:
    python replay_events.py narrate recordings/my_events.jsonl
    python replay_events.py seek recordings/my_events.jsonl --step 5
    python replay_events.py summary recordings/my_events.jsonl
"""

from __future__ import annotations

from pathlib import Path

import typer

from wargame_rl.wargame.envs.state import (
    JsonMatchCodec,
    ReplayController,
    StepEvent,
    StepNarrator,
)

app = typer.Typer(pretty_exceptions_enable=False)
narrator = StepNarrator()


def _load_log(file_path: str) -> ReplayController:
    """Load an event log from a JSONL file and return a ReplayController."""
    path = Path(file_path)
    if not path.exists():
        raise typer.BadParameter(f"File not found: {file_path}")
    codec = JsonMatchCodec()
    event_log = codec.decode(path.read_bytes())
    return ReplayController(event_log)


@app.command()
def narrate(
    file_path: str = typer.Argument(help="Path to the recorded event log JSONL file"),
) -> None:
    """Narrate every step of a recorded match in human-readable text."""
    controller = _load_log(file_path)
    snapshots = controller.iter_snapshots()

    for snapshot in snapshots:
        print(narrator.narrate(snapshot))
        print()
        print("-" * 60)
        print()


@app.command()
def seek(
    file_path: str = typer.Argument(help="Path to the recorded event log JSONL file"),
    step: int = typer.Option(..., help="Step number to seek to"),
) -> None:
    """Reconstruct and narrate the game state at a specific step."""
    controller = _load_log(file_path)
    try:
        snapshot = controller.seek(step)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    print(narrator.narrate(snapshot))


@app.command()
def summary(
    file_path: str = typer.Argument(help="Path to the recorded event log JSONL file"),
) -> None:
    """Print a summary of a recorded match."""
    path = Path(file_path)
    if not path.exists():
        raise typer.BadParameter(f"File not found: {file_path}")

    codec = JsonMatchCodec()
    event_log = codec.decode(path.read_bytes())
    controller = ReplayController(event_log)

    first_snap = controller.seek(controller.first_step)
    last_snap = controller.seek(controller.last_step)

    n_anchors = sum(
        1 for e in event_log.events if isinstance(e, StepEvent) and e.anchor is not None
    )

    file_size_kb = path.stat().st_size / 1024

    print(f"Match Event Log: {path.name}")
    print(f"{'=' * 50}")
    print(f"Total events:     {len(event_log)}")
    print(f"Steps:            {controller.total_steps}")
    print(f"Step range:       {controller.first_step} – {controller.last_step}")
    print(f"Anchor snapshots: {n_anchors}")
    print(f"Anchor interval:  {event_log.anchor_interval}")
    print(f"File size:        {file_size_kb:.1f} KB")
    print()
    print(f"Board:            {first_snap.board_width}x{first_snap.board_height}")
    print(f"Mission:          {first_snap.mission_type}")
    print(f"Player models:    {len(first_snap.player_models)}")
    print(f"Opponent models:  {len(first_snap.opponent_models)}")
    print(f"Objectives:       {len(first_snap.objectives)}")
    print()
    print(
        f"Final VP:         Player {last_snap.player_vp} – Opponent {last_snap.opponent_vp}"
    )
    print(
        f"Final alive:      Player {last_snap.player_alive_count} – Opponent {last_snap.opponent_alive_count}"
    )

    if last_snap.is_terminated:
        print("Outcome:          Terminated (game ended)")
    elif last_snap.is_truncated:
        print("Outcome:          Truncated (max steps)")
    else:
        print("Outcome:          In progress (recording ended mid-game)")


if __name__ == "__main__":
    app()
