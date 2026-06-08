#!/usr/bin/env python3
"""Analyze recorded match event logs for training evaluation.

Usage:
    python analyze_events.py report recordings/my_events.jsonl
    python analyze_events.py report recordings/my_events.jsonl --json
    python analyze_events.py compare recordings/run1.jsonl recordings/run2.jsonl
"""

from __future__ import annotations

from pathlib import Path

import typer

from wargame_rl.wargame.envs.state import (
    JsonMatchCodec,
    MatchAnalysis,
    ReplayController,
    analyze_match,
)

app = typer.Typer(pretty_exceptions_enable=False)


def _load_and_analyze(file_path: str) -> MatchAnalysis:
    """Load an event log and produce a MatchAnalysis."""
    path = Path(file_path)
    if not path.exists():
        raise typer.BadParameter(f"File not found: {file_path}")

    codec = JsonMatchCodec()
    event_log = codec.decode(path.read_bytes())
    controller = ReplayController(event_log)
    snapshots = controller.iter_snapshots()
    return analyze_match(snapshots, file_name=path.name)


@app.command()
def report(
    file_path: str = typer.Argument(help="Path to the recorded event log JSONL file"),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON instead of human-readable text"
    ),
) -> None:
    """Analyze a single recorded match and print a structured report."""
    analysis = _load_and_analyze(file_path)

    if json_output:
        print(analysis.model_dump_json(indent=2))
    else:
        print(analysis.to_text())


@app.command()
def compare(
    file_paths: list[str] = typer.Argument(
        help="Paths to event log JSONL files to compare"
    ),
) -> None:
    """Compare multiple recorded matches side-by-side."""
    if len(file_paths) < 2:
        typer.echo("Need at least 2 files to compare.", err=True)
        raise typer.Exit(1)

    analyses = [_load_and_analyze(f) for f in file_paths]

    # Header
    col_width = 22
    header = f"{'Metric':<30}"
    for a in analyses:
        name = (
            a.file[:col_width]
            if len(a.file) <= col_width
            else a.file[: col_width - 1] + "…"
        )
        header += f" {name:>{col_width}}"
    print(header)
    print("=" * len(header))

    rows: list[tuple[str, list[str]]] = [
        ("Steps", [str(a.steps) for a in analyses]),
        ("Outcome", [a.outcome for a in analyses]),
        ("Obj approach rate", [f"{a.objective_approach_rate:.1%}" for a in analyses]),
        ("Idle rate", [f"{a.idle_rate:.1%}" for a in analyses]),
        ("Edge contact rate", [f"{a.edge_contact_rate:.1%}" for a in analyses]),
        ("Mean dist to obj", [f"{a.mean_distance_to_objective:.1f}" for a in analyses]),
        ("Mean group distance", [f"{a.mean_group_distance:.1f}" for a in analyses]),
        (
            "Time to first obj",
            [str(a.time_to_first_objective or "never") for a in analyses],
        ),
        ("VP per step", [f"{a.vp_per_step:.3f}" for a in analyses]),
        (
            "Target sel. opt.",
            [_fmt_opt(a.target_selection_optimality) for a in analyses],
        ),
        ("Movement violations", [str(a.movement_violations) for a in analyses]),
        ("Bounds violations", [str(a.bounds_violations) for a in analyses]),
        ("Action entropy", [f"{a.action_entropy:.2f}" for a in analyses]),
        ("Oscillation rate", [f"{a.oscillation_rate:.1%}" for a in analyses]),
        ("Stagnation", [str(a.stagnation_detected) for a in analyses]),
        ("TACTICAL SCORE", [f"{a.tactical_score:.0f}/100" for a in analyses]),
    ]

    for label, values in rows:
        row = f"{label:<30}"
        for v in values:
            row += f" {v:>{col_width}}"
        print(row)

    # Issues section
    print()
    for a in analyses:
        if a.issues:
            print(f"Issues ({a.file}):")
            for issue in a.issues:
                print(f"  - {issue}")


def _fmt_opt(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1%}"


if __name__ == "__main__":
    app()
