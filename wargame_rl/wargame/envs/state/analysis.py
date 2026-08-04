"""Match analysis: compute structured metrics from a recorded event log.

Produces a MatchAnalysis report covering movement efficiency, tactical
quality, rule compliance, and degenerate behavior detection.
"""

from __future__ import annotations

import math
from collections import Counter

from pydantic import BaseModel, Field

from wargame_rl.wargame.envs.state.snapshot import GameStateSnapshot


class MatchAnalysis(BaseModel):
    """Structured analysis report for a recorded match."""

    file: str = ""
    steps: int = 0
    outcome: str = "unknown"

    # Movement efficiency
    objective_approach_rate: float = Field(
        default=0.0,
        description="Fraction of steps where closest objective distance decreased",
    )
    idle_rate: float = Field(
        default=0.0, description="Fraction of steps where Stay was chosen"
    )
    edge_contact_rate: float = Field(
        default=0.0, description="Fraction of model-steps at board edge"
    )
    mean_distance_to_objective: float = Field(
        default=0.0,
        description="Average closest-objective distance across all model-steps",
    )

    # Tactical quality
    mean_group_distance: float = Field(
        default=0.0,
        description="Average distance from a model to its nearest same-group model",
    )
    time_to_first_objective: int | None = Field(
        default=None,
        description="First step where any player model reaches an objective",
    )
    final_fraction_at_objectives: float = Field(
        default=0.0,
        description="Fraction of alive models on an objective at the final step",
    )
    peak_fraction_at_objectives: float = Field(
        default=0.0,
        description="Highest fraction of alive models on an objective in any step",
    )
    objective_drift_ratio: float = Field(
        default=1.0,
        description=(
            "peak / final occupancy. 1.0 means models held what they took; "
            "higher means they reached objectives and then left"
        ),
    )
    vp_per_step: float = Field(
        default=0.0, description="Total VP gained divided by episode steps"
    )
    target_selection_optimality: float | None = Field(
        default=None,
        description="Fraction of attacks that chose the highest expected-damage target",
    )

    # Rule compliance
    movement_violations: int = Field(
        default=0, description="Steps where a model moved further than max speed allows"
    )
    bounds_violations: int = Field(
        default=0, description="Snapshots where a model is outside board bounds"
    )

    # Degenerate behavior
    action_entropy: float = Field(
        default=0.0, description="Shannon entropy of action distribution (bits)"
    )
    oscillation_rate: float = Field(
        default=0.0,
        description="Fraction of models that returned to a previous position within 3 steps",
    )
    stagnation_detected: bool = Field(
        default=False,
        description="True if cumulative reward did not improve in the second half",
    )

    # Summary
    tactical_score: float = Field(default=0.0, description="Composite score 0-100")
    issues: list[str] = Field(
        default_factory=list, description="Human-readable issue flags"
    )

    def to_text(self) -> str:
        """Render as human-readable terminal report."""
        lines = [
            f"Match Analysis: {self.file}",
            "=" * 60,
            f"Steps: {self.steps} | Outcome: {self.outcome}",
            "",
            "MOVEMENT EFFICIENCY",
            f"  Objective approach rate: {self.objective_approach_rate:.1%}",
            f"  Idle rate:               {self.idle_rate:.1%}",
            f"  Edge contact rate:       {self.edge_contact_rate:.1%}",
            f"  Mean dist to objective:  {self.mean_distance_to_objective:.1f}",
            "",
            "TACTICAL QUALITY",
            f"  Mean group distance:     {self.mean_group_distance:.1f}",
            f"  Time to first objective: {self.time_to_first_objective or 'never'}",
            f"  VP per step:             {self.vp_per_step:.3f}",
            f"  On objectives (final):   {self.final_fraction_at_objectives:.2f}",
            f"  On objectives (peak):    {self.peak_fraction_at_objectives:.2f}",
            f"  Objective drift ratio:   {self.objective_drift_ratio:.2f}",
            f"  Target selection opt.:   {_fmt_opt(self.target_selection_optimality)}",
            "",
            "RULE COMPLIANCE",
            f"  Movement violations:     {self.movement_violations}",
            f"  Bounds violations:       {self.bounds_violations}",
            "",
            "DEGENERATE BEHAVIOR",
            f"  Action entropy:          {self.action_entropy:.2f} bits",
            f"  Oscillation rate:        {self.oscillation_rate:.1%}",
            f"  Stagnation detected:     {self.stagnation_detected}",
            "",
            f"TACTICAL SCORE: {self.tactical_score:.0f} / 100",
        ]
        if self.issues:
            lines.append("")
            lines.append("ISSUES:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        return "\n".join(lines)


def _fmt_opt(v: float | None) -> str:
    if v is None:
        return "N/A (no combat)"
    return f"{v:.1%}"


def analyze_match(
    snapshots: list[GameStateSnapshot],
    file_name: str = "",
    max_speed: int = 6,
) -> MatchAnalysis:
    """Compute all analysis metrics from an ordered list of snapshots.

    Args:
        snapshots: Ordered snapshots from ReplayController.iter_snapshots()
        file_name: Source filename for the report
        max_speed: Maximum movement distance per step (from env config n_speed_bins)
    """
    if not snapshots:
        return MatchAnalysis(file=file_name)

    first = snapshots[0]
    last = snapshots[-1]
    n_steps = len(snapshots) - 1

    outcome = "in_progress"
    if last.is_terminated:
        outcome = "terminated"
    elif last.is_truncated:
        outcome = "truncated"

    board_w = first.board_width
    board_h = first.board_height

    # Compute per-dimension metrics
    movement = _analyze_movement(snapshots, board_w, board_h, max_speed)
    tactical = _analyze_tactical(snapshots)
    rules = _analyze_rules(snapshots, board_w, board_h, max_speed)
    degenerate = _analyze_degenerate(snapshots)

    # Composite score
    issues: list[str] = []
    score = 50.0

    if movement.objective_approach_rate > 0.5:
        score += 15
    elif movement.objective_approach_rate < 0.2:
        score -= 15
        issues.append(
            f"Low objective approach rate ({movement.objective_approach_rate:.0%})"
        )

    if movement.idle_rate > 0.3:
        score -= 10
        issues.append(f"High idle rate ({movement.idle_rate:.0%})")

    if tactical.time_to_first_objective is not None:
        ratio = tactical.time_to_first_objective / max(n_steps, 1)
        if ratio < 0.5:
            score += 10
        elif ratio > 0.9:
            issues.append("Reached objective very late in episode")

    if tactical.vp_per_step > 0:
        score += 10

    if rules.movement_violations > 0:
        score -= 10
        issues.append(f"{rules.movement_violations} movement violations detected")
    if rules.bounds_violations > 0:
        score -= 10
        issues.append(f"{rules.bounds_violations} bounds violations detected")

    if degenerate.action_entropy < 1.0:
        score -= 15
        issues.append(
            f"Very low action entropy ({degenerate.action_entropy:.2f} bits) — likely stuck"
        )
    elif degenerate.action_entropy < 2.0:
        score -= 5
        issues.append(f"Low action diversity ({degenerate.action_entropy:.2f} bits)")

    if degenerate.oscillation_rate > 0.3:
        score -= 10
        issues.append(f"High oscillation ({degenerate.oscillation_rate:.0%})")

    if degenerate.stagnation_detected:
        score -= 10
        issues.append("Reward stagnation in second half of episode")

    if movement.edge_contact_rate > 0.3:
        score -= 5
        issues.append(f"Frequent edge contact ({movement.edge_contact_rate:.0%})")

    score = max(0.0, min(100.0, score))

    return MatchAnalysis(
        file=file_name,
        steps=n_steps,
        outcome=outcome,
        objective_approach_rate=movement.objective_approach_rate,
        idle_rate=movement.idle_rate,
        edge_contact_rate=movement.edge_contact_rate,
        mean_distance_to_objective=movement.mean_distance_to_objective,
        mean_group_distance=tactical.mean_group_distance,
        time_to_first_objective=tactical.time_to_first_objective,
        vp_per_step=tactical.vp_per_step,
        target_selection_optimality=tactical.target_selection_optimality,
        final_fraction_at_objectives=tactical.final_fraction_at_objectives,
        peak_fraction_at_objectives=tactical.peak_fraction_at_objectives,
        objective_drift_ratio=tactical.objective_drift_ratio,
        movement_violations=rules.movement_violations,
        bounds_violations=rules.bounds_violations,
        action_entropy=degenerate.action_entropy,
        oscillation_rate=degenerate.oscillation_rate,
        stagnation_detected=degenerate.stagnation_detected,
        tactical_score=score,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# Internal analysis passes
# ---------------------------------------------------------------------------


class _MovementMetrics(BaseModel):
    objective_approach_rate: float = 0.0
    idle_rate: float = 0.0
    edge_contact_rate: float = 0.0
    mean_distance_to_objective: float = 0.0


class _TacticalMetrics(BaseModel):
    mean_group_distance: float = 0.0
    time_to_first_objective: int | None = None
    vp_per_step: float = 0.0
    target_selection_optimality: float | None = None
    final_fraction_at_objectives: float = 0.0
    peak_fraction_at_objectives: float = 0.0
    objective_drift_ratio: float = 1.0


def _fraction_at_objectives(snapshot: GameStateSnapshot) -> float:
    """Fraction of alive player models standing on any objective."""
    alive = [m for m in snapshot.player_models if m.alive]
    if not alive:
        return 0.0
    return sum(1 for m in alive if any(m.at_objective)) / len(alive)


class _RuleMetrics(BaseModel):
    movement_violations: int = 0
    bounds_violations: int = 0


class _DegenerateMetrics(BaseModel):
    action_entropy: float = 0.0
    oscillation_rate: float = 0.0
    stagnation_detected: bool = False


def _analyze_movement(
    snapshots: list[GameStateSnapshot],
    board_w: int,
    board_h: int,
    max_speed: int,
) -> _MovementMetrics:
    if len(snapshots) < 2:
        return _MovementMetrics()

    approach_count = 0
    total_comparisons = 0
    idle_count = 0
    total_actions = 0
    edge_count = 0
    total_model_steps = 0
    all_distances: list[float] = []

    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        cur = snapshots[i]

        # Action analysis
        if cur.player_action_descriptions:
            for desc in cur.player_action_descriptions:
                total_actions += 1
                if desc == "Stay":
                    idle_count += 1

        # Per-model metrics
        for m_idx, (prev_m, cur_m) in enumerate(
            zip(prev.player_models, cur.player_models)
        ):
            if not cur_m.alive:
                continue
            total_model_steps += 1

            # Edge contact
            x, y = cur_m.location
            if x <= 0 or x >= board_w - 1 or y <= 0 or y >= board_h - 1:
                edge_count += 1

            # Objective approach
            if (
                prev_m.closest_objective_distance is not None
                and cur_m.closest_objective_distance is not None
            ):
                total_comparisons += 1
                if cur_m.closest_objective_distance < prev_m.closest_objective_distance:
                    approach_count += 1

            # Distance tracking
            if cur_m.closest_objective_distance is not None:
                all_distances.append(cur_m.closest_objective_distance)

    return _MovementMetrics(
        objective_approach_rate=(
            approach_count / total_comparisons if total_comparisons > 0 else 0.0
        ),
        idle_rate=idle_count / total_actions if total_actions > 0 else 0.0,
        edge_contact_rate=edge_count / total_model_steps
        if total_model_steps > 0
        else 0.0,
        mean_distance_to_objective=(
            sum(all_distances) / len(all_distances) if all_distances else 0.0
        ),
    )


def _analyze_tactical(snapshots: list[GameStateSnapshot]) -> _TacticalMetrics:
    if not snapshots:
        return _TacticalMetrics()

    # Group cohesion
    group_distances: list[float] = []
    time_to_first_obj: int | None = None
    total_optimal = 0
    total_attacks = 0

    for snap in snapshots:
        # Distance to the nearest *same-group* model, matching the definition
        # `group_cohesion` uses. An all-pairs army-wide mean measures dispersion
        # rather than cohesion, and ranks the two backwards: a policy that piles
        # every model onto one objective scores better than one that correctly
        # splits squads across three.
        alive_models = [m for m in snap.player_models if m.alive]
        nearest_same_group: list[float] = []
        for model in alive_models:
            squadmates = [
                other
                for other in alive_models
                if other is not model and other.group_id == model.group_id
            ]
            if not squadmates:
                continue
            nearest_same_group.append(
                min(math.dist(model.location, other.location) for other in squadmates)
            )
        if nearest_same_group:
            group_distances.append(sum(nearest_same_group) / len(nearest_same_group))

        # Time to first objective
        if time_to_first_obj is None:
            for m in snap.player_models:
                if m.alive and any(m.at_objective):
                    time_to_first_obj = snap.step
                    break

        # Target selection optimality
        if snap.player_combat_results:
            by_attacker: dict[int, list[float]] = {}
            for cr in snap.player_combat_results:
                by_attacker.setdefault(cr.attacker_idx, []).append(cr.expected_damage)

            for damages in by_attacker.values():
                total_attacks += 1
                if len(damages) == 1:
                    total_optimal += 1
                else:
                    actual = damages[0]
                    best = max(damages)
                    if actual >= best - 1e-6:
                        total_optimal += 1

    # VP efficiency
    first = snapshots[0]
    last = snapshots[-1]
    n_steps = len(snapshots) - 1
    vp_gained = last.player_vp - first.player_vp
    vp_per_step = vp_gained / n_steps if n_steps > 0 else 0.0

    # Occupancy over the episode. The drift ratio is the signal that started
    # this project's reward redesign: recorded matches showed the policy
    # reaching objectives and then leaving them, which every episode-level
    # metric reported as success because occupancy is judged at the last step.
    occupancy = [_fraction_at_objectives(s) for s in snapshots]
    final_occupancy = occupancy[-1] if occupancy else 0.0
    peak_occupancy = max(occupancy) if occupancy else 0.0

    return _TacticalMetrics(
        mean_group_distance=(
            sum(group_distances) / len(group_distances) if group_distances else 0.0
        ),
        time_to_first_objective=time_to_first_obj,
        vp_per_step=vp_per_step,
        target_selection_optimality=(
            total_optimal / total_attacks if total_attacks > 0 else None
        ),
        final_fraction_at_objectives=final_occupancy,
        peak_fraction_at_objectives=peak_occupancy,
        objective_drift_ratio=(
            peak_occupancy / final_occupancy if final_occupancy > 0 else float("inf")
        ),
    )


def _analyze_rules(
    snapshots: list[GameStateSnapshot],
    board_w: int,
    board_h: int,
    max_speed: int,
) -> _RuleMetrics:
    movement_violations = 0
    bounds_violations = 0

    for i, snap in enumerate(snapshots):
        for side_models in (snap.player_models, snap.opponent_models):
            for m in side_models:
                if not m.alive:
                    continue
                x, y = m.location
                if x < 0 or x >= board_w or y < 0 or y >= board_h:
                    bounds_violations += 1

        # Movement distance check (compare to previous step)
        if i > 0:
            prev = snapshots[i - 1]
            for prev_m, cur_m in zip(prev.player_models, snap.player_models):
                if not cur_m.alive or not prev_m.alive:
                    continue
                dx = cur_m.location[0] - prev_m.location[0]
                dy = cur_m.location[1] - prev_m.location[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > max_speed + 0.5:
                    movement_violations += 1

    return _RuleMetrics(
        movement_violations=movement_violations,
        bounds_violations=bounds_violations,
    )


def _analyze_degenerate(snapshots: list[GameStateSnapshot]) -> _DegenerateMetrics:
    if len(snapshots) < 2:
        return _DegenerateMetrics()

    # Action entropy
    action_counts: Counter[str] = Counter()
    for snap in snapshots:
        if snap.player_action_descriptions:
            for desc in snap.player_action_descriptions:
                action_counts[desc] += 1

    total_actions = sum(action_counts.values())
    entropy = 0.0
    if total_actions > 0:
        for count in action_counts.values():
            p = count / total_actions
            if p > 0:
                entropy -= p * math.log2(p)

    # Oscillation: a model *leaves* a cell and comes back within 3 steps.
    #
    # The `pos != previous` guard is what makes this measure oscillation rather
    # than stationarity. Without it, a model holding an objective — the exact
    # behaviour the reward is designed to produce — matched its own previous
    # position every step and scored ~70%, while a random walk scored ~5%.
    oscillation_events = 0
    total_model_steps = 0
    for m_idx in range(len(snapshots[0].player_models)):
        recent_positions: list[tuple[int, int]] = []
        for snap in snapshots:
            if m_idx >= len(snap.player_models):
                continue
            m = snap.player_models[m_idx]
            if not m.alive:
                continue
            pos = (m.location[0], m.location[1])
            total_model_steps += 1
            previous = recent_positions[-1] if recent_positions else None
            returned = pos in recent_positions[:-1] and pos != previous
            if returned:
                oscillation_events += 1
            recent_positions.append(pos)
            if len(recent_positions) > 4:
                recent_positions.pop(0)

    # Stagnation: compare cumulative reward in first vs second half
    rewards = [s.reward.total for s in snapshots[1:] if s.reward.total is not None]
    stagnation = False
    if len(rewards) >= 4:
        mid = len(rewards) // 2
        first_half_avg = sum(rewards[:mid]) / mid
        second_half_avg = sum(rewards[mid:]) / (len(rewards) - mid)
        if second_half_avg <= first_half_avg + 0.001:
            stagnation = True

    return _DegenerateMetrics(
        action_entropy=entropy,
        oscillation_rate=(
            oscillation_events / total_model_steps if total_model_steps > 0 else 0.0
        ),
        stagnation_detected=stagnation,
    )
