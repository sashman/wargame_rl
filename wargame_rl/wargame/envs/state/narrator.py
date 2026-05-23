"""StepNarrator: produce LLM-readable text summaries from a GameStateSnapshot.

Operates purely on snapshot data — no env or config dependency.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.state.snapshot import (
    CombatResultSnapshot,
    GameStateSnapshot,
    ModelSnapshot,
    ObjectiveSnapshot,
)


class StepNarrator:
    """Produces structured text summaries of game steps for LLM consumption."""

    def narrate(self, snapshot: GameStateSnapshot) -> str:
        """Produce a full text summary of the game state at this step."""
        sections = [
            self._header(snapshot),
            self._board(snapshot),
            self._force_status(snapshot),
            self._models(snapshot),
            self._objectives(snapshot),
            self._actions(snapshot),
            self._combat(snapshot),
            self._reward(snapshot),
            self._status(snapshot),
        ]
        return "\n\n".join(s for s in sections if s)

    def _header(self, s: GameStateSnapshot) -> str:
        clock = s.clock
        parts = [f"Step {s.step} / {s.max_steps}"]
        if clock.battle_round is not None:
            parts.append(f"Round {clock.battle_round}")
        if clock.battle_phase is not None:
            parts.append(clock.battle_phase.replace("_", " ").title())
        if clock.active_player is not None:
            parts.append(clock.active_player.replace("_", " ").title())
        return f"=== {' | '.join(parts)} ==="

    def _board(self, s: GameStateSnapshot) -> str:
        return f"BOARD: {s.board_width}x{s.board_height} | Mission: {s.mission_type}"

    def _force_status(self, s: GameStateSnapshot) -> str:
        lines = ["FORCE STATUS:"]
        lines.append(
            f"  Player: {s.player_alive_count} alive "
            f"({s.player_total_wounds} wounds) | "
            f"Opponent: {s.opponent_alive_count} alive "
            f"({s.opponent_total_wounds} wounds)"
        )
        lines.append(f"  Player VP: {s.player_vp} | Opponent VP: {s.opponent_vp}")
        return "\n".join(lines)

    def _model_line(self, idx: int, m: ModelSnapshot) -> str:
        status = "Alive" if m.alive else "Dead"
        wound_str = f"{m.current_wounds}/{m.max_wounds} wounds"
        pos = f"({m.location[0]}, {m.location[1]})"
        parts = [f"  Model {idx} [{status} {wound_str}] at {pos}"]
        if m.alive and m.closest_objective_idx is not None:
            dist = (
                f"{m.closest_objective_distance:.1f}"
                if m.closest_objective_distance is not None
                else "?"
            )
            parts.append(
                f"nearest objective: Obj {m.closest_objective_idx} ({dist} away)"
            )
        return " — ".join(parts)

    def _models(self, s: GameStateSnapshot) -> str:
        lines = ["PLAYER MODELS:"]
        for i, m in enumerate(s.player_models):
            lines.append(self._model_line(i, m))
        if s.opponent_models:
            lines.append("")
            lines.append("OPPONENT MODELS:")
            for i, m in enumerate(s.opponent_models):
                lines.append(self._model_line(i, m))
        return "\n".join(lines)

    def _objectives(self, s: GameStateSnapshot) -> str:
        lines = ["OBJECTIVES:"]
        for i, o in enumerate(s.objectives):
            ctrl = s.objective_control[i] if i < len(s.objective_control) else "?"
            pos = f"({o.location[0]}, {o.location[1]})"
            in_range = self._models_in_range_str(o)
            lines.append(f"  Obj {i} at {pos} — {ctrl} | {in_range}")
        return "\n".join(lines)

    def _models_in_range_str(self, o: ObjectiveSnapshot) -> str:
        parts: list[str] = []
        if o.player_models_in_range:
            ids = ", ".join(str(i) for i in o.player_models_in_range)
            parts.append(f"player models {ids}")
        if o.opponent_models_in_range:
            ids = ", ".join(str(i) for i in o.opponent_models_in_range)
            parts.append(f"opponent models {ids}")
        return ", ".join(parts) if parts else "no models in range"

    def _actions(self, s: GameStateSnapshot) -> str:
        if not s.player_action_descriptions:
            return ""
        lines = ["ACTIONS TAKEN:"]
        for i, desc in enumerate(s.player_action_descriptions):
            lines.append(f"  Model {i}: {desc}")
        return "\n".join(lines)

    def _combat_line(self, cr: CombatResultSnapshot, side: str) -> list[str]:
        target_side = "Opponent" if side == "Player" else "Player"
        main = (
            f"  {side} {cr.attacker_idx} -> {target_side} {cr.target_idx}: "
            f"{cr.hits} hits, {cr.wounds} wounds, "
            f"{cr.unsaved} unsaved, {cr.damage_dealt} damage dealt"
        )
        analytics = (
            f"    (expected {cr.expected_damage:.2f} dmg, "
            f"{cr.hit_probability * 100:.0f}% hit, "
            f"{cr.wound_probability * 100:.0f}% wound)"
        )
        return [main, analytics]

    def _combat(self, s: GameStateSnapshot) -> str:
        if not s.player_combat_results and not s.opponent_combat_results:
            return ""
        lines = ["COMBAT:"]
        for cr in s.player_combat_results:
            lines.extend(self._combat_line(cr, "Player"))
        for cr in s.opponent_combat_results:
            lines.extend(self._combat_line(cr, "Opponent"))
        return "\n".join(lines)

    def _reward(self, s: GameStateSnapshot) -> str:
        r = s.reward
        total_str = f"{r.total:.3f}" if r.total is not None else "None"
        lines = [f"REWARD [{r.phase_name}, phase {r.phase_index}]: {total_str}"]

        top_level = {k: v for k, v in r.breakdown.items() if "/" not in k}
        sub_level = {k: v for k, v in r.breakdown.items() if "/" in k}

        for key, val in top_level.items():
            lines.append(f"  {key}: {val:.3f}")
            prefix = f"{key}/"
            for sub_key, sub_val in sub_level.items():
                if sub_key.startswith(prefix):
                    label = sub_key[len(prefix) :]
                    lines.append(f"    {label}: {sub_val:.3f}")

        return "\n".join(lines)

    def _status(self, s: GameStateSnapshot) -> str:
        if s.is_terminated:
            return "STATUS: Terminated"
        if s.is_truncated:
            return "STATUS: Truncated"
        return "STATUS: In progress"
