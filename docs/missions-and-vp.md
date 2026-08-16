# Missions and Victory Points

The environment tracks **victory points (VP)** for both the player and the opponent. How and when VP are scored is determined by the **mission** selected in the environment config.

## Mission config

In your environment YAML you can set:

```yaml
mission:
  type: default   # or "none" to disable VP scoring
  params:
    vp_per_objective: 5
    cap_per_turn: 15
    min_round: 2
```

- **type**: Selects the VP calculator. `default` scores VP for controlled objectives at end of command phase from a given round. `none` disables VP (always 0).
- **params**: Passed to the calculator. For `default`, the params are `vp_per_objective` (default 5), `cap_per_turn` (default 15) and `min_round` (default 2). Omit for built-in defaults. Note these are read outside the mission too: `vp_gain` divides by `cap_per_turn`, and `player_vp_min` derives its threshold from all three — so changing them rescales the reward and moves phase gates.

If you omit `mission` entirely, the default mission is used (VP per controlled objective, cap per turn, scoring from round 2).

## When VP are scored

For the default mission, VP are scored **at the end of the command phase** for each side, **from round 2 onward**. The side whose command phase just ended receives VP for each objective they control (see below), up to a cap per turn. Both the player and the opponent score when their own command phase ends.

### ⚠ The cap binds at three objectives, and the real tables carry five or six

Scoring is `min(cap_per_turn, controlled * vp_per_objective)` = `min(15, held * 5)` at the defaults, so **a fourth objective you control is worth zero additional VP**. Over 19 scoring rounds the ceiling is 285, and on `configs/evaluation/maps/` every competent policy saturates its own half of the scoreboard — 270–278 of 285 for the scripted baselines. **Above the cap, `vp_margin` is decided entirely by the opponent's score**, i.e. by denial.

Two consequences worth knowing before shaping anything against objective count:

- **`held` stops ranking policies once the cap binds.** `squad_march_deny` holds **3.00** objectives and `squad_march_shoot` **4.00**, and they score level. See [metrics.md](metrics.md).
- **Read `plr VP` and `opp VP` separately** — `measure-maps` prints both beside the margin, because the margin alone cannot say which half moved.

The cap is the rule, not an artefact: `rules/constants.yaml` sets `primary_cap_per_round: 15`. Raising `vp_per_objective` to "uncap" the mission would void every baseline measured on these maps. See [the report](../reports/2026-08-16-the-cap-makes-it-a-denial-game.md).

## Objective control

An objective is **controlled** by the side with the **greater Level of Control** within range — the number of that side's **alive** models within the objective’s radius (each model contributes its Control Value, currently 1; dead models are excluded from the count). The side with **strictly more** models in range controls the objective and scores it; **equal totals (including zero) mean the objective is contested/uncontrolled** and neither side scores. Control is re-evaluated each time VP are scored. See [rules/14-objectives.md](rules/14-objectives.md#level-of-control) for the rule, and [rules/implementation-status.md](rules/implementation-status.md#objectives-and-scoring) for where this simplifies it.

## Env state and observation

- **Battle state**: `player_vp` and `opponent_vp` are cumulative; `player_vp_delta` and `opponent_vp_delta` are the VP added during the current env step (reset at the start of each step, and read by the `vp_gain` reward as well as by the renderer).
- **Info**: Each step’s info dict includes `player_vp`, `opponent_vp`, `player_vp_delta`, and `opponent_vp_delta`.
- **Observation**: The agent observation includes `player_vp`, `opponent_vp`, and `player_vp_delta` in the game-feature vector so the policy can condition on score and step-wise VP gain. For VP-based reward shaping and phase success (e.g. `vp_gain` calculator, `player_vp_min` criteria, optional `terminal_vp_bonus`), see [reward-phases.md](reward-phases.md).

## Adding mission types

New mission types are registered in the VP calculator registry. Each calculator implements: given the current view, the side that is scoring, the current round, and which side owns the “player” models, return the VP to add. The env calls the calculator only when the clock is at the command phase (and for the default mission, from round 2).
