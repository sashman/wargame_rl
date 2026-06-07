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
- **params**: Passed to the calculator. For `default`, common params are `vp_per_objective`, `cap_per_turn`, and `min_round`. Omit for built-in defaults.

If you omit `mission` entirely, the default mission is used (VP per controlled objective, cap per turn, scoring from round 2).

## When VP are scored

For the default mission, VP are scored **at the end of the command phase** for each side, **from round 2 onward**. The side whose command phase just ended receives VP for each objective they control (see below), up to a cap per turn. Both the player and the opponent score when their own command phase ends.

## Objective control

An objective is **controlled** by the side with the **greater Level of Control** within range — the number of that side's models within the objective’s radius (each model contributes its Objective Control value, currently 1). The side with **strictly more** models in range controls the objective and scores it; **equal totals (including zero) mean the objective is contested/uncontrolled** and neither side scores. Control is re-evaluated each time VP are scored. This matches the tabletop OC rule — see [tabletop-rules-reference.md](tabletop-rules-reference.md).

## Env state and observation

- **Battle state**: `player_vp` and `opponent_vp` are cumulative; `player_vp_delta` and `opponent_vp_delta` are the VP added during the current env step (for display).
- **Info**: Each step’s info dict includes `player_vp`, `opponent_vp`, `player_vp_delta`, and `opponent_vp_delta`.
- **Observation**: The agent observation includes `player_vp`, `opponent_vp`, and `player_vp_delta` in the game-feature vector so the policy can condition on score and step-wise VP gain. For VP-based reward shaping and phase success (e.g. `vp_gain` calculator, `player_vp_min` criteria, optional `terminal_vp_bonus`), see [reward-phases.md](reward-phases.md).

## Adding mission types

New mission types are registered in the VP calculator registry. Each calculator implements: given the current view, the side that is scoring, the current round, and which side owns the “player” models, return the VP to add. The env calls the calculator only when the clock is at the command phase (and for the default mission, from round 2).
