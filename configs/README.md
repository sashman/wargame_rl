# Configs

Environment configs, tiered by **what breaks if you change the file**.

| Directory | Changing a file here… | Lifetime |
|---|---|---|
| `golden/` | invalidates a published number | permanent |
| `experiments/` | affects one open question | delete when answered |
| `evaluation/maps/` | changes what final evaluation measures | permanent |
| `dev/` | breaks a test or a demo, never a result | permanent |

## `golden/` — results-bearing

Each of these backs a number quoted in `CLAUDE.md` or a report. Do not edit one
to try something; copy it into `experiments/` instead.

| Config | What it established | Comparable to |
|---|---|---|
| `25v25_shooting_opponent.yaml` | **Beats the shooting bar**: +28.4 vp_margin against `squad_march_shoot`'s +17.0, both seeds, n=100. The lever is `objective_hold.crowding_exponent`. [Report](../reports/2026-08-08-paying-the-pot-beats-the-bar.md) | `25v25_cover_control.yaml` |
| `25v25_cover_control.yaml` | Batch-3 control. Cover is not used even with a working geometry and a priced trade. [Report](../reports/2026-08-06-cover-signal-reason-geometry.md) | `25v25_shooting_opponent.yaml` |
| `25v25_single_phase.yaml` | The single-phase control for the curriculum question | `25v25_curriculum.yaml` |
| `25v25_curriculum.yaml` | Two-rung ladder; final phase identical to the control's, so the comparison isolates the curriculum | `25v25_single_phase.yaml` |

**The two scenarios are not comparable to each other.** The shooting pair faces
`scripted_advance_and_shoot` on terrain regenerated every episode with
`objective_min_separation`; the curriculum pair faces the non-shooting
`scripted_advance_to_objective` on fixed terrain. Switching a config's opponent
or terrain profile invalidates every baseline and agent score measured on it —
re-measure both sides, on identical seeds, or the comparison is meaningless.

Two of these are **frozen bit-identically** by `tests/test_reward_golden.py` and
`tests/test_observation_golden.py` (`25v25_single_phase`, `25v25_shooting_opponent`,
plus `dev/4v4_two_phases`). Any edit that changes a reward or an observation
fails those tests loudly. That is the intended behaviour, not an obstacle:
regenerate the fixtures only when you mean to change the numbers.

## `experiments/` — arms

Where a screen's arms live while its question is open. Copy the control, change
**one thing**, and give it a distinct `config_name` — run names are built from
it, and two arms with the same name write checkpoints into one directory and
score whichever process saved last.

Delete an arm once its question is answered; `git log -- configs/` restores any
of them. Every batch so far was disposed of this way (batch 1/2's arms, batch
3's `cover_reason`, batch 4's eight `25v25_beat_*`).

## `evaluation/maps/` — the real table layouts

Final evaluation. Training uses `random_terrain`, which is what makes a
positioning result falsifiable — but it never asks how the policy does on the
boards the game is actually played on.

A map is terrain only:

```yaml
name: table_01
terrain:
  - { footprint: [12, 8, 18, 14] }
  - { footprint: [27, 20, 33, 26] }
```

`just measure-maps <ckpt> configs/golden/25v25_shooting_opponent.yaml` runs the
golden scenario unchanged and swaps only `terrain`, once per map, reporting a
row each plus the spread. Quote it against the bar on the same maps:

```bash
just measure-maps squad_march_shoot configs/golden/25v25_shooting_opponent.yaml
```

There is deliberately **no config per map**. A 25v25 scenario is ~10 KB, so
copying it per map means every future reward change must be applied N times —
and the first one missed makes evaluation measure a different game from
training, without failing anything.

## `dev/` — fixtures and demos

Fast, small, and no result is ever quoted from them.

| Config | Used by |
|---|---|
| `ci_smoke.yaml` | `tests/test_z_e2e_training.py`, the README resume examples |
| `4v4_two_phases.yaml` | the `just train` default; frozen by both golden tests |
| `terrain_los_demo.yaml` | `tests/test_terrain_render.py`, `docs/terrain.md` |
| `tiny.yaml` | `train.py`'s default, the `just record` default |

## Conventions

- **Set `config_name`.** It leads the run name, and everything after it
  describes the *scenario* — which arms of one experiment deliberately share.
- **Repeated model entries use YAML anchors.** Five squads of five are one
  profile; the anchor single-sources it so a weapon change is one edit. YAML
  resolves anchors at parse time, so the config the env sees, and the copy
  dumped beside each checkpoint, is fully expanded either way.
- **Comment the *why*, with the measurement.** The golden configs carry the
  numbers that justify each term. That is why they are long, and it is the part
  worth keeping.
