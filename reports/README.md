# Experiment reports

> **Paths in these reports predate 2026-08-09.** Configs moved from
> `examples/env_config/` to `configs/{golden,dev,experiments}/`, and most of the
> configs named here were deleted once their question was answered. The old
> paths are left as written — a report records what was true when it was run.
> `git log -- examples/ configs/` finds any of them.

Findings from training experiments, kept for retrospection. Each report records what
was asked, what was measured, and what the measurement does and does not support.

## Index

| Report | Question | Outcome |
|---|---|---|
| **[2026-08-12 — The physics were off, and then the agent beat the bar](2026-08-12-the-physics-were-off-and-the-agent-beat-the-bar.md)** | Beat `scripted_advance_and_shoot` on the 25v25 shooting scenario. | **Achieved** — but first: `configs/golden/25v25_shooting_opponent.yaml` never set `base_radius`, so it took the old default of **0.0**, at which the whole of phase 03 is a documented no-op — no occlusion, **cover cannot occur at all**, no collision, centre-to-centre range. It was the pre-geometry scenario on the post-geometry engine. With 32mm bases, objectives as the largest unclustered middle-section ruins and terrain refitted to the 45 real layouts, the bar went +6.7 → +27.8 → **+38.0**. `objective_hold` 1.25→2.5 with `objective_coverage` 0.3→0.1 — same budget, global term to per-model — scores **+14.2 ± 6.1 paired** over the bar (64/89, one seed), and **+8.8 ± 5.8 zero-shot** on the goal's own config with terrain it never trained on (59/90, p ≈ 0.003). The control reproduces at +6.2/+4.8 across two seeds. **γ = 0.97 was the lead hypothesis and came last** (−8.8, retired at the 300 screen). Every baseline was marching its squads onto the objective *centroid*, because `radius_size` is 0.0 for an area by design. And **four claims died to proper pairing, all of them mine** — +8.0→+1.7, +10.2→+7.5, +7.6→+0.8, and a 300-epoch arm ranking that reversed |
| **[2026-08-11 — What a seven-pass audit found](2026-08-11-what-a-seven-pass-audit-found.md)** | Do the day's conclusions survive review? | **Four of them do not** — and the pattern matters more than any one: **every error was in how data was bucketed or paired, none was in the code under test.** The environment, reward and network were sound; the probes ran correctly and measured something other than what their author believed. The "transit trough" was a battle-phase artefact (walking pays **41% more** than loitering at fixed phase, not half); the local-optimum verdict read the **team scalar** while PPO trains on the per-model vector, where the same deviation *costs* the movers 29.4; +21.1 vp came from a `zip` that diverged after the first skipped episode; and a global reward term cannot create a preference between two things a model might do. Cost: one hour, no GPU — against ~7 GPU-hours per arm it would have prevented. Carries the outstanding items and the audit composition that worked |
| **[2026-08-11 — Spreading objectives lowers the bar, not the abandonment](2026-08-11-spreading-objectives-lowers-the-bar-not-the-abandonment.md)** | The agent abandons 38% of objectives. Is that reward weighting, or objectives packed into a 16" circle? | **Neither.** Abandonment across five weight configurations and two scenarios: 0.380, 0.370, 0.357, against the bar's 0.147. It is pushed off an objective **1.7%** of the time against the bar's 24.7% — it is not losing fights, it is not turning up. What *did* replicate: moving income from the **global** `objective_coverage` to the **per-model** `objective_hold` is worth **+9.8 vp** on both seeds, up from a weak +5.2 on the other scenario. Refuted: `overstack_penalty_per_extra` (zeroing it is null), and the prediction that spreading objectives would fix allocation — measured at 3.6 sigma as a natural experiment, it did not transfer to training. The deficit narrowed **-17.1 -> -9.9**, but by the **bar falling 8.8** while the agent stayed flat. Remaining hypothesis is representational: nothing encodes "an uncontested objective is worth a 20-inch walk", and the reward pays zero during the walk |
| **[2026-08-10 — Real terrain, and the abandonment gap](2026-08-10-real-terrain-and-the-abandonment-gap.md)** | First training on the continuous board with polygon terrain, model bases and cover. Does the agent still beat the shooting bar? | **No — +50.2 / +51.3 against the bar's +67.7**, two seeds within 1.1 vp, converged by epoch 680. The gap is entirely *conceded* VP: player VP is a dead heat, opponent VP is 20 higher. `measure-objective-split` says why — the agent **abandons 38% of objectives** against the bar's 16% while being pushed off only 0.7% of the time, and parks **8.6 surplus models** on the points it holds against the bar's 3.4. It is not losing objectives, it never goes. That is exactly what `objective_hold.crowding_exponent` prices, and it is too weak here. It is meanwhile *better at fighting*: firepower 1.76 v 0.88, alive 0.65 v 0.51, **exposure 0.245 v 0.438**. Getting here first required scrapping a run at epoch 470: the terrain profile had been tuned against `measure-terrain`'s own hidden-fraction metric, which rewards piece count, and `_coverage` billed bounding boxes — together giving **0.194 hidden against the real game's 0.088** and objectives that held 5 models instead of 18. Refitted to the 45 real layouts |
| **[2026-08-09 — TF32 costs 8.5 vp_margin](2026-08-09-tf32-costs-eight-vp.md)** | A fresh run of the golden config scored +21.2/+19.9 where the published figure was +28.4. What changed? | **TF32**, which shipped on by default on 2026-08-08 and had never been measured against a trained result. At matched epoch 1000, n=100 identical layouts: **s1 +30.8 → +21.2, s2 +27.4 → +19.9** — beating the bar by 12.1 becomes beating it by 3.6. The `--no-tf32` control reproduced the pre-TF32 run **bit-identically** (222/222 tensors, max abs diff 0.0), which proves TF32 is the whole effect *and* re-confirms determinism across every other change in the window. It buys **17.8%** of an epoch, not the 1.34x the update-only benchmark implies. **Now off by default.** The refuted claim — "below every effect size this project can resolve" — was inferred from the mantissa drop and a throughput benchmark, never measured. **Win rate would have missed it** (0.705 → 0.65, inside the ~7pp limit); `vp_margin` caught it. Lesson: a precision setting is a reward-affecting change |
| **[2026-08-08 — paying the pot beats the bar](2026-08-08-paying-the-pot-beats-the-bar.md)** | Beat `scripted_advance_and_shoot`. | **Achieved, on both seeds.** Pricing objective *crowding* — paying a point a fixed pot split between its occupants rather than every occupant the same wage — takes the agent from +2.5 to **+28.4 vp_margin**, past the `squad_march_shoot` bar at +17.0 (n=100, identical layouts, +30.1 and +26.6 on the two seeds). The lever is `objective_hold`'s `crowding_exponent`. **Confound-controlled:** the same weight with a *flat* exponent scores **−40.4**, worse than the control, piling 20 of 21 survivors onto one objective — so the exponent is worth 68 vp at fixed weight and "objectives just pay more" is refuted. Why it worked where two earlier anti-stacking levers failed: **it redistributes reward instead of destroying it** — a penalty and a discount both lower total objective income, so the policy reads either as "objectives pay less". Also: `last.ckpt` was **not the last epoch** (it held epochs 970/692/948/998), so every prior "at N epochs" score was really "at whatever epoch that run last improved" |
| **[2026-08-06 — beat the shooting opponent](2026-08-06-beat-the-shooting-opponent.md)** | Why does the agent lose to `scripted_advance_and_shoot`, and what actually closes the gap? | **The agent was never ahead of the bar** — batch 3 scored agent and baseline on different layout sets; on matched seeds it is 0.67 against 0.77. Two binding defects found, neither about cover: the reward makes objective **stacking provably profitable** (hold pays +0.25/step, the anti-stack knob charges at most 0.24), and **36–40% of shots are discarded** as overkill because the policy is factorized over models. Decoding targets autoregressively is worth **+10.5 vp_margin on the existing checkpoint with no retraining**. **Closed 2026-08-08 — every lever here measured null or negative**; the goal was met by [the follow-up](2026-08-08-paying-the-pot-beats-the-bar.md). Two conclusions carry inline corrections: the lowered prior on anti-stacking, and the diagnosis that the levers failed because occupancy was unobservable (adding the observation was worth +1.55 ± 4.5). All its numbers were scored through a `last.ckpt` that was not the last epoch |
| [2026-08-06 — cover: signal, reason, geometry](2026-08-06-cover-signal-reason-geometry.md) ⚠️ **corrected** | With all three blockers removed — 19.8% of the board hidden, a per-model LOS input, and losses priced — does the agent use cover? | **Still no**, and this time the null is clean. Exposure 0.092–0.110 across a 2x2, two seeds each. Side findings are stronger: the `models_lost` penalty is worth **+7 vp_margin** (non-overlapping seeds), the `observe_threat_count` input is **null**, and the penalty made the agent lose *more* models, not fewer — the opposite of the mechanism it was added for. `firepower_advantage` is broken as specified: `random` outscores every trained arm |
| [2026-08-05 — stochastic terrain and cover](2026-08-05-stochastic-terrain-and-cover.md) | Does the agent learn to use terrain for cover when the opponent shoots back? | **No — it learns range management.** Deleting every ruin moves exposure 0.116 → 0.120; return fire alone moves it 4.7x. At weapon range 24, where distance stops working, the policy collapses to 6.8% win. Reward *and* opponent curricula both unnecessary. Objective-placement defect found and fixed: 25% of episodes had overlapping objectives |
| **[2026-08-04 — correction: what was actually broken](2026-08-04-correction-what-was-actually-broken.md)** | **Read this first.** Is the policy learning at all? | **No.** The movement head never left initialisation; a 12-line heuristic beats 945 epochs of training 80% to 17%. Four further defects found; most conclusions below retracted |
| [2026-08-04 — reward-phase curriculum](2026-08-04-reward-phase-curriculum.md) | Why does the 25v25 curriculum never leave phase 0, and can it reach `win_at_the_end`? | ⚠️ Substantially retracted — training reward never left phase 0, so no cross-run conclusion holds |
| [2026-08-04 — mechanism defects](2026-08-04-mechanism-defects.md) | Detailed evidence for each defect found | D1/D2/D4 survive (arithmetic); D3/D5's numbers measured a near-random policy |
| [2026-08-04 — objective drift](2026-08-04-objective-drift.md) | What does the agent actually do during an episode? | Drift is real; ⚠️ its explanation is superseded — a uniform policy produces the same trace |
| [2026-08-04 — reward calculation changes](2026-08-04-reward-calculation-changes.md) | What changed in the reward, and what does the evidence support? | Reward became per-model; two calculators added, three dropped, one arithmetic bug fixed. Win 0.17 → 0.93–0.97 vs a 0.67 movement bar — but **confounded** with the PPO fixes that shipped alongside |

## Conventions

**Distinguish measurement from inference.** Every quantitative claim states how it was
obtained. Claims that are inferred rather than measured say so.

**Report negative results.** A refuted hypothesis costs a run to establish and costs
another run every time it is retried by someone who does not know it failed.

**State confounds.** Most runs here changed more than one variable, were stopped early,
and used a single seed. That is adequate for finding defects and inadequate for ranking
hyperparameters. Reports say which of the two they support.

**Prefer the live metric.** `success_rate` from a best checkpoint is not the same
quantity as the per-epoch `success_rate` a gate reads. See
[mechanism defects, D4](2026-08-04-mechanism-defects.md#d4--rungs-calibrated-against-the-wrong-distribution).

**Quote every number against a floor and a bar.** A `success_rate` alone says nothing —
17% read as progress until a scripted baseline scored 80% on the same board. Run
`just measure-baselines <env_config>` before interpreting any result; training logs
`eval/baseline_*` automatically. Note every number in these reports was measured against
`scripted_advance_to_objective`, an opponent that never fires — the `squad_march_shoot`
bar of 1.00 is partly an artefact of that, and falls to 0.60 against
`scripted_advance_and_shoot`.

**Verify the thing you are tuning is the thing being trained.** Seven runs tuned reward
phases whose rewards never reached the gradient.

## Reproducing

```bash
# Scripted baseline scores -- the floor and the bar. Run this before reading any result.
# Pass a seed_base to score on the same layouts as measure-checkpoint.
just measure-baselines <env_config> [n_episodes] [record] [seed_base]

# Score a checkpoint through the baselines' own code path, so the two are comparable
just measure-checkpoint <checkpoint> <env_config> [n_episodes] [record]

# Rolling-mean metrics for a run (point readings are n-sample binomials -- do not trust them)
just run-summary <run_id> [bucket_size]

# Per-phase criteria rates and the whole min_fraction curve for a checkpoint
just measure-phase-gates <checkpoint> <env_config> 40

# Behavioural analysis of a recorded match
just replay-summary <file>
just analyze <file>
just analyze-compare <files...>
```

Recordings live in `recordings/` (gitignored). Runs are in the `wargame_rl/wargame_rl`
Weights & Biases project; run IDs are given in each report.
