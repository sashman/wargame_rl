# Curriculum rung C2: the enemy squad on the point fires back — two seeds clear the mark, one is a hundredth short, and the whole-army control fails on all three

**Verdict first.** On the first guns rung (#340, arm #368) — C1's four
unarmed squads and four points with the enemy unit on its point armed
and firing (`hold_and_shoot`, range 12, one attack per model, the
shooting phase stepped) — the per-model arm warm-started from C1 reads
greedy success ****0.890 / 0.920 / 0.950**** at 245,760 rounds, n=100, `held` 2.87 / 2.88 / 3.00 of
3, turns 10.21 / 9.84 / 9.79 phase-clock against the script's 9.60 and the
whole-army control's 14.2–14.8. ****FAIL as pre-registered, by one seed at one hundredth under the 0.90 mark** (binomial SE 0.03; the pre-registration said the verdict would not be softened for a seed one SE under, and it is not). Seed 1 is also the one seed with an in-run dip under the drift clause (rolling 73% at 123k, from a raw 63% at 121k, recovered by 124k).** The control fails its
own criterion on every seed at its extended budget (**0.790 / 0.690 /
0.780** at 120 epochs) on a rung the script passes at 0.900: under A5's
clause that is a finding about the control, and it is the first rung on
the ladder where the per-model arm is the trainer that reaches the mark while the control does not: the arm is at the script's speed (turns 9.8–10.2 against 9.60) and the script's survival (`alive` 0.826 / 0.842 / 0.856 against the bar's 0.842), and the control arrives two and a half rounds late (14.2–14.8 of 16) and misses a point in a fifth to a third of episodes. The prediction on file — a pass on all three inside 40k rounds at about the bar's `alive` — is right on the speed and the survival and wrong by one seed and one hundredth on the mark.

## Provenance

| field | value |
|---|---|
| date | launched 2026-09-16 22:18 (control and arm together); first leg exited at 122,880 at 01:50–01:51 on 09-17, resumed in place 01:51, exited 03:43 / 03:44 / 03:43 |
| GPU / no-GPU | GPU (RTX 4090), beside A5b, T1 and the control's extension — twelve trainers at launch, six from 01:51 |
| seeds | 1 / 2 / 3; rollout layouts at seed×100+; each seed from C1's `last.pt` of the same seed (245,760 rounds on C1: 0.990 / 0.930 / 0.990), fresh optimizer — **not paired on init with the control** |
| n | 100 at seed base 700000 (final, the 122,880 readout, `alive`); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 20 at 700000 (census) |
| config | `configs/experiments/curriculum/c2.yaml` — unrefereed by design; success `all_objectives_occupied` counts our bodies; `skip_phases: [command, charge, fight]` so shooting is stepped (two agent phases per round: turns are phase-clock, 16 in an eight-round game) |
| decode | none on the per-model facade; K=1 on the control |
| paired | per episode against `squad_march_take` on identical seeds |
| comparator | `squad_march_take` (unarmed here), both facades identical: success 0.900, held 2.92, `alive` 0.842, `on_obj` 0.909, turns 9.60, vp −2.7 ± 0.8, coherent 0.932 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one declared unit, on the point at (35, 27), a rifle of range 12 and one attack each, firing at the nearest valid unit every shooting phase |
| budget | per-model 245,760 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`, resumed in place at 122,880 (#346: cadences explicit); control 60 epochs extended once to 120 (245,760 rounds) — the symmetric cap |
| code revision | `1b993eb` on `feature/curriculum-c2` (PR #369) at launch; the resumed leg on `236c69a` (docs only between them) |
| checkpoint | `last.pt` at 245,760; `checkpoints/per_model/per-model-c2-2026-09-16-22-18-5{9,9,8}-s{1,2,3}c2`; control `checkpoints/ppo-transformer-curriculum_c2-…-s{1,2,3}c2-ctl-x2/last.ckpt` |
| coherency | greedy 0.582 / 0.563 / 0.662 at 700000+; 0.63 / 0.56 / 0.65 greedy against **0.38 / 0.40 / 0.42** sampled at 900000+ |
| Wandb | `curriculum-c2`: per-model `3ek95n2i` / `1pam14x6` / `yzoqo4cw`, resumed `8fa0j8m6` / `m2rkug75` / `9khjcvpg`; control `pqkc7sp3` / `conri3cb` / `ews7lbi8` (to 60), `o640jkhb` / `ivu8b09r` / `az2s5lpc` (60–120) |
| pre-registration | `reports/2026-09-16-curriculum-C2-preregistration.md` at `1b993eb` (2026-09-16 22:18); amendment 1 at `edfe44d` (the control's 60-epoch read, the extension and the 245,760 budget, 22:50, before any per-model number); amendment 2 at `236c69a` (the control's 120-epoch read, 23:10, before any per-model number) |

## The read

| row | success | turns (phase-clock, 16 max) | vs bar, paired | held of 3 | on obj | alive | coherent | in-run: rounds to rolling 50 / 80 / 95%; last 40 |
|---|---|---|---|---|---|---|---|---|
| `squad_march_take` (unarmed) | 0.900 | 9.60 | — | 2.92 | 0.909 | 0.842 | 0.932 | — |
| **s1 at 245,760** | **0.890** | 10.21 | +0.61 ± 0.31 | 2.87 | 0.782 | 0.826 | 0.582 | 2.6k / 2.6k / 81k; 88.2 |
| **s2 at 245,760** | **0.920** | 9.84 | +0.24 ± 0.30 | 2.88 | 0.873 | 0.842 | 0.563 | 2.6k / 2.6k / 8.2k; 93.9 |
| **s3 at 245,760** | **0.950** | 9.79 | +0.19 ± 0.29 | 3.00 | 0.807 | 0.856 | 0.662 | 2.6k / 2.6k / 8.2k; 91.8 |
| s1 / s2 / s3 at 122,880 (readout) | 0.810 / 0.960 / 0.900 | 11.15 / 9.66 / 9.96 | +1.55 / +0.06 / +0.36 | 2.88 / 2.99 / 2.91 | 0.78 / 0.87 / 0.76 | — | 0.55 / 0.62 / 0.58 | — |
| control at 60 epochs | 0.710 / 0.690 / 0.760 | 13.8–14.5 | +4.2 to +4.9 | 2.73–2.76 | 0.93–0.97 | — | 0.79–0.81 | 80% at epoch never / never / 33 |
| **control at 120 epochs** | **0.790 / 0.690 / 0.780** | 14.39 / 14.81 / 14.18 | +4.79 / +5.21 / +4.58 | 2.84 / 2.83 / 2.80 | 0.96 / 0.97 / 0.96 | — | 0.74 / 0.80 / 0.80 | 80% at epoch 68 / 96 / 64; 90% never; last ten 0.73–0.90 |

The drift clause (no rolling dip under 0.75 after the first rolling
pass): seed 1 once, at 123,392 rounds (raw 63 at 121,344, in the first
leg; rolling 73.2; back at 90 by 124,416); seeds 2 and 3 never. Raw
in-run dips under 80 after the first rolling 95%: 22 / 12 / 15 of 322 /
465 / 465 evaluations. ⚠ The in-run rows after the resume at 122,880
are **n=20**, not 30: a resume drops `--n-eval-episodes` with the
cadences (#346, noted there); the n=100 reads are unaffected.

**Greedy against sampled** (900000+, n=100, paired): vp +0.9 ± 1.0 / +0.7 ± 0.8 / +2.0 ± 0.8; `held` 2.84 / 2.97 / 3.08
greedy against 2.76 / 2.98 / 2.96 sampled; `alive` 0.82–0.86 either
way; coherency 0.63 / 0.56 / 0.65 greedy against **0.38 / 0.40 / 0.42**
sampled; 80–86 decisions per episode. The sampled policy holds the
points and keeps its bodies alive as well as the greedy one and walks
the squads apart, as on every rung.

**Health panel, last quarter** (~480 updates per seed): explained variance 0.54 / 0.52 / 0.50, clip fraction 0.24 / 0.23 /
0.22, ratio p99 1.83 / 1.81 / 1.73, displacement entropy 0.99 / 0.98 /
1.19, declaration entropy 0.01–0.02, advantage std 0.41–0.48. The
regime's normal range on every line (C1: 0.46–0.52, 0.24–0.27,
1.88–1.91).

**The by-turn census** (n=20 on 700000+, bodies on objectives of 12
and points held of 3 at the end; which points are empty at the end, by
index 0–3 with index 2 the enemy's point at (35, 27)):

| policy | success (n=20) | turns | on objectives at the end | held at the end | episodes with each point empty | max stack |
|---|---|---|---|---|---|---|
| `squad_march_take` | 0.80 | 10.40 | 10.97 | 2.90 | 0 / 0 / **0.20** / 0.10 | 2.9 |
| s1 | 0.90 | 10.10 | 9.19 | 2.85 | 0.05 / 0.05 / 0.10 / 0.05 | 2.9 |
| s2 | 1.00 | 9.50 | 10.27 | 3.00 | 0 / 0 / 0 / 0 | 3.2 |
| s3 | 0.95 | 9.55 | 9.76 | 3.10 | 0 / 0 / 0.05 / 0 | 3.1 |

**The script's misses are at the enemy's point; the arm's are spread.**
At n=20 the script leaves the enemy's disc empty in a fifth of episodes
— the bound squad, under fire, is the one that fails to get on — where
seed 1 misses each point once in twenty and seeds 2 and 3 miss the
enemy's once or never. The arm ends with 9.2–10.3 bodies on points of
about ten alive (`alive` 0.83–0.86 of twelve): nearly every survivor is
standing on a point.

## What it says

- **FAIL on the letter by one hundredth on one seed, and the rung's
  question is answered the other way.** The clause is 0.90 on all three;
  seed 1 reads 0.890, one SE under, at the end of the budget. Seeds 2 and
  3 clear it. The record reads the letter as written (A1, A3, A4 and C1
  were all called FAIL at a cap or a plateau within an SE or two), and
  the same record says what the arm did here: it reached the mark on two
  seeds and sat a hundredth under on the third, at the script's speed
  and the script's survival, while the whole-army control on the same
  scenario and the same rounds failed on every seed by 0.11–0.21.
- **This is the first rung where the per-model arm is ahead of the
  control at equal rounds.** On A0–A4 and C1 the control passed and the
  arm caught up; on A5 the arm was worse. Here the control arrives in
  round seven of eight (turns 14.2–14.8 of 16) and misses a point in a
  quarter of episodes, with its bodies on the points it reaches
  (`on_obj` 0.96) — it is slow, not lost. The arm warm-started from C1
  was at 80% in-run by 2,560 rounds on every seed and at 95% by 8,192
  on two. What the C1 warm start carries is the approach; what C2 adds
  (a squad under fire on the way to one of the four points) costs the
  arm nothing the script does not pay: `alive` 0.826 / 0.842 / 0.856
  against the bar's 0.842.
- **The enemy's fire is not where the arm loses; the residual is C1's.**
  The census puts the script's own misses at the enemy's disc (a fifth
  of episodes at n=20) and the arm's misses across all four points at
  one in twenty each on seed 1 — the A3 allocation residual that C1
  inherited (0.96–0.97 at A3x's best, 0.93 on C1's seed 2), not the
  guns. Seed 1's dip at 121k rounds (raw 63%) is the one drift event on
  the rung; it recovered within 3k rounds and the panel is green.
- **The control's failure is speed under fire, and it is new.** C1's
  control passed 3 of 3 at 120 epochs two rounds behind the script
  (6.8–7.1 against 4.98); C2's is five rounds behind (14.2–14.8 phase
  clock, i.e. about seven rounds, against the script's 4.8) and never
  reaches a rolling 90% in-run. The same trainer, the same scenario but
  for three rifles, and it went from a round slower than the script to
  arriving with the game nearly over. Nothing in this rung's reward pays
  for a body, so the control is not hiding; it is walking around.
  Whether that is the whole-army policy's known stacking (§ Holding
  pays) meeting an occupied point under fire is a census for the control
  that this report did not run.
- **Coherency 0.56–0.66 greedy, 0.38–0.42 sampled** — unpaid, as on
  every rung; the enemy's fire does not change it.

## Standing consequence for the ladder (#317)

The rung was redesigned before launch because the phase facade **ends
the battle when the opponent is wiped** (#317), against the rules the
per-model facade follows; with both sides armed our twelve rifles wipe
the three blockers in two shooting phases and the scripted bar reads
0.380 on the phase facade against 1.000 on the per-model one, same
seeds (`just measure-bridge`: BRIDGE DIVERGES). So this rung arms their
side only, our guns are C3's axis, and **no whole-army control can be
read beside a per-model arm on any rung where the opponent can be
wiped** until #317 is fixed. Written into `CLAUDE.md` with this report.

## What was not done

- No recording of the per-model arm's failing episodes beyond the
  census.
- No arm with both sides armed (the bridge diverges; see above).
- The control's `alive` was not measured separately; its `on_obj`
  0.96–0.97 with `held` 2.80–2.84 says its bodies stand on the points
  they reach.
