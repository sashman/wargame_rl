# Curriculum rung D2: PPO from the escort clone — reward destroys a copied plan unless the weights are anchored to it, the critic is not the lever, and under the anchor reward improves nothing

**Verdict first.** Four arms on the rung that asks whether the per-model
PPO improves a policy that starts with the plan (#340; #376, #379, #378,
#380), all from D1b's clone (success 0.960, kill before arrival
1.00 on C3b's final-board scenario), three seeds each, 122,880 rounds,
read greedy at n=180 on 700000+:

| arm | change from D2 | verdict | success at the end | ordering | vp paired against the clone |
|---|---|---|---|---|---|
| **D2** | — (cold critic, no anchor) | **DESTROYS** | 0.028 / 0.183 / 0.028 | 0.88 / 0.89 / 0.89 | +10.6 / +16.5 / +21.5 against +59.6 |
| **D2b** | the clone's critic fitted first (policy bit-identical) | **DESTROYS** | 0.056 / 0.006 / 0.028 | 0.97 / 0.97 / 0.87 | +43.1 / +33.8 / +28.7 against +59.6 |
| **D2c** | the KL anchor to the clone (coef 10, target 0.03), cold critic | **HOLDS** | 0.944 / 0.956 / 0.956 | 1.00 / 1.00 / 1.00 | +0.7 ± 0.8 (t 0.86) / −0.6 ± 0.9 (t −0.68) / −1.4 ± 1.0 (t −1.49) |
| **D2d** | both | **HOLDS on one seed, under the HOLDS bound by 0.03 on two; DESTROYS on none** | 0.939 / 0.906 / 0.911 | 1.00 / 1.00 / 1.00 | −0.1 ± 0.8 (t −0.10) / −1.9 ± 1.0 (t −1.90) / −1.3 ± 1.0 (t −1.23) |

**The pre-registered guesses were right on every arm:** D2 DESTROYS by
the drift prediction (under 0.90 within 20,480 rounds on every seed and
never back), D2b DESTROYS the same way, D2c HOLDS 3/3 and IMPROVES on
none, D2d HOLDS on one seed and sits 0.03 under the HOLDS bound on two
(0.906 / 0.911 against 0.94, binomial SE 0.02 at n=180), DESTROYS on none,
IMPROVES on none. The whole-army record's diagnosis — the cold critic —
is not the mechanism on this trainer; the anchor is the lever.

## Provenance

| field | value |
|---|---|
| date | 2026-09-17: D2 launched 16:30, D2b and D2c 16:56, D2d 20:16 (when D2's three exited); read D2 20:19, D2b 21:04, D2c 21:20, D2d 22:37–22:57 |
| GPU / no-GPU | GPU (RTX 4090), up to nine trainers at once |
| seeds | 1 / 2 / 3, every arm from the same clone (`escort-c3b-1200-s0.pt`, or its critic-fitted twin for D2b and D2d) — three seeds off one warm start are not three samples of a policy, so every read is against the start itself, paired |
| n | 180 at seed base 700000 (final, the ordering census); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census); 300 games at 800000+ for the drift-from-teacher match |
| config | `configs/experiments/curriculum/c3b.yaml` — success on the final board after twelve rounds; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against the clone and the escort on identical seeds; per seed across arms on the same layouts |
| comparator | the clone at n=180: success 0.950, held 3.92, `alive` 0.958, on_obj 0.963, vp +59.6 ± 1.7, coherent 0.747, kill before arrival 1.00, blockers wiped 0.98. `scripted_escort` at n=180: success 0.978, held 3.97, on_obj 0.993, vp +62.6 ± 1.6, coherent 0.839. Reward from a start without the plan (C3b): 0.20 / 0.13 / 0.23 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, four wounds, eight attacks at range 12 |
| budget | 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`; no extension |
| code revision | D2 on `fde8088` (PR #374's tree); D2b and D2c on `8d2e9ad` (PR #377); D2d on `4e20344` (PR #377) |
| checkpoint | `last.pt` at 122,880; `checkpoints/per_model/per-model-c3b-*-s{1,2,3}d2{,b,c,d}` |
| coherency | greedy at 700000+: D2 0.41 / 0.40 / 0.42, D2b 0.49 / 0.58 / 0.38, D2c 0.78 / 0.78 / 0.75, D2d 0.77 / 0.75 / 0.76 (the clone 0.75); D2c against sampled 0.75–0.77 at 900000+ |
| Wandb | `curriculum-d2`: D2 `udqm4uzc` / `i36lut0r` / `gozwwmae`; D2b `5998xmnz` / `t8xm3yf2` / `j1zn8st7`; D2c `4n9vfyk1` / `le7mw85g` / `et51yp6f`; D2d `ziia57hp` / `jvcqlj5k` / `8qo65zat` |
| pre-registrations | D2 `reports/2026-09-17-curriculum-D2-preregistration.md` at `fde8088` (16:25); D2b and D2c at `8d2e9ad` (16:55); D2d at `4e20344` (PR #377) — each before its own numbers |

## The read

| row | success | held of 4 | on obj | alive | vp (± SE) | coherent | stat | vp paired against the clone |
|---|---|---|---|---|---|---|---|---|
| `scripted_escort` | 0.978 | 3.97 | 0.993 | 0.963 | +62.6 ± 1.6 | 0.839 | — | — |
| the clone (the start) | 0.950 | 3.92 | 0.963 | 0.958 | +59.6 ± 1.7 | 0.747 | 0.36 | — |
| **D2 s1 / s2 / s3** (cold critic, no anchor) | **0.028 / 0.183 / 0.028** | 1.44 / 1.88 / 1.81 | 0.43 / 0.48 / 0.52 | 0.55 / 0.56 / 0.60 | +10.6 / +16.5 / +21.5 | 0.41 / 0.40 / 0.42 | 0.04 / 0.07 / 0.26 | (not run: the success clause decides) |
| **D2b s1 / s2 / s3** (critic fitted) | **0.056 / 0.006 / 0.028** | 2.33 / 1.91 / 1.71 | 0.60 / 0.54 / 0.49 | 0.74 / 0.70 / 0.53 | +43.1 / +33.8 / +28.7 | 0.49 / 0.58 / 0.38 | 0.31 / 0.30 / 0.21 | (not run) |
| **D2c s1 / s2 / s3** (anchor) | **0.944 / 0.956 / 0.956** | 3.92 / 3.92 / 3.93 | 0.97 / 0.97 / 0.96 | 0.95 / 0.96 / 0.96 | +60.3 / +59.1 / +58.2 | 0.78 / 0.78 / 0.75 | 0.37 ×3 | +0.7 ± 0.8 (t 0.86, 47 of 88 differing) / −0.6 ± 0.9 (t −0.68, 40 of 92) / −1.4 ± 1.0 (t −1.49, 37 of 91); 88–92 of 180 episodes identical |
| **D2d s1 / s2 / s3** (both) | **0.939 / 0.906 / 0.911** | 3.92 / 3.84 / 3.87 | 0.96 / 0.95 / 0.96 | 0.96 / 0.94 / 0.94 | +59.6 / +57.7 / +58.4 | 0.77 / 0.75 / 0.76 | 0.36–0.37 | −0.1 ± 0.8 (t −0.10, 45 of 89 differing) / −1.9 ± 1.0 (t −1.90, 37 of 85) / −1.3 ± 1.0 (t −1.23, 43 of 91); 89–95 of 180 identical |

Every game runs the full 24 turns, so the vp column against the bar in
`measure-rung` is void (identical lengths); the paired vp here is
`just measure-paired` against the clone on the same 180 seeds.

**The ordering clause and the plan by phase** (n=180 on 700000+):

| policy | success (n=180) | first kill | blockers wiped (share) | first unarmed body on their point (share) | **kill precedes arrival** | alive | armed squad dead of 3 |
|---|---|---|---|---|---|---|---|
| the clone | 0.950 | 6.1 | 11.3 (0.98) | 13.3 (0.90) | 1.00 | 0.958 | 0.07 |
| D2 s1 / s2 / s3 | 0.03 / 0.18 / 0.03 | 7.5 / 10.7 / 9.4 | 0.39 / 0.74 / 0.60 | 0.46 / 0.44 / 0.29 | 0.88 / 0.89 / 0.89 | 0.55 / 0.56 / 0.60 | 1.37 / 0.87 / 0.55 |
| D2b s1 / s2 / s3 | 0.06 / 0.01 / 0.03 | 7.8 / 7.0 / 8.0 | 0.93 / 0.82 / 0.67 | 0.37 / 0.69 / 0.49 | 0.97 / 0.97 / 0.87 | 0.74 / 0.70 / 0.53 | 0.43 / 0.81 / 1.23 |
| D2c s1 / s2 / s3 | 0.94 / 0.96 / 0.96 | 6.1 / 6.0 / 6.1 | 0.99 / 0.99 / 0.99 | 0.82 / 0.87 / 0.89 | **1.00 / 1.00 / 1.00** | 0.95 / 0.96 / 0.96 | 0.08 / 0.07 / 0.05 |
| D2d s1 / s2 / s3 | 0.94 / 0.91 / 0.91 | 6.1 / 6.0 / 6.1 | 0.99 / 0.98 / 0.99 | 0.87 / 0.86 / 0.83 | **1.00 / 1.00 / 1.00** | 0.96 / 0.94 / 0.94 | 0.04 / 0.08 / 0.08 |

The destroyed arms do not lose the order first: D2b still fires first
in 87–97% of episodes and wipes the blockers in two thirds to nine
tenths. What they lose is the walk after it — bodies on points, and the
points held to the end (D2's census: 1.3–2.4 held at turn 24).

**In-run** (rolling five of n=30 on 500000+; the first point is the clone's own score):

| arm | first evaluation (512 rounds) | 2,560 | 5k | 20k | 60k | last forty, mean |
|---|---|---|---|---|---|---|
| D2 | 90 / 80 / 73 | 6 / 0 / 46 | 8 / 4 / 2 | 8 / 4 / 2 | 14 / 7 / 17 | under 20 on every seed |
| D2b | ~95 ×3 | 27 / 13 / 7 (6k) | — | 4 / 9 / 13 | 4 / 9 / 13 | 6–20 |
| D2c | ~95 ×3 | 98 / 99 / 99 (5k) | 98 / 99 / 99 | 97 / 97 / 96 | 99 / 100 / 99 | 97–100 |
| D2d | ~95 ×3 | 98 / 99 / 99 (8k) | — | 99 / 97 / 97 | 96 / 99 / 98 | 98–99 |

Both collapses are complete inside the first five thousand rounds, and
neither recovers by 122,880. Both holds are flat from the first
evaluation to the last.

**Drift from the teacher** (the final policy's held-out match against the escort's demonstrations, beside the clone's 0.72 on the displacement head):

| arm | selector | declaration | displacement | unit-pointer | joint |
|---|---|---|---|---|---|
| the clone, on the same 300 games | 0.83 | 0.99 | **0.94** | 1.00 | 0.79 |
| D2 s1 / s2 / s3 | 0.63 / 0.62 / 0.62 | 0.77 / 0.81 / 0.87 | **0.11 / 0.06 / 0.05** | 0.97 / 1.00 / 1.00 | 0.18 / 0.15 / 0.15 |
| D2b s1 / s2 / s3 | 0.62 / 0.64 / 0.62 | 0.84 / 0.86 / 0.85 | **0.08 / 0.06 / 0.06** | 1.00 ×3 | 0.16 / 0.17 / 0.17 |
| D2c s1 / s2 / s3 | 0.77 / 0.77 / 0.77 | 0.97 / 0.98 / 0.98 | **0.88 / 0.87 / 0.86** | 1.00 ×3 | 0.70 / 0.70 / 0.69 |
| D2d s1 / s2 / s3 | 0.77 / 0.77 / 0.77 | 0.98 / 0.97 / 0.98 | **0.88 / 0.88 / 0.87** | 1.00 ×3 | 0.70 / 0.70 / 0.70 |

Scored on the first 300 of the escort's demonstration games — games the
1,200-game clone was fitted on (its own match there is 0.94 on the
displacement head), so this is drift from the START, not held-out
fidelity. The destroyed arms' displacement heads are unrecognisable
(0.05–0.11); the anchored arms moved from 0.94 to 0.87 and stayed there.

**Greedy against sampled**: D2c: vp −1.6 ± 1.3 / +0.7 ± 1.0 / +0.3 ± 0.9, `held` 3.93–3.98 either way,
coherency 0.77–0.81 greedy against 0.75–0.77 sampled — the anchored policy
is sharp (displacement entropy 0.30). D2d: −1.4 ± 1.5 / +0.8 ± 1.4 / +0.7 ±
0.8, `held` 3.8–3.9 either way, D2c's twin. D2: +1.6 / +4.3 / −0.3, `held`
1.3–2.2. D2b: **+12.8 ± 4.0 / +2.8 ± 3.1 / +12.6 ± 4.8** — a diffuse policy
whose argmax plays better than its samples, the do-nothing fingerprint's
neighbour (stationary share 0.21–0.32).

**Health panel, last quarter**: D2 — explained variance 0.01 / (n/a) / 0.23, clip 0.34 / 0.37 / 0.41,
ratio p99 1.9–2.4, displacement entropy 1.5–2.3; D2b — explained variance
0.10 / 0.07 / 0.01, clip 0.33–0.37, entropy 1.8–2.2; D2c — explained
variance **0.66 / 0.68 / 0.67**, clip 0.28–0.29, ratio p99 **2.9**,
displacement entropy **0.30**; D2d — 0.69 / 0.68 / 0.67, clip 0.28–0.29,
p99 2.9, entropy 0.30, D2c's panel to the second decimal from a critic
that started at 0.74 instead of cold. Under the anchor the cold critic
learned (0.66 from initialisation) and the policy stayed sharp; the
ratio's 99th percentile at 2.9 is the anchor's gradient distorting the
importance ratio, not a trust region failing — read it beside `kl_ref`.

**The by-turn census** (n=20 on 700000+, points held of 4 and bodies on
points of 12 at turn 9 → 12 → 16 → 24):

| policy | success (n=20) | held / on points, 9 → 12 → 16 → 24 | empty points at the end (index; 2 the blockers') |
|---|---|---|---|
| `scripted_escort` | 1.00 | 0.7/1.8 → 1.3/3.5 → 3.3/9.5 → 4.0/12.0 | 0 / 0 / 0 / 0 |
| the clone | 0.90 | 0.6/1.6 → 1.3/3.4 → 3.3/9.1 → 3.8/11.4 | 0 / 0.05 / 0.05 / 0.10 |
| D2 s1 / s2 / s3 | 0.00 / 0.15 / 0.00 | 1.2/3.8 → 1.4/3.7 → 1.4/3.8 → 1.5/4.5 · 0.4/0.9 → 0.9/2.6 → 2.1/5.4 → 2.4/7.0 · 0.9/2.2 → 1.2/4.1 → 1.3/5.2 → 1.3/5.5 | 0.3 / 0.55 / 0.8 / 0.9 · 0.3 / 0.3 / 0.45 / 0.55 · 0.3 / 0.8 / 0.9 / 0.7 |
| D2b s1 / s2 / s3 | 0.00 / 0.05 / 0.10 | 1.0/2.7 → 1.8/4.8 → 2.2/7.1 → 2.1/7.4 · 0.6/0.8 → 1.8/5.1 → 2.0/6.5 → 1.9/6.1 · 1.3/3.2 → 2.1/5.6 → 2.0/5.9 → 1.9/6.2 | 0.6 / 0.2 / 0.6 / 0.55 · 0.9 / 0.65 / 0.4 / 0.15 · 0.4 / 0.35 / 0.7 / 0.65 |
| D2c s1 / s2 / s3 | 0.90 / 0.95 / 0.95 | 0.8/2.0 → 1.4/3.8 → 3.2/8.6 → 3.8/11.2 · 0.7/1.9 → 1.2/3.3 → 3.3/9.2 → 3.9/11.5 · 0.7/1.7 → 1.3/3.4 → 3.0/8.5 → 3.9/11.3 | 0 / 0.05 / 0.05 / 0.10 · 0 / 0.05 / 0.05 / 0.05 · 0 / 0.05 / 0.05 / 0.05 |
| D2d s1 / s2 / s3 | 0.95 / 0.90 / 0.90 | 0.7/2.0 → 1.4/3.6 → 3.5/9.5 → 3.9/11.2 · 0.7/1.9 → 1.3/3.5 → 3.3/8.8 → 3.9/11.4 · 0.7/1.8 → 1.4/3.8 → 3.3/8.7 → 3.7/11.2 | 0 / 0.05 / 0.05 / 0.05 · 0 / 0 / 0.10 / 0.05 · 0 / 0.10 / 0.10 / 0.10 |

The anchored arms' boards are the clone's to the first decimal at every
turn. The destroyed arms walk bodies in early (1–4 on points at turn 9
against the clone's 1.6, waiting) and end with half the board empty.

## What it says

- **Per-model PPO from a clone that holds the plan destroys it in two
  thousand rounds, and a fitted critic does not stop that.** D2 and D2b
  read 0.03–0.18 at the end from a 0.95 start, both gone by 5k rounds,
  both flat after. The whole-army record's diagnosis — a cold critic's
  noisy first advantages — was the obvious mechanism and it is not the
  one here: with the value head fitted to the teacher's returns
  (explained variance 0.74) the collapse is the same shape at the same
  speed. What the two collapses share is the update itself: at 128 rounds
  per update, five epochs over the rollout with `eps_clip` 0.2 move a
  sharp policy far from a start the critic cannot yet value, and once
  the unarmed squads walk in early the plan's payoff is gone from the
  data. The destroyed policies keep the order (they still fire first in
  87–97% of episodes) and lose the walk.
- **The KL anchor holds the plan, and reward under it changes nothing
  measurable.** D2c: 0.944 / 0.956 / 0.956 against the clone's 0.950,
  paired vp −1.4 to +0.7 with half the episodes identical move for move,
  the board the clone's to the first decimal at every turn, the policy
  0.87 from the teacher on the displacement head where the clone is
  0.94. HOLDS 3/3, IMPROVES on none — the pre-registered guess. The
  anchor's coefficient climbed to its cap on every seed (the measured
  drift sat at ~0.09 nats per decision per update whatever the
  coefficient, Adam normalising the penalty's gradient), and the critic
  it protected learned to 0.66 explained variance from a cold start.
  This is #332's answer: build it, and the per-model trainer can carry a
  clone the way the whole-army one carried the KL-anchored clone.
- **Both changes together are the anchor alone, a hair worse.** D2d
  reads 0.939 / 0.906 / 0.911 with the order intact, the board the
  clone's, the panel D2c's to the second decimal (explained variance
  0.68 either way — the cold critic under the anchor reached what the
  fitted one started at), paired vp −0.1 / −1.9 / −1.3 against the
  clone. Two seeds sit 0.03 under the HOLDS bound at a binomial SE of
  0.02, none under 0.90, none improved. The fitted critic under the
  anchor is worth nothing; whether it costs the 0.03 is inside the
  noise. The 2×2 is clean: the anchor's row holds, the critic's column
  does not matter.
- **Where the ladder stands.** The D rungs asked whether the set
  network can hold an ordered plan (yes, from imitation: 0.96 at 1,200
  games) and whether PPO can improve one it starts with. The answer at
  this budget is that PPO from the plan keeps it only under an anchor,
  and under that anchor it does not improve it. The improvement question
  is now a coefficient sweep — an anchor loose enough to let reward move
  the policy and tight enough to keep the plan — and a budget question
  the record already names for the whole-army anchor (`--kl-ref-target`
  0.03 held there too, and won two of four cells). **Rule: warm-start the
  per-model PPO from a clone only under a KL anchor to it; the critic
  is not the lever; and read the 2×2, not the diagnosis on file, before
  naming a mechanism.**

## What was not done

- No longer budget: every arm that held was flat by 20k rounds, and
  every arm that collapsed was gone by 5k.
- No sweep of the anchor's coefficient or target; the smoke's setting
  was the one run.
