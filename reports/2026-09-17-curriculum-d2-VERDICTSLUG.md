# Curriculum rung D2: PPO from the escort clone — VERDICTTITLE

**Verdict first.** Four arms on the rung that asks whether the per-model
PPO improves a policy that starts with the plan (#340; #376, #379, #378,
D2DISSUE), all from D1b's clone (success 0.960, kill before arrival
1.00 on C3b's final-board scenario), three seeds each, 122,880 rounds,
read greedy at n=180 on 700000+:

| arm | change from D2 | verdict | success at the end | ordering | vp paired against the clone |
|---|---|---|---|---|---|
| **D2** | — (cold critic, no anchor) | D2VERDICT | D2SUCC | D2ORD | D2VP |
| **D2b** | the clone's critic fitted first (policy bit-identical) | D2BVERDICT | D2BSUCC | D2BORD | D2BVP |
| **D2c** | the KL anchor to the clone (coef 10, target 0.03), cold critic | D2CVERDICT | D2CSUCC | D2CORD | D2CVP |
| **D2d** | both | D2DVERDICT | D2DSUCC | D2DORD | D2DVP |

VERDICTCLAUSE

## Provenance

| field | value |
|---|---|
| date | 2026-09-17: D2 launched 16:30, D2b and D2c 16:56, D2d D2DLAUNCH; read D2READ |
| GPU / no-GPU | GPU (RTX 4090), up to nine trainers at once |
| seeds | 1 / 2 / 3, every arm from the same clone (`escort-c3b-1200-s0.pt`, or its critic-fitted twin for D2b and D2d) — three seeds off one warm start are not three samples of a policy, so every read is against the start itself, paired |
| n | 180 at seed base 700000 (final, the ordering census); 30 at 500000 (in-run, every 512 rounds); 100 at 900000 (greedy against sampled); 20 at 700000 (by-turn census); 300 games at 800000+ for the drift-from-teacher match |
| config | `configs/experiments/curriculum/c3b.yaml` — success on the final board after twelve rounds; unrefereed by design |
| decode | none on the per-model facade |
| paired | per episode against the clone and the escort on identical seeds; per seed across arms on the same layouts |
| comparator | the clone at n=180: CLONEROW. `scripted_escort` at n=180: ESCORTROW. Reward from a start without the plan (C3b): 0.20 / 0.13 / 0.23 |
| opponent | `scripted_baseline` wrapping `hold_and_shoot`: three models, one unit, four wounds, eight attacks at range 12 |
| budget | 122,880 rounds at 128 per update (`--rollout-rounds 32 --num-rollout-envs 4`), `--ent-coef 0.003`; no extension |
| code revision | D2 on `fde8088` (PR #374's tree); D2b and D2c on `8d2e9ad` (PR #377); D2d on D2DREV |
| checkpoint | `last.pt` at 122,880; `checkpoints/per_model/per-model-c3b-*-s{1,2,3}d2{,b,c,d}` |
| coherency | COHERENCY |
| Wandb | `curriculum-d2`: D2 `udqm4uzc` / `i36lut0r` / `gozwwmae`; D2b `5998xmnz` / `t8xm3yf2` / `j1zn8st7`; D2c `4n9vfyk1` / `le7mw85g` / `et51yp6f`; D2d D2DIDS |
| pre-registrations | D2 `reports/2026-09-17-curriculum-D2-preregistration.md` at `fde8088` (16:25); D2b and D2c at `8d2e9ad` (16:55); D2d at D2DREV — each before its own numbers |

## The read

READTABLE

**The ordering clause and the plan by phase** (n=180 on 700000+):

ORDERTABLE

**In-run** (rolling five of n=30 on 500000+; the first point is the clone's own score):

CURVES

**Drift from the teacher** (the final policy's held-out match against the escort's demonstrations, beside the clone's 0.72 on the displacement head):

DRIFT

**Greedy against sampled**: EVALMODE

**Health panel, last quarter**: PANEL

CENSUS

## What it says

WHATITSAYS

## What was not done

- No longer budget: every arm that held was flat by 20k rounds, and
  every arm that collapsed was gone by 5k.
- No sweep of the anchor's coefficient or target; the smoke's setting
  was the one run.
