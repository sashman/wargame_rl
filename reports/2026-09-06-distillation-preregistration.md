# Pre-registration — distilling the DECODED policy, the route the record says is supported

Written 2026-09-06 **before the arm is launched**, after the decode stack was
measured to its ceiling
(`reports/2026-09-06-three-decode-knobs-and-none-of-them-pays.md`).

## Why this, and why now

The ladder needs roughly **+10 vp on `refereed` and +13 on `vs_shoot`**. Three
decode knobs delivered +5.2 on one cell between them, so the remaining route is
a better *policy*.

The record already names the route. `reallocation_decode.py`:

> **PLAY-TIME ONLY, like every other decode here.** Folding a decode into PPO
> means the executed action is not the sampled one, which measured **−51.8 vp**
> from scratch. **The supported route into the weights is distillation** — clone
> the decoded policy, then train from that basin.

The decode stack is worth **~+81 vp** over the raw network (`refereed`: −71.9
undecoded against +9.3 decoded). If any of that can be carried in the weights,
it dwarfs every lever measured in this goal.

## ⚠ It has been tried once, under-powered, and lost

§46: 2 seeds, **120 episodes x 8 epochs**, K=3 teacher with the decode on — the
clone "loses the operator's gain and falls below the plain teacher by 3.9". But
this repo's own stated house fidelity is **1200 demonstrations x 60 epochs**
(`docs/melee-teaching-goal.md` §30). That attempt used **10x fewer
demonstrations and 7.5x fewer epochs**. It is a weak prior, not a settled
result, and it is the reason this is worth one properly-sized run rather than
three.

## ⚠ A defect fixed to make this possible at all

`behaviour_clone.py` built its teacher **without `charge_decode`**, and the
demonstration cache key did not mention it. On a melee config that clones a
teacher which **declares charges it cannot execute** — a factored policy almost
never puts the one legal joint rung in every squadmate's top-K, which is why
`apply_charge_decode` exists. Fixed in this branch; default off, so every
existing clone and every cached collection is unaffected.

## The arm

1. **Distil** arm 4's decoded policy into a fresh network:
   `just behaviour-clone <arm4 s{n} last.ckpt> configs/experiments/25v25_maps_melee_approach.yaml 1200 60 checkpoints/decodeclone-s{n}.ckpt {n} 3 1 1`
   — house fidelity, K=3, reallocation on, charge decode on, demonstrations on
   the clone band (800000+).
2. **Train** from it exactly as arm 4 was, so the only difference is the basin:
   self-play, `--pool-anchor squad_march_take_charge,squad_march_shoot`,
   `--kl-ref-target 0.03 --kl-ref-coef 1.0` (anchored to the *new* clone),
   1000 epochs, `ent_coef` 0.003. **Three seeds.**
3. **Score** at n=180, seeds 700000+, six... **three** seeds, K=3, charge
   decode, reallocation, all four cells, paired per scenario against the same
   n=180 bar.

Control: arm 4 itself, same protocol — already measured.

⚠ **UNPAIRED by construction**: a different initialisation, so no weights are
shared. Seeds still fix the layout and dice streams.

## Bounds, fixed now

- **PASS**: all four cells reach **t > 2** against the n=180 bar. That is the
  goal.
- **PARTIAL**: the clone's own decoded score beats arm 4's on `refereed` **and**
  `vs_shoot` by more than 2 SE, without losing a cell — progress on the two
  cells that block, whether or not they resolve.
- **FAIL**: it does not beat arm 4 on either blocking cell. §46 is then
  confirmed at house fidelity and **distillation is closed**, not re-run bigger.

## Power, stated before the fact

Three seeds, unpaired, per-seed sd ~8 on these cells at n=180 gives SE ≈ 6.5 for
an arm-versus-control difference. Only differences above **~13 vp** are
resolvable. This is sized to detect a *large* effect, which is the only kind
worth having here — the decode gap it is trying to capture is +81.

## Predictions

- **The clone will not carry the joint properties.** The record's standing
  finding is that a per-model fit does not inherit a joint property: a
  98.3%-action-match clone of `squad_march_take` held **0.40** unit coherency
  against its teacher's 0.95. I expect the same failure mode here, softened by
  house fidelity.
- **Most likely outcome: FAIL**, and the value of running it is that it closes
  the last route the record itself nominates, at a known cost, rather than
  leaving it open as a plausible untried idea.
- If it PARTIALs, the follow-up is obvious and cheap: iterate (decode the
  student, distil again), which is the standard policy-improvement loop this
  repo has never run.
