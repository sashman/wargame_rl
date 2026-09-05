# Pre-registration — ARM 4: arm 3 extended to 1000 epochs

Written 2026-09-05, **before any 1000-epoch checkpoint exists**.

## Why

Arm 3 (two-anchor pool) at six seeds wins `vs_take`, `vs_deny` and `vs_shoot`
and **ties the refereed mirror at −0.12 SE**. That single cell is the goal.

Pool composition cannot fix it: the mirror is played against
`squad_march_take_charge`, and the arm that anchors on it **alone** (arm 1)
still only reaches +1.45, short of the ~+5 a win needs. So the remaining gap is
not the opponent distribution.

This project's own rule is *"screen at ~300 epochs, quote effect sizes at
1000+"*, and the historical 300→1000 gain is **~+8 vp** — the size required.
The anchored arm is the first lineage here that does **not** degrade with
training (headroom held at +67 to +78 against plain PPO's +40), so it is the
first for which extending is a reasonable bet rather than a known loss.

## The arm

`--resume-ckpt-path` from each arm 3 seed's epoch-300 `last.ckpt` to
`--max-epochs 1000`, everything else identical. **Three seeds** (1–3) as a
screen. ⚠ Resume was broken for anchored runs until today and is now verified by
resuming one to epoch 302; without that fix this arm was impossible.

## Bounds, fixed now

Primary readout: **refereed**, because that is the only cell not won.

- **PASS** (extend to six seeds): refereed six-... three-seed mean ≥ **+6.0**
  AND `vs_shoot` stays above **+60** (it must not give back the cell arm 3 won).
- **FAIL**: refereed ≤ arm 3's −5.95, or `vs_shoot` falls below +56.6.
- **INDETERMINATE** between.

⚠ **Power, stated before the numbers**: arm 3's refereed per-seed sd is ~12.8,
so three seeds give SE ≈ 7.4. A +12 improvement is 1.6 SE. **This screen cannot
establish a win on that cell**; it can only say whether six seeds are worth
spending. Written down because this project has read an underpowered screen as a
result twice.

⚠ **The record's own warning applies with full force**: the advance lever read
"+2.2, free" at 300 epochs and **−16.3 at 1000**, with two of three seeds
flipping sign. **A 1000-epoch result can reverse a 300-epoch one rather than
sharpen it.** If this arm comes back worse, that is a real finding about the
anchor and not a fluke to be re-run.

## Committed in advance
- Scored identically to every other row: n=45, seeds 700000+, K=3 + charge
  decode, all four cells, at a **verified** epoch (checked inside the file, not
  inferred from `last.ckpt` existing — that mistake cost a mislabelled table
  yesterday).
- Decode headroom reported whatever it says. If the cells improve while headroom
  collapses, the anchor stopped being the mechanism and that must be said.
