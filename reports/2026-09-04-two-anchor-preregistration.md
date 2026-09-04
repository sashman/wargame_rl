# Pre-registration — ARM 3: a two-anchor pool, aimed at the one cell that ties

Written 2026-09-04 after the six-seed verdict and **before this arm is launched**.

## What the verdict says the problem is

The KL-anchored arm at six seeds: `vs_take` **WON**, `vs_deny` **WON**, refereed
**ahead**, `vs_shoot` **tie at −0.15 SE**. The goal is conjunctive, so
`vs_shoot` alone blocks it.

**The pool never contained a shooting opponent.** Its floor was
`squad_march_take_charge` plus the learner's own snapshots, and the learner is
then scored against `squad_march_shoot`. The one matchup it never practised is
the one matchup it cannot win. That is a hypothesis, not a demonstration — the
arm is worse than its own CLONE on that cell (+56.03 v +58.45), which is equally
consistent with residual drift.

## The change

`SnapshotPool` takes a **list** of anchors, all never evicted;
`--pool-anchor` takes a comma-separated floor. Default unchanged
(`["squad_march_take"]`), so every existing run and recipe is unaffected.

**Arm 3** = arm 1 with `--pool-anchor squad_march_take_charge,squad_march_shoot`.
Everything else identical: same clones, same `--kl-ref-target 0.03`, 300 epochs,
same seeds, same scoring.

## Bounds, fixed now

Primary readout is `vs_shoot`, because that is what the change is for.
Comparator: **arm 1's own six-seed numbers**, and the bar.

- **SCREEN PASS** (worth taking to six seeds): `vs_shoot` ≥ **+62.0** mean over
  three seeds AND no other cell falls more than 5 vp below arm 1's mean.
- **SCREEN FAIL**: `vs_shoot` ≤ arm 1's +56.03, or any other cell collapses.
- **INDETERMINATE** between — report as such.

⚠ **Power, checked before writing the bound**: arm 1's `vs_shoot` per-seed sd is
**9.1**, so three seeds give SE ≈ 5.3. A +6 improvement is ~1.1 SE and is
**not** resolvable at three seeds. This screen therefore CANNOT establish the
effect — it can only decide whether six seeds are worth spending. Saying so in
advance, because this project has twice read an underpowered screen as a result.

⚠ **To WIN `vs_shoot` the six-seed mean must exceed +64.0** (bar +56.6 plus
2 SE at the observed spread). Arm 1 reached +56.03 and the clone +58.45, so this
needs a real **+6 to +8**, which no intervention measured today has produced on
that cell.

## Committed in advance
- Adding a second anchor **halves the share of pool draws that are the charging
  script**, so a `vs_take`/`vs_deny` regression is a predicted risk, not a
  surprise. That is why the bound has a no-collapse clause on the other cells.
- If `vs_shoot` rises and `vs_take`/`vs_deny` fall, the honest reading is that
  the pool trades matchups, not that the arm improved.
