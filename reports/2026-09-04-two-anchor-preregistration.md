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

---

# AMENDMENT 2026-09-05 — seeds 4-6 launched, and a correction to my own reading

**What I got wrong.** On seeing the three-seed screen I reported that arm 3
"trades matchups, not an improvement", citing `vs_shoot` +6.77 against `vs_take`
−6.97. Tested rather than eyeballed, only ONE cell moved:

| cell | paired diff v arm 1 | SE | t | signs |
|---|---|---|---|---|
| refereed | −6.23 | 7.58 | −0.82 | 2/3 |
| `vs_take` | −6.97 | 12.52 | −0.56 | 2/3 |
| `vs_deny` | +0.17 | 9.87 | +0.02 | 2/3 |
| **`vs_shoot`** | **+6.77** | 3.90 | **+1.73** | **3/3** |

The losses are **indistinguishable from zero**. I had pre-registered that a
trade was the predicted risk, then read the prediction into point estimates that
do not carry it. ⚠ **Registering a risk in advance makes it easier to see, not
truer** — the test still has to be run.

**The no-collapse clause still fails by the letter**: refereed is 5.22 below arm
1's six-seed mean against a 5.0 allowance. It fails on a difference that is
statistically zero, which is a defect in the clause — an absolute vp bound on a
cell whose three-seed SE is 7.58 cannot separate 5.22 from 0.

**Decision: take arm 3 to six seeds.** `vs_shoot` is the only cell blocking the
goal and this is the only intervention that has moved it. Seeds 4-6 launched on
the same config, clones, anchor list and dose.

**Bounds for the six-seed table, fixed now**, using the same rule as every other
route (WON = gap to bar > 2 SE):
- The claim to be tested is **`vs_shoot` WON**, i.e. six-seed mean above
  `+56.6 + 2 SE`.
- Report all four cells. If refereed or `vs_take` are genuinely below arm 1 at
  six seeds, the trade reading becomes admissible **then**, not now.
- ⚠ Arm 1 and arm 3 share seeds 1-3 clones and differ only in the pool's floor,
  so the per-seed difference is paired and is the estimator to quote.
