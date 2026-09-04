# Play-doctrine findings — what this project measured against the doctrine

⚠ **[docs/play-doctrine.md](play-doctrine.md) is an IMMUTABLE REFERENCE and must not
be edited.** It was collected from external sources and its value is that it says
what it said before we tested anything. This file is where our results go: one entry
per doctrine claim we have priced, additive, never rewriting the claim it answers.

Read the doctrine for the claim; read this file for what happened when we tried it.
Where the two disagree, **this file wins on fact and the doctrine keeps the claim**.

Entry format: the doctrine id and its claim in one line, the measurement with its n /
seeds / decode / config, the verdict, and what it licenses or forbids next.

---

## PREMISE ROWS — "There is no melee" / "There is no charge phase and no fight phase"

**Both premises are now FALSE on melee configs, and the closing they assumed away
is worth ~+30 vp.**

These are not numbered claims but *environment facts* the doctrine's table asserts,
and two of its strongest consequences hang off them: *"every measurement of a
movement feature here is provisional"* (because closing is priced only by what it
captures, never by what it threatens) and *"ground is sticky — nothing is levered
off a point except by being shot off it"*.

Measured 2026-09-04 at `d5ec7d4`, n=45, seeds 700000+, no decode, on
`configs/evaluation/25v25_maps_melee_approach_*`. The two scripts differ **only**
in whether `select_charge` returns a charge:

| cell | `squad_march_take` (never charges) | `squad_march_take_charge` | charging is worth |
|---|---|---|---|
| `vs_shoot` | +23.8 | **+56.6** | **+32.8** |
| refereed | −33.6 | **−5.3** | **+28.3** |

- **The premises are lifted where melee is on.** Closing now threatens as well as
  captures, and ground is no longer sticky: a unit can be levered off a point by
  contact.
- ⚠ **They still hold everywhere else.** `melee.enabled` defaults to False, so every
  golden config — and therefore most of the movement record, including the whole
  advance-move line — still sits inside the doctrine's premise. The provisionality
  warning it attaches is **not** retired; it is now scoped.
- ⚠ **This refutes a hypothesis of mine the same day.** I proposed that against a
  *non-charging* opponent (`vs_shoot`) declaring charges wastes the unit's shooting,
  and that this was why no policy beats the bar on that cell. Charging is worth
  **+32.8** there. **REFUTED** — the cell is hard because the bar plays it well
  (+56.6), not because charging costs anything.
- **Licensed next**: pricing any move type whose value is "arrive sooner" on a melee
  config, which the doctrine's own note says was impossible before.
  **Forbidden next**: quoting a pre-melee movement result as though the premise had
  been lifted for it.

## D-08 — "A surplus unit goes to the cheapest unheld point, then the cheapest enemy-held one"

**The claim's ordering is REFUTED for the trained agent; its priority is the reverse.**

Measured 2026-08-31, prompted by the user's read of the recordings (*"abandoned
objectives which can be controlled for +5 vp every turn"*). Six fold verdict
checkpoints, n=45, seeds 700000+, K=3, `configs/evaluation/25v25_maps_melee_fold_refereed.yaml`,
play-time reflex on **frozen weights**, paired against the same checkpoints' plain
rows. Prediction committed before the run (`/tmp/melee_realloc/PREDICTION.md`).

| decode | per-seed gain (s1–s6) | mean | t | positive |
|---|---|---|---|---|
| **contest** — surplus → opponent's WEAKEST-held point | +12.5/+25.2/+12.4/+2.5/−0.1/−2.6 | **+8.32 ± 4.25** | +1.95 | 4/6 |
| **spread** — surplus → nearest EMPTY point | −2.9/+17.3/−8.9/+5.5/+12.5/−13.9 | **+1.60 ± 5.00** | +0.32 | 3/6 |
| spread − contest | −15.4/−7.9/−21.3/+3.0/+12.6/−11.3 | −6.72 ± 5.09 | −1.32 | 2/6 |

- `spread` **fails** the instrument's own pre-registered kill (negative on 3 of 6
  seeds); `contest` clears it. The head-to-head favours contest but is **not**
  statistically settled — the sound statement is "one clears its kill and the other
  does not".
- ⚠ **The doctrine entry's stated reason for the blunt form's failure does not
  survive.** It attributes it to own VP saturating at three objectives. The agent
  holds **~1.5** and is nowhere near the cap, and empty ground still pays ~nothing
  there. Saturation was not the mechanism.
- ⚠ **My own prediction was wrong** (+2 to +6 predicted for `spread`; +1.6 measured,
  failing its kill), and the reasoning behind it — that the doctrine's script-measured
  verdict should not transfer to an unsaturated agent — was wrong with it.
- **Licensed next**: a surplus-to-weakest-enemy-point rule is worth ~+8 vp to this
  agent for zero training, i.e. **44% of the remaining gap to the §38 bar** on that
  cell. **Forbidden next**: re-running the neutral-ground form, on scripts or on an
  agent.
- **CLOSED same day by §40d, and the question was the wrong one.** Four candidate
  explanations were measured and every one refuted: the empty point is *closer*
  (17.6–21.6" against 19.0–21.9"), ~5× *safer* (0.89–1.33 enemies covering the
  destination against 4.41–5.26), the agent is *not* at the VP cap (holds 1.42–1.77,
  at cap on 2.2–16.9% of nominations), and the nomination does *not* churn more
  (same target on 57.8–86.8% of consecutive phases against contest's 63.4–76.8%).
  What is true instead: **neither** reflex raises our own held count (contest
  +0.02 ± 0.09, spread −0.06 ± 0.12), so no redirect measured here buys ground at
  all — the contest gain is denial and attrition. See D-02 below.

Full write-up: [melee-teaching-goal.md](melee-teaching-goal.md) §40c. The reflex's
post-referee value and the fold's zero-allocation result are §40b.

## D-02 — "Take the cap's worth of ground and no more; spend the rest on denial"

**Confirmed, on the trained agent, by the mechanism split of the reallocation reflex.**

Measured 2026-08-31 (details in [melee-teaching-goal.md](melee-teaching-goal.md) §40d).
Sending a surplus squad at the opponent's weakest-held objective is worth **+8.3 ± 4.25**
vp; sending the same squad to the nearest empty objective is worth **+1.6 ± 5.00** and
fails its kill. The paired mechanism split (3 seeds, n=45) shows the gain is **not**
ground taken — our own held moves **+0.02 ± 0.09** — but ground **denied** (**−0.17 ±
0.06** of theirs, 3/3) and army **destroyed** (**−4.3 ± 0.6 pp** of theirs, 3/3).

- The entry's prescription and its *reason* both hold here: spare capacity is worth more
  spent against the opponent's scoring than on adding to our own.
- ⚠ It holds for a reason the entry does not claim: the surplus squad's value at the
  contested point is largely that it **shoots**, not that it contests. Four candidate
  explanations for the empty-ground null (distance, destination danger, own VP cap,
  nomination churn) were each measured and each refuted — see §40d.
- **Licensed next**: narrowing the melee hunt declaration to *surplus units hunting
  objective-holding enemies* — a mask change on machinery that already exists.
