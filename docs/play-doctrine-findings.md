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
- **Open**: *why* neutral ground pays nothing. Standing candidate is the hold-hazard
  result (deaths happen walking *between* points) plus the possibility that a point
  still empty by round 8 is one neither side can cross to safely. Needs the reflex's
  own travel distance and en-route casualties instrumented; neither exists.

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
