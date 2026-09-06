# Stock-take: the melee-ladder goal, before the re-work

**Provenance.** 2026-09-06 · a retrospective, **not a measurement** · code
revision `0c8b3ef` · every number below is cited to the report that measured it.

The goal: *the agent beats `squad_march_take_charge` on the four-cell melee
ladder and the refereed head-to-head, at K=3, six seeds, pre-registered bounds.*

## 1. The headline: NOT MET, and the honest position is weaker than "close"

At n=180 with the agent and the bar measured on identical scenarios, paired:

| cell | gap to bar | t | seeds ahead |
|---|---|---|---|
| `refereed` | +4.21 | 0.65 | 4/6 |
| `vs_take` | +6.73 | 1.08 | 6/6 |
| `vs_deny` | +7.45 | 1.12 | 6/6 |
| `vs_shoot` | **−0.19** | −0.03 | 2/6 |

The agent is **slightly ahead on three matchups and level on the fourth**, and
none of it resolves. Resolving `vs_take`/`vs_deny` at these gaps needs n≈1450;
`vs_shoot` cannot be won by more measurement at all.

## 2. ⚠ The thing the re-work most needs to know

**Every agent-versus-bar number produced in this goal before 2026-09-06 was
measured with a broken instrument**, and that includes numbers this project
acted on. Two independent errors, both inflating:

- The **comparator was treated as a constant**. It is a fixed policy sampled
  over scenarios and carries the same noise the agent does: the bar's `vs_deny`
  value moved **+11.8 → +36.8** when remeasured at n=180. Every ladder row ever
  published here carried an unpropagated **±12.7**.
- The **across-seed SE (n=6) omits scenario noise**, which is common-mode
  because every seed is scored on the *same* scenarios. It read 2.8–8.1 SE on
  four cells a paired per-scenario estimator puts at t = −0.03 to 1.12.

Per-scenario sd is **81–89**, so SE ≈ 89/√n: **±12.7 at n=45**, ±6.3 at n=180,
±3.2 at n=720. **Every arm difference ever measured on this ladder is inside the
n=45 error bar.**

⚠ **What survives the correction and what does not, by size.** A +81 effect
(the decode stack) survives n=45 comfortably. A +5 to +15 effect does not — and
that is the size of nearly every lever this project has ever tuned.

## 3. What we can still claim

- **The decode stack is worth ~+81 vp.** Same weights on `refereed`: **−71.9**
  with a plain argmax, **+9.3** with joint decoding + charge decode +
  reallocation. Far too large to be noise at any n used here.
- **The reallocation decode is real**: **+9.57 ± 2.55, 6 of 6 seeds**, paired
  per scenario at **n=180**. The only lever measured this session that survived
  its own remeasurement.
- **PPO trains undecoded and is scored decoded, and spends the difference.**
  Decode headroom decays with training and predicts the score, 6/6 seeds. This
  motivated the KL anchor and remains the best structural account of why
  training gains do not show up where they are scored.
- **Self-play teaches charging from scratch.** A cold-started self-play run
  declares 14–16 charges an episode and completes 7.6–9.4 — more than the bar
  (5.8–6.3) and more than the trained agent (5.9–8.3). The behaviour-cloned warm
  start is **not** what supplies the charge.
- **Seed variance was never the binding constraint; episodes were.** Per-seed
  means span ~16 vp where scenarios span ±85. Six seeds was the wrong axis to
  spend on.

## 4. What shipped, and is worth keeping

| | where | state |
|---|---|---|
| **KL anchor to the warm start** (`--kl-ref-coef`, `--kl-ref-target`, adaptive coefficient, resume-safe) | PR #265 | works mechanically; ⚠ its **ladder rows are n=45 and void** |
| **Multi-anchor self-play pool** (`SnapshotPool` takes a sequence) | PR #265 | shipped |
| `just measure-melee-ladder` | PR #265 | shipped |
| **`max_redirects` / `exclude_groups` / `exclude_targets`** on the reallocation decode | PR #279 | **measured-rejected**, kept as control, defaults bit-identical |
| `charge_decode` threaded into `behaviour_clone` | PR #279 | **defect fix** — it was distilling teachers that declare charges they cannot execute |
| `record_gifs` passes `charge_decode`; pool logs its anchor names | stacked on #265 | defect fixes, regression test verified to fail on the old code |

Ten reports, four of them pre-registrations committed before their results
existed. Issues #277 (question), #278, #281 (arms) carry the verdicts.

## 5. What was retracted — five things, three of them my own predictions

1. **"All four cells WON"** at n=45 → no cell resolved at n=180.
2. **"`vs_shoot` is a +15.6 win"** → **−0.19**, dead level.
3. **Iterated reallocation** — I proposed it, wrote it, pre-registered **+3 to
   +8**, and it measured **−1.28** overall and **−4.33 (t=−3.17)** on `vs_deny`.
4. **"A cold-started policy never charges"** — true only of the non-self-play
   control; I generalised from a partial result before the self-play cells
   finished.
5. **`min_stack` 4 → 2** — a held-out sweep said +5.2 on `vs_shoot` (3/3,
   t≈2.9); confirmation gave **+0.51 (t=0.45)**, and the real effect was on two
   *other* cells. The tuning transferred in neither magnitude nor location.

## 6. Lessons the re-work should carry

**Measurement**

1. **Measure the comparator at the same n as the arm and propagate its SE.** A
   deterministic script is not a constant.
2. **Never quote an across-seed SE when every seed shares the evaluation
   scenarios.** Report the paired per-scenario estimator, or both.
3. **Raising n moves the mean, not just the interval.** A prediction that only
   narrows the interval assumes the small-n point estimate was unbiased.
4. **Check what a sign count would be under the effect you are claiming.** At
   sd ≈ 85 a true +9.5 effect predicts 24.4/45; observing 22/45 is not evidence
   against it. The "quote a t and a sign count" rule assumes the count is
   informative, and often it is not.
5. **A tuning sweep needs the same n discipline as the arm it feeds.**
6. **Arm-versus-arm paired comparisons are far stronger than agent-versus-bar**,
   because a shared initialisation and shared scenarios cancel most of the
   noise. Prefer designs that admit pairing; treat bar comparisons as needing
   an order of magnitude more episodes.

**Process**

7. **Pre-register with a power check written first.** Two criteria this session
   were unpassable or had an unhandled middle; both were recorded as defects in
   the test rather than as evidence.
8. **Predict before running, and record the prediction being wrong.** Three of
   mine were, and each wrong prediction taught more than the result.
9. **Never let a launcher's exit code stand as evidence a run happened.**
10. **`pgrep -f`/`pkill -f` match the issuing shell.** Two commands killed
    their own shell this session, surfacing only as a bare exit code.
11. **Do not edit code that running processes re-import.** Fresh `uv run` per
    job means a mid-sweep edit silently splits a sweep across two code versions.
12. **Prove a default-preserving change is bit-identical**, end to end, before
    trusting a result taken under it.

## 7. What the re-work should NOT carry

- ⚠ **Any pre-2026-09-06 agent-versus-bar figure**, including favourable ones.
  They are not wrong by direction so much as unmeasured.
- ⚠ **The decode as a source of further gains.** Four configurations measured;
  the stack is at its ceiling for this goal.
- ⚠ **More seeds as the answer to noise.** Episodes are the axis.
- ⚠ **Reward shaping for offence.** Three terms failed before this session, and
  nothing measured here revives them.

## 8. The one live thread

Distillation of the decoded policy at house fidelity (1200 demonstrations x 60
epochs, K=3 + reallocation + charge decode) —
`reports/2026-09-06-distillation-preregistration.md`. It is the last route the
record itself nominates, the prior attempt was 10x under-powered, and the
pre-registered prediction is **FAIL**. Running it closes the route rather than
leaving it as a plausible untried idea.
