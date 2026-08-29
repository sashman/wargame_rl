# The folded-commitment successor — design brief (pre-panel, pre-v10r)

## Goal-level status (as of 2026-08-30, `ce1e441`)

- **Level 1 (the agent CAN play correctly): MET, certified on a trained
  agent 2026-08-30.** Direct phase-participation census (v12 s5 last.ckpt,
  declare_refereed, n=10, K=3, casualties attributed to the phase they land
  in): the agent kills **13.0/ep shooting, 1.0/ep in its own FIGHT phase, and
  2.4/ep striking back in the opponent's fight phase** — it fights in melee
  on both sides of the engagement — while declaring plans (13–17/ep, apr
  census) and standing referee-certified charges (1.69/ep at n=45). Illegal
  play is impossible by construction: charge, pile-in, consolidate and (since
  `507d510`) the fall back all revert whole-unit on any after-move violation,
  each verified through `env.step`; the fall-back audit closed the last
  engine-side fidelity gap (opponent tears 9.0% → 0.5%, control level).
  Probe: session scratchpad `probe_l1_phases.py`.
- **Level 2 (reasonable vs the ladder): OPEN.** v12-lite REJECTED on every
  gate; every pre-referee number void; v10r retraining under corrected rules
  re-establishes the learned reference tonight.
- **Level 3 (beats the bar): OPEN.** The corrected bar is §38 (charging
  script: −5.3 refereed / +20.2 / +11.8 / +56.6). The fold arm is the funded
  attempt: mechanics landed (`58e6afe`), configs + two-stage pre-registration
  landed (`ce1e441`), launch blocked only on v10r's bounds.

Written 2026-08-30 at `e843d35`, while v10r trains. This is the starting document
for the successor's panel round and pre-registration. It captures the evidence
steers so they survive context loss; it decides nothing the panel should decide.

## What the evidence has already fixed

1. **The command-slot bottleneck is the binding constraint** (§35, memory
   `command-slot-bottleneck`). One leader action per command phase; each added
   declaration slice displaced charge declarations 8.55 → 5.89 → 2.68/ep.
   v11's hunt was rejected for halving the charging it was built to increase.
   Rule: fold commitments into ONE declaration; a hunt IS charge-intent.
2. **Execution payment works; the CHOICE fails** (§37a). The hold term paid
   3–5× the march term on every v12 seed — commitments are kept and collected.
   Home was never *chosen* as the commitment. Re-pricing execution (any pot,
   any dose) aims at the wrong stage. The successor targets the declaration
   distribution.
3. **The corrected game pays melee heavily** (§38): charging beats walking by
   +28 to +33 vp on all four ladder cells post-fall-back-referee. A
   hunt-capable policy has far more headroom than v11 had — v11's landing was
   measured on rules where a pinned unit could tear free, i.e. where the lock
   it was buying was leaky.
4. **v11's mechanics survive; its slot economics died.** `charge_target`
   (declare_targets), `declared_target_progress` (delta-paid, casualty-safe)
   and the census instruments are all committed and tested. The adjacency
   zero-gap bias of the progress family is documented and owed a distribution
   printout at landing.

## The candidate mechanism (for the panel to attack)

ONE plan slice of `6 + max_groups` values replacing the separate
`objective_target` (6) and `charge_target` (max_groups) slices: the leader
declares either "our plan is objective i" or "our plan is enemy unit j".
Declaring a hunt carries charge-intent: a hunt-planned unit auto-declares the
charge in every command phase in which the charge is legal (eligibility +
roll-reachability as today), WITHOUT consuming the leader's command action —
that is the fold, and it is what removes the §35 displacement.

Net action-space change: 114 = 108 + 6(objective_target); successor =
108 + (6 + max_groups) = 119 at max_groups 5. Width changes ⇒ **unpairable
against v10r** (net.py re-rolls every Gaussian on any width change — the
pairing myth is refuted on file). Protocol: six seeds unpaired vs v10r at six
seeds, the advance-arm protocol, plus a `dark_action_slices` shape-control if
carrying-cost needs isolating (KNOWN_ACTION_SLICES already admits both slices).

## Questions the panel must settle (not this document)

- Auto-charge semantics: every-legal-turn, or intent-that-the-policy-can-
  override? The §36(b) lesson (a forced override the policy does not replan
  around collects nothing) cuts against hard forcing; the §35 lesson cuts
  against making it cost the slot. There is a real tension here.
- What pays a hunt plan: `declared_target_progress` as-is (with its adjacency
  zero-gap), or plus a kept-commitment term for engagement (the §37a steer says
  the choice, not the payment, is the problem — so maybe nothing new pays).
- ⚠ **§38a WAS A SIGN FLIP — struck by §38b (panel A red team, chair-verified,
  hand-verified).** Corrected: post-referee garrisoning is worth **+7.6 ± 6.7**,
  the same sign as §36's +10.4, attenuated within noise. The garrison value
  REPLICATES. Licensed claim: "consistent with §36", never "proven". The fold
  still stands on §35's slot economics, not on home-guarding — but the
  correction revives the hold-plus-pin composite that plan exclusivity forbids;
  panel F2 (on-objective charge share) is pre-launch, with the pre-committed
  trigger: ≥70% on-objective standing charges AND retention failing at verdict
  ⇒ redesign plan semantics (objective plan with hunt rider) before any further
  12-seed round.
- Primary readouts: declaration-distribution census (home share, hunt share,
  P(nearest)), charge cells vs §38's bar, ladder vp. Bounds ONLY from v10r's
  landing — every pre-referee bound is void.

## Free screens available before any GPU

- Scripted 2×2 on the corrected rules: `squad_march_take_charge` (hunts
  implicitly) vs `squad_march_take` rows exist (§38); a scripted
  explicit-hunt variant would price the plan's value ceiling for a script.
- The charge census rider (`measure-charges`) now prints P(stood | gap) — the
  calibration table for any charge-value claim.
- The declaration census (`measure-declarations`) now prints the combined
  march+hold farm column — the §37a instrument.

## Iteration-speed rules (adopted 2026-08-30, user-prompted)

- **Two-stage pre-registration**: every arm binds a SCREEN clause at epoch 300
  (kill on unambiguous failure of the primary with all seeds trending against;
  else extend to 600) and a VERDICT clause at 600. Most arms fail; stop paying
  full price to confirm forecasts. Kill thresholds conservative — mode-settling
  is late (s5 flipped after epoch 450).
- **Standing auto-peek**: a detached script scores each live seed's newest
  checkpoint on the arm's primary every ~90 min into `trends.txt`. Peeks are
  forecasts, never verdicts.
- **CPU-first, always**: a script that can falsify the design (§38a killed the
  home motivation for 30 min of CPU) runs before anything trains.
- **One-time throughput buys, owed**: profile the melee config's step
  (`measure-throughput`, never run on it); A/B `bf16-mixed` over two seeds;
  `--eval-every-n-epochs 4` (legal on these single-phase configs, ~16%).
- NOT adopted: smaller-game screens (measures a different scenario — the
  costliest recorded error class), fewer seeds (seeds are the power lever).

## Standing traps this design must not re-trip

- Power-check every per-seed bound against the measured seed spread before
  writing it down (the −8/3-seeds bound failed 56% of the time on a true zero).
- The comparator is named BEFORE measuring; stamp the revision on every table.
- Nothing trains under enforcement; the fall-back referee is a RULE (like the
  charge referee), not a training referee — it stays on everywhere.
- A three-seed screen is a screen. Mode/usage oscillation is the convergence
  diagnostic (§35's lever-usage signal).

## Panel A verdict (2026-08-30, wf_bc5da8e3-5c3): LAUNCH_MODIFIED, unanimous 5/5, red team SERIOUS

Full output: workflow journal wf_bc5da8e3-5c3. The modifications are all header text,
census labels, and record corrections — no code redesign, no GPU. Apply M1–M8 in the
landing window before the first fold seed:

- **M1** farm screen: gains a declared_target income rider (or re-registers as
  objective+hold with the target column as NAMED DEBT) — as committed it references a
  column no instrument computes and cannot fire.
- **M2** eval assignment pinned: arm on fold_refereed, control on hunt_refereed;
  cross-decode is a labeled falsifier, never a primary.
- **M3** PRIMARY B → non-inferiority guardrail (pooled > −1.5× pooled SE AND no column
  below 1.5× its paired SE) — as committed it rejected a true-zero-vp mechanism win >50%
  of the time (the design's own most likely success mode for a lethality-neutral
  mechanic). Bounds power-checked against the PAIR's own spread; v10r numbers marked
  PROVISIONAL (melee family per-episode sd is 79–101, not the doctrine ~45–50).
- **M4** fold-vs-v10r ADOPTION GATE restored as pre-registered secondary (non-inferiority,
  1.5× unpaired SE; UNDERPOWERED clause if SE > ~15).
- **M5** screen kills THE FOLD ONLY (control runs to 600 — it is the only planned test of
  whether v11's rejection was partly a pre-referee artifact); screen ckpt = highest
  ppo-NNN ≥ 290; attrition rule written.
- **M6** six labels, one definition each — headlined by THE MISMATCH CENSUS: share of
  stood charges landing on the DECLARED group (the shipped grant is proximity-shaped:
  any-enemy gates + nearest-unit approach mask — three seats converged independently);
  pre-committed: >50% mismatched stood charges ⇒ A is not read as "declaration
  distribution improved"; follow-up is a target-gated grant, not a relaunch.
- **M7** licensed claim: full PASS = "enabled charging without slot cost", NEVER "learned
  hunt allocation" (one-profile armies make targeting degenerate); the flag bundles TWO
  mechanisms (grant + exclusivity) with the frozen-weights attribution probe pre-named;
  the pre-registration binds the SHIPPED form (two slices + post-hoc exclusivity).
- **M8** record corrections: §38b (done, dd75285); entities.py stale STAY comment and the
  misnamed test_out_of_range_hunts_are_not_granted_a_charge queued for the landing window.

Free work F2–F5 (CPU, one at a time, before launch): F2 on-objective charge share of the
bar; F3 six-seed frozen swap (the mechanical-floor comparator column); F4 hunt-plan vs
objective-plan per-model income on v11 seeds; F5 P(declared target = nearest). F6 the
scripted explicit-hunt pricer in the landing window. F7 epoch-0 state_dict assert at
launch. What died and the standing-belief challenges: see the chair's sections 6–7.

## Free-work results (2026-08-30 ~02:00, /tmp/panel_fold/out/)

- **F3 frozen-swap floor: +1.81 ± 0.29 stood/ep, 6/6 positive** (K=1, n=20/seed).
  The floor is real and large; it enters the verdict table as PRIMARY A's named
  comparator column. vp deltas at n=20 are noise and are not quoted.
- **F2 on-objective charge share: 70.2%** (33/47 standing charges of the bar
  engage a unit holding an objective). The pre-committed trigger's first
  conjunct (≥70%) is SATISFIED at the threshold — the redesign (objective plan
  with hunt rider) fires if hunt retention fails at verdict.
- **F4 income asymmetry SURVIVES its kill-check**: hunt-plan units earn
  0.0071–0.0092/model-step in the declared channels vs objective plans'
  0.0133–0.0285 — 2–4× less, **6/6 seeds** (s6 the widest: 0.0067 vs 0.0285).
  ⚠ Caveat: v11 has no
  exclusivity, so both-plan units classify as hunts in the probe — directional
  evidence, not confirmatory. Stays a LABEL risk, not a pot (per the nulls).
- **F5 march-annuity story KILLED**: P(new hunt = nearest enemy) 0.14–0.27
  across seeds, vs the objection's predicted >0.8. v11 target selection is
  already diverse; the gamma/march-dominance explanation for the declaration
  distribution is refuted at its own criterion.
