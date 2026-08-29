# The folded-commitment successor — design brief (pre-panel, pre-v10r)

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
- ~~Whether the home objective needs anything at all~~ **SETTLED same day
  (§38a): the garrison value is void and does not replicate** — pooled
  **−7.6 ± 6.7** on the corrected rules against +10.4 pre-referee. Home is at
  best free for the best script. The successor pursues the hunt fold on its own
  merits; home-guarding is no longer a design goal unless a future mission
  re-prices it.
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
