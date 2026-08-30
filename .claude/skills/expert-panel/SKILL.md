---
name: expert-panel
description: Runs an adversarial expert panel over a technical claim or decision, using parallel subagent workflows. Use when the user says "expert panel", "panel review", "fan out experts", "brainstorming session with experts", or asks to have a finding, diagnosis or plan pressure-tested by specialists.
---

# Expert panel

**Panels are refutation engines, not idea engines.** On this project's record: three
headline nominations, three refuted; roughly eight audits, roughly eight landed. Every rule
below optimises for the half that works.

Two modes. **`audit` is the default** — point a panel at one specific claim and demand
attack. **`generate`** is available, but say plainly to the user that its historical hit
rate here is 0 for 3, and that its output is *hypotheses to kill*, never conclusions. Its
value has come from side findings (an unswept hyperparameter, a statistics bug), not from
the headline.

⚠ **Never call `Workflow` unless the user has explicitly opted into multi-agent
orchestration.** Invoking this skill counts as that opt-in.

## The protocol

### 1. Write the brief before choosing the experts

The brief is most of the quality. Four mandatory sections:

- **Measured facts, with error bars.** No claim without its n, its SE and how it was decoded
  or scored. An expert given a bare number will build on it.
- **Measured nulls — an explicit "do not re-propose these" list.** Without it a third of the
  proposals will be things already refuted, and you will spend the red team's budget saying so.
- **How this project measures.** Pairing rules, seed minimums, what voids a comparison, the
  resolution floor. Proposals that ignore the local measurement discipline are unusable, and
  the experts cannot infer it.
- **The exact decision at stake, and what changes if the answer flips.** "Diagnose this" gets
  an essay; "this decides whether we spend 36 GPU-hours on X" gets a decision.

Point experts at the repository and name the files worth reading. They have tools; the good
ones will run measurements rather than theorise, and those are the reports worth having.

### 2. Two panels, uncoordinated, deliberately different lenses

Run them as two separate `Workflow` calls so neither sees the other. Convergence between
panels that never met is the strongest signal available.

⚠ **Distinguish convergence from correlated error.** In one round here, four experts
independently made the *same wrong argument* because they shared one upstream assumption
(centroid-to-centroid distance for objectives whose control test is an area). Several
experts agreeing is only evidence if they did not share the mistaken premise. When
agreement appears, check the premise before counting the votes.

### 2b. Budget the machine BEFORE launching — and stagger unless it is idle

⚠ **A panel is a fan-out of heavy jobs onto a box that is usually already busy.** Measured
2026-08-26: two 7-expert panels launched together drove load average to **160 on 24 cores**
(~6.7x oversubscribed) while three PPO runs were training. Cost: training fell 1.9 → 1.47
epochs/min and GPU utilisation dropped 11% → 3% — the trainers were starved of CPU for env
stepping. Nothing crashed and memory never tightened (17 GB free, zero swap I/O throughout),
so the whole bill was ~40 minutes of someone else's wall-clock. Cheap, avoidable, and paid
for no benefit: nothing depended on the two panels finishing in parallel.

**Check first, then decide.** One command, before the first `Workflow` call:

    uptime; nproc; free -g; nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv

Read it against what is already running:

| what you find | what to do |
|---|---|
| box idle, nothing training | launch both panels together — this is the fast path |
| training or another long job live | **stagger**: run panel A, wait for it, then panel B |
| load already > cores | stagger *and* cut each panel to 4–5 seats |
| memory tight or swap I/O non-zero | do not launch; that is the one that actually crashes things |

Staggering costs a panel's wall-clock and buys back the host job's throughput. It also
**improves** panel B: feed it panel A's findings, per §6, and the second panel is sharper for
having something to attack.

⚠ **A per-agent cap does not bound a fan-out.** The brief said *"run at most one measurement
at a time"* and every agent obeyed it — 14 agents × 1 job = 14 concurrent jobs. If you need a
ceiling, cap the number of *seats*, not each seat's behaviour.

⚠ **Tell agents which resource is contended, and check which one that actually is.** The same
brief told agents to prefer `CUDA_VISIBLE_DEVICES=` "so you do not contend with training".
That was backwards twice over: the training here is **CPU-bound on env stepping** while the
GPU sat at 11% with 13.6 GB free, so it pushed work onto the scarce resource and away from
the idle one — and it broke checkpoint loading outright, because checkpoints saved on CUDA
need `map_location` to load on CPU. Every checkpoint probe an agent attempted would have
died. **Name the contended resource only after measuring which one it is.**

### 3. Every proposal ships with a zero-cost falsifier

The single highest-value rule. Three theories were eliminated for **zero GPU** because each
arrived with a free screen. Make it a required schema field, not a hope:

    prediction      falsifiable, with a number
    free_screen     what can be checked on frozen weights / existing data, at zero cost
    kill_criterion  decided in advance
    pairable        yes/no; if no, the bridge that legitimises the comparison
    cost            in the currency that actually binds (GPU-hours, engineering days)

Also ask each expert **what they would refuse to spend on**. Negative information is
cheap here and rarely volunteered.

### 4. The red team gets a DUAL mandate

Not just "audit the proposals" — **audit the measurements the brief rests on**. Name the
specific instruments and tell it to read them. This mandate produced the most valuable
output of any round: a real bug in a measurement script, and the control the author had
failed to run.

Give it a verdict enum (`FATAL` / `SERIOUS` / `SURVIVABLE` / `SOUND`) and require that
`SOUND` be earned.

### 5. The chair leads with the measurement verdict

Order the synthesis: **is the evidence sound → convergence → the slate → the free work →
one fully specified first experiment → what died → challenges to standing beliefs.**

If the evidence is unsafe, everything downstream is noise. It belongs in section 1, never
buried at the end.

### 6. Feed each round the refutation of the last

A second round given "your consensus was tested and refuted, here is the data" is far
sharper than a fresh one. Include the retraction, the measurement that killed it, and an
instruction to attack the *new* claim before building on it.

## After the panel — the part that matters most

**1. Verify every load-bearing claim in the code yourself.** Panels assert wrong mechanisms
with total confidence. Real examples, each refuted in minutes:

- "it pays toward the objective's **centre point**" — false; those objectives are areas and
  the distance is to the outline
- "the existing control becomes a paired partner **for free**" — false; 73 of 110 shared
  tensors differ, because a narrower head consumes less RNG at init
- an arithmetic claim that closed a line of enquiry — overstated, and it was the author's own

Each would have misdirected real work. Budget an hour for this; it is the highest-return
hour in the process.

**2. Run the free falsifiers before believing anything.** Including the ones attached to
proposals you like.

**3. Hold proposals to their authors' own kill criteria.** When a free screen fails, the
proposals gated on it are unfunded *by their own rules* — say so explicitly rather than
letting a favourite survive on enthusiasm.

**4. Correct the record when the panel is right about you.** If a published claim falls,
retract it in the same document, keep the original visible, and write the lesson as a rule.
A retraction that is discoverable is worth more than the original claim was.

**5. Keep a scorecard.** Append to `.claude/skills/expert-panel/scorecard.md`: date, mode,
the nomination, its verdict, and the audits that landed. This turns "the panel was wrong
again" into a usable prior rather than a disappointment.

## Machinery

Each panel is one `Workflow` call:

- `phase('Panel')` — experts in `parallel()`, `effort: 'high'`, each with a `schema`
- `phase('RedTeam')` — one agent, given **all** panel output plus the dual mandate. This is a
  justified barrier: the red team needs every proposal at once to find shared failure modes.
- `phase('Synthesis')` — the chair, given the panel and the red team, ordered as in §5

Notes paid for in blood:

- **Scripts are plain JavaScript.** Backticks inside a template literal break the parse — build
  long briefs as an array of strings joined with `\n`, and avoid backticks entirely.
- Build the brief **once** as a constant and pass it to every agent, including the chair.
- Ask for `challenges_to_the_brief` in every schema. Some of the best findings arrive there.
- Panel agents write files into the repo. Check `git status` afterwards and move throwaway
  probes to a scratchpad — they are useful provenance but not repo-quality.
- Keep each panel to 6–8 agents. Two panels plus chairs lands around 14, which is a
  reasonable ceiling **for a machine with nothing else running** — see §2b, which is
  where that number goes wrong.
- **Give every brief an environment-safety block** naming what must not be touched: live
  training processes, the only copy of the weights, destructive recipes, git state, and any
  API known to destabilise running jobs. Fourteen agents with full tool access on a box
  carrying live runs broke nothing, and that block is why.

## Choosing lenses

For the specified panel, use whatever the user named. For the shadow panel, pick lenses that
attack the problem *differently* — not adjacent specialisms. If the first panel is
architectural, make the second one about measurement, incentives, and optimisation dynamics.

**Always include a methodologist** with an explicit mandate to audit the statistics and the
instruments. On this project that seat has never been wasted.
