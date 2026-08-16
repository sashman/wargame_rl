# Rules primitives

A rule in this specification is not a mechanism. It is a **composition** of a few
mechanisms, and its name is a label stuck on the result. This file is the set of
mechanisms and the grammar that composes them, so that a rule can be *authored* rather
than implemented.

The test it has to pass is not "does this describe the rules". It is:

> When a rule changes, how many places change with it?

For a decomposition that works, the answer is one, and that one place is data.

## The problem, in one example

`13` § Cover is a single sentence of effect: *worsen the Ranged Skill of that attack by 1*.
It reached the code as `in_cover: bool` on two functions — a parameter named after the
**rule** rather than after anything the rule does. Three separable things fused into that
name: what grants cover (a ruin clipping the corridor), what cover does (`skill + 1`), and
whether the target has it at all. Consequences, all three already real:

- `Stealth` (16) grants cover with no terrain anywhere near it, and has nothing to attach
  to.
- `[IGNORES COVER]` (16) denies cover, and there is no state to deny.
- Two functions each re-derived the label independently, and disagreed — the closed form
  ignored cover entirely for as long as cover existed.

None of those are oversights. They are what a label does when it is used as an interface.

## The shape of a rule

Every rule in chapters 03–16 fits one frame:

```
WHEN   a trigger fires
IF     a condition holds
FOR    a scope — which models, and for how long
DO     an effect
```

The specification is already written this way and says so in its own conventions: move and
shoot types are given as *eligible if* / *before* / *while* / *after* blocks. That is this
frame with the trigger implied by the phase.

Four slots, and the primitives below are the vocabularies each slot draws on.

---

## 1. Quantity

A named number on a model, weapon or unit, plus the **kind** that says what "improve"
means. The kind is the whole content: it decides the arithmetic, the clamp, and the
direction of every modifier that will ever touch it.

| Kind | Members | Improve | Clamp (`02` § Clamps) |
|---|---|---|---|
| `target` — roll at least this | RS, MS, Sv, InSv, Rv | subtract | RS/MS/Sv/InSv ≥ `2+`; RS/MS ≤ `6+`; Rv `4+`–`9+` |
| `penalty` — worsens someone else | AP | subtract | never worse than `0` |
| `magnitude` — a plain count | M, T, W, CV, R, A, S, D | add | mostly ≥ 1 |

```yaml
quantity: ranged-skill
kind: target
clamp: {best: 2, worst: 6}
```

Nothing else in the system needs to know that RS is a hit target. A rule says *worsen
ranged-skill by 1* and the kind supplies the sign and the bound.

## 2. Roll

A dice pool and a **face-value vector** `v ∈ ℝ⁶` — what one die is worth showing each face
— with value `g = (1/6)·Σⱼ v[j]`. [expected-damage.md](../expected-damage.md) develops this
and is the worked instance of this whole document; the short version is that abilities are
edits to cells of `v`, and that a representation of "threshold plus probability" cannot
hold the ones whose cells are neither 0 nor 1.

```yaml
roll: hit
pool: {per: attack-die}
target: {quantity: ranged-skill, of: attacker}
pinned: {1: fail, 6: critical}      # never movable by a modifier
```

`pinned` is the primitive's own answer to two special cases the code currently hand-writes:
a worsening modifier is worth nothing at a `6+` gate and an improving one is worth nothing
at `2+`, because those cells cannot move. Every roll in the spec is an instance — hit,
wound, save, resolve, backfire, advance, charge, roll-off — differing only in pool,
target, and which cells are pinned. A save has no critical; a resolve roll has neither
pinned cell and a 2D6 pool.

Rerolls are an operator on the assembled gate rather than a cell edit, which is why they
compose differently and are treated in `expected-damage.md` § 4 and § 7.

## 3. Selector

A predicate that names a set of models, and a **quantifier** when that set has to collapse
to one answer about a unit. The quantifier is not a detail; in most rules it *is* the rule.

```yaml
selector:
  of: target.models
  where: {not: fully-visible, from: attacker}
  quantifier: every        # any | every | every-pair | connected
```

- `any` — a unit is visible, engaged, within range, or carries a keyword, when **one**
  model is (`01` § Descriptors).
- `every` — a unit has cover, is *fully* visible, or is wholly within, only when **all**
  are.
- `every-pair` and `connected` — coherency's spread and single-group clauses.

Two selectors over one set are not one selector. `04` says a target unit must have *some*
model visible and *some* model in range and that they need not be the same model, which is
two `any` reductions, not one.

## 4. State

**The primitive that replaces labels.** A named fact attached to a subject, written by
rules and read by rules, and by nothing else.

```yaml
state: has-cover
subject: attack           # model | unit | objective | terrain | attack
duration: this-attack
```

The whole discipline is one sentence: **nothing computes a state where it is read.**
Sources set it, consumers read it, and neither knows the other exists. That is the
difference between `has-cover` and `in_cover: bool` — the boolean is computed by its
consumer, so the consumer owns the definition, and a second consumer owns a second one.

Three properties, each earned by a rule this spec already contains:

- **Many sources, one name.** Terrain and `Stealth` both grant cover. Sources are
  independent rules that happen to write the same state.
- **Deniable.** `[IGNORES COVER]` sets it off. A state can be switched; a computation
  inside a consumer cannot.
- **Directionality belongs in the name, not in a field.** Where a state is asymmetric, it
  is two states — being able to see and being seen — not one state plus a direction
  parameter, because a reader that has to interpret the direction is a reader that can
  interpret it wrongly. Sight here is exactly symmetric, which is *why* one name suffices
  and is worth recording as a fact rather than an accident.

Names are a closed vocabulary; their parameters are open. That split is already this
repo's convention — config models forbid unknown keys while `params` bags stay free-form.

The states this spec needs: `has-cover`, `visible`, `hidden`, `engaged`, `in-coherency`,
`suppressed`, `controlling`, `secured`, `below-half-strength`, `advanced-this-turn`,
`charged-this-turn`, `selected-to-move-this-phase`, `selected-to-shoot-this-phase`,
`has-fought-this-phase`.

## 5. Effect

The leaves, and the combinators over them. Leaves:

| Effect | Parameters |
|---|---|
| `modify-quantity` | quantity, op, value, optional scaling |
| `set-state` | state, on/off |
| `move` | max distance, before/while/after constraints |
| `lose-wounds` | amount, **spill policy** |
| `destroy` | **credit**: does this fire destruction triggers |
| `grant` | ability or keyword, duration |
| `roll` | any of § 2, with what success and failure mean |

Two of those parameters carry rules that would otherwise be separate mechanics. **Spill
policy** is the entire difference between ordinary damage, which stops at the model it was
allocated to, and piercing damage, which moves on. **Credit** is the entire difference
between a model destroyed by an attack and one removed for falling out of coherency, which
triggers nothing and counts for nobody.

Combinators, because effects nest:

```
sequence   — do these in order
choice     — the controlling player picks one
conditional— if <condition> then <effect> else <effect>
for-each   — over a selector's set
dice-gated — roll, and apply on a result
```

`modify-quantity` takes declarative **scaling** rather than arithmetic in code — *once per
N of some counted thing, rounded, capped* — so "+1 for every five models in the target
unit" is a value, not a function.

## 6. Trigger

When a rule is asked. A `(scope, edge)` pair — scope in {battle, round, turn, phase, step},
edge in {start, end} — plus an owner and an optional once-per limit, or an **event** raised
during resolution.

```yaml
trigger: {scope: phase, phase: shooting, edge: end, owner: active}
trigger: {event: attack-targets-unit}
trigger: {event: model-destroyed}
```

Ownership is load-bearing: *your* Movement phase and *the* Movement phase fire a different
number of times per round (`07`), and a once-per-phase limit binds only the player whose
rule it is (`01`).

**When several rules fire at one point**, `01` § Resolving simultaneous rules gives the
order: active player's compulsory, active player's optional, opposing player's compulsory,
opposing player's optional — and anything a resolution creates waits for the queue to
drain. That ordering belongs to the trigger system, once, rather than to each rule.

---

## Composition laws

Four orderings the primitives obey. They are the part most likely to be re-derived
inconsistently, so they belong in one place.

1. **Within a modifier**: replacements, then ×, then +, then ÷, then −, then round up
   (`02`). A replacement to `0`, `-` or `*` freezes the quantity and skips the rest.
2. **Modifiers versus rolls**: rerolls happen first, then modifiers. A rule reading an
   **unmodified** result reads between the two. A modifier to a *roll* is capped at ±1
   total; a modifier to a *quantity* is bounded by that quantity's clamp instead. These are
   different laws for different slots and collapsing them is a bug.
3. **Modifiers stay individually addressable until they are applied.** `[PSIONIC]` ignores
   *any or all* modifiers to an attack, and `02` § Ignoring modifiers makes the choice
   selective — keep the helpful, drop the harmful. Summing modifiers as they arrive makes
   that unrepresentable.
4. **Two rules that meet are a third fact.** Whether two rules stack, conflict, replace one
   another or are order-dependent is its own record, authored beside them, not logic hidden
   inside either.

---

## Rules rebuilt

The point of the vocabulary is that rules become entries. Cover, in full — four
independent entries, none of which names the others:

```yaml
# SOURCE — terrain
- when: {event: attack-targets-unit, attack: ranged}
  if:   {selector: {of: target.models, where: {not: fully-visible, from: attacker},
                    quantifier: every}}
  do:   {set-state: has-cover, on: attack, value: true}

# SOURCE — an ability, with no terrain involved at all
- when: {event: attack-targets-unit, attack: ranged}
  if:   {selector: {of: target.models, where: {has-ability: stealth},
                    quantifier: every}}
  do:   {set-state: has-cover, on: attack, value: true}

# SUPPRESSOR
- when: {event: attack-targets-unit, weapon-has: ignores-cover}
  do:   {set-state: has-cover, on: attack, value: false}

# CONSUMER — the only entry that knows what cover is worth
- when: {event: hit-roll, attack: ranged}
  if:   {state: has-cover, of: attack}
  do:   {modify-quantity: ranged-skill, op: worsen, value: 1}
```

Now run the change that prompted this document. *Benefit of cover does something else* is
an edit to the **last four lines and nothing else**. Adding `Stealth` is a new entry and
touches none of the others. `[IGNORES COVER]` is a third and touches none. The only
interaction — that the suppressor must land after the sources — is a composition law, not
control flow.

The same treatment, compressed, for the rest of what this spec covers:

| Rule | Entries |
|---|---|
| **Coherency** | selector (`every-pair` at 9", `connected` at 2") → `set-state: in-coherency`; a separate entry at the end-of-turn trigger reads that state and does `destroy(credit: false)` one model at a time |
| **Objective control** | selector (`any` model within range) → sum the `CV` quantity per side → compare → `set-state: controlling` at every phase-end trigger. Suppression sets `CV` to `-`, which is a quantity replacement, so it needs no rule of its own |
| **Engagement** | selector (`any` enemy model within 2") → `set-state: engaged`. Shooting eligibility, melee eligibility and fall-back all read that state; none of them measures a distance |
| **Advance blocks shooting** | `move` effect also does `set-state: advanced-this-turn` (duration: turn); the shooting eligibility entry reads it |
| **Charge** | `roll` (2D6, no pinned cells, result clamped to 12) → `move` with after-conditions → `grant: strikes-first, duration: turn` |
| **Cover, melee** | does not exist — and needs no rule saying so, because the consumer entry above is triggered on ranged hit rolls only |

That last row is the shape worth noticing. Under a boolean parameter, melee needs an
`in_cover` argument that is always False. Under a triggered entry, the absence is the
absence of an entry.

---

## Four fragility tests

Apply to any mechanic before implementing it. Each is a question with a required answer.

1. **Change what it does.** How many code sites change? — **One.** If two, the rule is
   being re-derived somewhere.
2. **Add a second source.** How many existing entries change? — **None.** If any, the
   consumer is computing the state rather than reading it.
3. **Deny it.** Can another rule switch it off without the consumer knowing? — **Yes.** If
   not, the condition is fused to its effect.
4. **Rename it.** Does anything but the name change? — **No.**

Test 4 has already been run on this project, at scale and by accident. This specification
renames every mechanic it inherited — the nerve check, the damage that skips saves, the
unit's profile — and not one mechanism changed in the renaming. A decomposition that
survives that was never coupled to the names; `in_cover` did not survive its first
encounter with a second consumer.

---

## Landing this in the environment

Two constraints on any implementation, both from this repo's existing rules rather than
from the model above.

**This is authoring-time data, not an inner-loop interpreter.** `env.step()` costs ~2.26 ms
and the reward path is memoised per model; a tree walked per attack die would be a
throughput regression measured in whole training runs. Resolve entries into the concrete
per-attack quantities once, at construction where possible and per (attacker, target unit)
otherwise — the same discipline that makes `expected_damage_matrix` cost one call per
distinct stat pair rather than one per model pair.

**Nothing here authorises a vocabulary entry with no consumer.** A state nothing reads and
a quantity nothing modifies are exactly the settable-but-inert fields this project has
already been bitten by. Add the entry with the code that reads it. Where a rule is
deliberately unimplemented, that belongs in
[`implementation-status.md`](implementation-status.md) as it does now — and an implementation
that cannot express an entry should say which one it dropped, rather than silently
resolving without it.
