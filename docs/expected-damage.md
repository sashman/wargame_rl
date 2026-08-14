# Expected damage

Closed-form expected damage for the [attack sequence](rules/05-attack-sequence.md), including the
abilities that normally force case-splitting. The result is that the whole calculation stays a
**single product** — cheap to evaluate, cheap to differentiate, and usable as an analytical check on
what the simulation produces.

This is a pricing reference for the abilities in
[16 — Ability reference](rules/16-ability-reference.md). Which of them are actually implemented is
[implementation-status.md](rules/implementation-status.md)'s job, not this document's.

`domain.shooting.expected_damage(weapon, defender)` implements equation (1) for the no-abilities
case — hit, wound, save and Damage only. Everything below is the generalisation: each ability is one
cell edit, so extending that function as abilities land is a table lookup rather than a
rederivation. See [shooting.md](shooting.md).

---

## 0. Notation

The attack sequence is a chain of **gates**: hit roll → wound roll → save roll. Each is a d6, and
the method represents a gate not as a probability but as a **face-value vector** — what one die is
worth when it shows each face.

For gate *i*, let `vᵢ ∈ ℝ⁶` be that vector and define the **gate value**

```
gᵢ = (1/6) · Σⱼ vᵢ[j]
```

With no abilities, `vᵢ` is 0s and 1s and `gᵢ` is just the pass probability. The whole point of the
method is that abilities are **edits to individual cells of `vᵢ`**, and cells may hold values
greater than 1 and non-integers.

The expected damage is then a pure product:

```
E[damage] = A · g_hit · g_wound · g_save · D · g_shrug                                      (1)
```

`A` = expected attack dice, `D` = expected Damage per unsaved wound, `g_save` = probability the save
**fails**, `g_shrug` = probability `Shrug` **fails**.

Everything below is a rule for computing one term.

### The five ability families

Every ability that touches the attack sequence is one of:

| Family | Effect on `vᵢ` | Examples |
|---|---|---|
| **Modifier** | shifts which faces pass | `[BRACED]`, `[IMPACT]`, Stealth |
| **Bonus** | a critical face is worth `1 + X` | `[EXTRA HITS X]` |
| **Bypass** | a critical face skips the *next* gate | `[PIERCING STRIKES]`, `[SHATTERING WOUNDS]` |
| **Threshold** | criticals occur on `N+`, not only on 6 | `[BANE-X Y+]` |
| **Reroll** | some faces get a second draw | `[PAIRED]` |

Plus **auto-pass** (`[AUTO-HIT]`), which sets `g_hit = 1`, and the dice-count abilities
(`[SCATTER]`, `[BURST X]`), which only touch `A`.

---

## 1. Why it's a product

The gates are independent conditional events, so their probabilities multiply, and multiplication is
commutative and associative. Two consequences the method leans on:

- **Order is free.** Evaluate the gates in whichever order is convenient.
- **A proportional gain anywhere is that same proportional gain in the output.** If `g_hit` improves
  by a factor `α`, the product improves by exactly `α`. A 16.7% better hit gate is 16.7% more
  damage.

**Right-to-left is not arbitrary.** §5 gives abilities whose cell values depend on the *next* gate's
value. Evaluating save → wound → hit means that value is always already known.

---

## 2. Averages of dice

The faces `1 … N` are in arithmetic progression, so the mean equals the mean of the endpoints:

```
E[dN] = (1 + N) / 2                                                                         (2)
```

d3 → 2, d6 → 3.5. For `k` dice plus a constant, `E[k·dN + c] = k(N+1)/2 + c`; e.g. `d6+2 → 5.5`,
`3d3 → 6`.

Because (1) is linear in `A` and `D`, substituting these means is exact — no independence assumption
beyond what (1) already uses. `[OVERLOAD X]` is a `+X` on `D`; `[SCATTER]` and `[BURST X]` are
additions to `A`. Where Damage combines scaling and offset modifiers, apply the usual precedence:
multiplication and division before addition and subtraction.

---

## 3. Saves, complements, AP

A group saving on `Sv+` fails on faces `1 … Sv−1`, so

```
g_save = (Sv − 1) / 6                                                                       (3)
```

`Shrug X+` takes the same complement: `g_shrug = (X − 1) / 6`.

One point of AP shifts the save to `(Sv+1)+`, giving `Sv/6`. The **proportional** gain is

```
gain from one point of AP = Sv / (Sv − 1)                                                   (4)
```

| Sv | `g_save` | with AP −1 | gain |
|---|---|---|---|
| 6+ | 5/6 | 6/6 | ×1.20 |
| 5+ | 4/6 | 5/6 | ×1.25 |
| 4+ | 3/6 | 4/6 | ×1.33 |
| 3+ | 2/6 | 3/6 | ×1.50 |
| 2+ | 1/6 | 2/6 | **×2.00** |

(4) is strictly decreasing in `Sv`: **AP is worth most against the best saves.** Since an
invulnerable save [ignores AP](rules/05-attack-sequence.md), a defender with an InSv caps `g_save`
from below regardless of how much AP is thrown at it — check the InSv before pricing AP at all.

---

## 4. Rerolls

### 4.1 The formula

Let the gate value before rerolling be `O`, and let `n` be the number of **die faces** eligible for
a reroll, where those faces are worth 0. A rerolled die is a fresh draw from the same gate, worth
`O` in expectation:

```
g = O + (n/6)·O = O · (6 + n) / 6                                                            (5)
```

- Reroll 1s → `n = 1`
- Reroll all failures (`[PAIRED]` on the wound roll) → `n` = the number of failing faces
- Reroll 1s when already passing on 2+ → these coincide, `n = 1`

**(5) requires the rerolled faces to be worth zero.** That is what makes it a clean multiplier
rather than a sum; §7 handles the case where they aren't.

Sanity check against the textbook form. Passing on `p`, rerolling all failures: `p + (1−p)p`. With
`p = 2/6`, `n = 4`: (5) gives `(2/6)(10/6) = 5/9`; direct gives `1/3 + (2/3)(1/3) = 5/9`. ✓

### 4.2 The 1/6 rule

Rewriting (5) as `g = O · (1 + n/6)`: **each rerolled face multiplies the gate by an additional
1/6 ≈ 16.7% of its original value.** Three faces → ×1.5.

This is linear in the *multiplier*, not compounding within a gate — one reroll per die, so the `n`
faces all pay back against the same base `O`.

**Across gates it does compound**, because (1) is a product. Rerolling 1s at two gates is

```
(7/6) × (7/6) = 49/36 ≈ ×1.36
```

not ×1.33.

### 4.3 Rerolls favour bad gates

(5) is a fixed multiplier `(6+n)/6`, but with *full* rerolls `n = 6 − 6·O`, so worse gates get
larger `n`.

| Gate | none | reroll 1s | full reroll |
|---|---|---|---|
| 6+ | 0.167 | 0.194 | 0.306 |
| 5+ | 0.333 | 0.389 | 0.556 |
| 4+ | 0.500 | 0.583 | 0.750 |
| 3+ | 0.667 | 0.778 | 0.889 |
| 2+ | 0.833 | 0.972 | 0.972 |

Full rerolls on 6s (0.306) ≈ passing on 5s. Full rerolls on 5s (0.556) beats passing on 4s. At a 2+
gate, "reroll 1s" and "reroll all failures" are the same rule.

---

## 5. Bypass — the reciprocal principle

The load-bearing idea, and the reason the method exists.

**Claim.** If a die at gate *i* is allowed to skip gate *i+1*, its value in `vᵢ` should be set to
`1 / g_{i+1}`.

**Proof.** In (1) every die passing gate *i* is subsequently multiplied by `g_{i+1}`. A die entered
with value `1/g_{i+1}` therefore contributes `(1/g_{i+1}) · g_{i+1} = 1` to the downstream product —
exactly the behaviour of a die that bypassed the gate and arrived intact. ∎

The exchange rate is *how many ordinary dice you would have to send at the next gate to get one
through*: the reciprocal of its pass rate.

**`[SHATTERING WOUNDS]`** — a critical wound ends the sequence and inflicts piercing damage, which
skips the save:

```
v_wound[6] = 1 / g_save                                                                     (6)
```

Piercing damage still meets `Shrug`, so `g_shrug` stays in the product. It also
[spills between models](rules/06-visibility-and-damage.md#piercing-damage) where ordinary damage
does not — irrelevant to expected damage, relevant to how it lands.

**`[PIERCING STRIKES]`** — a critical hit may auto-wound, skipping the wound roll:

```
v_hit[6] = 1 / g_wound                                                                      (7)
```

**This is what forces right-to-left evaluation.** `g_save` must be known before the wound table can
be filled in, and `g_wound` — *including* its own `[SHATTERING WOUNDS]` cell, its `[BANE]` faces and
its `[PAIRED]` reroll — before the hit table can be. Both dependencies point backwards; neither
points forwards.

**Corollary: no case-splitting.** The naive treatment of a skip is `(1/6)·X + (5/6)·Y` — a sum,
which destroys the product form of (1) and reintroduces ordering constraints. (6) and (7) express
the same expectation as a single cell edit.

### 5.1 When to decline `[PIERCING STRIKES]`

It is a choice, not a compulsion, and (7) prices the choice exactly. Taking it makes the crit cell
worth `1/g_wound`; declining leaves it worth 1 and the die rolls to wound normally.

```
Take [PIERCING STRIKES]  ⟺  g_wound < 1                                                     (8)
```

`g_wound` exceeds 1 whenever `[BANE-X Y+]` and `[SHATTERING WOUNDS]` are on the same profile against
a valid target — several faces each worth `1/g_save`. Against those profiles, auto-wounding
**throws damage away**: you trade a die worth `g_wound > 1` for exactly one ordinary wound. This is
the same threshold as §7, and it is the one place in the sequence where the obviously-good ability
is the wrong call.

---

## 6. Bonus, threshold, auto-pass

**`[EXTRA HITS X]`** — the critical hit still hits, and scores X more:

```
v_hit[6] = 1 + X                                                                            (9)
```

**`[BANE-X Y+]`** — every face from `Y` to `6` takes the critical value rather than 1. With
`c = 7 − Y` critical faces and critical value `V`, they contribute `cV/6` instead of `c/6`.

This is why `[BANE]` stacks explosively with `[SHATTERING WOUNDS]`: `V` is a reciprocal (up to 6),
and `c` multiplies it. `[BANE-X 3+]` plus `[SHATTERING WOUNDS]` against a 2+ save gives `c = 4`,
`V = 6` → `g_wound = 24/6 = 4`.

**`[AUTO-HIT]`** — `g_hit = 1`. Note the ability explicitly makes the hit **never critical**, so it
silently disables `[EXTRA HITS]` and `[PIERCING STRIKES]` on the same profile. Set those cells to
nothing rather than pricing them.

### 6.1 `[EXTRA HITS 1]` ≡ +1 to hit

A +1 modifier converts one failing face to a passing one: `g_hit += 1/6`. `[EXTRA HITS 1]` raises
the critical cell from 1 to 2: `g_hit += 1/6`. **Identical in expectation.**

Two caveats the identity hides. At a 2+ gate the modifier is worth nothing (an unmodified 1 always
fails) while the ability still pays — they diverge at the ceiling. And they differ in variance (§8).

### 6.2 `[EXTRA HITS X]` against `[PIERCING STRIKES]`

Both edit only `v_hit[6]`, and neither changes `g_wound`, so comparing the cell values decides it:

```
[EXTRA HITS X] beats [PIERCING STRIKES]  ⟺  1 + X > 1 / g_wound  ⟺  g_wound > 1/(1 + X)    (10)
```

For X = 1 the threshold is `g_wound > 1/2` — a 4+ wound roll with nothing else on the profile is the
break-even point. `[PIERCING STRIKES]` wins against high-Toughness targets, where the wound gate is
the bottleneck; `[EXTRA HITS]` wins against soft ones.

`g_wound` here is the *finished* wound gate, after `[BANE]`, `[SHATTERING WOUNDS]` and `[PAIRED]`.

---

## 7. Rerolling successes

Spending rerolls on non-critical **successes** to hunt for criticals. This is the one construction
that cannot be written as a multiplier, because (5)'s zero-value assumption fails.

Partition the 6 faces: `c` critical faces worth `V`, `s` plain successes worth 1, `f` failures
worth 0.

```
O = (cV + s) / 6                                                                           (11)
```

**Strategy A** — reroll failures only, the ordinary use of a reroll:

```
E_A = O + (f/6)·O = O·(6 + f)/6
```

**Strategy B** — reroll everything that isn't a critical:

```
E_B = (c/6)·V + ((s + f)/6)·O                                                              (12)
```

Subtracting:

```
E_B − E_A = (s/6)·(O − 1)                                                                  (13)
```

### The decision rule

```
Reroll successes  ⟺  O > 1
```

with `O` the gate value **before any rerolls**. (13) makes the intuition precise: you trade each
plain success, worth exactly 1, for a fresh die worth `O`. Good iff the average die beats a
guaranteed success — which requires some cell above 1, i.e. a bypass or bonus ability in play.

(13) also gives the **size** of the gain, which the rule alone doesn't: it scales with `s`. A gate
with `O` barely above 1 and few plain successes gains almost nothing.

---

## 8. Variance

Expectation is not the whole decision. The two critical-hit families move spread in opposite
directions:

- **Bypass reduces variance.** It removes a random gate for those dice — they clear the next gate
  with certainty instead of with probability `g_{i+1}`. Deleting a Bernoulli stage removes its
  variance.
- **Bonus increases variance.** It adds dice, each of which still runs the full remaining gauntlet,
  adding binomial variance.

The corollary at or near the (10) break-even, where the means coincide: **take the bypass when the
expected damage already suffices** (you want the mean delivered reliably) **and the bonus when it
doesn't** (you need the upper tail). Killing a model is a threshold event, so the right choice
depends on which side of the threshold the mean sits.

---

## 9. The recipe

1. `A` ← expected attack dice, via (2), plus `[SCATTER]` / `[BURST X]`.
2. `D` ← expected Damage, via (2), plus `[OVERLOAD X]`, in standard precedence order.
3. `g_shrug` ← `(X − 1)/6`, or 1 if the target has no `Shrug`.
4. `g_save` ← `(Sv − 1)/6` per (3), after AP; use the InSv instead where it is better.
5. `g_wound` ← 1 for passes; `1/g_save` in the critical cell for `[SHATTERING WOUNDS]` (6); extend
   critical faces down for `[BANE]`. Then `[PAIRED]` via (5), or §7 if `O > 1`.
6. `g_hit` ← 1 for passes; `1/g_wound` for `[PIERCING STRIKES]` (7) — but check (8) first — or
   `1+X` for `[EXTRA HITS X]` (9); then rerolls.
7. Multiply all six terms per (1).

No step produces a sum except §7, which is the only place order can bite you.

---

## 10. Worked examples

All four verified with exact rational arithmetic.

**No abilities** — 20 attacks, hit 3+, wound 5+, Sv 5+, Damage 1.

```
20 × 4/6 × 2/6 × 4/6 × 1 = 80/27 = 2.963
```

**Bypass against bonus** — 3d3 attacks, hit 3+, wound 3+, Sv 5+, Damage 3, with
`[SHATTERING WOUNDS]` and a choice of `[PIERCING STRIKES]` or `[EXTRA HITS 1]`.

- `A = 3 × 2 = 6`, `D = 3`
- `g_save = 4/6 = 2/3`
- `g_wound`: critical cell `= 1/(2/3) = 1.5` → `(0,0,1,1,1,1.5)` → `4.5/6 = 3/4`
- `g_hit` with `[EXTRA HITS 1]`: critical cell `= 2` → `(0,0,1,1,1,2)` → `5/6`
- `g_hit` with `[PIERCING STRIKES]`: `= 1/(3/4) = 4/3` → `4.333/6 = 0.722`
- `[EXTRA HITS 1]` wins, and (10) agrees: `g_wound = 0.75 > 0.5`

```
6 × 5/6 × 3/4 × 2/3 × 3 = 7.5
```

**Everything at once** — 10 attacks, hit 2+ with `[EXTRA HITS 1]` + `[PIERCING STRIKES]` + reroll
1s; wound 5+ with `[PAIRED]` + `[SHATTERING WOUNDS]`; Sv 4+; Damage d6+2.

- `A = 10`, `D = 5.5`
- `g_save = 3/6 = 1/2`
- `g_wound`: critical cell `= 1/(1/2) = 2` → `(0,0,0,0,1,2) = 3/6`; `[PAIRED]` is a full reroll,
  `n = 4` → `(3/6)(10/6) = 30/36 = 5/6`
- `g_hit`: `[PIERCING STRIKES]` cell `= 1/(5/6) = 1.2`, plus `[EXTRA HITS 1]` → `2.2` →
  `(0,1,1,1,1,2.2) = 6.2/6`; reroll 1s `n = 1` → `× 7/6`

```
10 × (6.2/6) × (7/6) × (30/36) × (1/2) × 5.5 = 27.63
```

Note the `[PIERCING STRIKES]` cell used the wound gate **after** its reroll. That is the
right-to-left discipline doing its job.

**Rerolling successes** — hit 3+ with `[PIERCING STRIKES]`, wound 6+.

- `g_wound = 1/6` → critical cell `= 6`
- `v_hit = (0,0,1,1,1,6)`, so `c=1, V=6, s=3, f=2`, `O = 9/6 = 1.5 > 1` → reroll successes
- `E_B = (1/6)(6) + (5/6)(1.5) = 2.25`, against `E_A = 1.5 × 8/6 = 2.0`

---

## 11. Pitfalls

**AP gains are neither constant nor additive.** By (4) the gain depends on the save being improved
*from*: the first point off a 5+ save is ×1.25, the second is improving from 6+ and gives ×1.20.
A single percentage per point is wrong at every save except by coincidence.

**Compare success-rerolling against the right baseline.** §7 competes against `E_A` — rerolling the
*failures*, which you would have done anyway — not against the gate with no rerolls at all.
Measuring against the bare gate folds the reroll's own value into the decision. In the §10 example
the honest gain is 2.25 against 2.00, about **+12.5%**; against the bare 1.5 it looks like +50%.

**The gain is not `(O−1)/O`.** That is the improvement on a single upgraded die. Only the `s`
plain-success faces are upgraded, so the gate gains `(s/6)(O−1)` per (13). With `O = 6.2/6` and
`s = 4` the per-die figure is 3.2% but the gate gains **1.8%**.

**Test `O > 1` on the pre-reroll value.** The post-reroll value is what success-rerolling competes
*against*, not the value of the die you would draw. Using it inverts the decision on marginal gates.

**`[AUTO-HIT]` deletes your critical-hit abilities.** No hit roll means no critical hit, so
`[EXTRA HITS]` and `[PIERCING STRIKES]` contribute nothing on that profile — `g_hit = 1` is the
whole story, and pricing the crit cell as well double-counts.
