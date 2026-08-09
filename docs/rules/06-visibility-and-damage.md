# 06 — Visibility and damage

Three concepts that sit outside the attack sequence but are used constantly by it.

## Visibility

Visibility between two models is decided by **line of sight**: whether an imaginary
straight line 1 mm wide can be drawn from any part of the observing model to any part of
the observed model.

While tracing it, ignore the other models in the observer's own unit and the other models
in the observed model's unit. Everything else — enemy models, third-party units, terrain —
can block.

| Term | Condition |
|---|---|
| **Model visible** | Any part of the observed model can be seen. |
| **Model fully visible** | Every part of the observed model that faces the observer can be seen — the only thing allowed to obstruct it is the model itself. |
| **Unit visible** | At least one model in the unit is visible. |
| **Unit fully visible** | Every model in the unit is fully visible. When checking this, the observer may see *through* other models of that same unit. |

Terrain layers additional restrictions on top of raw line of sight — see
[Terrain](13-terrain.md), which covers *hidden*, *obscuring* and *solid*.

Note that being **within range** of a weapon and being **visible** to it are independent:
a model can be in range without being visible.

### Describing visible units

When a rule mentions a visible unit but does not say visible *to whom*, it means visible
to the unit using the rule.

### Units not on the board

A unit that is not on the board (held back, or carried) is invisible in both directions,
and no distance can be measured to or from it. It cannot be selected or targeted by
anything requiring visibility or a distance — its own abilities excepted, since a unit is
always visible to itself and always within range of its own abilities.

Such a unit is still part of its player's army and can still be picked by rules that
select from an army or affect a whole army. Its controlling player must still make
suppression rolls for it in their Command phase if it is suppressed or at or below
half-strength.

## Piercing damage

Some attacks and abilities inflict **piercing damage**: Wounds that skip the hit, wound
and save rolls entirely.

Each time a unit suffers piercing damage, its controlling player resolves the following
once per point, until all of it is inflicted or the unit is destroyed:

1. **Select model** — take the first case that applies:
   - a non-`CHARACTER` model in the unit that has lost Wounds;
   - otherwise, any non-`CHARACTER` model in the unit;
   - otherwise, a `CHARACTER` model that has lost Wounds;
   - otherwise, any `CHARACTER` model.
2. **Resolve** — the selected model loses 1 Wound. At 0 Wounds it is destroyed.

Unlike ordinary damage, piercing damage does spill: each point is allocated separately, so
a surplus moves on to the next model rather than being lost.

When one pool of attacks inflicts both ordinary damage and piercing damage, resolve **all
the ordinary damage first**, then the piercing damage.

**Ordinary damage** means Wounds lost to a weapon's Damage characteristic. Piercing damage
dealt as part of an attack is still part of that attack.

## Backfire rolls

A **backfire roll** is a single D6 that punishes its own side. On a **1–2** the roll fails
and the unit suffers:

- 1 point of piercing damage; or
- 3 points instead, if every model in that unit is a `MONSTER`/`VEHICLE` model.

One roll is made per unit, not per model, unless a rule says otherwise. If several
backfire rolls are called for at once, make them all simultaneously.

Piercing damage from a backfire roll is allocated to the *unit* by the rules above — it
does not have to land on whichever model caused it.
