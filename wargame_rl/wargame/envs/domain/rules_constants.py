"""Rules constants, in inches.

The rules are specified in inches; the environment plays in its own coordinate
unit. This module is the runtime source for every universal rules quantity, and
``docs/rules/constants.yaml`` is the specification. ``tests/test_rules_constants.py``
asserts the two agree, so neither can drift.

The values here are *universal rules constants* only. Per-profile characteristics
-- a model's Move, a weapon's Range -- vary from model to model and belong in the
config, not here; the defaults for those live alongside their config fields.

**These are the rules' values, not necessarily the environment's.** Several
scenarios deliberately deviate, and the deviation lives in the config where it is
visible. Where the environment's default differs from a constant here, the gap is
recorded in ``docs/rules/implementation-status.md`` rather than silently
reconciled. Reading a constant from this module is therefore a statement that the
env implements that rule faithfully -- do not wire one in to "fix" a divergence
without measuring, because every shipped baseline was measured under the
env's value.

Nothing in this module reads a file. The YAML is documentation and is not shipped
as a runtime dependency.
"""

from __future__ import annotations

MM_PER_INCH = 25.4

# 01-core-concepts.md / 03-moving.md
# A model is engaged while an enemy is within this distance of its base.
ENGAGEMENT_RANGE_IN = 2.0

# Every model must be within this of at least one other model in its unit.
COHERENCY_NEAREST_IN = 2.0

# Every model must be within this of every other model in its unit.
COHERENCY_FURTHEST_IN = 9.0

# 02-unit-profiles.md
# The common infantry base. Real base sizes are per-profile; this is the default.
MODEL_BASE_DIAMETER_MM = 32.0

# Half the base diameter, in inches -- roughly 0.63".
MODEL_BASE_RADIUS_IN = MODEL_BASE_DIAMETER_MM / MM_PER_INCH / 2.0

# 13-terrain.md
# Worsen the Ranged Skill of an attack against a unit that has cover.
COVER_RANGED_SKILL_PENALTY = 1

# 14-objectives.md
# A model is within range of an objective marker while within this of it.
OBJECTIVE_MARKER_RANGE_IN = 3.0

# 07-battle-round.md
# Battle rounds in a standard game.
BATTLE_ROUNDS = 5
