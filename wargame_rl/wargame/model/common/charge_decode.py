"""The charge decode: execute a declared charge as a rigid unit translation.

⚠ **The measured failure this exists for.** Policies here DECLARE charges and
then do not make them. The bar declares 8.5–9.8 an episode and stands 5.8–6.3;
the fold-tb agent declares up to 8.9 and stands 0.6–1.6 (§42), and a clone of
the bar itself declares **12.9–13.9** and attempts only **2.0–2.5** (§44). The
declaration is learned; the move is not.

**Why a per-model policy cannot make the move.** A legal charge needs every
member of the unit to end engaged with ONE enemy unit, each closer than it
started, and the unit still coherent — a joint property of five simultaneous
choices. The script solves it constructively with a **rigid translation**: one
shared (angle, rung) for the whole unit, sized to put the nearest member just
inside engagement range, declined unless a legal rung covers it. A product of
five independent softmaxes rarely puts that one specific bin in every member's
top-K, which is why joint decoding at K=3 leaves ~80% of declarations
unattempted and K=5 recovers only +0.32 stood/ep (§42). This is the same
architectural limit the record measured for squad headings, where 83% of a
script-agent behavioural gap was the factored policy rather than skill.

So this decode does for the charge what `decode_joint_coherent` does for
formation and `apply_reallocation` does for surplus squads: it supplies the
JOINT move the architecture cannot express, for units the POLICY chose to
commit. The choice to charge stays the agent's; only the execution is decoded.

⚠ **It overrides only units that declared, and only where the script would
also move.** A unit the policy did not commit is untouched, and the env's own
referee still judges the result — a decoded charge that ends illegal is
reverted exactly like any other.

⚠ **PLAY-TIME ONLY**, like every decode here: folding one into PPO means the
executed action is not the sampled one, measured at −51.8 vp.
"""

from __future__ import annotations

from typing import Any

from wargame_rl.wargame.envs.baseline.scripted_squad_march_charge import (
    ScriptedSquadMarchTakeChargePolicy,
)
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

_ORACLE: ScriptedSquadMarchTakeChargePolicy | None = None


def _oracle() -> ScriptedSquadMarchTakeChargePolicy:
    """One shared instance — the policy is stateless across calls."""
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = ScriptedSquadMarchTakeChargePolicy()
    return _ORACLE


def apply_charge_decode(actions: list[int], env: Any) -> list[int]:
    """Replace declared units' charge-phase actions with the rigid charge move.

    A no-op outside the charge phase, with melee off, for units that did not
    declare, and wherever the constructive rule declines (no legal rung, or
    the unit is already incoherent so the translation would land it broken).
    """
    if env.game_clock_state.phase is not BattlePhase.charge:
        return actions
    if not env.config.melee.enabled:
        return actions
    models = env.wargame_models
    declared = {
        int(model.group_id)
        for model in models
        if model.is_alive and getattr(model, "declared_charge", False)
    }
    if not declared:
        return actions
    proposal = _oracle().select_charge(models, env)
    decoded = list(actions)
    changed = False
    for index, model in enumerate(models):
        if not model.is_alive or int(model.group_id) not in declared:
            continue
        suggestion = int(proposal.actions[index])
        if suggestion != STAY_ACTION:
            decoded[index] = suggestion
            changed = True
    return decoded if changed else actions
