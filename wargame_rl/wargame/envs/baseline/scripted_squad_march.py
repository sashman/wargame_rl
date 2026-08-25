"""Baseline: each squad marches to one objective as a body, then holds it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.policy import (
    BaselinePolicy,
    objective_extent,
    step_toward_objective,
)
from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.env_components.actions import (
    MOVE_TYPE_ADVANCE,
    MOVE_TYPE_CHARGE,
    STAY_ACTION,
)
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedSquadMarchPolicy(BaselinePolicy):
    """Send squad *k* to objective *k mod n_objectives*, moving as one body.

    The strongest scripted baseline and the reference bar for the learned
    policy. Two properties make it so, and both matter for what the agent is
    supposed to learn:

    - **Concentration.** Whole squads on whole objectives produce an uneven
      split (5 squads over 3 objectives gives 2/2/1, i.e. 10/10/5 models).
      Control is a strict count comparison, so concentrating beats spreading.
    - **Coherency.** Steering on the squad centroid rather than each model's
      own vector keeps squads together, which is legal under the tabletop
      rules that the per-model baselines violate.
    """

    def squad_objectives(
        self, models: list[WargameModel], env: WargameEnv, group_ids: list[int]
    ) -> list[WargameObjective]:
        """One objective per squad, in `group_ids` order.

        The seam subclasses change to play a different allocation while keeping
        this class's centroid-steered, coherency-preserving movement. The
        assignment here is fixed at squad *k* -> objective *k mod n* and never
        revised, which is what makes this baseline a stable reference bar.
        """
        objectives = env.objectives
        return [objectives[i % len(objectives)] for i in range(len(group_ids))]

    # Advance when a normal full move cannot reach the objective: run while far,
    # walk once close. Advancing forbids this unit shooting for the turn, so a
    # marching squad trades fire it mostly could not deliver at range for
    # arriving sooner.
    #
    # A toggle rather than a hardcoded rule because the trade is a real one and
    # this project measures rather than assumes: `advance=0` reproduces the
    # pre-Advance bar exactly, so the two are directly comparable on one config.
    #
    # ⚠ It is a per-SQUAD decision, matching how the env resolves it: the roll is
    # one D6 per unit and `advanced_this_turn` is marked for every model of any
    # group that advanced, so a squad splitting its choice loses the shooting
    # anyway and gains only part of the distance.
    # ⚠ **REJECTED AS A DEFAULT, and the measurement that rejected it is the 2x2.**
    # "Run while far, walk once close" costs its USER about 78 vp. Measured on
    # `25v25_maps_advance_refereed`, held-out nine, n=10, `squad_march_take`
    # both sides, vp_margin to the player:
    #
    #                        opponent walks   opponent advances
    #     player walks            -4.1              +72.7
    #     player advances        -81.8               -3.6
    #
    # The both-advance cell (-3.6) is indistinguishable from both-walk (-4.1),
    # which is why a first measurement read this as "Advance is worth +15.5 to
    # the bar". It is worth nothing to the bar: the two sides were handicapping
    # themselves by the same amount and the effects cancelled. **Never measure a
    # symmetric change with both sides changed at once.**
    #
    # The mechanism (`ActionHandler.best_advance_toward`) is kept, because
    # Advance is a core rule and a scripted bar that cannot use it is not a bar.
    # What is rejected is this HEURISTIC. A better rule has to price the
    # forfeited shooting, which this one never does.
    advance_when_out_of_reach: bool = False

    # The replacement, and the difference is the whole point: this one prices
    # the shooting it gives up. An advance costs the unit its ENTIRE turn of
    # fire (`advanced_this_turn`; no weapon here has the ability that would
    # permit firing after one), so the only advance that is free is one that
    # forfeits a shot the unit did not have. That is what the rejected
    # heuristic above never asked, and it is why it cost its user ~78 vp.
    #
    # Still per SQUAD, for the same reason: the roll is one D6 per unit and the
    # env marks every model of any group that advanced.
    advance_when_no_shot: bool = False

    # D-40, and the refinement the measurement above asked for. `no_shot` alone
    # is REJECTED -- 0 of 3 seed bases, -18.4 vp paired -- and the mechanism is
    # not the shot it gives up (those advances end inside an enemy's reach on
    # 4.1% of moves against walking's 22.4%). It is that arriving EARLY buys
    # turns standing forward under fire: episode exposure +10.8%, firepower
    # ratio 1.091 -> 1.004, `alive` 0.396 -> 0.349.
    #
    # Control is a headcount at the scoring moment, so being nearer buys
    # nothing and being THERE buys everything. This clause spends the D6 only
    # when it converts a two-turn approach into a one-turn arrival, which is
    # the only advance whose gain is not zero.
    advance_to_arrive: bool = False

    def _would_forfeit_a_shot(
        self,
        models: list[WargameModel],
        member_indices: list[int],
        env: WargameEnv,
        displacement: np.ndarray,
    ) -> bool:
        """Whether any member would have an enemy in range after a NORMAL move.

        Range only, and deliberately so. Line of sight can only ever REMOVE
        shots, so a squad with nothing in range has nothing to forfeit however
        the sight lines fall; a squad with something in range may still be
        blocked. Erring this way declines a few advances that were in fact free
        and never spends a shot the unit really had — which is the safe
        direction, because the shot is the expensive half of the trade.

        Evaluated at the post-move position rather than the current one: the
        movement phase resolves before the shooting phase, so what the unit
        gives up is the shot it would have had *after* walking.
        """
        enemies = [model for model in env.opponent_models if model.is_alive]
        if not enemies:
            return False
        enemy_positions = np.array([m.location for m in enemies], dtype=float)
        ranges = env.player_max_ranges
        for index in member_indices:
            destination = np.asarray(models[index].location, dtype=float) + displacement
            reach = float(ranges[index]) if ranges.size else 0.0
            if np.any(np.linalg.norm(enemy_positions - destination, axis=1) <= reach):
                return True
        return False

    def _squad_advance_decisions(
        self, models: list[WargameModel], env: WargameEnv
    ) -> dict[int, bool]:
        """Which squads should declare an advance this turn, by group id.

        Computed in the COMMAND phase, before anything has moved — which is
        exactly when the rules ask for the move type, and why the same geometry
        the movement step uses is still valid here.
        """
        objectives = env.objectives
        decisions: dict[int, bool] = {}
        if not objectives or not (
            self.advance_when_out_of_reach
            or self.advance_when_no_shot
            or self.advance_to_arrive
        ):
            return decisions

        speeds = env.player_action_handler.move_speeds
        group_ids = sorted({model.group_id for model in models})
        targets = self.squad_objectives(models, env, group_ids)
        for squad_index, group_id in enumerate(group_ids):
            member_indices = [
                i
                for i, model in enumerate(models)
                if model.group_id == group_id and model.is_alive
            ]
            if not member_indices:
                continue
            max_step = float(
                min(speeds[i] for i in member_indices) if speeds.size else 0.0
            )
            objective = targets[squad_index]
            radius = objective_extent(objective)
            centroid = np.mean(
                [models[i].location for i in member_indices], axis=0, dtype=float
            )
            lead = np.asarray(objective.location, dtype=float) - centroid
            lead_distance = float(np.linalg.norm(lead))
            out_of_reach = lead_distance > radius and lead_distance > max_step
            walk_step = (
                lead / lead_distance * min(max_step, lead_distance)
                if lead_distance > 0.0
                else np.zeros(2, dtype=float)
            )
            roll = float(models[member_indices[0]].advance_roll)
            arrives_this_turn = lead_distance - radius <= max_step + roll
            decisions[int(group_id)] = out_of_reach and (
                self.advance_when_out_of_reach
                or (
                    (self.advance_when_no_shot or self.advance_to_arrive)
                    and (arrives_this_turn or not self.advance_to_arrive)
                    and not self._would_forfeit_a_shot(
                        models, member_indices, env, walk_step
                    )
                )
            )
        return decisions

    # ⚠ **A bar that cannot use a core rule is not a bar**, and until this
    # existed no scripted baseline and no scripted opponent could charge at all.
    # That is the same failure this project already paid for on Advance: an
    # agent trained with melee on would have been scored against a policy
    # physically incapable of the mechanic under test, so the arm would have
    # measured `baseline/policy.py`, not the agent.
    #
    # Off by default, so every baseline figure ever measured here is unchanged
    # and a charging bar is opt-in and separately attributable.
    #
    # ⚠ **This is ONE heuristic and must never be reported as "the value of
    # melee".** Six independently hand-rolled charging scripts produced +6.5,
    # +48.0, +52.0, +59.2, +82.9 and +88.8 vp for nominally the same
    # measurement — a 14x spread — because each measured its own rule. Quote the
    # ablation and the 2x2, never one arm.
    charge_when_it_lands: bool = False

    def _charge_target(
        self, models: list[WargameModel], member_indices: list[int], env: WargameEnv
    ) -> tuple[int, int] | None:
        """The (member, enemy) pair this squad would charge, or None.

        The closest living pair, which is also the pair that decides whether the
        charge reaches: the referee asks that the unit end engaged with exactly
        one enemy unit, and the nearest member is the one that gets there first.
        """
        best: tuple[float, int, int] | None = None
        for i in member_indices:
            for j, enemy in enumerate(env.opponent_models):
                if not enemy.is_alive:
                    continue
                gap = float(
                    np.linalg.norm(
                        np.asarray(enemy.location, dtype=float)
                        - np.asarray(models[i].location, dtype=float)
                    )
                )
                if best is None or gap < best[0]:
                    best = (gap, i, j)
        return None if best is None else (best[1], best[2])

    def _reachable_charge_units(
        self, models: list[WargameModel], env: WargameEnv
    ) -> set[int]:
        """Units eligible to charge whose nearest member can actually reach.

        The declaration and the charge must ask the SAME question, or a unit
        declares a charge it then declines to make -- and a declaration makes
        STAY illegal for its members, so declining is not available: the unit
        would be forced into whatever rung the mask left rather than into no
        move at all.

        Asked from geometry rather than from `charge_legality`, because this
        runs in the COMMAND phase where that mask is empty by construction: it
        is gated on a declaration that has not been made yet.
        """
        handler = env.player_action_handler
        eligible = handler.charge_eligible_units(models, env.opponent_models)
        if not eligible:
            return set()
        quantities = env.rules_quantities
        contact = float(quantities.engagement_range) + 2.0 * float(
            quantities.base_radius
        )
        speeds = handler.move_speeds
        enemies = [m for m in env.opponent_models if m.is_alive]
        reachable: set[int] = set()
        for group_id in sorted(eligible):
            member_indices = [
                i
                for i, model in enumerate(models)
                if model.group_id == group_id and model.is_alive
            ]
            if not member_indices:
                continue
            if not evaluate_coherency(
                positions=np.array(
                    [models[i].location for i in member_indices], dtype=float
                ),
                group_ids=np.zeros(len(member_indices), dtype=np.intp),
                alive_mask=np.ones(len(member_indices), dtype=bool),
                base_radii=np.array(
                    [models[i].base_radius for i in member_indices], dtype=float
                ),
                nearest_distance=env.config.coherency.nearest_distance,
                furthest_distance=env.config.coherency.furthest_distance,
            ).all_coherent:
                continue
            gap = min(
                float(
                    np.linalg.norm(
                        np.asarray(enemy.location, dtype=float)
                        - np.asarray(models[i].location, dtype=float)
                    )
                )
                for i in member_indices
                for enemy in enemies
            )
            move = float(min(speeds[i] for i in member_indices)) if speeds.size else 0.0
            reach = min(move, float(models[member_indices[0]].charge_roll))
            if gap - contact <= reach:
                reachable.add(group_id)
        return reachable

    def select_charge(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Charge as a rigid body when the squad can actually reach contact.

        Three properties, each of them measured rather than assumed:

        - **Rigid translation.** Every member takes the same action, so relative
          positions — and therefore coherency — are preserved exactly. A
          per-model greedy charge stretches the squad, and a stretched squad
          both fails the coherency clause and clips a second enemy unit; both
          revert the whole charge.
        - **Only when it lands.** The step is sized to put the nearest member
          just inside engagement range, and the squad declines unless a legal
          rung covers it. This is the charge's analogue of `advance_to_arrive`:
          control is a headcount at the scoring moment, so being nearer buys
          nothing and being THERE buys everything.
        - **The mask decides legality, not this rule.** `charge_legality`
          already encodes eligibility (within declaration range, not already
          engaged, has not advanced or fallen back) and the 2D6 cap. Recomputing
          any of that here would be a second source of truth for the rules.

        ⚠ A declined or failed charge costs nothing: this phase runs after
        shooting, and the referee reverts an illegal charge to where the unit
        began. So the rule can afford to try. What a SUCCESSFUL charge costs is
        the next turn — an engaged unit cannot shoot, and cannot be shot at.
        That second half is the whole of the charge's measured value.
        """
        actions = [STAY_ACTION] * len(models)
        if not self.charge_when_it_lands or not env.config.melee.enabled:
            return WargameEnvAction(actions=actions)
        handler = env.player_action_handler
        legality = env.player_charge_legality
        movement = handler.movement_slice
        speeds = handler.move_speeds
        # Base to base, as the engagement predicate itself measures it.
        contact = float(env.rules_quantities.engagement_range) + 2.0 * float(
            env.rules_quantities.base_radius
        )

        for group_id in sorted({model.group_id for model in models}):
            member_indices = [
                i
                for i, model in enumerate(models)
                if model.group_id == group_id and model.is_alive
            ]
            # A unit with no legal rung is ineligible -- out of declaration
            # range, already engaged, or it advanced or fell back this turn.
            if not member_indices or not legality[member_indices].any():
                continue
            # ⚠ **Decline if the unit is ALREADY out of coherency.** The referee
            # judges coherency at the END of the charge, and a rigid translation
            # preserves formation exactly -- which is precisely why a squad that
            # was broken when it declared is still broken when it lands, and the
            # whole charge reverts.
            #
            # Measured: 82.2% of this policy's incoherent charge failures were on
            # units already incoherent BEFORE the move, and only 8 of 135 moved
            # charges were broken BY the move. There is also a selection effect
            # pulling the wrong way -- a stretched squad's nearest member is
            # nearer the enemy, so a broken unit looks MORE chargeable.
            #
            # ⚠ I published the opposite ("rigid translation preserves formation
            # exactly, so this is the RESOLVER -- do not attempt a fourth
            # movement-side fix") and an audit panel refuted it with four
            # independent probes. The inference was backwards, and the fix is
            # policy-side, so the movement prohibition never applied to it.
            if not evaluate_coherency(
                positions=np.array(
                    [models[i].location for i in member_indices], dtype=float
                ),
                group_ids=np.zeros(len(member_indices), dtype=np.intp),
                alive_mask=np.ones(len(member_indices), dtype=bool),
                base_radii=np.array(
                    [models[i].base_radius for i in member_indices], dtype=float
                ),
                nearest_distance=env.config.coherency.nearest_distance,
                furthest_distance=env.config.coherency.furthest_distance,
            ).all_coherent:
                continue
            pair = self._charge_target(models, member_indices, env)
            if pair is None:
                continue
            lead_index, enemy_index = pair
            lead = np.asarray(
                env.opponent_models[enemy_index].location, dtype=float
            ) - np.asarray(models[lead_index].location, dtype=float)
            gap = float(np.linalg.norm(lead))
            # Capped by the squad's slowest member, as the march is: a member
            # that cannot cover the shared step is left behind, and a stretched
            # squad fails the coherency clause the referee applies to all of it.
            # ⚠ **`min`, not `+`.** A charge is capped by the 2D6 (the mask) AND
            # by the movement ladder, whose longest rung is the model's Move
            # (`DEFERRED: charge.beyond_move_ladder`) -- the roll exceeds Move on
            # 59.1% of declarations, so it is usually Move that binds. Reading
            # this as `Move + roll`, the advance's rule, made the policy declare
            # charges from twice the distance it could cover and measured
            # **55.9%** of moved charges touching nobody.
            move = float(min(speeds[i] for i in member_indices)) if speeds.size else 0.0
            reach = min(move, float(models[lead_index].charge_roll))
            # The shortest step that can make contact. Below it the charge
            # cannot stand however the rungs fall, so the unit declines rather
            # than spending a declaration on it.
            if gap - contact > reach:
                continue
            # ⚠ **Aim THROUGH the enemy, not short of it**, and let its base
            # stop the move. Enemy bases are blockers in `resolve_move`, so a
            # step aimed at the enemy's own position comes to rest at base
            # contact -- inside engagement range by construction.
            #
            # Aiming at `gap - contact/2` instead, the obvious reading of "stop
            # just inside", measured **44.8%** of moved charges touching NOBODY:
            # `best_action_toward` rounds DOWN to a rung, so the deliberate
            # margin was spent on rounding, and anything the resolver deflected
            # fell short. Letting the blocker decide removes that whole class.
            needed = min(gap, reach)
            if needed <= 0.0:
                continue
            action = handler.best_action_toward(
                float(lead[0]),
                float(lead[1]),
                max_step_length=needed,
                model_idx=lead_index,
            )
            offset = action - movement.start
            if action == STAY_ACTION or offset < 0 or offset >= movement.size:
                continue
            # The mask is the authority. A rung this rule likes but the rules
            # forbid is not a charge, and emitting it would put an illegal
            # action past a policy that is meant to be a reference.
            if not all(legality[i][offset] for i in member_indices):
                continue
            for i in member_indices:
                actions[i] = action
        return WargameEnvAction(actions=actions)

    def select_command(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Declare each squad's move type, from its leader's action."""
        move_type = env.player_action_handler.move_type_slice
        actions = [STAY_ACTION] * len(models)
        # STAY already declares `normal`, so only an ADVANCE needs an action --
        # which also means a DARKENED slice (registered, valid in no phase, so
        # masked) is handled by emitting nothing rather than by a special case.
        if move_type is None or BattlePhase.command not in move_type.valid_phases:
            return WargameEnvAction(actions=actions)
        decisions = self._squad_advance_decisions(models, env)
        handler = env.player_action_handler
        advance = handler.move_type_action(MOVE_TYPE_ADVANCE)
        charge = handler.move_type_action(MOVE_TYPE_CHARGE)
        leaders: dict[int, int] = {}
        for index, model in enumerate(models):
            if model.is_alive:
                leaders.setdefault(int(model.group_id), index)
        # ⚠ A unit declares ONE move type, so advance wins where both apply --
        # it is the older behaviour and the one every measured baseline was
        # taken under. A charge declaration is only reached by a unit that was
        # not going to advance anyway.
        charging = self._squad_charge_declarations(models, env) if charge else {}
        for group_id, leader in leaders.items():
            if decisions.get(group_id) and advance is not None:
                actions[leader] = advance
            elif charging.get(group_id) and charge is not None:
                actions[leader] = charge
        return WargameEnvAction(actions=actions)

    def _squad_charge_declarations(
        self, models: list[WargameModel], env: WargameEnv
    ) -> dict[int, bool]:
        """Which squads would declare a charge this turn, per `select_charge`.

        Asks the same question the charge itself asks, so a unit never declares
        a charge it will then decline to make -- a declaration makes STAY
        illegal for its members, so declaring and not charging would force the
        unit into whatever rung the mask left rather than into no move at all.
        """
        if not self.charge_when_it_lands or not env.config.melee.enabled:
            return {}
        return {
            group_id: True for group_id in self._reachable_charge_units(models, env)
        }

    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """March each squad toward its objective, settling onto the disc on arrival."""
        objectives = env.objectives
        actions = [STAY_ACTION] * len(models)
        if not objectives:
            return WargameEnvAction(actions=actions)

        # Per SQUAD, and that distinction is load-bearing. A squad marches as a
        # body on one shared vector -- that is what keeps its formation rigid
        # and therefore legal -- so its step is capped by its own slowest
        # member, since a member that cannot cover the shared step is left
        # behind and breaks the property this policy relies on.
        #
        # Taking the minimum over the whole ARMY instead is identical while
        # every model is equally fast, and silently wrong the moment they are
        # not: one slow squad would cap a fast one, so a scripted bar could not
        # use a speed a learned policy can. That flatters the agent against a
        # hobbled bar, which is the most expensive class of error here.
        speeds = env.player_action_handler.move_speeds
        group_ids = sorted({model.group_id for model in models})
        targets = self.squad_objectives(models, env, group_ids)

        for squad_index, group_id in enumerate(group_ids):
            member_indices = [
                i
                for i, model in enumerate(models)
                if model.group_id == group_id and model.is_alive
            ]
            if not member_indices:
                continue
            max_step = float(
                min(speeds[i] for i in member_indices) if speeds.size else 0.0
            )

            objective = targets[squad_index]
            radius = objective_extent(objective)
            centroid = np.mean(
                [models[i].location for i in member_indices], axis=0, dtype=float
            )
            lead = np.asarray(objective.location, dtype=float) - centroid
            lead_distance = float(np.linalg.norm(lead))

            # READ the declaration rather than recompute it. The move type was
            # chosen in the command phase and the env is holding the squad to
            # it; deciding again here could disagree with what was declared, and
            # a long rung is masked for a squad that declared a normal move.
            squad_advances = bool(models[member_indices[0]].declared_advance)

            for i in member_indices:
                if lead_distance <= radius:
                    # The squad has arrived; each model settles onto the disc
                    # individually so the whole body ends up inside it.
                    actions[i] = step_toward_objective(models[i], objective, env, i)
                else:
                    # Every model follows the same squad vector, which keeps
                    # relative positions — and therefore coherency — intact.
                    step = min(max_step, lead_distance)
                    advance = (
                        env.player_action_handler.best_advance_toward(
                            float(lead[0]),
                            float(lead[1]),
                            advance_roll=float(models[i].advance_roll),
                            max_step_length=min(
                                lead_distance, max_step + float(models[i].advance_roll)
                            ),
                            model_idx=i,
                        )
                        if squad_advances
                        else None
                    )
                    actions[i] = (
                        advance
                        if advance is not None
                        else env.player_action_handler.best_action_toward(
                            float(lead[0]),
                            float(lead[1]),
                            max_step_length=step,
                            model_idx=i,
                        )
                    )

        return WargameEnvAction(actions=actions)


register_baseline("squad_march", ScriptedSquadMarchPolicy)
