"""Accumulates unit coherency over one episode, for the eval metrics.

Coherency has been measurable since `just measure-coherency`, but only offline,
against a finished checkpoint. That is the wrong place for it while a run is
*training under the rule*: a run enforcing coherent deployment can drift out of
formation for a thousand epochs and nothing in the dashboard would say so.

Sampled once per movement phase, because movement is the only thing that changes
formation -- shooting changes it only through casualties, which the rule
deliberately does not blame on the unit.

**Read the two numbers together.** `coherency_rate` alone is confounded with
squad size: a unit shot down to one model is coherent by definition, so a rising
rate can mean the units died rather than that they held together.
`models_out_of_coherency` does not have that failure mode, since a dead model
contributes nothing to it.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency


class CoherencyTracker:
    """Running totals for one force over one episode."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear accumulated counts. Called at the start of each episode."""
        self._units = 0
        self._units_coherent = 0
        self._models_out = 0
        self._samples = 0
        # The same three counts for the move as PROPOSED, before enforcement.
        self._intent_units = 0
        self._intent_units_coherent = 0
        self._intent_models_out = 0
        self._intent_samples = 0

    def record(
        self,
        positions: np.ndarray,
        group_ids: np.ndarray,
        alive_mask: np.ndarray,
        base_radii: np.ndarray,
        nearest_distance: float,
        furthest_distance: float,
    ) -> None:
        """Fold one phase's formation into the totals.

        Args:
            positions: (n, 2) model locations.
            group_ids: (n,) unit id per model.
            alive_mask: (n,) True for models still on the board.
            base_radii: (n,) base radius per model, for base-to-base distance.
            nearest_distance: The chain distance, in board units.
            furthest_distance: The spread distance, in board units.
        """
        report = evaluate_coherency(
            positions=positions,
            group_ids=group_ids,
            alive_mask=alive_mask,
            base_radii=base_radii,
            nearest_distance=nearest_distance,
            furthest_distance=furthest_distance,
        )
        if not report.units:
            return
        self._samples += 1
        for unit in report.units:
            self._units += 1
            if unit.coherent:
                self._units_coherent += 1
            else:
                # `member_coherency` is the rule's own per-model answer.
                # Counting `size - largest_component_size` instead measured the
                # CHAIN graph only, so a unit fully connected but overrunning
                # the 9" spread cap has one component and recorded **zero**
                # models adrift while being 100% in breach -- a six-model line
                # at 2.0 spacing spans 10.0 against the cap, has 2 models truly
                # adrift, and reported 0. `just measure-coherency` used the
                # right definition throughout, so the two disagreed.
                self._models_out += int((~unit.member_coherency).sum())

    def record_intent(self, counts: tuple[int, int, int] | None) -> None:
        """Fold one phase's PROPOSED formation into the intent totals.

        `counts` is `(units, units_coherent, models_out)` judged before
        enforcement edited anything, so these metrics answer "did the policy
        choose a legal move" where the others answer "is the board legal". Under
        enforcement the two diverge completely, and reporting only the second is
        how a policy intending 0.630 coherency was published at 1.000.
        """
        if counts is None:
            return
        units, coherent, models_out = counts
        if units == 0:
            return
        self._intent_samples += 1
        self._intent_units += units
        self._intent_units_coherent += coherent
        self._intent_models_out += models_out

    @property
    def intended_coherency_rate(self) -> float | None:
        """Share of unit-samples the POLICY put in coherency, before the revert."""
        if not self._intent_units:
            return None
        return self._intent_units_coherent / self._intent_units

    @property
    def intended_models_out_of_coherency(self) -> float | None:
        """Models per phase the policy left out of coherency, before the revert."""
        if not self._intent_samples:
            return None
        return self._intent_models_out / self._intent_samples

    @property
    def coherency_rate(self) -> float | None:
        """Share of unit-samples in coherency, None before anything is sampled."""
        if not self._units:
            return None
        return self._units_coherent / self._units

    @property
    def models_out_of_coherency(self) -> float | None:
        """Mean models outside their unit's coherent body, per sampled phase."""
        if not self._samples:
            return None
        return self._models_out / self._samples
