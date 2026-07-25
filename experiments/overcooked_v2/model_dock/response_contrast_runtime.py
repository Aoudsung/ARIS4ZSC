"""Device runtime for the matched decision-response deployment contrast."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from src.path_c.belief.update import likelihood_for_token, log_bayes_update
from src.path_c.evaluation.standard import PopulationManifest, StandardPairing
from src.path_c.probe.candidates import enumerate_candidates
from src.path_c.probe.controllers import select_action
from src.path_c.probe.scores import response_information_scores, sequential_scores

from .env_dock import visible_goal_safe_action_mask
from .formal_evaluation_runtime import (
    FormalPopulationRuntime,
    _SideState,
    _categorical_by_row,
    _split_row_keys,
)
from .response_dock import response_tokens


class _DecisionDetails(NamedTuple):
    action: Any
    base_action: Any
    reference_action: Any
    state: _SideState
    probed: Any
    reference_selected_candidate: Any
    opportunity: Any
    probabilities: Any
    reference_probabilities: Any


class _BranchState(NamedTuple):
    environment_state: Any
    observations: Any
    left: _SideState
    right: _SideState
    episode_start: Any
    episode_step: Any
    raw_return: Any
    probe_count: Any
    safe_candidate_opportunity_count: Any
    correct_delivery_count: Any
    wrong_delivery_count: Any
    indicator_cost: Any


class _ContrastState(NamedTuple):
    a1: _BranchState
    masked: _BranchState
    used: _BranchState
    triggered: Any
    first_probe_step: Any
    executed_environment_steps: Any
    unique_environment_steps: Any
    random_keys: Any


def _tree_select(mask: Any, selected: Any, alternative: Any) -> Any:
    import jax
    import jax.numpy as jnp

    boolean = jnp.asarray(mask, dtype=jnp.bool_)

    def choose(selected_leaf: Any, alternative_leaf: Any) -> Any:
        selected_value = jnp.asarray(selected_leaf)
        shape = boolean.shape + (1,) * (selected_value.ndim - boolean.ndim)
        return jnp.where(boolean.reshape(shape), selected_leaf, alternative_leaf)

    return jax.tree_util.tree_map(choose, selected, alternative)


def _probabilities_for_action(probabilities: Any, action: Any) -> Any:
    import jax.numpy as jnp

    index = jnp.asarray(action, dtype=jnp.int32)[:, None, None, None]
    return jnp.take_along_axis(
        probabilities,
        jnp.broadcast_to(
            index,
            (
                probabilities.shape[0],
                probabilities.shape[1],
                1,
                probabilities.shape[-1],
            ),
        ),
        axis=2,
    )[:, :, 0, :]


class ResponseContrastRuntime:
    """Evaluate A1, A2-mask, and A2-use from one coupled XP prefix."""

    def __init__(self, manifest: PopulationManifest) -> None:
        if (
            manifest.population_id != "decision_focused"
            or manifest.condition_id != "decision_focused"
            or manifest.controller != "registered_response_sequential_branch_v1"
            or manifest.source_type != "adaptation_checkpoint"
        ):
            raise ValueError(
                "The response contrast requires the formal decision-focused population."
            )
        self.population = FormalPopulationRuntime(manifest)
        self.manifest = manifest
        self.config = self.population.config
        self.model = self.population.model
        if self.config is None or self.model is None or not self.config.is_family_pool:
            raise ValueError(
                "The response contrast requires the family-level Path C model."
            )
        self._compiled = self._compile()

    def _decision_details(
        self,
        *,
        params: Mapping[str, Any],
        calibration: Any,
        side: _SideState,
        observations: Any,
        episode_start: Any,
        episode_step: Any,
        random_keys: Any,
    ) -> _DecisionDetails:
        import jax
        import jax.numpy as jnp

        split_keys = _split_row_keys(random_keys, 2)
        next_carry, output = self.model.apply(
            {"params": params},
            side.carry,
            observations[None, ...],
            episode_start[None, ...],
        )
        output = jax.tree_util.tree_map(lambda value: value[0], output)
        base_action = _categorical_by_row(split_keys[:, 0], output["actor_logits"])
        candidates = enumerate_candidates(
            safe_action_mask=visible_goal_safe_action_mask(
                observations, action_order=self.population.action_order
            ),
            base_action=base_action,
            episode_step=episode_step,
            budget_remaining=side.budget_remaining,
            candidate_window=self.config.probe.candidate_window,
        )
        sequential = sequential_scores(
            log_belief=side.log_belief,
            response_probabilities=output["transition_response_probabilities"],
            reward_estimates=output["reward_estimates"],
            next_values=output["next_values"],
            base_action=base_action,
            candidate_mask=candidates.mask,
            gamma=self.config.adaptation.gamma,
            probability_floor=self.config.probe.belief_probability_floor,
        )
        information = response_information_scores(
            log_belief=side.log_belief,
            response_probabilities=output["transition_response_probabilities"],
            probability_floor=self.config.probe.belief_probability_floor,
        )
        decision = select_action(
            self.config.controller,
            key=split_keys[:, 1],
            sequential=sequential,
            information_scores=information,
            candidate_mask=candidates.mask,
            base_action=base_action,
            budget_remaining=side.budget_remaining,
            decision_threshold=calibration[0],
            information_threshold=calibration[1],
            random_trigger_probability=calibration[2],
        )
        safe_values = jnp.where(candidates.mask, sequential.j_mask, -jnp.inf)
        best_candidate = jnp.argmax(safe_values, axis=-1)
        best_candidate_value = jnp.max(safe_values, axis=-1)
        reference_selected_candidate = (
            candidates.any_candidate
            & (side.budget_remaining > 0)
            & (best_candidate_value > sequential.v_base)
        )
        reference_action = jnp.where(
            reference_selected_candidate, best_candidate, base_action
        )
        return _DecisionDetails(
            action=decision.chosen_action,
            base_action=base_action,
            reference_action=reference_action,
            state=_SideState(
                carry=next_carry,
                log_belief=side.log_belief,
                budget_remaining=decision.budget_remaining,
            ),
            probed=decision.probed,
            reference_selected_candidate=reference_selected_candidate,
            opportunity=candidates.any_candidate,
            probabilities=_probabilities_for_action(
                output["response_probabilities"], decision.chosen_action
            ),
            reference_probabilities=_probabilities_for_action(
                output["response_probabilities"], reference_action
            ),
        )

    def _compile(self) -> Any:
        import jax
        import jax.numpy as jnp

        count = self.manifest.episodes_per_pairing
        probability_floor = self.config.probe.belief_probability_floor
        budget = self.config.probe.budget_per_episode
        environment = self.population.environment

        def initial_side() -> _SideState:
            from src.path_c.belief.update import uniform_log_belief

            return _SideState(
                carry=self.model.initial_carry(count),
                log_belief=uniform_log_belief(count, 4),
                budget_remaining=jnp.full((count,), budget, dtype=jnp.int32),
            )

        def run(
            left_params: Mapping[str, Any],
            right_params: Mapping[str, Any],
            left_calibration: Any,
            right_calibration: Any,
            episode_seeds: Any,
        ) -> _ContrastState:
            root_keys = jax.vmap(jax.random.PRNGKey)(episode_seeds)
            initial_keys = _split_row_keys(root_keys, 2)
            environment_state, observations = environment.reset_with_keys(
                initial_keys[:, 0]
            )

            def initial_branch() -> _BranchState:
                return _BranchState(
                    environment_state=environment_state,
                    observations=observations,
                    left=initial_side(),
                    right=initial_side(),
                    episode_start=jnp.ones((count,), dtype=jnp.bool_),
                    episode_step=jnp.zeros((count,), dtype=jnp.int32),
                    raw_return=jnp.zeros((count,), dtype=jnp.float32),
                    probe_count=jnp.zeros((count,), dtype=jnp.int32),
                    safe_candidate_opportunity_count=jnp.zeros(
                        (count,), dtype=jnp.int32
                    ),
                    correct_delivery_count=jnp.zeros((count,), dtype=jnp.int32),
                    wrong_delivery_count=jnp.zeros((count,), dtype=jnp.int32),
                    indicator_cost=jnp.zeros((count,), dtype=jnp.float32),
                )

            branch = initial_branch()
            initial = _ContrastState(
                a1=branch,
                masked=branch,
                used=branch,
                triggered=jnp.zeros((count,), dtype=jnp.bool_),
                first_probe_step=jnp.full((count,), -1, dtype=jnp.int32),
                executed_environment_steps=jnp.zeros((count,), dtype=jnp.int32),
                unique_environment_steps=jnp.zeros((count,), dtype=jnp.int32),
                random_keys=initial_keys[:, 1],
            )

            def advance(
                current: _BranchState,
                *,
                left_details: _DecisionDetails,
                right_details: _DecisionDetails,
                left_action: Any,
                left_probabilities: Any,
                left_probed: Any,
                skip_left_response: Any,
                step_keys: Any,
            ) -> _BranchState:
                actions = jnp.stack((left_action, right_details.action), axis=-1)
                next_environment, next_observations, reward, done, info = (
                    environment.step_with_keys(
                        current.environment_state, actions, step_keys
                    )
                )
                left_token = response_tokens(
                    current.observations[:, 0, ...],
                    next_observations[:, 0, ...],
                    done,
                    info,
                )
                right_token = response_tokens(
                    current.observations[:, 1, ...],
                    next_observations[:, 1, ...],
                    done,
                    info,
                )
                left_updated = log_bayes_update(
                    left_details.state.log_belief,
                    likelihood_for_token(left_probabilities, left_token),
                    probability_floor=probability_floor,
                )
                left_belief = jnp.where(
                    skip_left_response[:, None],
                    left_details.state.log_belief,
                    left_updated,
                )
                right_belief = log_bayes_update(
                    right_details.state.log_belief,
                    likelihood_for_token(
                        right_details.probabilities, right_token
                    ),
                    probability_floor=probability_floor,
                )
                return _BranchState(
                    environment_state=next_environment,
                    observations=next_observations,
                    left=left_details.state._replace(log_belief=left_belief),
                    right=right_details.state._replace(log_belief=right_belief),
                    episode_start=done,
                    episode_step=current.episode_step + 1,
                    raw_return=current.raw_return + reward,
                    probe_count=(
                        current.probe_count
                        + left_probed.astype(jnp.int32)
                        + right_details.probed.astype(jnp.int32)
                    ),
                    safe_candidate_opportunity_count=(
                        current.safe_candidate_opportunity_count
                        + left_details.opportunity.astype(jnp.int32)
                        + right_details.opportunity.astype(jnp.int32)
                    ),
                    correct_delivery_count=(
                        current.correct_delivery_count + info["correct_delivery"]
                    ),
                    wrong_delivery_count=(
                        current.wrong_delivery_count + info["wrong_delivery"]
                    ),
                    indicator_cost=current.indicator_cost + info["indicator_cost"],
                )

            def scan_step(
                current: _ContrastState, unused: Any
            ) -> tuple[_ContrastState, None]:
                del unused
                keys = _split_row_keys(current.random_keys, 4)
                pre_trigger = ~current.triggered
                a1_current = _tree_select(pre_trigger, current.used, current.a1)
                mask_current = _tree_select(
                    pre_trigger, current.used, current.masked
                )

                def details(branch: _BranchState) -> tuple[_DecisionDetails, _DecisionDetails]:
                    return (
                        self._decision_details(
                            params=left_params,
                            calibration=left_calibration,
                            side=branch.left,
                            observations=branch.observations[:, 0, ...],
                            episode_start=branch.episode_start,
                            episode_step=branch.episode_step,
                            random_keys=keys[:, 1],
                        ),
                        self._decision_details(
                            params=right_params,
                            calibration=right_calibration,
                            side=branch.right,
                            observations=branch.observations[:, 1, ...],
                            episode_start=branch.episode_start,
                            episode_step=branch.episode_step,
                            random_keys=keys[:, 2],
                        ),
                    )

                a1_left, a1_right = details(a1_current)
                mask_left, mask_right = details(mask_current)
                use_left, use_right = details(current.used)
                trigger_now = pre_trigger & use_left.probed

                a1_left_state = _tree_select(
                    pre_trigger, use_left.state, a1_left.state
                )
                a1_budget = jnp.maximum(
                    current.used.left.budget_remaining
                    - use_left.reference_selected_candidate.astype(jnp.int32),
                    0,
                )
                a1_left_state = a1_left_state._replace(
                    budget_remaining=jnp.where(
                        trigger_now,
                        a1_budget,
                        a1_left_state.budget_remaining,
                    )
                )
                a1_left = a1_left._replace(
                    state=a1_left_state,
                    opportunity=jnp.where(
                        pre_trigger, use_left.opportunity, a1_left.opportunity
                    ),
                )
                mask_left = mask_left._replace(
                    state=_tree_select(
                        pre_trigger, use_left.state, mask_left.state
                    ),
                    opportunity=jnp.where(
                        pre_trigger, use_left.opportunity, mask_left.opportunity
                    ),
                )
                a1_right = a1_right._replace(
                    state=_tree_select(
                        pre_trigger, use_right.state, a1_right.state
                    ),
                    opportunity=jnp.where(
                        pre_trigger, use_right.opportunity, a1_right.opportunity
                    ),
                    probabilities=jnp.where(
                        pre_trigger[:, None, None],
                        use_right.probabilities,
                        a1_right.probabilities,
                    ),
                    action=jnp.where(
                        pre_trigger, use_right.action, a1_right.action
                    ),
                    probed=jnp.where(
                        pre_trigger, use_right.probed, a1_right.probed
                    ),
                )
                mask_right = mask_right._replace(
                    state=_tree_select(
                        pre_trigger, use_right.state, mask_right.state
                    ),
                    opportunity=jnp.where(
                        pre_trigger, use_right.opportunity, mask_right.opportunity
                    ),
                    probabilities=jnp.where(
                        pre_trigger[:, None, None],
                        use_right.probabilities,
                        mask_right.probabilities,
                    ),
                    action=jnp.where(
                        pre_trigger, use_right.action, mask_right.action
                    ),
                    probed=jnp.where(
                        pre_trigger, use_right.probed, mask_right.probed
                    ),
                )

                a1_action = jnp.where(
                    trigger_now,
                    use_left.reference_action,
                    jnp.where(pre_trigger, use_left.action, a1_left.action),
                )
                a1_probabilities = jnp.where(
                    trigger_now[:, None, None],
                    use_left.reference_probabilities,
                    jnp.where(
                        pre_trigger[:, None, None],
                        use_left.probabilities,
                        a1_left.probabilities,
                    ),
                )
                a1_probed = jnp.where(
                    trigger_now,
                    use_left.reference_selected_candidate,
                    jnp.where(pre_trigger, use_left.probed, a1_left.probed),
                )
                mask_action = jnp.where(
                    pre_trigger, use_left.action, mask_left.action
                )
                mask_probabilities = jnp.where(
                    pre_trigger[:, None, None],
                    use_left.probabilities,
                    mask_left.probabilities,
                )
                mask_probed = jnp.where(
                    pre_trigger, use_left.probed, mask_left.probed
                )

                next_a1 = advance(
                    a1_current,
                    left_details=a1_left,
                    right_details=a1_right,
                    left_action=a1_action,
                    left_probabilities=a1_probabilities,
                    left_probed=a1_probed,
                    skip_left_response=trigger_now,
                    step_keys=keys[:, 3],
                )
                next_masked = advance(
                    mask_current,
                    left_details=mask_left,
                    right_details=mask_right,
                    left_action=mask_action,
                    left_probabilities=mask_probabilities,
                    left_probed=mask_probed,
                    skip_left_response=trigger_now,
                    step_keys=keys[:, 3],
                )
                next_used = advance(
                    current.used,
                    left_details=use_left,
                    right_details=use_right,
                    left_action=use_left.action,
                    left_probabilities=use_left.probabilities,
                    left_probed=use_left.probed,
                    skip_left_response=jnp.zeros_like(trigger_now),
                    step_keys=keys[:, 3],
                )
                triggered = current.triggered | trigger_now
                still_common = ~triggered
                next_a1 = _tree_select(still_common, next_used, next_a1)
                next_masked = _tree_select(still_common, next_used, next_masked)
                unique_increment = jnp.where(
                    current.triggered,
                    3,
                    jnp.where(trigger_now, 2, 1),
                )
                return _ContrastState(
                    a1=next_a1,
                    masked=next_masked,
                    used=next_used,
                    triggered=triggered,
                    first_probe_step=jnp.where(
                        trigger_now,
                        current.used.episode_step,
                        current.first_probe_step,
                    ),
                    executed_environment_steps=(
                        current.executed_environment_steps + 3
                    ),
                    unique_environment_steps=(
                        current.unique_environment_steps + unique_increment
                    ),
                    random_keys=keys[:, 0],
                ), None

            return jax.lax.scan(
                scan_step,
                initial,
                xs=None,
                length=self.config.environment.episode_steps,
            )[0]

        return jax.jit(run)

    def evaluate_pairing(
        self, pairing: StandardPairing, episode_seeds: Sequence[int]
    ) -> Sequence[Mapping[str, Any]]:
        import jax.numpy as jnp

        if pairing.split != "xp":
            raise ValueError("The response contrast accepts only cross-play pairings.")
        if len(episode_seeds) != self.manifest.episodes_per_pairing:
            raise ValueError("The response contrast requires exactly 500 seeds.")
        left = self.population.policies[pairing.outer_unit_0]
        right = self.population.policies[pairing.outer_unit_1]
        if left.calibration is None or right.calibration is None:
            raise ValueError("The response contrast requires deployment calibrations.")
        final = self._compiled(
            left.params,
            right.params,
            jnp.asarray(left.calibration),
            jnp.asarray(right.calibration),
            jnp.asarray(np.asarray(episode_seeds, dtype=np.uint32)),
        )
        if not (
            bool(np.asarray(final.a1.episode_start).all())
            and bool(np.asarray(final.masked.episode_start).all())
            and bool(np.asarray(final.used.episode_start).all())
        ):
            raise RuntimeError("A response-contrast cell did not finish all episodes.")

        def arrays(branch: _BranchState) -> Mapping[str, np.ndarray]:
            return {
                "raw_return": np.asarray(branch.raw_return),
                "correct_delivery_count": np.asarray(branch.correct_delivery_count),
                "wrong_delivery_count": np.asarray(branch.wrong_delivery_count),
                "probe_count": np.asarray(branch.probe_count),
                "safe_candidate_opportunity_count": np.asarray(
                    branch.safe_candidate_opportunity_count
                ),
                "indicator_cost": np.asarray(branch.indicator_cost),
            }

        branch_arrays = {
            "A1": arrays(final.a1),
            "A2-mask": arrays(final.masked),
            "A2-use": arrays(final.used),
        }
        triggered = np.asarray(final.triggered)
        first_probe = np.asarray(final.first_probe_step)
        executed_steps = np.asarray(final.executed_environment_steps)
        unique_steps = np.asarray(final.unique_environment_steps)
        maximum_budget = 2 * int(self.config.probe.budget_per_episode)
        results = []
        for index in range(self.manifest.episodes_per_pairing):
            branches = {}
            for branch, values in branch_arrays.items():
                branches[branch] = {
                    "raw_return": float(values["raw_return"][index]),
                    "correct_delivery_count": int(
                        values["correct_delivery_count"][index]
                    ),
                    "wrong_delivery_count": int(
                        values["wrong_delivery_count"][index]
                    ),
                    "probe_count": int(values["probe_count"][index]),
                    "safe_candidate_opportunity_count": int(
                        values["safe_candidate_opportunity_count"][index]
                    ),
                    "maximum_probe_budget": maximum_budget,
                    "indicator_cost": float(values["indicator_cost"][index]),
                }
            results.append(
                {
                    "triggered": bool(triggered[index]),
                    "first_probe_step": (
                        int(first_probe[index]) if triggered[index] else None
                    ),
                    "executed_environment_steps": int(executed_steps[index]),
                    "unique_environment_steps": int(unique_steps[index]),
                    "branches": branches,
                }
            )
        return results


__all__ = ["ResponseContrastRuntime"]
