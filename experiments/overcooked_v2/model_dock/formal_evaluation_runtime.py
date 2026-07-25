"""Device-side runtime for one standard ten-policy Path C population."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from src.path_c.belief.update import (
    likelihood_for_token,
    log_bayes_update,
    uniform_log_belief,
)
from src.path_c.evaluation.standard import PopulationManifest, StandardPairing
from src.path_c.model.adaptation_model import build_model
from src.path_c.model.checkpoint import load_checkpoint
from src.path_c.probe.candidates import enumerate_candidates
from src.path_c.probe.controllers import select_action
from src.path_c.probe.scores import response_information_scores, sequential_scores

from .env_dock import OvercookedV2VectorEnvironment, visible_goal_safe_action_mask
from .official_dock import OfficialBackboneDock
from .response_dock import registered_response_vocabulary, response_tokens


class _SideState(NamedTuple):
    carry: Any
    log_belief: Any
    budget_remaining: Any


class _EpisodeState(NamedTuple):
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
    random_keys: Any


def _split_row_keys(keys: Any, count: int) -> Any:
    import jax
    import jax.numpy as jnp

    values = jnp.asarray(keys)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Formal evaluation requires one legacy JAX key per episode row.")
    return jax.vmap(lambda value: jax.random.split(value, count))(values)


def _categorical_by_row(keys: Any, logits: Any) -> Any:
    import jax

    return jax.vmap(
        lambda key, row: jax.random.categorical(key, row, axis=-1)
    )(keys, logits)


def _calibration_vector(summary: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        (
            float(summary["decision_threshold"]),
            float(summary["information_threshold"]),
            float(summary["random_trigger_probability"]),
        ),
        dtype=np.float32,
    )


@dataclass(frozen=True)
class _LoadedPolicy:
    params: Mapping[str, Any]
    calibration: np.ndarray | None


class FormalPopulationRuntime:
    """Load ten policies and evaluate ordered cells with independent side state."""

    def __init__(self, manifest: PopulationManifest) -> None:
        self.manifest = manifest
        self.environment = OvercookedV2VectorEnvironment.create(
            num_envs=manifest.episodes_per_pairing,
            layout=manifest.layout,
        )
        first_unit = manifest.outer_units.units[0]
        self.dock = OfficialBackboneDock.from_launch_config(
            first_unit.backbone.launch_config_path
        )
        self.action_order = tuple(self.dock.action_order())
        self.vocabulary = registered_response_vocabulary()
        self._validate_official_network_contracts()
        if manifest.source_type == "adaptation_checkpoint":
            first_config = manifest.policies[0].resolved_config
            if first_config is None:
                raise RuntimeError("An adapted population has no resolved model config.")
            self.config = first_config
            self.model = build_model(
                num_prototypes=4,
                action_count=len(self.action_order),
                response_count=self.vocabulary.size,
                official_dimensions=self.dock.network_dimensions(),
                model_config=first_config.model,
            )
            self._validate_adapted_population_contracts()
            loaded = []
            for entry in manifest.policies:
                params, metadata, unused_manifest = load_checkpoint(entry.checkpoint_path)
                del unused_manifest
                if (
                    metadata.stage != "adaptation"
                    or metadata.environment_steps != 10_000_000
                    or metadata.condition_id != manifest.condition_id
                    or metadata.controller != manifest.controller
                    or metadata.response_vocabulary_sha256 != self.vocabulary.sha256
                    or entry.resolved_config is None
                    or metadata.config_sha256 != entry.resolved_config.config_sha256
                ):
                    raise ValueError(
                        "A formal adapted policy checkpoint changed its model, vocabulary, "
                        "condition, or 10,000,000-step contract."
                    )
                if entry.calibration_summary is None:
                    raise ValueError("An adapted policy has no bound calibration summary.")
                loaded.append(
                    _LoadedPolicy(
                        params=params,
                        calibration=_calibration_vector(entry.calibration_summary),
                    )
                )
            self.policies = tuple(loaded)
            self._compiled = self._compile_adapted_evaluator()
        else:
            self.config = None
            self.model = None
            loaded = []
            for entry, unit in zip(
                manifest.policies, manifest.outer_units.units, strict=True
            ):
                unit_dock = OfficialBackboneDock.from_launch_config(
                    unit.backbone.launch_config_path
                )
                if (
                    tuple(unit_dock.action_order()) != self.action_order
                    or unit_dock.network_dimensions() != self.dock.network_dimensions()
                ):
                    raise ValueError(
                        "Official backbone policies do not share one registered network contract."
                    )
                if entry.checkpoint_path != unit.backbone.checkpoint_path:
                    raise ValueError("An official backbone entry changed its outer-unit source.")
                loaded.append(
                    _LoadedPolicy(
                        params=unit_dock.official_parameter_tree(unit.backbone),
                        calibration=None,
                    )
                )
            self.policies = tuple(loaded)
            self.official_network_adapter = self.dock.frozen_policy(
                self.policies[0].params, policy_id="formal_standard_adapter"
            ).network_adapter
            self._compiled = self._compile_backbone_evaluator()

    def _validate_official_network_contracts(self) -> None:
        if self.action_order != (
            "right",
            "down",
            "left",
            "up",
            "stay",
            "interact",
        ):
            raise ValueError("Formal evaluation requires the official primitive action order.")
        for unit in self.manifest.outer_units.units[1:]:
            dock = OfficialBackboneDock.from_launch_config(
                unit.backbone.launch_config_path
            )
            if (
                tuple(dock.action_order()) != self.action_order
                or dock.network_dimensions() != self.dock.network_dimensions()
            ):
                raise ValueError("Outer-unit official networks have incompatible structures.")

    def _validate_adapted_population_contracts(self) -> None:
        assert self.config is not None
        reference = self.config
        for entry in self.manifest.policies:
            config = entry.resolved_config
            if config is None:
                raise ValueError("An adapted policy has no resolved configuration.")
            if (
                config.condition_id != reference.condition_id
                or config.controller != reference.controller
                or config.environment != reference.environment
                or config.model != reference.model
                or config.probe != reference.probe
                or config.adaptation != reference.adaptation
                or config.num_prototypes != 4
                or config.is_family_pool != reference.is_family_pool
            ):
                raise ValueError(
                    "A formal population mixes conditions or model/evaluation contracts."
                )

    def _adaptive_action(
        self,
        *,
        params: Mapping[str, Any],
        calibration: Any,
        side: _SideState,
        observations: Any,
        episode_start: Any,
        episode_step: Any,
        random_keys: Any,
    ) -> tuple[Any, _SideState, Any, Any, Any]:
        import jax
        import jax.numpy as jnp

        assert self.model is not None and self.config is not None
        split_keys = _split_row_keys(random_keys, 2)
        model_keys = split_keys[:, 0]
        controller_keys = split_keys[:, 1]
        next_carry, output = self.model.apply(
            {"params": params},
            side.carry,
            observations[None, ...],
            episode_start[None, ...],
        )
        output = jax.tree_util.tree_map(lambda value: value[0], output)
        base_action = _categorical_by_row(model_keys, output["actor_logits"])
        candidates = enumerate_candidates(
            safe_action_mask=visible_goal_safe_action_mask(
                observations, action_order=self.action_order
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
            key=controller_keys,
            sequential=sequential,
            information_scores=information,
            candidate_mask=candidates.mask,
            base_action=base_action,
            budget_remaining=side.budget_remaining,
            decision_threshold=calibration[0],
            information_threshold=calibration[1],
            random_trigger_probability=calibration[2],
        )
        action_index = decision.chosen_action[:, None, None, None]
        probabilities = jnp.take_along_axis(
            output["response_probabilities"],
            jnp.broadcast_to(
                action_index,
                (
                    observations.shape[0],
                    output["response_probabilities"].shape[1],
                    1,
                    output["response_probabilities"].shape[-1],
                ),
            ),
            axis=2,
        )[:, :, 0, :]
        return (
            decision.chosen_action,
            _SideState(
                carry=next_carry,
                log_belief=side.log_belief,
                budget_remaining=decision.budget_remaining,
            ),
            decision.probed,
            probabilities,
            candidates.any_candidate,
        )

    def _compile_adapted_evaluator(self) -> Any:
        import jax
        import jax.numpy as jnp

        assert self.model is not None and self.config is not None
        count = self.manifest.episodes_per_pairing
        probability_floor = self.config.probe.belief_probability_floor

        def run(
            left_params: Mapping[str, Any],
            right_params: Mapping[str, Any],
            left_calibration: Any,
            right_calibration: Any,
            episode_seeds: Any,
        ) -> _EpisodeState:
            root_keys = jax.vmap(jax.random.PRNGKey)(episode_seeds)
            initial_keys = _split_row_keys(root_keys, 2)
            environment_state, observations = self.environment.reset_with_keys(
                initial_keys[:, 0]
            )
            initial = _EpisodeState(
                environment_state=environment_state,
                observations=observations,
                left=_SideState(
                    carry=self.model.initial_carry(count),
                    log_belief=uniform_log_belief(count, 4),
                    budget_remaining=jnp.full(
                        (count,), self.config.probe.budget_per_episode, dtype=jnp.int32
                    ),
                ),
                right=_SideState(
                    carry=self.model.initial_carry(count),
                    log_belief=uniform_log_belief(count, 4),
                    budget_remaining=jnp.full(
                        (count,), self.config.probe.budget_per_episode, dtype=jnp.int32
                    ),
                ),
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
                random_keys=initial_keys[:, 1],
            )

            def step(current: _EpisodeState, unused: Any) -> tuple[_EpisodeState, None]:
                del unused
                keys = _split_row_keys(current.random_keys, 4)
                left_action, left_state, left_probe, left_probabilities, left_opportunity = self._adaptive_action(
                    params=left_params,
                    calibration=left_calibration,
                    side=current.left,
                    observations=current.observations[:, 0, ...],
                    episode_start=current.episode_start,
                    episode_step=current.episode_step,
                    random_keys=keys[:, 1],
                )
                right_action, right_state, right_probe, right_probabilities, right_opportunity = self._adaptive_action(
                    params=right_params,
                    calibration=right_calibration,
                    side=current.right,
                    observations=current.observations[:, 1, ...],
                    episode_start=current.episode_start,
                    episode_step=current.episode_step,
                    random_keys=keys[:, 2],
                )
                joint_actions = jnp.stack((left_action, right_action), axis=-1)
                next_environment_state, next_observations, reward, done, info = self.environment.step_with_keys(
                    current.environment_state, joint_actions, keys[:, 3]
                )

                def update_belief(
                    side: _SideState,
                    probabilities: Any,
                    previous_observation: Any,
                    next_observation: Any,
                ) -> _SideState:
                    token = response_tokens(
                        previous_observation, next_observation, done, info
                    )
                    return side._replace(
                        log_belief=log_bayes_update(
                            side.log_belief,
                            likelihood_for_token(probabilities, token),
                            probability_floor=probability_floor,
                        )
                    )

                left_state = update_belief(
                    left_state,
                    left_probabilities,
                    current.observations[:, 0, ...],
                    next_observations[:, 0, ...],
                )
                right_state = update_belief(
                    right_state,
                    right_probabilities,
                    current.observations[:, 1, ...],
                    next_observations[:, 1, ...],
                )
                return _EpisodeState(
                    environment_state=next_environment_state,
                    observations=next_observations,
                    left=left_state,
                    right=right_state,
                    episode_start=done,
                    episode_step=current.episode_step + 1,
                    raw_return=current.raw_return + reward,
                    probe_count=(
                        current.probe_count
                        + left_probe.astype(jnp.int32)
                        + right_probe.astype(jnp.int32)
                    ),
                    safe_candidate_opportunity_count=(
                        current.safe_candidate_opportunity_count
                        + left_opportunity.astype(jnp.int32)
                        + right_opportunity.astype(jnp.int32)
                    ),
                    correct_delivery_count=(
                        current.correct_delivery_count + info["correct_delivery"]
                    ),
                    wrong_delivery_count=(
                        current.wrong_delivery_count + info["wrong_delivery"]
                    ),
                    indicator_cost=current.indicator_cost + info["indicator_cost"],
                    random_keys=keys[:, 0],
                ), None

            return jax.lax.scan(
                step,
                initial,
                xs=None,
                length=self.config.environment.episode_steps,
            )[0]

        return jax.jit(run)

    def _compile_backbone_evaluator(self) -> Any:
        import jax
        import jax.numpy as jnp

        count = self.manifest.episodes_per_pairing

        def run(
            left_params: Mapping[str, Any],
            right_params: Mapping[str, Any],
            episode_seeds: Any,
        ) -> _EpisodeState:
            root_keys = jax.vmap(jax.random.PRNGKey)(episode_seeds)
            initial_keys = _split_row_keys(root_keys, 2)
            environment_state, observations = self.environment.reset_with_keys(
                initial_keys[:, 0]
            )
            zero_belief = jnp.zeros((count, 4), dtype=jnp.float32)
            zero_budget = jnp.zeros((count,), dtype=jnp.int32)
            initial = _EpisodeState(
                environment_state=environment_state,
                observations=observations,
                left=_SideState(
                    self.dock.initial_recurrent_state(count), zero_belief, zero_budget
                ),
                right=_SideState(
                    self.dock.initial_recurrent_state(count), zero_belief, zero_budget
                ),
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
                random_keys=initial_keys[:, 1],
            )

            def step(current: _EpisodeState, unused: Any) -> tuple[_EpisodeState, None]:
                del unused
                keys = _split_row_keys(current.random_keys, 4)
                left_carry, left_logits = self.official_network_adapter.apply_actor(
                    left_params,
                    current.left.carry,
                    current.observations[:, 0, ...],
                    current.episode_start,
                )
                right_carry, right_logits = self.official_network_adapter.apply_actor(
                    right_params,
                    current.right.carry,
                    current.observations[:, 1, ...],
                    current.episode_start,
                )
                actions = jnp.stack(
                    (
                        _categorical_by_row(keys[:, 1], left_logits),
                        _categorical_by_row(keys[:, 2], right_logits),
                    ),
                    axis=-1,
                )
                next_environment_state, next_observations, reward, done, info = self.environment.step_with_keys(
                    current.environment_state, actions, keys[:, 3]
                )
                return _EpisodeState(
                    environment_state=next_environment_state,
                    observations=next_observations,
                    left=current.left._replace(carry=left_carry),
                    right=current.right._replace(carry=right_carry),
                    episode_start=done,
                    episode_step=current.episode_step + 1,
                    raw_return=current.raw_return + reward,
                    probe_count=current.probe_count,
                    safe_candidate_opportunity_count=current.safe_candidate_opportunity_count,
                    correct_delivery_count=(
                        current.correct_delivery_count + info["correct_delivery"]
                    ),
                    wrong_delivery_count=(
                        current.wrong_delivery_count + info["wrong_delivery"]
                    ),
                    indicator_cost=current.indicator_cost + info["indicator_cost"],
                    random_keys=keys[:, 0],
                ), None

            return jax.lax.scan(
                step, initial, xs=None, length=self.environment.episode_steps
            )[0]

        return jax.jit(run)

    def evaluate_pairing(
        self, pairing: StandardPairing, episode_seeds: Sequence[int]
    ) -> Sequence[Mapping[str, Any]]:
        import jax.numpy as jnp

        if len(episode_seeds) != self.manifest.episodes_per_pairing:
            raise ValueError("Formal evaluation requires exactly 500 episode seeds.")
        left = self.policies[pairing.outer_unit_0]
        right = self.policies[pairing.outer_unit_1]
        seeds = jnp.asarray(np.asarray(episode_seeds, dtype=np.uint32))
        if self.manifest.source_type == "adaptation_checkpoint":
            assert left.calibration is not None and right.calibration is not None
            final = self._compiled(
                left.params,
                right.params,
                jnp.asarray(left.calibration),
                jnp.asarray(right.calibration),
                seeds,
            )
            maximum_probe_budget = 2 * int(self.config.probe.budget_per_episode)
        else:
            final = self._compiled(left.params, right.params, seeds)
            maximum_probe_budget = 0
        if not bool(np.asarray(final.episode_start).all()):
            raise RuntimeError("A standard matrix cell did not finish all 400-step episodes.")
        arrays = {
            "raw_return": np.asarray(final.raw_return),
            "probe_count": np.asarray(final.probe_count),
            "safe_candidate_opportunity_count": np.asarray(
                final.safe_candidate_opportunity_count
            ),
            "correct_delivery_count": np.asarray(final.correct_delivery_count),
            "wrong_delivery_count": np.asarray(final.wrong_delivery_count),
            "indicator_cost": np.asarray(final.indicator_cost),
        }
        return [
            {
                "raw_return": float(arrays["raw_return"][index]),
                "probe_count": int(arrays["probe_count"][index]),
                "safe_candidate_opportunity_count": int(
                    arrays["safe_candidate_opportunity_count"][index]
                ),
                "maximum_probe_budget": maximum_probe_budget,
                "correct_delivery_count": int(
                    arrays["correct_delivery_count"][index]
                ),
                "wrong_delivery_count": int(arrays["wrong_delivery_count"][index]),
                "indicator_cost": float(arrays["indicator_cost"][index]),
            }
            for index in range(self.manifest.episodes_per_pairing)
        ]


__all__ = ["FormalPopulationRuntime"]
