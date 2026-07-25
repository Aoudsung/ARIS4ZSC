"""Development pairing execution for adapted and frozen official policies."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from src.path_c.belief.update import (
    likelihood_for_token,
    log_bayes_update,
    uniform_log_belief,
)
from src.path_c.evaluation.pairing import development_pairings, execute_pairings
from src.path_c.evaluation.summary import summarize_rows
from src.path_c.probe.candidates import enumerate_candidates
from src.path_c.probe.controllers import select_action
from src.path_c.probe.scores import response_information_scores, sequential_scores

from .env_dock import OvercookedV2VectorEnvironment, visible_goal_safe_action_mask
from .response_dock import response_tokens


BACKBONE_EVALUATION_EPISODES_PER_PAIRING = 64


class _SideState(NamedTuple):
    carry: Any
    log_belief: Any
    budget_remaining: Any


class _EpisodeState(NamedTuple):
    env_state: Any
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
    random_key: Any


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _split_row_keys(keys: Any, count: int) -> Any:
    import jax
    import jax.numpy as jnp

    values = jnp.asarray(keys)
    if values.shape == (2,):
        return jax.random.split(values, count)
    if values.ndim == 2 and values.shape[1] == 2:
        return jax.vmap(lambda value: jax.random.split(value, count))(values)
    raise ValueError("Evaluation randomness requires one root key or one key per episode.")


def _categorical_by_row(keys: Any, logits: Any) -> Any:
    import jax
    import jax.numpy as jnp

    key_array = jnp.asarray(keys)
    values = jnp.asarray(logits)
    if key_array.shape == (2,):
        return jax.random.categorical(key_array, values, axis=-1)
    if key_array.shape != (values.shape[0], 2):
        raise ValueError("Categorical evaluation requires one random key per episode row.")
    return jax.vmap(
        lambda key, row: jax.random.categorical(key, row, axis=-1)
    )(key_array, values)


def _adaptive_action(
    *,
    side: _SideState,
    observations: Any,
    episode_start: Any,
    episode_step: Any,
    random_key: Any,
    model: Any,
    params: Mapping[str, Any],
    config: Any,
    action_order: Sequence[str],
    calibration_summary: Mapping[str, Any],
) -> tuple[Any, _SideState, Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    split_keys = _split_row_keys(random_key, 2)
    if split_keys.ndim == 2:
        model_key, controller_key = split_keys
    else:
        model_key, controller_key = split_keys[:, 0], split_keys[:, 1]
    next_carry, output = model.apply(
        {"params": params}, side.carry, observations[None, ...], episode_start[None, ...]
    )
    output = jax.tree_util.tree_map(lambda value: value[0], output)
    base_action = _categorical_by_row(model_key, output["actor_logits"])
    candidates = enumerate_candidates(
        safe_action_mask=visible_goal_safe_action_mask(
            observations, action_order=action_order
        ),
        base_action=base_action,
        episode_step=episode_step,
        budget_remaining=side.budget_remaining,
        candidate_window=config.probe.candidate_window,
    )
    sequential = sequential_scores(
        log_belief=side.log_belief,
        response_probabilities=output["transition_response_probabilities"],
        reward_estimates=output["reward_estimates"],
        next_values=output["next_values"],
        base_action=base_action,
        candidate_mask=candidates.mask,
        gamma=config.adaptation.gamma,
        probability_floor=config.probe.belief_probability_floor,
    )
    information = response_information_scores(
        log_belief=side.log_belief,
        response_probabilities=output["transition_response_probabilities"],
        probability_floor=config.probe.belief_probability_floor,
    )
    decision = select_action(
        config.controller,
        key=controller_key,
        sequential=sequential,
        information_scores=information,
        candidate_mask=candidates.mask,
        base_action=base_action,
        budget_remaining=side.budget_remaining,
        decision_threshold=float(calibration_summary["decision_threshold"]),
        information_threshold=float(calibration_summary["information_threshold"]),
        random_trigger_probability=float(calibration_summary["random_trigger_probability"]),
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
    next_side = _SideState(
        carry=next_carry,
        log_belief=side.log_belief,
        budget_remaining=decision.budget_remaining,
    )
    return (
        decision.chosen_action,
        next_side,
        decision.probed,
        probabilities,
        candidates.any_candidate,
    )


def _frozen_action(
    *, policy: Any, side: _SideState, observations: Any, episode_start: Any, random_key: Any
) -> tuple[Any, _SideState, Any, None, Any]:
    import jax.numpy as jnp

    next_carry, logits = policy.network_adapter.apply_actor(
        policy.params, side.carry, observations, episode_start
    )
    action = _categorical_by_row(random_key, logits)
    return (
        action,
        side._replace(carry=next_carry),
        jnp.zeros((observations.shape[0],), dtype=jnp.bool_),
        None,
        jnp.zeros((observations.shape[0],), dtype=jnp.bool_),
    )


def _evaluate_one_pairing(
    *,
    pairing: Any,
    episode_seeds: Sequence[int],
    built: Mapping[str, Any],
    params: Mapping[str, Any] | None,
    config: Any,
    calibration_summary: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    count = len(episode_seeds)
    environment = OvercookedV2VectorEnvironment.create(num_envs=count)
    seed_array = jnp.asarray(np.asarray(episode_seeds, dtype=np.uint32))
    root_keys = jax.vmap(jax.random.PRNGKey)(seed_array)
    initial_keys = _split_row_keys(root_keys, 2)
    reset_keys = initial_keys[:, 0]
    rollout_keys = initial_keys[:, 1]
    env_state, observations = environment.reset_with_keys(reset_keys)
    prototype_count = len(built["partners"])
    frozen_by_id = {
        policy.prototype_id: policy
        for policy in (*built["partners"], built["backbone_policy"])
    }

    def initial_side(source: str, policy_id: str) -> _SideState:
        carry = (
            built["model"].initial_carry(count)
            if source == "adaptation_checkpoint"
            else frozen_by_id[policy_id].initial_state(count)
        )
        return _SideState(
            carry=carry,
            log_belief=uniform_log_belief(count, prototype_count),
            budget_remaining=jnp.full(
                (count,), config.probe.budget_per_episode, dtype=jnp.int32
            ),
        )

    initial = _EpisodeState(
        env_state=env_state,
        observations=observations,
        left=initial_side(pairing.policy_0_source, pairing.policy_0_id),
        right=initial_side(pairing.policy_1_source, pairing.policy_1_id),
        episode_start=jnp.ones((count,), dtype=jnp.bool_),
        episode_step=jnp.zeros((count,), dtype=jnp.int32),
        raw_return=jnp.zeros((count,), dtype=jnp.float32),
        probe_count=jnp.zeros((count,), dtype=jnp.int32),
        safe_candidate_opportunity_count=jnp.zeros((count,), dtype=jnp.int32),
        correct_delivery_count=jnp.zeros((count,), dtype=jnp.int32),
        wrong_delivery_count=jnp.zeros((count,), dtype=jnp.int32),
        indicator_cost=jnp.zeros((count,), dtype=jnp.float32),
        random_key=rollout_keys,
    )
    action_order = tuple(built["dock"].action_order())

    def side_action(
        source: str,
        policy_id: str,
        side: _SideState,
        side_observation: Any,
        episode_start: Any,
        episode_step: Any,
        key: Any,
    ) -> tuple[Any, _SideState, Any, Any, Any]:
        if source == "adaptation_checkpoint":
            if params is None:
                raise RuntimeError("Adapted evaluation requires an adaptation checkpoint.")
            return _adaptive_action(
                side=side,
                observations=side_observation,
                episode_start=episode_start,
                episode_step=episode_step,
                random_key=key,
                model=built["model"],
                params=params,
                config=config,
                action_order=action_order,
                calibration_summary=calibration_summary,
            )
        return _frozen_action(
            policy=frozen_by_id[policy_id],
            side=side,
            observations=side_observation,
            episode_start=episode_start,
            random_key=key,
        )

    def step(current: _EpisodeState, unused: Any) -> tuple[_EpisodeState, None]:
        del unused
        split_keys = _split_row_keys(current.random_key, 4)
        next_key = split_keys[:, 0]
        left_key = split_keys[:, 1]
        right_key = split_keys[:, 2]
        environment_key = split_keys[:, 3]
        left_observation = current.observations[:, 0, ...]
        right_observation = current.observations[:, 1, ...]
        (
            left_action,
            left_state,
            left_probe,
            left_probabilities,
            left_opportunity,
        ) = side_action(
            pairing.policy_0_source,
            pairing.policy_0_id,
            current.left,
            left_observation,
            current.episode_start,
            current.episode_step,
            left_key,
        )
        (
            right_action,
            right_state,
            right_probe,
            right_probabilities,
            right_opportunity,
        ) = side_action(
            pairing.policy_1_source,
            pairing.policy_1_id,
            current.right,
            right_observation,
            current.episode_start,
            current.episode_step,
            right_key,
        )
        joint_actions = jnp.stack((left_action, right_action), axis=-1)
        next_env_state, next_observations, reward, done, info = environment.step_with_keys(
            current.env_state, joint_actions, environment_key
        )

        def update_response(
            source: str,
            side: _SideState,
            previous_observation: Any,
            next_observation: Any,
            probabilities: Any,
        ) -> _SideState:
            if source != "adaptation_checkpoint":
                return side
            if probabilities is None:
                raise RuntimeError("An adapted evaluation policy must expose response probabilities.")
            token = response_tokens(previous_observation, next_observation, done, info)
            likelihood = likelihood_for_token(probabilities, token)
            belief = log_bayes_update(
                side.log_belief,
                likelihood,
                probability_floor=config.probe.belief_probability_floor,
            )
            return side._replace(log_belief=belief)

        left_state = update_response(
            pairing.policy_0_source,
            left_state,
            left_observation,
            next_observations[:, 0, ...],
            left_probabilities,
        )
        right_state = update_response(
            pairing.policy_1_source,
            right_state,
            right_observation,
            next_observations[:, 1, ...],
            right_probabilities,
        )
        return _EpisodeState(
            env_state=next_env_state,
            observations=next_observations,
            left=left_state,
            right=right_state,
            episode_start=done,
            episode_step=current.episode_step + 1,
            raw_return=current.raw_return + reward,
            probe_count=current.probe_count + left_probe.astype(jnp.int32) + right_probe.astype(jnp.int32),
            safe_candidate_opportunity_count=(
                current.safe_candidate_opportunity_count
                + left_opportunity.astype(jnp.int32)
                + right_opportunity.astype(jnp.int32)
            ),
            correct_delivery_count=current.correct_delivery_count + info["correct_delivery"],
            wrong_delivery_count=current.wrong_delivery_count + info["wrong_delivery"],
            indicator_cost=current.indicator_cost + info["indicator_cost"],
            random_key=next_key,
        ), None

    final = jax.jit(
        lambda value: jax.lax.scan(step, value, xs=None, length=config.environment.episode_steps)[0]
    )(initial)
    if not bool(np.asarray(final.episode_start).all()):
        raise RuntimeError("Evaluation lanes did not all finish one 400-step episode.")
    host_return = np.asarray(final.raw_return)
    host_probe = np.asarray(final.probe_count)
    host_opportunities = np.asarray(final.safe_candidate_opportunity_count)
    host_correct = np.asarray(final.correct_delivery_count)
    host_wrong = np.asarray(final.wrong_delivery_count)
    host_cost = np.asarray(final.indicator_cost)
    adapted_side_count = sum(
        source == "adaptation_checkpoint"
        for source in (pairing.policy_0_source, pairing.policy_1_source)
    )
    maximum_probe_budget = adapted_side_count * config.probe.budget_per_episode
    return [
        {
            "raw_return": float(host_return[index]),
            "probe_count": int(host_probe[index]),
            "safe_candidate_opportunity_count": int(host_opportunities[index]),
            "maximum_probe_budget": maximum_probe_budget,
            "correct_delivery_count": int(host_correct[index]),
            "wrong_delivery_count": int(host_wrong[index]),
            "indicator_cost": float(host_cost[index]),
        }
        for index in range(count)
    ]


def run_development_evaluation(
    *,
    context: Any,
    built: Mapping[str, Any],
    params: Mapping[str, Any],
    config: Any,
    calibration_summary: Mapping[str, Any],
) -> Mapping[str, Path]:
    """Run the registered 1 + 8 development pairing schedule."""

    pairings = development_pairings(
        f"adapted_{config.condition_id}",
        [policy.prototype_id for policy in built["partners"]],
    )
    rows_path = context.stage_directory / "rows.jsonl"
    rows = execute_pairings(
        pairings,
        episodes_per_pairing=config.evaluation.episodes_per_pairing,
        evaluation_seed=config.seeds.evaluation_seed,
        evaluate_pairing=lambda pairing, seeds: _evaluate_one_pairing(
            pairing=pairing,
            episode_seeds=seeds,
            built=built,
            params=params,
            config=config,
            calibration_summary=calibration_summary,
        ),
        output_path=rows_path,
    )
    summary = dict(summarize_rows(rows))
    summary.update(
        {
            "run_kind": config.run_kind,
            "scientific_readout_allowed": config.scientific_readout_allowed,
            "episodes_per_pairing": config.evaluation.episodes_per_pairing,
            "effective_completed_episodes": len(rows),
            "effective_environment_steps": len(rows) * config.environment.episode_steps,
        }
    )
    summary_path = context.stage_directory / "summary.json"
    _atomic_json(summary_path, summary)
    return {"rows": rows_path, "summary": summary_path}


def run_backbone_evaluation(
    *,
    context: Any,
    built: Mapping[str, Any],
    config: Any,
) -> Mapping[str, Path]:
    """Evaluate the frozen official seed-100 backbone before adaptation."""

    backbone = built["backbone_policy"]
    pairings = development_pairings(
        backbone.prototype_id,
        [policy.prototype_id for policy in built["partners"]],
        focal_policy_source="frozen_official_backbone",
    )
    rows_path = context.stage_directory / "rows.jsonl"
    rows = execute_pairings(
        pairings,
        episodes_per_pairing=BACKBONE_EVALUATION_EPISODES_PER_PAIRING,
        evaluation_seed=config.seeds.evaluation_seed,
        evaluate_pairing=lambda pairing, seeds: _evaluate_one_pairing(
            pairing=pairing,
            episode_seeds=seeds,
            built=built,
            params=None,
            config=config,
            calibration_summary={},
        ),
        output_path=rows_path,
    )
    if any(int(row["probe_count"]) != 0 for row in rows):
        raise RuntimeError("The frozen backbone anchor unexpectedly recorded a probe.")
    summary = dict(summarize_rows(rows))
    summary.update(
        {
            "stage": "backbone_evaluation",
            "shared_across_conditions": True,
            "backbone_policy_id": backbone.prototype_id,
            "backbone_origin_sha256": config.backbone_init.flax_weights_sha256,
            "run_kind": config.run_kind,
            "scientific_readout_allowed": config.scientific_readout_allowed,
            "episodes_per_pairing": BACKBONE_EVALUATION_EPISODES_PER_PAIRING,
            "effective_completed_episodes": len(rows),
            "effective_environment_steps": len(rows)
            * config.environment.episode_steps,
            "adaptation_checkpoint_used": False,
        }
    )
    summary_path = context.stage_directory / "summary.json"
    _atomic_json(summary_path, summary)
    return {"rows": rows_path, "summary": summary_path}
