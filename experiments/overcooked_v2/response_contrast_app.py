"""Matched A1/A2-mask/A2-use response-contrast runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from experiments.overcooked_v2.deployment import (
    Deployment,
    load_deployment,
    reset_deployment_state,
    update_deployment_after_transition,
)
from experiments.overcooked_v2.official_adapter import VectorEnvironment
from src.path_c.evaluation import (
    Pairing,
    ResponseContrastRow,
    standard_episode_seed,
    summarize_response_contrast,
    validate_development_response_contrast_rows,
    validate_response_contrast_rows,
)
from src.path_c.method import policy_effect_trigger_tolerance
from src.path_c.runner import policy_action
from src.path_c.storage import read_parquet, write_json, write_parquet


class DeploymentStep(NamedTuple):
    state: Any
    action: Any
    output: Any
    record: Any


class _ContrastBranch(NamedTuple):
    environment_state: Any
    observations: Any
    left_state: Any
    right_state: Any
    raw_return: Any
    correct_delivery_count: Any
    wrong_delivery_count: Any
    indicator_activation_count: Any
    last_left_action: Any
    last_left_response: Any
    last_reward: Any


class _ContrastState(NamedTuple):
    a1: _ContrastBranch
    a2_mask: _ContrastBranch
    a2_use: _ContrastBranch
    triggered: Any
    trigger_step: Any
    trigger_tolerance: Any
    predicted_response_effect: Any
    predicted_policy_cost: Any
    predicted_net_effect: Any
    predicted_regularized_net_effect: Any
    predicted_policy_total_variation: Any
    predicted_policy_mediated_effect: Any
    predicted_next_policy_total_variation: Any
    maximum_action_net_value: Any
    maximum_action_policy_mediated_gain: Any
    executed_action_net_value: Any
    executed_action_response_value: Any
    executed_action_policy_mediated_gain: Any
    executed_action_expected_next_policy_tv: Any
    executed_action: Any
    maximum_net_action: Any
    post_response_belief_l1: Any
    first_left_action_difference_step: Any
    first_observation_difference_step: Any
    first_response_code_difference_step: Any
    first_reward_difference_step: Any
    left_action_difference_count: Any
    observation_difference_count: Any
    response_code_difference_count: Any
    reward_difference_count: Any


def _deployment_step(
    *,
    deployment: Deployment,
    state: Any,
    observations: Any,
    keys: Any,
    config: Any,
) -> DeploymentStep:
    next_state, action, output, record, unused_generic = policy_action(
        functions=deployment.functions(),
        params={
            "official": deployment.online_params,
            "heads": deployment.head_params,
        },
        policy_state=state,
        observations=observations,
        key=keys,
        deployment_mode="posterior_use",
        gamma=config.training.gamma,
        behavior_support=0.0,
    )
    del unused_generic
    return DeploymentStep(next_state, action, output, record)


def _masked_action_step(step: DeploymentStep, keys: Any) -> DeploymentStep:
    import jax

    action = jax.vmap(
        lambda key, logits: jax.random.categorical(key, logits)
    )(keys, step.output.mask_execution_logits)
    return step._replace(action=action)


def _advance_contrast_branch(
    *,
    branch: _ContrastBranch,
    left: Deployment,
    right: Deployment,
    left_step: DeploymentStep,
    right_step: DeploymentStep,
    environment: VectorEnvironment,
    environment_keys: Any,
    config: Any,
    mask_left_response: Any,
) -> _ContrastBranch:
    import jax.numpy as jnp

    actions = jnp.stack((left_step.action, right_step.action), axis=-1)
    next_environment, next_observations, rewards, dones, info = (
        environment.step_with_keys(
            branch.environment_state, actions, environment_keys
        )
    )
    terminal = info["terminal_observations"]
    mask = dones.reshape(
        dones.shape + (1,) * (next_observations[:, 0].ndim - 1)
    )
    left_next_observation = jnp.where(
        mask, terminal[:, 0], next_observations[:, 0]
    )
    right_next_observation = jnp.where(
        mask, terminal[:, 1], next_observations[:, 1]
    )
    left_state, left_response = update_deployment_after_transition(
        deployment=left,
        functions=left.functions(),
        state=left_step.state,
        output=left_step.output,
        observations=branch.observations[:, 0],
        actions=left_step.action,
        next_observations=left_next_observation,
        rewards=rewards,
        dones=dones,
        deployment_mode="posterior_use",
        terminal_response=config.model.response_count - 1,
        mask_response=mask_left_response,
    )
    right_state, unused_right_response = update_deployment_after_transition(
        deployment=right,
        functions=right.functions(),
        state=right_step.state,
        output=right_step.output,
        observations=branch.observations[:, 1],
        actions=right_step.action,
        next_observations=right_next_observation,
        rewards=rewards,
        dones=dones,
        deployment_mode="posterior_use",
        terminal_response=config.model.response_count - 1,
    )
    del unused_right_response
    return _ContrastBranch(
        environment_state=next_environment,
        observations=next_observations,
        left_state=left_state,
        right_state=right_state,
        raw_return=branch.raw_return + rewards,
        correct_delivery_count=(
            branch.correct_delivery_count + info["correct_delivery"]
        ),
        wrong_delivery_count=(
            branch.wrong_delivery_count + info["wrong_delivery"]
        ),
        indicator_activation_count=(
            branch.indicator_activation_count + info["indicator_activation"]
        ),
        last_left_action=left_step.action,
        last_left_response=left_response,
        last_reward=rewards,
    )


def contrast_pairing_batch(
    *,
    config: Any,
    left: Deployment,
    right: Deployment,
    pairing: Pairing,
    evaluation_seed: int,
) -> tuple[ResponseContrastRow, ...]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    count = config.evaluation.episodes_per_pairing
    template = VectorEnvironment.create(config)
    environment = VectorEnvironment(
        environment=template.environment,
        num_envs=count,
        episode_steps=config.environment.episode_steps,
    )
    seeds = np.asarray(
        [
            standard_episode_seed(
                evaluation_seed=evaluation_seed,
                layout=config.environment.layout,
                left_outer_unit_id=pairing.left_outer_unit_id,
                right_outer_unit_id=pairing.right_outer_unit_id,
                episode_index=index,
            )
            for index in range(count)
        ],
        dtype=np.uint32,
    )
    root_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(root_keys)
    environment_state, observations = environment.reset_with_keys(reset_keys)
    minus_one = jnp.full((count,), -1, dtype=jnp.int32)
    zeros_int = jnp.zeros((count,), dtype=jnp.int32)
    zeros_float = jnp.zeros((count,), dtype=jnp.float32)
    initial_branch = _ContrastBranch(
        environment_state=environment_state,
        observations=observations,
        left_state=reset_deployment_state(
            left, batch_size=count, config=config
        ),
        right_state=reset_deployment_state(
            right, batch_size=count, config=config
        ),
        raw_return=zeros_float,
        correct_delivery_count=zeros_int,
        wrong_delivery_count=zeros_int,
        indicator_activation_count=zeros_int,
        last_left_action=minus_one,
        last_left_response=minus_one,
        last_reward=zeros_float,
    )
    initial = _ContrastState(
        a1=initial_branch,
        a2_mask=initial_branch,
        a2_use=initial_branch,
        triggered=jnp.zeros((count,), dtype=jnp.bool_),
        trigger_step=minus_one,
        trigger_tolerance=zeros_float,
        predicted_response_effect=zeros_float,
        predicted_policy_cost=zeros_float,
        predicted_net_effect=zeros_float,
        predicted_regularized_net_effect=zeros_float,
        predicted_policy_total_variation=zeros_float,
        predicted_policy_mediated_effect=zeros_float,
        predicted_next_policy_total_variation=zeros_float,
        maximum_action_net_value=zeros_float,
        maximum_action_policy_mediated_gain=zeros_float,
        executed_action_net_value=zeros_float,
        executed_action_response_value=zeros_float,
        executed_action_policy_mediated_gain=zeros_float,
        executed_action_expected_next_policy_tv=zeros_float,
        executed_action=minus_one,
        maximum_net_action=minus_one,
        post_response_belief_l1=zeros_float,
        first_left_action_difference_step=minus_one,
        first_observation_difference_step=minus_one,
        first_response_code_difference_step=minus_one,
        first_reward_difference_step=minus_one,
        left_action_difference_count=zeros_int,
        observation_difference_count=zeros_int,
        response_code_difference_count=zeros_int,
        reward_difference_count=zeros_int,
    )

    def one_step(current: _ContrastState, step: Any) -> tuple[Any, None]:
        left_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 1 + 3 * step)
        )(root_keys)
        right_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 2 + 3 * step)
        )(root_keys)
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 3 + 3 * step)
        )(root_keys)

        a1_left = _deployment_step(
            deployment=left,
            state=current.a1.left_state,
            observations=current.a1.observations[:, 0],
            keys=left_keys,
            config=config,
        )
        a1_right = _deployment_step(
            deployment=right,
            state=current.a1.right_state,
            observations=current.a1.observations[:, 1],
            keys=right_keys,
            config=config,
        )
        mask_left = _deployment_step(
            deployment=left,
            state=current.a2_mask.left_state,
            observations=current.a2_mask.observations[:, 0],
            keys=left_keys,
            config=config,
        )
        mask_right = _deployment_step(
            deployment=right,
            state=current.a2_mask.right_state,
            observations=current.a2_mask.observations[:, 1],
            keys=right_keys,
            config=config,
        )
        use_left = _deployment_step(
            deployment=left,
            state=current.a2_use.left_state,
            observations=current.a2_use.observations[:, 0],
            keys=left_keys,
            config=config,
        )
        use_right = _deployment_step(
            deployment=right,
            state=current.a2_use.right_state,
            observations=current.a2_use.observations[:, 1],
            keys=right_keys,
            config=config,
        )

        tolerance = policy_effect_trigger_tolerance(
            use_left.output.j_use, use_left.output.j_mask
        )
        actionable = (
            use_left.record.executed_action_policy_mediated_gain > tolerance
        ) & (
            use_left.record.executed_action_expected_next_policy_tv
            >= config.evaluation.response_policy_tv_minimum
        )
        newly_triggered = (~current.triggered) & actionable

        masked_reference = _masked_action_step(use_left, left_keys)
        a1_left = a1_left._replace(
            action=jnp.where(
                newly_triggered,
                masked_reference.action,
                a1_left.action,
            )
        )
        mask_left = mask_left._replace(
            action=jnp.where(
                newly_triggered, use_left.action, mask_left.action
            )
        )
        next_a1 = _advance_contrast_branch(
            branch=current.a1,
            left=left,
            right=right,
            left_step=a1_left,
            right_step=a1_right,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_left_response=newly_triggered,
        )
        next_mask = _advance_contrast_branch(
            branch=current.a2_mask,
            left=left,
            right=right,
            left_step=mask_left,
            right_step=mask_right,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_left_response=newly_triggered,
        )
        next_use = _advance_contrast_branch(
            branch=current.a2_use,
            left=left,
            right=right,
            left_step=use_left,
            right_step=use_right,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_left_response=False,
        )

        active = current.triggered | newly_triggered
        action_difference = active & (
            next_mask.last_left_action != next_use.last_left_action
        )
        observation_difference = active & jnp.any(
            next_mask.observations != next_use.observations,
            axis=tuple(range(1, next_use.observations.ndim)),
        )
        response_difference = active & (
            next_mask.last_left_response != next_use.last_left_response
        )
        reward_difference = active & (
            next_mask.last_reward != next_use.last_reward
        )

        def capture(old: Any, new: Any) -> Any:
            return jnp.where(newly_triggered, new, old)

        def first(old: Any, condition: Any) -> Any:
            return jnp.where((old < 0) & condition, step, old)

        use_belief = jax.nn.softmax(
            next_use.left_state.slot_log_belief, axis=-1
        )
        mask_belief = jax.nn.softmax(
            next_mask.left_state.slot_log_belief, axis=-1
        )
        return _ContrastState(
            a1=next_a1,
            a2_mask=next_mask,
            a2_use=next_use,
            triggered=active,
            trigger_step=capture(current.trigger_step, step),
            trigger_tolerance=capture(
                current.trigger_tolerance, tolerance
            ),
            predicted_response_effect=capture(
                current.predicted_response_effect,
                use_left.output.predicted_response_effect,
            ),
            predicted_policy_cost=capture(
                current.predicted_policy_cost,
                use_left.output.predicted_policy_cost,
            ),
            predicted_net_effect=capture(
                current.predicted_net_effect,
                use_left.output.predicted_net_effect,
            ),
            predicted_regularized_net_effect=capture(
                current.predicted_regularized_net_effect,
                use_left.output.predicted_regularized_net_effect,
            ),
            predicted_policy_total_variation=capture(
                current.predicted_policy_total_variation,
                use_left.output.predicted_policy_total_variation,
            ),
            predicted_policy_mediated_effect=capture(
                current.predicted_policy_mediated_effect,
                use_left.output.predicted_policy_mediated_effect,
            ),
            predicted_next_policy_total_variation=capture(
                current.predicted_next_policy_total_variation,
                use_left.output.predicted_next_policy_total_variation,
            ),
            maximum_action_net_value=capture(
                current.maximum_action_net_value,
                use_left.record.maximum_action_net_value,
            ),
            maximum_action_policy_mediated_gain=capture(
                current.maximum_action_policy_mediated_gain,
                use_left.record.maximum_action_policy_mediated_gain,
            ),
            executed_action_net_value=capture(
                current.executed_action_net_value,
                use_left.record.executed_action_net_value,
            ),
            executed_action_response_value=capture(
                current.executed_action_response_value,
                use_left.record.executed_action_response_value,
            ),
            executed_action_policy_mediated_gain=capture(
                current.executed_action_policy_mediated_gain,
                use_left.record.executed_action_policy_mediated_gain,
            ),
            executed_action_expected_next_policy_tv=capture(
                current.executed_action_expected_next_policy_tv,
                use_left.record.executed_action_expected_next_policy_tv,
            ),
            executed_action=capture(
                current.executed_action, use_left.action
            ),
            maximum_net_action=capture(
                current.maximum_net_action,
                jnp.argmax(use_left.output.per_action_net_value, axis=-1),
            ),
            post_response_belief_l1=capture(
                current.post_response_belief_l1,
                jnp.sum(jnp.abs(use_belief - mask_belief), axis=-1),
            ),
            first_left_action_difference_step=first(
                current.first_left_action_difference_step,
                action_difference,
            ),
            first_observation_difference_step=first(
                current.first_observation_difference_step,
                observation_difference,
            ),
            first_response_code_difference_step=first(
                current.first_response_code_difference_step,
                response_difference,
            ),
            first_reward_difference_step=first(
                current.first_reward_difference_step,
                reward_difference,
            ),
            left_action_difference_count=(
                current.left_action_difference_count
                + action_difference.astype(jnp.int32)
            ),
            observation_difference_count=(
                current.observation_difference_count
                + observation_difference.astype(jnp.int32)
            ),
            response_code_difference_count=(
                current.response_code_difference_count
                + response_difference.astype(jnp.int32)
            ),
            reward_difference_count=(
                current.reward_difference_count
                + reward_difference.astype(jnp.int32)
            ),
        ), None

    final, unused = jax.lax.scan(
        one_step,
        initial,
        jnp.arange(config.environment.episode_steps),
    )
    del unused

    def optional_step(values: Any, lane: int) -> int | None:
        value = int(np.asarray(values[lane]))
        return None if value < 0 else value

    rows = []
    for lane in range(count):
        triggered = bool(np.asarray(final.triggered[lane]))

        def optional_float(values: Any) -> float | None:
            return float(np.asarray(values[lane])) if triggered else None

        def optional_int(values: Any) -> int | None:
            return int(np.asarray(values[lane])) if triggered else None

        rows.append(
            ResponseContrastRow(
                pairing_id=pairing.pairing_id,
                episode_index=lane,
                episode_seed=int(seeds[lane]),
                triggered=triggered,
                trigger_step=(
                    int(np.asarray(final.trigger_step[lane]))
                    if triggered
                    else None
                ),
                trigger_tolerance=optional_float(final.trigger_tolerance),
                predicted_response_effect=optional_float(
                    final.predicted_response_effect
                ),
                predicted_policy_cost=optional_float(
                    final.predicted_policy_cost
                ),
                predicted_net_effect=optional_float(
                    final.predicted_net_effect
                ),
                predicted_regularized_net_effect=optional_float(
                    final.predicted_regularized_net_effect
                ),
                predicted_policy_total_variation=optional_float(
                    final.predicted_policy_total_variation
                ),
                predicted_policy_mediated_effect=optional_float(
                    final.predicted_policy_mediated_effect
                ),
                predicted_next_policy_total_variation=optional_float(
                    final.predicted_next_policy_total_variation
                ),
                maximum_action_net_value=optional_float(
                    final.maximum_action_net_value
                ),
                maximum_action_policy_mediated_gain=optional_float(
                    final.maximum_action_policy_mediated_gain
                ),
                executed_action_net_value=optional_float(
                    final.executed_action_net_value
                ),
                executed_action_response_value=optional_float(
                    final.executed_action_response_value
                ),
                executed_action_policy_mediated_gain=optional_float(
                    final.executed_action_policy_mediated_gain
                ),
                executed_action_expected_next_policy_tv=optional_float(
                    final.executed_action_expected_next_policy_tv
                ),
                executed_action=optional_int(final.executed_action),
                maximum_net_action=optional_int(final.maximum_net_action),
                post_response_belief_l1=optional_float(
                    final.post_response_belief_l1
                ),
                first_left_action_difference_step=(
                    optional_step(
                        final.first_left_action_difference_step, lane
                    )
                    if triggered
                    else None
                ),
                first_observation_difference_step=(
                    optional_step(
                        final.first_observation_difference_step, lane
                    )
                    if triggered
                    else None
                ),
                first_response_code_difference_step=(
                    optional_step(
                        final.first_response_code_difference_step, lane
                    )
                    if triggered
                    else None
                ),
                first_reward_difference_step=(
                    optional_step(final.first_reward_difference_step, lane)
                    if triggered
                    else None
                ),
                left_action_difference_count=optional_int(
                    final.left_action_difference_count
                ),
                observation_difference_count=optional_int(
                    final.observation_difference_count
                ),
                response_code_difference_count=optional_int(
                    final.response_code_difference_count
                ),
                reward_difference_count=optional_int(
                    final.reward_difference_count
                ),
                environment_steps=3 * config.environment.episode_steps,
                a1_raw_return=float(np.asarray(final.a1.raw_return[lane])),
                a1_correct_delivery_count=int(
                    np.asarray(final.a1.correct_delivery_count[lane])
                ),
                a1_wrong_delivery_count=int(
                    np.asarray(final.a1.wrong_delivery_count[lane])
                ),
                a1_indicator_activation_count=int(
                    np.asarray(final.a1.indicator_activation_count[lane])
                ),
                a2_mask_raw_return=float(
                    np.asarray(final.a2_mask.raw_return[lane])
                ),
                a2_mask_correct_delivery_count=int(
                    np.asarray(final.a2_mask.correct_delivery_count[lane])
                ),
                a2_mask_wrong_delivery_count=int(
                    np.asarray(final.a2_mask.wrong_delivery_count[lane])
                ),
                a2_mask_indicator_activation_count=int(
                    np.asarray(final.a2_mask.indicator_activation_count[lane])
                ),
                a2_use_raw_return=float(
                    np.asarray(final.a2_use.raw_return[lane])
                ),
                a2_use_correct_delivery_count=int(
                    np.asarray(final.a2_use.correct_delivery_count[lane])
                ),
                a2_use_wrong_delivery_count=int(
                    np.asarray(final.a2_use.wrong_delivery_count[lane])
                ),
                a2_use_indicator_activation_count=int(
                    np.asarray(final.a2_use.indicator_activation_count[lane])
                ),
            )
        )
    return tuple(rows)


def evaluate_response_contrast(
    *,
    config: Any,
    population: Any,
    output: Path,
    evaluation_seed: int,
    resume: bool,
) -> tuple[int, int]:
    deployments = {
        entry.outer_unit_id: load_deployment(entry, config)
        for entry in population.entries
    }
    paths = []
    for left_id in range(10):
        for right_id in range(10):
            if left_id == right_id:
                continue
            pairing = Pairing(
                deployment_mode="posterior_use",
                split="xp",
                left_outer_unit_id=left_id,
                right_outer_unit_id=right_id,
            )
            path = output / pairing.pairing_id / "branches.parquet"
            if resume and path.is_file():
                if len(read_parquet(path)) != config.evaluation.episodes_per_pairing:
                    raise RuntimeError(
                        f"Incomplete response contrast output exists: {path}"
                    )
                paths.append(path)
                continue
            if path.exists():
                raise RuntimeError(
                    f"Response contrast output already exists: {path}"
                )
            rows = contrast_pairing_batch(
                config=config,
                left=deployments[left_id],
                right=deployments[right_id],
                pairing=pairing,
                evaluation_seed=evaluation_seed,
            )
            write_parquet(path, [row.to_mapping() for row in rows])
            paths.append(path)
    rows = [
        ResponseContrastRow(**row)
        for path in paths
        for row in read_parquet(path)
    ]
    validate_response_contrast_rows(
        rows,
        evaluation_seed=evaluation_seed,
        layout=config.environment.layout,
        episodes_per_pairing=config.evaluation.episodes_per_pairing,
    )
    write_json(
        output / "summary.json", summarize_response_contrast(rows)
    )
    return len(rows), sum(row.environment_steps for row in rows)


def evaluate_development_response_contrast(
    *,
    config: Any,
    population: Any,
    output: Path,
    evaluation_seed: int,
    resume: bool,
) -> tuple[int, int]:
    """Run matched response branches on one self-paired development policy."""

    path = output / "branches.parquet"
    if resume and path.is_file():
        if len(read_parquet(path)) != config.evaluation.episodes_per_pairing:
            raise RuntimeError(
                f"Incomplete development response contrast exists: {path}"
            )
    else:
        if path.exists():
            raise RuntimeError(
                f"Development response contrast already exists: {path}"
            )
        deployment = load_deployment(population.entries[0], config)
        rows = contrast_pairing_batch(
            config=config,
            left=deployment,
            right=deployment,
            pairing=Pairing("posterior_use", "sp", 0, 0),
            evaluation_seed=evaluation_seed,
        )
        write_parquet(path, [row.to_mapping() for row in rows])
    rows = [ResponseContrastRow(**row) for row in read_parquet(path)]
    validate_development_response_contrast_rows(
        rows,
        evaluation_seed=evaluation_seed,
        layout=config.environment.layout,
        episodes_per_pairing=config.evaluation.episodes_per_pairing,
    )
    write_json(
        output / "summary.json",
        {
            "run_kind": "development",
            "scientific_readout_allowed": False,
            "evaluation_protocol": "single_policy_self_pair_response_contrast",
            **summarize_response_contrast(rows),
        },
    )
    return len(rows), sum(row.environment_steps for row in rows)


__all__ = [
    "contrast_pairing_batch",
    "evaluate_development_response_contrast",
    "evaluate_response_contrast",
]
