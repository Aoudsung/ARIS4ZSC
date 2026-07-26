"""Matched A1, A2-mask, and A2-use behavior-consistent VQBC contrast."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

from src.path_c.vqbc.config import VQBCConfig
from src.path_c.vqbc.evaluation import VQBCPopulation, standard_episode_seed
from src.path_c.vqbc.response_contrast import (
    RESPONSE_CONTRAST_SCHEMA_VERSION,
    VQBCResponseContrastRow,
    policy_effect_trigger_tolerance,
    summarize_response_contrast,
)

from .env_dock import OvercookedV2VectorEnvironment
from .vqbc_evaluation_runtime import DeploymentStep, VQBCDeploymentPolicy


class _BranchState(NamedTuple):
    environment_state: Any
    observations: Any
    left_state: Any
    right_state: Any
    raw_return: Any
    last_left_action: Any
    last_left_response_code: Any
    last_reward: Any


class _ContrastBatchState(NamedTuple):
    a1: _BranchState
    a2_mask: _BranchState
    a2_use: _BranchState
    triggered: Any
    trigger_step: Any
    trigger_tolerance: Any
    predicted_response_effect: Any
    predicted_policy_cost: Any
    predicted_net_effect: Any
    predicted_regularized_net_effect: Any
    predicted_policy_total_variation: Any
    maximum_action_net_value: Any
    executed_action_net_value: Any
    executed_action_response_value: Any
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


class _DevelopmentEntry(NamedTuple):
    config: VQBCConfig
    checkpoint_path: Path


def _advance(
    *,
    branch: _BranchState,
    left_policy: VQBCDeploymentPolicy,
    right_policy: VQBCDeploymentPolicy,
    left_step: DeploymentStep,
    right_step: DeploymentStep,
    environment: Any,
    environment_keys: Any,
    left_mask_response: Any,
) -> _BranchState:
    import jax.numpy as jnp

    joint_actions = jnp.stack((left_step.action, right_step.action), axis=-1)
    (
        next_environment_state,
        next_observations,
        rewards,
        dones,
        info,
    ) = environment.step_with_keys(
        branch.environment_state, joint_actions, environment_keys
    )
    terminal = info["terminal_observations"]
    mask = dones.reshape(
        dones.shape + (1,) * (next_observations[:, 0].ndim - 1)
    )
    left_next = jnp.where(mask, terminal[:, 0], next_observations[:, 0])
    right_next = jnp.where(mask, terminal[:, 1], next_observations[:, 1])
    left_state, left_codes = left_policy.observe(
        left_step,
        observations=branch.observations[:, 0],
        next_observations=left_next,
        rewards=rewards,
        dones=dones,
        deployment_mode="posterior_use",
        mask_response=left_mask_response,
    )
    right_state, unused_right_codes = right_policy.observe(
        right_step,
        observations=branch.observations[:, 1],
        next_observations=right_next,
        rewards=rewards,
        dones=dones,
        deployment_mode="posterior_use",
    )
    del unused_right_codes
    return _BranchState(
        environment_state=next_environment_state,
        observations=next_observations,
        left_state=left_state,
        right_state=right_state,
        raw_return=branch.raw_return + rewards,
        last_left_action=left_step.action,
        last_left_response_code=left_codes,
        last_reward=rewards,
    )


def _a1_step_from_use(use_step: DeploymentStep, key: Any) -> DeploymentStep:
    """Sample the registered response-masked policy from the common prefix."""

    import jax
    import jax.numpy as jnp

    key_array = jnp.asarray(key)
    action = (
        jax.vmap(
            lambda lane_key, logits: jax.random.categorical(
                lane_key, logits, axis=-1
            )
        )(key_array, use_step.output.mask_execution_logits)
        if key_array.ndim == 2
        else jax.random.categorical(
            key_array, use_step.output.mask_execution_logits, axis=-1
        )
    )
    return DeploymentStep(
        use_step.state,
        action,
        use_step.output,
        use_step.decision,
    )


def _optional_step(value: Any, lane: int) -> int | None:
    observed = int(value[lane])
    return None if observed < 0 else observed


def _row(
    *,
    pairing_id: str,
    episode_index: int,
    episode_seed: int,
    triggered: bool,
    trigger_step: int | None,
    diagnostics: dict[str, Any] | None,
    a1_raw_return: float,
    a2_mask_raw_return: float,
    a2_use_raw_return: float,
) -> VQBCResponseContrastRow:
    payload = diagnostics or {}
    return VQBCResponseContrastRow(
        schema_version=RESPONSE_CONTRAST_SCHEMA_VERSION,
        pairing_id=pairing_id,
        episode_index=episode_index,
        episode_seed=episode_seed,
        triggered=triggered,
        trigger_step=trigger_step,
        trigger_tolerance=payload.get("trigger_tolerance"),
        predicted_response_effect=payload.get("predicted_response_effect"),
        predicted_policy_cost=payload.get("predicted_policy_cost"),
        predicted_net_effect=payload.get("predicted_net_effect"),
        predicted_regularized_net_effect=payload.get(
            "predicted_regularized_net_effect"
        ),
        predicted_policy_total_variation=payload.get(
            "predicted_policy_total_variation"
        ),
        maximum_action_net_value=payload.get("maximum_action_net_value"),
        executed_action_net_value=payload.get("executed_action_net_value"),
        executed_action_response_value=payload.get(
            "executed_action_response_value"
        ),
        executed_action=payload.get("executed_action"),
        maximum_net_action=payload.get("maximum_net_action"),
        post_response_belief_l1=payload.get("post_response_belief_l1"),
        first_left_action_difference_step=payload.get(
            "first_left_action_difference_step"
        ),
        first_observation_difference_step=payload.get(
            "first_observation_difference_step"
        ),
        first_response_code_difference_step=payload.get(
            "first_response_code_difference_step"
        ),
        first_reward_difference_step=payload.get(
            "first_reward_difference_step"
        ),
        left_action_difference_count=payload.get("left_action_difference_count"),
        observation_difference_count=payload.get("observation_difference_count"),
        response_code_difference_count=payload.get(
            "response_code_difference_count"
        ),
        reward_difference_count=payload.get("reward_difference_count"),
        a1_raw_return=a1_raw_return,
        a2_mask_raw_return=a2_mask_raw_return,
        a2_use_raw_return=a2_use_raw_return,
    )


def _make_contrast_batch_runner(
    *,
    left_policy: VQBCDeploymentPolicy,
    right_policy: VQBCDeploymentPolicy,
    layout: str,
    count: int,
) -> Any:
    """Compile complete matched episodes and preserve control-path diagnostics."""

    import jax
    import jax.numpy as jnp

    environment = OvercookedV2VectorEnvironment.create(num_envs=count, layout=layout)

    def nan_float() -> Any:
        return jnp.full((count,), jnp.nan, dtype=jnp.float32)

    def minus_one() -> Any:
        return jnp.full((count,), -1, dtype=jnp.int32)

    def zeros_int() -> Any:
        return jnp.zeros((count,), dtype=jnp.int32)

    def initial_branch(environment_state: Any, observations: Any) -> _BranchState:
        return _BranchState(
            environment_state=environment_state,
            observations=observations,
            left_state=left_policy.initial_state(count),
            right_state=right_policy.initial_state(count),
            raw_return=jnp.zeros((count,), dtype=jnp.float32),
            last_left_action=minus_one(),
            last_left_response_code=minus_one(),
            last_reward=jnp.zeros((count,), dtype=jnp.float32),
        )

    def run_episode_batch(base_keys: Any) -> _ContrastBatchState:
        environment_state, observations = environment.reset_with_keys(base_keys)
        shared = initial_branch(environment_state, observations)
        initial = _ContrastBatchState(
            a1=shared,
            a2_mask=shared,
            a2_use=shared,
            triggered=jnp.zeros((count,), dtype=jnp.bool_),
            trigger_step=minus_one(),
            trigger_tolerance=nan_float(),
            predicted_response_effect=nan_float(),
            predicted_policy_cost=nan_float(),
            predicted_net_effect=nan_float(),
            predicted_regularized_net_effect=nan_float(),
            predicted_policy_total_variation=nan_float(),
            maximum_action_net_value=nan_float(),
            executed_action_net_value=nan_float(),
            executed_action_response_value=nan_float(),
            executed_action=minus_one(),
            maximum_net_action=minus_one(),
            post_response_belief_l1=nan_float(),
            first_left_action_difference_step=minus_one(),
            first_observation_difference_step=minus_one(),
            first_response_code_difference_step=minus_one(),
            first_reward_difference_step=minus_one(),
            left_action_difference_count=zeros_int(),
            observation_difference_count=zeros_int(),
            response_code_difference_count=zeros_int(),
            reward_difference_count=zeros_int(),
        )

        def one_step(current: _ContrastBatchState, step_index: Any) -> tuple[Any, None]:
            step_keys = jax.vmap(lambda key: jax.random.fold_in(key, step_index))(
                base_keys
            )
            left_keys = jax.vmap(lambda key: jax.random.fold_in(key, 1))(step_keys)
            right_keys = jax.vmap(lambda key: jax.random.fold_in(key, 2))(step_keys)
            environment_keys = jax.vmap(lambda key: jax.random.fold_in(key, 3))(
                step_keys
            )

            a1_left = left_policy.act(
                current.a1.left_state,
                current.a1.observations[:, 0],
                left_keys,
                "posterior_use",
            )
            a1_right = right_policy.act(
                current.a1.right_state,
                current.a1.observations[:, 1],
                right_keys,
                "posterior_use",
            )
            mask_left = left_policy.act(
                current.a2_mask.left_state,
                current.a2_mask.observations[:, 0],
                left_keys,
                "posterior_use",
            )
            mask_right = right_policy.act(
                current.a2_mask.right_state,
                current.a2_mask.observations[:, 1],
                right_keys,
                "posterior_use",
            )
            use_left = left_policy.act(
                current.a2_use.left_state,
                current.a2_use.observations[:, 0],
                left_keys,
                "posterior_use",
            )
            use_right = right_policy.act(
                current.a2_use.right_state,
                current.a2_use.observations[:, 1],
                right_keys,
                "posterior_use",
            )

            predicted_net = use_left.output.predicted_net_effect
            tolerance = policy_effect_trigger_tolerance(
                use_left.output.j_use, use_left.output.j_mask
            )
            newly_triggered = (~current.triggered) & (predicted_net > tolerance)
            a1_trigger = _a1_step_from_use(use_left, left_keys)
            a1_left = DeploymentStep(
                a1_left.state,
                jnp.where(newly_triggered, a1_trigger.action, a1_left.action),
                a1_left.output,
                a1_left.decision,
            )
            # A2-mask executes exactly the A2-use action at the fork. Its only
            # intervention is refusing the response in the belief update.
            mask_left = DeploymentStep(
                mask_left.state,
                jnp.where(newly_triggered, use_left.action, mask_left.action),
                mask_left.output,
                mask_left.decision,
            )

            next_a1 = _advance(
                branch=current.a1,
                left_policy=left_policy,
                right_policy=right_policy,
                left_step=a1_left,
                right_step=a1_right,
                environment=environment,
                environment_keys=environment_keys,
                left_mask_response=newly_triggered,
            )
            next_mask = _advance(
                branch=current.a2_mask,
                left_policy=left_policy,
                right_policy=right_policy,
                left_step=mask_left,
                right_step=mask_right,
                environment=environment,
                environment_keys=environment_keys,
                left_mask_response=newly_triggered,
            )
            next_use = _advance(
                branch=current.a2_use,
                left_policy=left_policy,
                right_policy=right_policy,
                left_step=use_left,
                right_step=use_right,
                environment=environment,
                environment_keys=environment_keys,
                left_mask_response=False,
            )

            active = current.triggered | newly_triggered
            action_difference = active & (
                next_mask.last_left_action != next_use.last_left_action
            )
            observation_axes = tuple(range(1, next_use.observations.ndim))
            observation_difference = active & jnp.any(
                next_mask.observations != next_use.observations,
                axis=observation_axes,
            )
            response_difference = active & (
                next_mask.last_left_response_code
                != next_use.last_left_response_code
            )
            reward_difference = active & (
                jnp.abs(next_mask.last_reward - next_use.last_reward) > 1.0e-6
            )

            def capture(old: Any, new: Any) -> Any:
                return jnp.where(newly_triggered, new, old)

            def first(old: Any, condition: Any) -> Any:
                return jnp.where((old < 0) & condition, step_index, old)

            use_belief = jax.nn.softmax(
                next_use.left_state.slot_log_belief, axis=-1
            )
            mask_belief = jax.nn.softmax(
                next_mask.left_state.slot_log_belief, axis=-1
            )
            belief_l1 = jnp.sum(jnp.abs(use_belief - mask_belief), axis=-1)
            return _ContrastBatchState(
                a1=next_a1,
                a2_mask=next_mask,
                a2_use=next_use,
                triggered=active,
                trigger_step=capture(current.trigger_step, step_index),
                trigger_tolerance=capture(current.trigger_tolerance, tolerance),
                predicted_response_effect=capture(
                    current.predicted_response_effect,
                    use_left.output.predicted_response_effect,
                ),
                predicted_policy_cost=capture(
                    current.predicted_policy_cost,
                    use_left.output.predicted_policy_cost,
                ),
                predicted_net_effect=capture(
                    current.predicted_net_effect, predicted_net
                ),
                predicted_regularized_net_effect=capture(
                    current.predicted_regularized_net_effect,
                    use_left.output.predicted_regularized_net_effect,
                ),
                predicted_policy_total_variation=capture(
                    current.predicted_policy_total_variation,
                    use_left.output.predicted_policy_total_variation,
                ),
                maximum_action_net_value=capture(
                    current.maximum_action_net_value,
                    use_left.decision.maximum_action_net_value,
                ),
                executed_action_net_value=capture(
                    current.executed_action_net_value,
                    use_left.decision.executed_action_net_value,
                ),
                executed_action_response_value=capture(
                    current.executed_action_response_value,
                    use_left.decision.executed_action_response_value,
                ),
                executed_action=capture(current.executed_action, use_left.action),
                maximum_net_action=capture(
                    current.maximum_net_action,
                    jnp.argmax(use_left.output.per_action_net_value, axis=-1),
                ),
                post_response_belief_l1=capture(
                    current.post_response_belief_l1, belief_l1
                ),
                first_left_action_difference_step=first(
                    current.first_left_action_difference_step, action_difference
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
                    current.first_reward_difference_step, reward_difference
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
            one_step, initial, jnp.arange(400, dtype=jnp.int32)
        )
        del unused
        return final

    return jax.jit(run_episode_batch)


def _row_from_final(
    *,
    final: _ContrastBatchState,
    lane: int,
    pairing_id: str,
    episode_index: int,
    episode_seed: int,
) -> VQBCResponseContrastRow:
    import numpy as np

    did_trigger = bool(np.asarray(final.triggered[lane]))
    diagnostics = None
    trigger_step = None
    if did_trigger:
        trigger_step = int(np.asarray(final.trigger_step[lane]))
        diagnostics = {
            "trigger_tolerance": float(np.asarray(final.trigger_tolerance[lane])),
            "predicted_response_effect": float(
                np.asarray(final.predicted_response_effect[lane])
            ),
            "predicted_policy_cost": float(
                np.asarray(final.predicted_policy_cost[lane])
            ),
            "predicted_net_effect": float(
                np.asarray(final.predicted_net_effect[lane])
            ),
            "predicted_regularized_net_effect": float(
                np.asarray(final.predicted_regularized_net_effect[lane])
            ),
            "predicted_policy_total_variation": float(
                np.asarray(final.predicted_policy_total_variation[lane])
            ),
            "maximum_action_net_value": float(
                np.asarray(final.maximum_action_net_value[lane])
            ),
            "executed_action_net_value": float(
                np.asarray(final.executed_action_net_value[lane])
            ),
            "executed_action_response_value": float(
                np.asarray(final.executed_action_response_value[lane])
            ),
            "executed_action": int(np.asarray(final.executed_action[lane])),
            "maximum_net_action": int(np.asarray(final.maximum_net_action[lane])),
            "post_response_belief_l1": float(
                np.asarray(final.post_response_belief_l1[lane])
            ),
            "first_left_action_difference_step": _optional_step(
                np.asarray(final.first_left_action_difference_step), lane
            ),
            "first_observation_difference_step": _optional_step(
                np.asarray(final.first_observation_difference_step), lane
            ),
            "first_response_code_difference_step": _optional_step(
                np.asarray(final.first_response_code_difference_step), lane
            ),
            "first_reward_difference_step": _optional_step(
                np.asarray(final.first_reward_difference_step), lane
            ),
            "left_action_difference_count": int(
                np.asarray(final.left_action_difference_count[lane])
            ),
            "observation_difference_count": int(
                np.asarray(final.observation_difference_count[lane])
            ),
            "response_code_difference_count": int(
                np.asarray(final.response_code_difference_count[lane])
            ),
            "reward_difference_count": int(
                np.asarray(final.reward_difference_count[lane])
            ),
        }
    return _row(
        pairing_id=pairing_id,
        episode_index=episode_index,
        episode_seed=episode_seed,
        triggered=did_trigger,
        trigger_step=trigger_step,
        diagnostics=diagnostics,
        a1_raw_return=float(np.asarray(final.a1.raw_return[lane])),
        a2_mask_raw_return=float(np.asarray(final.a2_mask.raw_return[lane])),
        a2_use_raw_return=float(np.asarray(final.a2_use.raw_return[lane])),
    )


def _evaluate_response_contrast_episode(
    *,
    left_policy: VQBCDeploymentPolicy,
    right_policy: VQBCDeploymentPolicy,
    population_id: str,
    layout: str,
    left_outer_unit_id: int,
    right_outer_unit_id: int,
    episode_index: int,
) -> VQBCResponseContrastRow:
    import jax

    episode_seed = standard_episode_seed(
        population_id=population_id,
        layout=layout,
        left_outer_unit_id=left_outer_unit_id,
        right_outer_unit_id=right_outer_unit_id,
        episode_index=episode_index,
    )
    runner = _make_contrast_batch_runner(
        left_policy=left_policy,
        right_policy=right_policy,
        layout=layout,
        count=1,
    )
    final = runner(jax.random.PRNGKey(episode_seed)[None, ...])
    return _row_from_final(
        final=final,
        lane=0,
        pairing_id=f"{left_outer_unit_id:02d}_to_{right_outer_unit_id:02d}",
        episode_index=episode_index,
        episode_seed=episode_seed,
    )


def evaluate_response_contrast_episode(
    *,
    population: VQBCPopulation,
    left_outer_unit_id: int,
    right_outer_unit_id: int,
    episode_index: int,
) -> VQBCResponseContrastRow:
    if left_outer_unit_id == right_outer_unit_id:
        raise ValueError("The response contrast is defined on directed cross-play.")
    return _evaluate_response_contrast_episode(
        left_policy=VQBCDeploymentPolicy.load(population.entry(left_outer_unit_id)),
        right_policy=VQBCDeploymentPolicy.load(population.entry(right_outer_unit_id)),
        population_id=population.population_id,
        layout=population.layout,
        left_outer_unit_id=left_outer_unit_id,
        right_outer_unit_id=right_outer_unit_id,
        episode_index=episode_index,
    )


def _write_response_contrast(
    rows: list[VQBCResponseContrastRow], output_root: Path, *, run_kind: str
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "rows.jsonl"
    temporary = rows_path.with_name(f".{rows_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_mapping(), sort_keys=True) + "\n")
    temporary.replace(rows_path)
    summary_path = output_root / "summary.json"
    summary = {**summarize_response_contrast(rows), "run_kind": run_kind}
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(summary_path)
    return {"rows": rows_path, "summary": summary_path}


def run_response_contrast(population: VQBCPopulation) -> dict[str, Path]:
    rows: list[VQBCResponseContrastRow] = []
    for left in range(10):
        for right in range(10):
            if left == right:
                continue
            left_policy = VQBCDeploymentPolicy.load(population.entry(left))
            right_policy = VQBCDeploymentPolicy.load(population.entry(right))
            runner = _make_contrast_batch_runner(
                left_policy=left_policy,
                right_policy=right_policy,
                layout=population.layout,
                count=1,
            )
            import jax

            for episode_index in range(500):
                episode_seed = standard_episode_seed(
                    population_id=population.population_id,
                    layout=population.layout,
                    left_outer_unit_id=left,
                    right_outer_unit_id=right,
                    episode_index=episode_index,
                )
                final = runner(jax.random.PRNGKey(episode_seed)[None, ...])
                rows.append(
                    _row_from_final(
                        final=final,
                        lane=0,
                        pairing_id=f"{left:02d}_to_{right:02d}",
                        episode_index=episode_index,
                        episode_seed=episode_seed,
                    )
                )
    return _write_response_contrast(
        rows, population.output_root / "response_contrast", run_kind="formal"
    )


def run_development_response_contrast(
    *,
    resolved_config_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Run the matched response-use contrast for one development checkpoint."""

    config_path = Path(resolved_config_path).resolve()
    config = VQBCConfig.from_mapping(
        json.loads(config_path.read_text(encoding="utf-8")),
        base_dir=config_path.parent,
    )
    if config.run_kind != "development" or config.outer_unit is not None:
        raise ValueError("The one-checkpoint contrast accepts only a development config.")
    policy = VQBCDeploymentPolicy.load(
        _DevelopmentEntry(
            config=config,
            checkpoint_path=Path(checkpoint_path).resolve(),
        )
    )
    population_id = (
        f"path_c_vqbc_v4_2_development_{config.environment.layout}_"
        f"{config.seeds.model_seed}"
    )
    import jax
    import jax.numpy as jnp

    rows: list[VQBCResponseContrastRow] = []
    runners: dict[int, Any] = {}
    batch_size = config.environment.num_envs
    for start in range(0, config.evaluation.episodes_per_pairing, batch_size):
        indexes = tuple(
            range(
                start,
                min(start + batch_size, config.evaluation.episodes_per_pairing),
            )
        )
        count = len(indexes)
        if count not in runners:
            runners[count] = _make_contrast_batch_runner(
                left_policy=policy,
                right_policy=policy,
                layout=config.environment.layout,
                count=count,
            )
        seeds = tuple(
            standard_episode_seed(
                population_id=population_id,
                layout=config.environment.layout,
                left_outer_unit_id=0,
                right_outer_unit_id=0,
                episode_index=episode_index,
            )
            for episode_index in indexes
        )
        base_keys = jax.vmap(jax.random.PRNGKey)(
            jnp.asarray(seeds, dtype=jnp.uint32)
        )
        final = runners[count](base_keys)
        for lane, (episode_index, episode_seed) in enumerate(
            zip(indexes, seeds, strict=True)
        ):
            rows.append(
                _row_from_final(
                    final=final,
                    lane=lane,
                    pairing_id="00_to_00",
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                )
            )
    destination = (
        config.output_root / "development_response_contrast"
        if output_root is None
        else Path(output_root).resolve()
    )
    return _write_response_contrast(rows, destination, run_kind="development")


__all__ = [
    "evaluate_response_contrast_episode",
    "run_development_response_contrast",
    "run_response_contrast",
]
