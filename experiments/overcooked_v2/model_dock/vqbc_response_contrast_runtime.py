"""Matched A1, A2-mask, and A2-use response contrast for VQBC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

from src.path_c.vqbc.config import VQBCConfig
from src.path_c.vqbc.evaluation import (
    VQBCPopulation,
    standard_episode_seed,
)
from src.path_c.vqbc.policy import regularized_policy
from src.path_c.vqbc.response_contrast import (
    RESPONSE_CONTRAST_SCHEMA_VERSION,
    VQBCResponseContrastRow,
    information_trigger_tolerance,
    summarize_response_contrast,
)

from .env_dock import OvercookedV2VectorEnvironment
from .vqbc_evaluation_runtime import (
    DeploymentStep,
    VQBCDeploymentPolicy,
)


class _BranchState(NamedTuple):
    environment_state: Any
    observations: Any
    left_state: Any
    right_state: Any
    raw_return: Any


class _DevelopmentEntry(NamedTuple):
    config: VQBCConfig
    checkpoint_path: Path


def _step_keys(base_key: Any, step_index: int) -> tuple[Any, Any, Any]:
    import jax

    root = jax.random.fold_in(base_key, step_index)
    return (
        jax.random.fold_in(root, 1),
        jax.random.fold_in(root, 2),
        jax.random.fold_in(root, 3)[None, ...],
    )


def _advance(
    *,
    branch: _BranchState,
    left_policy: VQBCDeploymentPolicy,
    right_policy: VQBCDeploymentPolicy,
    left_step: DeploymentStep,
    right_step: DeploymentStep,
    environment: Any,
    environment_keys: Any,
    left_mask_response: bool,
) -> _BranchState:
    import jax.numpy as jnp

    joint_actions = jnp.stack(
        (left_step.action, right_step.action), axis=-1
    )
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
    left_state, unused_left_codes = left_policy.observe(
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
    del unused_left_codes, unused_right_codes
    return _BranchState(
        environment_state=next_environment_state,
        observations=next_observations,
        left_state=left_state,
        right_state=right_state,
        raw_return=branch.raw_return + rewards,
    )


def _finish_branch(
    *,
    branch: _BranchState,
    start_step: int,
    base_key: Any,
    left_policy: VQBCDeploymentPolicy,
    right_policy: VQBCDeploymentPolicy,
    environment: Any,
) -> _BranchState:
    for step_index in range(start_step, 400):
        left_key, right_key, environment_keys = _step_keys(
            base_key, step_index
        )
        left_step = left_policy.act(
            branch.left_state,
            branch.observations[:, 0],
            left_key,
            "posterior_use",
        )
        right_step = right_policy.act(
            branch.right_state,
            branch.observations[:, 1],
            right_key,
            "posterior_use",
        )
        branch = _advance(
            branch=branch,
            left_policy=left_policy,
            right_policy=right_policy,
            left_step=left_step,
            right_step=right_step,
            environment=environment,
            environment_keys=environment_keys,
            left_mask_response=False,
        )
    return branch


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
    import jax.numpy as jnp
    import numpy as np

    episode_seed = standard_episode_seed(
        population_id=population_id,
        layout=layout,
        left_outer_unit_id=left_outer_unit_id,
        right_outer_unit_id=right_outer_unit_id,
        episode_index=episode_index,
    )
    base_key = jax.random.PRNGKey(episode_seed)
    environment = OvercookedV2VectorEnvironment.create(
        num_envs=1, layout=layout
    )
    environment_state, observations = environment.reset_with_keys(
        base_key[None, ...]
    )
    shared = _BranchState(
        environment_state=environment_state,
        observations=observations,
        left_state=left_policy.initial_state(1),
        right_state=right_policy.initial_state(1),
        raw_return=jnp.zeros((1,), dtype=jnp.float32),
    )
    for step_index in range(400):
        left_key, right_key, environment_keys = _step_keys(
            base_key, step_index
        )
        left_step = left_policy.act(
            shared.left_state,
            shared.observations[:, 0],
            left_key,
            "posterior_use",
        )
        right_step = right_policy.act(
            shared.right_state,
            shared.observations[:, 1],
            right_key,
            "posterior_use",
        )
        information_net_value = (
            left_step.output.j_use
            - jnp.max(left_step.output.j_mask, axis=-1, keepdims=True)
        )
        maximum = float(np.asarray(jnp.max(information_net_value)))
        tolerance = float(
            np.asarray(
                information_trigger_tolerance(
                    left_step.output.j_use, left_step.output.j_mask
                )
            )
        )
        if maximum <= tolerance:
            shared = _advance(
                branch=shared,
                left_policy=left_policy,
                right_policy=right_policy,
                left_step=left_step,
                right_step=right_step,
                environment=environment,
                environment_keys=environment_keys,
                left_mask_response=False,
            )
            continue
        alpha = jnp.exp(left_step.state.log_temperature)
        a1_policy = regularized_policy(
            left_step.decision.reference_logits,
            left_step.output.j_mask,
            alpha,
        )
        a1_action = jax.random.categorical(
            left_key, a1_policy.logits, axis=-1
        )
        a1_left_step = DeploymentStep(
            left_step.state,
            a1_action,
            left_step.output,
            left_step.decision,
        )
        a1 = _advance(
            branch=shared,
            left_policy=left_policy,
            right_policy=right_policy,
            left_step=a1_left_step,
            right_step=right_step,
            environment=environment,
            environment_keys=environment_keys,
            left_mask_response=False,
        )
        a2_mask = _advance(
            branch=shared,
            left_policy=left_policy,
            right_policy=right_policy,
            left_step=left_step,
            right_step=right_step,
            environment=environment,
            environment_keys=environment_keys,
            left_mask_response=True,
        )
        a2_use = _advance(
            branch=shared,
            left_policy=left_policy,
            right_policy=right_policy,
            left_step=left_step,
            right_step=right_step,
            environment=environment,
            environment_keys=environment_keys,
            left_mask_response=False,
        )
        a1 = _finish_branch(
            branch=a1,
            start_step=step_index + 1,
            base_key=base_key,
            left_policy=left_policy,
            right_policy=right_policy,
            environment=environment,
        )
        a2_mask = _finish_branch(
            branch=a2_mask,
            start_step=step_index + 1,
            base_key=base_key,
            left_policy=left_policy,
            right_policy=right_policy,
            environment=environment,
        )
        a2_use = _finish_branch(
            branch=a2_use,
            start_step=step_index + 1,
            base_key=base_key,
            left_policy=left_policy,
            right_policy=right_policy,
            environment=environment,
        )
        return VQBCResponseContrastRow(
            schema_version=RESPONSE_CONTRAST_SCHEMA_VERSION,
            pairing_id=f"{left_outer_unit_id:02d}_to_{right_outer_unit_id:02d}",
            episode_index=episode_index,
            episode_seed=episode_seed,
            triggered=True,
            trigger_step=step_index,
            predicted_information_net_value=maximum,
            a1_raw_return=float(np.asarray(a1.raw_return[0])),
            a2_mask_raw_return=float(np.asarray(a2_mask.raw_return[0])),
            a2_use_raw_return=float(np.asarray(a2_use.raw_return[0])),
        )
    raw_return = float(np.asarray(shared.raw_return[0]))
    return VQBCResponseContrastRow(
        schema_version=RESPONSE_CONTRAST_SCHEMA_VERSION,
        pairing_id=f"{left_outer_unit_id:02d}_to_{right_outer_unit_id:02d}",
        episode_index=episode_index,
        episode_seed=episode_seed,
        triggered=False,
        trigger_step=None,
        predicted_information_net_value=None,
        a1_raw_return=raw_return,
        a2_mask_raw_return=raw_return,
        a2_use_raw_return=raw_return,
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
        left_policy=VQBCDeploymentPolicy.load(
            population.entry(left_outer_unit_id)
        ),
        right_policy=VQBCDeploymentPolicy.load(
            population.entry(right_outer_unit_id)
        ),
        population_id=population.population_id,
        layout=population.layout,
        left_outer_unit_id=left_outer_unit_id,
        right_outer_unit_id=right_outer_unit_id,
        episode_index=episode_index,
    )


def run_response_contrast(population: VQBCPopulation) -> dict[str, Path]:
    rows = []
    for left in range(10):
        for right in range(10):
            if left == right:
                continue
            for episode_index in range(500):
                rows.append(
                    evaluate_response_contrast_episode(
                        population=population,
                        left_outer_unit_id=left,
                        right_outer_unit_id=right,
                        episode_index=episode_index,
                    )
                )
    output_root = population.output_root / "response_contrast"
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "rows.jsonl"
    temporary = rows_path.with_name(f".{rows_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_mapping(), sort_keys=True) + "\n")
    temporary.replace(rows_path)
    summary_path = output_root / "summary.json"
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(
        json.dumps(
            summarize_response_contrast(rows), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return {"rows": rows_path, "summary": summary_path}


def _make_development_contrast_batch_runner(
    *,
    policy: VQBCDeploymentPolicy,
    layout: str,
    count: int,
) -> Any:
    import jax
    import jax.numpy as jnp

    environment = OvercookedV2VectorEnvironment.create(
        num_envs=count, layout=layout
    )

    def run_episode_batch(base_keys: Any) -> tuple[Any, ...]:
        environment_state, observations = environment.reset_with_keys(
            base_keys
        )
        shared = _BranchState(
            environment_state=environment_state,
            observations=observations,
            left_state=policy.initial_state(count),
            right_state=policy.initial_state(count),
            raw_return=jnp.zeros((count,), dtype=jnp.float32),
        )
        initial = (
            shared,
            shared,
            shared,
            jnp.zeros((count,), dtype=jnp.bool_),
            jnp.full((count,), -1, dtype=jnp.int32),
            jnp.full((count,), jnp.nan, dtype=jnp.float32),
        )

        def one_step(current: tuple[Any, ...], step_index: Any) -> tuple[Any, None]:
            (
                a1,
                a2_mask,
                a2_use,
                triggered,
                trigger_step,
                predicted_information_net_value,
            ) = current
            step_keys = jax.vmap(
                lambda key: jax.random.fold_in(key, step_index)
            )(base_keys)
            left_keys = jax.vmap(
                lambda key: jax.random.fold_in(key, 1)
            )(step_keys)
            right_keys = jax.vmap(
                lambda key: jax.random.fold_in(key, 2)
            )(step_keys)
            environment_keys = jax.vmap(
                lambda key: jax.random.fold_in(key, 3)
            )(step_keys)

            a1_left = policy.act(
                a1.left_state,
                a1.observations[:, 0],
                left_keys,
                "posterior_use",
            )
            a1_right = policy.act(
                a1.right_state,
                a1.observations[:, 1],
                right_keys,
                "posterior_use",
            )
            mask_left = policy.act(
                a2_mask.left_state,
                a2_mask.observations[:, 0],
                left_keys,
                "posterior_use",
            )
            mask_right = policy.act(
                a2_mask.right_state,
                a2_mask.observations[:, 1],
                right_keys,
                "posterior_use",
            )
            use_left = policy.act(
                a2_use.left_state,
                a2_use.observations[:, 0],
                left_keys,
                "posterior_use",
            )
            use_right = policy.act(
                a2_use.right_state,
                a2_use.observations[:, 1],
                right_keys,
                "posterior_use",
            )

            information_net_value = use_left.output.j_use - jnp.max(
                use_left.output.j_mask, axis=-1, keepdims=True
            )
            maximum = jnp.max(information_net_value, axis=-1)
            tolerance = information_trigger_tolerance(
                use_left.output.j_use, use_left.output.j_mask
            )
            newly_triggered = (~triggered) & (maximum > tolerance)
            a1_distribution = regularized_policy(
                use_left.decision.reference_logits,
                use_left.output.j_mask,
                jnp.exp(use_left.state.log_temperature),
            )
            a1_trigger_action = jax.vmap(
                lambda key, logits: jax.random.categorical(
                    key, logits, axis=-1
                )
            )(left_keys, a1_distribution.logits)
            a1_left = DeploymentStep(
                a1_left.state,
                jnp.where(
                    newly_triggered,
                    a1_trigger_action,
                    a1_left.action,
                ),
                a1_left.output,
                a1_left.decision,
            )
            mask_left = DeploymentStep(
                mask_left.state,
                jnp.where(
                    newly_triggered,
                    use_left.action,
                    mask_left.action,
                ),
                mask_left.output,
                mask_left.decision,
            )

            a1 = _advance(
                branch=a1,
                left_policy=policy,
                right_policy=policy,
                left_step=a1_left,
                right_step=a1_right,
                environment=environment,
                environment_keys=environment_keys,
                left_mask_response=False,
            )
            a2_mask = _advance(
                branch=a2_mask,
                left_policy=policy,
                right_policy=policy,
                left_step=mask_left,
                right_step=mask_right,
                environment=environment,
                environment_keys=environment_keys,
                left_mask_response=newly_triggered,
            )
            a2_use = _advance(
                branch=a2_use,
                left_policy=policy,
                right_policy=policy,
                left_step=use_left,
                right_step=use_right,
                environment=environment,
                environment_keys=environment_keys,
                left_mask_response=False,
            )
            return (
                a1,
                a2_mask,
                a2_use,
                triggered | newly_triggered,
                jnp.where(newly_triggered, step_index, trigger_step),
                jnp.where(
                    newly_triggered,
                    maximum,
                    predicted_information_net_value,
                ),
            ), None

        final, unused = jax.lax.scan(
            one_step, initial, jnp.arange(400, dtype=jnp.int32)
        )
        del unused
        return (
            final[3],
            final[4],
            final[5],
            final[0].raw_return,
            final[1].raw_return,
            final[2].raw_return,
        )

    return jax.jit(run_episode_batch)


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
        raise ValueError(
            "The one-checkpoint contrast accepts only a development config."
        )
    checkpoint = Path(checkpoint_path).resolve()
    policy = VQBCDeploymentPolicy.load(
        _DevelopmentEntry(config=config, checkpoint_path=checkpoint)
    )
    population_id = (
        f"path_c_vqbc_v4_development_{config.environment.layout}_"
        f"{config.seeds.model_seed}"
    )
    import jax
    import jax.numpy as jnp
    import numpy as np

    rows = []
    runners = {}
    batch_size = config.environment.num_envs
    for start in range(
        0, config.evaluation.episodes_per_pairing, batch_size
    ):
        indexes = tuple(
            range(
                start,
                min(
                    start + batch_size,
                    config.evaluation.episodes_per_pairing,
                ),
            )
        )
        count = len(indexes)
        if count not in runners:
            runners[count] = _make_development_contrast_batch_runner(
                policy=policy,
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
        (
            triggered,
            trigger_steps,
            predicted_values,
            a1_returns,
            a2_mask_returns,
            a2_use_returns,
        ) = runners[count](base_keys)
        triggered = np.asarray(triggered)
        trigger_steps = np.asarray(trigger_steps)
        predicted_values = np.asarray(predicted_values)
        a1_returns = np.asarray(a1_returns)
        a2_mask_returns = np.asarray(a2_mask_returns)
        a2_use_returns = np.asarray(a2_use_returns)
        for lane, (episode_index, episode_seed) in enumerate(
            zip(indexes, seeds, strict=True)
        ):
            did_trigger = bool(triggered[lane])
            rows.append(
                VQBCResponseContrastRow(
                    schema_version=RESPONSE_CONTRAST_SCHEMA_VERSION,
                    pairing_id="00_to_00",
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    triggered=did_trigger,
                    trigger_step=(
                        int(trigger_steps[lane])
                        if did_trigger
                        else None
                    ),
                    predicted_information_net_value=(
                        float(predicted_values[lane])
                        if did_trigger
                        else None
                    ),
                    a1_raw_return=float(a1_returns[lane]),
                    a2_mask_raw_return=float(a2_mask_returns[lane]),
                    a2_use_raw_return=float(a2_use_returns[lane]),
                )
            )
    destination = (
        config.output_root / "development_response_contrast"
        if output_root is None
        else Path(output_root).resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    rows_path = destination / "rows.jsonl"
    temporary = rows_path.with_name(f".{rows_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_mapping(), sort_keys=True) + "\n")
    temporary.replace(rows_path)
    summary_path = destination / "summary.json"
    summary = {
        **summarize_response_contrast(rows),
        "run_kind": "development",
    }
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return {"rows": rows_path, "summary": summary_path}


__all__ = [
    "evaluate_response_contrast_episode",
    "run_development_response_contrast",
    "run_response_contrast",
]
