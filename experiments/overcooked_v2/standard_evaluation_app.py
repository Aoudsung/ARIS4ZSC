"""Standard 10x10 SP/XP evaluation runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.overcooked_v2.deployment import (
    Deployment,
    load_deployment,
    reset_deployment_state,
    update_deployment_after_transition,
)
from experiments.overcooked_v2.official_adapter import VectorEnvironment
from src.path_c.evaluation import (
    EpisodeRow,
    Pairing,
    standard_episode_seed,
    standard_pairings,
    summarize_development_rows,
    summarize_standard_rows,
    validate_development_rows,
    validate_standard_rows,
)
from src.path_c.method import policy_effect_trigger_tolerance
from src.path_c.runner import policy_action
from src.path_c.storage import read_parquet, write_json, write_jsonl, write_parquet


def _jsonl_row_count(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            json.loads(line)
            count += 1
    return count


def pairing_batch(
    *,
    config: Any,
    left: Deployment,
    right: Deployment,
    pairing: Pairing,
    population_name: str,
    evaluation_seed: int,
) -> tuple[tuple[EpisodeRow, ...], Iterable[Mapping[str, Any]]]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    episode_count = config.evaluation.episodes_per_pairing
    template = VectorEnvironment.create(config)
    environment = VectorEnvironment(
        environment=template.environment,
        num_envs=episode_count,
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
            for index in range(episode_count)
        ],
        dtype=np.uint32,
    )
    root_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(root_keys)
    environment_state, observations = environment.reset_with_keys(reset_keys)
    left_state = reset_deployment_state(
        left, batch_size=episode_count, config=config
    )
    right_state = reset_deployment_state(
        right, batch_size=episode_count, config=config
    )
    left_functions = left.functions()
    right_functions = right.functions()

    def one_step(carry: tuple[Any, ...], step: Any) -> tuple[Any, tuple[Any, ...]]:
        (
            current_environment,
            current_observations,
            current_left,
            current_right,
        ) = carry
        left_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 1 + 3 * step)
        )(root_keys)
        right_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 2 + 3 * step)
        )(root_keys)
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 3 + 3 * step)
        )(root_keys)
        stepped_left, left_action, left_output, left_record, unused_left_generic = (
            policy_action(
                functions=left_functions,
                params={"official": left.online_params, "heads": left.head_params},
                policy_state=current_left,
                observations=current_observations[:, 0],
                key=left_keys,
                deployment_mode=pairing.deployment_mode,
                gamma=config.training.gamma,
                uncertainty_penalty=config.model.uncertainty_penalty,
                behavior_exploration_mix=0.0,
                behavior_uniform_floor=0.0,
            )
        )
        stepped_right, right_action, right_output, right_record, unused_right_generic = (
            policy_action(
                functions=right_functions,
                params={"official": right.online_params, "heads": right.head_params},
                policy_state=current_right,
                observations=current_observations[:, 1],
                key=right_keys,
                deployment_mode=pairing.deployment_mode,
                gamma=config.training.gamma,
                uncertainty_penalty=config.model.uncertainty_penalty,
                behavior_exploration_mix=0.0,
                behavior_uniform_floor=0.0,
            )
        )
        del unused_left_generic, unused_right_generic
        actions = jnp.stack((left_action, right_action), axis=-1)
        (
            next_environment,
            next_observations,
            rewards,
            dones,
            info,
        ) = environment.step_with_keys(
            current_environment, actions, environment_keys
        )
        terminal = info["terminal_observations"]
        observation_mask = dones.reshape(
            dones.shape + (1,) * (next_observations.ndim - 2)
        )
        response_next = jnp.where(
            observation_mask[:, None, ...], terminal, next_observations
        )
        next_left, left_response = update_deployment_after_transition(
            deployment=left,
            functions=left_functions,
            state=stepped_left,
            output=left_output,
            observations=current_observations[:, 0],
            actions=left_action,
            next_observations=response_next[:, 0],
            rewards=rewards,
            dones=dones,
            deployment_mode=pairing.deployment_mode,
            terminal_response=config.model.response_count - 1,
        )
        next_right, right_response = update_deployment_after_transition(
            deployment=right,
            functions=right_functions,
            state=stepped_right,
            output=right_output,
            observations=current_observations[:, 1],
            actions=right_action,
            next_observations=response_next[:, 1],
            rewards=rewards,
            dones=dones,
            deployment_mode=pairing.deployment_mode,
            terminal_response=config.model.response_count - 1,
        )
        return (
            next_environment,
            next_observations,
            next_left,
            next_right,
        ), (
            rewards,
            info["correct_delivery"],
            info["wrong_delivery"],
            info["indicator_activation"],
            left_record,
            right_record,
            left_response,
            right_response,
        )

    unused_final, recorded = jax.lax.scan(
        one_step,
        (environment_state, observations, left_state, right_state),
        jnp.arange(config.environment.episode_steps),
    )
    del unused_final
    (
        rewards,
        correct,
        wrong,
        indicator,
        left_records,
        right_records,
        left_codes,
        right_codes,
    ) = recorded
    trigger_left = (
        left_records.executed_action_predicted_gain_lower_score
        > policy_effect_trigger_tolerance(left_records.j_use, left_records.j_mask)
    ) & (
        left_records.executed_action_expected_next_policy_tv
        >= config.evaluation.response_policy_tv_minimum
    )
    trigger_right = (
        right_records.executed_action_predicted_gain_lower_score
        > policy_effect_trigger_tolerance(right_records.j_use, right_records.j_mask)
    ) & (
        right_records.executed_action_expected_next_policy_tv
        >= config.evaluation.response_policy_tv_minimum
    )
    response_count = config.model.response_count

    def count_codes(left_values: Any, right_values: Any) -> Any:
        return jnp.bincount(
            jnp.concatenate((left_values, right_values)),
            length=response_count,
        )

    counts = jax.vmap(count_codes, in_axes=(1, 1))(left_codes, right_codes)
    raw_returns = np.asarray(jnp.sum(rewards, axis=0))
    correct_totals = np.asarray(jnp.sum(correct, axis=0))
    wrong_totals = np.asarray(jnp.sum(wrong, axis=0))
    indicator_totals = np.asarray(jnp.sum(indicator, axis=0))
    cumulative_kl = np.asarray(
        jnp.sum(left_records.kl_divergence + right_records.kl_divergence, axis=0)
    )
    deviations = np.asarray(
        jnp.sum(
            (
                left_records.action
                != left_records.reference_greedy_action
            ).astype(jnp.int32)
            + (
                right_records.action
                != right_records.reference_greedy_action
            ).astype(jnp.int32),
            axis=0,
        )
    )
    class_means = np.asarray(
        jnp.mean(
            0.5
            * (
                left_records.value_class_count
                + right_records.value_class_count
            ),
            axis=0,
        )
    )
    entropy_means = np.asarray(
        jnp.mean(
            0.5
            * (left_records.belief_entropy + right_records.belief_entropy),
            axis=0,
        )
    )
    next_policy_tv_means = np.asarray(
        jnp.mean(
            0.5
            * (
                left_records.predicted_next_policy_total_variation
                + right_records.predicted_next_policy_total_variation
            ),
            axis=0,
        )
    )
    trigger_counts = np.asarray(
        jnp.sum(
            trigger_left.astype(jnp.int32) + trigger_right.astype(jnp.int32),
            axis=0,
        )
    )
    counts_host = np.asarray(counts)
    rows = tuple(
        EpisodeRow(
            population=population_name,
            layout=config.environment.layout,
            deployment_mode=pairing.deployment_mode,
            split=pairing.split,
            pairing_id=pairing.pairing_id,
            left_outer_unit_id=pairing.left_outer_unit_id,
            right_outer_unit_id=pairing.right_outer_unit_id,
            episode_index=index,
            episode_seed=int(seeds[index]),
            environment_steps=config.environment.episode_steps,
            raw_return=float(raw_returns[index]),
            correct_delivery_count=int(correct_totals[index]),
            wrong_delivery_count=int(wrong_totals[index]),
            indicator_activation_count=int(indicator_totals[index]),
            positive_policy_mediated_effect_count=int(
                trigger_counts[index]
            ),
            cumulative_kl=float(cumulative_kl[index]),
            reference_action_deviation_count=int(deviations[index]),
            mean_value_class_count=float(class_means[index]),
            mean_belief_entropy=float(entropy_means[index]),
            mean_predicted_next_policy_tv=float(
                next_policy_tv_means[index]
            ),
            response_code_counts=tuple(
                int(value) for value in counts_host[index]
            ),
        )
        for index in range(episode_count)
    )

    host_records = {
        "left": jax.tree_util.tree_map(lambda value: np.asarray(value), left_records),
        "right": jax.tree_util.tree_map(lambda value: np.asarray(value), right_records),
    }
    host_codes = {
        "left": np.asarray(left_codes),
        "right": np.asarray(right_codes),
    }

    def decisions() -> Iterable[Mapping[str, Any]]:
        for step in range(config.environment.episode_steps):
            for episode_index in range(episode_count):
                for side in ("left", "right"):
                    record = host_records[side]
                    yield {
                        "deployment_mode": pairing.deployment_mode,
                        "pairing_id": pairing.pairing_id,
                        "episode_index": episode_index,
                        "episode_seed": int(seeds[episode_index]),
                        "step": step,
                        "side": side,
                        "action": int(record.action[step, episode_index]),
                        "reference_logits": record.reference_logits[
                            step, episode_index
                        ].tolist(),
                        "execution_logits": record.execution_logits[
                            step, episode_index
                        ].tolist(),
                        "behavior_logits": record.behavior_logits[
                            step, episode_index
                        ].tolist(),
                        "mask_execution_logits": record.mask_execution_logits[
                            step, episode_index
                        ].tolist(),
                        "j_use": record.j_use[step, episode_index].tolist(),
                        "j_mask": record.j_mask[step, episode_index].tolist(),
                        "per_action_response_value": (
                            record.per_action_response_value[
                                step, episode_index
                            ].tolist()
                        ),
                        "per_action_net_value": record.per_action_net_value[
                            step, episode_index
                        ].tolist(),
                        "per_action_policy_mediated_gain": (
                            record.per_action_policy_mediated_gain[
                                step, episode_index
                            ].tolist()
                        ),
                        "per_action_predicted_gain_lower_score": (
                            record.per_action_predicted_gain_lower_score[
                                step, episode_index
                            ].tolist()
                        ),
                        "per_action_policy_gain_uncertainty": (
                            record.per_action_policy_gain_uncertainty[
                                step, episode_index
                            ].tolist()
                        ),
                        "per_action_expected_next_policy_tv": (
                            record.per_action_expected_next_policy_tv[
                                step, episode_index
                            ].tolist()
                        ),
                        "information_gain": record.information_gain[
                            step, episode_index
                        ].tolist(),
                        "response_code": int(
                            host_codes[side][step, episode_index]
                        ),
                        "belief_entropy": float(
                            record.belief_entropy[step, episode_index]
                        ),
                        "value_class_count": int(
                            record.value_class_count[step, episode_index]
                        ),
                        "kl_divergence": float(
                            record.kl_divergence[step, episode_index]
                        ),
                        "reference_greedy_action": int(
                            record.reference_greedy_action[
                                step, episode_index
                            ]
                        ),
                        "predicted_response_effect": float(
                            record.predicted_response_effect[
                                step, episode_index
                            ]
                        ),
                        "predicted_policy_cost": float(
                            record.predicted_policy_cost[
                                step, episode_index
                            ]
                        ),
                        "predicted_net_effect": float(
                            record.predicted_net_effect[
                                step, episode_index
                            ]
                        ),
                        "predicted_regularized_net_effect": float(
                            record.predicted_regularized_net_effect[
                                step, episode_index
                            ]
                        ),
                        "predicted_policy_total_variation": float(
                            record.predicted_policy_total_variation[
                                step, episode_index
                            ]
                        ),
                        "predicted_policy_mediated_effect": float(
                            record.predicted_policy_mediated_effect[
                                step, episode_index
                            ]
                        ),
                        "predicted_policy_gain_lower_score": float(
                            record.predicted_policy_gain_lower_score[
                                step, episode_index
                            ]
                        ),
                        "predicted_policy_gain_uncertainty": float(
                            record.predicted_policy_gain_uncertainty[
                                step, episode_index
                            ]
                        ),
                        "predicted_next_policy_total_variation": float(
                            record.predicted_next_policy_total_variation[
                                step, episode_index
                            ]
                        ),
                        "executed_action_response_value": float(
                            record.executed_action_response_value[
                                step, episode_index
                            ]
                        ),
                        "executed_action_net_value": float(
                            record.executed_action_net_value[
                                step, episode_index
                            ]
                        ),
                        "executed_action_policy_mediated_gain": float(
                            record.executed_action_policy_mediated_gain[
                                step, episode_index
                            ]
                        ),
                        "executed_action_predicted_gain_lower_score": float(
                            record.executed_action_predicted_gain_lower_score[
                                step, episode_index
                            ]
                        ),
                        "executed_action_policy_gain_uncertainty": float(
                            record.executed_action_policy_gain_uncertainty[
                                step, episode_index
                            ]
                        ),
                        "executed_action_expected_next_policy_tv": float(
                            record.executed_action_expected_next_policy_tv[
                                step, episode_index
                            ]
                        ),
                        "maximum_action_net_value": float(
                            record.maximum_action_net_value[
                                step, episode_index
                            ]
                        ),
                        "maximum_action_policy_mediated_gain": float(
                            record.maximum_action_policy_mediated_gain[
                                step, episode_index
                            ]
                        ),
                        "maximum_action_predicted_gain_lower_score": float(
                            record.maximum_action_predicted_gain_lower_score[
                                step, episode_index
                            ]
                        ),
                    }

    return rows, decisions()


def evaluate_standard(
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
    row_paths = []
    for pairing in standard_pairings(config.evaluation.deployment_modes):
        pairing_root = (
            output
            / pairing.deployment_mode
            / pairing.pairing_id
        )
        row_path = pairing_root / "episodes.parquet"
        decision_path = pairing_root / "decisions.jsonl"
        if resume and row_path.is_file() and decision_path.is_file():
            row_count = len(read_parquet(row_path))
            decision_count = _jsonl_row_count(decision_path)
            expected_decisions = (
                2
                * config.environment.episode_steps
                * config.evaluation.episodes_per_pairing
            )
            if (
                row_count != config.evaluation.episodes_per_pairing
                or decision_count != expected_decisions
            ):
                raise RuntimeError(
                    f"Incomplete evaluation output exists in {pairing_root}."
                )
            row_paths.append(row_path)
            continue
        if row_path.exists() or decision_path.exists():
            raise RuntimeError(
                f"Incomplete evaluation output exists in {pairing_root}."
            )
        rows, decisions = pairing_batch(
            config=config,
            left=deployments[pairing.left_outer_unit_id],
            right=deployments[pairing.right_outer_unit_id],
            pairing=pairing,
            population_name=population.name,
            evaluation_seed=evaluation_seed,
        )
        write_parquet(row_path, [row.to_mapping() for row in rows])
        write_jsonl(decision_path, decisions)
        row_paths.append(row_path)

    all_rows = [
        EpisodeRow(
            **{
                **row,
                "response_code_counts": tuple(row["response_code_counts"]),
            }
        )
        for path in row_paths
        for row in read_parquet(path)
    ]
    validate_standard_rows(
        all_rows,
        deployment_modes=config.evaluation.deployment_modes,
        episodes_per_pairing=config.evaluation.episodes_per_pairing,
    )
    write_json(
        output / "summary.json", summarize_standard_rows(all_rows)
    )
    return len(all_rows), sum(row.environment_steps for row in all_rows)


def evaluate_development_self_pairing(
    *,
    config: Any,
    population: Any,
    output: Path,
    evaluation_seed: int,
    resume: bool,
) -> tuple[int, int]:
    """Run matched self pairings for one development policy."""

    deployment = load_deployment(population.entries[0], config)
    row_paths = []
    for mode in config.evaluation.deployment_modes:
        root = output / mode
        row_path = root / "episodes.parquet"
        decision_path = root / "decisions.jsonl"
        if resume and row_path.is_file() and decision_path.is_file():
            expected_decisions = (
                2
                * config.environment.episode_steps
                * config.evaluation.episodes_per_pairing
            )
            if (
                len(read_parquet(row_path))
                != config.evaluation.episodes_per_pairing
                or _jsonl_row_count(decision_path) != expected_decisions
            ):
                raise RuntimeError(
                    f"Incomplete development output exists in {root}."
                )
            row_paths.append(row_path)
            continue
        if row_path.exists() or decision_path.exists():
            raise RuntimeError(
                f"Development output already exists in {root}."
            )
        rows, decisions = pairing_batch(
            config=config,
            left=deployment,
            right=deployment,
            pairing=Pairing(mode, "sp", 0, 0),
            population_name=population.name,
            evaluation_seed=evaluation_seed,
        )
        write_parquet(row_path, [row.to_mapping() for row in rows])
        write_jsonl(decision_path, decisions)
        row_paths.append(row_path)
    all_rows = [
        EpisodeRow(
            **{
                **row,
                "response_code_counts": tuple(row["response_code_counts"]),
            }
        )
        for path in row_paths
        for row in read_parquet(path)
    ]
    validate_development_rows(
        all_rows,
        deployment_modes=config.evaluation.deployment_modes,
        episodes_per_mode=config.evaluation.episodes_per_pairing,
    )
    write_json(output / "summary.json", summarize_development_rows(all_rows))
    return len(all_rows), sum(row.environment_steps for row in all_rows)



__all__ = [
    "evaluate_development_self_pairing",
    "evaluate_standard",
    "pairing_batch",
]
