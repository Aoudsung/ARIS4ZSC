"""Development-only evaluation of one VQBC ego against fixed official partners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from experiments.overcooked_v2.deployment import (
    Deployment,
    load_deployment,
    reset_deployment_state,
    update_deployment_after_transition,
)
from experiments.overcooked_v2.official_adapter import FrozenPartnerPool, VectorEnvironment
from src.path_c.experiment import (
    METHOD_VERSION,
    PopulationEntry,
    load_config,
    load_training_unit_manifest,
)
from src.path_c.method import DEPLOYMENT_MODES, policy_effect_trigger_tolerance
from src.path_c.runner import policy_action
from src.path_c.storage import (
    ensure_run_identity,
    read_parquet,
    write_json,
    write_jsonl,
    write_parquet,
    write_run_metadata,
)


@dataclass(frozen=True, slots=True)
class PartnerPanelRow:
    ego_outer_unit_id: int
    partner_index: int
    partner_label: str
    partner_checkpoint: str
    deployment_mode: str
    episode_index: int
    episode_seed: int
    environment_steps: int
    raw_return: float
    correct_delivery_count: int
    wrong_delivery_count: int
    indicator_activation_count: int
    positive_gain_lower_score_count: int
    cumulative_kl: float
    reference_action_deviation_count: int
    mean_value_class_count: float
    mean_belief_entropy: float
    mean_predicted_next_policy_tv: float
    response_code_counts: tuple[int, ...]

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["response_code_counts"] = list(self.response_code_counts)
        return payload


def select_panel_checkpoints(
    checkpoint_paths: Sequence[str | Path], *, final_only: bool
) -> tuple[Path, ...]:
    """Choose the final checkpoint from each registered three-checkpoint run."""

    paths = tuple(Path(value).resolve() for value in checkpoint_paths)
    if not paths:
        raise ValueError("The partner panel requires at least one checkpoint.")
    if not final_only:
        return paths
    if len(paths) % 3:
        raise ValueError(
            "Final-only panel selection requires start/midpoint/final triples."
        )
    return paths[2::3]


def panel_episode_seed(
    *,
    evaluation_seed: int,
    layout: str,
    ego_outer_unit_id: int,
    partner_index: int,
    episode_index: int,
) -> int:
    """Derive common random numbers without including deployment mode."""

    payload = (
        f"{int(evaluation_seed)}\0{layout}\0{int(ego_outer_unit_id)}\0"
        f"partner-{int(partner_index)}\0{int(episode_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def panel_batch(
    *,
    config: Any,
    deployment: Deployment,
    partner_pool: FrozenPartnerPool,
    partner_index: int,
    partner_label: str,
    partner_checkpoint: Path,
    deployment_mode: str,
    evaluation_seed: int,
) -> tuple[tuple[PartnerPanelRow, ...], Iterable[Mapping[str, Any]]]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    episode_count = int(config.evaluation.episodes_per_pairing)
    template = VectorEnvironment.create(config)
    environment = VectorEnvironment(
        environment=template.environment,
        num_envs=episode_count,
        episode_steps=config.environment.episode_steps,
    )
    seeds = np.asarray(
        [
            panel_episode_seed(
                evaluation_seed=evaluation_seed,
                layout=config.environment.layout,
                ego_outer_unit_id=deployment.outer_unit_id,
                partner_index=partner_index,
                episode_index=index,
            )
            for index in range(episode_count)
        ],
        dtype=np.uint32,
    )
    root_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(root_keys)
    environment_state, observations = environment.reset_with_keys(reset_keys)
    ego_state = reset_deployment_state(
        deployment, batch_size=episode_count, config=config
    )
    partner_carry = partner_pool.initial_carry(episode_count)
    partner_episode_start = jnp.ones((episode_count,), dtype=jnp.bool_)
    member_indexes = jnp.full(
        (episode_count,), int(partner_index), dtype=jnp.int32
    )
    functions = deployment.functions()

    def one_step(carry: tuple[Any, ...], step: Any) -> tuple[Any, tuple[Any, ...]]:
        (
            current_environment,
            current_observations,
            current_ego,
            current_partner_carry,
            current_partner_start,
        ) = carry
        ego_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 1 + 3 * step)
        )(root_keys)
        partner_key = jax.random.fold_in(
            jax.random.PRNGKey(int(evaluation_seed) ^ (partner_index + 1)), step
        )
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 3 + 3 * step)
        )(root_keys)
        stepped_ego, ego_action, ego_output, ego_record, unused_generic = (
            policy_action(
                functions=functions,
                params={
                    "official": deployment.online_params,
                    "heads": deployment.head_params,
                },
                policy_state=current_ego,
                observations=current_observations[:, 0],
                key=ego_keys,
                deployment_mode=deployment_mode,
                gamma=config.training.gamma,
                uncertainty_penalty=config.model.uncertainty_penalty,
                behavior_exploration_mix=0.0,
                behavior_uniform_floor=0.0,
            )
        )
        del unused_generic
        partner_action, next_partner_carry = partner_pool.step(
            member_indexes,
            current_observations[:, 1],
            current_partner_carry,
            current_partner_start,
            partner_key,
        )
        actions = jnp.stack((ego_action, partner_action), axis=-1)
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
            dones.shape + (1,) * (next_observations[:, 0].ndim - 1)
        )
        response_next = jnp.where(
            observation_mask, terminal[:, 0], next_observations[:, 0]
        )
        next_ego, response_code = update_deployment_after_transition(
            deployment=deployment,
            functions=functions,
            state=stepped_ego,
            output=ego_output,
            observations=current_observations[:, 0],
            actions=ego_action,
            next_observations=response_next,
            rewards=rewards,
            dones=dones,
            deployment_mode=deployment_mode,
            terminal_response=config.model.response_count - 1,
        )
        return (
            next_environment,
            next_observations,
            next_ego,
            next_partner_carry,
            dones,
        ), (
            rewards,
            info["correct_delivery"],
            info["wrong_delivery"],
            info["indicator_activation"],
            ego_record,
            response_code,
            partner_action,
            current_ego.slot_log_belief,
        )

    unused_final, recorded = jax.lax.scan(
        one_step,
        (
            environment_state,
            observations,
            ego_state,
            partner_carry,
            partner_episode_start,
        ),
        jnp.arange(config.environment.episode_steps),
    )
    del unused_final
    (
        rewards,
        correct,
        wrong,
        indicator,
        records,
        response_codes,
        partner_actions,
        slot_log_beliefs,
    ) = recorded

    trigger = (
        records.executed_action_predicted_gain_lower_score
        > policy_effect_trigger_tolerance(records.j_use, records.j_mask)
    ) & (
        records.executed_action_expected_next_policy_tv
        >= config.evaluation.response_policy_tv_minimum
    )
    response_count = int(config.model.response_count)
    counts = jax.vmap(
        lambda values: jnp.bincount(values, length=response_count), in_axes=1
    )(response_codes)
    raw_returns = np.asarray(jnp.sum(rewards, axis=0))
    correct_totals = np.asarray(jnp.sum(correct, axis=0))
    wrong_totals = np.asarray(jnp.sum(wrong, axis=0))
    indicator_totals = np.asarray(jnp.sum(indicator, axis=0))
    trigger_counts = np.asarray(jnp.sum(trigger.astype(jnp.int32), axis=0))
    cumulative_kl = np.asarray(jnp.sum(records.kl_divergence, axis=0))
    deviations = np.asarray(
        jnp.sum(
            (records.action != records.reference_greedy_action).astype(jnp.int32),
            axis=0,
        )
    )
    class_means = np.asarray(jnp.mean(records.value_class_count, axis=0))
    entropy_means = np.asarray(jnp.mean(records.belief_entropy, axis=0))
    next_tv_means = np.asarray(
        jnp.mean(records.predicted_next_policy_total_variation, axis=0)
    )
    counts_host = np.asarray(counts)
    rows = tuple(
        PartnerPanelRow(
            ego_outer_unit_id=deployment.outer_unit_id,
            partner_index=partner_index,
            partner_label=partner_label,
            partner_checkpoint=str(partner_checkpoint),
            deployment_mode=deployment_mode,
            episode_index=index,
            episode_seed=int(seeds[index]),
            environment_steps=config.environment.episode_steps,
            raw_return=float(raw_returns[index]),
            correct_delivery_count=int(correct_totals[index]),
            wrong_delivery_count=int(wrong_totals[index]),
            indicator_activation_count=int(indicator_totals[index]),
            positive_gain_lower_score_count=int(trigger_counts[index]),
            cumulative_kl=float(cumulative_kl[index]),
            reference_action_deviation_count=int(deviations[index]),
            mean_value_class_count=float(class_means[index]),
            mean_belief_entropy=float(entropy_means[index]),
            mean_predicted_next_policy_tv=float(next_tv_means[index]),
            response_code_counts=tuple(
                int(value) for value in counts_host[index]
            ),
        )
        for index in range(episode_count)
    )

    host_records = jax.tree_util.tree_map(lambda value: np.asarray(value), records)
    host_codes = np.asarray(response_codes)
    host_partner_actions = np.asarray(partner_actions)
    host_slot_log_beliefs = np.asarray(slot_log_beliefs)

    def decisions() -> Iterable[Mapping[str, Any]]:
        for step in range(config.environment.episode_steps):
            for episode_index in range(episode_count):
                yield {
                    "partner_index": int(partner_index),
                    "partner_label": partner_label,
                    "deployment_mode": deployment_mode,
                    "episode_index": episode_index,
                    "episode_seed": int(seeds[episode_index]),
                    "step": step,
                    "ego_action": int(host_records.action[step, episode_index]),
                    "partner_action": int(host_partner_actions[step, episode_index]),
                    "response_code": int(host_codes[step, episode_index]),
                    "reference_logits": host_records.reference_logits[
                        step, episode_index
                    ].tolist(),
                    "execution_logits": host_records.execution_logits[
                        step, episode_index
                    ].tolist(),
                    "slot_log_belief": host_slot_log_beliefs[
                        step, episode_index
                    ].tolist(),
                    "belief_entropy": float(
                        host_records.belief_entropy[step, episode_index]
                    ),
                    "value_class_count": int(
                        host_records.value_class_count[step, episode_index]
                    ),
                    "predicted_policy_gain_lower_score": float(
                        host_records.predicted_policy_gain_lower_score[
                            step, episode_index
                        ]
                    ),
                    "predicted_policy_gain_uncertainty": float(
                        host_records.predicted_policy_gain_uncertainty[
                            step, episode_index
                        ]
                    ),
                    "predicted_next_policy_total_variation": float(
                        host_records.predicted_next_policy_total_variation[
                            step, episode_index
                        ]
                    ),
                    "executed_action_predicted_gain_lower_score": float(
                        host_records.executed_action_predicted_gain_lower_score[
                            step, episode_index
                        ]
                    ),
                    "executed_action_policy_gain_uncertainty": float(
                        host_records.executed_action_policy_gain_uncertainty[
                            step, episode_index
                        ]
                    ),
                    "executed_action_expected_next_policy_tv": float(
                        host_records.executed_action_expected_next_policy_tv[
                            step, episode_index
                        ]
                    ),
                }

    return rows, decisions()


def _summary(rows: Sequence[PartnerPanelRow]) -> Mapping[str, Any]:
    grouped: dict[tuple[str, str], list[PartnerPanelRow]] = {}
    for row in rows:
        grouped.setdefault((row.partner_label, row.deployment_mode), []).append(row)
    result: dict[str, Any] = {}
    for (partner, mode), members in sorted(grouped.items()):
        result.setdefault(partner, {})[mode] = {
            "episode_count": len(members),
            "mean_raw_return": mean(row.raw_return for row in members),
            "mean_correct_deliveries": mean(
                row.correct_delivery_count for row in members
            ),
            "mean_wrong_deliveries": mean(
                row.wrong_delivery_count for row in members
            ),
            "mean_positive_gain_lower_score_count": mean(
                row.positive_gain_lower_score_count for row in members
            ),
            "mean_kl": mean(row.cumulative_kl for row in members),
        }
    for partner, modes in result.items():
        if "posterior_use" in modes and "prior_only" in modes:
            modes["posterior_minus_prior"] = (
                modes["posterior_use"]["mean_raw_return"]
                - modes["prior_only"]["mean_raw_return"]
            )
    return {"partners": result}


def run_partner_panel(args: Any) -> None:
    if args.run_kind not in {"mechanical", "development"}:
        raise ValueError(
            "The fixed-partner panel is available only for wiring checks and development."
        )
    config = load_config(args.config, run_kind=args.run_kind)
    episode_override = getattr(args, "episodes_per_pairing", None)
    unit_manifest = load_training_unit_manifest(
        args.unit_manifest,
        expected_layout=config.environment.layout,
        run_kind=args.run_kind,
    )
    unit = unit_manifest.unit(int(args.outer_unit))
    selected = select_panel_checkpoints(
        unit.partner_checkpoints, final_only=bool(args.final_only)
    )
    entry = PopulationEntry(
        outer_unit_id=int(args.outer_unit),
        run_directory=Path(args.run_directory).resolve(),
    )
    deployment = load_deployment(entry, config)
    if episode_override is not None:
        if args.run_kind != "mechanical":
            raise ValueError(
                "Only a mechanical wiring check may override panel episode count."
            )
        if int(episode_override) <= 0:
            raise ValueError("Panel episode count must be positive.")
        config = replace(
            config,
            evaluation=replace(
                config.evaluation,
                episodes_per_pairing=int(episode_override),
            ),
        )
    pool = FrozenPartnerPool.from_checkpoints(selected)
    if pool.network.layout != config.environment.layout:
        raise ValueError("Panel partners and evaluation layout differ.")

    output = Path(args.output).resolve()
    panel_manifest = {
        "ego_outer_unit_id": int(args.outer_unit),
        "run_directory": str(Path(args.run_directory).resolve()),
        "final_only": bool(args.final_only),
        "partner_checkpoints": [str(path) for path in selected],
        "deployment_modes": list(config.evaluation.deployment_modes),
    }
    ensure_run_identity(
        output,
        {
            "stage": "partner_panel",
            "method": METHOD_VERSION,
            "run_kind": config.run_kind,
            "layout": config.environment.layout,
            "config": config.to_mapping(),
            "seed": int(args.seed),
            "panel": panel_manifest,
        },
    )
    write_json(output / "resolved_config.json", config.to_mapping())
    write_json(output / "panel_manifest.json", panel_manifest)

    row_paths: list[Path] = []
    for partner_index, checkpoint in enumerate(selected):
        partner_label = f"partner_{partner_index:02d}"
        for mode in config.evaluation.deployment_modes:
            root = output / partner_label / mode
            row_path = root / "episodes.parquet"
            decision_path = root / "decisions.jsonl"
            if args.resume and row_path.is_file() and decision_path.is_file():
                if len(read_parquet(row_path)) != config.evaluation.episodes_per_pairing:
                    raise RuntimeError(f"Incomplete panel rows in {root}.")
                expected_decisions = (
                    config.environment.episode_steps
                    * config.evaluation.episodes_per_pairing
                )
                with decision_path.open("r", encoding="utf-8") as handle:
                    count = sum(1 for _ in handle)
                if count != expected_decisions:
                    raise RuntimeError(f"Incomplete panel decisions in {root}.")
                row_paths.append(row_path)
                continue
            if row_path.exists() or decision_path.exists():
                raise RuntimeError(f"Incomplete panel output exists in {root}.")
            rows, decisions = panel_batch(
                config=config,
                deployment=deployment,
                partner_pool=pool,
                partner_index=partner_index,
                partner_label=partner_label,
                partner_checkpoint=checkpoint,
                deployment_mode=mode,
                evaluation_seed=int(args.seed),
            )
            write_parquet(row_path, [row.to_mapping() for row in rows])
            write_jsonl(decision_path, decisions)
            row_paths.append(row_path)

    all_rows = [
        PartnerPanelRow(
            **{
                **row,
                "response_code_counts": tuple(row["response_code_counts"]),
            }
        )
        for path in row_paths
        for row in read_parquet(path)
    ]
    expected = (
        len(selected)
        * len(config.evaluation.deployment_modes)
        * config.evaluation.episodes_per_pairing
    )
    if len(all_rows) != expected:
        raise RuntimeError(
            f"Expected {expected} panel episode rows, received {len(all_rows)}."
        )
    matched: dict[tuple[int, int], set[int]] = {}
    for row in all_rows:
        matched.setdefault((row.partner_index, row.episode_index), set()).add(
            row.episode_seed
        )
    if any(len(values) != 1 for values in matched.values()):
        raise RuntimeError("Panel deployment modes do not share episode seeds.")
    write_json(output / "summary.json", _summary(all_rows))
    effective_steps = sum(row.environment_steps for row in all_rows)
    write_run_metadata(
        output / "run_metadata.json",
        config=config,
        seed=int(args.seed),
        effective_environment_steps=effective_steps,
        update_count=0,
        completed_episodes=len(all_rows),
    )
    print(f"Complete fixed-partner panel: {output}")


__all__ = [
    "PartnerPanelRow",
    "panel_batch",
    "panel_episode_seed",
    "run_partner_panel",
    "select_panel_checkpoints",
]
