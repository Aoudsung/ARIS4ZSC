"""Matched base/conditional evaluation on run-disjoint confirmatory partners."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from experiments.overcooked_v2.br_prox_app import empirical_local_br_prox_pairing
from experiments.overcooked_v2.deployment import (
    deployment_action,
    load_deployment,
    reset_deployment_state,
    update_after_transition,
)
from experiments.overcooked_v2.official_adapter import FrozenPartnerPool, VectorEnvironment
from src.path_c.calibration import predicted_policy_gain
from src.path_c.evaluation import (
    paired_adaptation_gain,
    summarize_evaluation_rows,
)
from src.path_c.experiment import load_config, load_partner_manifest
from src.path_c.statistics import (
    equal_partner_mechanism_mean,
    hierarchical_partner_bootstrap,
    one_sided_interval,
)
from src.path_c.storage import (
    ensure_run_identity,
    evaluation_identity,
    write_json,
    write_parquet,
)
from src.path_c.types import EvaluationRow


class _EvalWorld(NamedTuple):
    environment_state: Any
    observations: Any
    ego_state: Any
    partner_carry: Any
    partner_episode_start: Any
    raw_return: Any
    correct_deliveries: Any
    wrong_deliveries: Any
    adaptation_steps: Any
    support_sum: Any
    gain_sum: Any


def evaluation_episode_seed(
    *,
    root_seed: int,
    ego_run_id: str,
    partner_run_id: str,
    role: int,
    episode_index: int,
) -> int:
    payload = (
        f"{int(root_seed)}\0{ego_run_id}\0{partner_run_id}\0"
        f"role-{int(role)}\0episode-{int(episode_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _tree_reset(mask: Any, fresh: Any, current: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def one(left: Any, right: Any) -> Any:
        values = jnp.asarray(mask, dtype=jnp.bool_)
        expanded = values.reshape(values.shape + (1,) * (jnp.ndim(left) - values.ndim))
        return jnp.where(expanded, left, right)

    return jax.tree_util.tree_map(one, fresh, current)


def _paired_batch(
    *,
    config: Any,
    deployment: Any,
    partner_checkpoint: Path,
    partner_run_id: str,
    partner_mechanism: str,
    role: int,
    evaluation_seed: int,
) -> tuple[tuple[EvaluationRow, ...], tuple[EvaluationRow, ...], list[dict[str, Any]]]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    if role not in {0, 1}:
        raise ValueError("role must be zero or one.")
    count = int(config.evaluation.episodes_per_pairing)
    runtime = replace(config, environment=replace(config.environment, num_envs=count))
    environment = VectorEnvironment.create(runtime)
    pool = FrozenPartnerPool.from_checkpoints((partner_checkpoint,))
    seeds = np.asarray(
        [
            evaluation_episode_seed(
                root_seed=evaluation_seed,
                ego_run_id=deployment.ego_run_id,
                partner_run_id=partner_run_id,
                role=role,
                episode_index=index,
            )
            for index in range(count)
        ],
        dtype=np.uint32,
    )
    root_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(root_keys)
    environment_state, observations = environment.reset_with_keys(reset_keys)
    ego_initial = reset_deployment_state(
        deployment,
        batch_size=count,
        observation_shape=environment.observation_shape,
    )
    partner_initial = pool.initial_carry(count)
    starts = jnp.ones((count,), dtype=jnp.bool_)
    zeros = jnp.zeros((count,), dtype=jnp.float32)
    zeros_int = jnp.zeros((count,), dtype=jnp.int32)
    initial_world = _EvalWorld(
        environment_state=environment_state,
        observations=observations,
        ego_state=ego_initial,
        partner_carry=partner_initial,
        partner_episode_start=starts,
        raw_return=zeros,
        correct_deliveries=zeros_int,
        wrong_deliveries=zeros_int,
        adaptation_steps=zeros_int,
        support_sum=zeros,
        gain_sum=zeros,
    )

    def advance(
        world: _EvalWorld,
        *,
        force_base: bool,
        step: Any,
    ) -> tuple[_EvalWorld, dict[str, Any]]:
        ego_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 1 + 4 * step)
        )(root_keys)
        partner_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 2 + 4 * step)
        )(root_keys)
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 3 + 4 * step)
        )(root_keys)
        ego_observation = world.observations[:, role]
        partner_observation = world.observations[:, 1 - role]
        stepped, ego_action, output, unused_log_probability = deployment_action(
            deployment=deployment,
            state=world.ego_state,
            observation=ego_observation,
            keys=ego_keys,
            force_base=force_base,
        )
        del unused_log_probability
        member = jnp.zeros((count,), dtype=jnp.int32)
        partner_action, next_partner_carry = pool.step_with_keys(
            member,
            partner_observation,
            world.partner_carry,
            world.partner_episode_start,
            partner_keys,
        )
        joint = (
            jnp.stack((ego_action, partner_action), axis=-1)
            if role == 0
            else jnp.stack((partner_action, ego_action), axis=-1)
        )
        next_environment, next_observations, rewards, dones, info = (
            environment.step_with_keys(
                world.environment_state, joint, environment_keys
            )
        )
        terminal = info["terminal_observations"][:, role]
        observation_mask = dones.reshape(
            dones.shape
            + (1,) * (next_observations[:, role].ndim - dones.ndim)
        )
        ego_response_next = jnp.where(
            observation_mask, terminal, next_observations[:, role]
        )
        next_ego = update_after_transition(
            deployment=deployment,
            stepped_state=stepped,
            action=ego_action,
            reward=rewards,
            done=dones,
            next_observation=ego_response_next,
        )
        fresh_partner = pool.initial_carry(count)
        next_partner_carry = _tree_reset(dones, fresh_partner, next_partner_carry)
        gain = predicted_policy_gain(
            output.action_values,
            output.base_logits,
            output.base_logits + output.residual_logits,
        )
        candidate = _EvalWorld(
            environment_state=next_environment,
            observations=next_observations,
            ego_state=next_ego,
            partner_carry=next_partner_carry,
            partner_episode_start=dones,
            raw_return=world.raw_return + rewards,
            correct_deliveries=(
                world.correct_deliveries
                + jnp.asarray(info["correct_delivery"], dtype=jnp.int32)
            ),
            wrong_deliveries=(
                world.wrong_deliveries
                + jnp.asarray(info["wrong_delivery"], dtype=jnp.int32)
            ),
            adaptation_steps=world.adaptation_steps
            + (output.gate > 0.5).astype(jnp.int32),
            support_sum=world.support_sum + output.support_score,
            gain_sum=world.gain_sum + gain,
        )
        return candidate, {
            "ego_action": ego_action,
            "partner_action": partner_action,
            "reward": rewards,
            "gate": output.gate,
            "support": output.support_score,
            "predicted_gain": gain,
        }

    def one_step(
        carry: tuple[_EvalWorld, _EvalWorld], step: Any
    ) -> tuple[tuple[_EvalWorld, _EvalWorld], tuple[Any, Any]]:
        adapted, base = carry
        next_adapted, adapted_record = advance(
            adapted,
            force_base=not bool(config.evaluation.enable_adaptation),
            step=step,
        )
        next_base, base_record = advance(base, force_base=True, step=step)
        return (next_adapted, next_base), (adapted_record, base_record)

    (adapted_final, base_final), records = jax.lax.scan(
        one_step,
        (initial_world, initial_world),
        jnp.arange(config.environment.episode_steps),
    )
    adapted_records, base_records = records
    del base_records
    adapted_rows = []
    base_rows = []
    paired_rows: list[dict[str, Any]] = []
    for index in range(count):
        adapted_return = float(np.asarray(adapted_final.raw_return[index]))
        base_return = float(np.asarray(base_final.raw_return[index]))
        common = dict(
            ego_run_id=deployment.ego_run_id,
            partner_run_id=partner_run_id,
            partner_mechanism=partner_mechanism,
            episode_index=index,
            episode_seed=int(seeds[index]),
        )
        adapted_rows.append(
            EvaluationRow(
                **common,
                raw_return=adapted_return,
                correct_deliveries=int(
                    np.asarray(adapted_final.correct_deliveries[index])
                ),
                wrong_deliveries=int(
                    np.asarray(adapted_final.wrong_deliveries[index])
                ),
                adaptation_enabled_steps=int(
                    np.asarray(adapted_final.adaptation_steps[index])
                ),
                base_policy_steps=(
                    config.environment.episode_steps
                    - int(np.asarray(adapted_final.adaptation_steps[index]))
                ),
                mean_support_score=float(
                    np.asarray(adapted_final.support_sum[index])
                    / config.environment.episode_steps
                ),
                mean_predicted_gain=float(
                    np.asarray(adapted_final.gain_sum[index])
                    / config.environment.episode_steps
                ),
                negative_transfer=adapted_return < base_return,
            )
        )
        base_rows.append(
            EvaluationRow(
                **common,
                raw_return=base_return,
                correct_deliveries=int(
                    np.asarray(base_final.correct_deliveries[index])
                ),
                wrong_deliveries=int(
                    np.asarray(base_final.wrong_deliveries[index])
                ),
                adaptation_enabled_steps=0,
                base_policy_steps=config.environment.episode_steps,
                mean_support_score=float(
                    np.asarray(base_final.support_sum[index])
                    / config.environment.episode_steps
                ),
                mean_predicted_gain=float(
                    np.asarray(base_final.gain_sum[index])
                    / config.environment.episode_steps
                ),
                negative_transfer=False,
            )
        )
        paired_rows.append(
            {
                **common,
                "ego_role": role,
                "adapted_raw_return": adapted_return,
                "base_raw_return": base_return,
                "adaptation_gain": adapted_return - base_return,
                "negative_transfer": adapted_return < base_return,
                "adaptation_enabled_steps": int(
                    np.asarray(adapted_final.adaptation_steps[index])
                ),
                "mean_support_score": float(
                    np.asarray(adapted_final.support_sum[index])
                    / config.environment.episode_steps
                ),
                "mean_predicted_gain": float(
                    np.asarray(adapted_final.gain_sum[index])
                    / config.environment.episode_steps
                ),
                "first_adapted_action": int(
                    np.asarray(adapted_records["ego_action"][0, index])
                ),
            }
        )
    return tuple(adapted_rows), tuple(base_rows), paired_rows


def _row_mapping(row: EvaluationRow, condition: str) -> dict[str, Any]:
    return {"condition": condition, **row._asdict()}


def run_evaluation(args: argparse.Namespace) -> None:
    import numpy as np

    config = load_config(args.config, run_kind=args.run_kind)
    manifest = load_partner_manifest(
        args.partner_manifest,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    confirmatory = manifest.by_role("confirmatory")
    if not confirmatory:
        raise ValueError("Partner manifest has no confirmatory partners.")
    deployment_bundles = tuple(
        Path(value).resolve() for value in args.deployments
    )
    if not deployment_bundles:
        raise ValueError("At least one pruned deployment bundle is required.")
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        evaluation_identity(
            config=config,
            seed=int(args.seed),
            deployment_bundles=deployment_bundles,
            manifest=manifest,
        ),
    )

    adapted_all: list[EvaluationRow] = []
    base_all: list[EvaluationRow] = []
    paired_all: list[dict[str, Any]] = []
    br_prox_all: list[Mapping[str, Any]] = []
    roles = (0, 1) if config.evaluation.evaluate_both_roles else (0,)
    for deployment_bundle in deployment_bundles:
        deployment = load_deployment(deployment_bundle, config)
        for partner_index, partner in enumerate(confirmatory):
            for role in roles:
                adapted, base, paired = _paired_batch(
                    config=config,
                    deployment=deployment,
                    partner_checkpoint=partner.checkpoint,
                    partner_run_id=partner.run_id,
                    partner_mechanism=partner.generation_mechanism,
                    role=role,
                    evaluation_seed=int(args.seed),
                )
                adapted_all.extend(adapted)
                base_all.extend(base)
                paired_all.extend(paired)
                if config.evaluation.report_br_prox:
                    br_prox_all.extend(
                        empirical_local_br_prox_pairing(
                            config=config,
                            deployment=deployment,
                            partner_checkpoint=partner.checkpoint,
                            partner_run_id=partner.run_id,
                            partner_mechanism=partner.generation_mechanism,
                            partner_numeric_id=partner_index,
                            role=role,
                            evaluation_seed=int(args.seed),
                        )
                    )

    write_parquet(
        output / "episodes.parquet",
        [
            *(_row_mapping(row, "adapted") for row in adapted_all),
            *(_row_mapping(row, "base") for row in base_all),
        ],
    )
    write_parquet(output / "paired_episodes.parquet", paired_all)
    if br_prox_all:
        write_parquet(output / "local_br_prox_anchors.parquet", br_prox_all)
    adapted_summary = summarize_evaluation_rows(
        adapted_all,
        bootstrap_replicates=config.evaluation.bootstrap_replicates,
        seed=config.evaluation.evaluation_seed,
        alpha=config.evaluation.one_sided_alpha,
    )
    base_summary = summarize_evaluation_rows(
        base_all,
        bootstrap_replicates=config.evaluation.bootstrap_replicates,
        seed=config.evaluation.evaluation_seed + 1,
        alpha=config.evaluation.one_sided_alpha,
    )
    gains, egos, partners, mechanisms = paired_adaptation_gain(
        adapted_all, base_all
    )
    gain_bootstrap = hierarchical_partner_bootstrap(
        values=gains,
        ego_runs=egos,
        partner_runs=partners,
        mechanisms=mechanisms,
        replicates=config.evaluation.bootstrap_replicates,
        seed=config.evaluation.evaluation_seed + 2,
    )
    gain_interval = one_sided_interval(
        gain_bootstrap, config.evaluation.one_sided_alpha
    )
    if br_prox_all:
        br_values = [float(row["br_prox"]) for row in br_prox_all]
        br_regrets = [float(row["raw_local_br_regret"]) for row in br_prox_all]
        br_egos = [str(row["ego_run_id"]) for row in br_prox_all]
        br_partners = [str(row["partner_run_id"]) for row in br_prox_all]
        br_mechanisms = [str(row["partner_mechanism"]) for row in br_prox_all]
        br_bootstrap = hierarchical_partner_bootstrap(
            values=br_values,
            ego_runs=br_egos,
            partner_runs=br_partners,
            mechanisms=br_mechanisms,
            replicates=config.evaluation.bootstrap_replicates,
            seed=config.evaluation.evaluation_seed + 3,
        )
        regret_bootstrap = hierarchical_partner_bootstrap(
            values=br_regrets,
            ego_runs=br_egos,
            partner_runs=br_partners,
            mechanisms=br_mechanisms,
            replicates=config.evaluation.bootstrap_replicates,
            seed=config.evaluation.evaluation_seed + 4,
        )
        local_br_prox_summary: Mapping[str, Any] | None = {
            "scope": "one_action_deviation_with_frozen_deployed_continuation",
            "ground_truth": "independent_split_simulator_continuation_return",
            "lower_is_better": True,
            "anchor_rows": len(br_prox_all),
            "partner_mechanism_weighted_point_estimate": (
                equal_partner_mechanism_mean(
                    br_values, br_partners, br_mechanisms
                )
            ),
            "bootstrap": one_sided_interval(
                br_bootstrap, config.evaluation.one_sided_alpha
            ),
            "raw_local_regret_point_estimate": equal_partner_mechanism_mean(
                br_regrets, br_partners, br_mechanisms
            ),
            "raw_local_regret_bootstrap": one_sided_interval(
                regret_bootstrap, config.evaluation.one_sided_alpha
            ),
            "oracle_action_agreement": float(
                np.mean([bool(row["oracle_action_agreement"]) for row in br_prox_all])
            ),
            "fit_replicas": config.evaluation.br_prox_fit_replicas,
            "evaluation_replicas": config.evaluation.br_prox_evaluation_replicas,
            "continuation_horizon": config.evaluation.br_prox_continuation_horizon,
            "note": (
                "This is an empirical local one-action-deviation approximation, "
                "not the unrestricted full-policy best response."
            ),
        }
    else:
        local_br_prox_summary = None
    summary = {
        "adapted": adapted_summary,
        "base": base_summary,
        "adaptation_gain": {
            "partner_mechanism_weighted_point_estimate": equal_partner_mechanism_mean(
                gains, partners, mechanisms
            ),
            "bootstrap": gain_interval,
            "negative_transfer_rate": float(
                np.mean([row.negative_transfer for row in adapted_all])
            ),
        },
        "ego_run_count": len({row.ego_run_id for row in adapted_all}),
        "confirmatory_partner_run_count": len(
            {row.partner_run_id for row in adapted_all}
        ),
        "confirmatory_mechanism_count": len(
            {row.partner_mechanism for row in adapted_all}
        ),
        "episode_rows_per_condition": len(adapted_all),
        "local_br_prox": local_br_prox_summary,
        "scientific_readout_allowed": bool(
            args.allow_scientific_readout
            and len({row.ego_run_id for row in adapted_all})
            >= config.evaluation.minimum_ego_runs
            and all(
                len(
                    {
                        row.partner_run_id
                        for row in adapted_all
                        if row.partner_mechanism == mechanism
                    }
                )
                >= config.evaluation.minimum_partner_runs_per_mechanism
                for mechanism in {
                    row.partner_mechanism for row in adapted_all
                }
            )
            and len({row.partner_mechanism for row in adapted_all})
            >= config.evaluation.minimum_mechanisms
            and (
                not config.evaluation.report_br_prox
                or bool(br_prox_all)
            )
        ),
    }
    write_json(output / "summary.json", summary)
    print(f"Complete DELTA-ZSC confirmatory evaluation: {output}")


__all__ = ["evaluation_episode_seed", "run_evaluation"]
