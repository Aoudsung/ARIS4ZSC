"""Causal decision-value evaluation for unified DELTA-ZSC.

The intervention replaces only the categorical belief at a frozen source
state.  Task features, instantaneous partner geometry, analytic behaviour
statistics, base policy, decision-emission geometry, source partner, and CRN
continuation returns remain fixed.  No recurrent hidden states are spliced.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.overcooked_v2.official_adapter import FrozenPartnerPool, VectorEnvironment
from src.delta_zsc.anchors import collect_decision_anchors
from src.delta_zsc.deployment import load_deployment
from src.delta_zsc.mirror_policy import expected_action_values, kl_constrained_policy
from src.delta_zsc.model import JOINT_VARIANT
from src.delta_zsc.runner import collect_rollout, initialize_runner
from src.path_c.experiment import load_partner_manifest
from src.path_c.partner_sources import make_static_pool_partner_functions
from src.path_c.resources import ResourceLedger, gpu_hours_for_wall_seconds
from src.path_c.storage import runtime_provenance, sha256_path, write_parquet


def _environment(config: Any, num_envs: int) -> VectorEnvironment:
    return VectorEnvironment.create(
        SimpleNamespace(
            environment=SimpleNamespace(
                layout=config.environment.layout,
                episode_steps=config.environment.episode_steps,
                agent_view_size=config.environment.agent_view_size,
                indicate_successful_delivery=(
                    config.environment.indicate_successful_delivery
                ),
                num_envs=int(num_envs),
            )
        )
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _nearest_different_run_donors(
    task_features: np.ndarray, run_ids: np.ndarray
) -> np.ndarray:
    count = task_features.shape[0]
    distance = np.linalg.norm(
        task_features[:, None, :] - task_features[None, :, :], axis=-1
    )
    distance = np.where(run_ids[:, None] != run_ids[None, :], distance, np.inf)
    donor = np.argmin(distance, axis=1)
    if np.any(~np.isfinite(distance[np.arange(count), donor])):
        raise RuntimeError("Causal panel cannot match a donor from another partner run.")
    return donor.astype(np.int32)


def _run_bootstrap(
    values: np.ndarray, run_ids: np.ndarray, *, replicates: int, seed: int
) -> dict[str, Any]:
    unique = np.unique(run_ids)
    if unique.size < 2:
        raise ValueError("Causal inference requires at least two partner-run blocks.")
    run_means = np.asarray(
        [np.mean(values[run_ids == run]) for run in unique], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, unique.size, size=(int(replicates), unique.size))
    draws = np.mean(run_means[indexes], axis=1)
    return {
        "mean": float(np.mean(run_means)),
        "one_sided_lcb_95": float(np.quantile(draws, 0.05)),
        "interval_95": [
            float(value) for value in np.quantile(draws, (0.025, 0.975))
        ],
        "partner_run_count": int(unique.size),
    }


def run_causal_evaluation(args: Any) -> None:
    import jax
    import jax.numpy as jnp

    started = time.perf_counter()
    deployment = load_deployment(args.deployment)
    config = deployment.config
    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    runs = tuple(manifest.by_role(str(getattr(args, "partner_role", "confirmatory"))))
    if len(runs) < 2:
        raise ValueError("Causal evaluation needs at least two independent partners.")
    lanes = int(getattr(args, "num_envs", 64))
    if lanes <= 0 or lanes % 2:
        raise ValueError("Causal evaluation lanes must be a positive even number.")
    environment = _environment(config, lanes)
    pool = FrozenPartnerPool.from_checkpoints(
        [run.checkpoint for run in runs],
        parent_training_run_ids=[run.parent_training_run_id for run in runs],
    )
    probabilities = jnp.full((len(runs),), 1.0 / float(len(runs)))
    family_ids = jnp.arange(len(runs), dtype=jnp.int32)
    stages = jnp.ones((len(runs),), dtype=jnp.float32)
    partner_functions = make_static_pool_partner_functions(
        external_pool=pool,
        member_probabilities=probabilities,
        member_family_ids=family_ids,
        member_checkpoint_stages=stages,
    )
    root = jax.random.PRNGKey(int(getattr(args, "seed", 0)))
    runner = initialize_runner(
        environment=environment,
        task_hidden_dim=config.architecture.task_hidden_dim,
        component_count=config.method.latent_components,
        partner_functions=partner_functions,
        random_key=jax.random.fold_in(root, 1),
    )
    _, _, records = collect_rollout(
        state=runner,
        length=config.environment.episode_steps,
        environment=environment,
        agent=deployment.agent,
        base_params=deployment.base_params,
        latent_params=deployment.latent_params,
        partner_functions=partner_functions,
        partner_parameters=None,
        variant=deployment.variant,
        task_hidden_dim=config.architecture.task_hidden_dim,
        component_count=config.method.latent_components,
        shaping_factor=0.0,
        record_anchor_state=True,
    )
    anchor_count = int(getattr(args, "anchor_count", 64))
    anchors, budget = collect_decision_anchors(
        key=jax.random.fold_in(root, 2),
        records=records,
        environment=environment,
        agent=deployment.agent,
        base_params=deployment.base_params,
        latent_params=deployment.latent_params,
        variant=deployment.variant,
        partner_functions=partner_functions,
        partner_parameters=None,
        task_hidden_dim=config.architecture.task_hidden_dim,
        component_count=config.method.latent_components,
        anchor_count=anchor_count,
        fit_replicas=int(getattr(args, "fit_replicas", 4)),
        evaluation_replicas=int(getattr(args, "evaluation_replicas", 8)),
        continuation_horizon=config.method.continuation_horizon,
        gamma=config.ppo.gamma,
    )
    time_index = np.asarray(anchors.time_indexes, dtype=np.int32)
    lane_index = np.asarray(anchors.lane_indexes, dtype=np.int32)
    state = jax.tree_util.tree_map(
        lambda value: value[time_index, lane_index], records["agent_state"]
    )
    observations = jnp.asarray(records["observations"])[time_index, lane_index]
    _, output = deployment.agent.step(
        base_params=deployment.base_params,
        latent_params=deployment.latent_params,
        state=state,
        observation=observations,
        variant=JOINT_VARIANT,
    )
    run_ids = np.asarray(records["partner_run_ids"])[time_index, lane_index]
    task = np.asarray(output.base.task_features, dtype=np.float64)
    donor = _nearest_different_run_donors(task, run_ids)
    belief = jnp.asarray(output.latent.belief, dtype=jnp.float32)
    donor_belief = belief[jnp.asarray(donor)]
    decision_mean = jnp.asarray(output.latent.decision_mean, dtype=jnp.float32)
    correct_values = expected_action_values(belief, decision_mean)
    shuffled_values = expected_action_values(donor_belief, decision_mean)
    correct_policy = kl_constrained_policy(
        output.base.base_logits,
        correct_values,
        kl_budget=config.method.adaptation_kl_budget,
    )
    shuffled_policy = kl_constrained_policy(
        output.base.base_logits,
        shuffled_values,
        kl_budget=config.method.adaptation_kl_budget,
    )
    source_returns = jnp.asarray(anchors.evaluation_returns, dtype=jnp.float32)
    value_difference = jnp.sum(
        (correct_policy.probabilities - shuffled_policy.probabilities)
        * source_returns,
        axis=-1,
    )
    action_tv = 0.5 * jnp.sum(
        jnp.abs(correct_policy.probabilities - shuffled_policy.probabilities),
        axis=-1,
    )
    rows = [
        {
            "anchor_index": int(index),
            "partner_run_id": int(run_ids[index]),
            "donor_partner_run_id": int(run_ids[donor[index]]),
            "source_task_distance": float(
                np.linalg.norm(task[index] - task[donor[index]])
            ),
            "belief_tv": float(
                0.5
                * np.sum(
                    np.abs(
                        np.asarray(belief[index])
                        - np.asarray(donor_belief[index])
                    )
                )
            ),
            "policy_tv": float(np.asarray(action_tv[index])),
            "source_world_value_difference": float(
                np.asarray(value_difference[index])
            ),
        }
        for index in range(anchor_count)
    ]
    value_host = np.asarray(value_difference, dtype=np.float64)
    summary = _run_bootstrap(
        value_host,
        np.asarray(run_ids),
        replicates=config.evaluation.bootstrap_replicates,
        seed=int(getattr(args, "seed", 0)),
    )
    summary["passed"] = bool(summary["one_sided_lcb_95"] > 0.0)
    elapsed = time.perf_counter() - started
    ledger = ResourceLedger(
        mechanism_evaluation_steps=(
            lanes * config.environment.episode_steps
            + int(budget["counterfactual_continuation_steps"])
        ),
        gpu_hours=gpu_hours_for_wall_seconds(elapsed),
        wall_clock_hours=elapsed / 3600.0,
    )
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "causal_belief_value.parquet"
    write_parquet(raw_path, rows)
    _write_json(
        output_dir / "causal_belief_value.json",
        {
            "version": 1,
            "artifact_type": "unified_delta_causal_belief_value",
            "estimand": "same_source_world_correct_minus_task_matched_shuffled_belief",
            "layout": config.environment.layout,
            "variant": deployment.variant,
            "summary": summary,
            "raw": {"path": str(raw_path), "sha256": sha256_path(raw_path)},
            "sources": {
                "deployment": {
                    "path": str(Path(args.deployment).resolve()),
                    "sha256": sha256_path(args.deployment),
                },
                "partner_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_path(manifest_path),
                },
            },
            "resource_ledger": ledger.to_mapping(),
            "repository_runtime": runtime_provenance(),
        },
    )
    _write_json(output_dir / "resource_ledger.json", ledger.to_mapping())


__all__ = ["run_causal_evaluation"]
