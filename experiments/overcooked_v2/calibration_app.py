"""Run-disjoint partner-block conformal calibration for DELTA-ZSC."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from experiments.overcooked_v2.deployment import (
    deployable_parameters,
    export_deployment_bundle,
    load_training_model,
)
from experiments.overcooked_v2.official_policy import OfficialDeltaPolicy
from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    VectorEnvironment,
    validate_official_runtime,
)
from src.path_c.anchor_sampling import collect_anchor_batch, gather_time_lanes
from src.path_c.counterfactual_anchor import anchor_microbatch_candidates
from src.path_c.belief_set_encoder import mixture_moments
from src.path_c.decision_geometry import centered_action_values
from src.path_c.calibration import calibrate_adaptation_gate, calibration_to_mapping
from src.path_c.experiment import (
    METHOD_VERSION,
    load_config,
    load_partner_manifest,
    official_training_domain_keys,
)
from src.path_c.external_partner import make_external_partner_functions
from src.path_c.runner import collect_rollout, initialize_runner
from src.path_c.resources import (
    ResourceLedger,
    gpu_hours_for_wall_seconds,
    measure_policy_inference_latency_ms,
    parameter_count,
    peak_device_memory_bytes,
)
from src.path_c.storage import (
    calibration_identity,
    ensure_run_identity,
    pytree_fingerprint,
    validate_formal_repository_state,
    write_json,
    write_parquet,
    validate_registered_python_runtime,
)


def _host(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _batch_config(config: Any, *, num_envs: int, anchors: int | None = None) -> Any:
    environment = replace(config.environment, num_envs=int(num_envs))
    anchor_config = (
        config.anchors
        if anchors is None
        else replace(config.anchors, states_per_interval=int(anchors))
    )
    return replace(config, environment=environment, anchors=anchor_config)


def _collect_fixed_partner_rollout(
    *,
    config: Any,
    deployment: Any,
    checkpoint: Path,
    run_numeric_id: int,
    key: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Collect one full vectorized episode batch against one frozen run."""

    import jax
    import jax.numpy as jnp

    environment = VectorEnvironment.create(config)
    pool = FrozenPartnerPool.from_checkpoints((checkpoint,))
    partner_functions = make_external_partner_functions(
        pool=pool,
        member_indexes=jnp.asarray(0, dtype=jnp.int32),
        latent_dim=config.model.latent_dim,
        code_dim=config.partner_generator.code_dim,
        run_ids=jnp.asarray(run_numeric_id, dtype=jnp.int32),
    )
    runner_key, collect_key = jax.random.split(key)
    runner = initialize_runner(
        environment=environment,
        model_config=config.model,
        partner_functions=partner_functions,
        random_key=runner_key,
    )
    runner = runner._replace(random_key=collect_key)
    runner, batch, records = collect_rollout(
        state=runner,
        length=config.environment.episode_steps,
        environment=environment,
        model=deployment.model,
        params=deployment.params,
        model_config=config.model,
        partner_functions=partner_functions,
        partner_parameters=None,
        gate_values=jnp.ones((config.environment.num_envs,), dtype=jnp.float32),
        teacher_lane_mask=jnp.zeros(
            (config.environment.num_envs,), dtype=jnp.bool_
        ),
        official_shaping_factor=0.0,
    )
    return environment, partner_functions, runner, batch, records


def _training_support_latents(
    *,
    training_run: str | Path,
    config: Any,
    deployment: Any,
) -> Any:
    """Load final-model support collected from the complete training mixture."""

    import json
    import numpy as np

    root = Path(training_run).resolve()
    support_path = root / "records" / "training_support_latents.npz"
    metadata_path = root / "records" / "training_support_metadata.json"
    if not support_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "The training run lacks the final mixed-partner support collection. "
            "Re-run DELTA-ZSC v5 training rather than approximating support from "
            "only frozen external partners."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_fingerprint") != pytree_fingerprint(deployment.params):
        raise ValueError("Training support latents belong to another model checkpoint.")
    if metadata.get("config_fingerprint") != config.fingerprint:
        raise ValueError("Training support latent config differs from calibration.")
    with np.load(support_path, allow_pickle=False) as payload:
        means = np.asarray(payload["posterior_mean"], dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != config.model.latent_dim:
        raise ValueError("Training support posterior means have the wrong shape.")
    if means.shape[0] < 2 or not np.isfinite(means).all():
        raise ValueError("Training support posterior means are invalid.")
    return means


def run_calibration(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    started = time.perf_counter()
    config = load_config(args.config, run_kind=args.run_kind)
    if config.run_kind == "formal":
        validate_formal_repository_state()
        validate_registered_python_runtime()
        validate_official_runtime()
    manifest = load_partner_manifest(
        args.partner_manifest,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    calibration_runs = tuple(
        run
        for run in manifest.by_role("calibration")
        if run.owner_seed_index == int(args.seed_index)
    )
    if config.run_kind == "formal":
        unowned = [
            run.run_id
            for run in manifest.by_role("calibration")
            if run.owner_seed_index is None
        ]
        if unowned:
            raise ValueError(
                "Formal calibration partners must bind one DELTA outer run: "
                f"{unowned}"
            )
    if len({run.parent_training_run_id for run in calibration_runs}) < config.calibration.minimum_run_count:
        raise ValueError("Partner manifest does not meet the calibration run count.")
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        calibration_identity(
            config=config,
            seed_index=int(args.seed_index),
            training_run=args.training_run,
            manifest=manifest,
        ),
    )
    deployment = load_training_model(args.training_run, config)
    training_latents = _training_support_latents(
        training_run=args.training_run,
        config=config,
        deployment=deployment,
    )

    predicted_rows = []
    empirical_rows = []
    latent_rows = []
    run_labels: list[str] = []
    audit_rows: list[Mapping[str, Any]] = []
    root = jnp.asarray(
        official_training_domain_keys(int(args.seed_index))["calibration"],
        dtype=jnp.uint32,
    )
    microbatch_size: int | None = None
    candidates = anchor_microbatch_candidates(
        maximum_anchor_worlds=config.calibration.anchors_per_run,
        action_count=6,
        replicas=config.anchors.fit_replicas + config.anchors.evaluation_replicas,
    )
    counterfactual_steps = 0
    rollout_steps = 0
    for run_index, run in enumerate(calibration_runs):
        current = _batch_config(
            config,
            num_envs=config.calibration.episodes_per_run,
            anchors=config.calibration.anchors_per_run,
        )
        environment, partner_functions, unused_runner, unused_batch, records = (
            _collect_fixed_partner_rollout(
                config=current,
                deployment=deployment,
                checkpoint=run.checkpoint,
                run_numeric_id=20_000 + run_index,
                key=jax.random.fold_in(root, run_index),
            )
        )
        del unused_runner, unused_batch
        rollout_steps += (
            config.calibration.episodes_per_run * config.environment.episode_steps
        )
        collection_kwargs = dict(
            anchor_domain=20_000 + run_index,
            key=jax.random.fold_in(root, 100_000 + run_index),
            records=records,
            environment=environment,
            model=deployment.model,
            target_params=deployment.params,
            config=current,
            partner_functions=partner_functions,
            partner_parameters=None,
            enable_quotient_interventions=False,
        )
        if microbatch_size is None:
            failures = []
            for candidate in candidates:
                try:
                    anchors, unused_quotient, unused_codes, indexes = (
                        collect_anchor_batch(
                            **collection_kwargs, microbatch_size=int(candidate)
                        )
                    )
                    leaves = jax.tree_util.tree_leaves(anchors)
                    if leaves:
                        jax.block_until_ready(leaves[0])
                    microbatch_size = int(candidate)
                    break
                except Exception as error:
                    message = f"{type(error).__name__}: {error}".lower()
                    if not any(
                        token in message
                        for token in (
                            "out of memory",
                            "resource exhausted",
                            "resource_exhausted",
                        )
                    ):
                        raise
                    failures.append(f"{candidate}: {type(error).__name__}: {error}")
                    jax.clear_caches()
            else:
                raise RuntimeError(
                    "No registered calibration-anchor microbatch fits; budgets "
                    "were not reduced. " + " | ".join(failures)
                )
            write_json(
                output / "anchor_microbatch.json",
                {
                    "candidates": list(candidates),
                    "selected": microbatch_size,
                    "changes_scientific_samples": False,
                },
            )
        else:
            anchors, unused_quotient, unused_codes, indexes = collect_anchor_batch(
                **collection_kwargs, microbatch_size=microbatch_size
            )
        del unused_quotient, unused_codes
        counterfactual_steps += (
            int(anchors.anchor_ids.shape[0])
            * 6
            * (config.anchors.fit_replicas + config.anchors.evaluation_replicas)
            * config.anchors.continuation_horizon
        )
        predicted = centered_action_values(
            gather_time_lanes(records["action_values"], indexes)
        )
        mean, unused_variance = mixture_moments(
            records["mixture_logits"],
            records["mixture_means"],
            records["mixture_log_variances"],
        )
        del unused_variance
        latent = gather_time_lanes(mean, indexes)
        predicted_rows.append(predicted)
        empirical_rows.append(
            centered_action_values(anchors.evaluation_returns_by_action)
        )
        latent_rows.append(latent)
        run_labels.extend([run.run_id] * int(predicted.shape[0]))
        for local_index in range(int(predicted.shape[0])):
            audit_rows.append(
                {
                    "partner_run_id": run.run_id,
                    "parent_training_run_id": run.parent_training_run_id,
                    "generation_mechanism": run.generation_mechanism,
                    "anchor_index": local_index,
                    "predicted_action_values": _host(predicted[local_index]),
                    "fit_action_returns": _host(anchors.fit_returns_by_action[local_index]),
                    "evaluation_action_returns": _host(
                        anchors.evaluation_returns_by_action[local_index]
                    ),
                    "posterior_mean": _host(latent[local_index]),
                }
            )

    predicted_all = jnp.concatenate(predicted_rows, axis=0)
    empirical_all = jnp.concatenate(empirical_rows, axis=0)
    calibration_latents = jnp.concatenate(latent_rows, axis=0)
    artifact = calibrate_adaptation_gate(
        predicted_values=predicted_all,
        empirical_values=empirical_all,
        partner_run_ids=run_labels,
        training_support_latents=training_latents,
        calibration_latents=calibration_latents,
        alpha=config.calibration.alpha,
        support_quantile=config.calibration.support_quantile,
        return_lower_bound=config.anchors.return_lower_bound,
        return_upper_bound=config.anchors.return_upper_bound,
        evaluation_replicas=config.anchors.evaluation_replicas,
        action_count=6,
        model_fingerprint=pytree_fingerprint(
            deployable_parameters(deployment.params)
        ),
    )
    write_json(output / "calibration.json", calibration_to_mapping(artifact))
    write_parquet(output / "calibration_anchors.parquet", audit_rows)
    deployment_bundle = export_deployment_bundle(
        output / "deployment",
        source_training_run=args.training_run,
        deployment=deployment,
        calibration=artifact,
    )
    latency_observation = jnp.zeros(
        tuple(deployment.model.observation_shape), dtype=jnp.float32
    )
    inference_latency_ms = measure_policy_inference_latency_ms(
        OfficialDeltaPolicy(
            deployment.__class__(
                ego_run_id=deployment.ego_run_id,
                config=deployment.config,
                model=deployment.model,
                params=deployable_parameters(deployment.params),
                calibration=artifact,
            )
        ),
        latency_observation,
    )
    write_json(
        output / "run_metadata.json",
        {
            "method": METHOD_VERSION,
            "seed_index": int(args.seed_index),
            "calibration_partner_runs": len(calibration_runs),
            "calibration_anchor_rows": len(run_labels),
            "training_support_latent_rows": int(np.asarray(training_latents).shape[0]),
            "calibration_rollout_steps": rollout_steps,
            "calibration_counterfactual_steps": counterfactual_steps,
            "artifact": calibration_to_mapping(artifact),
            "deployment_bundle": str(deployment_bundle),
            "deployment_params_fingerprint": pytree_fingerprint(
                deployable_parameters(deployment.params)
            ),
            "scientific_readout_allowed": False,
        },
    )
    write_json(
        output / "resource_ledger.json",
        ResourceLedger(
            calibration_steps=rollout_steps + counterfactual_steps,
            gpu_hours=gpu_hours_for_wall_seconds(time.perf_counter() - started),
            peak_memory_bytes=peak_device_memory_bytes(),
            deployable_parameters=parameter_count(
                deployable_parameters(deployment.params)
            ),
            inference_latency_ms=inference_latency_ms,
        ).to_mapping(),
    )
    print(f"Complete DELTA-ZSC calibration: {output / 'calibration.json'}")
    print(f"Pruned deployment bundle: {deployment_bundle}")


__all__ = ["run_calibration"]
