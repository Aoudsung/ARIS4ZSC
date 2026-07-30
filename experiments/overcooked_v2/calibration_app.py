"""Run-disjoint partner-block conformal calibration for DELTA-ZSC."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.overcooked_v2.deployment import (
    deployable_parameters,
    export_deployment_bundle,
    load_training_model,
)
from experiments.overcooked_v2.official_adapter import FrozenPartnerPool, VectorEnvironment
from src.path_c.anchor_sampling import collect_anchor_batch, gather_time_lanes
from src.path_c.belief_set_encoder import mixture_moments
from src.path_c.decision_geometry import centered_action_values
from src.path_c.calibration import calibrate_adaptation_gate, calibration_to_mapping
from src.path_c.experiment import load_config, load_partner_manifest
from src.path_c.external_partner import make_external_partner_functions
from src.path_c.runner import collect_rollout, initialize_runner
from src.path_c.storage import (
    calibration_identity,
    ensure_run_identity,
    pytree_fingerprint,
    write_json,
    write_parquet,
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

    config = load_config(args.config, run_kind=args.run_kind)
    manifest = load_partner_manifest(
        args.partner_manifest,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    calibration_runs = manifest.by_role("calibration")
    if len({run.parent_training_run_id for run in calibration_runs}) < config.calibration.minimum_run_count:
        raise ValueError("Partner manifest does not meet the calibration run count.")
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        calibration_identity(
            config=config,
            seed=int(args.seed),
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
    root = jax.random.PRNGKey(int(args.seed))
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
        anchors, unused_quotient, unused_codes, indexes = collect_anchor_batch(
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
        del unused_quotient, unused_codes
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
    write_json(
        output / "run_metadata.json",
        {
            "method": "delta_zsc_v5_decision_equivalent_bayes_r1",
            "calibration_partner_runs": len(calibration_runs),
            "calibration_anchor_rows": len(run_labels),
            "training_support_latent_rows": int(np.asarray(training_latents).shape[0]),
            "artifact": calibration_to_mapping(artifact),
            "deployment_bundle": str(deployment_bundle),
            "deployment_params_fingerprint": pytree_fingerprint(
                deployable_parameters(deployment.params)
            ),
            "scientific_readout_allowed": False,
        },
    )
    print(f"Complete DELTA-ZSC calibration: {output / 'calibration.json'}")
    print(f"Pruned deployment bundle: {deployment_bundle}")


__all__ = ["run_calibration"]
