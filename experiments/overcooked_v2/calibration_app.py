"""Optional post-training safety calibration for DELTA-ZSC V6.

The primary method never imports or executes this wrapper.  This command
calibrates the direct full-context-versus-prior-context policy gain on
run-disjoint frozen partners and writes a separate ``DELTA-ZSC-E2E+Safety``
artifact that references, but never replaces, the primary deployment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.deployment import (
    PRIMARY_ARTIFACT_NAME,
    deployable_parameters,
    load_training_model,
)
from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    VectorEnvironment,
    validate_official_runtime,
)
from src.path_c.anchor_sampling import (
    AnchorRuntime,
    gather_time_lanes,
    make_anchor_functions,
    time_source_stratified_indexes,
    world_from_records,
)
from src.path_c.belief_set_encoder import prior_gaussian_summary
from src.path_c.calibration import (
    calibrate_safety_wrapper,
    calibration_to_mapping,
    empirical_policy_gain,
    predicted_policy_gain,
)
from src.path_c.counterfactual_anchor import collect_counterfactual_anchors
from src.path_c.experiment import (
    ENGINEERING_SEED_INDEX,
    METHOD_VERSION,
    load_config,
    load_partner_manifest,
    official_training_domain_keys,
)
from src.path_c.external_partner import make_external_partner_functions
from src.path_c.resources import ResourceLedger
from src.path_c.runner import collect_rollout, initialize_runner
from src.path_c.storage import (
    calibration_identity,
    ensure_run_identity,
    pytree_fingerprint,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
    write_parquet,
)


SAFETY_ARTIFACT_NAME = "DELTA-ZSC-E2E+Safety"
_MECHANISM_ALIASES = {
    "rnn-sp": "sp",
    "sp": "sp",
    "rnn-op": "op",
    "op": "op",
    "state-augmented": "sa",
    "sa": "sa",
    "fcp": "fcp",
}


def _owned_calibration_runs(manifest: Any, seed_index: int) -> tuple[Any, ...]:
    owner = None if int(seed_index) == ENGINEERING_SEED_INDEX else int(seed_index)
    return tuple(
        run
        for run in manifest.by_role("calibration")
        if run.owner_seed_index == owner
    )


def _validate_calibration_runs(runs: tuple[Any, ...], *, formal: bool) -> None:
    parent_ids = [str(run.parent_training_run_id) for run in runs]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Safety calibration requires independent partner-run blocks.")
    if formal:
        counts = Counter(_MECHANISM_ALIASES.get(run.generation_mechanism) for run in runs)
        if counts != Counter({"sp": 5, "op": 5, "sa": 5, "fcp": 5}):
            raise ValueError(
                "Formal V6 safety calibration requires five independent runs "
                "from each of SP, OP, SA, and FCP."
            )


def _training_support(training_run: str | Path, *, config: Any, params: Any) -> np.ndarray:
    path = Path(training_run).resolve() / "records" / "training_support_latents.json"
    if not path.is_file():
        raise FileNotFoundError(
            "V6 training support is missing; safety calibration cannot substitute "
            "external-only latent rows."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = pytree_fingerprint(deployable_parameters(params))
    if (
        payload.get("method") != METHOD_VERSION
        or payload.get("config_fingerprint") != config.fingerprint
        or payload.get("model_fingerprint") != expected
    ):
        raise ValueError("Safety support belongs to another V6 model or config.")
    values = np.asarray(payload.get("posterior_mean"), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != config.model.latent_dim:
        raise ValueError("Safety support latent shape differs from V6.")
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Safety support contains insufficient or non-finite rows.")
    return values


def _collect_partner_block(
    *, config: Any, deployment: Any, run: Any, key: Any, run_numeric_id: int
) -> tuple[Mapping[str, Any], int]:
    """Collect uniform legal-history anchors and direct policy-gain labels."""

    import jax
    import jax.numpy as jnp

    calibration_config = replace(
        config,
        environment=replace(
            config.environment, num_envs=int(config.calibration.episodes_per_run)
        ),
    )
    environment = VectorEnvironment.create(calibration_config)
    pool = FrozenPartnerPool.from_checkpoints(
        (run.checkpoint,),
        parent_training_run_ids=(run.parent_training_run_id,),
    )
    partner_functions = make_external_partner_functions(
        pool=pool,
        member_indexes=jnp.asarray(0, dtype=jnp.int32),
        latent_dim=config.model.latent_dim,
        code_dim=config.partner_generator.code_dim,
        run_ids=jnp.asarray(run_numeric_id, dtype=jnp.int32),
    )
    runner_key, rollout_key, selection_key, return_key = jax.random.split(key, 4)
    runner = initialize_runner(
        environment=environment,
        model_config=config.model,
        partner_functions=partner_functions,
        random_key=runner_key,
    )._replace(random_key=rollout_key)
    unused_runner, unused_batch, records = collect_rollout(
        state=runner,
        length=config.environment.episode_steps,
        environment=environment,
        model=deployment.model,
        params=deployment.params,
        target_params=deployment.params,
        model_config=config.model,
        partner_functions=partner_functions,
        partner_parameters=None,
        context_dropout_probability_value=0.0,
        context_dropout_root=jax.random.fold_in(rollout_key, 9_001),
        official_shaping_factor=0.0,
        record_mode="anchor_full",
    )
    del unused_runner, unused_batch
    indexes = time_source_stratified_indexes(
        selection_key,
        time_count=config.environment.episode_steps,
        environment_count=environment.num_envs,
        requested=config.calibration.anchors_per_run,
        source_values=records["partner_source"],
    )
    world = world_from_records(records, indexes)
    observations = gather_time_lanes(records["observations"], indexes)
    policy_states = world.ego_state
    count = int(indexes.shape[0])
    functions = make_anchor_functions(
        model=deployment.model,
        model_config=config.model,
        partner_functions=partner_functions,
        environment=environment,
    )
    labels = collect_counterfactual_anchors(
        anchor_ids=(run_numeric_id + 1) * 1_000_000 + jnp.arange(count, dtype=jnp.int64),
        root_keys=jax.random.split(return_key, count),
        world=world,
        rollout_flat_indexes=indexes,
        policy_states=policy_states,
        observations=observations,
        partner_codes=jnp.zeros(
            (count, config.partner_generator.code_dim), dtype=jnp.float32
        ),
        partner_sources=jnp.full((count,), 2, dtype=jnp.int32),
        partner_run_ids=jnp.full((count,), run_numeric_id, dtype=jnp.int32),
        functions=functions,
        action_count=6,
        fit_replicas=config.anchors.fit_replicas,
        evaluation_replicas=0,
        continuation_horizon=config.environment.episode_steps,
        discount=1.0,
        microbatch_size=6 * config.anchors.fit_replicas * min(count, 8),
        runtime=AnchorRuntime(deployment.params, None),
    )
    unused_state, model_output = deployment.model.apply(
        {"params": deployment.params},
        policy_states,
        observations,
        jnp.zeros((count,), dtype=jnp.bool_),
        method=deployment.model.step,
    )
    del unused_state
    prior_summary = prior_gaussian_summary(
        model_output.belief_summary, config.model.latent_dim
    )
    prior_logits = deployment.model.apply(
        {"params": deployment.params},
        model_output.task_features,
        prior_summary,
        method=deployment.model.policy_logits_from_features_and_summary,
    )
    action_values = jnp.minimum(model_output.raw_q1, model_output.raw_q2)
    predicted = predicted_policy_gain(
        action_values, model_output.policy_logits, prior_logits
    )
    empirical = empirical_policy_gain(
        labels.fit_returns_by_action, model_output.policy_logits, prior_logits
    )
    return {
        "predicted_gain": np.asarray(predicted),
        "empirical_gain": np.asarray(empirical),
        "posterior_mean": np.asarray(model_output.belief_mean),
        "full_context_logits": np.asarray(model_output.policy_logits),
        "prior_context_logits": np.asarray(prior_logits),
        "returns_by_action": np.asarray(labels.fit_returns_by_action),
        "flat_index": np.asarray(indexes),
    }, count * 6 * config.anchors.fit_replicas * config.environment.episode_steps


def run_safety_calibration(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

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
    runs = _owned_calibration_runs(manifest, int(args.seed_index))
    _validate_calibration_runs(runs, formal=config.run_kind == "formal")
    if len(runs) < config.calibration.minimum_run_count:
        raise RuntimeError(
            "Safety calibration has too few independent partner-run blocks; "
            "the optional wrapper is not exported."
        )
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
    support = _training_support(args.training_run, config=config, params=deployment.params)
    root = jnp.asarray(
        official_training_domain_keys(int(args.seed_index))["calibration"],
        dtype=jnp.uint32,
    )
    predicted_rows: list[np.ndarray] = []
    empirical_rows: list[np.ndarray] = []
    latent_rows: list[np.ndarray] = []
    labels: list[str] = []
    table: list[Mapping[str, Any]] = []
    attempted = 0
    for index, run in enumerate(runs):
        block, block_steps = _collect_partner_block(
            config=config,
            deployment=deployment,
            run=run,
            key=jax.random.fold_in(root, index),
            run_numeric_id=index,
        )
        attempted += int(block_steps)
        predicted_rows.append(block["predicted_gain"])
        empirical_rows.append(block["empirical_gain"])
        latent_rows.append(block["posterior_mean"])
        labels.extend([run.run_id] * int(block["predicted_gain"].shape[0]))
        for row_index in range(int(block["predicted_gain"].shape[0])):
            table.append(
                {
                    "partner_run_id": run.run_id,
                    "partner_mechanism": run.generation_mechanism,
                    "anchor_index": row_index,
                    "rollout_flat_index": int(block["flat_index"][row_index]),
                    "predicted_gain": float(block["predicted_gain"][row_index]),
                    "empirical_gain": float(block["empirical_gain"][row_index]),
                    "absolute_residual": float(
                        abs(
                            block["predicted_gain"][row_index]
                            - block["empirical_gain"][row_index]
                        )
                    ),
                }
            )
    predicted = np.concatenate(predicted_rows)
    empirical = np.concatenate(empirical_rows)
    latents = np.concatenate(latent_rows)
    model_words = np.frombuffer(
        bytes.fromhex(pytree_fingerprint(deployable_parameters(deployment.params)))[:8],
        dtype=">u4",
    ).astype(np.uint32)
    artifact = calibrate_safety_wrapper(
        predicted_gains=predicted,
        empirical_gains=empirical,
        partner_run_ids=labels,
        training_support_latents=support,
        calibration_latents=latents,
        alpha=config.calibration.alpha,
        support_quantile=config.calibration.support_quantile,
        model_fingerprint=model_words,
    )
    write_json(output / "safety_calibration.json", calibration_to_mapping(artifact))
    write_json(
        output / "safety_wrapper.json",
        {
            "artifact_name": SAFETY_ARTIFACT_NAME,
            "method": METHOD_VERSION,
            "primary_artifact_name": PRIMARY_ARTIFACT_NAME,
            "primary_deployment": str(
                Path(args.training_run).resolve() / "final_deployment"
            ),
            "primary_params_fingerprint": pytree_fingerprint(
                deployable_parameters(deployment.params)
            ),
            "selection_rule": "predicted_gain - run_block_radius > 0 and latent_support",
            "calibration": calibration_to_mapping(artifact),
            "primary_method_replacement": False,
        },
    )
    write_parquet(output / "gain_residuals.parquet", table)
    ledger = ResourceLedger(calibration_steps=attempted)
    write_json(output / "resource_ledger.json", ledger.to_mapping())
    write_json(
        output / "run_metadata.json",
        {
            "artifact_name": SAFETY_ARTIFACT_NAME,
            "partner_run_blocks": len(runs),
            "anchor_rows": len(table),
            "wall_seconds": time.perf_counter() - started,
            "scientific_readout": False,
            "note": "Optional safety calibration is not the primary DELTA-ZSC-E2E method.",
        },
    )
    print(f"Complete optional safety calibration: {output}")


__all__ = ["SAFETY_ARTIFACT_NAME", "run_safety_calibration"]
