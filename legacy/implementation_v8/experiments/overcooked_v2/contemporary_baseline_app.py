"""Register honest in-repository proxies for contemporary adaptation baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

from experiments.overcooked_v2.deployment import load_deployment
from src.path_c.experiment import OFFICIAL_SOURCE_COMMIT, load_config
from src.path_c.resources import ResourceLedger, parameter_count
from src.path_c.storage import sha256_path, write_json


CONTEMPORARY_PROXY_METHODS = (
    "history-context-proxy",
    "recbayes-filter-proxy",
    "deterministic-context-proxy",
)

_PROXY_VARIANTS = {
    "history-context-proxy": "r0",
    "recbayes-filter-proxy": "b1",
    "deterministic-context-proxy": "deterministic_context",
}

_PROXY_DESCRIPTIONS = {
    "history-context-proxy": (
        "full-history recurrent context proxy; not a faithful CooT/Transformer reproduction"
    ),
    "recbayes-filter-proxy": (
        "recurrent exact categorical response-filter proxy; not a faithful RecBayes reproduction"
    ),
    "deterministic-context-proxy": (
        "capacity-matched deterministic recurrent context without an exact Bayes posterior"
    ),
}


def run_build_contemporary_proxy_manifest(args: argparse.Namespace) -> None:
    method = str(args.method)
    expected_variant = _PROXY_VARIANTS[method]
    config = load_config(args.config, run_kind="formal")
    from experiments.overcooked_v2.official_evaluation_app import (
        _load_policy_manifest,
    )

    reference_path = Path(args.depi_reference_manifest).resolve()
    reference = _load_policy_manifest(
        reference_path, expected_layout=config.environment.layout
    )
    if reference["method"] != "depi":
        raise ValueError("The contemporary-proxy reference must be a DEPI manifest.")
    reference_pool_hashes = set()
    reference_ego_steps = set()
    for row in reference["runs"]:
        reference_deployment = Path(str(row["checkpoint"])).resolve()
        reference_bundle = json.loads(
            (reference_deployment / "deployment_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        reference_training = Path(
            str(reference_bundle["source_training_run"])
        ).resolve()
        reference_identity = json.loads(
            (reference_training / "run_identity.json").read_text(encoding="utf-8")
        )
        partner_manifest = reference_identity.get("partner_manifest")
        if not isinstance(partner_manifest, Mapping):
            raise ValueError("The DEPI reference has no frozen partner-pool lineage.")
        reference_pool_hashes.add(str(partner_manifest["sha256"]))
        reference_ledger = ResourceLedger.from_mapping(
            json.loads(
                (reference_training / "resource_ledger.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        reference_ego_steps.add(int(reference_ledger.ego_policy_steps))
    if len(reference_pool_hashes) != 1 or len(reference_ego_steps) != 1:
        raise ValueError("DEPI reference runs do not share one pool and interaction budget.")
    reference_pool_hash = next(iter(reference_pool_hashes))
    reference_steps = next(iter(reference_ego_steps))
    reference_capacity = int(reference["deployment_parameter_count"])
    deployments = tuple(Path(value).resolve() for value in args.deployments)
    if len(deployments) != 10:
        raise ValueError("A contemporary proxy manifest requires ten frozen deployments.")
    runs = []
    lineage: dict[tuple[str, str, Any, str], Mapping[str, Any]] = {}
    seed_indexes = set()
    partner_manifest_hashes = set()
    ego_steps = set()
    parameter_counts = set()
    for deployment_path in deployments:
        bundle = json.loads(
            (deployment_path / "deployment_bundle.json").read_text(encoding="utf-8")
        )
        training_run = Path(str(bundle["source_training_run"])).resolve()
        identity = json.loads(
            (training_run / "run_identity.json").read_text(encoding="utf-8")
        )
        if (
            identity.get("method_variant") != expected_variant
            or identity.get("layout") != config.environment.layout
        ):
            raise ValueError("Contemporary proxy deployment uses the wrong mechanism/layout.")
        seed_index = int(identity["seed_index"])
        if seed_index in seed_indexes or seed_index not in range(10):
            raise ValueError("Contemporary proxy seeds must be indexes 0..9 exactly once.")
        seed_indexes.add(seed_index)
        partner_manifest = identity.get("partner_manifest")
        if not isinstance(partner_manifest, Mapping):
            raise ValueError("Contemporary proxy has no frozen partner-pool lineage.")
        partner_manifest_hashes.add(str(partner_manifest["sha256"]))
        ledger = ResourceLedger.from_mapping(
            json.loads(
                (training_run / "resource_ledger.json").read_text(encoding="utf-8")
            )
        )
        ego_steps.add(int(ledger.ego_policy_steps))
        deployment = load_deployment(deployment_path, config)
        parameter_counts.add(parameter_count(deployment.params))
        run_id = str(identity["ego_run_id"])
        checkpoint_hash = sha256_path(deployment_path)
        runs.append(
            {
                "run_index": seed_index,
                "run_id": run_id,
                "parent_training_run_id": run_id,
                "co_training_group_id": None,
                "checkpoint": str(deployment_path),
                "checkpoint_sha256": checkpoint_hash,
            }
        )
        rows = partner_manifest.get("runs")
        if not isinstance(rows, list):
            raise ValueError("Contemporary proxy partner lineage is incomplete.")
        for row in rows:
            key = (
                str(row["checkpoint_sha256"]),
                str(row["parent_training_run_id"]),
                row["co_training_group_id"],
                f"proxy_training_{row['role']}",
            )
            lineage[key] = {
                "checkpoint_sha256": key[0],
                "parent_training_run_id": key[1],
                "co_training_group_id": key[2],
                "role": key[3],
            }
        lineage[(checkpoint_hash, run_id, None, "proxy_ego")] = {
            "checkpoint_sha256": checkpoint_hash,
            "parent_training_run_id": run_id,
            "co_training_group_id": None,
            "role": "proxy_ego",
        }
    if seed_indexes != set(range(10)):
        raise ValueError("Contemporary proxy deployments must cover seeds 0..9.")
    if len(partner_manifest_hashes) != 1 or len(ego_steps) != 1:
        raise ValueError("Contemporary proxy runs do not share pool and interaction budget.")
    if len(parameter_counts) != 1:
        raise ValueError("Contemporary proxy deployable capacities differ across seeds.")
    if (
        next(iter(partner_manifest_hashes)) != reference_pool_hash
        or next(iter(ego_steps)) != reference_steps
        or next(iter(parameter_counts)) != reference_capacity
    ):
        raise ValueError(
            "Contemporary proxy does not match DEPI's partner pool, interaction "
            "budget, and deployable capacity."
        )
    write_json(
        args.output,
        {
            "version": 1,
            "layout": config.environment.layout,
            "method": method,
            "policy_kind": "depi_deployment",
            "official_source_commit": OFFICIAL_SOURCE_COMMIT,
            "runs": sorted(runs, key=lambda row: int(row["run_index"])),
            "training_lineage": list(lineage.values()),
            "proxy_metadata": {
                "is_proxy": True,
                "not_a_published_method_reproduction": True,
                "closest_in_repository_variant": expected_variant,
                "description": _PROXY_DESCRIPTIONS[method],
                "legal_information_boundary": "SCIENTIFIC_SPEC L1-L5",
                "partner_manifest_sha256": next(iter(partner_manifest_hashes)),
                "ego_policy_steps_per_run": next(iter(ego_steps)),
                "deployable_parameter_count": next(iter(parameter_counts)),
                "depi_reference_manifest": {
                    "path": str(reference_path),
                    "sha256": sha256_path(reference_path),
                },
                "same_training_partner_pool": True,
                "same_ego_interaction_budget": True,
                "same_deployable_capacity": True,
            },
        },
    )


def run_common_proxy_evaluation(args: argparse.Namespace) -> None:
    """Evaluate all three honest proxies on the frozen Common-Partner panel."""

    import jax
    import numpy as np

    from experiments.overcooked_v2.common_partner_app import (
        COMMON_PARTNER_COUNTS,
        _load_common_panel_policies,
        _official_environment,
        _validate_common_panel,
    )
    from experiments.overcooked_v2.official_adapter import (
        official_pairing_rollouts,
        validate_official_runtime,
    )
    from experiments.overcooked_v2.official_evaluation_app import (
        _load_policies,
        _load_policy_manifest,
    )
    from src.path_c.evaluation import common_partner_summary
    from src.path_c.experiment import (
        OFFICIAL_EVALUATION_ROOT_SEED,
        load_partner_manifest,
    )
    from src.path_c.resources import ResourceLedger, gpu_hours_for_wall_seconds
    from src.path_c.storage import (
        ensure_run_identity,
        runtime_provenance,
        validate_formal_repository_state,
        validate_registered_python_runtime,
        write_parquet,
    )

    started = time.perf_counter()
    validate_formal_repository_state()
    validate_registered_python_runtime()
    runtime = validate_official_runtime()
    config = load_config(args.config, run_kind="formal")
    manifests = {}
    sources = {}
    for value in args.policy_manifest:
        method, raw_path = value.split("=", 1)
        if method in manifests or method not in CONTEMPORARY_PROXY_METHODS:
            raise ValueError("Common proxy manifests must cover each registered proxy once.")
        path = Path(raw_path).resolve()
        manifest = _load_policy_manifest(
            path, expected_layout=config.environment.layout
        )
        if manifest["method"] != method:
            raise ValueError("Common proxy policy-manifest label differs.")
        manifests[method] = manifest
        sources[method] = {"path": str(path), "sha256": sha256_path(path)}
    if set(manifests) != set(CONTEMPORARY_PROXY_METHODS):
        raise ValueError("Common proxy evaluation requires all three proxy manifests.")
    training_pools = {
        manifest["proxy_metadata"]["partner_manifest_sha256"]
        for manifest in manifests.values()
    }
    if len(training_pools) != 1:
        raise ValueError("Common proxies do not share the same training partner pool.")
    panel_path = Path(args.partner_manifest).resolve()
    panel = load_partner_manifest(
        panel_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    partners = _validate_common_panel(panel, manifests)
    partner_descriptors, partner_policies = _load_common_panel_policies(partners)
    environment = _official_environment(config)
    root = jax.random.PRNGKey(OFFICIAL_EVALUATION_ROOT_SEED)
    rows = []
    summaries = {}
    for method in CONTEMPORARY_PROXY_METHODS:
        method_rows = []
        ego_policies = _load_policies(manifests[method], config)
        for ego_index, ego in enumerate(ego_policies):
            for partner_index, (partner_run, partner) in enumerate(
                zip(partner_descriptors, partner_policies, strict=True)
            ):
                for ego_role in (0, 1):
                    left, right = (ego, partner) if ego_role == 0 else (partner, ego)
                    rollouts, unused_keys = official_pairing_rollouts(
                        left_policy=left,
                        right_policy=right,
                        environment=environment,
                        root_key=root,
                        episodes=500,
                    )
                    del unused_keys
                    returns = np.asarray(rollouts.total_reward, dtype=np.float64)
                    for episode_index, raw_return in enumerate(returns):
                        row = {
                            "layout": config.environment.layout,
                            "method": method,
                            "ego_run_index": ego_index,
                            "partner_run_id": partner_run.run_id,
                            "partner_run_index": partner_index,
                            "partner_mechanism": partner_run.generation_mechanism,
                            "ego_role": ego_role,
                            "episode_index": episode_index,
                            "raw_return": float(raw_return),
                        }
                        method_rows.append(row)
                        rows.append(row)
        summaries[method] = common_partner_summary(
            method_rows, expected_partner_counts=COMMON_PARTNER_COUNTS
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensure_run_identity(
        output,
        {
            "stage": "common-partner-contemporary-proxies",
            "layout": config.environment.layout,
            "official_runtime": runtime,
            "repository_runtime": runtime_provenance(),
            "config_fingerprint": config.fingerprint,
            "partner_manifest": {
                "path": str(panel_path),
                "sha256": sha256_path(panel_path),
            },
            "policy_manifests": sources,
            "evaluation_root_key": [0, OFFICIAL_EVALUATION_ROOT_SEED],
        },
    )
    raw_path = output / "common_proxy_episode_returns.parquet"
    write_parquet(raw_path, rows)
    elapsed_wall_seconds = time.perf_counter() - started
    ledger = ResourceLedger(
        evaluation_steps=len(rows) * 400,
        gpu_hours=gpu_hours_for_wall_seconds(elapsed_wall_seconds),
        wall_clock_hours=elapsed_wall_seconds / 3_600.0,
    )
    write_json(
        output / "common_proxy_summary.json",
        {
            "version": 1,
            "artifact_type": "depi_contemporary_proxy_common_partner_summary",
            "proxy_only": True,
            "not_published_method_reproductions": True,
            "layout": config.environment.layout,
            "methods": summaries,
            "sources": sources,
            "raw": {"path": str(raw_path), "sha256": sha256_path(raw_path)},
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())


__all__ = [
    "CONTEMPORARY_PROXY_METHODS",
    "run_build_contemporary_proxy_manifest",
    "run_common_proxy_evaluation",
]
