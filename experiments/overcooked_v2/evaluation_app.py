"""Raw-node Official and Common-Partner evaluation for DELTA-ZSC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from src.delta_zsc.config import METHOD_VERSION, load_config
from src.delta_zsc.manifest import load_partner_manifest
from src.delta_zsc.resources import ResourceLedger
from src.delta_zsc.storage import ensure_run_identity, read_json, sha256_path, write_json

from .deployment import load_deployment
from .official_policy import OfficialDELTAPolicy


POLICY_MANIFEST_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_policy_manifest(args: argparse.Namespace) -> None:
    kind = str(args.policy_kind)
    paths = tuple(Path(value).resolve() for value in args.policy)
    if len(paths) != int(args.run_count):
        raise ValueError("Policy manifest run count differs.")
    runs = []
    for index, path in enumerate(paths):
        if kind == "delta_deployment":
            deployment = load_deployment(path)
            if deployment.config.environment.layout != args.layout:
                raise ValueError("DELTA deployment layout differs.")
            identity = read_json(path / "deployment_bundle.json")
            run_id = deployment.ego_run_id
        else:
            if not path.exists():
                raise FileNotFoundError(path)
            identity = None
            run_id = f"{args.method}-run-{index}"
        runs.append(
            {
                "run_index": index,
                "run_id": run_id,
                "policy": str(path),
                "policy_sha256": sha256_path(path),
                "identity": identity,
            }
        )
    write_json(
        args.output,
        {
            "version": POLICY_MANIFEST_VERSION,
            "method": str(args.method),
            "layout": str(args.layout),
            "policy_kind": kind,
            "runs": runs,
        },
    )


def _load_policy(row: Mapping[str, Any], kind: str) -> Any:
    if kind == "delta_deployment":
        return OfficialDELTAPolicy(load_deployment(row["policy"]))
    from .official_adapter import official_policy, restore_official_checkpoint

    config, params = restore_official_checkpoint(row["policy"])
    return official_policy(params, config)


def run_evaluation(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    import jax

    from .official_adapter import (
        VectorEnvironment,
        official_pairing_rollouts,
        official_policy,
        restore_official_checkpoint,
        validate_official_runtime,
    )

    config = load_config(args.config, run_kind=args.run_kind)
    if config.run_kind == "formal":
        validate_official_runtime()
    manifest_path = Path(args.policy_manifest).resolve()
    policy_manifest = read_json(manifest_path)
    if (
        policy_manifest.get("version") != POLICY_MANIFEST_VERSION
        or policy_manifest.get("layout") != config.environment.layout
    ):
        raise ValueError("Policy manifest identity differs.")
    manifest_runs = policy_manifest.get("runs")
    if not isinstance(manifest_runs, list) or not manifest_runs:
        raise ValueError("Policy manifest has no runs.")
    if [int(row.get("run_index", -1)) for row in manifest_runs] != list(
        range(len(manifest_runs))
    ):
        raise ValueError("Policy manifest run indexes must be contiguous from zero.")
    for row in manifest_runs:
        policy_path = Path(str(row["policy"])).resolve()
        if sha256_path(policy_path) != str(row["policy_sha256"]):
            raise ValueError("Policy manifest checkpoint/deployment hash differs.")
    if config.run_kind == "formal" and len(manifest_runs) != config.evaluation.minimum_ego_runs:
        raise ValueError("Formal evaluation requires ten independent ego runs.")

    partner_manifest_path = Path(args.partner_manifest).resolve()
    partner_manifest = load_partner_manifest(
        partner_manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    partners = partner_manifest.by_role(str(args.partner_role))
    if not partners:
        raise ValueError("Evaluation partner role is empty.")
    if config.run_kind == "formal" and str(args.partner_role) == "confirmatory":
        aliases = {"rnn-sp": "sp", "rnn-op": "op", "state-augmented": "sa"}
        counts: dict[str, int] = {}
        for run in partners:
            mechanism = aliases.get(
                str(run.generation_mechanism).lower(),
                str(run.generation_mechanism).lower(),
            )
            counts[mechanism] = counts.get(mechanism, 0) + 1
        if len(counts) < 4 or min(counts.values()) < int(
            config.evaluation.minimum_partner_runs_per_mechanism
        ):
            raise ValueError(
                "Formal confirmatory evaluation requires at least four mechanisms "
                "and four independent parents per mechanism."
            )
    environment = VectorEnvironment.create(config).environment
    ego_policies = [
        _load_policy(row, policy_manifest["policy_kind"])
        for row in policy_manifest["runs"]
    ]
    partner_policies = []
    for run in partners:
        partner_config, partner_params = restore_official_checkpoint(run.checkpoint)
        partner_policies.append(official_policy(partner_params, partner_config))
    root = jax.random.PRNGKey(int(config.evaluation.evaluation_seed))
    rows = []
    for ego_index, ego in enumerate(ego_policies):
        for partner_index, (partner_run, partner) in enumerate(
            zip(partners, partner_policies, strict=True)
        ):
            roles = (0, 1) if config.evaluation.evaluate_both_roles else (0,)
            for role in roles:
                left, right = (ego, partner) if role == 0 else (partner, ego)
                key = jax.random.fold_in(
                    jax.random.fold_in(root, ego_index * 10_000 + partner_index), role
                )
                rollouts, _ = official_pairing_rollouts(
                    left_policy=left,
                    right_policy=right,
                    environment=environment,
                    root_key=key,
                    episodes=config.evaluation.episodes_per_pairing,
                )
                returns = np.asarray(rollouts.total_reward, dtype=np.float64)
                for episode, value in enumerate(returns):
                    rows.append(
                        {
                            "layout": config.environment.layout,
                            "method": policy_manifest["method"],
                            "ego_run_index": ego_index,
                            "ego_run_id": policy_manifest["runs"][ego_index]["run_id"],
                            "partner_run_index": partner_index,
                            "partner_run_id": partner_run.run_id,
                            "partner_mechanism": partner_run.generation_mechanism,
                            "ego_role": role,
                            "episode_index": episode,
                            "raw_return": float(value),
                        }
                    )
    output = Path(args.output).resolve()
    identity = {
        "stage": "official-evaluation",
        "layout": config.environment.layout,
        "method": policy_manifest["method"],
        "config_fingerprint": config.fingerprint,
        "policy_manifest": {"path": str(manifest_path), "sha256": sha256_path(manifest_path)},
        "partner_manifest": {
            "path": str(partner_manifest_path),
            "sha256": sha256_path(partner_manifest_path),
        },
        "partner_role": str(args.partner_role),
    }
    ensure_run_identity(output, identity)
    raw = output / "episode_returns.jsonl"
    _write_jsonl(raw, rows)
    elapsed = time.perf_counter() - started
    ledger = ResourceLedger(
        evaluation_steps=len(rows) * config.environment.episode_steps,
        measurement_wall_clock_hours=elapsed / 3600.0,
        measurement_gpu_hours=(
            elapsed / 3600.0
            if any(device.platform == "gpu" for device in jax.devices())
            else 0.0
        ),
    )
    write_json(
        output / "evaluation_summary.json",
        {
            "version": EVALUATION_SCHEMA_VERSION,
            "artifact_type": "delta_raw_evaluation",
            "layout": config.environment.layout,
            "method": policy_manifest["method"],
            "mean_return": float(np.mean([row["raw_return"] for row in rows])),
            "episode_count": len(rows),
            "raw": {"path": str(raw), "sha256": sha256_path(raw)},
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())


def _node_matrix(
    rows: list[Mapping[str, Any]]
) -> tuple[np.ndarray, list[int], list[str]]:
    """Return a run-index × partner-run matrix.

    Different methods necessarily have different run IDs.  Pairing is defined
    by the preregistered seed/run index, while the partner nodes must be the
    same immutable runs.  Requiring identical ego strings would incorrectly
    reject every cross-method comparison.
    """

    egos = sorted({int(row["ego_run_index"]) for row in rows})
    partners = sorted({str(row["partner_run_id"]) for row in rows})
    if egos != list(range(len(egos))):
        raise ValueError("Evaluation ego run indexes must be contiguous from zero.")
    matrix = np.empty((len(egos), len(partners)), dtype=np.float64)
    for i, ego in enumerate(egos):
        for j, partner in enumerate(partners):
            values = [
                float(row["raw_return"])
                for row in rows
                if int(row["ego_run_index"]) == ego
                and str(row["partner_run_id"]) == partner
            ]
            if not values:
                raise ValueError("Evaluation node matrix is incomplete.")
            matrix[i, j] = np.mean(values)
    return matrix, egos, partners


def _bootstrap_difference(
    left: np.ndarray, right: np.ndarray, *, replicates: int, seed: int
) -> Mapping[str, Any]:
    if left.shape != right.shape:
        raise ValueError("Paired node matrices differ.")
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(replicates),), dtype=np.float64)
    for index in range(int(replicates)):
        ego = rng.integers(0, left.shape[0], size=left.shape[0])
        partner = rng.integers(0, left.shape[1], size=left.shape[1])
        draws[index] = np.mean(left[np.ix_(ego, partner)] - right[np.ix_(ego, partner)])
    return {
        "estimate": float(np.mean(left - right)),
        "interval_95": [float(v) for v in np.quantile(draws, (0.025, 0.975))],
        "one_sided_lcb": float(np.quantile(draws, 0.05)),
    }


def summarize_evaluations(args: argparse.Namespace) -> None:
    sources = {}
    matrices = {}
    ego_indexes = partner_ids = None
    layout = None
    for value in args.evaluation:
        method, raw_path = value.split("=", 1)
        if method in matrices:
            raise ValueError(f"Duplicate evaluation method: {method}")
        directory = Path(raw_path).resolve()
        summary = read_json(directory / "evaluation_summary.json")
        if (
            summary.get("artifact_type") != "delta_raw_evaluation"
            or summary.get("method") != method
        ):
            raise ValueError("Evaluation summary label differs from its registered method.")
        raw = Path(summary["raw"]["path"])
        if sha256_path(raw) != summary["raw"]["sha256"]:
            raise ValueError("Raw evaluation hash differs.")
        rows = _read_jsonl(raw)
        if not rows or any(
            str(row.get("method")) != method
            or str(row.get("layout")) != str(summary.get("layout"))
            for row in rows
        ):
            raise ValueError("Raw evaluation method/layout identity differs.")
        matrix, current_egos, current_partners = _node_matrix(rows)
        current_layout = str(summary["layout"])
        if ego_indexes is None:
            ego_indexes, partner_ids, layout = (
                current_egos,
                current_partners,
                current_layout,
            )
        elif (
            current_egos != ego_indexes
            or current_partners != partner_ids
            or current_layout != layout
        ):
            raise ValueError(
                "Evaluation run-index, partner, or layout nodes differ across methods."
            )
        matrices[method] = matrix
        sources[method] = {"path": str(directory), "sha256": sha256_path(directory)}
    if "delta-active" not in matrices:
        raise ValueError("Summary requires delta-active.")
    baseline_methods = sorted(name for name in matrices if name != "delta-active")
    if not baseline_methods:
        raise ValueError("Official summary requires at least one same-protocol baseline.")
    strongest = max(
        baseline_methods, key=lambda name: float(np.mean(matrices[name]))
    )
    contrasts = {
        name: _bootstrap_difference(
            matrices["delta-active"],
            matrices[name],
            replicates=int(args.bootstrap_replicates),
            seed=int(args.seed) + index,
        )
        for index, name in enumerate(baseline_methods)
    }
    all_baselines_gate = bool(
        all(float(value["one_sided_lcb"]) > 0.0 for value in contrasts.values())
        and all(float(value["estimate"]) >= 20.0 for value in contrasts.values())
    )
    write_json(
        args.output,
        {
            "version": 2,
            "artifact_type": "delta_official_summary",
            "method": METHOD_VERSION,
            "layout": layout,
            "ego_run_indexes": ego_indexes,
            "partner_run_ids": partner_ids,
            "means": {name: float(np.mean(value)) for name, value in matrices.items()},
            "strongest_baseline": strongest,
            "delta_active_vs_strongest": contrasts[strongest],
            "delta_active_vs_each_baseline": contrasts,
            "all_baselines_material_superiority_gate": all_baselines_gate,
            "testing_rule": (
                "intersection_union: every registered baseline contrast must have "
                "one-sided LCB>0 and estimate>=20"
            ),
            "sources": sources,
        },
    )


__all__ = [
    "build_policy_manifest",
    "run_evaluation",
    "summarize_evaluations",
]
