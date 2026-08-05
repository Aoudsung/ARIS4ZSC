"""Feasible, principle-derived DELTA development matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import yaml

from src.delta_zsc.config import CONFIG_VERSION, METHOD_VERSION, RUN_BUDGETS
from src.delta_zsc.storage import ensure_run_identity, read_json, sha256_path, write_json

from .evaluation_app import build_policy_manifest, run_evaluation
from .training_app import run_training


MAIN_VARIANTS = (
    "history_rnn",
    "base",
    "response_only",
    "delta_passive",
    "delta_active",
    "history_rnn_extra",
    "base_extra",
)
K_SENSITIVITY_VARIANTS = ("delta_passive", "delta_active")
DEVELOPMENT_SEEDS = tuple(range(5))


def _anchor_budget(payload: Mapping[str, Any]) -> int:
    base = RUN_BUDGETS["development"].environment_steps
    anchors = payload["anchors"]
    method = payload["method"]
    triggers = base // int(anchors["interval_environment_steps"])
    return triggers * (
        int(anchors["states_per_trigger"])
        * 6
        * (int(anchors["fit_replicas"]) + int(anchors["evaluation_replicas"]))
        * int(method["continuation_horizon"])
    )


def _write_variant_config(
    source: Path,
    target: Path,
    *,
    variant: str,
    component_count: int,
) -> None:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["version"] = CONFIG_VERSION
    payload["method_variant"] = variant.removesuffix("_extra")
    payload["method"]["latent_components"] = int(component_count)
    payload["training"]["extra_ppo_environment_steps"] = (
        _anchor_budget(payload) if variant.endswith("_extra") else 0
    )
    # Controls without a decision emission never collect anchors.
    if payload["method_variant"] in {"history_rnn", "base", "response_only"}:
        payload["anchors"]["enabled"] = False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")




def _validate_training_entries(entries: list[Mapping[str, Any]]) -> None:
    """Mechanically enforce nested and total-budget comparability."""

    index = {
        (int(row["component_count"]), str(row["variant"]), int(row["seed_index"])): row
        for row in entries
    }
    if len(index) != len(entries):
        raise ValueError("Development matrix contains duplicate K/variant/seed rows.")
    for component_count in (2, 4, 8):
        variants = MAIN_VARIANTS if component_count == 4 else K_SENSITIVITY_VARIANTS
        for seed in DEVELOPMENT_SEEDS:
            rows = [index[(component_count, variant, seed)] for variant in variants]
            capacities = {
                int(row["resource_ledger"]["deployable_parameters"]) for row in rows
            }
            if len(capacities) != 1:
                raise ValueError("Deployable capacity differs across a paired development block.")
            if component_count != 4:
                passive = index[(component_count, "delta_passive", seed)]["resource_ledger"]
                active = index[(component_count, "delta_active", seed)]["resource_ledger"]
                if (
                    int(passive["ego_policy_steps"]) != int(active["ego_policy_steps"])
                    or int(passive["anchor_continuation_steps"])
                    != int(active["anchor_continuation_steps"])
                ):
                    raise ValueError("K-sensitivity passive/active budgets differ.")
                continue
            by_variant = {variant: index[(4, variant, seed)]["resource_ledger"] for variant in variants}
            base_ppo = {
                int(by_variant[name]["ego_policy_steps"])
                for name in (
                    "history_rnn", "base", "response_only", "delta_passive", "delta_active"
                )
            }
            if len(base_ppo) != 1:
                raise ValueError("Core development variants have different PPO budgets.")
            anchor_cost = int(by_variant["delta_active"]["anchor_continuation_steps"] )
            if (
                anchor_cost <= 0
                or int(by_variant["delta_passive"]["anchor_continuation_steps"]) != anchor_cost
            ):
                raise ValueError("Passive and active DELTA must share one sparse-anchor budget.")
            for name in ("history_rnn", "base", "response_only", "history_rnn_extra", "base_extra"):
                if int(by_variant[name]["anchor_continuation_steps"]) != 0:
                    raise ValueError(f"{name} must not collect privileged decision anchors.")
            ordinary = next(iter(base_ppo))
            for name in ("history_rnn_extra", "base_extra"):
                if int(by_variant[name]["ego_policy_steps"]) != ordinary + anchor_cost:
                    raise ValueError(f"{name} does not reallocate the exact anchor cost to PPO.")
                if int(by_variant[name]["marginal_training_simulator_steps"]) != int(
                    by_variant["delta_active"]["marginal_training_simulator_steps"]
                ):
                    raise ValueError(f"{name} and DELTA-active total marginal budgets differ.")

def run_development_matrix(args: argparse.Namespace) -> None:
    source = Path(args.config).resolve()
    manifest = Path(args.partner_manifest).resolve()
    output = Path(args.output).resolve()
    seeds = tuple(int(value) for value in args.seed_index)
    if tuple(sorted(seeds)) != DEVELOPMENT_SEEDS:
        raise ValueError("Development matrix uses seed indexes 0..4 exactly once.")
    entries = []
    for component_count in (2, 4, 8):
        variants = MAIN_VARIANTS if component_count == 4 else K_SENSITIVITY_VARIANTS
        for variant in variants:
            config_path = output / "configs" / f"k-{component_count}" / f"{variant}.yaml"
            _write_variant_config(
                source, config_path, variant=variant, component_count=component_count
            )
            for seed in seeds:
                run = output / "runs" / f"k-{component_count}" / variant / f"seed-{seed}"
                run_training(
                    SimpleNamespace(
                        config=str(config_path),
                        partner_manifest=str(manifest),
                        ego_run_id=f"delta-k{component_count}-{variant}-seed-{seed}",
                        seed_index=seed,
                        run_kind="development",
                        output=str(run),
                        resume=bool(args.resume),
                        require_cuda=bool(args.require_cuda),
                        skip_manifest_hash_check=bool(args.skip_manifest_hash_check),
                    )
                )
                identity = read_json(run / "run_identity.json")
                ledger = read_json(run / "resource_ledger.json")
                entries.append(
                    {
                        "component_count": component_count,
                        "variant": variant,
                        "seed_index": seed,
                        "run": str(run),
                        "deployment": str(run / "final_deployment"),
                        "config": {"path": str(config_path), "sha256": sha256_path(config_path)},
                        "config_fingerprint": identity["config_fingerprint"],
                        "resource_ledger": ledger,
                    }
                )
    _validate_training_entries(entries)
    result = {
        "version": 1,
        "artifact_type": "delta_development_matrix",
        "method": METHOD_VERSION,
        "source_config": {"path": str(source), "sha256": sha256_path(source)},
        "partner_manifest": {"path": str(manifest), "sha256": sha256_path(manifest)},
        "seeds": list(seeds),
        "main_k4_variants": list(MAIN_VARIANTS),
        "k_sensitivity_variants": list(K_SENSITIVITY_VARIANTS),
        "budget_capacity_alignment_passed": True,
        "entries": entries,
    }
    ensure_run_identity(
        output,
        {
            "stage": "development-matrix",
            "method": METHOD_VERSION,
            "source_config": result["source_config"],
            "partner_manifest": result["partner_manifest"],
        },
    )
    write_json(output / "development_matrix.json", result)


def evaluate_development_matrix(args: argparse.Namespace) -> None:
    matrix_path = Path(args.matrix).resolve()
    matrix = read_json(matrix_path)
    if matrix.get("artifact_type") != "delta_development_matrix":
        raise ValueError("Development matrix identity differs.")
    output = Path(args.output).resolve()
    evaluations = []
    groups = sorted(
        {
            (int(row["component_count"]), str(row["variant"]))
            for row in matrix["entries"]
        }
    )
    for component_count, variant in groups:
        rows = [
            row
            for row in matrix["entries"]
            if int(row["component_count"]) == component_count
            and row["variant"] == variant
        ]
        manifest_path = output / "policy_manifests" / f"k{component_count}-{variant}.json"
        build_policy_manifest(
            SimpleNamespace(
                policy_kind="delta_deployment",
                policy=[row["deployment"] for row in rows],
                run_count=len(rows),
                method=f"k{component_count}-{variant}",
                layout=args.layout,
                output=str(manifest_path),
            )
        )
        evaluation_dir = output / "evaluations" / f"k{component_count}" / variant
        run_evaluation(
            SimpleNamespace(
                config=rows[0]["config"]["path"],
                run_kind="development",
                policy_manifest=str(manifest_path),
                partner_manifest=args.partner_manifest,
                partner_role="development_coverage",
                skip_manifest_hash_check=bool(args.skip_manifest_hash_check),
                output=str(evaluation_dir),
            )
        )
        evaluations.append(
            {
                "component_count": component_count,
                "variant": variant,
                "directory": str(evaluation_dir),
                "sha256": sha256_path(evaluation_dir),
            }
        )
    write_json(
        output / "development_evaluations.json",
        {
            "version": 1,
            "artifact_type": "delta_development_evaluations",
            "matrix": {"path": str(matrix_path), "sha256": sha256_path(matrix_path)},
            "evaluations": evaluations,
        },
    )


def _rows(path: Path) -> list[Mapping[str, Any]]:
    summary = read_json(path / "evaluation_summary.json")
    raw = Path(summary["raw"]["path"])
    if sha256_path(raw) != summary["raw"]["sha256"]:
        raise ValueError("Development raw evaluation hash differs.")
    return [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line]


def _seed_scores(rows: list[Mapping[str, Any]]) -> np.ndarray:
    ids = sorted({int(row["ego_run_index"]) for row in rows})
    return np.asarray(
        [np.mean([float(row["raw_return"]) for row in rows if int(row["ego_run_index"]) == index]) for index in ids]
    )


def _paired(left: np.ndarray, right: np.ndarray, seed: int) -> Mapping[str, Any]:
    if left.shape != right.shape:
        raise ValueError("Paired development seed arrays differ.")
    difference = left - right
    rng = np.random.default_rng(seed)
    draw = np.mean(
        difference[rng.integers(0, difference.size, size=(9_999, difference.size))],
        axis=1,
    )
    return {
        "estimate": float(np.mean(difference)),
        "interval_95": [float(v) for v in np.quantile(draw, (0.025, 0.975))],
        "interval_99": [float(v) for v in np.quantile(draw, (0.005, 0.995))],
    }


def summarize_development_matrix(args: argparse.Namespace) -> None:
    evaluation = read_json(args.evaluations)
    index = {
        (int(row["component_count"]), str(row["variant"])): _seed_scores(
            _rows(Path(row["directory"]))
        )
        for row in evaluation["evaluations"]
    }
    k4 = {variant: index[(4, variant)] for variant in MAIN_VARIANTS}
    contrasts = {
        "decision_emission": _paired(k4["delta_passive"], k4["response_only"], 1),
        "active_voi": _paired(k4["delta_active"], k4["delta_passive"], 2),
        "active_vs_base": _paired(k4["delta_active"], k4["base"], 3),
        "active_vs_history_rnn": _paired(k4["delta_active"], k4["history_rnn"], 4),
        "active_vs_base_total_budget": _paired(k4["delta_active"], k4["base_extra"], 5),
        "active_vs_history_total_budget": _paired(
            k4["delta_active"], k4["history_rnn_extra"], 6
        ),
    }
    layouts = {
        str(read_json(Path(row["directory"]) / "evaluation_summary.json")["layout"])
        for row in evaluation["evaluations"]
    }
    if len(layouts) != 1:
        raise ValueError("A development summary must cover exactly one layout.")
    write_json(
        args.output,
        {
            "version": 2,
            "artifact_type": "delta_development_summary",
            "method": METHOD_VERSION,
            "layout": next(iter(layouts)),
            "k4_means": {name: float(np.mean(values)) for name, values in k4.items()},
            "primary_contrasts": contrasts,
            "k_sensitivity": {
                str(k): {
                    variant: float(np.mean(index[(k, variant)]))
                    for variant in K_SENSITIVITY_VARIANTS
                }
                for k in (2, 4, 8)
            },
            "sources": {
                "evaluations": {
                    "path": str(Path(args.evaluations).resolve()),
                    "sha256": sha256_path(args.evaluations),
                }
            },
        },
    )


__all__ = [
    "DEVELOPMENT_SEEDS",
    "K_SENSITIVITY_VARIANTS",
    "MAIN_VARIANTS",
    "evaluate_development_matrix",
    "run_development_matrix",
    "summarize_development_matrix",
]
