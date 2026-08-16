"""Build the registered CETR-ZSC claim report from evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.cetr_zsc.config import FORMAL_METHOD_LABEL, TAIL_MASS
from src.cetr_zsc.manifest import load_partner_manifest
from src.cetr_zsc.storage import read_json, write_json


BOOTSTRAP_REPLICATES = 9_999


def _artifact_payload(value: str | Path, *, preferred: tuple[str, ...]) -> Mapping[str, Any]:
    source = Path(value).resolve()
    if source.is_dir():
        for name in preferred:
            candidate = source / name
            if candidate.is_file():
                source = candidate
                break
        else:
            raise FileNotFoundError(f"No recognized artifact in {source}.")
    payload = read_json(source)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Artifact is not a JSON object: {source}")
    return payload


def _evaluation_summary(value: str | Path) -> Mapping[str, Any]:
    return _artifact_payload(value, preferred=("evaluation_summary.json",))


def _read_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    source = Path(path).resolve()
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _ego_index(row: Mapping[str, Any]) -> int:
    for field in ("ego_run_index", "seed_index"):
        if field in row and row[field] is not None:
            return int(row[field])
    raise ValueError(
        "Evaluation row lacks ego_run_index or seed_index for method alignment."
    )


def _common_units(summary: Mapping[str, Any]) -> dict[tuple[int, str], float]:
    if summary.get("evaluation_mode") != "common_partner":
        raise ValueError("Claim external inputs must be common-partner evaluations.")
    raw = summary.get("raw")
    partner = summary.get("partner_manifest")
    if not isinstance(raw, Mapping) or not isinstance(partner, Mapping):
        raise ValueError("Evaluation artifact lacks raw rows or partner manifest.")
    raw_path = Path(str(raw["path"])).resolve()
    partner_path = Path(str(partner["path"])).resolve()
    manifest = load_partner_manifest(
        partner_path,
        expected_layout=str(summary["layout"]),
        verify_files=False,
    )
    parent_by_run = {str(run.run_id): str(run.parent_training_run_id) for run in manifest.runs}
    totals: dict[tuple[int, str], list[float]] = {}
    for row in _read_jsonl(raw_path):
        partner_id = str(row["partner_run_id"])
        if partner_id not in parent_by_run:
            raise ValueError(f"Evaluation row references an unknown partner: {partner_id}")
        key = (_ego_index(row), parent_by_run[partner_id])
        totals.setdefault(key, []).append(float(row["raw_return"]))
    if not totals:
        raise ValueError("Evaluation artifact contains no external observations.")
    return {key: float(np.mean(values)) for key, values in totals.items()}


def _merge_evaluations(values: list[str]) -> tuple[dict[tuple[int, str], float], list[Mapping[str, Any]]]:
    if not values:
        raise ValueError("At least one evaluation artifact is required.")
    merged: dict[tuple[int, str], list[float]] = {}
    summaries = []
    for value in values:
        summary = _evaluation_summary(value)
        summaries.append(summary)
        method = str(summary.get("method", ""))
        if not method:
            raise ValueError("Evaluation summary lacks method identity.")
        for key, result in _common_units(summary).items():
            merged.setdefault(key, []).append(result)
    return {key: float(np.mean(result)) for key, result in merged.items()}, summaries


def _matrix(units: Mapping[tuple[int, str], float]) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...]]:
    egos = tuple(sorted({key[0] for key in units}))
    parents = tuple(sorted({key[1] for key in units}))
    if not egos or len(parents) < 2:
        raise ValueError("External claim input lacks enough ego runs or parent lineages.")
    expected = {(ego, parent) for ego in egos for parent in parents}
    if set(units) != expected:
        raise ValueError("External evaluation cells are incomplete.")
    values = np.asarray(
        [[float(units[(ego, parent)]) for parent in parents] for ego in egos],
        dtype=np.float64,
    )
    return values, egos, parents


def _cvar50(values: np.ndarray) -> float:
    count = max(1, int(np.ceil(values.shape[1] * float(TAIL_MASS))))
    ordered = np.sort(values, axis=1)[:, :count]
    return float(np.mean(ordered))


def _bootstrap_external(
    left: np.ndarray,
    right: np.ndarray,
    *,
    statistic: str,
    seed: int = 0,
) -> Mapping[str, Any]:
    if left.shape != right.shape:
        raise ValueError("CETR and FCP external cells are not aligned.")
    rng = np.random.default_rng(int(seed))
    ego_count, parent_count = left.shape
    point = (
        float(np.mean(left - right))
        if statistic == "mean"
        else float(_cvar50(left) - _cvar50(right))
    )
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        selected_ego = rng.integers(0, ego_count, size=ego_count)
        selected_parent = rng.integers(0, parent_count, size=parent_count)
        crossed_left = left[np.ix_(selected_ego, selected_parent)]
        crossed_right = right[np.ix_(selected_ego, selected_parent)]
        if statistic == "mean":
            draws[index] = float(np.mean(crossed_left - crossed_right))
        else:
            draws[index] = float(_cvar50(crossed_left) - _cvar50(crossed_right))
    return {
        "estimate": point,
        "interval_95": [float(value) for value in np.quantile(draws, (0.025, 0.975))],
        "lcb95": float(np.quantile(draws, 0.05)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "ego_run_x_parent_lineage",
    }


def _population_diagonal(value: str | Path) -> np.ndarray:
    payload = _artifact_payload(
        value,
        preferred=("population_matrix_summary.json", "evaluation_summary.json"),
    )
    if payload.get("evaluation_mode") == "population_matrix" and isinstance(
        payload.get("population_summary"), Mapping
    ):
        payload = _artifact_payload(
            str(payload["population_summary"]["path"]),
            preferred=("population_matrix_summary.json",),
        )
    diagonal = payload.get("sp_diagonal")
    if diagonal is None and payload.get("cell_means") is not None:
        diagonal = np.diag(np.asarray(payload["cell_means"], dtype=np.float64)).tolist()
    if not isinstance(diagonal, list) or not diagonal:
        raise ValueError("CETR population artifact lacks an SP diagonal.")
    return np.asarray(diagonal, dtype=np.float64)


def _bootstrap_sp(differences: np.ndarray, *, seed: int = 0) -> Mapping[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size != 10:
        raise ValueError("SP claim requires exactly ten paired seed differences.")
    rng = np.random.default_rng(int(seed))
    point = float(np.mean(values))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, values.size, size=values.size)
        draws[index] = float(np.mean(values[selected]))
    return {
        "estimate": point,
        "interval_95": [float(value) for value in np.quantile(draws, (0.025, 0.975))],
        "lcb95": float(np.quantile(draws, 0.05)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "seed",
    }


def _claim_paths(value: str | Path) -> tuple[Path, Path]:
    output = Path(value).resolve()
    if output.suffix.lower() == ".json":
        return output, output.with_suffix(".md")
    output.mkdir(parents=True, exist_ok=True)
    return output / "claim.json", output / "claim.md"


def _markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    contrasts = report["contrasts"]
    lines = [
        "# CETR-ZSC claim report",
        "",
        f"Decision: **{decision['status']}**",
        "",
        decision["rule"],
        "",
        "## Estimates",
        "",
        "| Quantity | CETR | FCP | Difference | 95% interval | LCB95 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| J_ext,mean | {report['estimates']['cetr']['j_ext_mean']:.6f} | "
            f"{report['estimates']['fcp']['j_ext_mean']:.6f} | "
            f"{contrasts['j_ext_mean']['estimate']:.6f} | "
            f"{contrasts['j_ext_mean']['interval_95']} | "
            f"{contrasts['j_ext_mean']['lcb95']:.6f} |"
        ),
        (
            f"| J_ext,CVaR50 | {report['estimates']['cetr']['j_ext_cvar50']:.6f} | "
            f"{report['estimates']['fcp']['j_ext_cvar50']:.6f} | "
            f"{contrasts['j_ext_cvar50']['estimate']:.6f} | "
            f"{contrasts['j_ext_cvar50']['interval_95']} | "
            f"{contrasts['j_ext_cvar50']['lcb95']:.6f} |"
        ),
        (
            f"| J_SP - tau_SP | {report['estimates']['cetr']['j_sp']:.6f} | "
            f"{report['estimates']['reference_tau_sp']:.6f} | "
            f"{contrasts['j_sp_minus_tau_sp']['estimate']:.6f} | "
            f"{contrasts['j_sp_minus_tau_sp']['interval_95']} | "
            f"{contrasts['j_sp_minus_tau_sp']['lcb95']:.6f} |"
        ),
        "",
        "## Decision gates",
        "",
        "- External mean LCB95 > 0: "
        + str(contrasts["j_ext_mean"]["lcb95"] > 0.0),
        "- External CVaR50 LCB95 > 0: "
        + str(contrasts["j_ext_cvar50"]["lcb95"] > 0.0),
        "- SP LCB95 >= 0: "
        + str(contrasts["j_sp_minus_tau_sp"]["lcb95"] >= 0.0),
    ]
    lines.extend(
        (
            "",
            "## Parent-lineage units",
            "",
            "| Ego run | Parent lineage | CETR | FCP | Difference |",
            "|---|---|---:|---:|---:|",
        )
    )
    lines.extend(
        "| {ego_run_id} | {parent_lineage} | {cetr_return:.6f} | {fcp_return:.6f} | {difference:.6f} |".format(**row)
        for row in report["unit_table"]
    )
    lines.extend(("", "## Per-seed SP differences", "", "| Seed | J_SP | tau_SP | Difference |", "|---:|---:|---:|---:|"))
    lines.extend(
        "| {seed_index} | {j_sp:.6f} | {tau_sp:.6f} | {difference:.6f} |".format(**row)
        for row in report["sp_unit_table"]
    )
    return "\n".join(lines) + "\n"


def build_claim(args: argparse.Namespace) -> None:
    cetr_units, cetr_summaries = _merge_evaluations(list(args.cetr_evaluation))
    baseline_units, baseline_summaries = _merge_evaluations(list(args.baseline_evaluation))
    cetr_values, cetr_egos, cetr_parents = _matrix(cetr_units)
    fcp_values, fcp_egos, fcp_parents = _matrix(baseline_units)
    if cetr_egos != fcp_egos or cetr_parents != fcp_parents:
        raise ValueError("CETR and FCP evaluations do not share the same panel cells.")

    baseline_methods = {str(summary.get("method")) for summary in baseline_summaries}
    if baseline_methods != {"fcp"}:
        raise ValueError("Baseline inputs must identify exactly the FCP evaluation.")

    diagonals = [_population_diagonal(value) for value in args.cetr_population]
    diagonal = np.concatenate(diagonals)
    if diagonal.size != 10:
        raise ValueError("CETR population SP diagonal must contain ten run indexes.")
    reference_payloads = [
        _artifact_payload(value, preferred=("reference_sp.json",))
        for value in args.reference_sp
    ]
    if len(reference_payloads) != 10:
        raise ValueError("SP claim requires ten reference artifacts, one per seed.")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for payload in reference_payloads:
        if (
            payload.get("artifact_type") != "cetr_reference_sp"
            or int(payload.get("version", -1)) != 2
        ):
            raise ValueError("Reference-SP artifact identity differs.")
        if "seed_index" not in payload:
            raise ValueError("Reference-SP artifact lacks seed_index for SP pairing.")
        seed = int(payload["seed_index"])
        if seed in by_seed:
            raise ValueError(f"Reference-SP seed index is duplicated: {seed}")
        by_seed[seed] = payload
    if set(by_seed) != set(range(10)):
        raise ValueError(
            "SP claim requires exactly one reference artifact for each seed index 0..9."
        )
    sp_differences = np.asarray(
        [float(diagonal[seed]) - float(by_seed[seed]["tau_sp"]) for seed in range(10)],
        dtype=np.float64,
    )
    tau_sp = float(np.mean([float(by_seed[seed]["tau_sp"]) for seed in range(10)]))

    contrasts = {
        "j_ext_mean": _bootstrap_external(cetr_values, fcp_values, statistic="mean"),
        "j_ext_cvar50": _bootstrap_external(cetr_values, fcp_values, statistic="cvar50"),
        "j_sp_minus_tau_sp": _bootstrap_sp(sp_differences),
    }
    sp_contrast = contrasts["j_sp_minus_tau_sp"]
    no_go_sp = float(sp_contrast["estimate"]) < 0.0 and float(
        sp_contrast["interval_95"][1]
    ) < 0.0
    no_go_external = (
        float(contrasts["j_ext_mean"]["estimate"]) <= 0.0
        and float(contrasts["j_ext_cvar50"]["estimate"]) <= 0.0
    )
    go = (
        float(contrasts["j_ext_mean"]["lcb95"]) > 0.0
        and float(contrasts["j_ext_cvar50"]["lcb95"]) > 0.0
        and float(sp_contrast["lcb95"]) >= 0.0
    )
    if go:
        status = "GO"
        rule = "All three registered lower-confidence-bound gates pass."
    elif no_go_sp or no_go_external:
        status = "NO-GO"
        rule = "At least one registered failure rule is met."
    else:
        status = "INCONCLUSIVE"
        rule = "The SP constraint is not a decisive failure, but the external gates are not jointly positive."

    unit_table = [
        {
            "ego_run_index": int(ego),
            "ego_run_id": str(ego),
            "parent_lineage": str(parent),
            "cetr_return": float(cetr_units[(ego, parent)]),
            "fcp_return": float(baseline_units[(ego, parent)]),
            "difference": float(cetr_units[(ego, parent)] - baseline_units[(ego, parent)]),
        }
        for ego in cetr_egos
        for parent in cetr_parents
    ]
    sp_unit_table = [
        {
            "seed_index": seed,
            "j_sp": float(diagonal[seed]),
            "tau_sp": float(by_seed[seed]["tau_sp"]),
            "difference": float(sp_differences[seed]),
        }
        for seed in range(10)
    ]
    report = {
        "version": 1,
        "artifact_type": "cetr_claim_report",
        "method": FORMAL_METHOD_LABEL,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "estimates": {
            "cetr": {
                "j_ext_mean": float(np.mean(cetr_values)),
                "j_ext_cvar50": _cvar50(cetr_values),
                "j_sp": float(np.mean(diagonal)),
            },
            "fcp": {
                "j_ext_mean": float(np.mean(fcp_values)),
                "j_ext_cvar50": _cvar50(fcp_values),
            },
            "reference_tau_sp": tau_sp,
            "reference_tau_sp_by_seed": [
                float(by_seed[seed]["tau_sp"]) for seed in range(10)
            ],
        },
        "contrasts": contrasts,
        "unit_table": unit_table,
        "sp_unit_table": sp_unit_table,
        "decision": {"status": status, "rule": rule},
        "panel": {
            "ego_run_indexes": list(cetr_egos),
            "ego_run_ids": [str(value) for value in cetr_egos],
            "parent_lineages": list(cetr_parents),
            "cetr_evaluations": list(args.cetr_evaluation),
            "baseline_evaluations": list(args.baseline_evaluation),
            "cetr_population": list(args.cetr_population),
            "reference_sp": list(args.reference_sp),
        },
        "sources": {
            "cetr_summaries": cetr_summaries,
            "baseline_summaries": baseline_summaries,
            "reference_sp": reference_payloads,
        },
    }
    json_path, markdown_path = _claim_paths(args.output)
    write_json(json_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")


__all__ = ["build_claim"]
