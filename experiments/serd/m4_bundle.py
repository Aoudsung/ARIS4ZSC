"""Writers for SERD M4 output bundles."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .fixture_env import CONTROL_FAMILIES
from .serd_core import (
    BalanceRow,
    BranchRecord,
    FamilySerd,
    WorstSerd,
    require_family_coverage,
    summarize_family_serd,
    summarize_pre_h_balance,
    summarize_worst_serd,
)


BRANCH_COLUMNS = [
    "probe_id",
    "policy",
    "domain",
    "disruption",
    "family",
    "no_shock_return",
    "branch_return",
    "shock_magnitude",
    "phi_pre_h",
]

FAMILY_COLUMNS = [
    "policy",
    "domain",
    "disruption",
    "family",
    "mean_serd",
    "ci95_low",
    "ci95_high",
    "n",
    "classification",
]

WORST_COLUMNS = [
    "policy",
    "domain",
    "disruption",
    "mean_serd_worst",
    "ci95_low",
    "ci95_high",
    "n",
    "classification",
    "limiting_family",
]

BALANCE_COLUMNS = [
    "policy",
    "domain",
    "disruption",
    "family",
    "covariate",
    "semantic_mean",
    "control_mean",
    "smd",
    "n",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def branch_rows(records: Iterable[BranchRecord]) -> list[dict[str, Any]]:
    rows = [asdict(record) for record in records]
    for row in rows:
        row["phi_pre_h"] = json.dumps(row["phi_pre_h"], sort_keys=True)
    return rows


def rows_from_dataclasses(items: Iterable[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]


def m4_status(
    family_rows: list[FamilySerd],
    balance_rows: list[BalanceRow],
    required_families: tuple[str, ...] = CONTROL_FAMILIES,
) -> str:
    missing = require_family_coverage(family_rows, required_families)
    if missing:
        return "M4_BLOCKED_FAMILY_COVERAGE"
    if not balance_rows:
        return "M4_BLOCKED_BALANCE_MISSING"
    return "M4_CLAIM_BEARING_SERD_READY_FOR_REVIEW"


def write_m4_bundle(
    *,
    output_dir: Path,
    semantic_records: list[BranchRecord],
    control_records: list[BranchRecord],
    provenance: dict[str, Any],
    run_config: dict[str, Any],
    acceptance_notes: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    epsilon_shock = float(run_config["epsilon_shock"])
    epsilon_phi = float(run_config["epsilon_phi"])
    delta_serd = float(run_config["delta_serd"])
    required_families = tuple(run_config.get("control_families", CONTROL_FAMILIES))

    family = summarize_family_serd(
        semantic_records,
        control_records,
        epsilon_shock=epsilon_shock,
        epsilon_phi=epsilon_phi,
        delta_serd=delta_serd,
    )
    worst = summarize_worst_serd(family, delta_serd=delta_serd)
    balance = summarize_pre_h_balance(
        semantic_records,
        control_records,
        epsilon_shock=epsilon_shock,
        epsilon_phi=epsilon_phi,
    )
    status = m4_status(family, balance, required_families=required_families)
    missing_families = require_family_coverage(family, required_families)

    all_branch_records = list(semantic_records) + list(control_records)
    branch_dicts = branch_rows(all_branch_records)
    family_dicts = rows_from_dataclasses(family)
    worst_dicts = rows_from_dataclasses(worst)
    balance_dicts = rows_from_dataclasses(balance)

    _write_csv(output_dir / "branch_records.csv", branch_dicts, BRANCH_COLUMNS)
    _write_csv(output_dir / "family_serd.csv", family_dicts, FAMILY_COLUMNS)
    _write_csv(output_dir / "worst_serd.csv", worst_dicts, WORST_COLUMNS)
    _write_csv(output_dir / "pre_h_balance.csv", balance_dicts, BALANCE_COLUMNS)
    _write_json(output_dir / "provenance.json", provenance)
    _write_json(output_dir / "run_config.json", run_config)

    policy_domain_pairs = sorted(
        {
            (record.policy, record.domain)
            for record in all_branch_records
        }
    )
    summary = {
        "m4_status": status,
        "claim_bearing": status == "M4_CLAIM_BEARING_SERD_READY_FOR_REVIEW",
        "policy_domain_pairs": [
            {"policy": policy, "domain": domain}
            for policy, domain in policy_domain_pairs
        ],
        "delta_serd": delta_serd,
        "n_branch_records": len(branch_dicts),
        "n_family_rows": len(family_dicts),
        "n_worst_rows": len(worst_dicts),
        "n_balance_rows": len(balance_dicts),
        "missing_families": missing_families,
    }
    _write_json(output_dir / "summary.json", summary)

    lines = [
        "# M4 SERD Bundle Acceptance",
        "",
        f"Status: {status}",
        "",
        "This file records bundle-local acceptance for the generated M4 output.",
        "Downstream workflow readiness still requires the project-level M4",
        "acceptance packet, run matrix, statistical decision plan, and",
        "power/seed escalation plan to be applied to this bundle.",
        "",
        "## Counts",
        "",
        f"- Branch records: {len(branch_dicts)}",
        f"- Family SERD rows: {len(family_dicts)}",
        f"- Worst SERD rows: {len(worst_dicts)}",
        f"- Balance rows: {len(balance_dicts)}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in acceptance_notes)
    lines.append("")
    (output_dir / "acceptance.md").write_text("\n".join(lines), encoding="utf-8")

    return summary
