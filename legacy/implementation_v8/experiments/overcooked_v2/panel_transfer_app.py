"""S1 transfer-value analysis (exploration track).

Registered design (docs/status/S1_EXECUTION_SPEC.md): partner runs are split
into development seeds 0-4 and holdout seeds 5-9.  For each partner type the
type-to-mode-run mapping is frozen on development partner runs only, identity
cells excluded everywhere, and the frozen mapping is then evaluated on holdout
partner runs.  Intervals use partner-run-level bootstrap, the registered
statistical unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEV_SEEDS = (0, 1, 2, 3, 4)
HOLDOUT_SEEDS = (5, 6, 7, 8, 9)
BOOTSTRAP_REPLICATES = 9_999
BOOTSTRAP_SEED = 20260803


def seed_of(run_id: str) -> int:
    return int(run_id.rsplit("-", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 transfer-value analysis.")
    parser.add_argument("--rows", required=True, help="Merged rows.json path.")
    parser.add_argument("--output", required=True, help="Summary json path.")
    args = parser.parse_args()

    rows = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    cells: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if "identity_match" in row["flags"]:
            continue
        cells.setdefault(
            (row["mode_run_id"], row["partner_run_id"]), []
        ).append(row["raw_return"])
    if not cells:
        raise RuntimeError("no run-disjoint cells found")
    means = {key: float(np.mean(values)) for key, values in cells.items()}

    partner_type = {row["partner_run_id"]: row["partner_type"] for row in rows}
    partner_runs = sorted(partner_type)
    mode_runs = sorted({row["mode_run_id"] for row in rows})
    dev_partners = [p for p in partner_runs if seed_of(p) in DEV_SEEDS]
    holdout_partners = [p for p in partner_runs if seed_of(p) in HOLDOUT_SEEDS]
    if not dev_partners or not holdout_partners:
        raise RuntimeError("development/holdout split is empty")
    types = sorted(set(partner_type.values()))

    mapping: dict[str, str] = {}
    mapping_scores: dict[str, float] = {}
    for z in types:
        dev_z = [p for p in dev_partners if partner_type[p] == z]
        best_run, best_score = "", -np.inf
        for mode in mode_runs:
            values = [means[(mode, p)] for p in dev_z if (mode, p) in means]
            if not values:
                continue
            score = float(np.mean(values))
            if score > best_score:
                best_run, best_score = mode, score
        if not best_run:
            raise RuntimeError(f"no valid mode candidate for type {z}")
        mapping[z] = best_run
        mapping_scores[z] = best_score

    def value_under(mode_for, partners):
        values = [
            means[(mode_for(p), p)] for p in partners if (mode_for(p), p) in means
        ]
        return values

    transfer_values = value_under(
        lambda p: mapping[partner_type[p]], holdout_partners
    )
    v_transfer = float(np.mean(transfer_values))

    fix_values_by_mode = {}
    for mode in mode_runs:
        values = [means[(mode, p)] for p in holdout_partners if (mode, p) in means]
        if values:
            fix_values_by_mode[mode] = values
    fix_mode = max(fix_values_by_mode, key=lambda m: float(np.mean(fix_values_by_mode[m])))
    v_fix_test = float(np.mean(fix_values_by_mode[fix_mode]))

    oracle_values = [
        max(means[(mode, p)] for mode in mode_runs if (mode, p) in means)
        for p in holdout_partners
    ]
    v_oracle_holdout = float(np.mean(oracle_values))

    common = [
        p
        for p in holdout_partners
        if (mapping[partner_type[p]], p) in means and (fix_mode, p) in means
    ]
    transfer_by_partner = {p: means[(mapping[partner_type[p]], p)] for p in common}
    fix_by_partner = {p: means[(fix_mode, p)] for p in common}
    arr_transfer = np.array([transfer_by_partner[p] for p in common])
    arr_fix = np.array([fix_by_partner[p] for p in common])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gaps = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(common), len(common))
        gaps[index] = arr_transfer[sample].mean() - arr_fix[sample].mean()
    lower, upper = np.quantile(gaps, [0.025, 0.975])

    summary = {
        "dev_seeds": list(DEV_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "frozen_mapping": mapping,
        "frozen_mapping_dev_scores": mapping_scores,
        "fix_mode_holdout": fix_mode,
        "v_transfer": v_transfer,
        "v_fix_test": v_fix_test,
        "transfer_gap": v_transfer - v_fix_test,
        "transfer_gap_ci95": [float(lower), float(upper)],
        "v_oracle_holdout_identity_free": v_oracle_holdout,
        "bootstrap_units": len(common),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "note": "identity cells excluded everywhere; bootstrap unit is the holdout partner run",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
