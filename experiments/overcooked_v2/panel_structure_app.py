"""L2 structure analysis: within-type ranking consistency (E0).

Uses the merged S1 panel rows to ask whether mode-run preference rankings
agree across partner runs of the same type.  A positive within-type Kendall
tau would indicate convention-level structure that is merely weak; tau near
zero supports the "no transferable structure" reading of the failed transfer
analysis.  Identity cells are excluded everywhere.  Exploration track,
``scientific_readout_allowed: false``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Tau-b with tie handling, dependency-free."""
    n = len(a)
    if n < 2:
        return float("nan")
    concordant = discordant = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0.0:
                tie_a += 1
                if db == 0.0:
                    tie_b += 1
                continue
            if db == 0.0:
                tie_b += 1
                continue
            if da * db > 0:
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt((concordant + discordant + tie_a) * (concordant + discordant + tie_b))
    if denom == 0.0:
        return float("nan")
    return float((concordant - discordant) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 ranking-structure analysis.")
    parser.add_argument("--rows", required=True, help="Merged rows.json path.")
    parser.add_argument("--output", required=True, help="Summary json path.")
    args = parser.parse_args()

    rows = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    cells: dict[tuple[str, str], list[float]] = {}
    partner_type: dict[str, str] = {}
    for row in rows:
        if "identity_match" in row["flags"]:
            continue
        cells.setdefault(
            (row["mode_run_id"], row["partner_run_id"]), []
        ).append(row["raw_return"])
        partner_type[row["partner_run_id"]] = row["partner_type"]
    if not cells:
        raise RuntimeError("no run-disjoint cells found")

    partner_runs = sorted(partner_type)
    mode_runs = sorted({key[0] for key in cells})

    rankings: dict[str, list[float]] = {}
    for partner in partner_runs:
        values = []
        for mode in mode_runs:
            cell = cells.get((mode, partner))
            values.append(float(np.mean(cell)) if cell else float("nan"))
        rankings[partner] = values

    def pairwise_tau(partners: list[str]) -> list[float]:
        taus = []
        for left, right in itertools.combinations(partners, 2):
            pairs = [
                (lv, rv)
                for lv, rv in zip(rankings[left], rankings[right])
                if not (np.isnan(lv) or np.isnan(rv))
            ]
            if len(pairs) < len(mode_runs):
                raise RuntimeError("incomplete cell coverage in ranking comparison")
            taus.append(
                kendall_tau([p[0] for p in pairs], [p[1] for p in pairs])
            )
        return taus

    types = sorted(set(partner_type.values()))
    within_type: dict[str, list[float]] = {
        z: pairwise_tau([p for p in partner_runs if partner_type[p] == z])
        for z in types
    }
    cross_type = pairwise_tau(partner_runs)
    within_flat = [tau for values in within_type.values() for tau in values]
    cross_only = [
        tau
        for left, right in itertools.combinations(partner_runs, 2)
        if partner_type[left] != partner_type[right]
        for tau in [
            kendall_tau(
                [
                    lv
                    for lv, rv in zip(rankings[left], rankings[right])
                    if not (np.isnan(lv) or np.isnan(rv))
                ],
                [
                    rv
                    for lv, rv in zip(rankings[left], rankings[right])
                    if not (np.isnan(lv) or np.isnan(rv))
                ],
            )
        ]
    ]

    summary = {
        "mode_runs": mode_runs,
        "within_type_tau_by_type": {
            z: {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "pairs": len(values),
            }
            for z, values in within_type.items()
        },
        "within_type_tau": {
            "mean": float(np.mean(within_flat)),
            "std": float(np.std(within_flat)),
            "min": float(np.min(within_flat)),
            "max": float(np.max(within_flat)),
            "n_pairs": len(within_flat),
        },
        "cross_type_tau": {
            "mean": float(np.mean(cross_only)),
            "std": float(np.std(cross_only)),
            "n_pairs": len(cross_only),
        },
        "reading_rule": (
            "within-type tau near zero supports no transferable convention "
            "structure; clearly positive tau would support weak-but-present "
            "structure"
        ),
        "scientific_readout_allowed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
