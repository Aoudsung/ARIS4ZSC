from __future__ import annotations

import json
from pathlib import Path

from experiments.overcooked_v2.delta_zsc import _parser
from src.delta_zsc.config import FORMAL_METHOD_LABEL, OFFICIAL_BASELINE_METHODS
from src.delta_zsc.storage import read_json, sha256_path, write_json


def _evaluation_fixture(root: Path, method: str, value: float) -> Path:
    directory = root / method
    directory.mkdir(parents=True)
    raw = directory / "episode_returns.jsonl"
    rows = []
    for ego in range(2):
        for partner in range(2):
            rows.append(
                {
                    "layout": "test_time_simple",
                    "method": method,
                    "ego_run_index": ego,
                    "ego_run_id": f"{method}-run-{ego}",
                    "partner_run_index": partner,
                    "partner_run_id": f"partner-{partner}",
                    "partner_mechanism": "fixture",
                    "ego_role": 0,
                    "episode_index": 0,
                    "raw_return": float(value + ego + partner),
                }
            )
    raw.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_json(
        directory / "evaluation_summary.json",
        {
            "version": 1,
            "artifact_type": "delta_raw_evaluation",
            "layout": "test_time_simple",
            "method": method,
            "mean_return": float(value),
            "episode_count": len(rows),
            "raw": {"path": str(raw), "sha256": sha256_path(raw)},
            "resource_ledger": {},
        },
    )
    return directory


def test_summarize_evaluations_cli_wires_seed_and_writes_h1_summary(
    tmp_path: Path,
) -> None:
    methods = (FORMAL_METHOD_LABEL, *OFFICIAL_BASELINE_METHODS)
    directories = {
        method: _evaluation_fixture(
            tmp_path / "evaluations",
            method,
            40.0 if method == FORMAL_METHOD_LABEL else 0.0,
        )
        for method in methods
    }
    output = tmp_path / "official_summary.json"
    argv = ["summarize-evaluations"]
    for method in methods:
        argv.extend(("--evaluation", f"{method}={directories[method]}"))
    argv.extend(
        (
            "--bootstrap-replicates",
            "99",
            "--seed",
            "17",
            "--output",
            str(output),
        )
    )
    args = _parser().parse_args(argv)
    assert args.seed == 17
    args.function(args)

    summary = read_json(output)
    assert summary["bootstrap_seed"] == 17
    assert set(summary["delta_active_vs_each_baseline"]) == set(
        OFFICIAL_BASELINE_METHODS
    )


def test_summarize_evaluations_cli_seed_defaults_to_zero() -> None:
    args = _parser().parse_args(
        [
            "summarize-evaluations",
            "--evaluation",
            "delta-active=/tmp/evaluation",
            "--output",
            "/tmp/summary.json",
        ]
    )
    assert args.seed == 0
