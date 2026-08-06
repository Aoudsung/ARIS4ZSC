from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments.overcooked_v2.delta_zsc import _parser
from src.delta_zsc.config import FORMAL_METHOD_LABEL, OFFICIAL_BASELINE_METHODS
from src.delta_zsc.storage import read_json, write_json


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
            "raw": {"path": str(raw)},
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


def test_development_matrix_launches_every_cell_in_a_fresh_process(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.overcooked_v2.development_matrix_app as matrix_app

    commands: list[list[str]] = []

    def fake_config(source, target, *, variant, component_count):
        del source, variant, component_count
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture: true\n", encoding="utf-8")

    def fake_subprocess(command, *, check):
        assert check is True
        commands.append(list(command))

    monkeypatch.setattr(matrix_app, "_write_variant_config", fake_config)
    monkeypatch.setattr(
        matrix_app,
        "_initializer_for_component_count",
        lambda source, component_count: Path(source) / f"k-{component_count}",
    )
    monkeypatch.setattr(matrix_app.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(
        matrix_app,
        "read_json",
        lambda unused: {"deployable_parameters": 1},
    )
    monkeypatch.setattr(matrix_app, "_validate_training_entries", lambda rows: None)
    monkeypatch.setattr(
        matrix_app,
        "_initializer_for_component_count",
        lambda root, count: Path(root) / f"k-{count}",
    )
    monkeypatch.setattr(matrix_app, "ensure_run_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(matrix_app, "write_json", lambda *args, **kwargs: None)
    matrix_app.run_development_matrix(
        SimpleNamespace(
            config=str(tmp_path / "source.yaml"),
            partner_manifest=str(tmp_path / "manifest.json"),
            output=str(tmp_path / "matrix"),
            seed_index=list(range(5)),
            resume=True,
            require_cuda=True,
            skip_manifest_file_check=True,
            semantic_initializer=str(tmp_path / "initializers"),
        )
    )
    assert len(commands) == 55
    assert all(command[1:4] == ["-m", "experiments.overcooked_v2.delta_zsc", "train"] for command in commands)
    assert all("--resume" in command for command in commands)
    assert all("--require-cuda" in command for command in commands)
    assert all("--semantic-initializer" in command for command in commands)
