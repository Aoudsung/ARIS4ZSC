from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.overcooked_v2.delta_zsc import _parser
from src.delta_zsc.config import (
    FORMAL_METHOD_LABEL,
    MANIFEST_VERSION,
    OFFICIAL_BASELINE_METHODS,
)
from src.delta_zsc.storage import read_json, write_json


def test_population_statistics_use_diagonal_and_ordered_off_diagonal_rows() -> None:
    from experiments.overcooked_v2.evaluation_app import _population_statistics

    cell_means = np.asarray(
        [
            [10.0, 1.0, 2.0],
            [3.0, 20.0, 4.0],
            [5.0, 6.0, 30.0],
        ]
    )
    cube = np.repeat(cell_means[:, :, None], 2, axis=2)
    observed = _population_statistics(cube)

    np.testing.assert_allclose(observed["sp_diagonal"], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(observed["xp_rows"], [1.5, 3.5, 5.5])
    np.testing.assert_allclose(observed["gap_rows"], [8.5, 16.5, 24.5])
    assert observed["sp_point"] == 20.0
    assert observed["xp_point"] == 3.5
    assert observed["gap_point"] == 16.5


def test_delta_only_population_summary_uses_layout_paper_values(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    cube = np.full((10, 10, 500), 2.0, dtype=np.float64)
    for index in range(10):
        cube[index, index] = 10.0

    monkeypatch.setattr(
        evaluation_app,
        "_validated_population_cube",
        lambda unused_directory, method: (
            {
                "layout": "grounded_coord_ring",
                "root_seed": 42,
                "method": method,
            },
            cube,
            {(0, 0, 0): (1, 2)},
        ),
    )
    output = tmp_path / "paper-comparison"
    evaluation_app.summarize_population_matrices(
        SimpleNamespace(
            evaluation=[f"delta-active={tmp_path / 'delta-evaluation'}"],
            output=str(output),
        )
    )

    summary = read_json(output / "population_matrix_summary.json")
    rows = {row["method"]: row for row in summary["methods"]}
    assert summary["comparison_mode"] == "paper_reference"
    assert summary["evaluated_methods"] == ["delta-active"]
    assert rows["fcp"]["paper_xp_verbatim"] == "6±46"
    assert rows["fcp"]["reproduced_xp_point"] is None
    assert rows["delta-active"]["reproduced_sp_point"] == 10.0
    assert rows["delta-active"]["reproduced_xp_point"] == 2.0
    assert rows["delta-active"]["reproduced_gap_point"] == 8.0


def _partner_row(checkpoint: Path, index: int) -> dict:
    return {
        "run_id": f"partner-{index}",
        "role": "confirmatory",
        "checkpoint": str(checkpoint),
        "parent_training_run_id": f"confirmatory-parent-{index}",
        "generation_mechanism": "fixture",
        "checkpoint_stage": 1.0,
        "hyperparameter_family": "fixture",
        "seed": index,
        "seed_index": index,
        "jax_prng_key": [0, index],
        "owner_seed_index": None,
        "co_training_group_id": None,
        "partner_type_id": None,
    }


def _evaluation_fixture(
    root: Path,
    method: str,
    value: float,
    *,
    partner_manifest: Path,
) -> Path:
    directory = root / method
    directory.mkdir(parents=True)
    raw = directory / "episode_returns.jsonl"
    rows = []
    for ego in range(2):
        for partner in range(2):
            for role in range(2):
                rows.append(
                    {
                        "evaluation_mode": "common_partner",
                        "layout": "test_time_simple",
                        "method": method,
                        "ego_run_index": ego,
                        "ego_run_id": f"{method}-run-{ego}",
                        "partner_run_index": partner,
                        "partner_run_id": f"partner-{partner}",
                        "partner_mechanism": "fixture",
                        "ego_role": role,
                        "episode_index": 0,
                        "environment_key": [ego + partner, role],
                        "raw_return": float(value + ego + partner + role),
                    }
                )
    raw.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    policy_manifest = directory / "policy_manifest.json"
    write_json(
        policy_manifest,
        {
            "version": 2,
            "method": method,
            "layout": "test_time_simple",
            "policy_kind": "official_checkpoint",
            "runs": [
                {
                    "run_index": index,
                    "run_id": f"{method}-run-{index}",
                    "policy": str(directory / f"policy-{index}"),
                    "identity": {
                        "parent_training_run_id": f"{method}-parent-{index}",
                        "co_training_group_id": None,
                    },
                }
                for index in range(2)
            ],
            "training_lineage": [],
        },
    )
    write_json(
        directory / "evaluation_summary.json",
        {
            "version": 2,
            "artifact_type": "delta_raw_evaluation",
            "evaluation_mode": "common_partner",
            "layout": "test_time_simple",
            "method": method,
            "mean_return": float(value),
            "episode_count": len(rows),
            "root_seed": 0,
            "key_schedule": (
                "fold_in_root_by_ego_partner_then_role_then_split_episode_keys"
            ),
            "observation_protocol": "default_non_permuted",
            "policy_manifest": {"path": str(policy_manifest)},
            "partner_manifest": {"path": str(partner_manifest)},
            "raw": {"path": str(raw)},
            "resource_ledger": {},
        },
    )
    return directory


def test_summarize_evaluations_cli_wires_seed_and_writes_h1_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.overcooked_v2.evaluation_app as evaluation_app

    monkeypatch.setattr(evaluation_app, "FORMAL_COMMON_EGO_RUNS", 2)
    monkeypatch.setattr(evaluation_app, "FORMAL_COMMON_PARTNER_RUNS", 2)
    monkeypatch.setattr(evaluation_app, "FORMAL_EVALUATION_EPISODES", 1)
    partner_manifest = tmp_path / "partner_manifest.json"
    write_json(
        partner_manifest,
        {
            "version": MANIFEST_VERSION,
            "layout": "test_time_simple",
            "runs": [
                _partner_row(tmp_path / f"partner-{index}", index)
                for index in range(2)
            ],
        },
    )
    methods = (FORMAL_METHOD_LABEL, *OFFICIAL_BASELINE_METHODS)
    directories = {
        method: _evaluation_fixture(
            tmp_path / "evaluations",
            method,
            40.0 if method == FORMAL_METHOD_LABEL else 0.0,
            partner_manifest=partner_manifest,
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
            sp_initializer_root=str(tmp_path / "sp-initializers"),
        )
    )
    assert len(commands) == 55
    assert all(command[1:4] == ["-m", "experiments.overcooked_v2.delta_zsc", "train"] for command in commands)
    assert all("--resume" in command for command in commands)
    assert all("--require-cuda" in command for command in commands)
    assert all("--semantic-initializer" in command for command in commands)
    assert all("--sp-initializer" in command for command in commands)
    for command in commands:
        seed = int(command[command.index("--seed-index") + 1])
        checkpoint = command[command.index("--sp-initializer") + 1]
        assert checkpoint.endswith(f"run-{seed}/ckpt_final")
