from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
                "execution_mode": "active",
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
                        "execution_mode": "active",
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
            "version": 3,
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
            "execution_mode": "active",
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
    assert summary["execution_mode"] == "active"
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


def test_common_partner_summary_rejects_raw_execution_mode_mismatch(
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
    raw = directories[FORMAL_METHOD_LABEL] / "episode_returns.jsonl"
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    rows[0]["execution_mode"] = "passive"
    raw.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        evaluation=[f"{method}={directories[method]}" for method in methods],
        bootstrap_replicates=99,
        seed=0,
        output=str(tmp_path / "official_summary.json"),
    )
    import pytest

    with pytest.raises(ValueError, match="raw matrix"):
        evaluation_app.summarize_evaluations(args)


def test_official_checkpoint_evaluation_rejects_diagnostic_execution_mode(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    import experiments.overcooked_v2.evaluation_app as evaluation_app

    monkeypatch.setattr(
        evaluation_app,
        "load_partner_manifest",
        lambda *args, **kwargs: SimpleNamespace(
            by_role=lambda unused_role: [SimpleNamespace(generation_mechanism="fixture")]
        ),
    )

    with pytest.raises(ValueError, match="Official checkpoints"):
        evaluation_app._run_common_partner(
            SimpleNamespace(
                partner_manifest=str(tmp_path / "unused.json"),
                skip_manifest_file_check=False,
                partner_role="confirmatory",
                execution_mode="passive",
            ),
            config=SimpleNamespace(
                run_kind="mechanical",
                environment=SimpleNamespace(layout="test_time_simple"),
            ),
            started=0.0,
            policy_manifest={"policy_kind": "official_checkpoint", "runs": []},
            policy_manifest_path=tmp_path / "unused-policy.json",
        )


def _formal_claim_fixture(
    root: Path, *, execution_mode: str = "active"
) -> SimpleNamespace:
    from src.delta_zsc.config import LAYOUTS, METHOD_VERSION

    official_args = []
    development_args = []
    intervention_args = []
    diagnostic_args = []
    for layout in LAYOUTS:
        official = root / f"{layout}-official.json"
        development = root / f"{layout}-development.json"
        intervention = root / f"{layout}-intervention.json"
        diagnostic = root / f"{layout}-diagnostic.json"
        write_json(
            official,
            {
                "version": 3,
                "artifact_type": "delta_official_summary",
                "method": METHOD_VERSION,
                "execution_mode": execution_mode,
                "layout": layout,
                "strongest_baseline": "sp",
                "delta_active_vs_strongest": {},
                "delta_active_vs_each_baseline": {},
                "all_baselines_material_superiority_gate": False,
            },
        )
        write_json(
            development,
            {
                "version": 2,
                "artifact_type": "delta_development_summary",
                "method": METHOD_VERSION,
                "layout": layout,
                "primary_contrasts": {
                    "decision_emission": {"interval_95": [-1.0, 1.0]},
                    "active_voi": {},
                },
            },
        )
        write_json(
            intervention,
            {
                "version": 2,
                "artifact_type": "delta_belief_value_intervention",
                "method": METHOD_VERSION,
                "execution_mode": "active",
                "layout": layout,
                "one_sided_lcb": -1.0,
            },
        )
        write_json(
            diagnostic,
            {
                "version": 6,
                "artifact_type": "delta_v6_posterior_predictive_diagnostics",
                "method": METHOD_VERSION,
                "layout": layout,
                "claim_role": "diagnostic_only",
            },
        )
        official_args.append(f"{layout}={official}")
        development_args.append(f"{layout}={development}")
        intervention_args.append(f"{layout}={intervention}")
        diagnostic_args.append(f"{layout}={diagnostic}")
    resource = root / "resource.json"
    write_json(
        resource,
        {"version": 1, "artifact_type": "delta_resource_report", "methods": [{}]},
    )
    return SimpleNamespace(
        official_summary=official_args,
        development_summary=development_args,
        belief_intervention=intervention_args,
        posterior_diagnostics=diagnostic_args,
        resource_report=str(resource),
        output=str(root / "claim.json"),
    )


def test_formal_claim_requires_active_v6_evidence(tmp_path: Path) -> None:
    from experiments.overcooked_v2.formal_claim_app import build_formal_claim_report

    with pytest.raises(ValueError, match="Official summary identity"):
        build_formal_claim_report(
            _formal_claim_fixture(tmp_path, execution_mode="reference_only")
        )


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
    assert len(commands) == 45
    assert all(command[1:4] == ["-m", "experiments.overcooked_v2.delta_zsc", "train"] for command in commands)
    assert all("--resume" in command for command in commands)
    assert all("--require-cuda" in command for command in commands)
    assert all("--semantic-initializer" in command for command in commands)
    assert all("--sp-initializer" in command for command in commands)
    for command in commands:
        seed = int(command[command.index("--seed-index") + 1])
        checkpoint = command[command.index("--sp-initializer") + 1]
        assert checkpoint.endswith(f"run-{seed}/ckpt_final")


def test_state_augmented_population_records_effective_environment_count(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.overcooked_v2.upstream_pipeline_app as upstream_app

    identities = []
    commands: list[list[str]] = []

    def fake_identity(unused_directory, identity):
        identities.append(identity)

    def fake_run_logged(command, *, output, name):
        del output, name
        commands.append(list(command))
        hydra_argument = next(
            item for item in command if item.startswith("hydra.run.dir=")
        )
        hydra_root = Path(hydra_argument.split("=", 1)[1])
        run_root = hydra_root / "runs" / "fixture"
        for index in range(10):
            (run_root / f"run_{index}" / "ckpt_final").mkdir(
                parents=True, exist_ok=True
            )
        return 1.0

    monkeypatch.setattr(upstream_app, "ensure_run_identity", fake_identity)
    monkeypatch.setattr(upstream_app, "_run_logged", fake_run_logged)
    monkeypatch.setattr(
        upstream_app,
        "_population_keys",
        lambda unused_seed, count: [[0, index] for index in range(count)],
    )
    monkeypatch.setattr(
        upstream_app,
        "restore_official_checkpoint",
        lambda unused: ({}, {"fixture": np.zeros((1,), dtype=np.float32)}),
    )
    monkeypatch.setattr(upstream_app, "parameter_count", lambda unused: 1)

    output = tmp_path / "state-augmented"
    upstream_app._run_official_population(
        method="state-augmented",
        layout="test_time_wide",
        root_seed=13_042,
        run_count=10,
        checkpoint_count=1,
        parent_prefix="wide-coverage-sa",
        output=output,
        co_training_groups=["shared"] * 10,
    )

    assert len(commands) == 1
    assert "++model.NUM_ENVS=128" in commands[0]
    assert identities[0]["state_augmented_num_envs"] == 128
    population = read_json(output / "population_training.json")
    assert population["state_augmented_num_envs"] == 128
    assert len(population["runs"]) == 10


def test_upstream_registers_four_partners_from_each_ten_run_sa_source(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.overcooked_v2.upstream_pipeline_app as upstream_app

    population_calls = []

    monkeypatch.setattr(upstream_app, "validate_official_runtime", lambda: None)
    monkeypatch.setattr(
        upstream_app,
        "load_config",
        lambda unused_path, run_kind: SimpleNamespace(
            environment=SimpleNamespace(layout="test_time_simple"),
            to_mapping=lambda: {"run_kind": run_kind},
        ),
    )

    def fake_population(**kwargs):
        population_calls.append(kwargs)
        return Path(kwargs["output"]) / "policy_manifest.json"

    def fake_fcp_population(*, purpose, count, layout, output):
        del purpose, count, layout
        root = Path(output)
        lineage = root / "training_lineage.json"
        write_json(lineage, [])
        ledger = root / "resource_ledger.json"
        write_json(ledger, {})
        return root / "population", lineage, ledger

    def fake_partner_rows(
        population, *, role, run_indexes, all_checkpoints
    ):
        del all_checkpoints
        return [
            {
                "population": str(population),
                "role": role,
                "run_index": int(index),
            }
            for index in run_indexes
        ]

    monkeypatch.setattr(upstream_app, "_run_official_population", fake_population)
    monkeypatch.setattr(upstream_app, "_build_fcp_population", fake_fcp_population)
    monkeypatch.setattr(upstream_app, "_baseline_command", lambda **unused: None)
    monkeypatch.setattr(upstream_app, "_partner_rows", fake_partner_rows)
    monkeypatch.setattr(upstream_app, "build_partner_manifest", lambda unused: None)

    output = tmp_path / "upstream"
    upstream_app.run_upstream(
        SimpleNamespace(config=str(tmp_path / "config.yaml"), output=str(output))
    )

    sa_calls = [
        call for call in population_calls if call["method"] == "state-augmented"
    ]
    assert len(sa_calls) == 2
    assert all(call["run_count"] == 10 for call in sa_calls)
    assert all(len(call["co_training_groups"]) == 10 for call in sa_calls)

    plan = read_json(output / "plans" / "partner_plan.json")
    sa_rows = [
        row for row in plan["runs"] if "state-augmented" in row["population"]
    ]
    assert len(sa_rows) == 8
    assert {
        (row["role"], row["run_index"]) for row in sa_rows
    } == {
        (role, index)
        for role in ("development_coverage", "confirmatory")
        for index in range(4)
    }
