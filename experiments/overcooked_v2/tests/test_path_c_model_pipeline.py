from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.path_c.contracts.config import (
    FAMILY_POOL_SCHEMA_VERSION,
    PathCFormalTemplateV3,
    PathCModelConfig,
)
from src.path_c.pipeline.run import (
    FAMILY_POOL_STAGES,
    LEGACY_STAGES,
    STAGES,
    run_pipeline,
)
from src.path_c.evaluation.standard import canonical_sha256
from experiments.overcooked_v2.scripts.evaluate_path_c_model import (
    _dependency_roots as _evaluation_dependency_roots,
    _pairing_input,
)
from experiments.overcooked_v2.scripts.run_path_c_model import (
    _pipeline_dependency_roots,
)
from experiments.overcooked_v2.scripts.run_path_c_formal_standard import (
    CONDITIONS as FORMAL_CONDITIONS,
    JAX_MEMORY_FRACTION,
    POPULATIONS as FORMAL_POPULATIONS,
    _archive_stale_formal_contract,
    _archive_stale_mechanical_smoke,
    _cuda_environment,
    _formal_paths,
    _launch_payload,
    _parse_gpus,
    _shared_stage_receipts,
    _template_paths,
    expected_official_task_identities,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "path_c_model_development_decision_focused.yaml"
)
V3_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "path_c_model_v3_formal_decision_focused.yaml"
)


def _config(
    tmp_path: Path,
    *,
    condition: str = "decision_focused",
    gamma: float | None = None,
) -> PathCModelConfig:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    controllers = {
        "decision_focused": "registered_response_sequential_branch_v1",
        "random_safe_probe": "registered_random_safe_probe_v1",
        "no_probe": "off",
        "generic_response_information": "generic_response_information",
    }
    payload["condition_id"] = condition
    payload["controller"] = controllers[condition]
    payload["output_root"] = str(tmp_path / "output")
    if gamma is not None:
        payload["stages"]["adaptation"]["gamma"] = gamma
    return PathCModelConfig.from_mapping(payload, base_dir=CONFIG_PATH.parent)


def _family_config(
    tmp_path: Path, *, condition: str = "decision_focused"
) -> PathCModelConfig:
    from experiments.overcooked_v2.tests.test_path_c_model_contracts import (
        _family_outer_units_manifest,
    )

    manifest = _family_outer_units_manifest(tmp_path / "outer_manifest")
    payload = yaml.safe_load(V3_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["condition_id"] = condition
    payload["controller"] = {
        "decision_focused": "registered_response_sequential_branch_v1",
        "no_probe": "off",
    }[condition]
    payload["output_root"] = str(tmp_path / "output")
    template = PathCFormalTemplateV3.from_mapping(
        payload, base_dir=V3_CONFIG_PATH.parent
    )
    config = template.resolve(manifest, outer_unit_id=0)
    assert config.schema_version == FAMILY_POOL_SCHEMA_VERSION
    return config


def _runners(calls: dict[str, int]):
    def runner(context):
        calls[context.stage] = calls.get(context.stage, 0) + 1
        target = context.stage_directory / f"{context.stage}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "stage": context.stage,
                    "input": context.stage_input_sha256,
                    "call": calls[context.stage],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {"primary": target}

    return {stage: runner for stage in STAGES}


def test_stage_receipts_and_resume_require_unchanged_inputs_and_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    calls: dict[str, int] = {}
    config = _config(tmp_path)
    runners = _runners(calls)
    first = run_pipeline(
        config,
        source_roots=(source,),
        stage_runners=runners,
        through="adaptation",
    )
    assert calls == {stage: 1 for stage in LEGACY_STAGES}
    for stage in LEGACY_STAGES:
        receipt = Path(first["stages"][stage]["receipt_path"])
        assert receipt.is_file()
        assert json.loads(receipt.read_text(encoding="utf-8"))["stage"] == stage
    run_pipeline(
        config,
        source_roots=(source,),
        stage_runners=runners,
        resume=True,
        through="adaptation",
    )
    assert calls == {stage: 1 for stage in LEGACY_STAGES}
    adaptation_artifact = Path(
        first["stages"]["adaptation"]["artifacts"]["primary"]["path"]
    )
    adaptation_artifact.write_text("tampered", encoding="utf-8")
    run_pipeline(
        config,
        source_roots=(source,),
        stage_runners=runners,
        resume=True,
        through="adaptation",
    )
    assert calls["adaptation"] == 2


def test_shared_stages_reuse_across_conditions_but_adaptation_does_not(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    calls: dict[str, int] = {}
    runners = _runners(calls)
    first = _config(tmp_path, condition="decision_focused")
    second = _config(tmp_path, condition="no_probe")
    run_pipeline(
        first,
        source_roots=(source,),
        stage_runners=runners,
        through="adaptation",
    )
    run_pipeline(
        second,
        source_roots=(source,),
        stage_runners=runners,
        through="adaptation",
    )
    assert calls["pool_check"] == 1
    assert calls["prefit"] == 1
    assert calls["calibration"] == 1
    assert calls["adaptation"] == 2


def test_family_pipeline_shares_training_calibration_but_not_deployment_calibration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    calls: dict[str, int] = {}
    runners = _runners(calls)
    first = _family_config(tmp_path, condition="decision_focused")
    second = _family_config(tmp_path, condition="no_probe")
    run_pipeline(
        first,
        source_roots=(source,),
        stage_runners=runners,
        through="deployment_calibration",
    )
    run_pipeline(
        second,
        source_roots=(source,),
        stage_runners=runners,
        through="deployment_calibration",
    )
    assert calls == {
        "pool_check": 1,
        "prefit": 1,
        "training_calibration": 1,
        "adaptation": 2,
        "deployment_calibration": 2,
    }
    source.write_text("VALUE = 2\n", encoding="utf-8")
    run_pipeline(
        first,
        source_roots=(source,),
        stage_runners=runners,
        resume=True,
        through="deployment_calibration",
    )
    assert calls == {
        "pool_check": 2,
        "prefit": 2,
        "training_calibration": 2,
        "adaptation": 3,
        "deployment_calibration": 3,
    }


def test_source_drift_changes_binding_and_reruns_every_stage(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    calls: dict[str, int] = {}
    config = _config(tmp_path)
    runners = _runners(calls)
    first = run_pipeline(
        config,
        source_roots=(source,),
        stage_runners=runners,
        resume=True,
        through="adaptation",
    )
    first_hash = first["stages"]["pool_check"]["stage_input_sha256"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = run_pipeline(
        config,
        source_roots=(source,),
        stage_runners=runners,
        resume=True,
        through="adaptation",
    )
    assert second["stages"]["pool_check"]["stage_input_sha256"] != first_hash
    assert calls == {stage: 2 for stage in LEGACY_STAGES}

    changed_config = _config(tmp_path, gamma=0.98)
    third = run_pipeline(
        changed_config,
        source_roots=(source,),
        stage_runners=runners,
        resume=True,
        through="adaptation",
    )
    assert third["stages"]["pool_check"]["stage_input_sha256"] != second[
        "stages"
    ]["pool_check"]["stage_input_sha256"]
    assert calls == {stage: 3 for stage in LEGACY_STAGES}


def test_through_stops_after_requested_stage(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    calls: dict[str, int] = {}
    state = run_pipeline(
        _config(tmp_path),
        source_roots=(source,),
        stage_runners=_runners(calls),
        through="calibration",
    )
    assert set(state["stages"]) == {"pool_check", "prefit", "calibration"}
    assert calls == {"pool_check": 1, "prefit": 1, "calibration": 1}

    calls.clear()
    state = run_pipeline(
        _config(tmp_path / "adaptation"),
        source_roots=(source,),
        stage_runners=_runners(calls),
        through="adaptation",
    )
    assert set(state["stages"]) == {
        "pool_check",
        "prefit",
        "calibration",
        "adaptation",
    }
    assert calls == {
        "pool_check": 1,
        "prefit": 1,
        "calibration": 1,
        "adaptation": 1,
    }


def test_official_runtime_dependencies_and_launch_configs_are_pipeline_inputs() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    config_root = repository_root / "experiments" / "overcooked_v2" / "configs"
    roots = _pipeline_dependency_roots(
        repository_root,
        backbone_launch=config_root / "path_c_official_sp_simple_seed100.yaml",
        partner_launches=[
            config_root / "path_c_official_sp_simple_seed101.yaml",
            config_root / "path_c_official_sp_simple_seed102.yaml",
            config_root / "path_c_official_op_simple_seed201.yaml",
            config_root / "path_c_official_op_simple_seed202.yaml",
        ],
        layout="test_time_simple",
    )
    resolved = {path.resolve() for path in roots}
    overcooked_root = repository_root / "experiments" / "overcooked_v2"
    expected_dependencies = {
        overcooked_root / "official" / "overcooked_v2_experiments_adapter.py",
        overcooked_root / "path_c_official_artifact.py",
        overcooked_root / "path_c_flax_policy.py",
        overcooked_root / "path_c_official_evidence.py",
        overcooked_root / "path_c_pool_admission.py",
        overcooked_root / "path_c_response_summary.py",
        overcooked_root / "path_c_seed.py",
        overcooked_root / "path_c_standard_training.py",
    }
    assert {path.resolve() for path in expected_dependencies} <= resolved
    assert (config_root / "path_c_official_sp_simple_seed100.yaml").resolve() in resolved

    wide_roots = _pipeline_dependency_roots(
        repository_root,
        backbone_launch=config_root / "path_c_official_sp_wide_seed100.yaml",
        partner_launches=[
            config_root / "path_c_official_sp_wide_seed100.yaml",
            config_root / "path_c_official_op_wide_seed201.yaml",
        ],
        layout="test_time_wide",
    )
    wide_resolved = {path.resolve() for path in wide_roots}
    assert (
        overcooked_root
        / "official"
        / "overcooked_v2_experiments_wide_adapter.py"
    ).resolve() in wide_resolved
    assert (
        overcooked_root
        / "scripts"
        / "run_path_c_official_wide_training.py"
    ).resolve() in wide_resolved


def test_standard_evaluator_hashes_the_audited_official_dependency_closure() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    overcooked_root = repository_root / "experiments" / "overcooked_v2"
    source = SimpleNamespace(
        launch_config_path=overcooked_root
        / "configs"
        / "path_c_official_sp_simple_seed100.yaml"
    )
    manifest = SimpleNamespace(
        path=repository_root / "population.json",
        layout="test_time_simple",
        outer_units=SimpleNamespace(
            path=repository_root / "outer_units.json",
            units=(SimpleNamespace(sources=(source,)),),
        ),
    )
    roots = {
        path.resolve()
        for path in _evaluation_dependency_roots(repository_root, manifest)
    }
    assert {
        (overcooked_root / "path_c_flax_policy.py").resolve(),
        (overcooked_root / "path_c_official_evidence.py").resolve(),
        (overcooked_root / "path_c_pool_admission.py").resolve(),
        (overcooked_root / "path_c_seed.py").resolve(),
        (overcooked_root / "path_c_standard_training.py").resolve(),
        (
            overcooked_root
            / "official"
            / "path_c_family_pool_training.py"
        ).resolve(),
    } <= roots
    manifest.layout = "test_time_wide"
    wide_roots = {
        path.resolve()
        for path in _evaluation_dependency_roots(repository_root, manifest)
    }
    assert (
        overcooked_root
        / "official"
        / "overcooked_v2_experiments_wide_adapter.py"
    ).resolve() in wide_roots


def test_each_explicit_dependency_drift_reruns_shared_pool_check(tmp_path: Path) -> None:
    dependency_names = (
        "official_adapter.py",
        "official_artifact.py",
        "path_c_flax_policy.py",
        "path_c_official_evidence.py",
        "path_c_pool_admission.py",
        "response_summary.py",
        "path_c_seed.py",
        "path_c_standard_training.py",
        "official_launch.yaml",
    )
    dependencies = tuple(tmp_path / name for name in dependency_names)
    for index, path in enumerate(dependencies):
        path.write_text(f"VERSION = {index}\n", encoding="utf-8")
    calls: dict[str, int] = {}
    config = _config(tmp_path)
    runners = _runners(calls)
    run_pipeline(
        config,
        source_roots=dependencies,
        stage_runners=runners,
        resume=True,
        through="pool_check",
    )
    for expected_calls, path in enumerate(dependencies, start=2):
        path.write_text(
            path.read_text(encoding="utf-8") + "CHANGED = 1\n",
            encoding="utf-8",
        )
        run_pipeline(
            config,
            source_roots=dependencies,
            stage_runners=runners,
            resume=True,
            through="pool_check",
        )
        assert calls["pool_check"] == expected_calls


def test_startup_guard_report_requires_all_three_explicit_passes() -> None:
    from experiments.overcooked_v2.model_dock.stage_runtime import (
        _startup_guard_report,
    )

    names = (
        "observation_layout",
        "delivery_counters",
        "official_warm_start_parity",
    )

    def report(failed: str | None = None):
        environment = {
            "observation_layout": {"passed": failed != "observation_layout"},
            "delivery_counters": {"passed": failed != "delivery_counters"},
        }
        parity = {"passed": failed != "official_warm_start_parity"}
        return _startup_guard_report(environment, parity)

    assert report()["all_passed"] is True
    for name in names:
        with pytest.raises(RuntimeError, match=name):
            report(name)
    with pytest.raises(RuntimeError, match="delivery_counters"):
        _startup_guard_report(
            {"observation_layout": {"passed": True}},
            {"passed": True},
        )


def test_pool_check_registers_partner_and_startup_guard_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.overcooked_v2.model_dock.stage_runtime import PathCStageRuntime

    runtime = PathCStageRuntime(
        _config(tmp_path),
        backbone_launch_config=tmp_path / "backbone.yaml",
        partner_launch_configs=tuple(
            tmp_path / f"partner_{index}.yaml" for index in range(4)
        ),
    )
    monkeypatch.setattr(
        "experiments.overcooked_v2.model_dock.stage_runtime.validate_checkpoint_reference",
        lambda reference, launch_config_path: {
            "checkpoint_path": str(reference.checkpoint_path),
            "launch_config_path": str(launch_config_path),
            "verified": True,
        },
    )
    monkeypatch.setattr(
        runtime,
        "_build",
        lambda: {
            "startup_guards": {
                "schema_version": "path_c_startup_guards_v1",
                "all_passed": True,
            }
        },
    )
    context = SimpleNamespace(
        config=runtime.config,
        stage_directory=tmp_path / "shared" / "pool_check",
    )
    artifacts = runtime.pool_check(context)
    assert set(artifacts) == {"pool_check", "startup_guards"}
    assert json.loads(artifacts["startup_guards"].read_text(encoding="utf-8"))[
        "all_passed"
    ] is True


def test_standard_evaluation_binding_changes_with_manifest_policy_and_source_drift() -> None:
    def policy(unit: int, checkpoint_hash: str, calibration_hash: str):
        return SimpleNamespace(
            outer_unit_id=unit,
            policy_id=f"outer_unit_{unit:02d}",
            source_type="adaptation_checkpoint",
            checkpoint_path=Path(f"/formal/unit_{unit}"),
            checkpoint_manifest_sha256=checkpoint_hash,
            model_weights_sha256=f"{unit + 10:064x}",
            calibration_summary_path=Path(f"/formal/unit_{unit}/calibration.json"),
            calibration_summary_sha256=calibration_hash,
        )

    left = policy(0, "1" * 64, "2" * 64)
    right = policy(1, "3" * 64, "4" * 64)
    pairing = SimpleNamespace(
        pairing_id="outer_unit_00__outer_unit_01",
        split="xp",
        outer_unit_0=0,
        outer_unit_1=1,
        policy_0=left,
        policy_1=right,
    )
    units = tuple(
        SimpleNamespace(seeds=SimpleNamespace(evaluation_seed=9000 + index))
        for index in range(10)
    )
    outer = SimpleNamespace(
        sha256="5" * 64,
        seed_root=2_026_072_101,
        unit=lambda index: units[index],
    )

    def bound(
        *, manifest_hash: str = "6" * 64, source_hash: str = "7" * 64,
        left_policy=left,
    ) -> str:
        manifest = SimpleNamespace(
            sha256=manifest_hash,
            outer_units=outer,
            layout="test_time_simple",
            episodes_per_pairing=500,
        )
        current_pairing = SimpleNamespace(**vars(pairing))
        current_pairing.policy_0 = left_policy
        return canonical_sha256(
            _pairing_input(
                manifest,
                current_pairing,
                ({"path": "/source.py", "sha256": source_hash},),
            )
        )

    original = bound()
    assert bound(manifest_hash="8" * 64) != original
    assert bound(source_hash="9" * 64) != original
    assert bound(left_policy=policy(0, "a" * 64, "2" * 64)) != original
    assert bound(left_policy=policy(0, "1" * 64, "b" * 64)) != original


def test_formal_orchestrator_allocates_fifty_independent_official_runs() -> None:
    simple = expected_official_task_identities("test_time_simple")
    wide = expected_official_task_identities("test_time_wide")
    for layout, tasks in (
        ("test_time_simple", simple),
        ("test_time_wide", wide),
    ):
        assert len(tasks) == 50
        assert len({row["task_id"] for row in tasks}) == 50
        assert len({row["training_seed"] for row in tasks}) == 50
        assert sum(row["experiment_variant"] == "rnn-sp" for row in tasks) == 30
        assert sum(row["experiment_variant"] == "rnn-op" for row in tasks) == 20
        assert {row["outer_unit_id"] for row in tasks} == set(range(10))
        assert {row["layout"] for row in tasks} == {layout}
    assert {row["training_seed"] for row in simple}.isdisjoint(
        {row["training_seed"] for row in wide}
    )
    assert FORMAL_CONDITIONS == (
        "decision_focused",
        "no_probe",
        "random_safe_probe",
        "generic_response_information",
    )
    assert FORMAL_POPULATIONS == ("pre_adaptation_backbone", *FORMAL_CONDITIONS)


def test_formal_orchestrator_permits_all_eight_gpus_and_reserves_calibration_memory() -> None:
    assert _parse_gpus("0,1,2,3,4,5,6,7") == tuple(range(8))
    assert _parse_gpus("5") == (5,)
    with pytest.raises(Exception, match="between 0 and 7"):
        _parse_gpus("8")
    with pytest.raises(Exception, match="distinct"):
        _parse_gpus("4,4")
    environment = _cuda_environment(5)
    assert environment["CUDA_VISIBLE_DEVICES"] == "5"
    assert environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] == JAX_MEMORY_FRACTION
    assert float(JAX_MEMORY_FRACTION) == pytest.approx(0.90)


def test_formal_orchestrator_archives_only_stale_mechanical_smoke(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    logs = control / "logs"
    artifacts = control / "mechanical_smoke_artifacts"
    logs.mkdir(parents=True)
    artifacts.mkdir()
    receipt = control / "mechanical_smoke.json"
    receipt.write_text('{"passed": true}\n', encoding="utf-8")
    (artifacts / "summary.json").write_text("{}\n", encoding="utf-8")
    (logs / "mechanical_smoke.log").write_text("old\n", encoding="utf-8")
    preserved = control / "upstream_receipt.json"
    preserved.write_text('{"complete": true}\n', encoding="utf-8")
    archive = _archive_stale_mechanical_smoke(
        {
            "control": control,
            "logs": logs,
            "mechanical_smoke": receipt,
        }
    )
    assert (archive / "mechanical_smoke.json").is_file()
    assert (archive / "mechanical_smoke_artifacts" / "summary.json").is_file()
    assert (archive / "mechanical_smoke.log").is_file()
    assert (archive / "archive_receipt.json").is_file()
    assert not receipt.exists()
    assert not artifacts.exists()
    assert preserved.read_text(encoding="utf-8") == '{"complete": true}\n'


def test_formal_orchestrator_archives_stale_contract_without_upstream_artifacts(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    control.mkdir()
    contract = control / "formal_execution_contract.json"
    contract.write_text('{"source": "old"}\n', encoding="utf-8")
    upstream = tmp_path / "upstream" / "checkpoint"
    upstream.mkdir(parents=True)
    checkpoint = upstream / "manifest.json"
    checkpoint.write_text('{"complete": true}\n', encoding="utf-8")
    archive = _archive_stale_formal_contract(
        {
            "control": control,
            "contract": contract,
        }
    )
    assert (archive / "formal_execution_contract.json").is_file()
    assert (archive / "archive_receipt.json").is_file()
    assert not contract.exists()
    assert checkpoint.read_text(encoding="utf-8") == '{"complete": true}\n'


def test_formal_layouts_use_separate_roots_manifests_and_templates() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    simple_paths = _formal_paths(
        repository_root, None, layout="test_time_simple"
    )
    wide_paths = _formal_paths(
        repository_root, None, layout="test_time_wide"
    )
    assert simple_paths["formal_root"] != wide_paths["formal_root"]
    assert simple_paths["outer_manifest"] != wide_paths["outer_manifest"]
    assert simple_paths["reference_acceptance_report"] == (
        simple_paths["control"] / "reference_acceptance_report.json"
    )
    assert (
        wide_paths["reference_acceptance_report"]
        == simple_paths["reference_acceptance_report"]
    )
    wide_templates = _template_paths(
        repository_root, layout="test_time_wide"
    )
    assert all(path.is_file() for path in wide_templates.values())
    assert wide_templates["sp"].name == "path_c_official_sp_wide_seed100.yaml"
    assert (
        wide_templates["decision_focused"].name
        == "path_c_model_v3_formal_wide_decision_focused.yaml"
    )
    launch = _launch_payload(
        template_path=_template_paths(
            repository_root, layout="test_time_simple"
        )["sp"],
        seed=123,
        output_dir=Path("/tmp/path_c_formal_launch_test"),
        reference_acceptance_report=simple_paths[
            "reference_acceptance_report"
        ],
    )
    assert launch["type_a_acceptance"]["reference_acceptance_report"] == str(
        simple_paths["reference_acceptance_report"].resolve()
    )


def test_formal_shared_stage_receipts_keep_condition_labels_and_reject_drift() -> None:
    states = {}
    for unit in range(10):
        for condition in FORMAL_CONDITIONS:
            states[f"outer_unit_{unit:02d}__{condition}"] = {
                "stages": {
                    stage: {"receipt_sha256": f"{unit:02d}-{stage}"}
                    for stage in (
                        "pool_check",
                        "prefit",
                        "training_calibration",
                    )
                }
            }

    receipts = _shared_stage_receipts(states)
    assert set(receipts[0]["prefit"]) == set(FORMAL_CONDITIONS)
    assert len(set(receipts[0]["prefit"].values())) == 1

    states["outer_unit_03__no_probe"]["stages"]["prefit"][
        "receipt_sha256"
    ] = "changed"
    with pytest.raises(RuntimeError, match="reuse one shared stage"):
        _shared_stage_receipts(states)
