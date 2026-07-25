"""R015 统一入口自动组装真实冻结输入的测试定义。"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import pytest
import yaml

from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline
from experiments.overcooked_v2.scripts import run_path_c_r015_pipeline as cli


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _materialize_fixture(tmp_path: Path) -> dict[str, object]:
    design_dir = tmp_path / "design"
    support_dir = tmp_path / "support"
    design_path = design_dir / "design_protocol.yaml"
    support_path = support_dir / "partner_support.yaml"
    environment_path = support_dir / "environment.yaml"
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text("{}\n", encoding="utf-8")

    design_protocol = yaml.safe_load(
        (CONFIG_DIR / "path_c_r015_design_data_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    support_registration = yaml.safe_load(
        (CONFIG_DIR / "path_c_r015_partner_support_simple.yaml").read_text(
            encoding="utf-8"
        )
    )
    support_registration["environment_config"] = "environment.yaml"
    support_registration["output_dir"] = "output"

    seeds = (100, 101, 102, 201, 202)
    candidate_by_seed = {
        int(value["training_seed"]): dict(value)
        for value in support_registration["candidates"]
    }
    ego = dict(design_protocol["support"]["ego_candidate"])
    candidate_by_seed[100] = ego
    manifests: dict[int, Path] = {}
    checkpoints: dict[int, Path] = {}
    dependency_files = {
        str(tmp_path / "external/jaxmarl/environments/overcooked_v2/settings.py"): {
            "sha256": "a" * 64
        },
        str(tmp_path / "external/jaxmarl/environments/overcooked_v2/overcooked.py"): {
            "sha256": "b" * 64
        },
        str(tmp_path / "external/jaxmarl/environments/overcooked_v2/layouts.py"): {
            "sha256": "c" * 64
        },
        str(tmp_path / "external/jaxmarl/environments/overcooked_v2/common.py"): {
            "sha256": "d" * 64
        },
    }
    for seed in seeds:
        owner = design_dir if seed == 100 else support_dir
        manifest_path = owner / "manifests" / f"seed_{seed}.json"
        checkpoint_path = owner / "checkpoints" / f"seed_{seed}"
        manifest = {
            "schema_version": "path_c_official_training_artifact_v2",
            "seed": seed,
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "format": "orbax_pytree",
                "parameter_tree_path": ["runner_state", "train_state", "params"],
            },
        }
        if seed == 100:
            manifest["training_implementation_dependencies"] = {
                "files": dependency_files
            }
        _write_json(manifest_path, manifest)
        manifests[seed] = manifest_path
        checkpoints[seed] = checkpoint_path
        candidate = candidate_by_seed[seed]
        candidate["training_manifest_path"] = str(
            manifest_path.relative_to(owner)
        )
        candidate["checkpoint_path"] = str(checkpoint_path.relative_to(owner))
        candidate["training_config_path"] = f"configs/seed_{seed}.yaml"
        candidate_by_seed[seed] = candidate

    design_protocol["support"]["ego_candidate"] = candidate_by_seed[100]
    design_protocol["support"]["partner_support_registration"] = os.path.relpath(
        support_path, design_dir
    )
    design_protocol["support"]["partner_support_report"] = "evidence/support.json"
    design_protocol["support"]["partner_support_episode_evidence"] = (
        "evidence/episodes.jsonl"
    )
    support_registration["candidates"] = [
        candidate_by_seed[seed] for seed in seeds if seed != 100
    ]
    _write_yaml(design_path, design_protocol)
    _write_yaml(support_path, support_registration)
    return {
        "design_path": design_path,
        "support_path": support_path,
        "design_protocol": design_protocol,
        "support_registration": support_registration,
        "manifests": manifests,
        "checkpoints": checkpoints,
        "dependency_files": dependency_files,
    }


def test_materialized_freeze_inputs_bind_all_five_runs_and_environment_sources(
    tmp_path,
):
    fixture = _materialize_fixture(tmp_path)
    selection = tmp_path / "run/design/design_selection_report.json"
    result = pipeline.materialize_r015_freeze_inputs(
        design_protocol_path=fixture["design_path"],
        partner_support_registration_path=fixture["support_path"],
        design_selection_report_path=selection,
    )
    assert result["schema_version"] == pipeline.FREEZE_INPUT_SCHEMA
    checkpoints = result["checkpoints"]
    assert [value["training_seed"] for value in checkpoints] == [
        100,
        101,
        102,
        201,
        202,
    ]
    assert [value["role"] for value in checkpoints] == [
        "ego",
        "partner",
        "partner",
        "partner",
        "partner",
    ]
    assert {value["format"] for value in checkpoints} == {"orbax_pytree"}
    assert all(
        value["parameter_tree_path"] == [
            "runner_state",
            "train_state",
            "params",
        ]
        for value in checkpoints
    )
    assert [Path(value["path"]) for value in checkpoints] == [
        fixture["checkpoints"][seed].resolve()
        for seed in (100, 101, 102, 201, 202)
    ]
    assert result["environment_sources"] == {
        "settings": next(
            path for path in fixture["dependency_files"] if path.endswith("settings.py")
        ),
        "overcooked": next(
            path
            for path in fixture["dependency_files"]
            if path.endswith("overcooked.py")
        ),
        "layouts": next(
            path for path in fixture["dependency_files"] if path.endswith("layouts.py")
        ),
        "common": next(
            path for path in fixture["dependency_files"] if path.endswith("common.py")
        ),
    }
    assert Path(result["support_registration_path"]) == fixture[
        "support_path"
    ].resolve()
    assert Path(result["support_report_path"]) == (
        fixture["design_path"].parent / "evidence/support.json"
    ).resolve()
    assert Path(result["support_episode_evidence_path"]) == (
        fixture["design_path"].parent / "evidence/episodes.jsonl"
    ).resolve()


def test_materialized_freeze_inputs_resolve_candidates_from_their_owner_files(
    tmp_path,
):
    fixture = _materialize_fixture(tmp_path)
    result = pipeline.materialize_r015_freeze_inputs(
        design_protocol_path=fixture["design_path"],
        partner_support_registration_path=fixture["support_path"],
        design_selection_report_path=tmp_path / "selection.json",
    )
    by_seed = {value["training_seed"]: value for value in result["checkpoints"]}
    assert Path(by_seed[100]["training_manifest_path"]).parent.parent == (
        fixture["design_path"].parent
    )
    for seed in (101, 102, 201, 202):
        assert Path(by_seed[seed]["training_manifest_path"]).parent.parent == (
            fixture["support_path"].parent
        )


def test_materialized_freeze_inputs_fail_closed_on_support_or_manifest_drift(
    tmp_path,
):
    fixture = _materialize_fixture(tmp_path)
    arguments = {
        "design_protocol_path": fixture["design_path"],
        "partner_support_registration_path": fixture["support_path"],
        "design_selection_report_path": tmp_path / "selection.json",
    }
    another_support = tmp_path / "another/support.yaml"
    _write_yaml(another_support, fixture["support_registration"])
    with pytest.raises(ValueError, match="支持登记与设计协议不一致"):
        pipeline.materialize_r015_freeze_inputs(
            **{**arguments, "partner_support_registration_path": another_support}
        )

    manifest_path = fixture["manifests"][101]
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_seed = copy.deepcopy(original)
    changed_seed["seed"] = 999
    _write_json(manifest_path, changed_seed)
    with pytest.raises(ValueError, match="训练清单身份不正确"):
        pipeline.materialize_r015_freeze_inputs(**arguments)
    _write_json(manifest_path, original)
    changed_checkpoint = copy.deepcopy(original)
    changed_checkpoint["checkpoint"]["path"] = str(tmp_path / "wrong-checkpoint")
    _write_json(manifest_path, changed_checkpoint)
    with pytest.raises(ValueError, match="指向不同 checkpoint"):
        pipeline.materialize_r015_freeze_inputs(**arguments)


def _direct_arguments(tmp_path: Path, fixture: dict[str, object]) -> argparse.Namespace:
    preregistration = tmp_path / "preregistration.yaml"
    pilot = tmp_path / "pilot.yaml"
    authorization = tmp_path / "authorization.yaml"
    for path in (preregistration, pilot, authorization):
        path.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        output_root=tmp_path / "run",
        authorization_file=authorization,
        design_protocol=fixture["design_path"],
        freeze_inputs=None,
        preregistration_template=preregistration,
        partner_support_registration=fixture["support_path"],
        pilot_protocol=pilot,
        formal_runtime_config=None,
    )


def test_direct_cli_prebinds_generated_files_before_design_and_reuses_them(
    tmp_path,
):
    from experiments.overcooked_v2.path_c_pool_admission import (
        load_r015_partner_support_config,
    )

    fixture = _materialize_fixture(tmp_path)
    arguments = _direct_arguments(tmp_path, fixture)
    first_config = cli._direct_config(arguments)
    output_root = arguments.output_root.resolve()
    freeze_path = output_root / "pipeline_freeze_inputs.generated.json"
    runtime_path = output_root / "pipeline_formal_runtime.generated.yaml"
    assert freeze_path.is_file()
    assert runtime_path.is_file()
    first_payload = yaml.safe_load(first_config.read_text(encoding="utf-8"))
    assert first_payload["paths"]["freeze_inputs"] == str(freeze_path)
    assert first_payload["paths"]["formal_runtime_config"] == str(runtime_path)
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    assert Path(runtime["support"]["ego_candidate"]["checkpoint_path"]).is_absolute()
    assert Path(runtime["support"]["ego_candidate"]["training_manifest_path"]).is_absolute()
    assert Path(runtime["support"]["partner_support_registration"]).is_absolute()
    loaded_support = load_r015_partner_support_config(
        runtime["support"]["partner_support_registration"]
    )
    assert len(loaded_support["candidates"]) == 4
    assert all(
        Path(candidate["checkpoint_path"]).is_absolute()
        for candidate in loaded_support["candidates"]
    )
    loaded = pipeline.load_r015_pipeline_config(first_config)
    first_binding = pipeline._build_run_binding(
        config_path=first_config,
        config=loaded,
        paths=pipeline._pipeline_paths(
            loaded,
            config_path=first_config,
            through="design",
        ),
    )

    # 先运行到 design 后只补入后续授权，不得让同一入口路径或运行绑定漂移。
    arguments.authorization_file.write_text(
        "design: authorized\nfreeze: authorized\nformal: authorized\n",
        encoding="utf-8",
    )
    second_config = cli._direct_config(arguments)
    assert second_config == first_config
    assert yaml.safe_load(second_config.read_text(encoding="utf-8")) == first_payload
    reloaded = pipeline.load_r015_pipeline_config(second_config)
    second_binding = pipeline._build_run_binding(
        config_path=second_config,
        config=reloaded,
        paths=pipeline._pipeline_paths(
            reloaded,
            config_path=second_config,
            through="formal",
        ),
    )
    assert second_binding == first_binding


@pytest.mark.parametrize(
    "generated_name",
    (
        "pipeline_freeze_inputs.generated.json",
        "pipeline_formal_runtime.generated.yaml",
    ),
)
def test_direct_cli_refuses_generated_file_drift(tmp_path, generated_name):
    fixture = _materialize_fixture(tmp_path)
    arguments = _direct_arguments(tmp_path, fixture)
    cli._direct_config(arguments)
    generated = arguments.output_root / generated_name
    generated.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="自动冻结输入已经改变|另一组统一入口路径"):
        cli._direct_config(arguments)


def test_cli_generation_is_unconditional_on_the_requested_terminal_stage():
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert 'output_root / "pipeline_formal_runtime.generated.yaml"' in source
    assert 'output_root / "pipeline_freeze_inputs.generated.json"' in source
    direct = source[source.index("def _direct_config") : source.index("def main")]
    assert "arguments.through" not in direct
