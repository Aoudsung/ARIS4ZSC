from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import yaml

from src.path_c.contracts.config import (
    CONDITION_CONTROLLERS,
    PathCFormalTemplateV2,
    PathCFormalTemplateV3,
    PathCModelConfig,
)
from src.path_c.contracts.outer_units import (
    FAMILY_POOL_FORMAL_SEED_ROOT,
    FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION,
    FORMAL_SEED_ROOT,
    OTHER_PLAY_FAMILY,
    OUTER_UNITS_SCHEMA_VERSION,
    SELF_PLAY_FAMILY,
    OuterUnitsManifest,
    derive_formal_seed,
    derive_family_pool_seed,
)


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "configs"
CONFIG_VARIANTS = {
    "development": {
        "decision_focused": "path_c_model_development_decision_focused.yaml",
        "no_probe": "path_c_model_development_no_probe.yaml",
        "random_safe_probe": "path_c_model_development_random_safe_probe.yaml",
        "generic_response_information": (
            "path_c_model_development_generic_response_information.yaml"
        ),
    },
    "development_seed2": {
        "decision_focused": "path_c_model_development_seed2_decision_focused.yaml",
        "no_probe": "path_c_model_development_seed2_no_probe.yaml",
        "random_safe_probe": "path_c_model_development_seed2_random_safe_probe.yaml",
        "generic_response_information": (
            "path_c_model_development_seed2_generic_response_information.yaml"
        ),
    },
    "formal": {
        "decision_focused": "path_c_model_formal_decision_focused.yaml",
        "no_probe": "path_c_model_formal_no_probe.yaml",
        "random_safe_probe": "path_c_model_formal_random_safe_probe.yaml",
        "generic_response_information": (
            "path_c_model_formal_generic_response_information.yaml"
        ),
    },
    "formal_wide": {
        "decision_focused": "path_c_model_formal_wide_decision_focused.yaml",
        "no_probe": "path_c_model_formal_wide_no_probe.yaml",
        "random_safe_probe": "path_c_model_formal_wide_random_safe_probe.yaml",
        "generic_response_information": (
            "path_c_model_formal_wide_generic_response_information.yaml"
        ),
    },
    "family_pool_formal": {
        "decision_focused": "path_c_model_v3_formal_decision_focused.yaml",
        "no_probe": "path_c_model_v3_formal_no_probe.yaml",
        "random_safe_probe": "path_c_model_v3_formal_random_safe_probe.yaml",
        "generic_response_information": (
            "path_c_model_v3_formal_generic_response_information.yaml"
        ),
    },
    "family_pool_formal_wide": {
        "decision_focused": "path_c_model_v3_formal_wide_decision_focused.yaml",
        "no_probe": "path_c_model_v3_formal_wide_no_probe.yaml",
        "random_safe_probe": "path_c_model_v3_formal_wide_random_safe_probe.yaml",
        "generic_response_information": (
            "path_c_model_v3_formal_wide_generic_response_information.yaml"
        ),
    },
}
# Each config group shares one run_kind; the second-seed development repeat
# keeps run_kind "development" while using its own seeds and output root.
RUN_KIND_FOR_GROUP = {
    "development": "development",
    "development_seed2": "development",
    "formal": "formal",
    "formal_wide": "formal",
    "family_pool_formal": "formal",
    "family_pool_formal_wide": "formal",
}


def _mapping(name: str = "path_c_model_development_decision_focused.yaml") -> dict:
    payload = yaml.safe_load((CONFIG_DIRECTORY / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _outer_units_manifest(
    tmp_path: Path, *, layout: str = "test_time_simple"
) -> OuterUnitsManifest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    units = []
    for unit_id in range(10):
        sources = []
        source_specs = (
            ("backbone", SELF_PLAY_FAMILY, "official_backbone", 0),
            ("self_play_0", SELF_PLAY_FAMILY, "official_partner_self_play", 0),
            ("self_play_1", SELF_PLAY_FAMILY, "official_partner_self_play", 1),
            ("other_play_0", OTHER_PLAY_FAMILY, "official_partner_other_play", 0),
            ("other_play_1", OTHER_PLAY_FAMILY, "official_partner_other_play", 1),
        )
        for member_id, family, role, member_index in source_specs:
            launch = tmp_path / f"unit_{unit_id}_{member_id}.yaml"
            launch.write_text(f"unit: {unit_id}\nmember: {member_id}\n", encoding="utf-8")
            sources.append(
                {
                    "member_id": member_id,
                    "family_id": family,
                    "training_seed": derive_formal_seed(
                        layout=layout,
                        outer_unit_id=unit_id,
                        role=role,
                        member_index=member_index,
                    ),
                    "checkpoint_path": str(
                        tmp_path / "checkpoints" / f"unit_{unit_id}_{member_id}"
                    ),
                    "flax_weights_sha256": hashlib.sha256(
                        f"weights-{unit_id}-{member_id}".encode()
                    ).hexdigest(),
                    "training_run_id": (
                        f"formal-{layout}-unit-{unit_id}-{member_id}"
                    ),
                    "launch_config_path": str(launch),
                    "launch_config_sha256": hashlib.sha256(launch.read_bytes()).hexdigest(),
                }
            )
        units.append(
            {
                "outer_unit_id": unit_id,
                "backbone": sources[0],
                "partners": sources[1:],
                "seeds": {
                    name: derive_formal_seed(
                        layout=layout,
                        outer_unit_id=unit_id,
                        role=role,
                    )
                    for name, role in (
                        ("model_seed", "model"),
                        ("environment_seed", "environment"),
                        ("evaluation_seed", "evaluation"),
                    )
                },
            }
        )
    payload = {
        "schema_version": OUTER_UNITS_SCHEMA_VERSION,
        "layout": layout,
        "seed_root": FORMAL_SEED_ROOT,
        "units": units,
    }
    return OuterUnitsManifest.from_mapping(
        payload,
        base_dir=tmp_path,
        path=tmp_path / "outer_units.json",
        verify_files=True,
    )


def _family_outer_units_manifest(
    tmp_path: Path, *, layout: str = "test_time_simple"
) -> OuterUnitsManifest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    units = []
    for unit_id in range(10):
        sources = []
        for member_id, family, role, member_index in (
            ("backbone", SELF_PLAY_FAMILY, "official_backbone", 0),
            ("self_play_0", SELF_PLAY_FAMILY, "official_partner_self_play", 0),
            ("self_play_1", SELF_PLAY_FAMILY, "official_partner_self_play", 1),
            ("other_play_0", OTHER_PLAY_FAMILY, "official_partner_other_play", 0),
            ("other_play_1", OTHER_PLAY_FAMILY, "official_partner_other_play", 1),
        ):
            launch = tmp_path / f"unit_{unit_id}_{member_id}.yaml"
            launch.write_text(f"unit: {unit_id}\nmember: {member_id}\n")
            run_id = f"family-{layout}-unit-{unit_id}-{member_id}"
            launch_sha256 = hashlib.sha256(launch.read_bytes()).hexdigest()
            history = []
            for checkpoint_index in range(3):
                checkpoint = (
                    tmp_path
                    / "checkpoints"
                    / f"unit_{unit_id}_{member_id}_{checkpoint_index}"
                )
                checkpoint.mkdir(parents=True, exist_ok=True)
                weights = hashlib.sha256(
                    f"history-{unit_id}-{member_id}-{checkpoint_index}".encode()
                ).hexdigest()
                history.append(
                    {
                        "checkpoint_index": checkpoint_index,
                        "update_step": checkpoint_index,
                        "effective_environment_steps": checkpoint_index,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            f"checkpoint-{unit_id}-{member_id}-{checkpoint_index}".encode()
                        ).hexdigest(),
                        "flax_weights_sha256": weights,
                    }
                )
            final_checkpoint = (
                tmp_path / "final" / f"unit_{unit_id}_{member_id}"
            ).resolve()
            variant = "rnn-sp" if family == SELF_PLAY_FAMILY else "rnn-op"
            update_count = 457 if variant == "rnn-sp" else 1_831
            effective_steps = (
                29_949_952 if variant == "rnn-sp" else 29_999_104
            )
            metrics = (
                tmp_path / "admissions" / f"unit_{unit_id}_{member_id}_metrics.json"
            )
            metrics.parent.mkdir(parents=True, exist_ok=True)
            metrics.write_text(
                json.dumps(
                    {
                        "schema_version": "path_c_official_training_metrics_v1",
                        "scientific_readout_allowed": False,
                        "experiment_variant": variant,
                        "seed": derive_family_pool_seed(
                            layout=layout,
                            outer_unit_id=unit_id,
                            role=role,
                            member_index=member_index,
                        ),
                        "effective_environment_steps": effective_steps,
                        "completed_episode_count": 1,
                        "metric_record_count": update_count,
                        "metrics_by_update": {
                            "returned_episode_returns": [110.0] * update_count
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            admission = (
                tmp_path / "admissions" / f"unit_{unit_id}_{member_id}.json"
            )
            admission.parent.mkdir(parents=True, exist_ok=True)
            admission.write_text(
                json.dumps(
                    {
                        "schema_version": "path_c_official_ability_admission_v1",
                        "rule_id": "official_final_quarter_ability_v1",
                        "run_kind": "formal",
                        "scientific_readout_allowed": False,
                        "experiment_variant": variant,
                        "training_seed": derive_family_pool_seed(
                            layout=layout,
                            outer_unit_id=unit_id,
                            role=role,
                            member_index=member_index,
                        ),
                        "effective_environment_steps": effective_steps,
                        "passed": True,
                        "history_selection_consulted_return": False,
                        "training_run_id": run_id,
                        "launch_config_sha256": launch_sha256,
                        "checkpoint_path": str(final_checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            f"final-{unit_id}-{member_id}".encode()
                        ).hexdigest(),
                        "model_weights_sha256": history[-1][
                            "flax_weights_sha256"
                        ],
                        "metric": "returned_episode_returns",
                        "metric_meaning": "official_wrapper_raw_episode_return",
                        "metric_record_count": update_count,
                        "quarter_partition": (
                            "contiguous_integer_update_quarters_v1"
                        ),
                        "quarter_boundaries_zero_based": [
                            index * update_count // 4 for index in range(5)
                        ],
                        "quarter_means": [110.0, 110.0, 110.0, 110.0],
                        "final_quarter_mean_raw_return": 110.0,
                        "peak_quarter_mean_raw_return": 110.0,
                        "final_to_peak_quarter_ratio": 1.0,
                        "minimum_final_quarter_mean_raw_return": 100.0,
                        "minimum_final_to_peak_quarter_ratio": 0.9,
                        "performance_threshold_pass": True,
                        "no_collapse_threshold_pass": True,
                        "selection_effect": (
                            "audit_only_include_all_fixed_seeds_v1"
                        ),
                        "included_in_population": True,
                        "seed_replacement_allowed": False,
                        "retraining_until_pass_allowed": False,
                        "training_metrics_path": str(metrics),
                        "training_metrics_sha256": hashlib.sha256(
                            metrics.read_bytes()
                        ).hexdigest(),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            sources.append(
                {
                    "member_id": member_id,
                    "family_id": family,
                    "training_seed": derive_family_pool_seed(
                        layout=layout,
                        outer_unit_id=unit_id,
                        role=role,
                        member_index=member_index,
                    ),
                    "checkpoint_path": str(final_checkpoint),
                    "flax_weights_sha256": history[-1][
                        "flax_weights_sha256"
                    ],
                    "training_run_id": run_id,
                    "launch_config_path": str(launch),
                    "launch_config_sha256": launch_sha256,
                    "ability_admission_path": str(admission),
                    "ability_admission_sha256": hashlib.sha256(
                        admission.read_bytes()
                    ).hexdigest(),
                    "checkpoint_history": history,
                }
            )
        units.append(
            {
                "outer_unit_id": unit_id,
                "backbone": sources[0],
                "partners": sources[1:],
                "seeds": {
                    name: derive_family_pool_seed(
                        layout=layout,
                        outer_unit_id=unit_id,
                        role=role,
                    )
                    for name, role in (
                        ("model_seed", "model"),
                        ("environment_seed", "environment"),
                        ("evaluation_seed", "evaluation"),
                    )
                },
            }
        )
    return OuterUnitsManifest.from_mapping(
        {
            "schema_version": FAMILY_POOL_OUTER_UNITS_SCHEMA_VERSION,
            "layout": layout,
            "seed_root": FAMILY_POOL_FORMAL_SEED_ROOT,
            "units": units,
        },
        base_dir=tmp_path,
        path=tmp_path / "family_outer_units.json",
        verify_files=True,
    )


def test_development_and_formal_configs_resolve_paths_and_registered_budgets(
    tmp_path: Path,
) -> None:
    development_path = CONFIG_DIRECTORY / "path_c_model_development_decision_focused.yaml"
    development = PathCModelConfig.from_mapping(
        _mapping(), base_dir=development_path.parent
    )
    assert development.output_root.is_absolute()
    assert development.backbone_init.checkpoint_path.is_absolute()
    assert development.scientific_readout_allowed is False
    assert development.environment.num_envs == 32
    assert development.prefit.env_steps == 204_800
    assert development.probe.decision_null_quantile == 0.95
    assert development.probe.information_quantile == 0.80
    assert development.output_root.name == "path_c_complete_model_zero_anchored_development"
    formal_path = CONFIG_DIRECTORY / "path_c_model_formal_decision_focused.yaml"
    formal_template = PathCFormalTemplateV2.from_mapping(
        _mapping(formal_path.name), base_dir=formal_path.parent
    )
    formal = formal_template.resolve(_outer_units_manifest(tmp_path), outer_unit_id=0)
    assert formal.scientific_readout_allowed is False
    assert formal.environment.num_envs == 250
    assert formal.prefit.env_steps == 1_000_000
    assert formal.calibration.episodes == 500
    assert formal.adaptation.env_steps == 10_000_000
    assert formal.evaluation.episodes_per_pairing == 500
    assert formal.evaluation.schedule == "external_standard_matrix"
    assert formal.formal_outer_unit is not None
    assert formal.formal_outer_unit.outer_unit_id == 0
    wide_path = (
        CONFIG_DIRECTORY / "path_c_model_formal_wide_decision_focused.yaml"
    )
    wide_template = PathCFormalTemplateV2.from_mapping(
        _mapping(wide_path.name), base_dir=wide_path.parent
    )
    wide = wide_template.resolve(
        _outer_units_manifest(
            tmp_path / "wide_manifest", layout="test_time_wide"
        ),
        outer_unit_id=0,
    )
    assert wide.environment.layout == "test_time_wide"
    assert wide.output_root.name == "outer_unit_00"
    assert wide.output_root.parent.name == "test_time_wide"


@pytest.mark.parametrize("condition,controller", sorted(CONDITION_CONTROLLERS.items()))
def test_all_four_condition_controller_pairs_are_accepted(
    condition: str, controller: str
) -> None:
    payload = _mapping()
    payload["condition_id"] = condition
    payload["controller"] = controller
    parsed = PathCModelConfig.from_mapping(payload, base_dir=CONFIG_DIRECTORY)
    assert parsed.condition_id == condition
    assert parsed.controller == controller


def test_all_config_groups_differ_internally_only_by_condition_and_controller() -> None:
    for group, variants in CONFIG_VARIANTS.items():
        base_payload = _mapping(variants["decision_focused"])
        base_comparable = copy.deepcopy(base_payload)
        base_comparable.pop("condition_id")
        base_comparable.pop("controller")
        formal_group = "formal" in group
        template_class = (
            PathCFormalTemplateV3
            if group.startswith("family_pool")
            else PathCFormalTemplateV2
        )
        base_config = (
            template_class.from_mapping(base_payload, base_dir=CONFIG_DIRECTORY)
            if formal_group
            else PathCModelConfig.from_mapping(base_payload, base_dir=CONFIG_DIRECTORY)
        )
        for condition, filename in variants.items():
            payload = _mapping(filename)
            assert payload["run_kind"] == RUN_KIND_FOR_GROUP[group]
            assert payload["condition_id"] == condition
            assert payload["controller"] == CONDITION_CONTROLLERS[condition]
            comparable = copy.deepcopy(payload)
            comparable.pop("condition_id")
            comparable.pop("controller")
            assert comparable == base_comparable
            parsed = (
                template_class.from_mapping(payload, base_dir=CONFIG_DIRECTORY)
                if formal_group
                else PathCModelConfig.from_mapping(payload, base_dir=CONFIG_DIRECTORY)
            )
            if formal_group:
                assert parsed.environment == base_config.environment
                assert parsed.prefit == base_config.prefit
                assert parsed.adaptation == base_config.adaptation
            else:
                assert parsed.shared_stage_mapping() == base_config.shared_stage_mapping()


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda value: value.update(scientific_readout_allowed=True), "Development"),
        (lambda value: value.update(controller="off"), "does not match"),
        (lambda value: value.update(unexpected=True), "unknown"),
        (
            lambda value: value["probe"]["threshold"].update(
                decision_null_quantile=0.90
            ),
            "0.95",
        ),
    ),
)
def test_configuration_rejects_readout_controller_unknown_key_and_quantile(
    mutation, match: str
) -> None:
    payload = _mapping()
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        PathCModelConfig.from_mapping(payload, base_dir=CONFIG_DIRECTORY)


def test_formal_budget_and_manual_threshold_are_rejected() -> None:
    payload = _mapping("path_c_model_formal_decision_focused.yaml")
    payload["stages"]["adaptation"]["env_steps"] -= 100_000
    with pytest.raises(ValueError, match="Formal runs"):
        PathCFormalTemplateV2.from_mapping(payload, base_dir=CONFIG_DIRECTORY)
    payload = _mapping("path_c_model_formal_decision_focused.yaml")
    payload["probe"]["threshold"] = {
        "source": "manual",
        "decision_null_quantile": 0.95,
        "information_quantile": 0.80,
        "manual_value": 0.0,
    }
    with pytest.raises(ValueError, match="calibration"):
        PathCFormalTemplateV2.from_mapping(payload, base_dir=CONFIG_DIRECTORY)

    development = _mapping()
    development["probe"]["threshold"] = {
        "source": "manual",
        "decision_null_quantile": 0.95,
        "information_quantile": 0.80,
        "manual_value": -1.0,
    }
    parsed = PathCModelConfig.from_mapping(development, base_dir=CONFIG_DIRECTORY)
    assert parsed.probe.manual_threshold == -1.0


def test_partner_pool_requires_four_unique_runs_and_two_per_family() -> None:
    payload = _mapping()
    payload["partner_pool"]["members"][1]["training_run_id"] = payload[
        "partner_pool"
    ]["members"][0]["training_run_id"]
    with pytest.raises(ValueError, match="distinct training run"):
        PathCModelConfig.from_mapping(payload, base_dir=CONFIG_DIRECTORY)
    payload = _mapping()
    payload["partner_pool"]["members"][1]["family_id"] = (
        "official_rnn_op_other_play_v1"
    )
    with pytest.raises(ValueError, match="two checkpoints"):
        PathCModelConfig.from_mapping(payload, base_dir=CONFIG_DIRECTORY)


def test_partner_pool_rejects_an_unregistered_checkpoint_source() -> None:
    payload = _mapping()
    payload["partner_pool"]["members"][0]["checkpoint_path"] = (
        "../../../results/path_c_official_sp_simple/seed_103/run_0/ckpt_final"
    )
    with pytest.raises(ValueError, match="registered seeds"):
        PathCModelConfig.from_mapping(payload, base_dir=CONFIG_DIRECTORY)


def test_shared_stage_hash_input_excludes_only_condition_specific_work() -> None:
    first = PathCModelConfig.from_mapping(_mapping(), base_dir=CONFIG_DIRECTORY)
    changed = _mapping()
    changed["condition_id"] = "no_probe"
    changed["controller"] = "off"
    second = PathCModelConfig.from_mapping(changed, base_dir=CONFIG_DIRECTORY)
    assert first.config_sha256 != second.config_sha256
    assert first.shared_stage_mapping() == second.shared_stage_mapping()
    changed_gamma = _mapping()
    changed_gamma["stages"]["adaptation"]["gamma"] = 0.98
    third = PathCModelConfig.from_mapping(changed_gamma, base_dir=CONFIG_DIRECTORY)
    assert first.shared_stage_mapping() != third.shared_stage_mapping()


def test_formal_outer_units_are_complete_disjoint_and_shared_only_within_unit(
    tmp_path: Path,
) -> None:
    manifest = _outer_units_manifest(tmp_path)
    assert len(manifest.units) == 10
    assert all(len(unit.sources) == 5 for unit in manifest.units)
    all_paths = [source.checkpoint_path for unit in manifest.units for source in unit.sources]
    all_runs = [source.training_run_id for unit in manifest.units for source in unit.sources]
    all_seeds = [
        source.training_seed for unit in manifest.units for source in unit.sources
    ] + [seed for unit in manifest.units for seed in unit.seeds.to_mapping().values()]
    assert len(set(all_paths)) == 50
    assert len(set(all_runs)) == 50
    assert len(set(all_seeds)) == 80

    decision = PathCFormalTemplateV2.from_mapping(
        _mapping("path_c_model_formal_decision_focused.yaml"),
        base_dir=CONFIG_DIRECTORY,
    )
    no_probe = PathCFormalTemplateV2.from_mapping(
        _mapping("path_c_model_formal_no_probe.yaml"),
        base_dir=CONFIG_DIRECTORY,
    )
    decision_unit_0 = decision.resolve(manifest, outer_unit_id=0)
    no_probe_unit_0 = no_probe.resolve(manifest, outer_unit_id=0)
    decision_unit_1 = decision.resolve(manifest, outer_unit_id=1)
    assert decision_unit_0.shared_stage_mapping() == no_probe_unit_0.shared_stage_mapping()
    assert decision_unit_0.shared_stage_mapping() != decision_unit_1.shared_stage_mapping()
    assert decision_unit_0.output_root != decision_unit_1.output_root


def test_wide_formal_templates_and_official_launches_use_the_wide_layout(
) -> None:
    from experiments.overcooked_v2.official.overcooked_v2_experiments_wide_adapter import (
        compose_official_wide_training_config,
        validate_official_wide_launch_config,
    )

    for filename in (
        "path_c_official_sp_wide_seed100.yaml",
        "path_c_official_op_wide_seed201.yaml",
    ):
        payload = _mapping(filename)
        validate_official_wide_launch_config(payload)
        resolved = compose_official_wide_training_config(payload)
        assert resolved["env"]["ENV_KWARGS"]["layout"] == "test_time_wide"


def test_outer_unit_manifest_rejects_reused_source_identity(tmp_path: Path) -> None:
    manifest = _outer_units_manifest(tmp_path)
    payload = {
        "schema_version": manifest.schema_version,
        "layout": manifest.layout,
        "seed_root": manifest.seed_root,
        "units": [unit.to_mapping() for unit in manifest.units],
    }
    payload["units"][1]["backbone"]["training_run_id"] = payload["units"][0][
        "backbone"
    ]["training_run_id"]
    with pytest.raises(ValueError, match="training run"):
        OuterUnitsManifest.from_mapping(
            payload, base_dir=tmp_path, verify_files=True
        )


def test_family_pool_v3_resolves_four_equal_families_and_three_checkpoint_histories(
    tmp_path: Path,
) -> None:
    manifest = _family_outer_units_manifest(tmp_path)
    template = PathCFormalTemplateV3.from_mapping(
        _mapping("path_c_model_v3_formal_decision_focused.yaml"),
        base_dir=CONFIG_DIRECTORY,
    )
    config = template.resolve(manifest, outer_unit_id=0)
    assert config.is_family_pool
    assert config.num_prototypes == 4
    assert config.family_pool is not None
    assert config.family_pool.family_weights == (0.25, 0.25, 0.25, 0.25)
    assert len(config.partner_pool) == 15
    assert {
        prototype: sum(
            member.prototype_index == prototype for member in config.partner_pool
        )
        for prototype in (1, 2, 3)
    } == {1: 3, 2: 6, 3: 6}
    assert all(len(source.checkpoint_history) == 3 for source in manifest.units[0].sources)
    assert config.calibration.episodes == 500
    assert config.deployment_calibration is not None
    assert config.deployment_calibration.episodes == 500
    serialized = config.to_mapping()
    reparsed = PathCModelConfig.from_mapping(serialized, base_dir=tmp_path)
    assert reparsed.config_sha256 == config.config_sha256
    manifest_payload = {
        "schema_version": manifest.schema_version,
        "layout": manifest.layout,
        "seed_root": manifest.seed_root,
        "units": [unit.to_mapping() for unit in manifest.units],
    }
    admission_path = manifest.units[0].backbone.ability_admission_path
    assert admission_path is not None
    admission_payload = json.loads(admission_path.read_text(encoding="utf-8"))
    metrics_path = Path(admission_payload["training_metrics_path"])
    original_metrics = metrics_path.read_text(encoding="utf-8")
    metrics_path.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="raw metric evidence changed"):
        OuterUnitsManifest.from_mapping(
            manifest_payload,
            base_dir=tmp_path,
            verify_files=True,
        )
    metrics_path.write_text(original_metrics, encoding="utf-8")
    admission_path.write_text('{"passed": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="ability-admission report hash"):
        OuterUnitsManifest.from_mapping(
            manifest_payload,
            base_dir=tmp_path,
            verify_files=True,
        )


def test_family_pool_below_threshold_observation_remains_in_population(
    tmp_path: Path,
) -> None:
    from experiments.overcooked_v2.official.path_c_family_pool_training import (
        evaluate_final_policy_ability_admission,
    )
    from experiments.overcooked_v2.scripts.run_path_c_formal_standard import (
        _write_or_validate_official_ability_admission,
    )

    manifest = _family_outer_units_manifest(tmp_path)
    payload = {
        "schema_version": manifest.schema_version,
        "layout": manifest.layout,
        "seed_root": manifest.seed_root,
        "units": [unit.to_mapping() for unit in manifest.units],
    }
    source = payload["units"][0]["backbone"]
    admission_path = Path(source["ability_admission_path"])
    previous = json.loads(admission_path.read_text(encoding="utf-8"))
    metrics_path = Path(previous["training_metrics_path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    update_count = int(metrics["metric_record_count"])
    metrics["metrics_by_update"]["returned_episode_returns"] = [90.0] * update_count
    metrics_path.write_text(
        json.dumps(metrics, sort_keys=True),
        encoding="utf-8",
    )
    observation = dict(
        evaluate_final_policy_ability_admission(
            metrics,
            training_run_id=source["training_run_id"],
            launch_config_sha256=source["launch_config_sha256"],
            checkpoint_path=source["checkpoint_path"],
            checkpoint_sha256=previous["checkpoint_sha256"],
            model_weights_sha256=source["flax_weights_sha256"],
        )
    )
    observation.update(
        {
            "training_metrics_path": str(metrics_path),
            "training_metrics_sha256": hashlib.sha256(
                metrics_path.read_bytes()
            ).hexdigest(),
        }
    )
    admission_path.write_text(
        json.dumps(observation, sort_keys=True),
        encoding="utf-8",
    )
    source["ability_admission_sha256"] = hashlib.sha256(
        admission_path.read_bytes()
    ).hexdigest()
    reparsed = OuterUnitsManifest.from_mapping(
        payload,
        base_dir=tmp_path,
        verify_files=True,
    )
    assert len(reparsed.units) == 10
    assert observation["passed"] is False
    assert observation["included_in_population"] is True
    writer_source = inspect.getsource(
        _write_or_validate_official_ability_admission
    )
    assert "failed ability admission" not in writer_source
    assert 'report["passed"] is not True' not in writer_source


def test_family_pool_seed_domain_has_no_layout_unit_role_or_checkpoint_collision() -> None:
    values = {
        derive_family_pool_seed(
            layout=layout,
            outer_unit_id=unit,
            role=role,
            member_index=member,
            checkpoint_index=checkpoint,
            stage=stage,
        )
        for layout in ("test_time_simple", "test_time_wide")
        for unit in range(10)
        for role in ("model", "environment", "evaluation", "partner")
        for member in range(2)
        for checkpoint in range(3)
        for stage in ("prefit", "adaptation", "evaluation")
    }
    assert len(values) == 2 * 10 * 4 * 2 * 3 * 3


def test_official_history_contract_forbids_return_selection() -> None:
    from experiments.overcooked_v2.official.path_c_family_pool_training import (
        evaluate_final_policy_ability_admission,
        validate_checkpoint_history,
    )

    rows = [
        {
            "checkpoint_index": index,
            "update_step": index,
            "effective_environment_steps": index,
            "path": f"/tmp/checkpoint_{index}",
            "format": "orbax_pytree_directory_v1",
            "parameter_tree_path": ["params"],
            "selection_rule": "mechanical_official_update_schedule_v1",
            "checkpoint_sha256": f"{index + 1:064x}",
            "model_weights_sha256": f"{index + 4:064x}",
        }
        for index in range(3)
    ]
    assert len(
        validate_checkpoint_history(
            rows,
            final_weights_sha256=rows[-1]["model_weights_sha256"],
            verify_files=False,
        )
    ) == 3
    changed = copy.deepcopy(rows)
    changed[1]["selection_rule"] = "best_return_checkpoint"
    with pytest.raises(ValueError, match="return selection"):
        validate_checkpoint_history(
            changed,
            final_weights_sha256=rows[-1]["model_weights_sha256"],
            verify_files=False,
        )
    admitted = evaluate_final_policy_ability_admission(
        {
            "schema_version": "path_c_official_training_metrics_v1",
            "experiment_variant": "rnn-sp",
            "seed": 123,
            "effective_environment_steps": 29_949_952,
            "metric_record_count": 457,
            "metrics_by_update": {
                "returned_episode_returns": [110.0] * 457
            },
        },
        training_run_id="run-123",
        launch_config_sha256="1" * 64,
        checkpoint_path="/tmp/final",
        checkpoint_sha256="2" * 64,
        model_weights_sha256="3" * 64,
    )
    assert admitted["passed"] is True
    assert admitted["history_selection_consulted_return"] is False
    collapsed = evaluate_final_policy_ability_admission(
        {
            "schema_version": "path_c_official_training_metrics_v1",
            "experiment_variant": "rnn-sp",
            "seed": 124,
            "effective_environment_steps": 29_949_952,
            "metric_record_count": 457,
            "metrics_by_update": {
                "returned_episode_returns": (
                    [120.0] * (3 * 457 // 4) + [80.0] * (457 - 3 * 457 // 4)
                )
            },
        },
        training_run_id="run-124",
        launch_config_sha256="4" * 64,
        checkpoint_path="/tmp/final-collapsed",
        checkpoint_sha256="5" * 64,
        model_weights_sha256="6" * 64,
    )
    assert collapsed["passed"] is False
    assert collapsed["performance_threshold_pass"] is False
    assert collapsed["selection_effect"] == "audit_only_include_all_fixed_seeds_v1"
    assert collapsed["included_in_population"] is True
    assert collapsed["seed_replacement_allowed"] is False
    assert collapsed["retraining_until_pass_allowed"] is False


def test_official_history_file_verification_is_runtime_free_and_detects_tamper(
    tmp_path: Path,
) -> None:
    from experiments.overcooked_v2.official.path_c_family_pool_training import (
        checkpoint_artifact_sha256,
        validate_checkpoint_history,
    )

    rows = []
    for index in range(3):
        checkpoint = tmp_path / f"checkpoint_{index}"
        checkpoint.mkdir()
        (checkpoint / "payload.bin").write_bytes(
            f"scheduled-checkpoint-{index}".encode("utf-8")
        )
        rows.append(
            {
                "checkpoint_index": index,
                "update_step": index + 1,
                "effective_environment_steps": (index + 1) * 100,
                "path": str(checkpoint),
                "format": "orbax_pytree_directory_v1",
                "parameter_tree_path": ["params"],
                "selection_rule": "mechanical_official_update_schedule_v1",
                "checkpoint_sha256": checkpoint_artifact_sha256(checkpoint),
                "model_weights_sha256": f"{index + 10:064x}",
            }
        )
    assert len(
        validate_checkpoint_history(
            rows,
            final_weights_sha256=rows[-1]["model_weights_sha256"],
            verify_files=True,
        )
    ) == 3
    validation_source = inspect.getsource(validate_checkpoint_history)
    for forbidden in (
        "overcooked_v2_experiments_adapter",
        "path_c_official_artifact",
        "restore_official_checkpoint",
        "import jax",
    ):
        assert forbidden not in validation_source
    (tmp_path / "checkpoint_1" / "payload.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content hash changed"):
        validate_checkpoint_history(
            rows,
            final_weights_sha256=rows[-1]["model_weights_sha256"],
            verify_files=True,
        )
