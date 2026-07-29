from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.path_c.experiment import (
    METHOD_VERSION,
    Population,
    PopulationEntry,
    load_config,
    load_population,
    write_population,
)
from src.path_c.storage import (
    CompleteConsoleLog,
    ensure_run_identity,
    evaluation_identity,
    read_array_chunks,
    read_run_identity,
    restore_checkpoint_step,
    training_identity,
    write_array_chunks,
    write_jsonl,
    write_parquet,
)
from src.path_c.evaluation import summarize_responsibility_records

ROOT = Path(__file__).resolve().parents[3]
SIMPLE_CONFIG = ROOT / "experiments/overcooked_v2/configs/path_c_simple.yaml"
WIDE_CONFIG = ROOT / "experiments/overcooked_v2/configs/path_c_wide.yaml"


def test_run_kind_selects_registered_budget() -> None:
    mechanical = load_config(SIMPLE_CONFIG, run_kind="mechanical")
    development = load_config(SIMPLE_CONFIG, run_kind="development")
    formal = load_config(SIMPLE_CONFIG, run_kind="formal")
    assert mechanical.environment.num_envs == 4
    assert mechanical.training.environment_steps == 1_600
    assert mechanical.training.minibatches_per_epoch == 1
    assert mechanical.training.checkpoint_interval_environment_steps == 1_600
    assert development.environment.num_envs == 32
    assert development.training.environment_steps == 1_228_800
    assert development.training.minibatches_per_epoch == 8
    assert formal.environment.num_envs == 250
    assert formal.training.environment_steps == 11_000_000
    assert formal.training.minibatches_per_epoch == 50
    assert development.training.behavior_exploration_mix == 0.25
    assert development.training.behavior_uniform_floor == 0.02
    assert development.training.retrace_lambda == 0.9
    assert development.model.uncertainty_penalty == 1.0
    assert development.kl.bisection_iterations == 24
    assert development.evaluation.response_policy_tv_minimum == 0.001
    assert METHOD_VERSION == "path_c_v4_4_retrace_calibrated_control_r1"


def test_layout_configs_differ_only_by_layout() -> None:
    simple = load_config(SIMPLE_CONFIG, run_kind="development")
    wide = load_config(WIDE_CONFIG, run_kind="development")
    assert simple.environment.layout == "test_time_simple"
    assert wide.environment.layout == "test_time_wide"
    assert replace(simple, environment=wide.environment) == wide


def test_unknown_configuration_field_fails(tmp_path: Path) -> None:
    import yaml

    payload = yaml.safe_load(SIMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["training"]["silent_limit"] = 10
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path, run_kind="development")


def test_run_identity_is_single_literal_source(tmp_path: Path) -> None:
    config = load_config(SIMPLE_CONFIG, run_kind="development")
    expected = training_identity(
        config=config,
        seed=100,
        outer_unit_id=0,
        reference_checkpoint=tmp_path / "reference",
        partner_checkpoints=(tmp_path / "partner-a", tmp_path / "partner-b"),
    )
    ensure_run_identity(tmp_path / "run", expected)
    assert read_run_identity(tmp_path / "run") == expected

    changed = dict(expected)
    changed["outer_unit_id"] = 1
    with pytest.raises(RuntimeError, match="different experiment"):
        ensure_run_identity(tmp_path / "run", changed)


def test_evaluation_identity_binds_population_and_seed(tmp_path: Path) -> None:
    config = load_config(SIMPLE_CONFIG, run_kind="development")
    population = Population(
        name="development",
        layout="test_time_simple",
        evaluation_kind="standard_matrix",
        entries=tuple(
            PopulationEntry(index, tmp_path / f"run-{index}") for index in range(10)
        ),
    )
    first = evaluation_identity(
        config=config, seed=17, population=population.to_mapping()
    )
    second = evaluation_identity(
        config=config, seed=18, population=population.to_mapping()
    )
    assert first["method"] == METHOD_VERSION
    assert first != second


def test_population_records_training_run_directories_only(tmp_path: Path) -> None:
    population = Population(
        name="posterior-use",
        layout="test_time_simple",
        evaluation_kind="standard_matrix",
        entries=tuple(
            PopulationEntry(index, tmp_path / f"run-{index}") for index in range(10)
        ),
    )
    path = write_population(tmp_path / "population.json", population)
    assert load_population(path) == population
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["policies"][0]) == {"outer_unit_id", "run_directory"}


def test_jsonl_and_array_chunks_are_lossless(tmp_path: Path) -> None:
    long_text = "响应🙂\n制表符\t引号\"" * 100
    rows = ({"index": index, "text": long_text + str(index)} for index in range(257))
    path = write_jsonl(tmp_path / "records.jsonl", rows)
    restored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(restored) == 257
    assert restored[-1]["text"] == long_text + "256"

    values = np.arange(2_317 * 7, dtype=np.float32).reshape(2_317, 7)
    chunks = write_array_chunks(
        tmp_path / "arrays", name="trajectory", values=values, rows_per_chunk=97
    )
    np.testing.assert_array_equal(read_array_chunks(chunks), values)


def test_console_log_keeps_exception_trace(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="完整异常消息"):
        with CompleteConsoleLog(tmp_path / "logs"):
            raise RuntimeError("完整异常消息")
    stderr = (tmp_path / "logs/stderr.log").read_text(encoding="utf-8")
    assert "Traceback" in stderr
    assert "RuntimeError: 完整异常消息" in stderr


def test_exact_checkpoint_restore_never_selects_a_different_step() -> None:
    pytest.importorskip("orbax.checkpoint")

    class LiteralManager:
        def all_steps(self):
            return (100, 200)

        def restore(self, step, *, args):
            del args
            return {"restored_step": step}

    manager = LiteralManager()
    assert restore_checkpoint_step(manager, step=100) == {"restored_step": 100}
    with pytest.raises(FileNotFoundError, match="1228800"):
        restore_checkpoint_step(manager, step=1_228_800)


def test_audit_parameter_snapshot_detects_in_memory_change() -> None:
    from types import SimpleNamespace
    from experiments.overcooked_v2.counterfactual_audit_app import (
        deployment_parameter_snapshot,
    )

    deployment = SimpleNamespace(
        reference_params={"weight": np.asarray([1.0, 2.0])},
        online_params={"weight": np.asarray([3.0])},
        head_params={"weight": np.asarray([4.0])},
        codebook={"embedding": np.asarray([[5.0]])},
        log_temperature=np.asarray(0.0),
        generic_log_temperature=np.asarray(0.0),
    )
    before = deployment_parameter_snapshot(deployment)
    deployment.reference_params["weight"][0] = 9.0
    after = deployment_parameter_snapshot(deployment)
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(before, after, strict=True)
    )



def test_responsibility_rows_preserve_every_epoch_lane_and_slot() -> None:
    from types import SimpleNamespace
    from experiments.overcooked_v2.training_app import _responsibility_rows

    assignments = SimpleNamespace(
        responsibilities=np.asarray([[0.25, 0.75], [0.6, 0.4]], dtype=np.float32),
        td_energies=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        response_energies=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        reward_energies=np.asarray([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
        next_q_use_energies=np.asarray([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32),
        next_q_mask_energies=np.asarray([[17.0, 18.0], [19.0, 20.0]], dtype=np.float32),
        next_reference_energy=np.asarray([21.0, 22.0], dtype=np.float32),
        bootstrap_mask=np.asarray([[True, False], [True, True]]),
        importance_ratio_mean=np.asarray([0.9, 1.1], dtype=np.float32),
        trace_coefficient_mean=np.asarray([0.8, 0.7], dtype=np.float32),
    )
    records = {
        "partner_members": np.asarray([[3, 4]], dtype=np.int32),
        "episode_ids": np.asarray([[100, 101]], dtype=np.int64),
    }
    rows = list(
        _responsibility_rows(assignments, records, update_count=7, epoch=2)
    )
    assert len(rows) == 4
    expected = {
        "update_count": 7,
        "epoch": 2,
        "environment_index": 0,
        "episode_id": 100,
        "audit_partner_member": 3,
        "slot": 0,
        "responsibility": 0.25,
        "td_energy": 1.0,
        "response_nll_energy": 5.0,
        "reward_nll_energy": 9.0,
        "next_q_use_nll_energy": 13.0,
        "next_q_mask_nll_energy": 17.0,
        "next_reference_mse_energy": 21.0,
        "bootstrap_available": True,
        "mean_importance_ratio": 0.9,
        "mean_trace_coefficient": 0.8,
    }
    assert set(rows[0]) == set(expected)
    for key, value in expected.items():
        if isinstance(value, float):
            assert rows[0][key] == pytest.approx(value)
        else:
            assert rows[0][key] == value


def test_responsibility_summary_uses_literal_recorded_masses_and_returns() -> None:
    rows = (
        {
            "update_count": 1,
            "epoch": 0,
            "environment_index": 2,
            "episode_id": 7,
            "audit_partner_member": 4,
            "slot": 0,
            "responsibility": 0.75,
            "td_energy": 1.0,
        },
        {
            "update_count": 1,
            "epoch": 0,
            "environment_index": 2,
            "episode_id": 7,
            "audit_partner_member": 4,
            "slot": 1,
            "responsibility": 0.25,
            "td_energy": 3.0,
        },
    )
    summary = summarize_responsibility_records(rows, {(1, 2, 7): 8.0})
    assert summary["assignment_count"] == 1
    assert summary["mean_td_energy_margin"] == 2.0
    assert summary["slots"]["0"]["responsibility_mass"] == 0.75
    assert summary["slots"]["1"]["responsibility_mass"] == 0.25
    assert summary["slots"]["0"]["responsibility_weighted_mean_return"] == 8.0
    assert (
        summary["slots"]["0"][
            "responsibility_weighted_return_standard_deviation"
        ]
        == 0.0
    )
    assert summary["slots"]["0"]["responsibility_weighted_return_minimum"] == 8.0
    assert summary["slots"]["0"]["responsibility_weighted_return_maximum"] == 8.0
    assert summary["slots"]["0"]["dominant_partner_counts"] == {"4": 1}
    assert summary["historical_task_stage"] == "not_collected"


def test_legacy_v44_response_reader_maps_only_the_registered_columns(
    tmp_path: Path,
) -> None:
    from experiments.overcooked_v2.counterfactual_audit_app import (
        read_legacy_v44_response_rows,
    )
    from src.path_c.evaluation import ResponseContrastRow

    current = ResponseContrastRow(
        pairing_id="00_to_00",
        episode_index=0,
        episode_seed=1,
        triggered=False,
        trigger_step=None,
        trigger_tolerance=None,
        predicted_response_effect=None,
        predicted_policy_cost=None,
        predicted_net_effect=None,
        predicted_regularized_net_effect=None,
        predicted_policy_total_variation=None,
        predicted_policy_mediated_effect=None,
        predicted_policy_gain_lower_score=None,
        predicted_policy_gain_uncertainty=None,
        predicted_next_policy_total_variation=None,
        maximum_action_net_value=None,
        maximum_action_policy_mediated_gain=None,
        maximum_action_predicted_gain_lower_score=None,
        executed_action_net_value=None,
        executed_action_response_value=None,
        executed_action_policy_mediated_gain=None,
        executed_action_predicted_gain_lower_score=None,
        executed_action_policy_gain_uncertainty=None,
        executed_action_expected_next_policy_tv=None,
        executed_action=None,
        maximum_net_action=None,
        post_response_belief_l1=None,
        first_left_action_difference_step=None,
        first_observation_difference_step=None,
        first_response_code_difference_step=None,
        first_reward_difference_step=None,
        left_action_difference_count=None,
        observation_difference_count=None,
        response_code_difference_count=None,
        reward_difference_count=None,
        environment_steps=1_200,
        a1_raw_return=0.0,
        a1_correct_delivery_count=0,
        a1_wrong_delivery_count=0,
        a1_indicator_activation_count=0,
        a2_mask_raw_return=0.0,
        a2_mask_correct_delivery_count=0,
        a2_mask_wrong_delivery_count=0,
        a2_mask_indicator_activation_count=0,
        a2_use_raw_return=0.0,
        a2_use_correct_delivery_count=0,
        a2_use_wrong_delivery_count=0,
        a2_use_indicator_activation_count=0,
    ).to_mapping()
    legacy_names = {
        "predicted_policy_gain_lower_score": "predicted_policy_mediated_effect_lcb",
        "maximum_action_predicted_gain_lower_score": "maximum_action_policy_mediated_gain_lcb",
        "executed_action_predicted_gain_lower_score": "executed_action_policy_mediated_gain_lcb",
    }
    for new, old in legacy_names.items():
        current[old] = current.pop(new)
    path = write_parquet(tmp_path / "legacy.parquet", [current])
    restored = read_legacy_v44_response_rows(path)
    assert len(restored) == 1
    assert restored[0].executed_action_predicted_gain_lower_score is None

    current_schema = dict(current)
    for new, old in legacy_names.items():
        current_schema[new] = current_schema.pop(old)
    write_parquet(tmp_path / "current.parquet", [current_schema])
    with pytest.raises(ValueError, match="registered schema"):
        read_legacy_v44_response_rows(tmp_path / "current.parquet")


def test_legacy_v44_panel_readers_map_only_registered_columns(
    tmp_path: Path,
) -> None:
    from experiments.overcooked_v2.counterfactual_audit_app import (
        read_legacy_v44_panel_episode_rows,
        read_legacy_v44_panel_trigger_decisions,
    )

    episode_path = write_parquet(
        tmp_path / "episodes.parquet",
        [
            {
                "episode_index": 3,
                "episode_seed": 17,
                "positive_policy_gain_lcb_count": 1,
            }
        ],
    )
    decision = {
        "episode_index": 3,
        "episode_seed": 17,
        "step": 9,
        "predicted_policy_mediated_effect_lcb": 0.25,
        "executed_action_policy_mediated_gain_lcb": 0.125,
    }
    decision_path = write_jsonl(
        tmp_path / "decisions.jsonl", [decision]
    )
    episodes = read_legacy_v44_panel_episode_rows(episode_path)
    decisions = read_legacy_v44_panel_trigger_decisions(
        decision_path, {3: 9}
    )
    assert episodes[0]["positive_gain_lower_score_count"] == 1
    assert decisions[3]["predicted_policy_gain_lower_score"] == 0.25
    assert decisions[3]["executed_action_predicted_gain_lower_score"] == 0.125

    write_parquet(
        tmp_path / "current_episodes.parquet",
        [
            {
                "episode_index": 3,
                "episode_seed": 17,
                "positive_gain_lower_score_count": 1,
            }
        ],
    )
    with pytest.raises(ValueError, match="unexpected schema"):
        read_legacy_v44_panel_episode_rows(
            tmp_path / "current_episodes.parquet"
        )


def test_panel_production_replay_requires_every_decision_field_exact(
    tmp_path: Path,
) -> None:
    from experiments.overcooked_v2.counterfactual_audit_app import (
        _validate_and_collect_panel_decisions,
    )

    current_rows = []
    legacy_rows = []
    for step in range(2):
        for episode in range(2):
            value = float(10 * step + episode)
            current = {
                "partner_index": 0,
                "partner_label": "partner_00",
                "deployment_mode": "posterior_use",
                "episode_index": episode,
                "episode_seed": 100 + episode,
                "step": step,
                "ego_action": episode,
                "partner_action": step,
                "response_code": step + episode,
                "reference_logits": [value, value + 1.0],
                "execution_logits": [value + 2.0, value + 3.0],
                "slot_log_belief": [-0.5, -1.5],
                "belief_entropy": value + 4.0,
                "value_class_count": 2,
                "predicted_policy_gain_lower_score": value + 5.0,
                "predicted_policy_gain_uncertainty": value + 6.0,
                "predicted_next_policy_total_variation": value + 7.0,
                "executed_action_predicted_gain_lower_score": value + 8.0,
                "executed_action_policy_gain_uncertainty": value + 9.0,
                "executed_action_expected_next_policy_tv": value + 10.0,
            }
            legacy = dict(current)
            legacy["predicted_policy_mediated_effect_lcb"] = legacy.pop(
                "predicted_policy_gain_lower_score"
            )
            legacy["executed_action_policy_mediated_gain_lcb"] = legacy.pop(
                "executed_action_predicted_gain_lower_score"
            )
            current_rows.append(current)
            legacy_rows.append(legacy)
    path = write_jsonl(tmp_path / "decisions.jsonl", legacy_rows)
    evidence = _validate_and_collect_panel_decisions(
        frozen_path=path,
        replayed=iter(current_rows),
        episode_steps=2,
        episode_count=2,
    )
    np.testing.assert_array_equal(
        evidence.ego_action, np.asarray([[0, 1], [0, 1]])
    )
    assert evidence.reference_logits.shape == (2, 2, 2)

    changed = [dict(row) for row in current_rows]
    changed[2]["response_code"] = 99
    with pytest.raises(RuntimeError, match="response_code"):
        _validate_and_collect_panel_decisions(
            frozen_path=path,
            replayed=iter(changed),
            episode_steps=2,
            episode_count=2,
        )


def test_reconstruction_separates_two_partner_state_types(
    tmp_path: Path,
) -> None:
    from src.path_c.storage import read_parquet

    rows = [
        {
            "trigger_id": "self",
            "pre_partner_policy_state": {"carry": [1.0, 2.0]},
            "post_partner_policy_state": {"carry": [3.0, 4.0]},
            "pre_fixed_partner_carry": None,
            "post_fixed_partner_carry": None,
        },
        {
            "trigger_id": "fixed",
            "pre_partner_policy_state": None,
            "post_partner_policy_state": None,
            "pre_fixed_partner_carry": [5.0, 6.0],
            "post_fixed_partner_carry": [7.0, 8.0],
        },
    ]
    path = write_parquet(tmp_path / "reconstruction.parquet", rows)
    restored = read_parquet(path)
    assert restored[0]["pre_partner_policy_state"] == {"carry": [1.0, 2.0]}
    assert restored[1]["pre_fixed_partner_carry"] == [5.0, 6.0]
