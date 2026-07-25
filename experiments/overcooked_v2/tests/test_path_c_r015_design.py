"""R015 独立设计数据选择与待签冻结清单的静态测试定义。"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.overcooked_v2 import path_c_r015_design as design
from experiments.overcooked_v2.path_c_r015_pilot import run_r015_pilot
from experiments.overcooked_v2.path_c_r015_controller import (
    HiddenStateParticleV1,
    NestedHiddenStateBranchScheduleV1,
    StratifiedParticleBeliefV1,
)
from experiments.overcooked_v2.scripts import (
    build_r015_freeze_manifest as freeze_manifest_cli,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_design_and_pilot_entrypoints_fail_closed_without_runtime_authorization():
    with pytest.raises(PermissionError, match="recorded authorization"):
        design.run_r015_design({})
    with pytest.raises(PermissionError, match="recorded authorization"):
        run_r015_pilot({})


def test_design_entrypoint_rejects_effect_fields_before_any_path_read():
    with pytest.raises(ValueError, match="forbidden effect field"):
        design.run_r015_design(
            {
                "_execution_authorized": True,
                "Delta_net": 0.0,
            }
        )


def test_design_history_jsonl_recovers_only_a_truncated_final_record(tmp_path):
    path = tmp_path / "design_histories.jsonl"
    path.write_bytes(b'{"design_sequence_index":1}\n{"design_sequence_')
    records, needs_repair = design._load_recoverable_jsonl_prefix(path)
    assert records == [{"design_sequence_index": 1}]
    assert needs_repair is True
    design._rewrite_jsonl_atomic(path, records)
    assert design._load_complete_jsonl(path) == records


def test_design_history_jsonl_rejects_a_corrupt_completed_line(tmp_path):
    path = tmp_path / "design_histories.jsonl"
    path.write_bytes(b'{"design_sequence_index":1}\nnot-json\n')
    with pytest.raises(ValueError, match="corrupt completed line"):
        design._load_recoverable_jsonl_prefix(path)


def test_filter_device_equivalence_entrypoint_fails_closed_without_authorization():
    assert design.FILTER_DEVICE_EQUIVALENCE_CANDIDATE == (
        256,
        "adaptive_ess_below_half_v1",
    )
    with pytest.raises(PermissionError, match="recorded authorization"):
        design.run_r015_filter_device_equivalence({})


def test_filter_device_equivalence_keeps_the_verified_fixed_batch_step_path():
    source = inspect.getsource(design.run_r015_filter_device_equivalence)
    assert "_reference_filter_group_trace(" in source
    assert source.index("replay_filter_candidate_device(") < source.index(
        "_reference_filter_group_trace("
    )
    grouped = inspect.getsource(design._reference_filter_group_trace)
    assert "executor.update_beliefs_v1_diagnostic(" in grouped
    assert "close_zero_support=True" in grouped
    assert "captured_outputs" in grouped
    assert "selected_source_index_trace" in grouped
    assert "active_indices = [" in grouped
    assert "tuple(contexts[index][\"belief\"] for index in active_indices)" in grouped
    assert "if active_indices:" in grouped
    assert (
        "expected_outputs = len(active_indices) * 4 * particles_per_prototype"
        in grouped
    )


def test_filter_device_prefix_diagnostic_cannot_be_mistaken_for_the_full_gate():
    source = inspect.getsource(design.run_r015_filter_device_equivalence)
    assert "FILTER_DEVICE_PREFIX_DIAGNOSTIC_SCHEMA" in source
    assert '"diagnostic_only": diagnostic_only' in source
    assert "environment_step_limit=compared_steps" in source
    assert "return_state_diagnostics=diagnostic_only" in source
    assert '"partner_recurrent_state_before"' in source
    assert '"agent_0_observation_before"' in source
    assert "if not proof[\"passed\"] and not diagnostic_only:" in source
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_path_c_r015_design.py"
    ).read_text(encoding="utf-8")
    assert "--filter-device-diagnostic-steps" in launcher
    assert "Filter diagnostics require an explicit output path." in launcher


def test_filter_candidate_formal_path_uses_one_device_scan_not_host_step_dispatch():
    from experiments.overcooked_v2.official.r015_runtime_bridge import (
        OCV2R015ProductionBackendV1,
    )

    source = inspect.getsource(design.run_r015_design)
    assert "replay_filter_candidate_device" in source
    assert "_replay_filter_episode_group(" not in source
    bridge_source = (
        Path(__file__).resolve().parents[1]
        / "official"
        / "r015_runtime_bridge.py"
    ).read_text(encoding="utf-8")
    assert "jax.jit(compiled_filter_scan)" in bridge_source
    assert "jax.lax.scan(" in bridge_source
    assert "adapter.env.step_env" in bridge_source
    assert "jax.vmap(" in bridge_source
    assert "def actor_apply_by_lane(" not in bridge_source
    assert "def freeze_closed_lanes(" in bridge_source
    assert "effective_due = due & ~next_closed[:, None]" in bridge_source
    assert "def _compiled_actor_boundary(" in bridge_source
    assert "inline=False" in bridge_source
    assert bridge_source.count("self._compiled_actor_boundary(") >= 2

    # 过滤候选只估计四个伙伴原型的后验；五个延续策略成员只在执行和规划阶段使用。
    filter_scan_source = inspect.getsource(
        OCV2R015ProductionBackendV1.run_offline_filter_scan_v2
    )
    assert "initial_continuation_states" not in filter_scan_source
    assert "selected_continuation_states" not in filter_scan_source
    assert "continuation_logit_checksums" not in filter_scan_source
    assert "continuation_params" not in filter_scan_source


def test_design_history_collection_uses_one_scan_per_partner_batch():
    from experiments.overcooked_v2.official.r015_runtime_bridge import (
        OCV2R015ProductionBackendV1,
    )

    source = inspect.getsource(
        OCV2R015ProductionBackendV1.collect_design_history_batch
    )
    assert "jax.lax.scan(" in source
    assert "jax.vmap(adapter.env.step_env)" in source
    assert "compiled = jax.jit(compiled_batch)" in source
    assert "jax.device_get((initial_observation, trace))" in source
    assert '"host_sync_inside_environment_loop": False' in source
    assert "step_joint_from_state" not in source
    assert "for step in range(400)" in source  # 只在调用前构造登记动作键。
    assert source.index("for step in range(400)") < source.index(
        "def compiled_batch("
    )


def test_design_history_device_path_preserves_registered_seed_and_action_keys():
    from experiments.overcooked_v2.official.r015_runtime_bridge import (
        OCV2R015ProductionBackendV1,
    )

    source = inspect.getsource(
        OCV2R015ProductionBackendV1.collect_design_history_batch
    )
    assert 'derive_controller_key(root_key, "ego_action", step)' in source
    assert 'derive_controller_key(root_key, "partner_action", step)' in source
    assert "derive_ocv2_execution_seed(value)" in source
    assert "reset_pairs = jax.vmap(jax.random.split)(reset_roots)" in source
    assert "snapshot_keys = reset_pairs[:, 0]" in source
    assert "adapter.env.reset" in source
    assert "environment_pairs = jax.vmap(jax.random.split)(environment_keys)" in source
    assert "environment_pairs[:, 1]" in source


def test_design_entry_collects_four_fixed_batches_and_records_throughput():
    source = inspect.getsource(design.run_r015_design)
    assert "backend.collect_design_history_batch(" in source
    assert "_collect_design_history(" not in source
    assert 'batch_dir = output_dir / "design_history_batches"' in source
    assert '"path_c_r015_design_history_batch_v1"' in source
    assert '"r015_design_history_jit_scan_vmap_v1"' in source
    assert 'throughput.get("true_environment_transitions") != 8000' in source
    assert 'throughput.get("active_lane_batch_width") != 20' in source
    assert '"design_history_collection_report.json"' in source
    assert '"true_environment_transitions": total_history_transitions' in source
    assert '"compiled_batch_calls": sum(' in source
    assert '"host_sync_inside_environment_loop": False' in source
    assert "Delta_net" not in source
    assert "Delta_response" not in source
    assert "Delta_cost" not in source


def test_design_history_batch_resume_is_bound_before_canonical_append():
    source = inspect.getsource(design.run_r015_design)
    assert 'batch_binding_sha256 = canonical_sha256(batch_binding)' in source
    assert "if batch_path.is_file():" in source
    assert 'batch.get("batch_binding") != batch_binding' in source
    assert 'batch.get("records_sha256") != canonical_sha256(batch_records)' in source
    assert source.index("_write_json_atomic(batch_path, batch)") < source.index(
        "_append_jsonl_record(histories_path, record)"
    )
    assert '"design_source_sha256": design_source_sha256' in source
    assert '"runtime_bridge_source_sha256": bridge_source_sha256' in source


def test_filter_device_keys_are_derived_inside_the_scan_without_step_seed_tensors():
    from experiments.overcooked_v2.official.r015_runtime_bridge import (
        OCV2R015ProductionBackendV1,
    )

    source = inspect.getsource(OCV2R015ProductionBackendV1.run_offline_filter_scan_v2)
    bridge_source = (
        Path(__file__).resolve().parents[1]
        / "official"
        / "r015_runtime_bridge.py"
    ).read_text(encoding="utf-8")
    assert "jax.random.fold_in" in source
    assert "environment_seeds = np.empty" not in source
    assert "partner_seeds = np.empty_like" not in source
    assert "all_environment_seeds" not in source
    assert "all_partner_seeds" not in source
    assert "r015_filter_device_fold_in_keys_v2" in bridge_source


def test_filter_partner_networks_only_process_their_own_prototype_particles():
    from experiments.overcooked_v2.official.r015_runtime_bridge import (
        OCV2R015ProductionBackendV1,
    )

    source = inspect.getsource(OCV2R015ProductionBackendV1.run_offline_filter_scan_v2)
    assert "prototype_actor_slot_count = lane_count * particles_per_prototype" in source
    assert "policy.initial_state(prototype_actor_slot_count)" in source
    assert "policy.initial_state(total_slots)" not in source
    assert "partner_candidates = tuple" not in source
    assert "for params in partner_params" not in source


def test_filter_microbatch_width_keeps_4096_parent_particle_slots():
    assert design.FILTER_PARENT_SLOT_TARGET == 4096
    assert design.filter_lane_batch_size(64) == 16
    assert design.filter_lane_batch_size(128) == 8
    assert design.filter_lane_batch_size(256) == 4
    for particles_per_prototype in (64, 128, 256):
        assert (
            design.filter_lane_batch_size(particles_per_prototype)
            * 4
            * particles_per_prototype
            == design.FILTER_PARENT_SLOT_TARGET
        )
    with pytest.raises(ValueError, match="divide the fixed parent-slot target"):
        design.filter_lane_batch_size(96)


def test_logarithmic_systematic_resampling_uses_strict_cdf_comparison():
    cumulative = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    positions = np.asarray(
        [0.0, 0.25, np.nextafter(0.25, 1.0), 0.5, 0.75, 1.0],
        dtype=np.float64,
    )
    indices = design.systematic_resample_indices_logarithmic(
        cumulative,
        positions,
    )
    assert np.array_equal(indices, np.asarray([0, 0, 1, 1, 2, 3]))
    source = inspect.getsource(design.systematic_resample_indices_logarithmic)
    assert "side=\"left\"" in source or "side='left'" in source
    assert "[..., :, None]" not in source
    assert "[..., None, :]" not in source


def test_filter_microbatch_merge_preserves_lane_order_and_throughput_counts():
    chunks = [
        {
            "lane_start": 2,
            "lane_count": 2,
            "lane_results": [
                {"lane_index": 2, "value": "c"},
                {"lane_index": 3, "value": "d"},
            ],
            "throughput": {
                "true_particle_environment_transitions": 200,
                "compiled_batch_calls": 2,
                "jit_compilations": 1,
                "active_lane_count": 2,
                "active_particle_batch_width": 512,
                "wall_seconds": 2.0,
                "host_sync_inside_environment_loop": False,
            },
        },
        {
            "lane_start": 0,
            "lane_count": 2,
            "lane_results": [
                {"lane_index": 0, "value": "a"},
                {"lane_index": 1, "value": "b"},
            ],
            "throughput": {
                "true_particle_environment_transitions": 100,
                "compiled_batch_calls": 1,
                "jit_compilations": 1,
                "active_lane_count": 2,
                "active_particle_batch_width": 512,
                "wall_seconds": 1.0,
                "host_sync_inside_environment_loop": False,
            },
        },
    ]
    merged = design.merge_filter_microbatch_results(
        chunks,
        expected_lane_count=4,
    )
    assert [row["lane_index"] for row in merged["lane_results"]] == [0, 1, 2, 3]
    assert merged["throughput"] == {
        "true_particle_environment_transitions": 300,
        "compiled_batch_calls": 3,
        "jit_compilations": 2,
        "active_lane_count": 4,
        "active_particle_batch_width": 512,
        "wall_seconds": 3.0,
        "true_transitions_per_second": 100.0,
        "host_sync_inside_environment_loop": False,
    }

    duplicate_lane = copy.deepcopy(chunks)
    duplicate_lane[0]["lane_results"][0]["lane_index"] = 1
    with pytest.raises(ValueError, match="exactly once in canonical order"):
        design.merge_filter_microbatch_results(
            duplicate_lane,
            expected_lane_count=4,
        )


def test_planning_uses_one_branch_head_update_then_scans_the_remaining_suffix():
    from experiments.overcooked_v2.path_c_r015_full_horizon import (
        OCV2R015FullHorizonExecutorV1,
    )

    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.rollout_planning_length_bucket_v2
    )
    assert "run_planning_branch_head_batch_v2" in source
    assert 'key_contract="filter_transition_v1"' not in source
    assert "total_steps=1" in source
    assert "total_steps=remaining_steps - 1" in source
    assert "suffix_outputs" in source


def test_filter_equivalence_digest_is_dtype_shape_and_byte_sensitive():
    base = np.asarray([[0.25, 0.75]], dtype=np.float32)
    assert design._array_sha256(base, domain="fixture") == design._array_sha256(
        base.copy(), domain="fixture"
    )
    assert design._array_sha256(base, domain="fixture") != design._array_sha256(
        base.astype(np.float64), domain="fixture"
    )
    assert design._array_sha256(base, domain="fixture") != design._array_sha256(
        base.reshape((2, 1)), domain="fixture"
    )


def test_filter_equivalence_uses_bytes_and_reports_the_first_difference():
    reference = np.asarray([[0.0, 0.5]], dtype=np.float64)
    device = np.asarray([[-0.0, 0.5]], dtype=np.float64)
    assert not design._arrays_bitwise_equal(reference, device)
    mismatch = design._first_bitwise_mismatch(reference, device)
    assert mismatch == {
        "index": [0, 0],
        "environment_step": 1,
        "left_bytes_hex": "0000000000000000",
        "right_bytes_hex": "0000000000000080",
        "left_float_hex": "0x0.0p+0",
        "right_float_hex": "-0x0.0p+0",
    }


def test_filter_evidence_canonicalizes_only_roundoff_above_the_ess_upper_bound():
    from experiments.overcooked_v2.path_c_r015_full_horizon import (
        _canonical_ess_fraction_for_evidence,
    )

    assert _canonical_ess_fraction_for_evidence(0.25) == 0.25
    assert _canonical_ess_fraction_for_evidence(1.0 + 2.0e-14) == 1.0
    with pytest.raises(ValueError, match="mathematical range"):
        _canonical_ess_fraction_for_evidence(0.0)
    with pytest.raises(ValueError, match="mathematical range"):
        _canonical_ess_fraction_for_evidence(1.0 + 2.0e-12)


def test_design_protocol_binds_local_three_class_one_step_response():
    protocol = design.load_r015_design_protocol(
        CONFIG_DIR / "path_c_r015_design_data_protocol.yaml"
    )
    assert protocol["environment"]["history_collection_action_rule"] == (
        "ego_seed100_official_categorical_no_probe_v1"
    )
    assert protocol["role_isolation"]["filter_initialization_coordinates"] == [
        "filter_repeat_namespace",
        "particles_per_prototype",
        "resampling_timing",
        "design_sequence_index",
    ]
    assert protocol["response_summary"] == {
        "implementation_constant": "REGISTERED_LOCAL_RESPONSE_SPEC",
        "response_classes": ["visible", "unseen", "local_non_agent_change"],
        "latency_bin_upper_bounds": [1],
        "source_fields": "adjacent_official_local_observations_only",
        "evaluator_partner_action_allowed": False,
    }


def _sha(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _posterior_pair(tv: float):
    left = {"p0": 0.25, "p1": 0.25, "p2": 0.25, "p3": 0.25}
    right = {"p0": 0.25 + tv, "p1": 0.25 - tv, "p2": 0.25, "p3": 0.25}
    return left, right


def test_filter_s1_s2_s3_hand_boundaries_use_registered_nearest_rank_rule():
    passing = design.summarize_filter_statistics(
        zero_support_closed=[True, True] + [False] * 98,
        posterior_pairs=[_posterior_pair(0.0)] * 8
        + [_posterior_pair(0.05), _posterior_pair(0.06)],
        ess_fractions=[0.2] + [1.0] * 9,
    )
    assert passing["s1_zero_support_close_rate"] == pytest.approx(0.02)
    assert passing["s2_posterior_tv_p90"] == pytest.approx(0.05)
    assert passing["s3_ess_fraction_p10"] == pytest.approx(0.2)
    assert passing["passed"] is True

    s1_fail = design.summarize_filter_statistics(
        zero_support_closed=[True, True, True] + [False] * 97,
        posterior_pairs=[_posterior_pair(0.0)] * 10,
        ess_fractions=[0.2] * 10,
    )
    s2_fail = design.summarize_filter_statistics(
        zero_support_closed=[False] * 100,
        posterior_pairs=[_posterior_pair(0.0)] * 8
        + [_posterior_pair(0.050001)] * 2,
        ess_fractions=[0.2] * 10,
    )
    s3_fail = design.summarize_filter_statistics(
        zero_support_closed=[False] * 100,
        posterior_pairs=[_posterior_pair(0.0)] * 10,
        ess_fractions=[0.199999] + [1.0] * 9,
    )
    assert s1_fail["s1_pass"] is False
    assert s2_fail["s2_pass"] is False
    assert s3_fail["s3_pass"] is False


def _filter_result(particles: int, timing: str, *, passed: bool):
    return {
        "schema_version": "path_c_r015_filter_candidate_result_v2",
        "selection_rule_id": design.FILTER_SELECTION_RULE_ID,
        "particles_per_prototype": particles,
        "filter_algorithm_id": design.FILTER_ALGORITHM_ID,
        "resampling_algorithm": design.FILTER_RESAMPLING_ALGORITHM_ID,
        "resampling_timing": timing,
        "candidate_evaluation_status": design.FILTER_CANDIDATE_COMPLETE_STATUS,
        "total_episode_count": 80,
        "processed_episode_count": 80,
        "zero_support_close_count": 0 if passed else 3,
        "s1_zero_support_close_rate_lower_bound": 0.0 if passed else 0.0375,
        "conditioned_opening_prefix_cache_binding_sha256": "2" * 64,
        "conditioned_opening_prefix_sha256_by_particle_count": {
            str(particles): "3" * 64,
        },
        "conditioned_opening_attempt_sha256_by_particle_count": {
            str(particles): "5" * 64,
        },
        "conditioned_opening_prefix_scope": "all_design_episodes",
        "design_episode_set_sha256": "1" * 64,
        "s1_zero_support_close_rate": 0.0 if passed else 0.0375,
        "s2_posterior_tv_p90": 0.0,
        "s3_ess_fraction_p10": 1.0,
        "s1_pass": passed,
        "s2_pass": True,
        "s3_pass": True,
        "s2_status": "computed",
        "s3_status": "computed",
        "passed": passed,
    }


def _s1_stopped_filter_result(
    particles: int,
    timing: str,
    *,
    processed_episode_count: int,
    zero_support_close_count: int,
):
    return {
        "schema_version": "path_c_r015_filter_candidate_result_v2",
        "selection_rule_id": design.FILTER_SELECTION_RULE_ID,
        "particles_per_prototype": particles,
        "filter_algorithm_id": design.FILTER_ALGORITHM_ID,
        "resampling_algorithm": design.FILTER_RESAMPLING_ALGORITHM_ID,
        "resampling_timing": timing,
        "candidate_evaluation_status": design.FILTER_CANDIDATE_S1_STOP_STATUS,
        "total_episode_count": 80,
        "processed_episode_count": processed_episode_count,
        "zero_support_close_count": zero_support_close_count,
        "s1_zero_support_close_rate_lower_bound": zero_support_close_count / 80,
        "conditioned_opening_prefix_cache_binding_sha256": "2" * 64,
        "conditioned_opening_prefix_sha256_by_particle_count": {
            str(particles): "4" * 64,
        },
        "conditioned_opening_attempt_sha256_by_particle_count": {
            str(particles): "6" * 64,
        },
        "conditioned_opening_prefix_scope": "processed_episode_prefix",
        "design_episode_set_sha256": "1" * 64,
        "s1_zero_support_close_rate": None,
        "s2_posterior_tv_p90": None,
        "s3_ess_fraction_p10": None,
        "s1_pass": False,
        "s2_pass": None,
        "s3_pass": None,
        "s2_status": "not_computed_due_to_irreversible_s1_failure",
        "s3_status": "not_computed_due_to_irreversible_s1_failure",
        "passed": False,
    }


def test_filter_selection_uses_particles_then_adaptive_first():
    results = [
        _filter_result(64, "adaptive_ess_below_half_v1", passed=False),
        _filter_result(64, "every_environment_step_v1", passed=True),
    ]
    selected = design.select_filter_candidate(results)
    assert selected["selected_candidate"]["particles_per_prototype"] == 64
    assert selected["selected_candidate"]["resampling_timing"] == (
        "every_environment_step_v1"
    )

    results = [_filter_result(64, "adaptive_ess_below_half_v1", passed=True)]
    selected = design.select_filter_candidate(results)
    assert selected["selected_candidate"]["resampling_timing"] == (
        "adaptive_ess_below_half_v1"
    )
    assert selected["selected_candidate"]["resampling_ess_fraction_threshold"] == 0.5


def test_filter_selection_rejects_a_higher_candidate_after_the_first_pass():
    results = [
        _filter_result(64, "adaptive_ess_below_half_v1", passed=True),
        _filter_result(64, "every_environment_step_v1", passed=False),
    ]
    with pytest.raises(ValueError, match="continued after the first passing"):
        design.select_filter_candidate(results)


def test_filter_attempt_digest_compares_only_equal_processed_episode_scopes():
    """不同 S1 早停前缀可有不同尝试摘要；同范围摘要漂移仍须拒绝。"""

    first = _s1_stopped_filter_result(
        64,
        "adaptive_ess_below_half_v1",
        processed_episode_count=2,
        zero_support_close_count=2,
    )
    second = _s1_stopped_filter_result(
        64,
        "every_environment_step_v1",
        processed_episode_count=3,
        zero_support_close_count=2,
    )
    second["conditioned_opening_attempt_sha256_by_particle_count"] = {
        "64": "7" * 64
    }
    remaining = [
        _filter_result(128, "adaptive_ess_below_half_v1", passed=False),
        _filter_result(128, "every_environment_step_v1", passed=False),
        _filter_result(256, "adaptive_ess_below_half_v1", passed=False),
        _filter_result(256, "every_environment_step_v1", passed=False),
    ]
    selection = design.select_filter_candidate([first, second, *remaining])
    assert selection["status"] == "filter_design_precision_infeasible"
    assert selection[
        "conditioned_opening_attempt_sha256_by_particle_count"
    ]["64"] == "7" * 64

    same_scope = copy.deepcopy(second)
    same_scope["processed_episode_count"] = 2
    with pytest.raises(ValueError, match="changed an existing opening attempt"):
        design.select_filter_candidate([first, same_scope])


def test_filter_selection_accepts_irreversible_s1_prefix_failure_then_continues():
    results = [
        _s1_stopped_filter_result(
            64,
            "adaptive_ess_below_half_v1",
            processed_episode_count=8,
            zero_support_close_count=2,
        ),
        _filter_result(64, "every_environment_step_v1", passed=True),
    ]
    selection = design.select_filter_candidate(results)
    assert selection["status"] == "selected"
    assert selection["selected_candidate"]["resampling_timing"] == (
        "every_environment_step_v1"
    )
    assert results[0]["s1_zero_support_close_rate_lower_bound"] == 0.025
    assert results[0]["s2_status"] == (
        "not_computed_due_to_irreversible_s1_failure"
    )
    assert results[0]["s3_status"] == (
        "not_computed_due_to_irreversible_s1_failure"
    )


def test_filter_selection_rejects_prefix_that_did_not_cross_s1_threshold():
    result = _s1_stopped_filter_result(
        64,
        "adaptive_ess_below_half_v1",
        processed_episode_count=8,
        zero_support_close_count=1,
    )
    with pytest.raises(ValueError, match="fail-closed"):
        design.select_filter_candidate([result])


def test_filter_selection_rejects_s2_or_s3_readout_after_s1_prefix_stop():
    result = dict(
        _s1_stopped_filter_result(
            64,
            "adaptive_ess_below_half_v1",
            processed_episode_count=8,
            zero_support_close_count=2,
        )
    )
    result["s2_posterior_tv_p90"] = 0.0
    with pytest.raises(ValueError, match="fail-closed"):
        design.select_filter_candidate([result])


def test_filter_selection_requires_all_80_episodes_for_exact_s1_s2_s3():
    result = dict(
        _filter_result(64, "adaptive_ess_below_half_v1", passed=True)
    )
    result["processed_episode_count"] = 79
    with pytest.raises(ValueError, match="missing exact S1--S3"):
        design.select_filter_candidate([result])


def test_filter_s1_second_closed_episode_is_an_irreversible_failure():
    assert design.s1_irreversible_failure(
        closed_episode_count=1,
        total_episode_count=80,
        threshold=0.02,
    ) is False
    assert design.s1_irreversible_failure(
        closed_episode_count=2,
        total_episode_count=80,
        threshold=0.02,
    ) is True
    with pytest.raises(ValueError, match="closed episode count"):
        design.s1_irreversible_failure(
            closed_episode_count=81,
            total_episode_count=80,
            threshold=0.02,
        )


def test_design_runner_stops_failed_filter_candidate_at_irreversible_s1_and_is_lazy():
    source = inspect.getsource(design.run_r015_design)
    assert "s1_irreversible_failure(" in source
    assert "irreversible_s1_failure" in source
    assert "select_filter_candidate(filter_results)" in source
    assert "break" in source
    assert "return_particle_weight_trace=False" in source
    assert "selected_source_index_trace" not in source
    assert "particle_weight_trace" not in source


def test_adaptive_filter_resamples_only_when_pre_resample_ess_is_below_half():
    def factory(prototype_id, index, key):
        return HiddenStateParticleV1(
            prototype_id=prototype_id,
            state_sha256=_sha((prototype_id, index, key)),
            state={"index": index},
            weight=1.0,
        )

    belief = StratifiedParticleBeliefV1.initialize(
        prototype_ids=("p0", "p1", "p2", "p3"),
        particles_per_prototype=3,
        registered_prototype_prior={f"p{index}": 0.25 for index in range(4)},
        resampling_algorithm="systematic_per_prototype_v1",
        resampling_interval_environment_steps=1,
        resampling_timing="adaptive_ess_below_half_v1",
        resampling_ess_fraction_threshold=0.5,
        initialization_key="2" * 64,
        factory=factory,
    )
    record = {
        "official_local_observation": {"step": 1},
        "ego_action_history": ["stay"],
        "raw_team_reward_history": [0.0],
        "episode_boundaries": [False],
    }

    def transition(particle, official_record, key):
        return (
            HiddenStateParticleV1(
                prototype_id=particle.prototype_id,
                state_sha256=_sha((particle.state_sha256, key)),
                state=particle.state,
                weight=1.0,
            ),
        )

    uniform = belief.update(
        official_record=record,
        update_key="3" * 64,
        transition=transition,
        compatibility=lambda particle, official_record: 1.0,
    )
    assert uniform.last_resampled_prototypes == ()
    assert all(
        value == pytest.approx(1.0)
        for value in uniform.last_pre_resample_ess_fraction_by_prototype.values()
    )

    concentrated = belief.update(
        official_record=record,
        update_key="4" * 64,
        transition=transition,
        compatibility=lambda particle, official_record: (
            1.0 if particle.state["index"] == 0 else 0.01
        ),
    )
    assert concentrated.last_resampled_prototypes == ("p0", "p1", "p2", "p3")
    assert all(
        value < 0.5
        for value in concentrated.last_pre_resample_ess_fraction_by_prototype.values()
    )


def _planning_evidence(branch_count: int, matching: int, tmp_path: Path):
    # v2 轮换顺序：块序 = (回合序号, 伙伴原型编号)，四伙伴在同一回合序号内轮换，
    # 与 first_200_design_consultations_partner_interleaved_v2 的冻结排序一致。
    partner_ids = ("proto-a", "proto-b", "proto-c", "proto-d")
    belief = StratifiedParticleBeliefV1.initialize(
        prototype_ids=partner_ids,
        particles_per_prototype=64,
        registered_prototype_prior={value: 0.25 for value in partner_ids},
        resampling_algorithm="systematic_per_prototype_v1",
        resampling_interval_environment_steps=1,
        initialization_key="1" * 64,
        factory=lambda prototype_id, index, key: HiddenStateParticleV1(
            prototype_id=prototype_id,
            state_sha256=design.canonical_sha256(
                [prototype_id, index, key]
            ),
            state={"prototype_id": prototype_id, "index": index},
            weight=1.0,
        ),
    )
    points = []
    checkpoint_refs = []
    for index in range(200):
        block = index // 20
        episode_index = block // len(partner_ids) + 1
        partner_id = partner_ids[block % len(partner_ids)]
        r_decision = {
            "masked_reference_action": "base",
            "probe_decision": "no_probe",
        }
        doubled_decision = dict(r_decision)
        if index >= matching:
            doubled_decision["probe_decision"] = "interact"
        episode_seed = 10_000 + index
        block_id = f"block-{block:03d}"
        environment_step = design.CONSULTATION_STEPS[index % 20]
        design_episode_id = f"episode-{partner_id}-{episode_index:03d}"
        checkpoint_sha256 = _sha(
            ["selected-filter-checkpoint", design_episode_id, environment_step]
        )
        checkpoint_refs.append(
            {
                "design_episode_id": design_episode_id,
                "environment_step": environment_step,
                "sha256": checkpoint_sha256,
            }
        )
        planning_key = design._six_coordinate_random_key(
            audit_unit_id=block_id,
            partner_prototype_id=partner_id,
            episode_seed=episode_seed,
            purpose="value_planning",
            environment_step=environment_step,
        )
        score_key = design._six_coordinate_random_key(
            audit_unit_id=block_id,
            partner_prototype_id=partner_id,
            episode_seed=episode_seed,
            purpose="score_estimation",
            environment_step=environment_step,
        )
        r_schedule = NestedHiddenStateBranchScheduleV1.build(
            belief=belief,
            planning_key=planning_key,
            sample_count=branch_count,
        )
        doubled_schedule = NestedHiddenStateBranchScheduleV1.build(
            belief=belief,
            planning_key=planning_key,
            sample_count=2 * branch_count,
        )

        def branch(sample):
            payload = {
                **sample.to_evidence(),
                "branch_belief_rule_id": design.BRANCH_BELIEF_RULE_ID,
                "v_base_return": float(sample.canonical_slot_16),
                "base_trajectory_sha256": _sha(
                    ("base", sample.canonical_slot_16)
                ),
                "base_committed_member_id": "proto-a",
                "base_frozen_belief_sha256": "a" * 64,
                "candidates": {
                    probe_id: {
                        "j_mask_return": 0.0,
                        "j_use_return": 0.0,
                        "mask_branch_head_belief_sha256": "b" * 64,
                        "use_branch_head_belief_sha256": "c" * 64,
                        "mask_committed_member_id": "proto-a",
                        "use_committed_member_id": "proto-a",
                        "branch_belief_rule_id": design.BRANCH_BELIEF_RULE_ID,
                    }
                    for probe_id in design.REGISTERED_PROBE_IDS
                },
            }
            payload["branch_rollout_sha256"] = (
                design._planning_branch_rollout_sha256(payload)
            )
            return payload

        r_branches = [branch(sample) for sample in r_schedule.samples]
        doubled_branches = [branch(sample) for sample in doubled_schedule.samples]
        remaining = 400 - design.CONSULTATION_STEPS[index % 20]
        new_samples = 2 * branch_count if branch_count == 2 else branch_count
        r_cost = {
            "schema_version": "path_c_r015_planning_cost_v2",
            "sample_count": branch_count,
            "remaining_environment_steps": remaining,
            "estimator_real_environment_transitions": branch_count
            * (13 * remaining - 6),
            "estimator_paired_suffix_batched_lane_time_steps": branch_count
            * 7
            * remaining,
            "new_sample_count": 0,
            "new_real_environment_transitions": 0,
            "new_paired_suffix_batched_lane_time_steps": 0,
            "new_branch_head_particle_transitions": 0,
            "particle_sequence_sha256": r_schedule.particle_sequence_sha256,
            "root_phase_uint64": r_schedule.root_phase_uint64,
        }
        doubled_cost = {
            "schema_version": "path_c_r015_planning_cost_v2",
            "sample_count": 2 * branch_count,
            "remaining_environment_steps": remaining,
            "estimator_real_environment_transitions": 2
            * branch_count
            * (13 * remaining - 6),
            "estimator_paired_suffix_batched_lane_time_steps": 2
            * branch_count
            * 7
            * remaining,
            "new_sample_count": new_samples,
            "new_real_environment_transitions": new_samples
            * (13 * remaining - 6),
            "new_paired_suffix_batched_lane_time_steps": new_samples
            * 7
            * remaining,
            "new_branch_head_particle_transitions": new_samples * 6 * 256,
            "particle_sequence_sha256": doubled_schedule.particle_sequence_sha256,
            "root_phase_uint64": doubled_schedule.root_phase_uint64,
        }
        points.append(
            {
                "consultation_id": f"point-{branch_count}-{index}",
                "design_block_id": block_id,
                "design_episode_id": design_episode_id,
                "partner_prototype_id": partner_id,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "environment_step": environment_step,
                "filter_checkpoint_sha256": checkpoint_sha256,
                "planning_random_key": planning_key,
                "score_random_key": score_key,
                "r_decision": r_decision,
                "doubled_r_decision": doubled_decision,
                "r_planning_branches": r_branches,
                "doubled_r_planning_branches": doubled_branches,
                "r_cost": r_cost,
                "doubled_r_cost": doubled_cost,
            }
        )
    new_logical = sum(
        point["doubled_r_cost"]["new_real_environment_transitions"]
        for point in points
    )
    new_particles = sum(
        point["doubled_r_cost"]["new_branch_head_particle_transitions"]
        for point in points
    )
    checkpoint_manifest_path = (
        tmp_path
        / f"selected_filter_checkpoints_r{branch_count}_{matching}.json"
    )
    checkpoint_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": design.SELECTED_FILTER_CHECKPOINT_MANIFEST_SCHEMA,
                "scientific_readout_allowed": False,
                "fixed_binding": {
                    "conditioned_opening_prefix_cache_binding_sha256": "2" * 64,
                },
                "point_count": len(checkpoint_refs),
                "ordered_checkpoint_refs": checkpoint_refs,
                "device_execution": {
                    "filter_algorithm_id": design.FILTER_ALGORITHM_ID,
                    "filter_key_contract": design.FILTER_DEVICE_KEY_CONTRACT_ID,
                    "requested_conditioned_opening_prefix_cache_binding_sha256": (
                        "2" * 64
                    ),
                    "conditioned_opening_prefix_cache_binding_sha256": "2" * 64,
                    "conditioned_opening_prefix_particles_used_per_prototype": 64,
                    "selected_replay_opening_cache_mode": "reused_complete_prefix",
                    "host_sync_inside_environment_loop": False,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    new_samples_per_point = 2 * branch_count if branch_count == 2 else branch_count
    bucket_reports = []
    active_batch_sizes = []
    for consultation_step in design.CONSULTATION_STEPS:
        remaining = 400 - consultation_step
        bucket_sample_count = 10 * new_samples_per_point
        bucket_active_sizes = [
            bucket_sample_count,
            6 * bucket_sample_count,
            6 * bucket_sample_count * 256,
            12 * bucket_sample_count,
        ]
        active_batch_sizes.extend(bucket_active_sizes)
        bucket_reports.append(
            {
                "schema_version": (
                    "path_c_r015_planning_length_bucket_execution_v2"
                ),
                "remaining_environment_steps": remaining,
                "new_sample_count": bucket_sample_count,
                "true_environment_transitions": (
                    bucket_sample_count * (13 * remaining - 6)
                ),
                "branch_head_particle_transitions": (
                    bucket_sample_count * 6 * 256
                ),
                "compiled_batch_calls": 4,
                "active_batch_sizes": bucket_active_sizes,
                "host_sync_inside_environment_loop": False,
            }
        )
    compiled_calls = sum(
        report["compiled_batch_calls"] for report in bucket_reports
    )
    return {
        "schema_version": design.PLANNING_EVIDENCE_SCHEMA,
        "scientific_readout_allowed": False,
        "branch_count": branch_count,
        "doubled_branch_count": 2 * branch_count,
        "common_random_numbers": True,
        "full_remaining_episode": True,
        "registered_probe_count": 6,
        "prototype_count": 4,
        "consultation_point_sampling_rule_id": (
            design.PLANNING_POINT_SAMPLING_RULE_ID
        ),
        "planning_filter_stream_id": design.PLANNING_FILTER_STREAM_ID,
        "branch_sampling_rule_id": design.BRANCH_SAMPLING_RULE_ID,
        "branch_belief_rule_id": design.BRANCH_BELIEF_RULE_ID,
        "grid_evaluation_rule_id": design.GRID_EVALUATION_RULE_ID,
        "nested_branch_counts": list(design.NESTED_BRANCH_COUNTS),
        "runtime_source_bindings": {
            name: {"path": f"/{name}.py", "sha256": _sha(name)}
            for name in (
                "design_runner",
                "controller",
                "full_horizon_executor",
                "runtime_bridge",
            )
        },
        "selected_filter_checkpoint_manifest": {
            "path": str(checkpoint_manifest_path),
            "sha256": design.file_sha256(checkpoint_manifest_path),
            "point_count": len(checkpoint_refs),
        },
        "total_particle_count": 256,
        "new_real_environment_transitions": new_logical,
        "new_branch_head_particle_transitions": new_particles,
        "compiled_batch_calls": compiled_calls,
        "active_batch_sizes": active_batch_sizes,
        "length_bucket_device_execution": bucket_reports,
        "host_sync_inside_environment_loop": False,
        "wall_seconds": 1.0,
        "new_real_transition_throughput_per_second": float(new_logical),
        "consultation_points": points,
    }


def test_planning_exact_tuple_agreement_boundary_and_first_passing_r(tmp_path):
    r2 = design.evaluate_planning_candidate(_planning_evidence(2, 189, tmp_path))
    r4 = design.evaluate_planning_candidate(_planning_evidence(4, 190, tmp_path))
    r8 = design.evaluate_planning_candidate(_planning_evidence(8, 200, tmp_path))
    assert r2["exact_decision_agreement"] == pytest.approx(0.945)
    assert r2["passed"] is False
    assert r4["exact_decision_agreement"] == pytest.approx(0.95)
    assert r4["passed"] is True
    selected = design.select_planning_branch_count([r2, r4])
    assert selected["selected_branch_count"] == 4
    assert set(r4["new_real_environment_transitions_per_block"]) == {
        f"block-{index:03d}" for index in range(10)
    }


def test_planning_v2_exact_costs_and_tampering_fail_closed(tmp_path):
    evidence = _planning_evidence(2, 200, tmp_path)
    result = design.evaluate_planning_candidate(evidence)
    assert sum(result["new_real_environment_transitions_per_point"]) == 3_650_800
    assert sum(result["new_branch_head_particle_transitions_per_point"]) == (
        4_800 * 256
    )
    cumulative = [
        design.evaluate_planning_candidate(_planning_evidence(count, 0, tmp_path))
        for count in (2, 4, 8)
    ]
    assert sum(
        sum(item["new_real_environment_transitions_per_point"])
        for item in cumulative
    ) == 14_603_200
    assert sum(
        sum(item["new_branch_head_particle_transitions_per_point"])
        for item in cumulative
    ) == 19_200 * 256
    legacy = copy.deepcopy(evidence)
    legacy["schema_version"] = "path_c_r015_planning_design_evidence_v1"
    with pytest.raises(ValueError, match="wrong schema"):
        design.evaluate_planning_candidate(legacy)
    changed_slot = copy.deepcopy(evidence)
    changed_slot["consultation_points"][0]["doubled_r_planning_branches"][0][
        "sample_slot"
    ] = 1
    with pytest.raises(ValueError, match="phase, slot, or random key"):
        design.evaluate_planning_candidate(changed_slot)
    changed_cost = copy.deepcopy(evidence)
    changed_cost["consultation_points"][0]["doubled_r_cost"][
        "compiled_batch_calls"
    ] = 5
    with pytest.raises(ValueError, match="point cost may not claim"):
        design.evaluate_planning_candidate(changed_cost)
    changed_sync = copy.deepcopy(evidence)
    changed_sync["consultation_points"][0]["doubled_r_cost"][
        "host_sync_inside_environment_loop"
    ] = True
    with pytest.raises(ValueError, match="point cost may not claim"):
        design.evaluate_planning_candidate(changed_sync)


def test_lazy_selection_rejects_evidence_after_first_pass(tmp_path):
    r2 = design.evaluate_planning_candidate(_planning_evidence(2, 200, tmp_path))
    r4 = design.evaluate_planning_candidate(_planning_evidence(4, 200, tmp_path))
    with pytest.raises(ValueError, match="continued after the first passing"):
        design.select_planning_branch_count([r2, r4])


def test_design_protocol_and_seed_roles_are_static_and_disjoint():
    protocol = design.load_r015_design_protocol(
        CONFIG_DIR / "path_c_r015_design_data_protocol.yaml"
    )
    assert protocol["filter_selection"]["particles_per_prototype"] == [64, 128, 256]
    assert protocol["planning_selection"]["branch_counts"] == [2, 4, 8]
    point_rule = protocol["planning_selection"]["consultation_point_sampling"]
    assert point_rule["planning_filter_stream_id"] == (
        "r015_filter_repeat_0_20260718_v2"
    )
    assert point_rule["second_filter_stream_role"] == (
        "posterior_repeat_stability_only"
    )
    planning_scope = protocol["planning_selection"]["planning_scope"]
    assert planning_scope["continuation_controller"] == (
        "map_prototype_committed_cook_v1"
    )
    assert planning_scope["planning_routing_frequency"] == (
        "commit_once_at_branch_head"
    )
    assert planning_scope["execution_routing_frequency"] == (
        "update_online_each_environment_step"
    )
    result = design.validate_seed_role_isolation(
        design_seeds=[10, 11],
        admission_seeds=[8115],
        pilot_seeds=[20, 21],
        formal_seeds=[30],
    )
    assert result["passed"] is True
    with pytest.raises(ValueError, match="overlap"):
        design.validate_seed_role_isolation(
            design_seeds=[10, 11],
            admission_seeds=[8115],
            pilot_seeds=[11, 20],
        )


def test_pilot_protocol_binds_eighty_blocks_nine_checks_and_formal_exclusion():
    pilot = yaml.safe_load(
        (CONFIG_DIR / "path_c_r015_pilot.yaml").read_text(encoding="utf-8")
    )
    assert pilot["status"] == "static_registered_not_frozen"
    assert pilot["scientific_readout_allowed"] is False
    assert pilot["sampling"]["total_paired_blocks"] == 80
    assert set(pilot["wiring_checks"]) == {
        "information_isolation",
        "filter_normalization_and_repeat_stability",
        "response_use_pairing",
        "random_key_naming",
        "zero_probe_interpretability",
        "artifact_counts_read_back",
        "lockstep_structural_zero",
        "shared_firing_indicator",
        "firing_count_input_isolation",
    }
    assert pilot["outputs"]["required_header"]["formal_interval_includes_pilot"] is False
    assert pilot["execution_boundary"]["gpu_hour_estimate_is_not_a_stop_limit"] is True


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _binding(path: Path):
    return {"path": str(path), "sha256": design.file_sha256(path)}


def _selection_files(tmp_path: Path):
    protocol = _write(tmp_path / "design-protocol.yaml", "static protocol\n")
    filter_bindings = [
        _binding(
            _write(
                tmp_path / "filter-0.json",
                json.dumps(
                    {
                        "schema_version": design.FILTER_EVIDENCE_SCHEMA,
                        "scientific_readout_allowed": False,
                    }
                )
                + "\n",
            )
        )
    ]
    planning_bindings = [
        _binding(
            _write(
                tmp_path / f"planning-{index}.json",
                json.dumps(
                    {
                        "schema_version": design.PLANNING_EVIDENCE_SCHEMA,
                        "scientific_readout_allowed": False,
                    }
                )
                + "\n",
            )
        )
        for index in range(2)
    ]
    selected_filter = {
        "particles_per_prototype": 64,
        "filter_algorithm_id": design.FILTER_ALGORITHM_ID,
        "resampling_algorithm": design.FILTER_RESAMPLING_ALGORITHM_ID,
        "resampling_timing": "adaptive_ess_below_half_v1",
        "resampling_ess_fraction_threshold": 0.5,
        "resampling_interval_environment_steps": 1,
    }
    report = {
        "schema_version": design.DESIGN_SELECTION_REPORT_SCHEMA,
        "scientific_readout_allowed": False,
        "status": "selected",
        "protocol_path": str(protocol),
        "protocol_sha256": design.file_sha256(protocol),
        "filter_evidence": filter_bindings,
        "planning_evidence": planning_bindings,
        "filter_selection": {
            "status": "selected",
            "selected_candidate": selected_filter,
        },
        "planning_selection": {
            "status": "selected",
            "selected_branch_count": 4,
        },
    }
    report_path = tmp_path / "selection.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, report, filter_bindings, planning_bindings, selected_filter


def _ready_manifest(tmp_path: Path, monkeypatch):
    report_path, report, filter_bindings, planning_bindings, selected_filter = (
        _selection_files(tmp_path)
    )
    common_file = _write(tmp_path / "bound.txt", "bound\n")
    settings_file = _write(tmp_path / "settings.py", "DELIVERY_REWARD = 20\n")
    environment = {
        "settings": _binding(settings_file),
        **{
            name: _binding(common_file)
            for name in ("overcooked", "layouts", "common")
        },
    }
    semantics = {
        name: _binding(common_file)
        for name in (
            "response_vocabulary",
            "probe_registry",
            "ego_history_contract",
            "response_projection",
            "filter_implementation",
            "planner_implementation",
            "continuation_controller",
        )
    }
    execution_artifacts = {
        name: _binding(common_file) for name in design.EXECUTION_ARTIFACTS
    }
    preregistration_artifacts = {
        name: _binding(common_file)
        for name in design.PREFLIGHT_PREREGISTRATION_ARTIFACTS
    }
    protocol_artifacts = {
        "design_data_protocol": _binding(Path(report["protocol_path"])),
        "pilot_protocol": _binding(common_file),
    }
    texts = {
        name: _binding(common_file)
        for name in ("audit_spec", "preregistration", "return_bound_derivation")
    }
    checkpoints = []
    checkpoint_contracts = (
        (100, "ego", design.OFFICIAL_SP_FAMILY_ID, 29_949_952),
        (101, "partner", design.OFFICIAL_SP_FAMILY_ID, 29_949_952),
        (102, "partner", design.OFFICIAL_SP_FAMILY_ID, 29_949_952),
        (201, "partner", design.OFFICIAL_OP_FAMILY_ID, 29_999_104),
        (202, "partner", design.OFFICIAL_OP_FAMILY_ID, 29_999_104),
    )
    for index, (seed, role, family_id, effective_steps) in enumerate(
        checkpoint_contracts
    ):
        checkpoint_path = _write(tmp_path / f"checkpoint-{index}", f"checkpoint-{index}")
        training_manifest = _write(
            tmp_path / f"training-manifest-{index}.json", "{}\n"
        )
        checkpoints.append(
            {
                "artifact_id": f"artifact-{index}",
                "role": role,
                "training_seed": seed,
                "family_id": family_id,
                "effective_environment_steps": effective_steps,
                "path": str(checkpoint_path),
                "checkpoint_sha256": design.checkpoint_artifact_sha256(checkpoint_path),
                "model_weights_hash_domain": design.FLAX_WEIGHTS_HASH_DOMAIN,
                "model_weights_sha256": _sha(f"weights-{index}"),
                "training_run_id": _sha(f"run-{index}"),
                "format": "fixture",
                "parameter_tree_path": ["params"],
                "training_manifest_path": str(training_manifest),
                "training_manifest_sha256": design.file_sha256(training_manifest),
            }
        )
    support_report = _write(
        tmp_path / "support.json", json.dumps({"support_complete": True}) + "\n"
    )
    support_registration = _write(tmp_path / "support-registration.yaml", "support\n")
    support_evidence = _write(tmp_path / "support.jsonl", "{}\n" * 1600)
    manifest = {
        "schema_version": design.FREEZE_MANIFEST_SCHEMA,
        "freeze_status": "ready_for_authorized_freeze_flip",
        "scientific_readout_allowed": False,
        "delivery_reward_readback": design.parse_delivery_reward(settings_file),
        "practical_margin_fraction": 0.25,
        "practical_margin": 5.0,
        "environment_sources": environment,
        "checkpoints": checkpoints,
        "semantic_artifacts": semantics,
        "execution_artifacts": execution_artifacts,
        "preregistration_artifacts": preregistration_artifacts,
        "protocol_artifacts": protocol_artifacts,
        "design_selection": {
            "report_path": str(report_path),
            "report_sha256": design.file_sha256(report_path),
            "filter": selected_filter,
            "planning_branch_count": 4,
            "planning_branch_sampling_rule_id": design.BRANCH_SAMPLING_RULE_ID,
            "planning_branch_belief_rule_id": design.BRANCH_BELIEF_RULE_ID,
            "planning_grid_evaluation_rule_id": design.GRID_EVALUATION_RULE_ID,
            "filter_evidence": filter_bindings,
            "planning_evidence": planning_bindings,
        },
        "partner_support": {
            "registration_path": str(support_registration),
            "registration_sha256": design.file_sha256(support_registration),
            "report_path": str(support_report),
            "report_sha256": design.file_sha256(support_report),
            "episode_evidence_path": str(support_evidence),
            "episode_evidence_sha256": design.file_sha256(support_evidence),
            "episode_evidence_rows": 1600,
            "candidate_ids": [f"candidate-{index}" for index in range(4)],
        },
        "text_artifacts": texts,
        "pilot_wiring_report": {
            "status": "not_run_in_registered_sequence",
            "path": None,
            "sha256": None,
        },
        "full_state_upper_bound": {"status": "not_provided"},
        "type_b_signoff": {
            "alpha": 0.25,
            "authorization_quote": "用户授权原文",
            "authorization_quote_sha256": _sha("用户授权原文"),
            "status": "signed",
        },
    }
    monkeypatch.setattr(
        design,
        "load_flax_parameter_tree",
        lambda path, **_: {"path": str(path)},
    )
    checkpoint_weights = {
        str(checkpoint["path"]): checkpoint["model_weights_sha256"]
        for checkpoint in checkpoints
    }
    monkeypatch.setattr(
        design,
        "flax_weights_sha256",
        lambda params: checkpoint_weights[str(params["path"])],
    )
    monkeypatch.setattr(
        design,
        "_validate_partner_support_binding",
        lambda **_: dict(manifest["partner_support"]),
    )
    monkeypatch.setattr(
        design,
        "_validate_selected_design_evidence",
        lambda *_args, **_kwargs: (
            selected_filter,
            report["planning_selection"],
            64,
            4,
        ),
    )
    return manifest


def test_ready_freeze_manifest_accepts_complete_content_bindings(tmp_path, monkeypatch):
    result = design.validate_ready_freeze_manifest(_ready_manifest(tmp_path, monkeypatch))
    assert result["ready_for_authorized_freeze_flip"] is True


def test_preregistration_fill_covers_every_prepilot_binding_and_keeps_honesty(
    tmp_path, monkeypatch
):
    preregistration = yaml.safe_load(
        (CONFIG_DIR / "path_c_r015_preregistration.yaml").read_text(encoding="utf-8")
    )
    manifest = _ready_manifest(tmp_path, monkeypatch)
    fill_values = design.preregistration_fill_values(manifest)
    filled = design.apply_preregistration_fill_values(preregistration, fill_values)
    validation = design.validate_prefilled_preregistration(filled)
    assert validation["ready_for_second_manifest_pass"] is True
    assert filled["controller"]["hidden_state_filter"]["particles_per_prototype"] == 64
    assert filled["controller"]["planning"]["branches_per_candidate"] == 4
    assert filled["partner_support"]["registration_sha256"] == (
        manifest["partner_support"]["registration_sha256"]
    )
    assert all(
        filled["artifacts"][name]["sha256"] != "pending"
        for name in design.PREFLIGHT_PREREGISTRATION_ARTIFACTS
    )
    assert filled["artifacts"]["pilot_wiring_report"] == {
        "path": "pending",
        "sha256": "pending",
    }
    assert filled["experiment"]["freeze_status"] == "static_registered_not_frozen"
    assert filled["scientific_readout_allowed"] is False
    assert preregistration["controller"]["planning"]["branches_per_candidate"] == (
        "pending_design_data"
    )


def test_preregistration_fill_rejects_partial_or_extra_update_set():
    preregistration = yaml.safe_load(
        (CONFIG_DIR / "path_c_r015_preregistration.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="incomplete or changes extra"):
        design.apply_preregistration_fill_values(
            preregistration,
            {
                "schema_version": design.PREREGISTRATION_FILL_SCHEMA,
                "freeze_status_after_copy": "static_registered_not_frozen",
                "scientific_readout_allowed": False,
                "field_updates": {
                    "controller.hidden_state_filter.particles_per_prototype": 64
                },
            },
        )


def test_second_manifest_pass_changes_only_preregistration_text_path(tmp_path):
    original_preregistration = _write(tmp_path / "prereg-old.yaml", "old\n")
    filled_preregistration = _write(tmp_path / "prereg-filled.yaml", "filled\n")
    inputs = {
        "schema_version": design.FREEZE_INPUT_SCHEMA,
        "text_artifacts": {
            "audit_spec": "audit.md",
            "preregistration": str(original_preregistration),
            "return_bound_derivation": "bound.md",
        },
        "unchanged": {"value": 7},
    }
    second = design.freeze_inputs_with_preregistration_path(
        inputs,
        filled_preregistration,
    )
    assert second["text_artifacts"]["preregistration"] == str(
        filled_preregistration.resolve()
    )
    assert second["unchanged"] == inputs["unchanged"]
    assert inputs["text_artifacts"]["preregistration"] == str(original_preregistration)


def test_two_pass_cli_requires_the_first_pass_preregistration_identity(tmp_path):
    preregistration = _write(tmp_path / "preregistration.yaml", "status: template\n")
    inputs = {
        "text_artifacts": {
            "preregistration": _binding(preregistration),
        }
    }
    freeze_manifest_cli._verify_first_pass_preregistration_identity(
        inputs, preregistration
    )
    replacement = _write(tmp_path / "replacement.yaml", "status: replacement\n")
    with pytest.raises(ValueError, match="first-pass input binding"):
        freeze_manifest_cli._verify_first_pass_preregistration_identity(
            inputs, replacement
        )


def test_freeze_file_bindings_accept_generated_content_addressed_inputs(tmp_path):
    artifact = _write(tmp_path / "generated-artifact.json", "{}\n")
    binding = _binding(artifact)
    assert design._bind_file_mapping(
        {"generated": binding}, field="generated_artifacts"
    ) == {"generated": binding}

    changed = copy.deepcopy(binding)
    changed["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash does not match"):
        design._bind_file_mapping(
            {"generated": changed}, field="generated_artifacts"
        )


def test_checkpoint_binding_accepts_and_rechecks_generated_readback_fields(
    tmp_path, monkeypatch
):
    checkpoint = _write(tmp_path / "checkpoint", "checkpoint\n")
    checkpoint_sha256 = _sha("checkpoint-artifact")
    weights_sha256 = _sha("checkpoint-weights")
    run_id = _sha("training-run")
    manifest_path = tmp_path / "training-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "path_c_official_training_artifact_v2",
                "training_config_path": "training.yaml",
                "seed": 100,
                "training_run_id": run_id,
                "checkpoint": {
                    "path": str(checkpoint),
                    "format": "fixture",
                    "parameter_tree_path": ["params"],
                    "checkpoint_sha256": checkpoint_sha256,
                    "model_weights_hash_domain": design.FLAX_WEIGHTS_HASH_DOMAIN,
                    "model_weights_sha256": weights_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        design, "load_flax_parameter_tree", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        design,
        "checkpoint_artifact_sha256",
        lambda _path: checkpoint_sha256,
    )
    monkeypatch.setattr(design, "flax_weights_sha256", lambda _params: weights_sha256)
    monkeypatch.setattr(
        design,
        "validate_official_artifact_manifest",
        lambda *_args, **_kwargs: {
            "family_id": design.OFFICIAL_SP_FAMILY_ID,
            "snapshot_environment_steps": 29_949_952,
            "layout": "test_time_simple",
        },
    )
    rich_input = {
        "artifact_id": "ego-seed-100",
        "role": "ego",
        "training_seed": 100,
        "path": str(checkpoint),
        "format": "fixture",
        "parameter_tree_path": ["params"],
        "training_manifest_path": str(manifest_path),
        "family_id": design.OFFICIAL_SP_FAMILY_ID,
        "effective_environment_steps": 29_949_952,
        "checkpoint_sha256": checkpoint_sha256,
        "model_weights_hash_domain": design.FLAX_WEIGHTS_HASH_DOMAIN,
        "model_weights_sha256": weights_sha256,
        "training_run_id": run_id,
        "training_manifest_sha256": design.file_sha256(manifest_path),
    }
    assert design._bind_checkpoint(rich_input)["checkpoint_sha256"] == checkpoint_sha256
    changed = copy.deepcopy(rich_input)
    changed["model_weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="actual artifact read-back"):
        design._bind_checkpoint(changed)


def test_signed_manifest_requires_explicit_authorization_and_keeps_readout_closed(
    tmp_path, monkeypatch
):
    pending = _ready_manifest(tmp_path, monkeypatch)
    pending["freeze_status"] = "pending_type_b_signature"
    pending["type_b_signoff"] = {
        "alpha": 0.25,
        "authorization_quote": "pending_authorization_quote",
        "status": "pending",
    }
    ready = design.sign_ready_freeze_manifest(
        pending,
        authorization_quote="用户逐字授权原文",
    )
    assert ready["freeze_status"] == "ready_for_authorized_freeze_flip"
    assert ready["scientific_readout_allowed"] is False
    frozen = design.authorize_pilot_freeze_manifest(
        ready,
        authorization_quote="用户逐字授权原文",
    )
    assert frozen["freeze_status"] == "frozen"
    assert frozen["scientific_readout_allowed"] is False
    assert design.validate_authorized_pilot_freeze_manifest(frozen)[
        "authorized_for_registered_pilot"
    ] is True
    tampered = copy.deepcopy(frozen)
    tampered["pilot_freeze_authorization"]["ready_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another ready manifest"):
        design.validate_authorized_pilot_freeze_manifest(tampered)
    with pytest.raises(ValueError, match="repeat the signed quote"):
        design.authorize_pilot_freeze_manifest(
            ready,
            authorization_quote="另一段文字",
        )


def test_completed_pilot_binding_only_fills_the_registered_slot(tmp_path, monkeypatch):
    ready = _ready_manifest(tmp_path, monkeypatch)
    frozen = design.authorize_pilot_freeze_manifest(
        ready,
        authorization_quote="用户授权原文",
    )
    pilot_report = _write(
        tmp_path / "pilot-report.json",
        json.dumps(
            {
                "schema_version": "path_c_r015_pilot_wiring_report_v2",
                "scientific_readout_allowed": False,
                "wiring_checks": {
                    name: True for name in design.PILOT_WIRING_CHECKS
                },
            }
        )
        + "\n",
    )
    completed = design.bind_completed_pilot_report(
        frozen,
        pilot_report_path=pilot_report,
    )
    assert completed["pilot_wiring_report"] == {
        "status": "passed",
        "path": str(pilot_report.resolve()),
        "sha256": design.file_sha256(pilot_report),
    }
    assert completed["design_selection"] == frozen["design_selection"]
    assert completed["scientific_readout_allowed"] is False
    preregistration = yaml.safe_load(
        (CONFIG_DIR / "path_c_r015_preregistration.yaml").read_text(encoding="utf-8")
    )
    prefilled = design.apply_preregistration_fill_values(
        preregistration,
        design.preregistration_fill_values(ready),
    )
    pilot_bound_preregistration = (
        design.bind_completed_pilot_report_to_preregistration(prefilled, completed)
    )
    assert pilot_bound_preregistration["artifacts"]["pilot_wiring_report"] == {
        "path": str(pilot_report.resolve()),
        "sha256": design.file_sha256(pilot_report),
    }
    assert pilot_bound_preregistration["experiment"]["freeze_status"] == (
        "static_registered_not_frozen"
    )
    assert pilot_bound_preregistration["scientific_readout_allowed"] is False


def test_formal_freeze_requires_new_authorization_and_changes_only_formal_slots(
    tmp_path, monkeypatch
):
    ready = _ready_manifest(tmp_path, monkeypatch)
    preregistration = yaml.safe_load(
        (CONFIG_DIR / "path_c_r015_preregistration.yaml").read_text(encoding="utf-8")
    )
    prefilled = design.apply_preregistration_fill_values(
        preregistration,
        design.preregistration_fill_values(ready),
    )
    prefilled_path = _write(
        tmp_path / "prefilled-preregistration.yaml",
        yaml.safe_dump(prefilled, sort_keys=False, allow_unicode=True),
    )
    ready["text_artifacts"]["preregistration"] = _binding(prefilled_path)
    frozen_for_pilot = design.authorize_pilot_freeze_manifest(
        ready,
        authorization_quote="用户授权原文",
    )
    pilot_report = _write(
        tmp_path / "pilot-formal-freeze.json",
        json.dumps(
            {
                "schema_version": "path_c_r015_pilot_wiring_report_v2",
                "scientific_readout_allowed": False,
                "wiring_checks": {
                    name: True for name in design.PILOT_WIRING_CHECKS
                },
            }
        )
        + "\n",
    )
    completed_manifest = design.bind_completed_pilot_report(
        frozen_for_pilot,
        pilot_report_path=pilot_report,
    )
    pilot_bound_preregistration = (
        design.bind_completed_pilot_report_to_preregistration(
            prefilled,
            completed_manifest,
        )
    )
    original = copy.deepcopy(pilot_bound_preregistration)
    firing_paths = [tmp_path / f"fires-{index}.json" for index in range(5)]
    formal_manifest, formal_preregistration = (
        design.authorize_formal_freeze_after_pilot(
            completed_manifest,
            pilot_bound_preregistration,
            authorization_quote="新的正式运行授权原文",
            formal_dataset_path=tmp_path / "formal-blocks.jsonl",
            firing_count_checkpoint_paths=firing_paths,
            formal_view_record_path=tmp_path / "formal-view.json",
        )
    )
    assert pilot_bound_preregistration == original
    assert formal_manifest["freeze_status"] == "frozen"
    assert formal_manifest["scientific_readout_allowed"] is True
    assert formal_preregistration["status"] == "frozen"
    assert formal_preregistration["experiment"]["freeze_status"] == "frozen"
    assert formal_preregistration["scientific_readout_allowed"] is True
    assert formal_preregistration["formal_data"] == {
        "dataset_path": str((tmp_path / "formal-blocks.jsonl").resolve()),
        "firing_count_checkpoint_paths": [str(path.resolve()) for path in firing_paths],
        "view_consumption_record_path": str((tmp_path / "formal-view.json").resolve()),
    }
    assert formal_preregistration["artifacts"]["pilot_wiring_report"] == (
        pilot_bound_preregistration["artifacts"]["pilot_wiring_report"]
    )
    with pytest.raises(ValueError, match="new authorization quote"):
        design.authorize_formal_freeze_after_pilot(
            completed_manifest,
            pilot_bound_preregistration,
            authorization_quote="用户授权原文",
            formal_dataset_path=tmp_path / "formal-blocks.jsonl",
            firing_count_checkpoint_paths=firing_paths,
            formal_view_record_path=tmp_path / "formal-view.json",
        )
    drifted = copy.deepcopy(pilot_bound_preregistration)
    drifted["statistics"]["n_rounds_max"] = 2499
    with pytest.raises(ValueError, match="scientific fields drifted"):
        design.authorize_formal_freeze_after_pilot(
            completed_manifest,
            drifted,
            authorization_quote="新的正式运行授权原文",
            formal_dataset_path=tmp_path / "formal-blocks.jsonl",
            firing_count_checkpoint_paths=firing_paths,
            formal_view_record_path=tmp_path / "formal-view.json",
        )


def test_ready_freeze_manifest_rejects_missing_hash_pending_and_empty_signoff(
    tmp_path, monkeypatch
):
    manifest = _ready_manifest(tmp_path, monkeypatch)
    missing_hash = copy.deepcopy(manifest)
    del missing_hash["semantic_artifacts"]["probe_registry"]["sha256"]
    with pytest.raises(ValueError, match="hash"):
        design.validate_ready_freeze_manifest(missing_hash)

    pending = copy.deepcopy(manifest)
    pending["semantic_artifacts"]["probe_registry"]["path"] = "pending"
    with pytest.raises(ValueError, match="pending"):
        design.validate_ready_freeze_manifest(pending)

    empty_signoff = copy.deepcopy(manifest)
    empty_signoff["type_b_signoff"]["authorization_quote"] = ""
    with pytest.raises(ValueError, match="quote"):
        design.validate_ready_freeze_manifest(empty_signoff)


def test_ready_freeze_manifest_rejects_selection_or_evidence_mismatch(
    tmp_path, monkeypatch
):
    manifest = _ready_manifest(tmp_path, monkeypatch)
    selection_mismatch = copy.deepcopy(manifest)
    selection_mismatch["design_selection"]["planning_branch_count"] = 8
    with pytest.raises(ValueError, match="differs"):
        design.validate_ready_freeze_manifest(selection_mismatch)

    evidence_mismatch = copy.deepcopy(manifest)
    evidence_mismatch["design_selection"]["filter_evidence"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs|hash"):
        design.validate_ready_freeze_manifest(evidence_mismatch)

    checkpoint_mismatch = copy.deepcopy(manifest)
    checkpoint_mismatch["checkpoints"][4]["training_seed"] = 203
    with pytest.raises(ValueError, match="signed production set"):
        design.validate_ready_freeze_manifest(checkpoint_mismatch)


def test_pending_freeze_builder_reads_reward_support_and_design_hashes(
    tmp_path, monkeypatch
):
    settings = _write(tmp_path / "settings.py", "DELIVERY_REWARD = 20\n")
    environment_sources = {
        "settings": settings,
        "overcooked": _write(tmp_path / "overcooked.py", "# overcooked\n"),
        "layouts": _write(tmp_path / "layouts.py", "# layouts\n"),
        "common": _write(tmp_path / "common.py", "# common\n"),
    }
    report_path, selection_report, _, _, selected_filter = _selection_files(tmp_path)
    support_report = _write(
        tmp_path / "support-report.json", json.dumps({"support_complete": True})
    )
    support_registration = _write(tmp_path / "support-registration.yaml", "support\n")
    support_rows = _write(tmp_path / "support-rows.jsonl", "{}\n" * 1600)
    semantic_paths = {
        name: _write(tmp_path / f"{name}.py", f"# {name}\n")
        for name in (
            "response_vocabulary",
            "probe_registry",
            "ego_history_contract",
            "response_projection",
            "filter_implementation",
            "planner_implementation",
            "continuation_controller",
        )
    }
    text_paths = {
        name: _write(tmp_path / f"{name}.md", f"# {name}\n")
        for name in ("audit_spec", "preregistration", "return_bound_derivation")
    }
    common_bound = _write(tmp_path / "prepilot-bound.txt", "bound\n")
    execution_paths = {
        name: common_bound for name in design.EXECUTION_ARTIFACTS
    }
    preregistration_paths = {
        name: common_bound for name in design.PREFLIGHT_PREREGISTRATION_ARTIFACTS
    }
    pilot_protocol = _write(tmp_path / "pilot-protocol.yaml", "pilot\n")

    def fake_checkpoint(entry):
        index = int(entry["training_seed"])
        expected = design.EXPECTED_FREEZE_CHECKPOINTS[index]
        return {
            "artifact_id": entry["artifact_id"],
            "role": entry["role"],
            "training_seed": index,
            "family_id": expected["family_id"],
            "effective_environment_steps": expected[
                "effective_environment_steps"
            ],
            "path": str(tmp_path / f"checkpoint-{index}"),
            "checkpoint_sha256": _sha(f"checkpoint-{index}"),
            "model_weights_hash_domain": design.FLAX_WEIGHTS_HASH_DOMAIN,
            "model_weights_sha256": _sha(f"weights-{index}"),
            "training_run_id": _sha(f"run-{index}"),
            "format": "fixture",
            "parameter_tree_path": ["params"],
            "training_manifest_path": str(tmp_path / f"manifest-{index}"),
            "training_manifest_sha256": _sha(f"manifest-{index}"),
        }

    monkeypatch.setattr(design, "_bind_checkpoint", fake_checkpoint)
    monkeypatch.setattr(
        design,
        "_validate_selected_design_evidence",
        lambda *_args, **_kwargs: (
            selected_filter,
            selection_report["planning_selection"],
            64,
            4,
        ),
    )
    monkeypatch.setattr(
        design,
        "_validate_partner_support_binding",
        lambda **_: {
            "registration_path": str(support_registration.resolve()),
            "registration_sha256": design.file_sha256(support_registration),
            "report_path": str(support_report.resolve()),
            "report_sha256": design.file_sha256(support_report),
            "episode_evidence_path": str(support_rows.resolve()),
            "episode_evidence_sha256": design.file_sha256(support_rows),
            "episode_evidence_rows": 1600,
            "candidate_ids": [f"candidate-{index}" for index in range(4)],
        },
    )
    checkpoint_contracts = (
        (100, "ego"),
        (101, "partner"),
        (102, "partner"),
        (201, "partner"),
        (202, "partner"),
    )
    checkpoints = [
        {
            "artifact_id": f"artifact-{index}",
            "role": role,
            "training_seed": seed,
            "path": "unused",
            "format": "unused",
            "parameter_tree_path": ["params"],
            "training_manifest_path": "unused",
        }
        for index, (seed, role) in enumerate(checkpoint_contracts)
    ]
    manifest = design.build_pending_freeze_manifest(
        {
            "schema_version": design.FREEZE_INPUT_SCHEMA,
            "settings_path": str(settings),
            "environment_sources": {
                name: str(path) for name, path in environment_sources.items()
            },
            "checkpoints": checkpoints,
            "design_selection_report_path": str(report_path),
            "support_registration_path": str(support_registration),
            "support_report_path": str(support_report),
            "support_episode_evidence_path": str(support_rows),
            "semantic_artifacts": {
                name: str(path) for name, path in semantic_paths.items()
            },
            "execution_artifacts": {
                name: str(path) for name, path in execution_paths.items()
            },
            "preregistration_artifacts": {
                name: str(path) for name, path in preregistration_paths.items()
            },
            "protocol_artifacts": {
                "design_data_protocol": str(
                    json.loads(report_path.read_text(encoding="utf-8"))["protocol_path"]
                ),
                "pilot_protocol": str(pilot_protocol),
            },
            "text_artifacts": {name: str(path) for name, path in text_paths.items()},
        }
    )
    assert manifest["freeze_status"] == "pending_type_b_signature"
    assert manifest["practical_margin"] == 5.0
    assert manifest["partner_support"]["episode_evidence_rows"] == 1600
