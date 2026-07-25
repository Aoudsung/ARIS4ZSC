"""R015 从设计数据到正式轮次的端到端实现合同测试定义。

这里的“端到端”是指同一公开入口真正调用设计数据、机械选择、两遍冻结、
试点和正式轮次实现；它不是把若干诊断命令拼成一份运行计划。
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from experiments.overcooked_v2 import path_c_r015_design as design
from experiments.overcooked_v2 import path_c_r015 as adjudication
from experiments.overcooked_v2 import path_c_r015_artifacts as artifacts
from experiments.overcooked_v2 import path_c_r015_runtime as formal_runtime
from experiments.overcooked_v2.official import r015_runtime_bridge as runtime_bridge
from experiments.overcooked_v2.official.r015_runtime_bridge import (
    OCV2R015ProductionBackendV1,
)
from experiments.overcooked_v2 import path_c_r015_full_horizon as horizon
from experiments.overcooked_v2.path_c_r015_full_horizon import (
    OCV2R015FullHorizonExecutorV1,
)
from experiments.overcooked_v2.path_c_r015_controller import FullHorizonRolloutV1
from experiments.overcooked_v2.path_c_r015_controller import PlanningBatchRolloutsV1
from experiments.overcooked_v2.path_c_r015_controller import R015_CONSULTATION_STEPS


ROOT = Path(__file__).resolve().parents[3]


def _scan_body(source: str) -> str:
    """取设备扫描内部正文，避免把扫描结束后的宿主读回误判为逐步同步。"""

    start = source.index("def scan_step")
    end = source.index("jax.lax.scan", start)
    return source[start:end]


def test_fully_adapted_filter_marginalizes_six_actions_and_recipe_outcomes():
    """每个粒子必须对动作与可达配方结果求和，而不是抽一个动作或配方。"""

    pure_step = getattr(
        OCV2R015ProductionBackendV1,
        "_advance_online_filter_v2_device",
    )
    source = inspect.getsource(pure_step)

    # 六个官方动作的概率必须全部参与似然；抽一个 categorical 动作不是边缘化。
    assert "OFFICIAL_ACTION_ORDER" in source
    assert "len(OFFICIAL_ACTION_ORDER)" in source
    assert "jax.vmap" in source
    assert "log_action_probability" in source
    assert "log_joint_probability" in source
    assert "jax.scipy.special.logsumexp" in source
    assert "axis=(-2, -1)" in source
    assert "action_outcome_likelihood = jnp.exp" in source
    # 条件后继只能在动作和配方结果的精确 logsumexp 以后抽取。
    assert source.index("action_outcome_likelihood = jnp.exp") < source.index(
        "conditional_cdf = jnp.cumsum"
    )
    assert "conditional_uniform" in source
    assert "jax.random.categorical" not in source

    # 正确交付会触发随机配方更新，所以所有 possible_recipes 都必须进入条件后继。
    assert "possible_recipes" in source
    assert "new_correct_delivery" in source
    assert "sample_recipe_on_delivery" in source
    assert "conditional_successor" in source


def test_non_delivery_transition_has_one_result_quality_not_recipe_duplicates():
    """没有正确交付时，配方列表不得把同一环境结果重复计权。"""

    source = inspect.getsource(
        OCV2R015ProductionBackendV1._advance_online_filter_v2_device
    )
    assert "randomized_delivery" in source
    assert "outcome_indices == 0" in source
    assert "1.0 / recipe_count" in source
    # 非交付只有第 0 个结果槽质量为 1，其余 possible_recipes 槽质量为 0。
    assert "jnp.asarray(1.0, dtype=jnp.float64)" in source
    assert "jnp.asarray(0.0, dtype=jnp.float64)" in source


def test_fully_adapted_filter_diagnostics_bind_likelihood_and_conditional_state():
    source = inspect.getsource(
        OCV2R015ProductionBackendV1._advance_online_filter_v2_device
    )
    for field in (
        "predictive_likelihood",
        "predictive_mass_by_prototype",
        "action_marginal_compatible_probability",
        "outcome_compatible",
        "conditional_successor_indices",
        "conditional_partner_actions",
        "conditional_recipe_outcomes",
        "particle_environment_transition_count",
        "finite_outcome_check_count",
    ):
        assert field in source


def test_offline_online_and_branch_head_share_one_pure_filter_step():
    """离线选择、真实在线执行和规划分支头不得各自维护一套过滤语义。"""

    shared_name = "_advance_online_filter_v2_device"
    offline = inspect.getsource(
        OCV2R015ProductionBackendV1.run_offline_filter_scan_v2
    )
    online = inspect.getsource(
        OCV2R015ProductionBackendV1.advance_online_filter_v2
    )
    full_horizon = inspect.getsource(OCV2R015FullHorizonExecutorV1)

    assert shared_name in offline
    assert shared_name in online
    assert "advance_online_filter_v2(" in full_horizon
    assert "reweight_online_filter_v2_response(" in full_horizon
    assert "official_history_particle_filter_v1" not in full_horizon
    assert "update_beliefs(" not in full_horizon


def test_generic_offline_filter_name_cannot_bypass_the_v2_runtime():
    """第二版端到端路径不得意外调用会推进五成员的第一版诊断核。"""

    assert not hasattr(OCV2R015ProductionBackendV1, "run_offline_filter_scan")
    assert hasattr(
        OCV2R015ProductionBackendV1,
        "run_offline_filter_scan_v1_diagnostic",
    )
    formal_source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.replay_filter_candidate_device_v2
    )
    assert 'backend_method_name="run_offline_filter_scan_v2"' in formal_source


def test_planning_branch_head_uses_one_batched_v2_update_not_particle_loops():
    entry = inspect.getsource(OCV2R015FullHorizonExecutorV1.rollout_planning_batch)
    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.rollout_planning_length_bucket_v2
    )
    assert "rollout_planning_length_bucket_v2(" in entry
    assert "R015OnlineParticleBeliefV2" in entry
    assert "run_planning_branch_head_batch_v2" in entry
    assert "run_planning_branch_head_batch_v2" in source
    assert "belief.update(" not in source
    assert "for particle_index, particle in enumerate(belief.particles)" not in source
    assert 'key_contract="filter_transition_v1"' not in source
    # 基线后缀、六个共享探查头、屏蔽/使用后缀各是一个设备批次。
    assert source.count("run_frozen_trajectory_batch(") == 3
    branch_head = inspect.getsource(
        OCV2R015ProductionBackendV1.run_planning_branch_head_batch_v2
    )
    assert "_advance_online_filter_v2_device(" in branch_head
    assert "reweight_online_filter_v2_response(" in branch_head
    assert "jax.jit" in branch_head
    assert "jax.vmap" in branch_head
    for field in (
        "masked_prototype_masses",
        "used_prototype_masses",
        "masked_closed",
        "used_closed",
        "masked_committed_member_indices",
        "used_committed_member_indices",
        "response_match_counts_by_prototype",
        "response_match_masses_by_prototype",
    ):
        assert f'"{field}"' in branch_head
    assert "unique = jnp.sum(tied, axis=-1) == 1" in branch_head
    assert "jnp.where(unique, prototype_choice, baseline_index)" in branch_head
    assert '"masked_filter_state"' not in branch_head
    assert '"used_filter_state"' not in branch_head
    assert "unstack_online_filter_states_v2" not in source
    assert "materialize_online_filter_state_v2" not in source


def test_planning_reuses_one_bound_belief_digest_per_consultation():
    """完整粒子信念每个咨询点只摘要一次，所有样本证据复用该绑定。"""

    grouped = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.rollout_planning_length_bucket_v2
    )
    direct = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.rollout_planning_batch
    )
    cache_key = inspect.getsource(
        OCV2R015FullHorizonExecutorV1._planning_batch_cache_key
    )
    assert grouped.count("_belief_sha256(request.belief)") == 1
    assert "request.filter_checkpoint_sha256 != belief_sha256" in grouped
    assert "frozen_belief_sha256=request_belief_sha256[request_index]" in grouped
    assert "belief_sha256=request_belief_sha256[request_index]" in grouped
    assert direct.count("_belief_sha256(belief)") == 1
    assert "_belief_sha256(" not in cache_key


def test_frozen_trajectory_batch_derives_duplicate_root_key_streams_once():
    """相同分支根键只派生一次，逆索引广播后仍保留每车道正式摘要。"""

    source = inspect.getsource(
        OCV2R015ProductionBackendV1.run_frozen_trajectory_batch
    )
    for token in (
        "unique_step_keys",
        "unique_index_by_key",
        "inverse_indices",
        "future_random_key_derivation",
        "unique_root_count",
        "requested_lane_count",
        "derived_step_key_count",
        "reused_lane_count",
        "derivation_contract_id",
    ):
        assert token in source
    assert "for step_key in unique_step_keys:" in source
    assert "jnp.stack(unique_environment_keys)[inverse]" in source
    assert "jnp.stack(unique_partner_action_keys)[inverse]" in source
    assert "jnp.stack(unique_continuation_action_keys[member_id])[inverse]" in source
    formal_summary = inspect.getsource(horizon._future_random_summary)
    for field in (
        "future_random_root_key",
        "future_random_derivation_contract_id",
        "future_random_step_count",
        "future_random_sequence_sha256",
    ):
        assert field in formal_summary
    assert "future_random_key_derivation" not in formal_summary
    assert "future_random_key_derivation" not in inspect.getsource(
        OCV2R015FullHorizonExecutorV1._batch_output_sha256
    )
    assert "future_random_key_derivation" not in inspect.getsource(
        OCV2R015FullHorizonExecutorV1._batch_state_summaries
    )


def test_frozen_trajectory_batch_runs_each_partner_lane_in_exactly_one_actor_group():
    """伙伴策略按真实原型分组前向，返回顺序仍是登记的抽样顺序。"""

    source = inspect.getsource(
        OCV2R015ProductionBackendV1.run_frozen_trajectory_batch
    )
    assert "partner_index_values = tuple(" in source
    assert "original_partner_index_values = tuple(" in source
    assert "execution_order = tuple(" in source
    assert "original_lane_by_execution_lane = execution_order" in source
    for lane_value in (
        "kernels",
        "branch_keys",
        "forced_first_actions",
        "committed_member_ids",
        "active_lengths",
        "normalized_step_keys",
    ):
        assert f"{lane_value} = reorder_lanes({lane_value})" in source
    assert "partner_rows_by_prototype = tuple(" in source
    assert "if selected_index == prototype_id" in source
    assert "assigned_partner_rows = sorted(" in source
    assert "if assigned_partner_rows != list(range(lane_count)):" in source
    assert "must partition every planning lane once" in source
    assert "for params, host_rows in zip(" in source
    assert "partner_params, partner_rows_by_prototype" in source
    assert "if len(host_rows) == 0:" in source
    assert "selected_state = jax.tree_util.tree_map(" in source
    assert 'selected_observation = obs["agent_0"][rows]' in source
    assert "next_partner_state = jax.tree_util.tree_map(" in source
    assert "partner_logits = partner_logits.at[rows].set(selected_logits)" in source
    # 设备输入已经稳定排列，编译缓存只需绑定四个原型的车道数量；返回时必须
    # 恢复登记的原始车道顺序。
    cache_source = source[source.index("cache_key = (") :]
    assert "partner_lane_counts," in cache_source
    assert "partner_index_values," not in cache_source.split(")", 1)[0]
    assert "original_lane = original_lane_by_execution_lane[row]" in source
    assert "results[original_lane] =" in source
    assert "did not restore every canonical lane" in source
    # 伙伴侧不能恢复成四个网络分别处理全部车道；延续侧则必须保留五成员全推进。
    scan_start = source.index("def scan_step(")
    scan_end = source.index("return keep_inactive", scan_start)
    scan = source[scan_start:scan_end]
    assert "actor_apply(\n                        params,\n                        selected_state," in scan
    assert "actor_apply(params, current_partner_state" not in scan
    assert "for params, state in zip(continuation_params, current_members):" in scan
    assert "next_members.append(next_state)" in scan
    assert "member_logits.append(logits)" in scan


def test_planning_length_bucket_prepares_one_stream_per_sample_and_masks_padding():
    """固定图只扩展计算形状；逻辑轨迹仍共用每个样本的一条登记随机流。"""

    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.rollout_planning_length_bucket_v2
    )
    assert "static_base_steps = 399" in source
    assert "static_suffix_steps = 398" in source
    assert "prepared_future_streams = []" in source
    assert "for branch_key in branch_keys:" in source
    assert "generated_step_count=static_base_steps" in source
    assert "logical_step_count=remaining_steps" in source
    assert "prepared_future_streams.append(stream)" in source
    # 三次设备扫描都显式复用同一完整流或它的首步/后缀，不能另派生随机键。
    assert "explicit_step_keys=tuple(prepared_future_streams)" in source
    assert "(prepared_future_streams[flat_index][0],)" in source
    assert "prepared_future_streams[flat_index][1:]" in source
    assert "active_steps=(remaining_steps,) * sample_lane_count" in source
    assert "active_steps=(remaining_steps - 1,) * len(suffix_kernels)" in source
    assert "computed_environment_transitions_including_padding" in source
    assert '"true_environment_transitions": true_environment_transitions' in source
    assert '"future_random_key_derivation": {' in source


def test_selected_filter_replay_is_the_design_to_planning_state_handoff():
    """机械选中后只重放 20 个咨询边界，不把 400 步全状态写入证据。"""

    backend = inspect.getsource(
        OCV2R015ProductionBackendV1.replay_selected_consultation_states_v2
    )
    executor = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.replay_selected_consultation_states_v2
    )
    design_source = inspect.getsource(
        design._load_or_build_selected_filter_checkpoints_v2
    )
    assert 'steps != tuple(range(1, 97, 5))' in backend
    assert "for consultation_step in steps:" in backend
    assert "checkpoint_filters.append(carry[0])" in backend
    assert "checkpoint_members.append(carry[1])" in backend
    assert '"consultation_point_count": len(all_points)' in backend
    assert "all_environment_step_states" not in backend
    assert 'getattr(\n            self.production_backend,\n            "replay_selected_consultation_states_v2"' in executor
    assert "scan.get(\"consultation_states\")" in design_source
    assert "len(histories) * len(CONSULTATION_STEPS)" in design_source
    assert "cached_payloads[batch_start:batch_stop]" in backend
    assert "cached_prefix_state=cached_state" in backend
    assert "cached_particles_per_prototype=cached_particles" in backend


def test_selected_filter_replay_has_the_registered_history_boundary_order():
    """第 t 步先让延续成员读 history[t-1]，过滤器再读 history[t]。"""

    source = inspect.getsource(
        OCV2R015ProductionBackendV1.replay_selected_consultation_states_v2
    )
    assert "for record in history[:maximum_step]" in source
    assert "for record in history[1 : maximum_step + 1]" in source
    assert source.index("member_observation_values[:, segment]") < source.index(
        "next_observation_values[:, segment]"
    )
    scan_start = source.index("def scan_step")
    scan_end = source.index("for consultation_step in steps:", scan_start)
    scan = source[scan_start:scan_end]
    assert scan.index("_compiled_actor_boundary_with_start(") < scan.index(
        "_advance_online_filter_v2_device("
    )
    assert '"official_local_observation": next_observation' in scan


def test_planning_branch_head_rejects_an_unregistered_response_token():
    """回应只允许 visible、unseen 和 local_non_agent_change 的登记编号。"""

    backend = object.__new__(OCV2R015ProductionBackendV1)
    with pytest.raises(ValueError, match="unknown response token"):
        backend.run_planning_branch_head_batch_v2(
            adapter=None,
            filter_state={
                "prototype_masses": [[0.25, 0.25, 0.25, 0.25]],
                "within_prototype_weights": [
                    [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
                ],
            },
            official_step_records={},
            environment_step=1,
            resampling_timing="adaptive_ess_below_half_v1",
            response_tokens=[999],
        )


def test_zero_device_call_planning_batches_are_only_prepared_views():
    with pytest.raises(ValueError, match="requires compiled calls"):
        PlanningBatchRolloutsV1(
            samples=(),
            compiled_batch_calls=0,
            active_batch_sizes=(),
            host_sync_inside_environment_loop=False,
        )
    prepared = PlanningBatchRolloutsV1(
        samples=(),
        compiled_batch_calls=0,
        active_batch_sizes=(),
        host_sync_inside_environment_loop=False,
        prepared_view=True,
    )
    assert prepared.prepared_view is True
    with pytest.raises(ValueError, match="may not claim device work"):
        PlanningBatchRolloutsV1(
            samples=(),
            compiled_batch_calls=1,
            active_batch_sizes=(1,),
            host_sync_inside_environment_loop=False,
            prepared_view=True,
        )


def test_formal_filter_wrappers_expose_materialization_without_changing_kernel():
    """执行层可以在咨询点读出粒子，但读出不得成为另一种状态更新。"""

    for method_name in (
        "initialize_online_filter_v2",
        "advance_online_filter_v2",
        "run_offline_filter_scan_v2",
        "materialize_online_filter_v2",
        "reweight_online_filter_v2_response",
    ):
        assert callable(getattr(OCV2R015ProductionBackendV1, method_name, None))
    materialize = inspect.getsource(
        OCV2R015ProductionBackendV1.materialize_online_filter_v2
    )
    assert "_advance_online_filter_v2_device" not in materialize


def test_response_reweighting_is_per_particle_and_accepts_the_observed_token():
    """B_use 读取回应后必须重加权完整粒子云，不能只替换四个原型质量。"""

    method = OCV2R015ProductionBackendV1.reweight_online_filter_v2_response
    assert "response_tokens" in inspect.signature(method).parameters
    source = inspect.getsource(method)
    assert "within_prototype_weights" in source
    assert "response_tokens" in source
    assert 'filter_state["last_predicted_response_tokens"]' in source
    assert "matched_total = (" in source
    assert "current_masses[..., None]" in source
    assert "predicted == observed[..., None, None]" in source
    assert "response_tokens is not None and external_weight_form" in source
    assert "jnp.all(\n                matched_prototype_mass > 0.0" in source
    assert "closed" in source
    step = inspect.getsource(
        OCV2R015ProductionBackendV1._advance_online_filter_v2_device
    )
    assert '"last_predicted_response_tokens"' in step


def test_filter_and_executed_episode_loops_are_jit_scan_vmap_without_host_sync():
    """400 个环境步留在设备扫描内；宿主只在整批或咨询边界读取结果。"""

    offline = inspect.getsource(
        OCV2R015ProductionBackendV1.run_offline_filter_scan_v2
    )
    online = inspect.getsource(
        OCV2R015ProductionBackendV1.run_online_trajectory_batch
    )
    planning = inspect.getsource(
        OCV2R015ProductionBackendV1.run_frozen_trajectory_batch
    )
    for source in (offline, online, planning):
        assert "jax.jit" in source
        assert "jax.lax.scan" in source
        assert "jax.vmap" in source
        body = _scan_body(source)
        assert "jax.device_get" not in body
        assert re.search(r"(?<!j)np\.asarray\(", body) is None
        assert ".item(" not in body
        assert '"host_sync_inside_environment_loop": False' in source


def test_online_trajectory_materialization_hashes_each_recurrent_state_once():
    """相邻步骤共享同一循环状态摘要，不重复摘要同一份设备结果。"""

    source = inspect.getsource(OCV2R015FullHorizonExecutorV1._run)
    device_source = inspect.getsource(
        OCV2R015ProductionBackendV1.run_online_trajectory_batch
    )
    assert 'trace["partner_state_before"]' not in source
    assert 'trace["continuation_states_before"]' not in source
    assert '"partner_state_before"' not in device_source
    assert '"continuation_states_before"' not in device_source
    assert '"partner_state_after": selected_partner_state' in device_source
    assert '"continuation_states_after": selected_members' in device_source
    assert source.count("_library_states_sha256(") == 2
    assert source.count(
        'domain="path_c_r015_partner_recurrent_state_v1"'
    ) == 2
    assert "controller_before_sha256 = controller_after_sha256" in source
    assert "partner_before_sha256 = partner_after_sha256" in source
    assert source.index(
        "controller_after_sha256 = _library_states_sha256(controller_after)"
    ) < source.index("trajectory.append(")
    assert source.index(
        'partner_after_sha256 = _pytree_sha256('
    ) < source.index("trajectory.append(")


def _opening_cache_binding_fixture():
    history = {
        "design_episode_id": "design-episode-1",
        "design_sequence_index": 1,
        "official_history": (
            {
                "official_local_observation": [0.0],
                "episode_boundaries": [True],
            },
        ),
    }
    stream_id = "r015_filter_repeat_0_20260718_v2"
    source_files = {"runtime_bridge.py": "a" * 64}
    binding = {
        "schema_version": horizon.R015_FILTER_OPENING_PREFIX_CACHE_SCHEMA,
        "cache_contract_id": horizon.R015_FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID,
        "filter_algorithm_id": horizon.R015_FILTER_ALGORITHM_ID_V2,
        "filter_key_contract": horizon.R015_FILTER_KEY_CONTRACT_V2,
        "maximum_particles_per_prototype": 256,
        "stored_state_scope": "conditioned_opening_particles_only",
        "post_resampling_state_reuse_allowed": False,
        "protocol_file_sha256": "b" * 64,
        "protocol_payload_sha256": "c" * 64,
        "design_histories_file_sha256": "d" * 64,
        "design_histories_payload_sha256": "e" * 64,
        "source_file_sha256": source_files,
        "source_closure_sha256": horizon.canonical_sha256(source_files),
        "lane_bindings": (
            {
                "design_episode_id": history["design_episode_id"],
                "design_sequence_index": history["design_sequence_index"],
                "filter_stream_id": stream_id,
                "filter_initialization_key": horizon.canonical_sha256(
                    [
                        horizon.R015_FILTER_KEY_CONTRACT_V2,
                        stream_id,
                        history["design_sequence_index"],
                    ]
                ),
                "official_history_sha256": horizon.canonical_sha256(
                    history["official_history"]
                ),
            },
        ),
    }
    binding["cache_binding_sha256"] = horizon.canonical_sha256(binding)
    return binding, ((history, stream_id),)


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("protocol_file_sha256", "1" * 64),
        ("source_closure_sha256", "2" * 64),
    ),
)
def test_opening_prefix_cache_rejects_protocol_or_source_binding_drift(
    field,
    replacement,
):
    binding, lanes = _opening_cache_binding_fixture()
    binding[field] = replacement
    with pytest.raises(ValueError, match="digest is invalid"):
        OCV2R015FullHorizonExecutorV1._validate_conditioned_opening_prefix_cache_binding(
            binding,
            history_stream_pairs=lanes,
        )


def test_opening_prefix_cache_rejects_history_or_initialization_key_drift():
    binding, lanes = _opening_cache_binding_fixture()
    changed_history = dict(lanes[0][0])
    changed_history["official_history"] = (
        {
            "official_local_observation": [1.0],
            "episode_boundaries": [True],
        },
    )
    with pytest.raises(ValueError, match="changed its lanes"):
        OCV2R015FullHorizonExecutorV1._validate_conditioned_opening_prefix_cache_binding(
            binding,
            history_stream_pairs=((changed_history, lanes[0][1]),),
        )

    changed_key = dict(binding)
    changed_key["lane_bindings"] = (
        {
            **binding["lane_bindings"][0],
            "filter_initialization_key": "3" * 64,
        },
    )
    with pytest.raises(ValueError, match="digest is invalid"):
        OCV2R015FullHorizonExecutorV1._validate_conditioned_opening_prefix_cache_binding(
            changed_key,
            history_stream_pairs=lanes,
        )


def test_opening_prefix_cache_reuses_only_the_conditioned_initial_prefix():
    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1._replay_filter_candidate_device_single_batch
    )
    v2_key_branch = source[
        source.index("elif (\n                filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V2") :
        source.index("else:\n                raise ValueError", source.index("elif (\n                filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V2"))
    ]
    assert "resampling_timing" not in v2_key_branch
    assert "particles_per_prototype" not in v2_key_branch
    assert "common_cached_count" in source
    assert "np.sum(lane_accepted_counts - common_cached_count)" in source
    assert "common_cached_count\n                * len(expected_cache_keys)" in source
    assert "changed an existing opening prefix" in source
    assert '"post_resampling_state_reuse_count": 0' in source


def test_opening_prefix_cache_growth_is_64_then_64_then_128_per_prototype():
    """64→128→256 只生成新增槽；既有前缀摘要必须保持不变。"""

    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1._replay_filter_candidate_device_single_batch
    )
    assert "if materialized < before" in source
    assert "raw_digests.get(count) != digest" in source
    increments = tuple(current - previous for previous, current in zip((0, 64, 128), (64, 128, 256)))
    assert increments == (64, 64, 128)


def test_failed_opening_attempt_is_bound_but_not_published_as_a_reusable_prefix():
    """失败的 256 槽尝试要可追溯，但不能冒充完整 256 槽前缀。"""

    source = inspect.getsource(
        OCV2R015ProductionBackendV1.run_offline_filter_scan_v2
    )
    attempt_start = source.index("def opening_attempt_digest(")
    attempt_end = source.index("for lane, initialization_key", attempt_start)
    attempt_digest = source[attempt_start:attempt_end]
    for field in (
        'value["opening_accepted_counts"]',
        'value["opening_proposal_counts"]',
        'value["closed"]',
    ):
        assert field in attempt_digest

    # 只有四个原型都已接受到该档粒子数，才发布该档可复用摘要。
    assert "and np.all(accepted_counts >= count)" in source
    assert "materialized = max(previous_count, completed_count)" in source
    assert "elif isinstance(previous, Mapping) and previous_count > 0:" in source
    assert 'stored_payload = previous.get("payload")' in source
    assert "if count <= completed_count:" in source
    assert "previous_digests[count_key] = realized_digest" in source
    assert "previous_digests[requested_key]" not in source
    # 请求档无论成功与否都只写入单独的尝试摘要账本。
    assert "previous_attempt_digests[requested_key] = realized_attempt_digest" in source
    assert '"opening_attempt_sha256_by_particle_count"' in source
    assert '**({"payload": stored_payload} if materialized > 0 else {})' in source


def test_selected_replay_rebuilds_every_lane_when_any_opening_prefix_is_incomplete():
    """混合完整与不完整前缀时整批从登记根键重建，不能拼接两种粒子序列。"""

    source = inspect.getsource(
        OCV2R015ProductionBackendV1.replay_selected_consultation_states_v2
    )
    assert "cached_payloads: tuple[Mapping[str, Any], ...] | None = None" in source
    assert "complete_cache = len(present) == lane_count and all(" in source
    assert (
        'int(entry["materialized_particles_per_prototype"])\n'
        "                >= int(particles_per_prototype)"
    ) in source
    assert 'and isinstance(entry.get("payload"), Mapping)' in source
    assert "if complete_cache:" in source
    assert source.index("if complete_cache:") < source.index(
        "cached_payloads = tuple("
    )
    assert "if cached_payloads is None:\n                batch_cached_state = None" in source
    assert "cached_prefix_state=cached_state" in source
    assert "cached_particles_per_prototype=cached_particles" in source

    initialize_start = source.index("def compiled_replay(")
    initialize_end = source.index("def scan_step", initialize_start)
    initialize = source[initialize_start:initialize_end]
    # 未完成缓存把 cached_particles 留为 0，因此初始化器必须从原根键构造；
    # 它不能把缓存中的 closed=False 覆盖到一次已关闭的开局尝试上。
    assert "cached_state" in initialize
    assert "cached_particles_per_prototype=cached_particles" in initialize


def test_selected_replay_uses_only_a_complete_hot_cache_or_a_whole_cold_rebuild():
    """进程内缓存缺失时两个缓存参数都为 None，checkpoint 仍绑定原缓存摘要。"""

    replay = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.replay_selected_consultation_states_v2
    )
    assert "selected_opening_prefixes = None" in replay
    assert "selected_opening_prefix_binding_sha256 = None" in replay
    assert "if cache_record is not None:" in replay
    assert "def owns_complete_selected_prefix(" in replay
    assert (
        "entry.get(\"binding_sha256\")\n"
        "                        == conditioned_opening_prefix_cache_binding_sha256"
    ) in replay
    assert "materialized >= int(particles_per_prototype)" in replay
    assert 'isinstance(entry.get("payload"), Mapping)' in replay
    assert "_is_sha256(prefix_digests.get(requested_count_key))" in replay
    assert "if all(" in replay
    assert "selected_opening_prefixes = {" in replay
    assert 'selected_opening_cache_mode = "cold_rebuild_from_registered_roots"' in replay
    assert 'selected_opening_cache_mode = "reused_complete_prefix"' in replay
    assert "conditioned_opening_prefix_cache=selected_opening_prefixes" in replay
    assert (
        "conditioned_opening_prefix_cache_binding_sha256=(\n"
        "                selected_opening_prefix_binding_sha256"
    ) in replay
    assert '"requested_conditioned_opening_prefix_cache_binding_sha256"' in replay
    assert '"selected_replay_opening_cache_mode"' in replay

    checkpointing = inspect.getsource(
        design._load_or_build_selected_filter_checkpoints_v2
    )
    assert '"conditioned_opening_prefix_cache_binding_sha256": (' in checkpointing
    assert "conditioned_opening_prefix_cache_binding_sha256" in checkpointing
    assert 'manifest.get("fixed_binding") != fixed_binding' in checkpointing
    assert '"selected_replay_opening_cache_mode"' in checkpointing
    assert '"reused_complete_prefix"' in checkpointing
    assert '"cold_rebuild_from_registered_roots"' in checkpointing
    assert (
        '"requested_conditioned_opening_prefix_cache_binding_sha256"'
        in checkpointing
    )
    assert '"conditioned_opening_prefix_cache_binding_sha256"' in checkpointing
    assert '"conditioned_opening_prefix_particles_used_per_prototype"' in checkpointing


def test_filter_evidence_requires_the_requested_attempt_not_a_false_complete_prefix():
    """S1 关闭候选必须记录请求尝试，但不要求失败请求具有可复用前缀。"""

    source = inspect.getsource(design.evaluate_filter_candidate)
    assert 'opening_attempt_digests.get(str(particles))' in source
    assert "Filter evidence lacks its conditioned-opening attempt digest." in source
    assert 'if bool(summary["passed"]) and not _is_sha256(' in source
    assert 'opening_prefix_digests.get(str(particles))' in source
    stopped_return = source[
        source.index('"not_computed_due_to_irreversible_s1_failure"') :
        source.index("summary = summarize_filter_statistics(")
    ]
    assert "opening_prefix_digests.get" not in stopped_return


def test_filter_cache_accounting_follows_backend_lane_order_and_common_prefix():
    """新增与复用槽数从后端逐车道接受计数计算，不能按缓存字典顺序猜测。"""

    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1._replay_filter_candidate_device_single_batch
    )
    assert "expected_cache_keys = tuple(initialization_keys)" in source
    assert "len(set(expected_cache_keys)) != len(expected_cache_keys)" in source
    assert 'opening_diagnostics.get("opening_accepted_counts")' in source
    assert "for lane_index, cache_key in enumerate(expected_cache_keys):" in source
    assert "lane_accepted_counts = opening_accepted_counts[lane_index]" in source
    assert "common_cached_count\n                * len(expected_cache_keys)" in source
    assert "np.sum(lane_accepted_counts - common_cached_count)" in source

    merged = inspect.getsource(
        OCV2R015FullHorizonExecutorV1._merge_filter_trace_results
    )
    assert '"microbatch_conditioned_opening_prefix_particles_used_per_prototype"' in merged
    assert '"conditioned_opening_prefix_generated_particle_slots"' in merged
    assert '"conditioned_opening_prefix_reused_particle_slots"' in merged
    validator = inspect.getsource(design.evaluate_filter_candidate)
    assert "len(cached_particles_by_microbatch) != len(microbatch_lane_counts)" in validator
    assert "any(value > particles for value in cached_particles_by_microbatch)" in validator
    assert "reused_opening_slots != expected_reused_opening_slots" in validator
    assert "generated_opening_slots > maximum_generated_opening_slots" in validator


def test_formal_filter_stops_only_between_fixed_microbatches_after_s1_is_lost():
    """第二个关闭回合出现后不得再提交一个 4096 槽设备批次。"""

    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.replay_filter_candidate_device_v2
    )
    assert "stop_at_irreversible_s1_failure" in source
    assert "R015_FILTER_S1_EARLY_STOP_CLOSED_EPISODE_COUNT" in source
    assert "lanes_per_batch % R015_FILTER_REPEAT_COUNT_PER_EPISODE" in source
    assert source.index("batches.append(batch)") < source.index(
        "len(closed_episode_ids) >="
    ) < source.index("break")
    assert '"status": status' in source
    assert '"total_episode_count": R015_FILTER_DESIGN_EPISODE_COUNT' in source
    assert '"processed_episode_count": len(processed_episode_ids)' in source
    assert '"s1_close_rate_lower_bound"' in source


def test_design_candidate_evidence_marks_s2_and_s3_uncomputed_after_s1_stop():
    source = inspect.getsource(design.run_r015_design)
    assert "stop_at_irreversible_s1_failure=True" in source
    assert "s1_irreversible_failure(" in source
    assert '"failed_from_fixed_denominator_lower_bound"' in source
    assert source.count('"not_computed_due_to_irreversible_s1_failure"') >= 2
    assert 'else []' in source
    assert source.index("candidate_result = evaluate_filter_candidate(") < source.index(
        "_write_json_atomic(evidence_path, evidence)"
    )
    assert '"candidate_result": dict(candidate_result)' in source


def test_opening_prefix_cache_is_memory_only_and_crash_recovery_rebuilds_it():
    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1._conditioned_opening_prefix_cache
    )
    assert "self._conditioned_opening_prefix_caches.get" in source
    assert '"conditioned_opening_prefixes": {}' in source
    assert "open(" not in source
    assert "read_text(" not in source
    assert "pickle" not in source


def test_opening_prefix_cache_logic_lives_in_offline_scan_not_public_initializer():
    initializer = inspect.getsource(
        OCV2R015ProductionBackendV1.initialize_online_filter_v2
    )
    assert "conditioned_opening_prefix_cache" not in initializer
    offline = inspect.getsource(
        OCV2R015ProductionBackendV1.run_offline_filter_scan_v2
    )
    for local_name in (
        "cached_particles",
        "cached_opening_state",
        "cached_proposal_counts_before",
    ):
        assert f"{local_name} =" in offline
    assert "cache_binding_sha256" in offline


def test_planning_random_evidence_uses_the_registered_compact_field_names():
    fields = FullHorizonRolloutV1.__dataclass_fields__
    assert "future_random_derivation_contract_id" in fields
    assert "future_random_step_count" in fields
    assert "future_random_key_derivation_contract" not in fields
    assert "future_random_sequence_length" not in fields
    assert "future_random_keys" not in fields


def test_planning_random_evidence_rejects_the_retired_field_names():
    root_key = "4" * 64
    old = {
        "common_random_key": root_key,
        "future_random_root_key": root_key,
        "future_random_key_derivation_contract": (
            design.FUTURE_RANDOM_KEY_DERIVATION_CONTRACT
        ),
        "future_random_sequence_length": 3,
        "future_random_sequence_sha256": horizon.canonical_sha256(
            [
                horizon.derive_controller_key(
                    root_key,
                    "future_environment",
                    index,
                )
                for index in range(3)
            ]
        ),
    }
    with pytest.raises(ValueError, match="future-random contract"):
        design._validate_planning_branch_random_summary(
            old,
            remaining_environment_steps=3,
        )


def test_executed_episode_scans_one_step_then_five_steps_between_consultations():
    assert R015_CONSULTATION_STEPS == tuple(range(1, 100, 5))
    source = inspect.getsource(OCV2R015FullHorizonExecutorV1.run_paired_block)
    assert "self.initialize_belief(" in source
    assert "initialize_belief_v1_diagnostic(" not in source
    assert "stop_step - environment_step" in source
    assert '"actual_episode"' in source
    assert '"future_environment"' in source
    assert "explicit_step_keys=tuple(" in source


def test_formal_belief_initializer_is_only_a_thin_v2_filter_adapter():
    """正式块可经 helper 初始化，但 helper 不得保留另一套 v1 语义。"""

    source = inspect.getsource(OCV2R015FullHorizonExecutorV1.initialize_belief)
    assert "getattr(" in source
    assert '"initialize_online_filter_v2"' in source
    assert "device_state = initializer(" in source
    assert "_materialize_online_particle_belief_v2(" in source
    assert "initialize_belief_v1_diagnostic(" not in source
    assert "official_history_particle_filter_v1" not in source
    materialize = inspect.getsource(horizon._materialize_online_particle_belief_v2)
    assert 'getattr(production_backend, "materialize_online_filter_v2"' in materialize
    assert 'getattr(production_backend, "reweight_online_filter_v2_response"' in materialize


def test_fire_execution_uses_one_two_lane_head_and_one_three_lane_suffix_scan():
    source = inspect.getsource(OCV2R015FullHorizonExecutorV1.run_paired_block)
    assert source.count("run_online_trajectory_batch(") == 2
    assert source.count("copy.deepcopy(branch.kernel)") >= 2
    assert "suffix_starts = (a1_head.state, masked_start, used_start)" in source
    assert "branch_keys=(post_key, post_key, post_key)" in source
    assert "explicit_step_keys=(suffix_keys, suffix_keys, suffix_keys)" in source


def test_online_scan_advances_all_five_members_then_routes_from_posterior():
    source = inspect.getsource(
        OCV2R015ProductionBackendV1.run_online_trajectory_batch
    )
    assert "for member_id, member_state in zip(member_ids, current_members)" in source
    assert "next_members.append(next_state)" in source
    assert "posterior = self.online_filter_prototype_masses_v2" in source
    assert "unique = jnp.sum(tied, axis=-1) == 1" in source
    assert "jnp.where(unique, prototype_choice, baseline_index)" in source
    assert source.index("next_members.append(next_state)") < source.index(
        "selected_logits = choose_member_rows"
    )


def test_online_scan_cache_reuses_only_the_graph_not_previous_segment_values():
    """第二次命中同一静态图时必须读取本段动作、长度和回合位置。"""

    source = inspect.getsource(
        OCV2R015ProductionBackendV1.run_online_trajectory_batch
    )
    compiled_signature = source[
        source.index("def compiled_scan(") : source.index("):", source.index("def compiled_scan("))
    ]
    for name in (
        "current_forced_indices",
        "current_active_lengths",
        "current_initial_environment_steps",
    ):
        assert name in compiled_signature
    assert "step_index < current_active_lengths" in source
    assert "current_initial_environment_steps + step_index" in source
    assert (
        "current_initial_environment_steps + step_index + 1" in source
    )
    assert "current_forced_indices >= 0" in source
    compiled_call = source[source.index("final, trace = compiled(") :]
    assert "forced_indices," in compiled_call
    assert "active_lengths_array," in compiled_call
    assert "initial_environment_steps," in compiled_call
    cache_key = source[
        source.index('"online_execution_v2"') : source.index(
            "compiled = self._online_execution_scan_cache.get", source.index('"online_execution_v2"')
        )
    ]
    assert "forced_indices" not in cache_key
    assert "active_lengths_array" not in cache_key
    assert "initial_environment_steps" not in cache_key


def test_online_scan_second_cache_hit_reads_new_action_length_and_start(
    monkeypatch,
):
    """同一已缓存设备图的第二段不得沿用第一段的动态输入。"""

    import jax
    import jax.numpy as jnp

    prototype_ids = tuple(f"prototype-{index}" for index in range(4))
    baseline_id = "ego-100"
    backend = OCV2R015ProductionBackendV1(
        policies={
            member_id: SimpleNamespace()
            for member_id in (*prototype_ids, baseline_id)
        },
        prototype_ids=prototype_ids,
        baseline_member_id=baseline_id,
    )
    compiled_calls = []

    def cached_graph(
        initial_carry,
        all_inputs,
        forced_indices,
        active_lengths,
        initial_environment_steps,
    ):
        compiled_calls.append(
            (
                tuple(int(value) for value in forced_indices.tolist()),
                tuple(int(value) for value in active_lengths.tolist()),
                tuple(int(value) for value in initial_environment_steps.tolist()),
            )
        )
        step_count = int(all_inputs[0].shape[0])
        active = (
            jnp.arange(step_count, dtype=jnp.int32)[:, None]
            < active_lengths[None, :]
        )
        zeros = jnp.zeros_like(active, dtype=jnp.int32)
        return initial_carry, {
            "active": active,
            "filter_particle_environment_transitions": zeros,
            "computed_filter_particle_environment_transitions": zeros,
            "received_forced_indices": forced_indices,
            "received_initial_environment_steps": initial_environment_steps,
        }

    monkeypatch.setattr(jax, "jit", lambda function: cached_graph)

    member_ids = (*prototype_ids, baseline_id)

    def kernel(environment_step):
        return SimpleNamespace(
            partner_prototype_id=prototype_ids[0],
            environment_step=environment_step,
            snapshot=SimpleNamespace(
                state=jnp.asarray([0], dtype=jnp.int32),
                raw_obs={
                    "agent_0": jnp.zeros((2,), dtype=jnp.float32),
                    "agent_1": jnp.zeros((2,), dtype=jnp.float32),
                },
                key=jax.random.PRNGKey(environment_step),
            ),
            partner_recurrent_state=jnp.zeros((1, 2), dtype=jnp.float32),
            continuation_states=SimpleNamespace(
                by_member_id={
                    member_id: jnp.zeros((1, 2), dtype=jnp.float32)
                    for member_id in member_ids
                }
            ),
        )

    def run_segment(*, environment_step, forced_action, active_steps):
        return backend.run_online_trajectory_batch(
            adapter=SimpleNamespace(),
            kernels=(kernel(environment_step),),
            filter_state={
                "update_count": jnp.asarray(
                    [environment_step], dtype=jnp.int32
                )
            },
            branch_keys=("1" * 64,),
            total_steps=2,
            key_offset=0,
            forced_first_actions=(forced_action,),
            resampling_timing="adaptive_ess_below_half_v1",
            active_steps=(active_steps,),
            explicit_step_keys=(("2" * 64, "3" * 64),),
        )

    first = run_segment(
        environment_step=0,
        forced_action=runtime_bridge.OFFICIAL_ACTION_ORDER[0],
        active_steps=2,
    )
    second = run_segment(
        environment_step=7,
        forced_action=runtime_bridge.OFFICIAL_ACTION_ORDER[-1],
        active_steps=1,
    )

    assert len(backend._online_execution_scan_cache) == 1
    assert first["throughput"]["jit_compilations"] == 1
    assert second["throughput"]["jit_compilations"] == 0
    assert compiled_calls == [
        ((0,), (2,), (0,)),
        ((len(runtime_bridge.OFFICIAL_ACTION_ORDER) - 1,), (1,), (7,)),
    ]
    assert second["throughput"]["true_environment_transitions"] == 1
    assert tuple(
        int(value)
        for value in second["trace"]["received_initial_environment_steps"]
    ) == (7,)


def test_online_scan_closes_a_lane_when_the_v2_filter_loses_support():
    source = inspect.getsource(
        OCV2R015ProductionBackendV1.run_online_trajectory_batch
    )
    assert 'selected_filter_closed = active & advanced["closed"]' in source
    assert "next_terminated = terminated | selected_done | selected_filter_closed" in source
    assert '"filter_closed": selected_filter_closed' in source


@pytest.mark.parametrize(
    "particles_per_prototype,expected_lanes",
    ((64, 16), (128, 8), (256, 4)),
)
def test_safety_microbatch_width_never_exceeds_4096_parent_particle_slots(
    particles_per_prototype,
    expected_lanes,
):
    width, lanes, formal = OCV2R015FullHorizonExecutorV1._filter_parent_slot_schedule(
        particles_per_prototype=particles_per_prototype,
        type_a_parent_slot_batch_width=None,
    )
    assert (width, lanes, formal) == (4096, expected_lanes, True)


def test_safety_microbatches_preserve_all_1116_registered_branches():
    source = inspect.getsource(OCV2R015FullHorizonExecutorV1.evaluate_safety)
    assert "repetitions: int = 279" in source
    assert "for prototype_id in self.prototype_ids" in source
    assert "R015_FILTER_MICROBATCH_SCHEDULE_ID" in source
    assert "for microbatch_start in range(" in source
    assert "parent_slot_widths" in source
    assert "width > parent_slot_batch_width" in source
    assert "sum(active_lane_counts) != repetitions" in source
    assert source.count("run_safety_trajectory_batch(") == 1
    assert '"schema_version": "path_c_r015_safety_branch_evidence_v2"' in source
    assert '"branch_key": branch_key' in source
    assert '"branches": tuple(branch_evidence)' in source
    assert 4 * 279 == 1116


def test_execution_throughput_reports_true_padding_filter_and_call_costs():
    source = inspect.getsource(OCV2R015FullHorizonExecutorV1.run_paired_block)
    for field in (
        "true_environment_transitions",
        "computed_environment_transitions_including_masked_padding",
        "filter_particle_environment_transitions",
        "compiled_batch_calls",
        "host_sync_inside_environment_loop",
    ):
        assert field in source
    assert '"host_sync_inside_environment_loop": any(' in source


def test_pipeline_is_one_real_entrypoint_through_formal_not_a_command_plan():
    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    assert pipeline.R015_PIPELINE_STAGE_ORDER == (
        "design",
        "freeze",
        "pilot",
        "formal",
    )
    source = inspect.getsource(pipeline.run_r015_pipeline)
    stage_calls = (
        "_run_design_stage(",
        "_run_freeze_stage(",
        "_run_pilot_stage(",
        "_run_formal_stage(",
    )
    for call in stage_calls:
        assert call in source
    assert [source.index(call) for call in stage_calls] == sorted(
        source.index(call) for call in stage_calls
    )
    module_source = inspect.getsource(pipeline)
    assert "run_r015_design(" in module_source
    assert "run_r015_pilot(" in module_source
    assert "run_r015_formal(" in module_source
    freeze_source = inspect.getsource(pipeline._run_freeze_stage)
    assert "first_manifest" in freeze_source
    assert "second_manifest" in freeze_source
    assert "subprocess" not in source
    assert "command_plan" not in source


def test_pipeline_cli_can_resume_and_run_through_formal():
    launcher = (
        ROOT
        / "experiments"
        / "overcooked_v2"
        / "scripts"
        / "run_path_c_r015_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "--through" in launcher
    assert "choices=R015_PIPELINE_STAGE_ORDER" in launcher
    assert "--resume" in launcher
    assert "--authorized" in launcher
    assert "--formal-root-seed" in launcher
    assert "run_r015_pipeline(" in launcher
    assert "run_path_c_r015_design.py" not in launcher
    assert "run_path_c_r015_pilot.py" not in launcher


def test_pipeline_cli_accepts_existing_stage_files_without_a_new_template():
    """统一入口可直接绑定现有合同文件，并把该绑定原子写入恢复目录。"""

    launcher = (
        ROOT
        / "experiments"
        / "overcooked_v2"
        / "scripts"
        / "run_path_c_r015_pipeline.py"
    ).read_text(encoding="utf-8")
    for argument in (
        "--design-protocol",
        "--freeze-inputs",
        "--preregistration-template",
        "--partner-support-registration",
        "--pilot-protocol",
        "--formal-runtime-config",
        "--output-root",
        "--authorization-file",
    ):
        assert argument in launcher
    assert 'output_root / "pipeline_invocation_config.yaml"' in launcher
    assert "temporary.replace(path)" in launcher
    assert "--config 不能与逐项路径同时使用" in launcher


def test_pipeline_binds_one_support_registration_across_every_stage_file():
    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    source = inspect.getsource(pipeline._validate_explicit_support_registration)
    for declaration in (
        "design_protocol.support.partner_support_registration",
        "freeze_inputs.support_registration_path",
        "preregistration.partner_support.registration_path",
        "pilot_protocol.frozen_inputs.support_registration",
        "formal_runtime.support.partner_support_registration",
    ):
        assert declaration in source
    assert "path.resolve() != explicit" in source
    assert "if explicit is None:" in source
    assert "raise ValueError" in source[source.index("if explicit is None:") :]
    assert "return None" not in source
    binding = inspect.getsource(pipeline._build_run_binding)
    assert '"partner_support_registration"' in binding


def test_pipeline_run_binding_covers_the_complete_end_to_end_source_closure():
    """设计、冻结、试点或正式执行所用源码漂移都必须改变统一运行摘要。"""

    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    relative = {
        path.relative_to(Path(pipeline.__file__).resolve().parent).as_posix()
        for path in pipeline._source_paths()
    }
    assert {
        "batched_rollout.py",
        "env_adapter.py",
        "path_c_backbone_ppo.py",
        "path_c_r015.py",
        "path_c_r015_controller.py",
        "path_c_r015_design.py",
        "path_c_r015_pilot.py",
        "path_c_r015_runtime.py",
        "path_c_r015_full_horizon.py",
        "path_c_r015_artifacts.py",
        "path_c_response_probe.py",
        "path_c_response_summary.py",
        "path_c_flax_policy.py",
        "path_c_official_artifact.py",
        "path_c_official_evidence.py",
        "path_c_pool_admission.py",
        "path_c_seed.py",
        "path_c_sequence.py",
        "path_c_standard.py",
        "path_c_standard_diagnostics.py",
        "path_c_standard_training.py",
        "official/overcooked_v2_experiments_adapter.py",
        "official/r015_runtime_bridge.py",
    }.issubset(relative)
    binding = inspect.getsource(pipeline._build_run_binding)
    assert "for path in _source_paths()" in binding
    assert '"source_sha256"' in binding


def test_custom_pipeline_config_always_requires_partner_support_registration(
    tmp_path,
):
    """即使只运行设计阶段，也必须先绑定同一份四伙伴 checkpoint 登记。"""

    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    config_path = tmp_path / "pipeline.yaml"
    with pytest.raises(ValueError, match="partner_support_registration"):
        pipeline._pipeline_paths(
            {
                "paths": {
                    "output_root": str(tmp_path / "run"),
                    "design_protocol": str(tmp_path / "design.yaml"),
                }
            },
            config_path=config_path,
            through="design",
        )
    with pytest.raises(ValueError, match="显式伙伴支持登记"):
        pipeline._validate_explicit_support_registration(
            paths={"design_protocol": tmp_path / "design.yaml"},
            through="design",
        )


def test_freeze_requires_and_validates_the_pilot_support_declaration(tmp_path):
    """冻结阶段已经消费试点模板，因此此时就必须绑定同一支持登记。"""

    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    support = tmp_path / "support.yaml"
    support.write_text("{}\n", encoding="utf-8")
    design_path = tmp_path / "design.yaml"
    design_path.write_text(
        "support:\n  partner_support_registration: support.yaml\n",
        encoding="utf-8",
    )
    freeze_path = tmp_path / "freeze.yaml"
    freeze_path.write_text(
        "support_registration_path: support.yaml\n",
        encoding="utf-8",
    )
    preregistration_path = tmp_path / "preregistration.yaml"
    preregistration_path.write_text(
        "partner_support:\n  registration_path: support.yaml\n",
        encoding="utf-8",
    )
    raw_paths = {
        "output_root": str(tmp_path / "run"),
        "design_protocol": str(design_path),
        "partner_support_registration": str(support),
        "freeze_inputs": str(freeze_path),
        "preregistration_template": str(preregistration_path),
    }
    with pytest.raises(ValueError, match="pilot_protocol"):
        pipeline._pipeline_paths(
            {"paths": raw_paths},
            config_path=tmp_path / "pipeline.yaml",
            through="freeze",
        )

    pilot_path = tmp_path / "pilot.yaml"
    pilot_path.write_text(
        "frozen_inputs:\n  support_registration: another-support.yaml\n",
        encoding="utf-8",
    )
    paths = pipeline._pipeline_paths(
        {"paths": {**raw_paths, "pilot_protocol": str(pilot_path)}},
        config_path=tmp_path / "pipeline.yaml",
        through="freeze",
    )
    with pytest.raises(ValueError, match="pilot_protocol"):
        pipeline._validate_explicit_support_registration(
            paths=paths,
            through="freeze",
        )


def test_freeze_stage_materializes_the_only_pilot_protocol_that_can_run():
    """冻结阶段填入真实路径，试点阶段只接收该已解析产物。"""

    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    freeze_source = inspect.getsource(pipeline._run_freeze_stage)
    assert 'resolved_pilot_protocol["status"] = "resolved_for_authorized_pilot"' in (
        freeze_source
    )
    for binding in (
        '"preregistration": str(freeze_dir / "prefilled_preregistration.yaml")',
        '"design_selection_report": str(selection_path)',
        '"signed_freeze_manifest": str(freeze_dir / "pilot_frozen_manifest.json")',
        '"support_registration": str(support_registration_path)',
        'resolved_outputs["report_path"]',
        'freeze_artifacts["resolved_pilot_protocol"]["path"]',
    ):
        target = (
            inspect.getsource(pipeline.run_r015_pipeline)
            if binding.startswith("freeze_artifacts")
            else freeze_source
        )
        assert binding in target
    assert 'resolved_pilot_path = freeze_dir / "resolved_pilot_protocol.yaml"' in (
        freeze_source
    )
    assert 'source_overrides["pilot_protocol"] = str(resolved_pilot_path)' in (
        freeze_source
    )
    assert '"resolved_pilot_protocol": _artifact(resolved_pilot_path)' in (
        freeze_source
    )


def test_pipeline_requires_an_explicit_formal_root_seed_before_freeze():
    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    source = inspect.getsource(pipeline.run_r015_pipeline)
    assert "formal_root_seed" in inspect.signature(pipeline.run_r015_pipeline).parameters
    assert 'target_index >= R015_PIPELINE_STAGE_ORDER.index("freeze")' in source
    assert "not isinstance(formal_root_seed, int)" in source
    assert "formal_root_seed < 0" in source
    freeze_source = inspect.getsource(pipeline._run_freeze_stage)
    assert "formal_root_seed=formal_root_seed" in freeze_source
    seed_binding = inspect.getsource(pipeline._bind_formal_root_seed)
    assert "existing is not None and existing != formal_root_seed" in seed_binding
    assert 'state["formal_sampling_registered_before_pilot"] = True' in seed_binding
    assert "_bind_formal_root_seed(" in source
    assert '"formal_sampling_schedule"' in freeze_source


def test_pipeline_result_reports_every_completed_stage_and_terminal_stage():
    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    source = inspect.getsource(pipeline.run_r015_pipeline)
    for field in (
        "schema_version",
        "status",
        "completed_stages",
        "terminal_stage",
        "state_path",
        "stage_artifacts",
    ):
        assert field in source


def test_pipeline_formal_preregistration_defaults_to_the_generated_freeze_file(
    tmp_path,
):
    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    config_path = tmp_path / "pipeline.yaml"
    output_root = tmp_path / "run"
    config = {
        "paths": {
            "output_root": str(output_root),
            "design_protocol": "design.yaml",
            "partner_support_registration": "support.yaml",
            "freeze_inputs": "freeze.json",
            "preregistration_template": "prereg.yaml",
            "pilot_protocol": "pilot.yaml",
            "formal_runtime_config": "runtime.yaml",
        }
    }
    paths = pipeline._pipeline_paths(
        config,
        config_path=config_path,
        through="formal",
    )
    assert paths["formal_preregistration"] == (
        output_root / "freeze" / "formal_frozen_preregistration.yaml"
    ).resolve()


def test_pipeline_formal_factory_loads_real_support_and_production_backend():
    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    source = inspect.getsource(pipeline.build_official_r015_formal_producer)
    assert "R015PartnerSupportSpec.from_mapping" in source
    assert "_require_frozen_support()" in source
    assert "build_official_r015_production_backend(" in source
    assert "frozen_manifest_path" in inspect.signature(
        pipeline.build_official_r015_formal_producer
    ).parameters
    assert "_OfficialR015FormalBlockProducer(" in source
    assert "support_spec=support_spec" in source
    assert "ego_candidate=ego_candidate" in source
    assert "frozen_manifest_path=frozen_manifest_path" in source
    assert "preregistration_path=preregistration_path" in source
    assert "placeholder" not in source.lower()


def test_formal_relative_artifact_paths_use_runtime_config_directory(
    tmp_path,
    monkeypatch,
):
    """从任意当前目录启动时，正式配置的相对路径仍相对配置文件。"""

    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    runtime_directory = tmp_path / "registered-runtime"
    runtime_directory.mkdir()
    unrelated_working_directory = tmp_path / "another-working-directory"
    unrelated_working_directory.mkdir()
    monkeypatch.chdir(unrelated_working_directory)
    resolved = runtime_bridge._resolve_r015_candidate_artifact_paths(
        {
            "candidate_id": "ego-100",
            "checkpoint_path": "artifacts/checkpoint",
            "training_config_path": "configs/train.yaml",
            "training_manifest_path": "artifacts/manifest.json",
        },
        config_base_path=runtime_directory,
    )
    assert Path(resolved["checkpoint_path"]) == (
        runtime_directory / "artifacts" / "checkpoint"
    ).resolve()
    assert Path(resolved["training_config_path"]) == (
        runtime_directory / "configs" / "train.yaml"
    ).resolve()
    assert Path(resolved["training_manifest_path"]) == (
        runtime_directory / "artifacts" / "manifest.json"
    ).resolve()
    assert pipeline._support_registration_path(
        {"support": {"partner_support_registration": "support.yaml"}},
        runtime_config_path=runtime_directory / "runtime.yaml",
    ) == (runtime_directory / "support.yaml").resolve()
    factory_source = inspect.getsource(
        pipeline.build_official_r015_formal_producer
    )
    assert "config_base_path=runtime_config_path.parent" in factory_source
    assert "_resolve_r015_candidate_artifact_paths(" in factory_source


def test_pipeline_formal_adapter_rejects_a_v1_block_from_the_executor():
    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    class _OldExecutor:
        @staticmethod
        def produce_formal_paired_block(*, request, preregistration):
            return {"schema_version": "path_c_r015_paired_block_v1"}

    # 此测试只隔离适配器的 schema 拒绝；冻结身份本身由下面的五成员夹具覆盖。
    producer = object.__new__(pipeline._OfficialR015FormalBlockProducer)
    producer.backend = SimpleNamespace(full_horizon_executor=_OldExecutor())
    producer.preregistration = object()
    producer._verify_formal_freeze_identity = lambda *, refresh_files: None
    request = formal_runtime.R015FormalBlockRequestV1(
        round_index=1,
        audit_unit_id="a" * 64,
        prototype_id="prototype-0",
        episode_seed=7,
        ego_position=1,
        mechanical_attempt_index=0,
    )
    with pytest.raises(ValueError, match="非 v2 配对块"):
        producer.produce_formal_paired_block(request)


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("path", "/another/checkpoint"),
        ("checkpoint_sha256", "1" * 64),
        ("model_weights_sha256", "2" * 64),
        ("training_run_id", "3" * 64),
    ),
)
def test_loaded_five_member_identity_rejects_same_id_with_changed_provenance(
    tmp_path,
    monkeypatch,
    field,
    replacement,
):
    """候选名不变也不能掩盖路径、文件、权重或训练运行身份漂移。"""

    prototype_ids = tuple(f"prototype-{index}" for index in range(4))
    baseline_id = "ego-100"
    member_ids = (*prototype_ids, baseline_id)
    policies = {
        member_id: SimpleNamespace(
            policy=SimpleNamespace(params={"member_id": member_id})
        )
        for member_id in member_ids
    }
    identities = {}
    for index, member_id in enumerate(member_ids):
        identities[member_id] = {
            "artifact_id": member_id,
            "training_seed": 100 + index,
            "path": str((tmp_path / f"checkpoint-{index}").resolve()),
            "checkpoint_sha256": f"{index + 4:x}" * 64,
            "model_weights_hash_domain": runtime_bridge.FLAX_WEIGHTS_HASH_DOMAIN,
            "model_weights_sha256": f"{index + 9:x}" * 64,
            "training_run_id": f"{index + 1:x}" * 64,
            "training_manifest_path": str(
                (tmp_path / f"manifest-{index}.json").resolve()
            ),
            "training_manifest_sha256": f"{index + 2:x}" * 64,
        }
    backend = OCV2R015ProductionBackendV1(
        policies=policies,
        prototype_ids=prototype_ids,
        baseline_member_id=baseline_id,
        checkpoint_identities=identities,
    )
    monkeypatch.setattr(
        runtime_bridge,
        "flax_weights_sha256",
        lambda params: identities[str(params["member_id"])][
            "model_weights_sha256"
        ],
    )
    frozen = {"checkpoints": [dict(identities[value]) for value in member_ids]}
    frozen["checkpoints"][0][field] = replacement
    with pytest.raises(ValueError, match="differs from its frozen identity"):
        backend.verify_frozen_checkpoint_identities(
            frozen,
            refresh_files=False,
        )


def test_formal_identity_reuses_hash_only_while_immutable_parameter_leaves_match(
    tmp_path,
    monkeypatch,
):
    """同一不可变参数树不重复搬回设备数据；替换叶节点仍重算并拒绝。"""

    prototype_ids = tuple(f"prototype-{index}" for index in range(4))
    baseline_id = "ego-100"
    member_ids = (*prototype_ids, baseline_id)
    leaves = {member_id: object() for member_id in member_ids}
    policies = {
        member_id: SimpleNamespace(
            policy=SimpleNamespace(
                params={"member_id": member_id, "leaf": leaves[member_id]}
            )
        )
        for member_id in member_ids
    }
    identities = {
        member_id: {
            "artifact_id": member_id,
            "training_seed": 100 + index,
            "path": str((tmp_path / f"checkpoint-{index}").resolve()),
            "checkpoint_sha256": f"{index + 4:x}" * 64,
            "model_weights_hash_domain": runtime_bridge.FLAX_WEIGHTS_HASH_DOMAIN,
            "model_weights_sha256": f"{index + 9:x}" * 64,
            "training_run_id": f"{index + 1:x}" * 64,
            "training_manifest_path": str(
                (tmp_path / f"manifest-{index}.json").resolve()
            ),
            "training_manifest_sha256": f"{index + 2:x}" * 64,
        }
        for index, member_id in enumerate(member_ids)
    }
    weights_by_leaf = {
        id(leaves[member_id]): identities[member_id]["model_weights_sha256"]
        for member_id in member_ids
    }
    hash_calls: list[object] = []

    def fake_weights_sha256(params):
        hash_calls.append(params["leaf"])
        return weights_by_leaf[id(params["leaf"])]

    monkeypatch.setattr(runtime_bridge, "flax_weights_sha256", fake_weights_sha256)
    monkeypatch.setattr(
        runtime_bridge,
        "_immutable_jax_parameter_snapshot",
        lambda params: ("fake-tree", (params["leaf"],)),
    )
    backend = OCV2R015ProductionBackendV1(
        policies=policies,
        prototype_ids=prototype_ids,
        baseline_member_id=baseline_id,
        checkpoint_identities=identities,
    )
    frozen = {"checkpoints": [dict(identities[value]) for value in member_ids]}

    backend.verify_frozen_checkpoint_identities(frozen, refresh_files=False)
    assert len(hash_calls) == 5
    backend.verify_frozen_checkpoint_identities(frozen, refresh_files=False)
    assert len(hash_calls) == 5

    changed_member = member_ids[0]
    changed_leaf = object()
    weights_by_leaf[id(changed_leaf)] = "0" * 64
    policies[changed_member].policy.params = {
        "member_id": changed_member,
        "leaf": changed_leaf,
    }
    with pytest.raises(ValueError, match="identity changed"):
        backend.verify_frozen_checkpoint_identities(frozen, refresh_files=False)
    assert len(hash_calls) == 6


def test_unified_pipeline_really_invokes_all_four_stages_in_order(
    tmp_path,
    monkeypatch,
):
    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    calls: list[str] = []

    def artifact(name: str):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        return pipeline._artifact(path)

    design_artifacts = {"design_selection_report": artifact("selection")}
    freeze_artifacts = {
        "pilot_frozen_manifest": artifact("pilot_frozen_manifest"),
        "prefilled_preregistration": artifact("prefilled_preregistration"),
        "resolved_pilot_protocol": artifact("resolved_pilot_protocol"),
    }
    pilot_artifacts = {
        "pilot_bound_manifest": artifact("pilot_bound_manifest"),
        "pilot_bound_preregistration": artifact("pilot_bound_preregistration"),
    }
    formal_artifacts = {"formal_terminal_result": artifact("formal_terminal")}

    def fake_design(**kwargs):
        calls.append("design")
        return {"status": "selected"}, design_artifacts

    def fake_freeze(**kwargs):
        calls.append("freeze")
        return {"schema_version": "freeze"}, freeze_artifacts

    pilot_arguments = {}

    def fake_pilot(**kwargs):
        calls.append("pilot")
        pilot_arguments.update(kwargs)
        return {
            "wiring_checks": {f"check_{index}": True for index in range(9)}
        }, pilot_artifacts

    def fake_formal(**kwargs):
        calls.append("formal")
        return {
            "termination_kind": "firing_count_registered_negative",
            "effect_value_read": False,
        }, formal_artifacts

    monkeypatch.setattr(pipeline, "_run_design_stage", fake_design)
    monkeypatch.setattr(pipeline, "_run_freeze_stage", fake_freeze)
    monkeypatch.setattr(pipeline, "_run_pilot_stage", fake_pilot)
    monkeypatch.setattr(pipeline, "_run_formal_stage", fake_formal)
    monkeypatch.setattr(
        pipeline,
        "_build_run_binding",
        lambda **kwargs: {
            "schema_version": pipeline.R015_PIPELINE_RUN_BINDING_SCHEMA,
            "test_binding": True,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_explicit_support_registration",
        lambda **kwargs: {"path": "support.yaml", "sha256": "a" * 64},
    )

    for name in (
        "design.yaml",
        "freeze.json",
        "prereg.yaml",
        "pilot.yaml",
        "runtime.yaml",
        "support.yaml",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("schema_version: path_c_r015_pipeline_v1\n", encoding="utf-8")
    authorization = {
        stage: {
            "authorized": True,
            "reference": f"signed-{stage}",
            **(
                {"authorization_quote": f"signed quote for {stage}"}
                if stage in {"freeze", "formal"}
                else {}
            ),
        }
        for stage in pipeline.R015_PIPELINE_STAGE_ORDER
    }
    config = {
        "schema_version": pipeline.R015_PIPELINE_CONFIG_SCHEMA,
        "paths": {
            "output_root": str(tmp_path / "run"),
            "design_protocol": str(tmp_path / "design.yaml"),
            "partner_support_registration": str(tmp_path / "support.yaml"),
            "freeze_inputs": str(tmp_path / "freeze.json"),
            "preregistration_template": str(tmp_path / "prereg.yaml"),
            "pilot_protocol": str(tmp_path / "pilot.yaml"),
            "formal_runtime_config": str(tmp_path / "runtime.yaml"),
        },
        "authorization": authorization,
    }
    result = pipeline.run_r015_pipeline(
        config,
        config_path=config_path,
        through="formal",
        resume=False,
        execution_authorized=True,
        formal_root_seed=8115,
    )
    assert calls == list(pipeline.R015_PIPELINE_STAGE_ORDER)
    assert pilot_arguments["resolved_pilot_protocol_path"] == Path(
        freeze_artifacts["resolved_pilot_protocol"]["path"]
    )
    assert result["completed_stages"] == list(pipeline.R015_PIPELINE_STAGE_ORDER)
    assert result["terminal_stage"] is None
    assert result["status"] == "completed_through_requested_stage"


def _formal_identity_payload(schema_version: str) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "phase": "formal",
        "round_index": 1,
        "audit_unit_id": "a" * 64,
        "prototype_id": "prototype-0",
        "episode_seed": 7,
        "ego_position": 1,
        "mechanical_attempt_index": 0,
        "runtime_binding_sha256": "b" * 64,
        "formal_effect_look_number": 1,
        "pilot_data": False,
        "firing_indicator": False,
    }


@pytest.mark.parametrize(
    "schema_version",
    (
        "path_c_r015_paired_block_v1",
        "path_c_r015_executed_paired_block_v2",
    ),
)
def test_formal_runtime_rejects_old_or_not_yet_replayed_blocks(schema_version):
    expected = (1, "a" * 64, "prototype-0", 7, 1, 0)
    with pytest.raises(ValueError, match="non-formal block schema"):
        formal_runtime._validate_formal_block_identity_without_effect_read(
            _formal_identity_payload(schema_version),
            expected=expected,
            runtime_binding_sha256="b" * 64,
        )


def test_formal_producer_upgrades_real_execution_only_after_three_replays():
    producer = getattr(
        OCV2R015FullHorizonExecutorV1,
        "produce_formal_paired_block",
    )
    source = inspect.getsource(producer)
    assert source.count("run_paired_block(") == 2
    assert "build_r015_replay_backend(" in source
    assert "verify_r015_trace_manifest(" in source
    assert "verify_r015_probe_decision(" in source
    assert "verify_r015_safety_comparison(" in source
    assert "validate_r015_paired_block(" in source
    conversion = inspect.getsource(horizon._formal_components_from_executed)
    assert "path_c_r015_executed_paired_block_v2" in conversion
    assert "path_c_r015_paired_block_v2" in conversion
    assert "pending_independent_replay" not in source
    assert "placeholder" not in source.lower()


def _formal_failure_test_preregistration():
    return SimpleNamespace(
        _require_frozen_support=lambda: {"prototype-0": object()},
        support_spec=SimpleNamespace(
            candidates=(SimpleNamespace(candidate_id="prototype-0"),)
        ),
        controller=SimpleNamespace(
            particles_per_prototype=64,
            resampling_timing="adaptive_ess_below_half_v1",
            planning_branches_per_candidate=2,
        ),
    )


def _formal_failure_test_request():
    return formal_runtime.R015FormalBlockRequestV1(
        round_index=1,
        audit_unit_id="a" * 64,
        prototype_id="prototype-0",
        episode_seed=7,
        ego_position=1,
        mechanical_attempt_index=0,
    )


@pytest.mark.parametrize(
    "exception_type",
    (RuntimeError, ValueError, TypeError, FloatingPointError),
)
def test_formal_producer_propagates_program_identity_and_numeric_errors(
    exception_type,
):
    """程序、身份、类型和数值错误必须停止正式链。

    这些错误不能伪装成可替换块。
    """

    executor = object.__new__(OCV2R015FullHorizonExecutorV1)

    def fail(**_arguments):
        raise exception_type("deliberate formal failure")

    executor.run_paired_block = fail
    executor.formal_freeze_identity_verifier = lambda: None
    with pytest.raises(exception_type, match="deliberate formal failure"):
        executor.produce_formal_paired_block(
            request=_formal_failure_test_request(),
            preregistration=_formal_failure_test_preregistration(),
        )


def test_formal_producer_only_converts_the_explicit_mechanical_block_error():
    """只有真实回合的专用机械失效异常可返回无部分证据的失效结果。"""

    executor = object.__new__(OCV2R015FullHorizonExecutorV1)

    def fail(**_arguments):
        raise formal_runtime.R015MechanicalBlockInvalidError(
            "online filter lost support"
        )

    executor.run_paired_block = fail
    executor.formal_freeze_identity_verifier = lambda: None
    result = executor.produce_formal_paired_block(
        request=_formal_failure_test_request(),
        preregistration=_formal_failure_test_preregistration(),
    )
    assert result == formal_runtime.R015FormalBlockProductionResultV1(
        mechanically_valid=False,
        paired_block=None,
        invalid_reason=(
            "R015MechanicalBlockInvalidError: online filter lost support"
        ),
    )

    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.produce_formal_paired_block
    )
    assert "except R015MechanicalBlockInvalidError as error:" in source
    assert "actual_execution_completed = False" in source
    assert "actual_execution_completed = True" in source
    assert "if actual_execution_completed:" in source
    for error_name in (
        "RuntimeError",
        "ValueError",
        "TypeError",
        "FloatingPointError",
    ):
        assert f"except {error_name}" not in source


def test_formal_producer_propagates_a_mechanical_error_from_independent_replay():
    """真执行完成后再出现机械异常属于重放不一致，必须停止正式链。"""

    executor = object.__new__(OCV2R015FullHorizonExecutorV1)
    call_count = 0

    def execute_then_fail(**_arguments):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {}
        raise formal_runtime.R015MechanicalBlockInvalidError(
            "independent replay diverged"
        )

    executor.run_paired_block = execute_then_fail
    executor.formal_freeze_identity_verifier = lambda: None
    with pytest.raises(
        formal_runtime.R015MechanicalBlockInvalidError,
        match="independent replay diverged",
    ):
        executor.produce_formal_paired_block(
            request=_formal_failure_test_request(),
            preregistration=_formal_failure_test_preregistration(),
        )
    assert call_count == 2


def test_true_episode_mechanical_invalidity_has_only_registered_raise_sites():
    """真实轨迹、规划分支或回应重加权零支持只可请求整块处理。"""

    run_source = inspect.getsource(OCV2R015FullHorizonExecutorV1._run)
    paired_source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.run_paired_block
    )
    assert run_source.count("raise R015MechanicalBlockInvalidError(") == 2
    assert "closed before the requested segment boundary" in run_source
    assert "online filter lost support during real execution" in run_source
    assert "returned an invalid active mask" in run_source
    assert "changed the filter-closed shape" in run_source
    assert paired_source.count(
        "except _R015OnlineFilterZeroSupportError as error:"
    ) == 3
    assert paired_source.count("raise R015MechanicalBlockInvalidError(") == 3
    assert "lost support at the true episode opening" in paired_source
    assert "planning branch lost finite-particle support" in paired_source
    assert "current probe response lost finite-particle support" in paired_source


@pytest.mark.parametrize(
    "weights,message",
    (
        ((0.0, 0.0, 0.0, 0.0), "all support"),
        ((0.0, 1.0, 1.0, 1.0), "one prototype"),
    ),
)
def test_current_response_zero_support_uses_the_dedicated_filter_exception(
    weights,
    message,
):
    """回应不可能或删除任一登记原型时，不能写入零后验继续执行。"""

    belief = object.__new__(horizon.R015OnlineParticleBeliefV2)
    object.__setattr__(belief, "prototype_ids", ("p0", "p1", "p2", "p3"))
    object.__setattr__(belief, "particles_per_prototype", 1)
    object.__setattr__(belief, "particles", (object(),) * 4)
    expected = (
        "Current response has zero probability under all support"
        if message == "all support"
        else "Current response removed one prototype from support"
    )
    with pytest.raises(horizon._R015OnlineFilterZeroSupportError, match=expected):
        belief._from_total_weights(weights)


def test_planning_closed_flags_and_split_response_become_whole_block_invalidity():
    """规划头或真实回应的有限粒子支持耗尽不能伪造成零值规划结果。"""

    planning = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.rollout_planning_length_bucket_v2
    )
    assert 'np.any(compact["masked_closed"])' in planning
    assert 'np.any(compact["used_closed"])' in planning
    closed_index = planning.index('np.any(compact["masked_closed"])')
    assert planning.index(
        "raise _R015OnlineFilterZeroSupportError(", closed_index
    ) > closed_index
    assert "planning branch lost one prototype's support" in planning
    assert '"j_mask_return": 0.0' not in planning[closed_index:]
    assert '"j_use_return": 0.0' not in planning[closed_index:]

    paired = inspect.getsource(OCV2R015FullHorizonExecutorV1.run_paired_block)
    consult = paired.index("decision = controller.consult(")
    consult_catch = paired.index(
        "except _R015OnlineFilterZeroSupportError as error:", consult
    )
    assert paired.index("raise R015MechanicalBlockInvalidError(", consult_catch) > (
        consult_catch
    )
    split = paired.index("self._split_probe_response(")
    split_catch = paired.index(
        "except _R015OnlineFilterZeroSupportError as error:", split
    )
    assert paired.index("raise R015MechanicalBlockInvalidError(", split_catch) > (
        split_catch
    )


def test_formal_actual_and_independent_replay_each_execute_fresh_planning():
    """独立重放可复用编译程序，但不得复用第一次正式执行的规划结果。"""

    block_source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.run_paired_block
    )
    producer_source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.produce_formal_paired_block
    )
    assert "self._prepared_planning_batches.clear()" in block_source
    assert block_source.index("self._prepared_planning_batches.clear()") < (
        block_source.index("observations, _ = self.adapter.reset(episode_seed)")
    )
    assert producer_source.count("self.run_paired_block(**arguments)") == 2
    assert producer_source.index("executed = self.run_paired_block(**arguments)") < (
        producer_source.index(
            "independently_replayed = self.run_paired_block(**arguments)"
        )
    )
    assert '"independent_replay_reused_compiled_device_programs": True' in (
        producer_source
    )
    assert 'executed["planning_real_environment_transitions"]' in producer_source
    assert (
        'independently_replayed[\n                            "planning_real_environment_transitions"'
        in producer_source
    )
    assert "decision.consultation.cost_accounting" in block_source
    assert "decision.consultation.device_execution" in block_source
    assert "if self.formal_freeze_identity_verifier is None:" in producer_source
    assert producer_source.count("self.formal_freeze_identity_verifier()") == 1
    assert producer_source.index("self.formal_freeze_identity_verifier()") < (
        producer_source.index(
            "independently_replayed = self.run_paired_block(**arguments)"
        )
    )

    from experiments.overcooked_v2 import path_c_r015_pipeline as pipeline

    adapter_source = inspect.getsource(
        pipeline._OfficialR015FormalBlockProducer.produce_formal_paired_block
    )
    assert adapter_source.count(
        "self._verify_formal_freeze_identity(refresh_files=False)"
    ) == 1
    assert adapter_source.index(
        "self._verify_formal_freeze_identity(refresh_files=False)"
    ) < adapter_source.index(
        "self.backend.full_horizon_executor.produce_formal_paired_block("
    )


def test_formal_producer_emits_the_exact_three_registered_replay_schemas():
    """冻结清单只接受登记的重放证据，不能用通用 verified 包装代替。"""

    replay_methods = {
        "path_c_r015_trace_replay_result_v2": (
            horizon.OCV2R015ReplayBackendV1.replay_trace
        ),
        "path_c_r015_decision_evidence_verification_v1": (
            horizon.OCV2R015ReplayBackendV1.replay_probe_decision
        ),
        "path_c_r015_safety_branch_verification_v2": (
            horizon.OCV2R015ReplayBackendV1.replay_safety_comparison
        ),
    }
    assert {
        artifacts.ARTIFACT_SCHEMAS[name]
        for name in (
            "trace_replay_verifier",
            "decision_evidence_verifier",
            "safety_branch_evidence_verifier",
        )
    } == set(replay_methods)
    source = "\n".join(inspect.getsource(method) for method in replay_methods.values())
    for schema_version, method in replay_methods.items():
        assert schema_version in inspect.getsource(method)
    for retired_or_generic in (
        "path_c_r015_trace_replay_verification_v1",
        "path_c_r015_probe_decision_replay_verification_v1",
        "path_c_r015_safety_comparison_replay_verification_v1",
        "generic_replay_verification_v1",
    ):
        assert retired_or_generic not in source


def test_safety_rejection_uses_its_selected_consultation_without_faking_a_fire():
    request = formal_runtime.R015FormalBlockRequestV1(
        round_index=1,
        audit_unit_id="a" * 64,
        prototype_id="prototype-0",
        episode_seed=7,
        ego_position=1,
        mechanical_attempt_index=0,
    )
    executed = {
        "probe_fired": False,
        "probe_step": None,
        "no_probe_reason": "safety_rejected",
        "consultations": (
            {
                "environment_step": 6,
                "selected_for_safety_probe_id": "probe-0",
            },
        ),
        "safety_comparisons": (
            {
                "prototype_id": "prototype-0",
                "repetitions": 1,
                "wrong_delivery_count": 0,
                "positive_posterior_support": True,
                "compatible_hidden_state_reconstructed": True,
                "branches": (
                    {
                        "branch_key": "b" * 64,
                        "source_particle_state_sha256": "c" * 64,
                        "replayable_environment_steps": (),
                        "wrong_delivery_detected": False,
                    },
                ),
            },
        ),
    }
    preregistration = SimpleNamespace(
        artifacts={
            "random_key_derivation": SimpleNamespace(sha256="d" * 64)
        }
    )
    comparisons = horizon._formal_safety_comparisons(
        executed,
        preregistration=preregistration,
        request=request,
    )
    assert len(comparisons) == 1
    assert comparisons[0]["environment_step"] == 6
    conversion = inspect.getsource(horizon._formal_components_from_executed)
    assert '"probe_step": probe_step if probe_fired else None' in conversion


@pytest.mark.parametrize(
    "loader_name,artifact_name,validator_name,callable_name",
    (
        (
            "_load_trace_replay_verifier",
            "trace_replay_verifier",
            "_validate_trace_replay_verifier_manifest",
            "verify_r015_trace_manifest",
        ),
        (
            "_load_decision_evidence_verifier",
            "decision_evidence_verifier",
            "_validate_decision_evidence_verifier_manifest",
            "verify_r015_probe_decision",
        ),
        (
            "_load_safety_branch_evidence_verifier",
            "safety_branch_evidence_verifier",
            "_validate_safety_branch_evidence_verifier_manifest",
            "verify_r015_safety_comparison",
        ),
    ),
)
def test_dynamic_replay_loader_can_bind_runtime_module_with_dataclasses(
    tmp_path,
    monkeypatch,
    loader_name,
    artifact_name,
    validator_name,
    callable_name,
):
    """动态装载时先登记模块，使 future annotations 与 dataclass 可解析。"""

    implementation_path = Path(formal_runtime.__file__).resolve()
    manifest_path = tmp_path / f"{artifact_name}.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "implementation_path": str(implementation_path),
        "implementation_sha256": "a" * 64,
    }
    monkeypatch.setattr(adjudication, "_load_json_or_yaml", lambda path: manifest)
    monkeypatch.setattr(adjudication, validator_name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        adjudication,
        "_validate_manifest_implementation",
        lambda *args, **kwargs: None,
    )
    preregistration = SimpleNamespace(
        artifacts={artifact_name: SimpleNamespace(path=manifest_path)}
    )
    verifier = getattr(adjudication, loader_name)(preregistration)
    assert verifier is getattr(sys.modules[verifier.__module__], callable_name)


def test_dynamic_replay_loaders_register_before_exec_and_clean_up_on_failure():
    for loader in (
        adjudication._load_trace_replay_verifier,
        adjudication._load_decision_evidence_verifier,
        adjudication._load_safety_branch_evidence_verifier,
    ):
        source = inspect.getsource(loader)
        registered = source.index("sys.modules[module_name] = module")
        executed = source.index("module_spec.loader.exec_module(module)")
        cleaned = source.index("sys.modules.pop(module_name, None)")
        assert registered < executed < cleaned


def test_one_formal_round_contains_one_block_for_each_prototype():
    prototypes = tuple(f"prototype-{index}" for index in range(4))
    schedule = formal_runtime.build_r015_formal_sampling_schedule(
        prototype_ids=prototypes,
        root_seed=8115,
    )
    entries = schedule["entries"]
    assert schedule["schema_version"] == (
        "path_c_r015_formal_sampling_schedule_v4"
    )
    assert schedule["round_sampling_contract"] == (
        "iid_round_vectors_ego_cook_seat_with_prefrozen_replacements_v4"
    )
    assert schedule["mechanical_replacement_attempts_per_coordinate"] == 32
    assert schedule["replacement_episode_seed_derivation_id"] == (
        "sha256_registered_coordinate_low32_allow_collisions_v1"
    )
    assert schedule["replacement_episode_seed_collision_policy"] == (
        "allow_value_collisions_without_redraw"
    )
    assert len(entries) == 2500 * 4
    for offset in range(0, len(entries), 4):
        one_round = entries[offset : offset + 4]
        assert {item["round_index"] for item in one_round} == {
            offset // 4 + 1
        }
        assert tuple(item["prototype_id"] for item in one_round) == prototypes
        assert all(item["ego_position"] == 1 for item in one_round)
        assert all(
            len(item["episode_seeds_by_attempt_index"]) == 32
            for item in one_round
        )
    builder_source = inspect.getsource(
        formal_runtime.build_r015_formal_sampling_schedule
    )
    assert "collision_nonce" not in builder_source
    assert "seen_episode" not in builder_source
    assert '"r015_formal_prefrozen_replacement_seed_v4"' in builder_source
    assert "int(digest[-8:], 16)" in builder_source

    retired_schedule = dict(schedule)
    retired_schedule["schema_version"] = "path_c_r015_formal_sampling_schedule_v3"
    with pytest.raises(ValueError, match="wrong contract"):
        artifacts._validate_formal_sampling_schedule(
            retired_schedule,
            prototype_ids=prototypes,
        )


def test_formal_filter_root_uses_the_prefrozen_episode_seed_not_attempt_number():
    """替换块的过滤随机根绑定预先登记的 seed，而不是运行时重试编号。"""

    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.produce_formal_paired_block
    )
    start = source.index("filter_initialization_key = canonical_sha256(")
    end = source.index("episode_random_key =", start)
    filter_root_expression = source[start:end]
    assert '"r015_formal_filter_initialization_v4"' in filter_root_expression
    assert "request.episode_seed" in filter_root_expression
    assert "request.mechanical_attempt_index" not in filter_root_expression

    runtime_source = inspect.getsource(formal_runtime.run_r015_formal)
    assert "episode_seed = attempt_episode_seeds[attempt_index]" in runtime_source
    assert "episode_seed=episode_seed" in runtime_source
    assert "mechanical_attempt_index=attempt_index" in runtime_source


class _EffectReadForbidden(dict):
    """次数检查若尝试读取回报或配对差就立刻失败。"""

    _forbidden = {
        "groups",
        "arms",
        "D_net",
        "D_response",
        "D_cost",
        "episode_return",
    }

    def get(self, key, default=None):
        if key in self._forbidden:
            raise AssertionError(f"count-only path read effect field {key}")
        return super().get(key, default)


def test_formal_block_ledger_reads_identity_and_firing_indicator_only():
    payload = _EffectReadForbidden(
        _formal_identity_payload("path_c_r015_paired_block_v2")
    )
    expected = (1, "a" * 64, "prototype-0", 7, 1, 0)
    assert (
        formal_runtime._validate_formal_block_identity_without_effect_read(
            payload,
            expected=expected,
            runtime_binding_sha256="b" * 64,
        )
        is False
    )


def test_formal_resume_accepts_only_the_prefrozen_replacement_prefix(tmp_path):
    attempt_seeds = tuple(range(100, 132))
    expected = ((1, "a" * 64, "prototype-0", attempt_seeds, 1),)
    binding = "b" * 64
    invalid_path = tmp_path / "invalid.jsonl"
    row = {
        "schema_version": "path_c_r015_mechanical_replacement_attempt_v2",
        "sequence_index": 0,
        "round_index": 1,
        "audit_unit_id": "a" * 64,
        "prototype_id": "prototype-0",
        "episode_seed": attempt_seeds[0],
        "ego_position": 1,
        "mechanical_attempt_index": 0,
        "reason": "mechanical invalidity",
        "schedule_bound_seed": True,
        "outcome_field_read": False,
        "runtime_binding_sha256": "b" * 64,
    }
    invalid_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert formal_runtime._reconcile_formal_invalid_attempts(
        invalid_path=invalid_path,
        expected_coordinates=expected,
        ledger_rows=(),
        runtime_binding_sha256=binding,
    ) == {0: 1}

    runtime_source = inspect.getsource(formal_runtime.run_r015_formal)
    recovered = runtime_source.index(
        "selected_attempt_index = invalid_counts.get(sequence_index, 0)"
    )
    resumed_range = runtime_source.index(
        "for attempt_index in range(\n"
        "                selected_attempt_index,",
        recovered,
    )
    assert recovered < resumed_range

    tampered = {**row, "episode_seed": attempt_seeds[1]}
    invalid_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen replacement seed"):
        formal_runtime._reconcile_formal_invalid_attempts(
            invalid_path=invalid_path,
            expected_coordinates=expected,
            ledger_rows=(),
            runtime_binding_sha256=binding,
        )


def test_formal_resume_binds_the_successful_attempt_to_block_and_ledger(tmp_path):
    attempt_seeds = tuple(range(200, 232))
    expected = ((1, "a" * 64, "prototype-0", attempt_seeds, 1),)
    block_directory = tmp_path / "blocks"
    block_directory.mkdir()
    block_path = block_directory / "block.json"
    binding = "b" * 64
    block = {
        **_formal_identity_payload("path_c_r015_paired_block_v2"),
        "episode_seed": attempt_seeds[1],
        "mechanical_attempt_index": 1,
        "runtime_binding_sha256": binding,
    }
    block_path.write_text(json.dumps(block) + "\n", encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = {
        "schema_version": "path_c_r015_formal_block_ledger_v2",
        "sequence_index": 0,
        "round_index": 1,
        "audit_unit_id": "a" * 64,
        "prototype_id": "prototype-0",
        "episode_seed": attempt_seeds[1],
        "ego_position": 1,
        "mechanical_attempt_index": 1,
        "firing_indicator": False,
        "block_path": str(block_path),
        "block_sha256": artifacts._file_sha256(block_path),
        "runtime_binding_sha256": binding,
    }
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    rows, counts = formal_runtime._reconcile_formal_ledger(
        ledger_path=ledger_path,
        expected_coordinates=expected,
        block_directory=block_directory,
        runtime_binding_sha256=binding,
    )
    assert len(rows) == 1
    assert counts == {"prototype-0": 0}

    changed_runtime_block = {**block, "runtime_binding_sha256": "c" * 64}
    block_path.write_text(
        json.dumps(changed_runtime_block) + "\n", encoding="utf-8"
    )
    ledger["block_sha256"] = artifacts._file_sha256(block_path)
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime binding"):
        formal_runtime._reconcile_formal_ledger(
            ledger_path=ledger_path,
            expected_coordinates=expected,
            block_directory=block_directory,
            runtime_binding_sha256=binding,
        )

    block_path.write_text(json.dumps(block) + "\n", encoding="utf-8")
    ledger["block_sha256"] = artifacts._file_sha256(block_path)
    ledger["episode_seed"] = attempt_seeds[2]
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen schedule"):
        formal_runtime._reconcile_formal_ledger(
            ledger_path=ledger_path,
            expected_coordinates=expected,
            block_directory=block_directory,
            runtime_binding_sha256=binding,
        )


def test_final_dataset_rereads_both_ledgers_and_requires_complete_invalid_prefix(
    tmp_path,
):
    binding = "b" * 64
    attempt_seeds = tuple(range(300, 332))
    audit_unit_id = "a" * 64
    schedule = frozenset(
        {(1, audit_unit_id, "prototype-0", attempt_seeds, 1)}
    )
    block_directory = tmp_path / "paired_blocks"
    block_directory.mkdir()
    block_path = block_directory / "round_0001__prototype-0.json"
    block = {
        **_formal_identity_payload("path_c_r015_paired_block_v2"),
        "audit_unit_id": audit_unit_id,
        "episode_seed": attempt_seeds[1],
        "mechanical_attempt_index": 1,
        "runtime_binding_sha256": binding,
    }
    block_path.write_text(json.dumps(block) + "\n", encoding="utf-8")
    invalid_path = tmp_path / "mechanically_invalid_attempts.jsonl"
    invalid_row = {
        "schema_version": "path_c_r015_mechanical_replacement_attempt_v2",
        "sequence_index": 0,
        "round_index": 1,
        "audit_unit_id": audit_unit_id,
        "prototype_id": "prototype-0",
        "episode_seed": attempt_seeds[0],
        "ego_position": 1,
        "mechanical_attempt_index": 0,
        "reason": "mechanical invalidity",
        "schedule_bound_seed": True,
        "outcome_field_read": False,
        "runtime_binding_sha256": binding,
    }
    invalid_path.write_text(json.dumps(invalid_row) + "\n", encoding="utf-8")
    ledger_path = tmp_path / "formal_block_ledger.jsonl"
    ledger_row = {
        "schema_version": "path_c_r015_formal_block_ledger_v2",
        "sequence_index": 0,
        "round_index": 1,
        "audit_unit_id": audit_unit_id,
        "prototype_id": "prototype-0",
        "episode_seed": attempt_seeds[1],
        "ego_position": 1,
        "mechanical_attempt_index": 1,
        "firing_indicator": False,
        "block_path": str(block_path),
        "block_sha256": artifacts._file_sha256(block_path),
        "runtime_binding_sha256": binding,
    }
    ledger_path.write_text(json.dumps(ledger_row) + "\n", encoding="utf-8")

    dataset = {
        "runtime_binding_sha256": binding,
        "formal_block_ledger": {
            "path": str(ledger_path),
            "sha256": artifacts._file_sha256(ledger_path),
        },
        "mechanically_invalid_attempts": {
            "path": str(invalid_path),
            "sha256": artifacts._file_sha256(invalid_path),
        },
    }
    assert adjudication._validate_final_formal_execution_ledgers(
        dataset,
        raw_blocks=(block,),
        expected_sampling_schedule=schedule,
        prototype_order=("prototype-0",),
        expected_runtime_binding_sha256=binding,
    ) == binding
    with pytest.raises(ValueError, match="frozen runtime binding"):
        adjudication._validate_final_formal_execution_ledgers(
            dataset,
            raw_blocks=(block,),
            expected_sampling_schedule=schedule,
            prototype_order=("prototype-0",),
            expected_runtime_binding_sha256="c" * 64,
        )

    forged_block = {
        **block,
        "episode_seed": attempt_seeds[2],
        "mechanical_attempt_index": 2,
    }
    block_path.write_text(json.dumps(forged_block) + "\n", encoding="utf-8")
    forged_ledger = {
        **ledger_row,
        "episode_seed": attempt_seeds[2],
        "mechanical_attempt_index": 2,
        "block_sha256": artifacts._file_sha256(block_path),
    }
    ledger_path.write_text(json.dumps(forged_ledger) + "\n", encoding="utf-8")
    forged_dataset = {
        **dataset,
        "formal_block_ledger": {
            "path": str(ledger_path),
            "sha256": artifacts._file_sha256(ledger_path),
        },
    }
    with pytest.raises(ValueError, match="complete invalid prefix"):
        adjudication._validate_final_formal_execution_ledgers(
            forged_dataset,
            raw_blocks=(forged_block,),
            expected_sampling_schedule=schedule,
            prototype_order=("prototype-0",),
            expected_runtime_binding_sha256=binding,
        )


def test_dataset_assembly_binds_runtime_and_both_execution_ledgers():
    source = inspect.getsource(
        formal_runtime._assemble_formal_dataset_without_effect_read
    )
    assert '"path_c_r015_formal_dataset_v3"' in source
    assert '"runtime_binding_sha256"' in source
    assert '"formal_block_ledger"' in source
    assert '"mechanically_invalid_attempts"' in source
    assert "existing_bytes != candidate_bytes" in source
    runtime_source = inspect.getsource(formal_runtime.run_r015_formal)
    assert "_ensure_empty_jsonl_atomic(invalid_path)" in runtime_source
    assert '"runtime_binding_sha256": runtime_binding_sha256' in runtime_source
    assert "expected_runtime_binding_sha256=runtime_binding_sha256" in runtime_source
    final_source = inspect.getsource(adjudication.load_and_adjudicate_r015)
    assert "_validate_final_formal_execution_ledgers(" in final_source
    assert final_source.index("_validate_final_formal_execution_ledgers(") < (
        final_source.index("validate_r015_paired_block(")
    )


def test_count_checkpoint_contains_no_effect_value(tmp_path):
    path = tmp_path / "counts.json"
    payload = formal_runtime._write_count_checkpoint(
        path=path,
        checkpoint_round=400,
        firing_counts={f"prototype-{index}": 0 for index in range(4)},
    )
    assert set(payload) == {
        "schema_version",
        "experiment_id",
        "checkpoint_round_count",
        "firing_counts_by_prototype",
    }
    encoded = json.dumps(payload, sort_keys=True)
    for forbidden in ("return", "delta", "difference", "variance", "groups"):
        assert forbidden not in encoded.lower()


def test_formal_runtime_checks_each_firing_checkpoint_only_at_round_completion():
    """下一轮前三块不能用尚未完成的新一轮计数重写上一 checkpoint。"""

    source = inspect.getsource(formal_runtime.run_r015_formal)
    assert "round_completed_now = (" in source
    assert (
        "if round_completed_now and completed_round in (\n"
        "            preregistration.statistics.kill_checkpoint_rounds\n"
        "        ):"
    ) in source


def test_zero_fire_formal_run_stops_after_exactly_1600_paired_blocks():
    prototype_ids = tuple(f"prototype-{index}" for index in range(4))
    statistics = SimpleNamespace(
        kill_checkpoint_rounds=(200, 400, 800, 1600, 2500),
        partner_prototype_count=4,
        alpha_rho_per_checkpoint=0.001,
        alpha_rho_per_prototype_cp=0.00025,
        paired_difference_absolute_bound=800.0,
        practical_margin=5.0,
    )

    def decision(round_count: int):
        return adjudication.evaluate_r015_firing_count_checkpoint(
            {
                "schema_version": "path_c_r015_firing_count_checkpoint_v1",
                "experiment_id": "R015",
                "checkpoint_round_count": round_count,
                "firing_counts_by_prototype": {
                    prototype_id: 0 for prototype_id in prototype_ids
                },
            },
            statistics=statistics,
            expected_prototype_ids=prototype_ids,
        )

    assert decision(200).verdict == "CONTINUE"
    stopped = decision(400)
    assert stopped.verdict == "REGISTERED_NEGATIVE"
    assert stopped.total_fires == 0
    assert stopped.method == "zero_count_exponential_bound"
    assert 400 * len(prototype_ids) == 1600


def test_count_only_stop_precedes_dataset_assembly_and_effect_view():
    source = inspect.getsource(formal_runtime.run_r015_formal)
    first_kill = source.index("evaluate_r015_firing_count_checkpoint(")
    dataset = source.index("_assemble_formal_dataset_without_effect_read(")
    effect_view = source.rindex("load_and_adjudicate_r015(")
    assert first_kill < dataset < effect_view
    assert '"effect_value_read": False' in inspect.getsource(
        formal_runtime._kill_terminal_payload
    )
    assert '"effect_value_read": True' in inspect.getsource(
        formal_runtime._effect_terminal_payload
    )
    assert "formal_view_record_path.exists()" in source


def test_single_effect_view_is_only_after_all_2500_rounds():
    source = inspect.getsource(formal_runtime.run_r015_formal)
    effect_view = source.rindex("load_and_adjudicate_r015(")
    assert source.count("load_and_adjudicate_r015(") == 1
    prefix = source[:effect_view]
    assert "formal_paired_block_budget" in prefix
    assert "five count-only checkpoints" in prefix
    terminal_source = inspect.getsource(formal_runtime._effect_terminal_payload)
    assert "n_rounds_max" in terminal_source
    assert '"termination_kind": "single_effect_view_completed"' in terminal_source


def test_orphan_block_is_regenerated_with_the_same_frozen_request_before_counting():
    source = inspect.getsource(formal_runtime.run_r015_formal)
    orphan_start = source.index("if block_path.exists():")
    orphan_end = source.index("else:", orphan_start)
    orphan_path = source[orphan_start:orphan_end]
    assert "recovery_request = R015FormalBlockRequestV1(" in orphan_path
    assert "producer.produce_formal_paired_block(recovery_request)" in orphan_path
    assert "regenerated_block = _bind_formal_runtime_to_block(" in orphan_path
    assert orphan_path.count("_formal_orphan_replay_projection(") == 2
    assert orphan_path.index("_formal_orphan_replay_projection(") < orphan_path.index(
        "_validate_formal_block_identity_without_effect_read("
    )


def test_orphan_replay_projection_excludes_only_valid_nondeterministic_telemetry():
    execution = {
        "true_environment_transitions": 400,
        "computed_environment_transitions_including_masked_padding": 400,
        "compiled_batch_calls": 20,
        "jit_compilations": 1,
        "filter_particle_environment_transitions": 409600,
        "computed_filter_particle_environment_transitions": 409600,
        "segment_count": 20,
        "active_lane_batch_widths": [1] * 20,
        "wall_seconds": 1.0,
        "true_transitions_per_second": 400.0,
        "host_sync_inside_environment_loop": False,
    }
    telemetry = {
        "schema_version": "path_c_r015_formal_production_telemetry_v1",
        "actual_execution_wall_seconds": 1.0,
        "independent_replay_wall_seconds": 2.0,
        "independent_replay_reused_compiled_device_programs": True,
        "actual_execution": execution,
        "independent_replay_execution": execution,
        "actual_planning_real_environment_transitions": 730160,
        "independent_replay_planning_real_environment_transitions": 730160,
        "effect_values_in_telemetry": False,
    }
    persisted = {
        "trajectory_sha256": "a" * 64,
        "random_key_sha256": "b" * 64,
        "probe_decision": {
            "consultations": [
                {
                    "cost_accounting": {
                        "new_real_environment_transitions": 730160
                    }
                }
            ]
        },
        "mechanical_execution_telemetry": telemetry,
    }
    regenerated = {
        **persisted,
        "mechanical_execution_telemetry": {
            **telemetry,
            "actual_execution_wall_seconds": 3.0,
            "independent_replay_wall_seconds": 4.0,
            "actual_execution": {
                **execution,
                "wall_seconds": 3.0,
                "true_transitions_per_second": 133.33333333333334,
                "jit_compilations": 0,
            },
            "independent_replay_execution": {
                **execution,
                "wall_seconds": 4.0,
                "true_transitions_per_second": 100.0,
                "jit_compilations": 0,
            },
        },
    }
    assert formal_runtime._formal_orphan_replay_projection(
        persisted
    ) == formal_runtime._formal_orphan_replay_projection(regenerated)

    changed_transition_count = {
        **regenerated,
        "mechanical_execution_telemetry": {
            **regenerated["mechanical_execution_telemetry"],
            "actual_execution": {
                **regenerated["mechanical_execution_telemetry"]["actual_execution"],
                "true_environment_transitions": 399,
            },
            "independent_replay_execution": {
                **regenerated["mechanical_execution_telemetry"][
                    "independent_replay_execution"
                ],
                "true_environment_transitions": 399,
            },
        },
    }
    assert formal_runtime._formal_orphan_replay_projection(
        persisted
    ) != formal_runtime._formal_orphan_replay_projection(changed_transition_count)

    for invalid_time in (-1.0, float("inf"), float("nan"), True):
        invalid = {
            **persisted,
            "mechanical_execution_telemetry": {
                **telemetry,
                "actual_execution_wall_seconds": invalid_time,
            },
        }
        with pytest.raises(ValueError, match="nonnegative telemetry field"):
            formal_runtime._formal_orphan_replay_projection(invalid)

    invalid_rate = {
        **persisted,
        "mechanical_execution_telemetry": {
            **telemetry,
            "actual_execution": {
                **execution,
                "true_transitions_per_second": True,
            },
        },
    }
    with pytest.raises(ValueError, match="nonnegative telemetry field"):
        formal_runtime._formal_orphan_replay_projection(invalid_rate)


def test_effect_terminal_and_completed_receipt_form_an_acyclic_digest_chain(tmp_path):
    dataset_path = tmp_path / "formal_dataset.json"
    dataset_path.write_text("{}\n", encoding="utf-8")
    support_path = tmp_path / "support.json"
    support_path.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "view_receipt.json"
    terminal_path = tmp_path / "terminal.json"
    preregistration = SimpleNamespace(
        formal_dataset_path=dataset_path,
        source_sha256="a" * 64,
        statistics=SimpleNamespace(n_rounds_max=2500),
    )
    receipt_base = formal_runtime._effect_receipt_base(
        preregistration=preregistration,
        support_report_sha256=artifacts._file_sha256(support_path),
    )
    initial_receipt = {
        **receipt_base,
        "status": "consumed_before_adjudication",
    }
    formal_runtime._write_json_atomic(receipt_path, initial_receipt)
    started_sha256 = artifacts._file_sha256(receipt_path)
    decision_payload = {"verdict": "UNDECIDABLE"}
    terminal = {
        "schema_version": "path_c_r015_formal_terminal_v2",
        "formal_effect_view_started_sha256": started_sha256,
        "decision_sha256": formal_runtime._decision_sha256(decision_payload),
        "decision": decision_payload,
    }
    formal_runtime._write_json_atomic(terminal_path, terminal)
    receipt = formal_runtime._complete_or_validate_effect_receipt(
        receipt_path=receipt_path,
        receipt_base=receipt_base,
        terminal_path=terminal_path,
        terminal_payload=terminal,
    )
    assert receipt["status"] == "completed"
    assert receipt["started_receipt_sha256"] == started_sha256
    assert receipt["terminal_sha256"] == artifacts._file_sha256(terminal_path)
    assert receipt["decision_sha256"] == terminal["decision_sha256"]
    assert "formal_view_record_sha256" not in terminal


def test_existing_terminal_is_reconciled_after_ledgers_and_never_returned_shallowly():
    source = inspect.getsource(formal_runtime.run_r015_formal)
    terminal_seen = source.index("terminal_exists = terminal_path.exists()")
    ledger_reconciled = source.index("_reconcile_formal_ledger(")
    recovered_checkpoints = source.index(
        "evaluate_r015_firing_count_checkpoint(", ledger_reconciled
    )
    effect_recovery = source.index("if terminal_exists:", recovered_checkpoints)
    assert terminal_seen < ledger_reconciled < recovered_checkpoints < effect_recovery
    assert "return terminal\n" not in source[terminal_seen:ledger_reconciled]
    assert "_publish_or_verify_terminal(terminal_path, terminal)" in source
    assert "_complete_or_validate_effect_receipt(" in source
