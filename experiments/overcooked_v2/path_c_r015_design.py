"""R015 独立设计数据的机械选择与待签冻结清单装配。

本模块不运行正式审计，也不读取正式配对差。它只把预先登记的过滤器和规划候选规则
应用到独立设计证据，并把已经存在的文件按内容摘要装配成待签清单。
ESS（effective sample size）表示有效样本量，只用于判断粒子权重是否过度集中。
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from experiments.overcooked_v2.path_c_official_artifact import (
    FLAX_WEIGHTS_HASH_DOMAIN,
    OFFICIAL_EFFECTIVE_STEP_CONTRACTS,
    OFFICIAL_OP_FAMILY_ID,
    OFFICIAL_SP_FAMILY_ID,
    checkpoint_artifact_sha256,
    flax_weights_sha256,
    load_flax_parameter_tree,
    validate_official_artifact_manifest,
)


DESIGN_PROTOCOL_SCHEMA = "path_c_r015_design_data_protocol_v2"
FILTER_EVIDENCE_SCHEMA = "path_c_r015_filter_design_evidence_v2"
PLANNING_EVIDENCE_SCHEMA = "path_c_r015_planning_design_evidence_v2"
PLANNING_CANDIDATE_RESULT_SCHEMA = "path_c_r015_planning_candidate_result_v2"
PLANNING_SELECTION_SCHEMA = "path_c_r015_planning_selection_v2"
DESIGN_SELECTION_REPORT_SCHEMA = "path_c_r015_design_selection_report_v2"
SELECTED_FILTER_CHECKPOINT_SCHEMA = (
    "path_c_r015_selected_filter_consultation_checkpoint_v1"
)
SELECTED_FILTER_CHECKPOINT_MANIFEST_SCHEMA = (
    "path_c_r015_selected_filter_checkpoint_manifest_v1"
)
SELECTED_FILTER_CHECKPOINT_CODEC_ID = "path_c_r015_checkpoint_pickle_codec_v1"
FREEZE_INPUT_SCHEMA = "path_c_r015_freeze_manifest_inputs_v2"
FREEZE_MANIFEST_SCHEMA = "path_c_r015_freeze_manifest_v2"
FREEZE_VALIDATION_SCHEMA = "path_c_r015_freeze_manifest_validation_v2"
PREREGISTRATION_FILL_SCHEMA = "path_c_r015_preregistration_fill_values_v2"
FREEZE_ASSEMBLY_REPORT_SCHEMA = "path_c_r015_two_pass_freeze_assembly_v1"
FILTER_SELECTION_RULE_ID = "r015_filter_s1_s2_s3_first_pass_v2"
PLANNING_SELECTION_RULE_ID = "r015_r_vs_2r_exact_decision_agreement_v1"
DESIGN_SEED_CONTRACT_ID = "r015_design_role_isolation_v2"
PERCENTILE_RULE_ID = "nearest_rank_empirical_percentile_v1"

FILTER_PARTICLE_COUNTS = (64, 128, 256)
FILTER_RESAMPLING_TIMINGS = (
    "adaptive_ess_below_half_v1",
    "every_environment_step_v1",
)
FILTER_CANDIDATE_ORDER = tuple(
    (particle_count, timing)
    for particle_count in FILTER_PARTICLE_COUNTS
    for timing in FILTER_RESAMPLING_TIMINGS
)
FILTER_ALGORITHM_ID = "official_history_fully_adapted_particle_filter_v2"
FILTER_RESAMPLING_ALGORITHM_ID = "strict_systematic_per_prototype_v2"
FILTER_DEVICE_KEY_CONTRACT_ID = "r015_filter_device_fold_in_keys_v2"
FILTER_DEVICE_EXECUTION_ID = "r015_filter_jit_scan_vmap_v2"
FILTER_DEVICE_EQUIVALENCE_SCHEMA_V1 = "path_c_r015_filter_device_equivalence_v1"
FILTER_DEVICE_EQUIVALENCE_SCHEMA = "path_c_r015_filter_device_equivalence_v2"
FILTER_DEVICE_PREFIX_DIAGNOSTIC_SCHEMA = (
    "path_c_r015_filter_device_prefix_diagnostic_v1"
)
FILTER_DEVICE_EQUIVALENCE_CANDIDATE = (
    256,
    "adaptive_ess_below_half_v1",
)
SUPERSEDED_V1_DESIGN_HISTORIES_SHA256 = (
    "55bae9cd26b079cc628b0b18537f4fa7e9172fb880b5108f4c952bd38a180232"
)
PLANNING_BRANCH_COUNTS = (2, 4, 8)
CONSULTATION_STEPS = tuple(range(1, 101, 5))
REGISTERED_PROBE_IDS = ("up", "down", "right", "left", "stay", "interact")
# v2（数据前修订，Fable 复核 2026-07-15）：按回合序号在四个伙伴间轮换排序，
# 避免"最前 200 点"全部来自字典序第一个伙伴的信念域。
PLANNING_POINT_SAMPLING_RULE_ID = (
    "first_200_design_consultations_partner_interleaved_v2"
)
PLANNING_FILTER_STREAM_ID = "r015_filter_repeat_0_20260718_v2"
CONTINUATION_CONTROLLER_ID = "map_prototype_committed_cook_v1"
BRANCH_SAMPLING_RULE_ID = "sampled_hidden_state_branches_v1"
BRANCH_BELIEF_RULE_ID = "frozen_belief_branch_continuation_v1"
GRID_EVALUATION_RULE_ID = "lazy_ascending_first_pass"
FUTURE_RANDOM_KEY_DERIVATION_CONTRACT = (
    "path_c_r015_controller_key_v1_future_environment_index_v1"
)
NESTED_BRANCH_COUNTS = (2, 4, 8, 16)
UINT64_MODULUS = 1 << 64
S1_MAXIMUM_CLOSE_RATE = 0.02
S2_MAXIMUM_TV_P90 = 0.05
S3_MINIMUM_ESS_FRACTION_P10 = 0.2
PLANNING_MINIMUM_AGREEMENT = 0.95
PLANNING_MINIMUM_POINT_COUNT = 200
FILTER_DESIGN_EPISODE_COUNT = 80
FILTER_PARENT_SLOT_TARGET = 4096
FILTER_REPEAT_COUNT_PER_EPISODE = 2
FILTER_S1_EARLY_STOP_CLOSED_EPISODE_COUNT = 2
FILTER_CANDIDATE_COMPLETE_STATUS = "complete"
FILTER_CANDIDATE_S1_STOP_STATUS = "s1_irreversible_failure"
FILTER_OPENING_PREFIX_CACHE_SCHEMA = (
    "path_c_r015_conditioned_opening_particle_prefix_cache_v1"
)
FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID = (
    "r015_shared_conditioned_opening_256_particle_prefix_v1"
)
FILTER_OPENING_PREFIX_MAXIMUM_PARTICLES_PER_PROTOTYPE = 256
EXPECTED_DELIVERY_REWARD = 20.0
PRACTICAL_MARGIN_FRACTION = 0.25
PREFLIGHT_PREREGISTRATION_ARTIFACTS = frozenset(
    {
        "environment_config",
        "environment_source",
        "ego_evidence_contract",
        "formal_sampling_schedule",
        "response_projection",
        "response_vocabulary",
        "probe_registry",
        "official_history_filter",
        "continuation_planner",
        "continuation_controller",
        "decision_evidence_verifier",
        "return_bound_derivation",
        "wrong_delivery_detector",
        "planning_stability_report",
        "random_key_derivation",
        "safety_branch_evidence_verifier",
        "trace_replay_verifier",
    }
)
EXECUTION_ARTIFACTS = frozenset(
    {
        "production_runtime_bridge",
        "full_horizon_executor",
        "replay_backend",
    }
)
PROTOCOL_ARTIFACTS = frozenset({"design_data_protocol", "pilot_protocol"})
PILOT_WIRING_CHECKS = frozenset(
    {
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
)
EXPECTED_FREEZE_CHECKPOINTS = {
    100: {
        "role": "ego",
        "family_id": OFFICIAL_SP_FAMILY_ID,
        "effective_environment_steps": OFFICIAL_EFFECTIVE_STEP_CONTRACTS["rnn-sp"][
            "effective_environment_steps"
        ],
    },
    101: {
        "role": "partner",
        "family_id": OFFICIAL_SP_FAMILY_ID,
        "effective_environment_steps": OFFICIAL_EFFECTIVE_STEP_CONTRACTS["rnn-sp"][
            "effective_environment_steps"
        ],
    },
    102: {
        "role": "partner",
        "family_id": OFFICIAL_SP_FAMILY_ID,
        "effective_environment_steps": OFFICIAL_EFFECTIVE_STEP_CONTRACTS["rnn-sp"][
            "effective_environment_steps"
        ],
    },
    201: {
        "role": "partner",
        "family_id": OFFICIAL_OP_FAMILY_ID,
        "effective_environment_steps": OFFICIAL_EFFECTIVE_STEP_CONTRACTS["rnn-op"][
            "effective_environment_steps"
        ],
    },
    202: {
        "role": "partner",
        "family_id": OFFICIAL_OP_FAMILY_ID,
        "effective_environment_steps": OFFICIAL_EFFECTIVE_STEP_CONTRACTS["rnn-op"][
            "effective_environment_steps"
        ],
    },
}
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_HEX)


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return int(value)


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return int(value)


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence.")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _controller_key(root_key: str, *coordinates: object) -> str:
    if not _is_sha256(root_key):
        raise ValueError("R015 controller root key must be SHA-256.")
    return canonical_sha256(
        {
            "schema_version": "path_c_r015_controller_key_v1",
            "root_key": root_key,
            "coordinates": [str(item) for item in coordinates],
        }
    )


def _six_coordinate_random_key(
    *,
    audit_unit_id: str,
    partner_prototype_id: str,
    episode_seed: int,
    purpose: str,
    environment_step: int,
    branch_index: int = 0,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "path_c_r015_random_key_v1",
            "coordinates": {
                "audit_unit_id": str(audit_unit_id),
                "partner_prototype_id": str(partner_prototype_id),
                "episode_seed": _nonnegative_int(
                    episode_seed, field="episode_seed"
                ),
                "purpose": str(purpose),
                "environment_step": _nonnegative_int(
                    environment_step, field="environment_step"
                ),
                "branch_index": _nonnegative_int(
                    branch_index, field="branch_index"
                ),
            },
        }
    )


def _planning_branch_rollout_sha256(branch: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "canonical_slot_16": branch.get("canonical_slot_16"),
            "source_particle_index": branch.get("source_particle_index"),
            "source_particle_prototype_id": branch.get(
                "source_particle_prototype_id"
            ),
            "source_particle_state_sha256": branch.get(
                "source_particle_state_sha256"
            ),
            "source_belief_filter_algorithm_id": branch.get(
                "source_belief_filter_algorithm_id"
            ),
            "source_belief_resampling_algorithm": branch.get(
                "source_belief_resampling_algorithm"
            ),
            "common_random_key": branch.get("common_random_key"),
            "v_base_return": branch.get("v_base_return"),
            "base_trajectory_sha256": branch.get("base_trajectory_sha256"),
            "future_random_root_key": branch.get("future_random_root_key"),
            "future_random_derivation_contract_id": branch.get(
                "future_random_derivation_contract_id"
            ),
            "future_random_step_count": branch.get(
                "future_random_step_count"
            ),
            "future_random_sequence_sha256": branch.get(
                "future_random_sequence_sha256"
            ),
            "base_committed_member_id": branch.get("base_committed_member_id"),
            "base_frozen_belief_sha256": branch.get(
                "base_frozen_belief_sha256"
            ),
            "base_ego_action_sequence_sha256": branch.get(
                "base_ego_action_sequence_sha256"
            ),
            "base_partner_action_sequence_sha256": branch.get(
                "base_partner_action_sequence_sha256"
            ),
            "base_final_environment_state_sha256": branch.get(
                "base_final_environment_state_sha256"
            ),
            "base_final_partner_state_sha256": branch.get(
                "base_final_partner_state_sha256"
            ),
            "base_final_continuation_states_sha256": branch.get(
                "base_final_continuation_states_sha256"
            ),
            "candidates": branch.get("candidates"),
        }
    )


def _validate_planning_branch_random_summary(
    branch: Mapping[str, Any],
    *,
    remaining_environment_steps: int,
) -> None:
    """从根随机键重建完整序列，拒绝缺失或不一致的紧凑证据。"""

    root_key = branch.get("future_random_root_key")
    if root_key != branch.get("common_random_key") or not _is_sha256(root_key):
        raise ValueError("Planning branch changed its future-random root key.")
    if branch.get("future_random_derivation_contract_id") != (
        FUTURE_RANDOM_KEY_DERIVATION_CONTRACT
    ) or branch.get("future_random_step_count") != (
        remaining_environment_steps
    ):
        raise ValueError("Planning branch changed its future-random contract.")
    expected_digest = canonical_sha256(
        [
            _controller_key(root_key, "future_environment", index)
            for index in range(remaining_environment_steps)
        ]
    )
    if branch.get("future_random_sequence_sha256") != expected_digest:
        raise ValueError("Planning branch future-random sequence digest changed.")


def _canonical_nested_slot(
    sample_count: int, sample_slot: int, phase_uint64: int
) -> int:
    count = sample_count
    slot = sample_slot
    phase = phase_uint64
    while count < 16:
        doubled = 2 * phase
        parity = doubled // UINT64_MODULUS
        phase = doubled % UINT64_MODULUS
        slot = 2 * slot + int(parity)
        count *= 2
    return slot


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Required R015 file is missing: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _filter_opening_lane_binding(
    history: Mapping[str, Any],
    stream_id: str,
) -> Mapping[str, Any]:
    """绑定一条历史、一条重放流和候选无关的初始化根键。"""

    sequence_index = _positive_int(
        history.get("design_sequence_index"),
        field="filter opening-cache design_sequence_index",
    )
    episode_id = str(history.get("design_episode_id", ""))
    official_history = _require_sequence(
        history.get("official_history"),
        field="filter opening-cache official_history",
    )
    if not episode_id or len(official_history) != 401:
        raise ValueError("Filter opening cache requires one complete design history.")
    initialization_key = canonical_sha256(
        [FILTER_DEVICE_KEY_CONTRACT_ID, str(stream_id), sequence_index]
    )
    return {
        "design_episode_id": episode_id,
        "design_sequence_index": sequence_index,
        "filter_stream_id": str(stream_id),
        "filter_initialization_key": initialization_key,
        "official_history_sha256": canonical_sha256(official_history),
    }


def build_filter_opening_prefix_cache_binding(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    histories_path: Path,
    histories: Sequence[Mapping[str, Any]],
    stream_ids: Sequence[str],
) -> Mapping[str, Any]:
    """建立候选无关的开局粒子缓存绑定。

    缓存只保存已经按首个官方局部观测条件化的 256 个规范粒子前缀。
    粒子数为 64 或 128 的候选只能读取该前缀；不同重采样时机在第一次
    实际分歧后不得把扫描状态写回这个缓存。
    """

    resolved_protocol = Path(protocol_path).resolve()
    resolved_histories = Path(histories_path).resolve()
    source_root = Path(__file__).resolve().parent
    source_files = (
        Path(__file__).resolve(),
        (source_root / "path_c_r015_full_horizon.py").resolve(),
        (source_root / "official" / "r015_runtime_bridge.py").resolve(),
    )
    source_sha256 = {
        str(path.relative_to(source_root.parent)): file_sha256(path)
        for path in source_files
    }
    lane_bindings = [
        _filter_opening_lane_binding(history, str(stream_id))
        for history in histories
        for stream_id in stream_ids
    ]
    core = {
        "schema_version": FILTER_OPENING_PREFIX_CACHE_SCHEMA,
        "cache_contract_id": FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID,
        "filter_algorithm_id": FILTER_ALGORITHM_ID,
        "filter_key_contract": FILTER_DEVICE_KEY_CONTRACT_ID,
        "maximum_particles_per_prototype": (
            FILTER_OPENING_PREFIX_MAXIMUM_PARTICLES_PER_PROTOTYPE
        ),
        "protocol_file_sha256": file_sha256(resolved_protocol),
        "protocol_payload_sha256": canonical_sha256(protocol),
        "design_histories_file_sha256": file_sha256(resolved_histories),
        "design_histories_payload_sha256": canonical_sha256(histories),
        "source_file_sha256": source_sha256,
        "source_closure_sha256": canonical_sha256(source_sha256),
        "lane_bindings": lane_bindings,
        "stored_state_scope": "conditioned_opening_particles_only",
        "post_resampling_state_reuse_allowed": False,
    }
    return {**core, "cache_binding_sha256": canonical_sha256(core)}


def filter_opening_prefix_cache_binding_sha256(
    binding: Mapping[str, Any],
) -> str:
    """验证开局缓存本身的内容摘要和候选无关范围。"""

    actual = dict(_require_mapping(binding, field="filter opening-cache binding"))
    digest = actual.pop("cache_binding_sha256", None)
    if not _is_sha256(digest) or digest != canonical_sha256(actual):
        raise ValueError("Filter opening cache binding digest is invalid.")
    if (
        actual.get("schema_version") != FILTER_OPENING_PREFIX_CACHE_SCHEMA
        or actual.get("cache_contract_id")
        != FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID
        or actual.get("filter_algorithm_id") != FILTER_ALGORITHM_ID
        or actual.get("filter_key_contract") != FILTER_DEVICE_KEY_CONTRACT_ID
        or actual.get("maximum_particles_per_prototype")
        != FILTER_OPENING_PREFIX_MAXIMUM_PARTICLES_PER_PROTOTYPE
        or actual.get("stored_state_scope")
        != "conditioned_opening_particles_only"
        or actual.get("post_resampling_state_reuse_allowed") is not False
    ):
        raise ValueError("Filter opening cache changed its registered scope.")
    if not _is_sha256(actual.get("protocol_file_sha256")) or not _is_sha256(
        actual.get("protocol_payload_sha256")
    ) or not _is_sha256(actual.get("design_histories_file_sha256")) or not (
        _is_sha256(actual.get("design_histories_payload_sha256"))
    ) or not _is_sha256(actual.get("source_closure_sha256")):
        raise ValueError("Filter opening cache lacks a source or data binding.")
    source_files = _require_mapping(
        actual.get("source_file_sha256"), field="filter opening-cache sources"
    )
    if not source_files or any(not _is_sha256(value) for value in source_files.values()):
        raise ValueError("Filter opening cache source closure is incomplete.")
    if actual["source_closure_sha256"] != canonical_sha256(dict(source_files)):
        raise ValueError("Filter opening cache source closure digest is invalid.")
    lane_bindings = _require_sequence(
        actual.get("lane_bindings"), field="filter opening-cache lanes"
    )
    if not lane_bindings:
        raise ValueError("Filter opening cache contains no lane bindings.")
    lane_keys: set[tuple[str, str]] = set()
    for raw_lane in lane_bindings:
        lane = _require_mapping(raw_lane, field="filter opening-cache lane")
        if set(lane) != {
            "design_episode_id",
            "design_sequence_index",
            "filter_stream_id",
            "filter_initialization_key",
            "official_history_sha256",
        }:
            raise ValueError("Filter opening cache lane has an unknown field.")
        sequence_index = _positive_int(
            lane.get("design_sequence_index"),
            field="filter opening-cache lane design_sequence_index",
        )
        stream_id = str(lane.get("filter_stream_id", ""))
        expected_initialization_key = canonical_sha256(
            [FILTER_DEVICE_KEY_CONTRACT_ID, stream_id, sequence_index]
        )
        if not _is_sha256(lane.get("filter_initialization_key")) or not _is_sha256(
            lane.get("official_history_sha256")
        ) or lane.get("filter_initialization_key") != expected_initialization_key:
            raise ValueError("Filter opening cache lane lacks a history or key digest.")
        coordinate = (
            str(lane.get("design_episode_id", "")),
            stream_id,
        )
        if not all(coordinate) or coordinate in lane_keys:
            raise ValueError("Filter opening cache lanes are missing or duplicated.")
        lane_keys.add(coordinate)
    return str(digest)


def validate_filter_opening_prefix_cache_binding(
    binding: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> str:
    """拒绝来源、配置、初始化键或历史摘要不一致的开局缓存。"""

    digest = filter_opening_prefix_cache_binding_sha256(binding)
    expected_digest = filter_opening_prefix_cache_binding_sha256(expected)
    if dict(binding) != dict(expected) or digest != expected_digest:
        raise ValueError("Filter opening cache does not match the current source and data.")
    return digest


def load_mapping(path: str | Path) -> Mapping[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Required R015 mapping is missing: {target}")
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    return _require_mapping(payload, field=str(target))


def nearest_rank_percentile(values: Sequence[float], probability: float) -> float:
    """Return the registered empirical percentile without interpolation."""

    if not values:
        raise ValueError("An empirical percentile requires at least one value.")
    p = _finite(probability, field="percentile probability")
    if not 0.0 < p <= 1.0:
        raise ValueError("Percentile probability must lie in (0,1].")
    ordered = sorted(_finite(value, field="percentile value") for value in values)
    rank = max(1, math.ceil(p * len(ordered)))
    return ordered[rank - 1]


def zero_support_close_rate(closed_flags: Sequence[bool]) -> float:
    if not closed_flags or any(type(value) is not bool for value in closed_flags):
        raise ValueError("Zero-support evidence must be a non-empty Boolean sequence.")
    return sum(closed_flags) / len(closed_flags)


def filter_lane_batch_size(
    particles_per_prototype: int,
    *,
    prototype_count: int = 4,
) -> int:
    """返回恰好装满 4096 个父粒子槽的历史车道数。"""

    particles = _positive_int(
        particles_per_prototype,
        field="filter lane particles_per_prototype",
    )
    prototypes = _positive_int(prototype_count, field="filter lane prototype_count")
    slots_per_lane = particles * prototypes
    if FILTER_PARENT_SLOT_TARGET % slots_per_lane != 0:
        raise ValueError("Filter lanes must divide the fixed parent-slot target.")
    return FILTER_PARENT_SLOT_TARGET // slots_per_lane


def systematic_resample_indices_logarithmic(
    cumulative_weights: Any,
    positions: Any,
) -> np.ndarray:
    """用严格 ``cdf < position`` 规则找到系统重采样父粒子。"""

    cumulative = np.asarray(cumulative_weights)
    requested = np.asarray(positions)
    if cumulative.ndim != 1 or requested.ndim != 1 or cumulative.size == 0:
        raise ValueError("Systematic resampling requires two non-empty vectors.")
    if np.any(np.diff(cumulative) < 0.0):
        raise ValueError("Systematic resampling CDF must be non-decreasing.")
    if cumulative[-1] < 1.0 - 1.0e-12:
        raise ValueError("Systematic resampling CDF must end at one.")
    # ``side="left"`` is exactly the number of cumulative entries strictly below
    # each position, matching the registered cdf < position comparison.
    result = np.searchsorted(cumulative, requested, side="left")
    return np.minimum(result, cumulative.size - 1).astype(np.int32, copy=False)


def merge_filter_microbatch_results(
    chunks: Sequence[Mapping[str, Any]],
    *,
    expected_lane_count: int,
) -> Mapping[str, Any]:
    """按全局车道序号合并批次，并只对机械吞吐计数求和。"""

    lane_count = _positive_int(expected_lane_count, field="expected filter lane count")
    rows: list[Mapping[str, Any]] = []
    throughputs: list[Mapping[str, Any]] = []
    for raw_chunk in chunks:
        chunk = _require_mapping(raw_chunk, field="filter microbatch")
        chunk_rows = tuple(
            _require_mapping(item, field="filter microbatch lane")
            for item in _require_sequence(
                chunk.get("lane_results"), field="filter microbatch lanes"
            )
        )
        declared = _nonnegative_int(
            chunk.get("lane_count"), field="filter microbatch lane_count"
        )
        if declared != len(chunk_rows):
            raise ValueError("Filter microbatch lane count does not match its rows.")
        rows.extend(chunk_rows)
        throughputs.append(
            _require_mapping(chunk.get("throughput"), field="filter microbatch throughput")
        )
    try:
        ordered = sorted(rows, key=lambda item: int(item["lane_index"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Every filter lane requires one integer lane_index.") from error
    if [int(item["lane_index"]) for item in ordered] != list(range(lane_count)):
        raise ValueError("Filter lanes must appear exactly once in canonical order.")
    if any(bool(item.get("host_sync_inside_environment_loop")) for item in throughputs):
        raise ValueError("A filter microbatch synchronized with the host inside its scan.")
    transitions = sum(
        _nonnegative_int(
            item.get("true_particle_environment_transitions"),
            field="filter microbatch true transitions",
        )
        for item in throughputs
    )
    wall_seconds = sum(
        _finite(item.get("wall_seconds"), field="filter microbatch wall seconds")
        for item in throughputs
    )
    throughput = {
        "true_particle_environment_transitions": transitions,
        "compiled_batch_calls": sum(
            _nonnegative_int(item.get("compiled_batch_calls"), field="compiled calls")
            for item in throughputs
        ),
        "jit_compilations": sum(
            _nonnegative_int(item.get("jit_compilations"), field="jit compilations")
            for item in throughputs
        ),
        "active_lane_count": lane_count,
        "active_particle_batch_width": max(
            _positive_int(
                item.get("active_particle_batch_width"),
                field="active particle batch width",
            )
            for item in throughputs
        ),
        "wall_seconds": wall_seconds,
        "true_transitions_per_second": (
            transitions / wall_seconds if wall_seconds > 0.0 else None
        ),
        "host_sync_inside_environment_loop": False,
    }
    return {"lane_results": ordered, "throughput": throughput}


def s1_irreversible_failure(
    closed_episode_count: int,
    *,
    total_episode_count: int = FILTER_DESIGN_EPISODE_COUNT,
    threshold: float = S1_MAXIMUM_CLOSE_RATE,
) -> bool:
    """判断已经发生的关闭次数是否使最终 S1 必然失败。"""

    closed = _nonnegative_int(closed_episode_count, field="S1 closed episode count")
    total = _positive_int(total_episode_count, field="S1 total episode count")
    limit = _finite(threshold, field="S1 threshold")
    if not 0.0 <= limit <= 1.0 or closed > total:
        raise ValueError("S1 irreversible-failure inputs are outside their range.")
    return closed / total > limit


@dataclass(frozen=True)
class FilterS1Progress:
    """记录零支持关闭次数；一旦最终 S1 必定失败就保持失败状态。"""

    total_episode_count: int
    processed_episode_count: int = 0
    zero_support_close_count: int = 0

    def __post_init__(self) -> None:
        total = _positive_int(
            self.total_episode_count,
            field="S1 total episode count",
        )
        processed = _nonnegative_int(
            self.processed_episode_count,
            field="S1 processed episode count",
        )
        closed = _nonnegative_int(
            self.zero_support_close_count,
            field="S1 zero-support close count",
        )
        if processed > total:
            raise ValueError("S1 processed episode count exceeds its frozen total.")
        if closed > processed:
            raise ValueError("S1 close count exceeds its processed episode count.")

    @property
    def maximum_passing_close_count(self) -> int:
        candidate = math.floor(S1_MAXIMUM_CLOSE_RATE * self.total_episode_count)
        while candidate / self.total_episode_count > S1_MAXIMUM_CLOSE_RATE:
            candidate -= 1
        while (
            candidate + 1
        ) / self.total_episode_count <= S1_MAXIMUM_CLOSE_RATE:
            candidate += 1
        return candidate

    @property
    def irreversible_failure(self) -> bool:
        return self.zero_support_close_count > self.maximum_passing_close_count

    @property
    def complete(self) -> bool:
        return self.processed_episode_count == self.total_episode_count


def start_filter_s1_progress(
    *,
    total_episode_count: int = FILTER_DESIGN_EPISODE_COUNT,
) -> FilterS1Progress:
    """建立不含证据内容的 S1 运行进度；默认 80 回合最多允许关闭 1 次。"""

    return FilterS1Progress(total_episode_count=total_episode_count)


def advance_filter_s1_progress(
    progress: FilterS1Progress,
    *,
    closed_for_zero_support: bool,
) -> FilterS1Progress:
    """加入一个回合；第二次关闭后默认状态不可逆地标记 S1 失败。"""

    if not isinstance(progress, FilterS1Progress):
        raise TypeError("S1 progress must be FilterS1Progress.")
    if type(closed_for_zero_support) is not bool:
        raise TypeError("S1 zero-support closure must be Boolean.")
    if progress.irreversible_failure:
        raise RuntimeError("S1 already failed irreversibly; no later episode can repair it.")
    if progress.complete:
        raise RuntimeError("S1 progress already contains its frozen episode count.")
    return FilterS1Progress(
        total_episode_count=progress.total_episode_count,
        processed_episode_count=progress.processed_episode_count + 1,
        zero_support_close_count=(
            progress.zero_support_close_count + int(closed_for_zero_support)
        ),
    )


def total_variation_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    if not left or set(left) != set(right):
        raise ValueError("Posterior vectors must have the same non-empty support.")
    left_values = {
        str(key): _finite(value, field=f"left posterior {key}")
        for key, value in left.items()
    }
    right_values = {
        str(key): _finite(value, field=f"right posterior {key}")
        for key, value in right.items()
    }
    if any(value < 0.0 for value in (*left_values.values(), *right_values.values())):
        raise ValueError("Posterior probabilities must be non-negative.")
    if not math.isclose(sum(left_values.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The left posterior is not normalized.")
    if not math.isclose(sum(right_values.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The right posterior is not normalized.")
    return 0.5 * sum(
        abs(left_values[key] - right_values[key]) for key in left_values
    )


def summarize_filter_statistics(
    *,
    zero_support_closed: Sequence[bool],
    posterior_pairs: Sequence[tuple[Mapping[str, float], Mapping[str, float]]],
    ess_fractions: Sequence[float],
) -> Mapping[str, Any]:
    """Compute S1--S3 from already isolated design evidence."""

    tv_values = [total_variation_distance(left, right) for left, right in posterior_pairs]
    if not tv_values:
        raise ValueError("S2 requires at least one paired consultation posterior.")
    if not ess_fractions:
        raise ValueError("S3 requires at least one consultation ESS fraction.")
    normalized_ess = [
        _finite(value, field="ESS fraction") for value in ess_fractions
    ]
    if any(not 0.0 < value <= 1.0 for value in normalized_ess):
        raise ValueError("ESS/N values must lie in (0,1].")
    s1 = zero_support_close_rate(zero_support_closed)
    s2 = nearest_rank_percentile(tv_values, 0.90)
    s3 = nearest_rank_percentile(normalized_ess, 0.10)
    return {
        "percentile_rule_id": PERCENTILE_RULE_ID,
        "s1_zero_support_close_rate": s1,
        "s2_posterior_tv_p90": s2,
        "s3_ess_fraction_p10": s3,
        "s1_pass": s1 <= S1_MAXIMUM_CLOSE_RATE,
        "s2_pass": s2 <= S2_MAXIMUM_TV_P90,
        "s3_pass": s3 >= S3_MINIMUM_ESS_FRACTION_P10,
        "passed": (
            s1 <= S1_MAXIMUM_CLOSE_RATE
            and s2 <= S2_MAXIMUM_TV_P90
            and s3 >= S3_MINIMUM_ESS_FRACTION_P10
        ),
        "episode_count": len(zero_support_closed),
        "paired_consultation_count": len(tv_values),
        "ess_observation_count": len(normalized_ess),
    }


def _normalized_posterior(
    value: Any,
    *,
    prototype_ids: tuple[str, ...],
    field: str,
) -> Mapping[str, float]:
    raw = _require_mapping(value, field=field)
    if set(raw) != set(prototype_ids):
        raise ValueError(f"{field} must cover the registered four prototypes.")
    normalized = {
        prototype_id: _finite(raw[prototype_id], field=f"{field}.{prototype_id}")
        for prototype_id in prototype_ids
    }
    if any(value < 0.0 for value in normalized.values()) or not math.isclose(
        sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"{field} must be a normalized posterior.")
    return normalized


def evaluate_filter_candidate(
    evidence: Mapping[str, Any],
    *,
    prototype_ids: Sequence[str],
    required_episodes_per_prototype: int = 20,
) -> Mapping[str, Any]:
    """验证一个完整候选，或验证已被 S1 固定分母判死的规范前缀。"""

    if evidence.get("schema_version") != FILTER_EVIDENCE_SCHEMA:
        raise ValueError("Filter design evidence has the wrong schema.")
    embedded_result = evidence.get("candidate_result")

    def finish(result: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized = dict(result)
        if embedded_result is not None and dict(
            _require_mapping(
                embedded_result,
                field="embedded filter candidate result",
            )
        ) != normalized:
            raise ValueError("Embedded filter candidate result does not match evidence.")
        return normalized
    if evidence.get("scientific_readout_allowed") is not False or evidence.get(
        "filter_algorithm_id"
    ) != FILTER_ALGORITHM_ID or evidence.get("filter_key_contract") != (
        FILTER_DEVICE_KEY_CONTRACT_ID
    ):
        raise ValueError("Filter evidence changed its honesty or algorithm identity.")
    particles = _positive_int(
        evidence.get("particles_per_prototype"),
        field="filter evidence particles_per_prototype",
    )
    timing = str(evidence.get("resampling_timing", ""))
    if (particles, timing) not in FILTER_CANDIDATE_ORDER:
        raise ValueError("Filter evidence uses an unregistered candidate.")
    if evidence.get("resampling_algorithm") != FILTER_RESAMPLING_ALGORITHM_ID:
        raise ValueError("Filter evidence changed the systematic resampling rule.")
    opening_cache_binding = _require_mapping(
        evidence.get("conditioned_opening_prefix_cache_binding"),
        field="filter opening-cache binding",
    )
    opening_cache_binding_sha256 = filter_opening_prefix_cache_binding_sha256(
        opening_cache_binding
    )
    raw_opening_prefix_digests = _require_mapping(
        evidence.get("conditioned_opening_prefix_sha256_by_particle_count"),
        field="filter conditioned-opening prefix digests",
    )
    opening_prefix_digests = {
        str(count): str(digest)
        for count, digest in raw_opening_prefix_digests.items()
    }
    if any(
        count not in {"64", "128", "256"} or not _is_sha256(digest)
        for count, digest in opening_prefix_digests.items()
    ):
        raise ValueError("Filter evidence has an invalid conditioned-opening prefix digest.")
    raw_opening_attempt_digests = _require_mapping(
        evidence.get("conditioned_opening_attempt_sha256_by_particle_count"),
        field="filter conditioned-opening attempt digests",
    )
    opening_attempt_digests = {
        str(count): str(digest)
        for count, digest in raw_opening_attempt_digests.items()
    }
    if any(
        count not in {"64", "128", "256"} or not _is_sha256(digest)
        for count, digest in opening_attempt_digests.items()
    ) or not _is_sha256(opening_attempt_digests.get(str(particles))):
        raise ValueError("Filter evidence lacks its conditioned-opening attempt digest.")
    throughput = _require_mapping(evidence.get("throughput"), field="filter throughput")
    if throughput.get("execution_id") != FILTER_DEVICE_EXECUTION_ID or throughput.get(
        "filter_algorithm_id"
    ) != FILTER_ALGORITHM_ID or throughput.get("filter_key_contract") != (
        FILTER_DEVICE_KEY_CONTRACT_ID
    ) or throughput.get("microbatch_schedule_id") != (
        "fixed_4096_parent_particle_slots_v1"
    ) or throughput.get("formal_fixed_parent_slot_schedule") is not True or (
        throughput.get("configured_parent_slot_batch_width")
        != FILTER_PARENT_SLOT_TARGET
    ) or throughput.get("lanes_per_microbatch") != filter_lane_batch_size(particles) or (
        throughput.get("host_sync_inside_environment_loop") is not False
    ) or throughput.get("conditioned_opening_prefix_cache_contract_id") != (
        FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID
    ) or throughput.get("conditioned_opening_prefix_cache_binding_sha256") != (
        opening_cache_binding_sha256
    ) or not 0 <= _nonnegative_int(
        throughput.get("conditioned_opening_prefix_particles_used_per_prototype"),
        field="filter reused opening-prefix particles",
    ) <= particles or throughput.get("post_resampling_state_reuse_count") != 0:
        raise ValueError("Filter evidence did not use the registered device schedule.")
    ids = tuple(str(item) for item in prototype_ids)
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("Filter design requires four distinct prototypes.")
    episodes = _require_sequence(evidence.get("episodes"), field="filter episodes")
    required = _positive_int(
        required_episodes_per_prototype,
        field="required episodes per prototype",
    )
    total_episode_count = len(ids) * required
    if total_episode_count != FILTER_DESIGN_EPISODE_COUNT:
        raise ValueError("Filter design evidence must retain the frozen 80 episodes.")
    progress = _require_mapping(
        evidence.get("candidate_progress"), field="filter candidate progress"
    )
    progress_status = str(progress.get("status", ""))
    processed_episode_count = _positive_int(
        progress.get("processed_episode_count"),
        field="filter processed episode count",
    )
    zero_support_close_count = _nonnegative_int(
        progress.get("zero_support_close_count"),
        field="filter zero-support close count",
    )
    if progress_status not in {
        FILTER_CANDIDATE_COMPLETE_STATUS,
        FILTER_CANDIDATE_S1_STOP_STATUS,
    } or progress.get("total_episode_count") != total_episode_count or (
        progress.get("processed_lane_count")
        != processed_episode_count * FILTER_REPEAT_COUNT_PER_EPISODE
    ) or progress.get("stop_candidate_at_closed_episode_count") != (
        FILTER_S1_EARLY_STOP_CLOSED_EPISODE_COUNT
    ):
        raise ValueError("Filter candidate progress changed the frozen S1 rule.")
    if processed_episode_count != len(episodes):
        raise ValueError("Filter candidate progress does not match its episode prefix.")
    processed_lane_count = processed_episode_count * FILTER_REPEAT_COUNT_PER_EPISODE
    microbatch_lane_counts = tuple(
        _positive_int(value, field="filter microbatch active lane count")
        for value in _require_sequence(
            throughput.get("microbatch_active_lane_counts"),
            field="filter microbatch active lane counts",
        )
    )
    if throughput.get("active_lane_count") != processed_lane_count or sum(
        microbatch_lane_counts
    ) != processed_lane_count or throughput.get("microbatch_count") != len(
        microbatch_lane_counts
    ) or throughput.get("compiled_batch_calls") != len(microbatch_lane_counts) or any(
        value != filter_lane_batch_size(particles)
        for value in microbatch_lane_counts
    ):
        raise ValueError("Filter throughput does not match the processed episode prefix.")
    cached_particles_by_microbatch = tuple(
        _nonnegative_int(
            value,
            field="filter microbatch reused opening-prefix particles",
        )
        for value in _require_sequence(
            throughput.get(
                "microbatch_conditioned_opening_prefix_particles_used_per_prototype"
            ),
            field="filter microbatch reused opening-prefix particles",
        )
    )
    reused_opening_slots = _nonnegative_int(
        throughput.get("conditioned_opening_prefix_reused_particle_slots"),
        field="filter reused opening-prefix slots",
    )
    generated_opening_slots = _nonnegative_int(
        throughput.get("conditioned_opening_prefix_generated_particle_slots"),
        field="filter generated opening-prefix slots",
    )
    expected_reused_opening_slots = sum(
        cached * lanes * len(ids)
        for cached, lanes in zip(
            cached_particles_by_microbatch,
            microbatch_lane_counts,
        )
    )
    maximum_generated_opening_slots = sum(
        (particles - cached) * lanes * len(ids)
        for cached, lanes in zip(
            cached_particles_by_microbatch,
            microbatch_lane_counts,
        )
    )
    if (
        len(cached_particles_by_microbatch) != len(microbatch_lane_counts)
        or any(value > particles for value in cached_particles_by_microbatch)
        or min(cached_particles_by_microbatch, default=0)
        != throughput.get(
            "conditioned_opening_prefix_particles_used_per_prototype"
        )
        or reused_opening_slots != expected_reused_opening_slots
        or generated_opening_slots > maximum_generated_opening_slots
    ):
        raise ValueError("Filter opening-prefix throughput accounting is inconsistent.")
    if progress_status == FILTER_CANDIDATE_COMPLETE_STATUS:
        if processed_episode_count != total_episode_count:
            raise ValueError("A complete filter candidate must contain all 80 episodes.")
    elif not 0 < processed_episode_count < total_episode_count:
        raise ValueError("An S1-stopped filter candidate must be a proper episode prefix.")
    metric_status = _require_mapping(
        evidence.get("metric_computation"), field="filter metric computation"
    )
    expected_metric_status = (
        {"s1": "computed", "s2": "computed", "s3": "computed"}
        if progress_status == FILTER_CANDIDATE_COMPLETE_STATUS
        else {
            "s1": "failed_from_fixed_denominator_lower_bound",
            "s2": "not_computed_due_to_irreversible_s1_failure",
            "s3": "not_computed_due_to_irreversible_s1_failure",
        }
    )
    if dict(metric_status) != expected_metric_status:
        raise ValueError("Filter evidence metric status does not match its progress.")

    bound_lanes = tuple(
        _require_mapping(item, field="filter opening-cache lane")
        for item in _require_sequence(
            opening_cache_binding.get("lane_bindings"),
            field="filter opening-cache lanes",
        )
    )
    canonical_episode_order: list[Mapping[str, Any]] = []
    bound_lane_count_by_episode: dict[str, int] = {}
    previous_bound_episode_id: str | None = None
    for lane in bound_lanes:
        episode_id = str(lane.get("design_episode_id", ""))
        sequence_index = _positive_int(
            lane.get("design_sequence_index"),
            field="filter opening-cache design sequence index",
        )
        bound_lane_count_by_episode[episode_id] = (
            bound_lane_count_by_episode.get(episode_id, 0) + 1
        )
        if episode_id != previous_bound_episode_id:
            if episode_id in bound_lane_count_by_episode and (
                bound_lane_count_by_episode[episode_id] != 1
            ):
                raise ValueError("Filter opening-cache episode lanes are not adjacent.")
            canonical_episode_order.append(
                {
                    "design_episode_id": episode_id,
                    "design_sequence_index": sequence_index,
                }
            )
            previous_bound_episode_id = episode_id
        elif canonical_episode_order[-1]["design_sequence_index"] != sequence_index:
            raise ValueError("Filter opening-cache repeats changed the design sequence.")
    if len(canonical_episode_order) != total_episode_count or any(
        count != FILTER_REPEAT_COUNT_PER_EPISODE
        for count in bound_lane_count_by_episode.values()
    ):
        raise ValueError("Filter opening cache must bind two repeats for all 80 episodes.")
    design_episode_set_sha256 = canonical_sha256(canonical_episode_order)

    counts = {prototype_id: 0 for prototype_id in ids}
    episode_ids: set[str] = set()
    episode_seeds: set[int] = set()
    closed_flags: list[bool] = []
    posterior_pairs: list[
        tuple[Mapping[str, float], Mapping[str, float]]
    ] = []
    ess_fractions: list[float] = []
    episode_identity: list[Mapping[str, Any]] = []
    expected_opening_cache_coordinates: set[tuple[str, int, str, str]] = set()
    for raw_episode in episodes:
        episode = _require_mapping(raw_episode, field="filter episode")
        episode_id = str(episode.get("design_episode_id", ""))
        partner_id = str(episode.get("partner_prototype_id", ""))
        seed = _nonnegative_int(episode.get("episode_seed"), field="design episode seed")
        design_sequence_index = _positive_int(
            episode.get("design_sequence_index"),
            field="design sequence index",
        )
        if not episode_id or episode_id in episode_ids or seed in episode_seeds:
            raise ValueError("Design episode ids and seeds must each be unique.")
        if partner_id not in counts:
            raise ValueError("Filter evidence names an unregistered partner prototype.")
        episode_ids.add(episode_id)
        episode_seeds.add(seed)
        counts[partner_id] += 1
        closed = episode.get("closed_for_zero_support")
        if type(closed) is not bool:
            raise ValueError("Every design episode must state zero-support closure.")
        closed_flags.append(closed)
        repeats = _require_sequence(episode.get("filter_repeats"), field="filter repeats")
        if len(repeats) != 2:
            raise ValueError("Every design episode requires two independent filter repeats.")
        by_stream: dict[str, Mapping[int, Mapping[str, Any]]] = {}
        repeat_closed_flags: list[bool] = []
        for raw_repeat in repeats:
            repeat = _require_mapping(raw_repeat, field="filter repeat")
            stream_id = str(repeat.get("filter_stream_id", ""))
            if not stream_id or stream_id in by_stream:
                raise ValueError("Filter repeat stream ids must be distinct and non-empty.")
            expected_initialization_key = canonical_sha256(
                [
                    FILTER_DEVICE_KEY_CONTRACT_ID,
                    stream_id,
                    design_sequence_index,
                ]
            )
            if repeat.get("filter_initialization_key") != expected_initialization_key:
                raise ValueError("Filter repeat initialization key changed its v2 coordinates.")
            expected_opening_cache_coordinates.add(
                (
                    episode_id,
                    design_sequence_index,
                    stream_id,
                    expected_initialization_key,
                )
            )
            repeat_closed = repeat.get("closed_for_zero_support")
            if type(repeat_closed) is not bool:
                raise ValueError("Each filter repeat must state zero-support closure.")
            repeat_closed_flags.append(repeat_closed)
            consultations = _require_sequence(
                repeat.get("consultations"), field="filter consultations"
            )
            by_step: dict[int, Mapping[str, Any]] = {}
            if progress_status == FILTER_CANDIDATE_S1_STOP_STATUS:
                # S1 已由固定 80 回合分母判死；前缀内即使已有咨询点，也不读取
                # 后验或有效样本量来计算、覆盖或排序 S2/S3。
                by_stream[stream_id] = by_step
                continue
            for raw_consultation in consultations:
                consultation = _require_mapping(
                    raw_consultation, field="filter consultation"
                )
                step = _positive_int(
                    consultation.get("environment_step"),
                    field="filter consultation environment_step",
                )
                if step not in CONSULTATION_STEPS or step in by_step:
                    raise ValueError("Filter consultations must use unique registered steps.")
                posterior = _normalized_posterior(
                    consultation.get("prototype_posterior"),
                    prototype_ids=ids,
                    field="filter consultation prototype_posterior",
                )
                raw_ess = _require_mapping(
                    consultation.get("pre_resample_ess_fraction_by_prototype"),
                    field="filter consultation ESS fractions",
                )
                if set(raw_ess) != set(ids):
                    raise ValueError("ESS evidence must cover all four prototypes.")
                for prototype_id in ids:
                    ess = _finite(
                        raw_ess[prototype_id],
                        field=f"ESS fraction {prototype_id}",
                    )
                    if not 0.0 < ess <= 1.0:
                        raise ValueError("ESS/N values must lie in (0,1].")
                    ess_fractions.append(ess)
                by_step[step] = {
                    "prototype_posterior": posterior,
                }
            if not repeat_closed and tuple(sorted(by_step)) != CONSULTATION_STEPS:
                raise ValueError("A complete design episode must expose all consultations.")
            by_stream[stream_id] = by_step
        if closed is not any(repeat_closed_flags):
            raise ValueError("Episode zero-support closure must equal the two-repeat union.")
        stream_steps = [set(value) for value in by_stream.values()]
        first, second = tuple(by_stream.values())
        for step in sorted(stream_steps[0].intersection(stream_steps[1])):
            posterior_pairs.append(
                (
                    first[step]["prototype_posterior"],
                    second[step]["prototype_posterior"],
                )
            )
        episode_identity.append(
            {
                "design_episode_id": episode_id,
                "partner_prototype_id": partner_id,
                "episode_seed": seed,
                "design_sequence_index": design_sequence_index,
            }
        )
    processed_identity_prefix = [
        {
            "design_episode_id": item["design_episode_id"],
            "design_sequence_index": item["design_sequence_index"],
        }
        for item in episode_identity
    ]
    if processed_identity_prefix != canonical_episode_order[:processed_episode_count]:
        raise ValueError("Filter candidate episodes are not the canonical design prefix.")
    if progress_status == FILTER_CANDIDATE_COMPLETE_STATUS and any(
        count != required for count in counts.values()
    ):
        raise ValueError("A complete filter design must contain 20 episodes per prototype.")
    bound_opening_cache_coordinates = {
        (
            str(lane["design_episode_id"]),
            int(lane["design_sequence_index"]),
            str(lane["filter_stream_id"]),
            str(lane["filter_initialization_key"]),
        )
        for lane in _require_sequence(
            opening_cache_binding.get("lane_bindings"),
            field="filter opening-cache lanes",
        )
    }
    if progress_status == FILTER_CANDIDATE_COMPLETE_STATUS:
        if bound_opening_cache_coordinates != expected_opening_cache_coordinates:
            raise ValueError("Filter opening cache does not cover the evidence lanes exactly.")
    elif not expected_opening_cache_coordinates < bound_opening_cache_coordinates:
        raise ValueError("S1-stopped evidence is not a strict opening-cache prefix.")

    actual_close_count = sum(closed_flags)
    if zero_support_close_count != actual_close_count:
        raise ValueError("Filter candidate progress changed its zero-support close count.")
    s1_lower_bound = zero_support_close_count / total_episode_count
    if not math.isclose(
        _finite(
            progress.get("s1_close_rate_lower_bound"),
            field="filter S1 close-rate lower bound",
        ),
        s1_lower_bound,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Filter candidate S1 lower bound does not match its counts.")

    common_result = {
        "schema_version": "path_c_r015_filter_candidate_result_v2",
        "selection_rule_id": FILTER_SELECTION_RULE_ID,
        "particles_per_prototype": particles,
        "filter_algorithm_id": FILTER_ALGORITHM_ID,
        "resampling_algorithm": FILTER_RESAMPLING_ALGORITHM_ID,
        "resampling_timing": timing,
        "candidate_evaluation_status": progress_status,
        "total_episode_count": total_episode_count,
        "processed_episode_count": processed_episode_count,
        "zero_support_close_count": zero_support_close_count,
        "s1_zero_support_close_rate_lower_bound": s1_lower_bound,
        "conditioned_opening_prefix_cache_binding_sha256": (
            opening_cache_binding_sha256
        ),
        "conditioned_opening_prefix_sha256_by_particle_count": (
            opening_prefix_digests
        ),
        "conditioned_opening_attempt_sha256_by_particle_count": (
            opening_attempt_digests
        ),
        "conditioned_opening_prefix_scope": (
            "all_design_episodes"
            if progress_status == FILTER_CANDIDATE_COMPLETE_STATUS
            else "processed_episode_prefix"
        ),
        "design_episode_set_sha256": design_episode_set_sha256,
    }
    if progress_status == FILTER_CANDIDATE_S1_STOP_STATUS:
        if not s1_irreversible_failure(
            zero_support_close_count,
            total_episode_count=total_episode_count,
            threshold=S1_MAXIMUM_CLOSE_RATE,
        ) or s1_lower_bound <= S1_MAXIMUM_CLOSE_RATE:
            raise ValueError("S1-stopped evidence did not prove irreversible failure.")
        return finish({
            **common_result,
            "s1_zero_support_close_rate": None,
            "s2_posterior_tv_p90": None,
            "s3_ess_fraction_p10": None,
            "s1_pass": False,
            "s2_pass": None,
            "s3_pass": None,
            "s2_status": "not_computed_due_to_irreversible_s1_failure",
            "s3_status": "not_computed_due_to_irreversible_s1_failure",
            "passed": False,
            "episode_count": processed_episode_count,
            "paired_consultation_count": None,
            "ess_observation_count": None,
        })

    summary = summarize_filter_statistics(
        zero_support_closed=closed_flags,
        posterior_pairs=posterior_pairs,
        ess_fractions=ess_fractions,
    )
    if not math.isclose(
        float(summary["s1_zero_support_close_rate"]),
        s1_lower_bound,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Complete filter S1 does not match its progress counts.")
    if bool(summary["passed"]) and not _is_sha256(
        opening_prefix_digests.get(str(particles))
    ):
        raise ValueError("A selected-capable filter lacks a complete opening prefix.")
    return finish({
        **common_result,
        "s2_status": "computed",
        "s3_status": "computed",
        **summary,
    })


def select_filter_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not 1 <= len(candidate_results) <= len(FILTER_CANDIDATE_ORDER):
        raise ValueError("Filter selection requires a non-empty registered prefix.")
    by_candidate: dict[tuple[int, str], Mapping[str, Any]] = {}
    episode_set_sha256: str | None = None
    opening_cache_binding_sha256: str | None = None
    opening_prefix_digest_by_particle_count: dict[str, str] = {}
    opening_attempt_digest_by_particle_count: dict[str, str] = {}
    opening_attempt_scope_by_particle_count: dict[str, tuple[int, str]] = {}
    for raw_result in candidate_results:
        result = _require_mapping(raw_result, field="filter candidate result")
        if result.get("schema_version") != "path_c_r015_filter_candidate_result_v2":
            raise ValueError("Filter candidate result has the wrong schema.")
        if result.get("selection_rule_id") != FILTER_SELECTION_RULE_ID or result.get(
            "resampling_algorithm"
        ) != FILTER_RESAMPLING_ALGORITHM_ID or result.get(
            "filter_algorithm_id"
        ) != FILTER_ALGORITHM_ID:
            raise ValueError("Filter candidate result changed the registered mechanics.")
        key = (
            _positive_int(
                result.get("particles_per_prototype"),
                field="filter result particles_per_prototype",
            ),
            str(result.get("resampling_timing", "")),
        )
        if key not in FILTER_CANDIDATE_ORDER or key in by_candidate:
            raise ValueError("Filter candidate results are missing or duplicated.")
        digest = result.get("design_episode_set_sha256")
        if not _is_sha256(digest):
            raise ValueError("Filter candidate result lacks a design-episode digest.")
        if episode_set_sha256 is None:
            episode_set_sha256 = str(digest)
        elif digest != episode_set_sha256:
            raise ValueError("Filter candidates were not evaluated on the same episodes.")
        cache_digest = result.get(
            "conditioned_opening_prefix_cache_binding_sha256"
        )
        if not _is_sha256(cache_digest):
            raise ValueError("Filter candidate result lacks an opening-cache binding.")
        if opening_cache_binding_sha256 is None:
            opening_cache_binding_sha256 = str(cache_digest)
        elif cache_digest != opening_cache_binding_sha256:
            raise ValueError("Filter candidates did not share one bound opening prefix.")
        prefix_digests = _require_mapping(
            result.get("conditioned_opening_prefix_sha256_by_particle_count"),
            field="filter candidate conditioned-opening prefix digests",
        )
        for count, digest_value in prefix_digests.items():
            count_key = str(count)
            digest_text = str(digest_value)
            if count_key not in {"64", "128", "256"} or not _is_sha256(
                digest_text
            ):
                raise ValueError("Filter candidate opening-prefix digest is invalid.")
            if result.get("conditioned_opening_prefix_scope") == "all_design_episodes":
                previous_digest = opening_prefix_digest_by_particle_count.get(count_key)
                if previous_digest is not None and previous_digest != digest_text:
                    raise ValueError("Filter candidate changed an existing opening prefix.")
                opening_prefix_digest_by_particle_count[count_key] = digest_text
        attempt_digests = _require_mapping(
            result.get("conditioned_opening_attempt_sha256_by_particle_count"),
            field="filter candidate conditioned-opening attempt digests",
        )
        requested_count_key = str(key[0])
        if not _is_sha256(attempt_digests.get(requested_count_key)):
            raise ValueError("Filter candidate lacks its requested opening attempt.")
        attempt_processed_episode_count = _positive_int(
            result.get("processed_episode_count"),
            field="filter result processed episode count",
        )
        for count, digest_value in attempt_digests.items():
            count_key = str(count)
            digest_text = str(digest_value)
            if count_key not in {"64", "128", "256"} or not _is_sha256(
                digest_text
            ):
                raise ValueError("Filter candidate opening-attempt digest is invalid.")
            previous_scope = opening_attempt_scope_by_particle_count.get(count_key)
            if previous_scope is not None and previous_scope[0] == (
                attempt_processed_episode_count
            ) and previous_scope[1] != digest_text:
                raise ValueError("Filter candidate changed an existing opening attempt.")
            if previous_scope is None or attempt_processed_episode_count > (
                previous_scope[0]
            ):
                opening_attempt_scope_by_particle_count[count_key] = (
                    attempt_processed_episode_count,
                    digest_text,
                )
                opening_attempt_digest_by_particle_count[count_key] = digest_text
        status = str(result.get("candidate_evaluation_status", ""))
        if result.get("total_episode_count") != FILTER_DESIGN_EPISODE_COUNT:
            raise ValueError("Filter candidate changed the frozen 80-episode denominator.")
        processed = attempt_processed_episode_count
        closed = _nonnegative_int(
            result.get("zero_support_close_count"),
            field="filter result zero-support close count",
        )
        lower_bound = _finite(
            result.get("s1_zero_support_close_rate_lower_bound"),
            field="filter result S1 lower bound",
        )
        if closed > processed or not math.isclose(
            lower_bound,
            closed / FILTER_DESIGN_EPISODE_COUNT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Filter candidate S1 progress is inconsistent.")
        if status == FILTER_CANDIDATE_S1_STOP_STATUS:
            if not 0 < processed < FILTER_DESIGN_EPISODE_COUNT or not (
                lower_bound > S1_MAXIMUM_CLOSE_RATE
            ) or result.get("conditioned_opening_prefix_scope") != (
                "processed_episode_prefix"
            ) or any(
                result.get(name) is not None
                for name in (
                    "s1_zero_support_close_rate",
                    "s2_posterior_tv_p90",
                    "s3_ess_fraction_p10",
                    "s2_pass",
                    "s3_pass",
                )
            ) or result.get("s1_pass") is not False or result.get(
                "s2_status"
            ) != "not_computed_due_to_irreversible_s1_failure" or result.get(
                "s3_status"
            ) != "not_computed_due_to_irreversible_s1_failure" or result.get(
                "passed"
            ) is not False:
                raise ValueError("Filter S1 prefix failure is not fail-closed.")
        elif status == FILTER_CANDIDATE_COMPLETE_STATUS:
            if processed != FILTER_DESIGN_EPISODE_COUNT or result.get(
                "conditioned_opening_prefix_scope"
            ) != "all_design_episodes" or result.get("s2_status") != (
                "computed"
            ) or result.get("s3_status") != "computed":
                raise ValueError("Complete filter result is missing exact S1--S3.")
            s1 = _finite(
                result.get("s1_zero_support_close_rate"), field="filter result S1"
            )
            s2 = _finite(result.get("s2_posterior_tv_p90"), field="filter result S2")
            s3 = _finite(result.get("s3_ess_fraction_p10"), field="filter result S3")
            expected_flags = {
                "s1_pass": s1 <= S1_MAXIMUM_CLOSE_RATE,
                "s2_pass": s2 <= S2_MAXIMUM_TV_P90,
                "s3_pass": s3 >= S3_MINIMUM_ESS_FRACTION_P10,
            }
            if any(
                result.get(name) is not value
                for name, value in expected_flags.items()
            ):
                raise ValueError("Filter candidate S1--S3 flags do not match their values.")
            expected_pass = all(expected_flags.values())
            if result.get("passed") is not expected_pass:
                raise ValueError("Filter candidate pass status does not match S1--S3.")
        else:
            raise ValueError("Filter candidate result has an unknown completion status.")
        by_candidate[key] = result
    expected_prefix = FILTER_CANDIDATE_ORDER[: len(candidate_results)]
    if tuple(by_candidate) != expected_prefix:
        raise ValueError("Filter results are not the frozen ascending prefix.")
    ordered = [by_candidate[key] for key in expected_prefix]
    selected = next((result for result in ordered if result["passed"]), None)
    if selected is not None and selected is not ordered[-1]:
        raise ValueError("Filter evidence continued after the first passing candidate.")
    if selected is None:
        if len(ordered) != len(FILTER_CANDIDATE_ORDER):
            raise ValueError("Filter evaluation stopped before a pass or the final candidate.")
        return {
            "schema_version": "path_c_r015_filter_selection_v2",
            "selection_rule_id": FILTER_SELECTION_RULE_ID,
            "status": "filter_design_precision_infeasible",
            "selected_candidate": None,
            "conditioned_opening_prefix_cache_binding_sha256": (
                opening_cache_binding_sha256
            ),
            "conditioned_opening_prefix_sha256_by_particle_count": dict(
                opening_prefix_digest_by_particle_count
            ),
            "conditioned_opening_attempt_sha256_by_particle_count": dict(
                opening_attempt_digest_by_particle_count
            ),
            "ordered_results": ordered,
        }
    return {
        "schema_version": "path_c_r015_filter_selection_v2",
        "selection_rule_id": FILTER_SELECTION_RULE_ID,
        "status": "selected",
        "selected_candidate": {
            "particles_per_prototype": selected["particles_per_prototype"],
            "filter_algorithm_id": FILTER_ALGORITHM_ID,
            "resampling_algorithm": FILTER_RESAMPLING_ALGORITHM_ID,
            "resampling_timing": selected["resampling_timing"],
            "resampling_ess_fraction_threshold": (
                0.5
                if selected["resampling_timing"] == "adaptive_ess_below_half_v1"
                else None
            ),
            "resampling_interval_environment_steps": 1,
        },
        "design_episode_set_sha256": episode_set_sha256,
        "conditioned_opening_prefix_cache_binding_sha256": (
            opening_cache_binding_sha256
        ),
        "conditioned_opening_prefix_sha256_by_particle_count": dict(
            opening_prefix_digest_by_particle_count
        ),
        "conditioned_opening_attempt_sha256_by_particle_count": dict(
            opening_attempt_digest_by_particle_count
        ),
        "ordered_results": ordered,
    }


def evaluate_planning_candidate(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    if evidence.get("schema_version") != PLANNING_EVIDENCE_SCHEMA:
        raise ValueError("Planning design evidence has the wrong schema.")
    if evidence.get("scientific_readout_allowed") is not False:
        raise ValueError("Planning design evidence cannot allow scientific readout.")
    branch_count = _positive_int(
        evidence.get("branch_count"), field="planning branch_count"
    )
    doubled = _positive_int(
        evidence.get("doubled_branch_count"), field="planning doubled_branch_count"
    )
    if branch_count not in PLANNING_BRANCH_COUNTS or doubled != 2 * branch_count:
        raise ValueError("Planning evidence does not compare a registered R with 2R.")
    if evidence.get("common_random_numbers") is not True:
        raise ValueError("Planning comparison must use common random numbers.")
    if evidence.get("full_remaining_episode") is not True:
        raise ValueError("Planning comparison must run through environment step 400.")
    if evidence.get("registered_probe_count") != 6 or evidence.get(
        "prototype_count"
    ) != 4:
        raise ValueError("Planning comparison changed the probe or prototype support.")
    if evidence.get("consultation_point_sampling_rule_id") != (
        PLANNING_POINT_SAMPLING_RULE_ID
    ):
        raise ValueError("Planning consultation-point sampling rule changed.")
    if evidence.get("planning_filter_stream_id") != PLANNING_FILTER_STREAM_ID:
        raise ValueError("Planning comparison changed the frozen filter stream.")
    if evidence.get("branch_sampling_rule_id") != BRANCH_SAMPLING_RULE_ID or (
        evidence.get("branch_belief_rule_id") != BRANCH_BELIEF_RULE_ID
    ) or evidence.get("grid_evaluation_rule_id") != GRID_EVALUATION_RULE_ID:
        raise ValueError("Planning comparison changed the registered planning rules.")
    if tuple(evidence.get("nested_branch_counts", ())) != NESTED_BRANCH_COUNTS:
        raise ValueError("Planning comparison changed the nested sample counts.")
    runtime_sources = _require_mapping(
        evidence.get("runtime_source_bindings"), field="planning runtime sources"
    )
    if set(runtime_sources) != {
        "design_runner",
        "controller",
        "full_horizon_executor",
        "runtime_bridge",
    } or any(
        not _is_sha256(
            _require_mapping(binding, field="planning runtime binding").get(
                "sha256"
            )
        )
        for binding in runtime_sources.values()
    ):
        raise ValueError("Planning evidence lacks its runtime source bindings.")
    checkpoint_binding = _require_mapping(
        evidence.get("selected_filter_checkpoint_manifest"),
        field="selected-filter checkpoint manifest binding",
    )
    if set(checkpoint_binding) != {"path", "sha256", "point_count"} or (
        checkpoint_binding.get("point_count") != PLANNING_MINIMUM_POINT_COUNT
    ) or not _is_sha256(checkpoint_binding.get("sha256")) or (
        file_sha256(checkpoint_binding.get("path"))
        != checkpoint_binding.get("sha256")
    ):
        raise ValueError("Planning evidence lacks its complete filter checkpoints.")
    checkpoint_manifest = load_mapping(checkpoint_binding["path"])
    if checkpoint_manifest.get("schema_version") != (
        SELECTED_FILTER_CHECKPOINT_MANIFEST_SCHEMA
    ) or checkpoint_manifest.get("scientific_readout_allowed") is not False:
        raise ValueError("Planning filter checkpoint manifest has the wrong schema.")
    checkpoint_refs = tuple(
        _require_mapping(value, field="planning filter checkpoint reference")
        for value in _require_sequence(
            checkpoint_manifest.get("ordered_checkpoint_refs"),
            field="planning filter checkpoint references",
        )
    )
    checkpoint_sha_by_coordinate = {
        (
            str(ref.get("design_episode_id", "")),
            _positive_int(
                ref.get("environment_step"),
                field="planning filter checkpoint environment step",
            ),
        ): str(ref.get("sha256", ""))
        for ref in checkpoint_refs
    }
    if len(checkpoint_sha_by_coordinate) != PLANNING_MINIMUM_POINT_COUNT or any(
        not _is_sha256(value) for value in checkpoint_sha_by_coordinate.values()
    ):
        raise ValueError("Planning filter checkpoint references are incomplete.")
    total_particles = _positive_int(
        evidence.get("total_particle_count"), field="planning total particle count"
    )
    points = _require_sequence(evidence.get("consultation_points"), field="planning points")
    if len(points) != PLANNING_MINIMUM_POINT_COUNT:
        raise ValueError("Planning comparison requires exactly the first 200 points.")
    identifiers: set[str] = set()
    ordered_coordinates: list[tuple[str, int]] = []
    matching = 0
    logical_steps_r: list[int] = []
    logical_steps_2r: list[int] = []
    new_logical_steps: list[int] = []
    new_particle_steps: list[int] = []
    block_totals: dict[str, int] = {}
    for raw_point in points:
        point = _require_mapping(raw_point, field="planning point")
        point_id = str(point.get("consultation_id", ""))
        block_id = str(point.get("design_block_id", ""))
        episode_id = str(point.get("design_episode_id", ""))
        environment_step = _positive_int(
            point.get("environment_step"), field="planning environment_step"
        )
        if not point_id or point_id in identifiers or not block_id:
            raise ValueError("Planning point identifiers must be unique and non-empty.")
        if not episode_id or environment_step not in CONSULTATION_STEPS:
            raise ValueError("Planning points must identify a registered consultation.")
        if point.get("filter_checkpoint_sha256") != (
            checkpoint_sha_by_coordinate.get((episode_id, environment_step))
        ):
            raise ValueError("A planning point changed its selected-filter checkpoint.")
        partner_id = str(point.get("partner_prototype_id", ""))
        episode_index = _positive_int(
            point.get("episode_index"), field="planning episode_index"
        )
        episode_seed = _nonnegative_int(
            point.get("episode_seed"), field="planning episode seed"
        )
        if not partner_id:
            raise ValueError("Planning points must carry the partner prototype id.")
        identifiers.add(point_id)
        ordered_coordinates.append((episode_index, partner_id, environment_step))
        planning_key = point.get("planning_random_key")
        score_key = point.get("score_random_key")
        expected_planning_key = _six_coordinate_random_key(
            audit_unit_id=block_id,
            partner_prototype_id=partner_id,
            episode_seed=episode_seed,
            purpose="value_planning",
            environment_step=environment_step,
        )
        expected_score_key = _six_coordinate_random_key(
            audit_unit_id=block_id,
            partner_prototype_id=partner_id,
            episode_seed=episode_seed,
            purpose="score_estimation",
            environment_step=environment_step,
        )
        if planning_key != expected_planning_key or score_key != expected_score_key:
            raise ValueError("Planning point random keys are missing or changed.")
        r_decision = _require_mapping(point.get("r_decision"), field="R decision")
        doubled_decision = _require_mapping(
            point.get("doubled_r_decision"), field="2R decision"
        )
        required_decision_fields = {"masked_reference_action", "probe_decision"}
        if set(r_decision) != required_decision_fields or set(
            doubled_decision
        ) != required_decision_fields:
            raise ValueError("Planning decisions must contain exactly the registered tuple.")
        if any(
            not isinstance(decision[field], str) or not decision[field]
            for decision in (r_decision, doubled_decision)
            for field in required_decision_fields
        ):
            raise ValueError("Planning decision tuple values must be non-empty strings.")
        if r_decision["masked_reference_action"] not in {"base", *REGISTERED_PROBE_IDS} or (
            doubled_decision["masked_reference_action"]
            not in {"base", *REGISTERED_PROBE_IDS}
        ):
            raise ValueError("Planning masked-reference action is unregistered.")
        if r_decision["probe_decision"] not in {"no_probe", *REGISTERED_PROBE_IDS} or (
            doubled_decision["probe_decision"]
            not in {"no_probe", *REGISTERED_PROBE_IDS}
        ):
            raise ValueError("Planning probe decision is unregistered.")
        if dict(r_decision) == dict(doubled_decision):
            matching += 1
        remaining = 400 - environment_step
        r_cost = _require_mapping(point.get("r_cost"), field="planning R cost")
        doubled_cost = _require_mapping(
            point.get("doubled_r_cost"), field="planning 2R cost"
        )
        for cost, count, label in (
            (r_cost, branch_count, "R"),
            (doubled_cost, doubled, "2R"),
        ):
            if cost.get("schema_version") != "path_c_r015_planning_cost_v2" or (
                _positive_int(cost.get("sample_count"), field=f"{label} sample count")
                != count
            ) or _positive_int(
                cost.get("remaining_environment_steps"),
                field=f"{label} remaining steps",
            ) != remaining:
                raise ValueError("Planning cost record changed its sample count or horizon.")
            expected = count * (13 * remaining - 6)
            if _nonnegative_int(
                cost.get("estimator_real_environment_transitions"),
                field=f"{label} real environment transitions",
            ) != expected:
                raise ValueError("Planning cost does not equal R times (13h-6).")
            if _nonnegative_int(
                cost.get("estimator_paired_suffix_batched_lane_time_steps"),
                field=f"{label} batched lane time steps",
            ) != count * 7 * remaining:
                raise ValueError("Planning batched lane time does not equal 7Rh.")
        expected_new_samples = doubled if branch_count == 2 else branch_count
        if _nonnegative_int(
            r_cost.get("new_sample_count"), field="R new sample count"
        ) != 0 or _nonnegative_int(
            r_cost.get("new_real_environment_transitions"),
            field="R new real environment transitions",
        ) != 0:
            raise ValueError("The R estimate must be reconstructed from verified 2R evidence.")
        if _nonnegative_int(
            r_cost.get("new_paired_suffix_batched_lane_time_steps"),
            field="R new batched lane time steps",
        ) != 0:
            raise ValueError("The reconstructed R estimate may not add batched work.")
        if _positive_int(
            doubled_cost.get("new_sample_count"), field="2R new sample count"
        ) != expected_new_samples or _nonnegative_int(
            doubled_cost.get("new_real_environment_transitions"),
            field="2R new real environment transitions",
        ) != expected_new_samples * (13 * remaining - 6):
            raise ValueError("The lazy grid recomputed old samples or skipped new samples.")
        if _nonnegative_int(
            doubled_cost.get("new_paired_suffix_batched_lane_time_steps"),
            field="2R new batched lane time steps",
        ) != expected_new_samples * 7 * remaining:
            raise ValueError("Planning batched lane time changed.")
        expected_particle_steps = expected_new_samples * 6 * total_particles
        if _nonnegative_int(
            doubled_cost.get("new_branch_head_particle_transitions"),
            field="2R branch-head particle transitions",
        ) != expected_particle_steps:
            raise ValueError("Planning branch-head particle-transition count changed.")
        # 设备调用属于跨咨询点的长度分桶批次，不能诚实地摊回单个咨询点。
        # 每点证据只固定语义成本；实际编译调用、活动批宽和墙钟在候选顶层核验。
        for cost, label in ((r_cost, "R"), (doubled_cost, "2R")):
            if any(
                field in cost
                for field in (
                    "compiled_batch_calls",
                    "active_batch_sizes",
                    "wall_seconds",
                    "new_real_transition_throughput_per_second",
                )
            ):
                raise ValueError(
                    f"{label} point cost may not claim a share of device batch work."
                )
        r_branches = _require_sequence(
            point.get("r_planning_branches"), field="R planning branches"
        )
        doubled_branches = _require_sequence(
            point.get("doubled_r_planning_branches"), field="2R planning branches"
        )
        if len(r_branches) != branch_count or len(doubled_branches) != doubled:
            raise ValueError("Planning evidence has the wrong number of sample slots.")
        root_phase = int(str(_controller_key(
            planning_key,
            BRANCH_SAMPLING_RULE_ID,
            "systematic_phase_uint64",
        ))[:16], 16)
        r_phase = (root_phase * (branch_count // 2)) % UINT64_MODULUS
        doubled_phase = (root_phase * (doubled // 2)) % UINT64_MODULUS
        if doubled_phase != (2 * r_phase) % UINT64_MODULUS:
            raise RuntimeError("Registered nested systematic phases are inconsistent.")
        r_by_canonical: dict[int, Mapping[str, Any]] = {}
        for expected_slot, raw_branch in enumerate(r_branches):
            branch = _require_mapping(raw_branch, field="R planning branch")
            if branch.get("branch_sampling_rule_id") != BRANCH_SAMPLING_RULE_ID or (
                branch.get("branch_belief_rule_id") != BRANCH_BELIEF_RULE_ID
            ) or branch.get("sample_count") != branch_count or (
                branch.get("estimator_weight") != 1.0 / branch_count
            ):
                raise ValueError("An R sample changed its registered rule or weight.")
            if branch.get("sample_slot") != expected_slot or branch.get(
                "systematic_phase_uint64"
            ) != r_phase or branch.get("systematic_position_numerator") != (
                r_phase + expected_slot * UINT64_MODULUS
            ) or branch.get("systematic_position_denominator") != (
                branch_count * UINT64_MODULUS
            ) or branch.get("common_random_key") != _controller_key(
                planning_key,
                BRANCH_SAMPLING_RULE_ID,
                "canonical_slot_16",
                branch.get("canonical_slot_16"),
            ):
                raise ValueError("An R sample changed its phase, slot, or random key.")
            expected_canonical = _canonical_nested_slot(
                branch_count, expected_slot, r_phase
            )
            expected_lower_slot = None
            if branch_count > 2:
                lower_phase = (
                    root_phase * (branch_count // 4)
                ) % UINT64_MODULUS
                lower_parity = (2 * lower_phase) // UINT64_MODULUS
                if expected_slot % 2 == lower_parity:
                    expected_lower_slot = (
                        expected_slot - int(lower_parity)
                    ) // 2
            if branch.get("canonical_slot_16") != expected_canonical or branch.get(
                "paired_lower_slot"
            ) != expected_lower_slot:
                raise ValueError("An R sample changed its nested slot mapping.")
            _validate_planning_branch_random_summary(
                branch,
                remaining_environment_steps=remaining,
            )
            if branch.get("branch_rollout_sha256") != (
                _planning_branch_rollout_sha256(branch)
            ):
                raise ValueError("An R sample return digest was changed.")
            canonical_slot = _nonnegative_int(
                branch.get("canonical_slot_16"), field="R canonical slot"
            )
            if canonical_slot >= 16 or canonical_slot in r_by_canonical:
                raise ValueError("R sample canonical slots are invalid or duplicated.")
            r_by_canonical[canonical_slot] = branch
        doubled_by_canonical: dict[int, Mapping[str, Any]] = {}
        for expected_slot, raw_branch in enumerate(doubled_branches):
            branch = _require_mapping(raw_branch, field="2R planning branch")
            if branch.get("branch_sampling_rule_id") != BRANCH_SAMPLING_RULE_ID or (
                branch.get("branch_belief_rule_id") != BRANCH_BELIEF_RULE_ID
            ) or branch.get("sample_count") != doubled or (
                branch.get("estimator_weight") != 1.0 / doubled
            ):
                raise ValueError("A 2R sample changed its registered rule or weight.")
            if branch.get("sample_slot") != expected_slot or branch.get(
                "systematic_phase_uint64"
            ) != doubled_phase or branch.get("systematic_position_numerator") != (
                doubled_phase + expected_slot * UINT64_MODULUS
            ) or branch.get("systematic_position_denominator") != (
                doubled * UINT64_MODULUS
            ) or branch.get("common_random_key") != _controller_key(
                planning_key,
                BRANCH_SAMPLING_RULE_ID,
                "canonical_slot_16",
                branch.get("canonical_slot_16"),
            ):
                raise ValueError("A 2R sample changed its phase, slot, or random key.")
            expected_canonical = _canonical_nested_slot(
                doubled, expected_slot, doubled_phase
            )
            lower_parity = (2 * r_phase) // UINT64_MODULUS
            expected_lower_slot = None
            if expected_slot % 2 == lower_parity:
                expected_lower_slot = (
                    expected_slot - int(lower_parity)
                ) // 2
            if branch.get("canonical_slot_16") != expected_canonical or branch.get(
                "paired_lower_slot"
            ) != expected_lower_slot:
                raise ValueError("A 2R sample changed its nested slot mapping.")
            _validate_planning_branch_random_summary(
                branch,
                remaining_environment_steps=remaining,
            )
            if branch.get("branch_rollout_sha256") != (
                _planning_branch_rollout_sha256(branch)
            ):
                raise ValueError("A 2R sample return digest was changed.")
            canonical_slot = _nonnegative_int(
                branch.get("canonical_slot_16"), field="2R canonical slot"
            )
            if canonical_slot >= 16 or canonical_slot in doubled_by_canonical:
                raise ValueError("2R sample canonical slots are invalid or duplicated.")
            doubled_by_canonical[canonical_slot] = branch
        if not set(r_by_canonical).issubset(doubled_by_canonical):
            raise ValueError("The R sample slots are not a strict subset of 2R.")
        for canonical_slot, lower in r_by_canonical.items():
            upper = doubled_by_canonical[canonical_slot]
            for field in (
                "source_particle_index",
                "source_particle_prototype_id",
                "source_particle_state_sha256",
                "common_random_key",
                "branch_rollout_sha256",
            ):
                if lower.get(field) != upper.get(field):
                    raise ValueError("A reused nested sample changed its source or return.")
        r_steps = branch_count * (13 * remaining - 6)
        doubled_steps = doubled * (13 * remaining - 6)
        new_steps = expected_new_samples * (13 * remaining - 6)
        logical_steps_r.append(r_steps)
        logical_steps_2r.append(doubled_steps)
        new_logical_steps.append(new_steps)
        new_particle_steps.append(expected_particle_steps)
        block_totals[block_id] = block_totals.get(block_id, 0) + new_steps
    if ordered_coordinates != sorted(ordered_coordinates) or len(
        set(ordered_coordinates)
    ) != len(ordered_coordinates):
        raise ValueError(
            "Planning points are not the frozen partner-interleaved first 200 set "
            "(order: episode_index, partner_prototype_id, environment_step)."
        )
    total_new_logical = sum(new_logical_steps)
    total_new_particles = sum(new_particle_steps)
    if _nonnegative_int(
        evidence.get("new_real_environment_transitions"),
        field="planning total new logical transitions",
    ) != total_new_logical or _nonnegative_int(
        evidence.get("new_branch_head_particle_transitions"),
        field="planning total branch-head particle transitions",
    ) != total_new_particles:
        raise ValueError("Planning evidence totals do not match its consultation points.")
    bucket_reports = tuple(
        _require_mapping(value, field="planning length-bucket execution")
        for value in _require_sequence(
            evidence.get("length_bucket_device_execution"),
            field="planning length-bucket executions",
        )
    )
    expected_horizons = tuple(400 - value for value in CONSULTATION_STEPS)
    if tuple(
        _positive_int(
            report.get("remaining_environment_steps"),
            field="planning length-bucket horizon",
        )
        for report in bucket_reports
    ) != expected_horizons:
        raise ValueError("Planning length buckets changed the registered horizons.")
    if any(
        report.get("schema_version")
        != "path_c_r015_planning_length_bucket_execution_v2"
        or report.get("compiled_batch_calls") != 4
        or report.get("host_sync_inside_environment_loop") is not False
        for report in bucket_reports
    ):
        raise ValueError("A planning length bucket changed its four device stages.")
    bucket_sample_count = 0
    bucket_true_transitions = 0
    bucket_computed_transitions = 0
    bucket_particle_transitions = 0
    bucket_compiled_calls = 0
    bucket_active_sizes: list[int] = []
    for report in bucket_reports:
        sample_count = _positive_int(
            report.get("new_sample_count"),
            field="planning length-bucket new samples",
        )
        horizon = int(report["remaining_environment_steps"])
        active_sizes = tuple(
            _positive_int(value, field="planning length-bucket active size")
            for value in _require_sequence(
                report.get("active_batch_sizes"),
                field="planning length-bucket active sizes",
            )
        )
        if active_sizes != (
            sample_count,
            6 * sample_count,
            6 * sample_count * total_particles,
            12 * sample_count,
        ) or report.get("true_environment_transitions") != (
            sample_count * (13 * horizon - 6)
        ) or report.get("branch_head_particle_transitions") != (
            sample_count * 6 * total_particles
        ):
            raise ValueError("A planning length bucket changed its exact cost.")
        computed_with_padding = _positive_int(
            report.get("computed_environment_transitions_including_padding"),
            field="planning padded device transitions",
        )
        if computed_with_padding != sample_count * (399 + 6 + 12 * 398) or (
            computed_with_padding < int(report["true_environment_transitions"])
        ) or report.get("static_scan_step_counts") != {
            "base": 399,
            "shared_probe_head": 1,
            "paired_suffix": 398,
        }:
            raise ValueError("A planning length bucket changed its fixed-shape work.")
        bucket_sample_count += sample_count
        bucket_true_transitions += int(report["true_environment_transitions"])
        bucket_computed_transitions += computed_with_padding
        bucket_particle_transitions += int(
            report["branch_head_particle_transitions"]
        )
        bucket_compiled_calls += int(report["compiled_batch_calls"])
        bucket_active_sizes.extend(active_sizes)
    expected_new_samples_total = (
        (doubled if branch_count == 2 else branch_count)
        * PLANNING_MINIMUM_POINT_COUNT
    )
    if bucket_sample_count != expected_new_samples_total or (
        bucket_true_transitions != total_new_logical
    ) or bucket_particle_transitions != total_new_particles:
        raise ValueError("Planning length-bucket totals changed the registered work.")
    if _positive_int(
        evidence.get("computed_environment_transitions_including_padding"),
        field="planning total padded device transitions",
    ) != bucket_computed_transitions:
        raise ValueError("Planning padded device cost does not match its buckets.")
    compiled_batch_calls = _positive_int(
        evidence.get("compiled_batch_calls"),
        field="planning candidate compiled batch calls",
    )
    active_batch_sizes = tuple(
        _positive_int(value, field="planning candidate active batch size")
        for value in _require_sequence(
            evidence.get("active_batch_sizes"),
            field="planning candidate active batch sizes",
        )
    )
    if not active_batch_sizes:
        raise ValueError("Planning candidate must report its actual device batch sizes.")
    if compiled_batch_calls != bucket_compiled_calls or active_batch_sizes != tuple(
        bucket_active_sizes
    ):
        raise ValueError("Planning top-level device counts do not match its buckets.")
    if evidence.get("host_sync_inside_environment_loop") is not False:
        raise ValueError("Planning synchronized with the host inside a compiled suffix loop.")
    wall_seconds = _finite(
        evidence.get("wall_seconds"), field="planning wall seconds"
    )
    if wall_seconds < 0.0:
        raise ValueError("Planning wall time cannot be negative.")
    throughput = evidence.get("new_real_transition_throughput_per_second")
    if wall_seconds > 0.0:
        if not math.isclose(
            _finite(throughput, field="planning transition throughput"),
            total_new_logical / wall_seconds,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Planning transition throughput does not match its cost.")
    elif throughput is not None:
        raise ValueError("Zero planning wall time requires a null throughput.")
    agreement = matching / len(points)
    return {
        "schema_version": PLANNING_CANDIDATE_RESULT_SCHEMA,
        "selection_rule_id": PLANNING_SELECTION_RULE_ID,
        "branch_count": branch_count,
        "doubled_branch_count": doubled,
        "planning_filter_stream_id": PLANNING_FILTER_STREAM_ID,
        "consultation_point_count": len(points),
        "exact_decision_match_count": matching,
        "exact_decision_agreement": agreement,
        "passed": agreement >= PLANNING_MINIMUM_AGREEMENT,
        "real_environment_transitions_per_point_r": logical_steps_r,
        "real_environment_transitions_per_point_2r": logical_steps_2r,
        "new_real_environment_transitions_per_point": new_logical_steps,
        "new_branch_head_particle_transitions_per_point": new_particle_steps,
        "compiled_batch_calls": compiled_batch_calls,
        "active_batch_sizes": list(active_batch_sizes),
        "computed_environment_transitions_including_padding": (
            bucket_computed_transitions
        ),
        "new_real_environment_transitions_per_block": dict(sorted(block_totals.items())),
    }


def select_planning_branch_count(
    candidate_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not 1 <= len(candidate_results) <= len(PLANNING_BRANCH_COUNTS):
        raise ValueError("Planning selection requires a non-empty ascending result prefix.")
    by_branch: dict[int, Mapping[str, Any]] = {}
    for raw_result in candidate_results:
        result = _require_mapping(raw_result, field="planning candidate result")
        if result.get("schema_version") != PLANNING_CANDIDATE_RESULT_SCHEMA:
            raise ValueError("Planning candidate result has the wrong schema.")
        if result.get("selection_rule_id") != PLANNING_SELECTION_RULE_ID:
            raise ValueError("Planning candidate result changed the selection rule.")
        branch_count = _positive_int(
            result.get("branch_count"), field="planning result branch_count"
        )
        if branch_count not in PLANNING_BRANCH_COUNTS or branch_count in by_branch:
            raise ValueError("Planning results are missing or duplicated.")
        if _positive_int(
            result.get("doubled_branch_count"),
            field="planning result doubled branch count",
        ) != 2 * branch_count:
            raise ValueError("Planning result does not bind 2R.")
        point_count = _positive_int(
            result.get("consultation_point_count"),
            field="planning result consultation count",
        )
        match_count = _nonnegative_int(
            result.get("exact_decision_match_count"),
            field="planning result exact match count",
        )
        if match_count > point_count:
            raise ValueError("Planning exact match count exceeds its point count.")
        agreement = _finite(
            result.get("exact_decision_agreement"),
            field="planning exact decision agreement",
        )
        if not math.isclose(
            agreement,
            match_count / point_count,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Planning agreement does not match its counts.")
        expected_pass = agreement >= PLANNING_MINIMUM_AGREEMENT
        if result.get("passed") is not expected_pass:
            raise ValueError("Planning pass status does not match exact agreement.")
        if point_count != PLANNING_MINIMUM_POINT_COUNT:
            raise ValueError("Planning result does not contain the frozen first 200 points.")
        by_branch[branch_count] = result
    expected_prefix = PLANNING_BRANCH_COUNTS[: len(candidate_results)]
    if tuple(by_branch) != expected_prefix:
        raise ValueError("Planning results are not the frozen ascending prefix.")
    ordered = [by_branch[value] for value in expected_prefix]
    selected = next((result for result in ordered if result["passed"]), None)
    if selected is not None and selected is not ordered[-1]:
        raise ValueError("Planning evidence continued after the first passing grid.")
    if selected is None:
        if len(ordered) != len(PLANNING_BRANCH_COUNTS):
            raise ValueError("Planning stopped before a pass or the final registered grid.")
        return {
            "schema_version": PLANNING_SELECTION_SCHEMA,
            "selection_rule_id": PLANNING_SELECTION_RULE_ID,
            "status": "planning_design_precision_infeasible",
            "selected_branch_count": None,
            "ordered_results": ordered,
        }
    return {
        "schema_version": PLANNING_SELECTION_SCHEMA,
        "selection_rule_id": PLANNING_SELECTION_RULE_ID,
        "status": "selected",
        "selected_branch_count": selected["branch_count"],
        "ordered_results": ordered,
    }


def validate_seed_role_isolation(
    *,
    design_seeds: Sequence[int],
    admission_seeds: Sequence[int],
    pilot_seeds: Sequence[int],
    formal_seeds: Sequence[int] = (),
) -> Mapping[str, Any]:
    groups = {}
    for name, raw_values in (
        ("design", design_seeds),
        ("admission", admission_seeds),
        ("pilot", pilot_seeds),
        ("formal", formal_seeds),
    ):
        values = tuple(
            _nonnegative_int(value, field=f"{name} seed") for value in raw_values
        )
        if len(set(values)) != len(values):
            raise ValueError(f"{name} seed set contains a duplicate.")
        groups[name] = set(values)
    for left_index, left_name in enumerate(groups):
        for right_name in tuple(groups)[left_index + 1 :]:
            overlap = groups[left_name].intersection(groups[right_name])
            if overlap:
                raise ValueError(
                    f"R015 seed roles overlap between {left_name} and {right_name}."
                )
    return {
        "schema_version": "path_c_r015_seed_role_isolation_report_v1",
        "contract_id": DESIGN_SEED_CONTRACT_ID,
        "passed": True,
        "seed_counts": {name: len(values) for name, values in groups.items()},
        "seed_set_sha256": {
            name: canonical_sha256(sorted(values)) for name, values in groups.items()
        },
    }


def load_r015_design_protocol(path: str | Path) -> Mapping[str, Any]:
    protocol_path = Path(path).resolve()
    payload = dict(load_mapping(protocol_path))
    if payload.get("schema_version") != DESIGN_PROTOCOL_SCHEMA:
        raise ValueError("R015 design protocol has the wrong schema.")
    if payload.get("status") != "static_registered_not_frozen" or payload.get(
        "scientific_readout_allowed"
    ) is not False:
        raise ValueError("R015 design protocol honesty markers changed.")
    filtering = _require_mapping(payload.get("filter_selection"), field="filter_selection")
    if tuple(filtering.get("particles_per_prototype", ())) != FILTER_PARTICLE_COUNTS:
        raise ValueError("R015 filter particle grid changed.")
    if tuple(filtering.get("resampling_timings", ())) != FILTER_RESAMPLING_TIMINGS:
        raise ValueError("R015 filter resampling timing order changed.")
    if filtering.get("selection_rule_id") != FILTER_SELECTION_RULE_ID:
        raise ValueError("R015 filter selection rule changed.")
    if filtering.get("mode") != FILTER_ALGORITHM_ID or filtering.get(
        "resampling_algorithm"
    ) != FILTER_RESAMPLING_ALGORITHM_ID or tuple(
        filtering.get("consultation_steps", ())
    ) != CONSULTATION_STEPS or filtering.get("percentile_rule_id") != PERCENTILE_RULE_ID:
        raise ValueError("R015 filter mechanics changed.")
    device_execution = _require_mapping(
        filtering.get("device_execution"), field="filter_selection.device_execution"
    )
    if device_execution.get("implementation_id") != FILTER_DEVICE_EXECUTION_ID or (
        device_execution.get("key_contract") != FILTER_DEVICE_KEY_CONTRACT_ID
    ) or device_execution.get("parent_particle_slot_target_per_microbatch") != (
        FILTER_PARENT_SLOT_TARGET
    ) or device_execution.get("group_partner_network_by_prototype") is not True or (
        device_execution.get("continuation_member_forwards_in_filter_candidate_stage")
        != 0
    ):
        raise ValueError("R015 filter device execution contract changed.")
    expected_lane_schedule = {
        particles: filter_lane_batch_size(particles)
        for particles in FILTER_PARTICLE_COUNTS
    }
    raw_lane_schedule = _require_mapping(
        device_execution.get("lanes_per_microbatch_by_particles_per_prototype"),
        field="filter_selection.device_execution.lanes_per_microbatch",
    )
    normalized_lane_schedule = {
        int(particles): int(lanes) for particles, lanes in raw_lane_schedule.items()
    }
    if normalized_lane_schedule != expected_lane_schedule:
        raise ValueError("R015 filter microbatch lane schedule changed.")
    irreversible_s1 = _require_mapping(
        filtering.get("irreversible_s1_failure"),
        field="filter_selection.irreversible_s1_failure",
    )
    if dict(irreversible_s1) != {
        "fixed_episode_denominator": FILTER_DESIGN_EPISODE_COUNT,
        "stop_candidate_at_closed_episode_count": (
            FILTER_S1_EARLY_STOP_CLOSED_EPISODE_COUNT
        ),
        "resulting_minimum_close_fraction": 0.025,
        "s2_or_s3_may_be_read_to_override_failure": False,
    }:
        raise ValueError("R015 irreversible S1 failure rule changed.")
    for field, threshold_field, expected in (
        ("s1", "pass_if_at_most", S1_MAXIMUM_CLOSE_RATE),
        ("s2", "pass_if_at_most", S2_MAXIMUM_TV_P90),
        ("s3", "pass_if_at_least", S3_MINIMUM_ESS_FRACTION_P10),
    ):
        rule = _require_mapping(filtering.get(field), field=f"filter_selection.{field}")
        if not math.isclose(
            _finite(rule.get(threshold_field), field=f"{field} threshold"),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"R015 {field} threshold changed.")
    support = dict(_require_mapping(payload.get("support"), field="support"))
    ego_candidate = dict(
        _require_mapping(support.get("ego_candidate"), field="support.ego_candidate")
    )
    if {
        "candidate_id": ego_candidate.get("candidate_id"),
        "family_id": ego_candidate.get("family_id"),
        "training_seed": ego_candidate.get("training_seed"),
        "expected_environment_steps": ego_candidate.get(
            "expected_environment_steps"
        ),
    } != {
        "candidate_id": "official_rnn_sp_ippo_v1_seed100_step29949952",
        "family_id": OFFICIAL_SP_FAMILY_ID,
        "training_seed": 100,
        "expected_environment_steps": 29949952,
    }:
        raise ValueError("R015 design ego checkpoint registration changed.")
    for field in ("checkpoint_path", "training_config_path", "training_manifest_path"):
        raw_target = str(ego_candidate.get(field, ""))
        if not raw_target:
            raise ValueError("R015 ego candidate artifact path is missing.")
        target = Path(raw_target)
        if not target.is_absolute():
            target = (protocol_path.parent / target).resolve()
        ego_candidate[field] = str(target)
    support["ego_candidate"] = ego_candidate
    payload["support"] = support
    if support.get("design_episodes_per_prototype") != 20 or support.get(
        "total_design_episodes"
    ) != 80 or len(tuple(support.get("partner_prototype_ids", ()))) != 4:
        raise ValueError("R015 filter design episode allocation changed.")
    environment = _require_mapping(payload.get("environment"), field="environment")
    if environment.get("history_collection_action_rule") != (
        "ego_seed100_official_categorical_no_probe_v1"
    ):
        raise ValueError("R015 design history collection action rule changed.")
    response_summary = _require_mapping(
        payload.get("response_summary"), field="response_summary"
    )
    if response_summary != {
        "implementation_constant": "REGISTERED_LOCAL_RESPONSE_SPEC",
        "response_classes": ["visible", "unseen", "local_non_agent_change"],
        "latency_bin_upper_bounds": [1],
        "source_fields": "adjacent_official_local_observations_only",
        "evaluator_partner_action_allowed": False,
    }:
        raise ValueError("R015 design response-summary registration changed.")
    planning = _require_mapping(payload.get("planning_selection"), field="planning_selection")
    if tuple(planning.get("branch_counts", ())) != PLANNING_BRANCH_COUNTS or planning.get(
        "selection_rule_id"
    ) != PLANNING_SELECTION_RULE_ID:
        raise ValueError("R015 planning selection rule changed.")
    point_sampling = _require_mapping(
        planning.get("consultation_point_sampling"),
        field="planning consultation_point_sampling",
    )
    if point_sampling.get("rule_id") != PLANNING_POINT_SAMPLING_RULE_ID or (
        point_sampling.get("take_first") != PLANNING_MINIMUM_POINT_COUNT
    ) or point_sampling.get("minimum_points") != PLANNING_MINIMUM_POINT_COUNT:
        raise ValueError("R015 planning point sample changed.")
    if point_sampling.get("planning_filter_stream_id") != PLANNING_FILTER_STREAM_ID or (
        point_sampling.get("second_filter_stream_role")
        != "posterior_repeat_stability_only"
    ):
        raise ValueError("R015 planning filter-stream role changed.")
    planning_scope = _require_mapping(
        planning.get("planning_scope"), field="planning planning_scope"
    )
    expected_scope = {
        "registered_probe_scripts": 6,
        "include_base_action": True,
        "registered_prototypes": 4,
        "horizon": "full_remaining_episode_to_step_400",
        "continuation_controller": CONTINUATION_CONTROLLER_ID,
        "planning_routing_frequency": "commit_once_at_branch_head",
        "execution_routing_frequency": "update_online_each_environment_step",
        "shared_member_selection_rule": "unique_exact_map_else_ego_seed100",
    }
    if dict(planning_scope) != expected_scope:
        raise ValueError("R015 planning changed its continuation controller.")
    if planning.get("branch_sampling") != BRANCH_SAMPLING_RULE_ID or planning.get(
        "branch_belief"
    ) != BRANCH_BELIEF_RULE_ID or planning.get("grid_evaluation") != (
        GRID_EVALUATION_RULE_ID
    ) or tuple(planning.get("nested_branch_counts", ())) != NESTED_BRANCH_COUNTS or (
        planning.get("reuse_verified_lower_grid_evidence") is not True
    ):
        raise ValueError("R015 nested planning registration changed.")
    if dict(
        _require_mapping(
            planning.get("cost_accounting"), field="planning cost_accounting"
        )
    ) != {
        "per_consultation_semantic_costs": {
            "required_fields": [
                "real_environment_transitions",
                "paired_suffix_batched_lane_time_steps",
                "branch_head_particle_transitions",
            ],
            "record_real_environment_transitions": True,
            "record_paired_suffix_batched_lane_time_steps": True,
            "record_branch_head_particle_transitions": True,
        },
        "per_planning_candidate_device_execution": {
            "record_scope": "planning_candidate_top_level",
            "group_by": ["remaining_horizon_length", "device_batch_index"],
            "required_fields": [
                "remaining_horizon_length",
                "device_batch_index",
                "compiled_batch_calls",
                "active_batch_size",
                "wall_seconds",
                "throughput",
            ],
            "record_compiled_batch_calls": True,
            "record_active_batch_size": True,
            "record_wall_seconds": True,
            "record_real_transition_throughput": True,
            "record_host_sync_inside_environment_loop": True,
            "candidate_throughput_formula": (
                "candidate_total_real_environment_transitions/"
                "candidate_total_wall_seconds"
            ),
        },
        "per_consultation_device_metric_allocation_allowed": False,
        "cross_consultation_vmap_pseudo_allocation_allowed": False,
        "device_execution_metrics_allowed_as_planning_selection_input": False,
    }:
        raise ValueError("R015 planning cost-accounting fields changed.")
    if not math.isclose(
        _finite(
            planning.get("pass_if_exact_agreement_at_least"),
            field="planning agreement threshold",
        ),
        PLANNING_MINIMUM_AGREEMENT,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or tuple(planning.get("exact_match_tuple", ())) != (
        "masked_reference_action",
        "probe_decision_candidate_id_or_no_probe",
    ):
        raise ValueError("R015 planning exact-match contract changed.")
    isolation = _require_mapping(payload.get("role_isolation"), field="role_isolation")
    if isolation.get("contract_id") != DESIGN_SEED_CONTRACT_ID or isolation.get(
        "filter_key_contract"
    ) != FILTER_DEVICE_KEY_CONTRACT_ID:
        raise ValueError("R015 design role-isolation contract changed.")
    return payload


def build_design_selection_report(
    *,
    protocol_path: str | Path,
    filter_evidence_paths: Sequence[str | Path],
    planning_evidence_paths: Sequence[str | Path],
) -> Mapping[str, Any]:
    """Read all registered candidate evidence and return one mechanical result."""

    protocol = load_r015_design_protocol(protocol_path)
    support = _require_mapping(protocol.get("support"), field="support")
    prototype_ids = tuple(
        str(value)
        for value in _require_sequence(
            support.get("partner_prototype_ids"), field="support partner ids"
        )
    )
    if not 1 <= len(filter_evidence_paths) <= len(FILTER_CANDIDATE_ORDER):
        raise ValueError("Design selection requires an ascending filter evidence prefix.")
    filter_results = [
        evaluate_filter_candidate(load_mapping(path), prototype_ids=prototype_ids)
        for path in filter_evidence_paths
    ]
    filter_bindings = [
        {"path": str(Path(path)), "sha256": file_sha256(path)}
        for path in filter_evidence_paths
    ]
    filter_selection = select_filter_candidate(filter_results)
    if filter_selection["status"] != "selected":
        return {
            "schema_version": DESIGN_SELECTION_REPORT_SCHEMA,
            "scientific_readout_allowed": False,
            "status": "filter_design_precision_infeasible",
            "protocol_path": str(Path(protocol_path)),
            "protocol_sha256": file_sha256(protocol_path),
            "filter_evidence": filter_bindings,
            "planning_evidence": [],
            "filter_selection": filter_selection,
            "planning_selection": None,
        }
    if not 1 <= len(planning_evidence_paths) <= 3:
        raise ValueError("A selected filter requires an ascending planning evidence prefix.")
    planning_bindings = [
        {"path": str(Path(path)), "sha256": file_sha256(path)}
        for path in planning_evidence_paths
    ]
    planning_results = [
        evaluate_planning_candidate(load_mapping(path))
        for path in planning_evidence_paths
    ]
    planning_selection = select_planning_branch_count(planning_results)
    status = (
        "selected"
        if planning_selection["status"] == "selected"
        else "planning_design_precision_infeasible"
    )
    return {
        "schema_version": DESIGN_SELECTION_REPORT_SCHEMA,
        "scientific_readout_allowed": False,
        "status": status,
        "protocol_path": str(Path(protocol_path)),
        "protocol_sha256": file_sha256(protocol_path),
        "filter_evidence": filter_bindings,
        "planning_evidence": planning_bindings,
        "filter_selection": filter_selection,
        "planning_selection": planning_selection,
    }


def parse_delivery_reward(settings_path: str | Path) -> Mapping[str, Any]:
    target = Path(settings_path)
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    matches: list[tuple[float, int]] = []
    for node in tree.body:
        name: str | None = None
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value_node = node.value
        if name != "DELIVERY_REWARD" or value_node is None:
            continue
        value = ast.literal_eval(value_node)
        matches.append((_finite(value, field="DELIVERY_REWARD"), node.lineno))
    if len(matches) != 1:
        raise ValueError("settings.py must define DELIVERY_REWARD exactly once.")
    reward, line_number = matches[0]
    if not math.isclose(reward, EXPECTED_DELIVERY_REWARD, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The installed delivery reward differs from the signed contract.")
    return {
        "value": reward,
        "source_path": str(target.resolve()),
        "source_sha256": file_sha256(target),
        "line_number": line_number,
        "source_line": source.splitlines()[line_number - 1].strip(),
    }


def _bind_file_mapping(raw: Any, *, field: str) -> Mapping[str, Any]:
    mapping = _require_mapping(raw, field=field)
    if not mapping:
        raise ValueError(f"{field} cannot be empty.")
    result: dict[str, Mapping[str, str]] = {}
    for name, raw_binding in sorted(mapping.items()):
        if isinstance(raw_binding, Mapping):
            if set(raw_binding) != {"path", "sha256"}:
                raise ValueError(
                    f"{field}.{name} binding must contain only path and sha256."
                )
            path = Path(str(raw_binding["path"])).resolve()
            observed_sha256 = file_sha256(path)
            if raw_binding.get("sha256") != observed_sha256:
                raise ValueError(f"{field}.{name} file hash does not match.")
        else:
            path = Path(str(raw_binding)).resolve()
            observed_sha256 = file_sha256(path)
        result[str(name)] = {
            "path": str(path),
            "sha256": observed_sha256,
        }
    return result


def _bind_checkpoint(raw: Any) -> Mapping[str, Any]:
    entry = _require_mapping(raw, field="checkpoint input")
    required = {
        "artifact_id",
        "role",
        "training_seed",
        "path",
        "format",
        "parameter_tree_path",
        "training_manifest_path",
    }
    derived = {
        "family_id",
        "effective_environment_steps",
        "checkpoint_sha256",
        "model_weights_hash_domain",
        "model_weights_sha256",
        "training_run_id",
        "training_manifest_sha256",
    }
    if not required.issubset(entry) or set(entry).difference(required | derived):
        raise ValueError("Freeze checkpoint input has the wrong fields.")
    training_seed = _nonnegative_int(
        entry["training_seed"], field="checkpoint training_seed"
    )
    role = str(entry["role"])
    expected = EXPECTED_FREEZE_CHECKPOINTS.get(training_seed)
    if expected is None or role != expected["role"]:
        raise ValueError("Freeze checkpoint seed or role differs from the signed set.")
    parameter_path = tuple(
        str(value)
        for value in _require_sequence(
            entry["parameter_tree_path"], field="checkpoint parameter_tree_path"
        )
    )
    params = load_flax_parameter_tree(
        entry["path"],
        checkpoint_format=str(entry["format"]),
        parameter_tree_path=parameter_path,
    )
    manifest_path = Path(str(entry["training_manifest_path"]))
    manifest = load_mapping(manifest_path)
    if manifest.get("schema_version") != "path_c_official_training_artifact_v2":
        raise ValueError("R015 freeze requires an official v2 training manifest.")
    checkpoint = _require_mapping(
        manifest.get("checkpoint"), field="official training checkpoint"
    )
    verified = validate_official_artifact_manifest(
        manifest.get("training_config_path"),
        manifest_path,
        expected_checkpoint_path=entry["path"],
        params=params,
    )
    checkpoint_sha256 = checkpoint_artifact_sha256(entry["path"])
    weights_sha256 = flax_weights_sha256(params)
    if Path(str(checkpoint.get("path"))).resolve() != Path(str(entry["path"])).resolve():
        raise ValueError("Official training manifest binds another checkpoint path.")
    if checkpoint.get("format") != entry["format"] or tuple(
        checkpoint.get("parameter_tree_path", ())
    ) != parameter_path:
        raise ValueError("Official training manifest checkpoint interface changed.")
    if checkpoint.get("checkpoint_sha256") != checkpoint_sha256 or checkpoint.get(
        "model_weights_sha256"
    ) != weights_sha256 or checkpoint.get("model_weights_hash_domain") != (
        FLAX_WEIGHTS_HASH_DOMAIN
    ):
        raise ValueError("Official training manifest checkpoint hashes do not match.")
    if manifest.get("seed") != training_seed or not _is_sha256(
        manifest.get("training_run_id")
    ):
        raise ValueError("Official training manifest run identity does not match.")
    if verified.get("family_id") != expected["family_id"] or verified.get(
        "snapshot_environment_steps"
    ) != expected["effective_environment_steps"] or verified.get("layout") != (
        "test_time_simple"
    ):
        raise ValueError("Official checkpoint differs from the signed R015 production set.")
    result = {
        "artifact_id": str(entry["artifact_id"]),
        "role": role,
        "training_seed": training_seed,
        "family_id": str(verified["family_id"]),
        "effective_environment_steps": int(verified["snapshot_environment_steps"]),
        "path": str(Path(str(entry["path"])).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "model_weights_hash_domain": FLAX_WEIGHTS_HASH_DOMAIN,
        "model_weights_sha256": weights_sha256,
        "training_run_id": str(manifest["training_run_id"]),
        "format": str(entry["format"]),
        "parameter_tree_path": list(parameter_path),
        "training_manifest_path": str(manifest_path.resolve()),
        "training_manifest_sha256": file_sha256(manifest_path),
    }
    for field in derived.intersection(entry):
        if entry[field] != result[field]:
            raise ValueError(
                f"Freeze checkpoint input {field} differs from actual artifact read-back."
            )
    return result


def _count_nonempty_lines(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _validate_selected_design_evidence(
    selection: Mapping[str, Any],
    *,
    prototype_ids: Sequence[str] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], int, int]:
    """重读新版过滤和规划证据，并返回唯一机械选择。"""

    filter_selection = _require_mapping(
        selection.get("filter_selection"), field="selection filter_selection"
    )
    planning_selection = _require_mapping(
        selection.get("planning_selection"), field="selection planning_selection"
    )
    if filter_selection.get("status") != "selected" or planning_selection.get(
        "status"
    ) != "selected":
        raise ValueError("Freeze inputs contain an infeasible design selection.")
    selected_filter = _require_mapping(
        filter_selection.get("selected_candidate"), field="selected filter candidate"
    )
    selected_timing = str(selected_filter.get("resampling_timing", ""))
    selected_particles = _positive_int(
        selected_filter.get("particles_per_prototype"),
        field="selected particles_per_prototype",
    )
    expected_threshold = 0.5 if selected_timing == "adaptive_ess_below_half_v1" else None
    if (selected_particles, selected_timing) not in FILTER_CANDIDATE_ORDER or (
        selected_filter.get("filter_algorithm_id") != FILTER_ALGORITHM_ID
    ) or selected_filter.get("resampling_algorithm") != (
        FILTER_RESAMPLING_ALGORITHM_ID
    ) or selected_filter.get("resampling_interval_environment_steps") != 1 or (
        selected_filter.get("resampling_ess_fraction_threshold") != expected_threshold
    ):
        raise ValueError("Design selection names an unregistered v2 filter candidate.")
    selected_branch_count = _positive_int(
        planning_selection.get("selected_branch_count"),
        field="selected planning branch count",
    )
    if selected_branch_count not in PLANNING_BRANCH_COUNTS:
        raise ValueError("Design selection names an unregistered planning branch count.")
    expected_counts = {
        "filter_evidence": FILTER_CANDIDATE_ORDER.index(
            (selected_particles, selected_timing)
        )
        + 1,
        "planning_evidence": PLANNING_BRANCH_COUNTS.index(selected_branch_count) + 1,
    }
    expected_schemas = {
        "filter_evidence": FILTER_EVIDENCE_SCHEMA,
        "planning_evidence": PLANNING_EVIDENCE_SCHEMA,
    }
    loaded_evidence: dict[str, list[Mapping[str, Any]]] = {
        "filter_evidence": [],
        "planning_evidence": [],
    }
    for evidence_field, expected_count in expected_counts.items():
        evidence = _require_sequence(selection.get(evidence_field), field=evidence_field)
        if len(evidence) != expected_count:
            raise ValueError("Design selection evidence file count is incomplete.")
        for entry in evidence:
            binding = _require_mapping(entry, field=f"{evidence_field} entry")
            if not _is_sha256(binding.get("sha256")) or file_sha256(
                binding.get("path")
            ) != binding.get("sha256"):
                raise ValueError("Design selection evidence hash does not match its file.")
            payload = load_mapping(binding.get("path"))
            if payload.get("schema_version") != expected_schemas[evidence_field] or (
                payload.get("scientific_readout_allowed") is not False
            ):
                raise ValueError("Freeze inputs require v2 no-readout design evidence.")
            loaded_evidence[evidence_field].append(payload)
    if prototype_ids is not None:
        recomputed_filter = select_filter_candidate(
            [
                evaluate_filter_candidate(payload, prototype_ids=prototype_ids)
                for payload in loaded_evidence["filter_evidence"]
            ]
        )
        recomputed_planning = select_planning_branch_count(
            [
                evaluate_planning_candidate(payload)
                for payload in loaded_evidence["planning_evidence"]
            ]
        )
        if recomputed_filter != filter_selection or recomputed_planning != planning_selection:
            raise ValueError("R015 design selection does not recompute from its v2 evidence.")
    return (
        selected_filter,
        planning_selection,
        selected_particles,
        selected_branch_count,
    )


def _validate_partner_support_binding(
    *,
    registration_path: Path,
    report_path: Path,
    episode_evidence_path: Path,
    checkpoints: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """从登记、逐回合证据和 checkpoint 重新确认四个伙伴均已准入。"""

    from experiments.overcooked_v2.path_c_pool_admission import (
        R015PartnerSupportSpec,
        load_r015_partner_support_config,
        validate_r015_support_report_evidence,
    )

    support_spec = R015PartnerSupportSpec.from_mapping(
        load_r015_partner_support_config(registration_path)
    )
    report = load_mapping(report_path)
    members = validate_r015_support_report_evidence(support_spec, report)
    if report.get("support_complete") is not True or len(members) != 4:
        raise ValueError("R015 freeze requires four admitted partner candidates.")
    evidence = _require_mapping(
        report.get("episode_evidence"), field="support episode_evidence"
    )
    if Path(str(evidence.get("path"))).resolve() != episode_evidence_path.resolve():
        raise ValueError("R015 support report binds another episode-evidence file.")
    if evidence.get("sha256") != file_sha256(episode_evidence_path) or (
        _count_nonempty_lines(episode_evidence_path) != 1600
    ):
        raise ValueError("R015 support evidence hash or 1,600-row count changed.")
    partner_checkpoints = {
        int(item["training_seed"]): item
        for item in checkpoints
        if item.get("role") == "partner"
    }
    candidates_by_seed = {
        int(candidate.training_seed): candidate for candidate in support_spec.candidates
    }
    if set(partner_checkpoints) != set(candidates_by_seed):
        raise ValueError("R015 freeze partner checkpoints differ from support registration.")
    for seed, candidate in candidates_by_seed.items():
        member = _require_mapping(members.get(candidate.candidate_id), field="support member")
        checkpoint = partner_checkpoints[seed]
        for report_field, checkpoint_field in (
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("model_weights_sha256", "model_weights_sha256"),
            ("training_run_id", "training_run_id"),
        ):
            if member.get(report_field) != checkpoint.get(checkpoint_field):
                raise ValueError("R015 support member and freeze checkpoint identity differ.")
    return {
        "registration_path": str(registration_path.resolve()),
        "registration_sha256": file_sha256(registration_path),
        "report_path": str(report_path.resolve()),
        "report_sha256": file_sha256(report_path),
        "episode_evidence_path": str(episode_evidence_path.resolve()),
        "episode_evidence_sha256": file_sha256(episode_evidence_path),
        "episode_evidence_rows": 1600,
        "candidate_ids": [
            candidate.candidate_id for candidate in support_spec.candidates
        ],
    }


def build_pending_freeze_manifest(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build a content-addressed manifest without changing preregistration status."""

    if inputs.get("schema_version") != FREEZE_INPUT_SCHEMA:
        raise ValueError("R015 freeze inputs have the wrong schema.")
    settings = parse_delivery_reward(inputs.get("settings_path"))
    environment_sources = _bind_file_mapping(
        inputs.get("environment_sources"), field="environment_sources"
    )
    if set(environment_sources) != {"settings", "overcooked", "layouts", "common"}:
        raise ValueError("Freeze inputs must bind the four registered environment files.")
    if environment_sources["settings"]["sha256"] != settings["source_sha256"]:
        raise ValueError("settings.py bindings disagree.")
    checkpoint_inputs = _require_sequence(inputs.get("checkpoints"), field="checkpoints")
    if len(checkpoint_inputs) != 5:
        raise ValueError("R015 freeze requires one ego and four partner checkpoints.")
    checkpoints = [_bind_checkpoint(entry) for entry in checkpoint_inputs]
    checkpoint_contract = {
        entry["training_seed"]: {
            "role": entry["role"],
            "family_id": entry["family_id"],
            "effective_environment_steps": entry["effective_environment_steps"],
        }
        for entry in checkpoints
    }
    if checkpoint_contract != EXPECTED_FREEZE_CHECKPOINTS:
        raise ValueError("R015 freeze checkpoints differ from the signed production set.")
    for identity_field in (
        "checkpoint_sha256",
        "model_weights_sha256",
        "training_run_id",
        "artifact_id",
    ):
        values = [entry[identity_field] for entry in checkpoints]
        if len(set(values)) != len(values):
            raise ValueError(f"R015 checkpoint {identity_field} values must be distinct.")

    selection_path = Path(str(inputs.get("design_selection_report_path")))
    selection = load_mapping(selection_path)
    if selection.get("schema_version") != DESIGN_SELECTION_REPORT_SCHEMA or selection.get(
        "status"
    ) != "selected":
        raise ValueError("Freeze inputs require a successful design selection report.")
    if selection.get("scientific_readout_allowed") is not False:
        raise ValueError("Design selection cannot authorize a scientific readout.")
    if not _is_sha256(selection.get("protocol_sha256")) or file_sha256(
        selection.get("protocol_path")
    ) != selection.get("protocol_sha256"):
        raise ValueError("Design selection protocol hash does not match its file.")
    (
        selected_filter,
        planning_selection,
        selected_particles,
        selected_branch_count,
    ) = _validate_selected_design_evidence(selection)

    support_report_path = Path(str(inputs.get("support_report_path")))
    support_evidence_path = Path(str(inputs.get("support_episode_evidence_path")))
    support_registration_path = Path(str(inputs.get("support_registration_path")))
    support_binding = _validate_partner_support_binding(
        registration_path=support_registration_path,
        report_path=support_report_path,
        episode_evidence_path=support_evidence_path,
        checkpoints=checkpoints,
    )
    _validate_selected_design_evidence(
        selection,
        prototype_ids=tuple(support_binding["candidate_ids"]),
    )

    semantic_artifacts = _bind_file_mapping(
        inputs.get("semantic_artifacts"), field="semantic_artifacts"
    )
    required_semantics = {
        "response_vocabulary",
        "probe_registry",
        "ego_history_contract",
        "response_projection",
        "filter_implementation",
        "planner_implementation",
        "continuation_controller",
    }
    if set(semantic_artifacts) != required_semantics:
        raise ValueError("R015 semantic artifact set is incomplete.")
    execution_artifacts = _bind_file_mapping(
        inputs.get("execution_artifacts"), field="execution_artifacts"
    )
    if set(execution_artifacts) != EXECUTION_ARTIFACTS:
        raise ValueError("R015 executor and replay artifact set is incomplete.")
    preregistration_artifacts = _bind_file_mapping(
        inputs.get("preregistration_artifacts"), field="preregistration_artifacts"
    )
    if set(preregistration_artifacts) != PREFLIGHT_PREREGISTRATION_ARTIFACTS:
        raise ValueError("R015 preregistration artifact set is incomplete before pilot.")
    protocol_artifacts = _bind_file_mapping(
        inputs.get("protocol_artifacts"), field="protocol_artifacts"
    )
    if set(protocol_artifacts) != PROTOCOL_ARTIFACTS:
        raise ValueError("R015 design and pilot protocol bindings are incomplete.")
    if protocol_artifacts["design_data_protocol"]["sha256"] != selection.get(
        "protocol_sha256"
    ) or Path(protocol_artifacts["design_data_protocol"]["path"]).resolve() != Path(
        str(selection.get("protocol_path"))
    ).resolve():
        raise ValueError("R015 design protocol binding differs from selection evidence.")
    text_artifacts = _bind_file_mapping(inputs.get("text_artifacts"), field="text_artifacts")
    if set(text_artifacts) != {"audit_spec", "preregistration", "return_bound_derivation"}:
        raise ValueError("R015 text artifact set is incomplete.")

    return {
        "schema_version": FREEZE_MANIFEST_SCHEMA,
        "freeze_status": "pending_type_b_signature",
        "scientific_readout_allowed": False,
        "delivery_reward_readback": settings,
        "practical_margin_fraction": PRACTICAL_MARGIN_FRACTION,
        "practical_margin": PRACTICAL_MARGIN_FRACTION * settings["value"],
        "environment_sources": environment_sources,
        "checkpoints": checkpoints,
        "semantic_artifacts": semantic_artifacts,
        "execution_artifacts": execution_artifacts,
        "preregistration_artifacts": preregistration_artifacts,
        "protocol_artifacts": protocol_artifacts,
        "design_selection": {
            "report_path": str(selection_path.resolve()),
            "report_sha256": file_sha256(selection_path),
            "filter": copy.deepcopy(selected_filter),
            "planning_branch_count": selected_branch_count,
            "planning_branch_sampling_rule_id": BRANCH_SAMPLING_RULE_ID,
            "planning_branch_belief_rule_id": BRANCH_BELIEF_RULE_ID,
            "planning_grid_evaluation_rule_id": GRID_EVALUATION_RULE_ID,
            "filter_evidence": copy.deepcopy(selection["filter_evidence"]),
            "planning_evidence": copy.deepcopy(selection["planning_evidence"]),
        },
        "partner_support": support_binding,
        "text_artifacts": text_artifacts,
        "pilot_wiring_report": {
            "status": "not_run_in_registered_sequence",
            "path": None,
            "sha256": None,
        },
        "full_state_upper_bound": {"status": "not_provided"},
        "type_b_signoff": {
            "alpha": PRACTICAL_MARGIN_FRACTION,
            "authorization_quote": "pending_authorization_quote",
            "status": "pending",
        },
    }


def _pending_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            paths.extend(_pending_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            paths.extend(_pending_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and "pending" in value.lower():
        paths.append(prefix)
    return paths


def validate_ready_freeze_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject a manifest until every mechanical binding and Type-B quote is ready."""

    if manifest.get("schema_version") != FREEZE_MANIFEST_SCHEMA:
        raise ValueError("R015 freeze manifest has the wrong schema.")
    if manifest.get("freeze_status") != "ready_for_authorized_freeze_flip":
        raise ValueError("R015 freeze manifest is not in the signed pre-flip state.")
    if manifest.get("scientific_readout_allowed") is not False:
        raise ValueError("A pre-freeze R015 manifest cannot allow scientific readout.")
    reward_readback = _require_mapping(
        manifest.get("delivery_reward_readback"), field="delivery_reward_readback"
    )
    reward = _finite(reward_readback.get("value"), field="delivery reward")
    if not math.isclose(reward, EXPECTED_DELIVERY_REWARD, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("R015 frozen delivery reward differs from 20.")
    if not math.isclose(
        _finite(manifest.get("practical_margin_fraction"), field="margin fraction"),
        PRACTICAL_MARGIN_FRACTION,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        _finite(manifest.get("practical_margin"), field="practical margin"),
        PRACTICAL_MARGIN_FRACTION * reward,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("R015 frozen practical margin differs from 0.25 times reward.")
    pending = _pending_paths(manifest)
    if pending:
        raise ValueError("R015 freeze manifest retains pending value(s): " + ", ".join(pending))
    signoff = _require_mapping(manifest.get("type_b_signoff"), field="type_b_signoff")
    quote = signoff.get("authorization_quote")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("R015 Type-B authorization quote is empty.")
    if signoff.get("authorization_quote_sha256") != hashlib.sha256(
        quote.encode("utf-8")
    ).hexdigest():
        raise ValueError("R015 Type-B authorization quote hash does not match.")
    if not math.isclose(
        _finite(signoff.get("alpha"), field="Type-B alpha"),
        PRACTICAL_MARGIN_FRACTION,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("R015 Type-B alpha differs from 0.25.")
    if signoff.get("status") != "signed":
        raise ValueError("R015 Type-B signoff is not marked signed.")
    if dict(
        _require_mapping(
            manifest.get("full_state_upper_bound"), field="full_state_upper_bound"
        )
    ) != {"status": "not_provided"}:
        raise ValueError("R015 full-state upper bound must remain not provided.")
    selection = _require_mapping(manifest.get("design_selection"), field="design_selection")
    if not _is_sha256(selection.get("report_sha256")) or file_sha256(
        selection.get("report_path")
    ) != selection.get("report_sha256"):
        raise ValueError("R015 design selection report hash does not match.")
    selection_report = load_mapping(selection.get("report_path"))
    if selection_report.get("schema_version") != DESIGN_SELECTION_REPORT_SCHEMA or (
        selection_report.get("status") != "selected"
    ):
        raise ValueError("R015 design selection report is not a successful selection.")
    if selection_report.get("scientific_readout_allowed") is not False or not _is_sha256(
        selection_report.get("protocol_sha256")
    ) or file_sha256(selection_report.get("protocol_path")) != selection_report.get(
        "protocol_sha256"
    ):
        raise ValueError("R015 design selection protocol binding is invalid.")
    report_filter = _require_mapping(
        selection_report.get("filter_selection"), field="selection report filter"
    )
    report_planning = _require_mapping(
        selection_report.get("planning_selection"), field="selection report planning"
    )
    if selection.get("filter") != report_filter.get("selected_candidate") or (
        selection.get("planning_branch_count")
        != report_planning.get("selected_branch_count")
    ):
        raise ValueError("R015 frozen selection differs from its evidence report.")
    if selection.get("planning_branch_sampling_rule_id") != (
        BRANCH_SAMPLING_RULE_ID
    ) or selection.get("planning_branch_belief_rule_id") != (
        BRANCH_BELIEF_RULE_ID
    ) or selection.get("planning_grid_evaluation_rule_id") != (
        GRID_EVALUATION_RULE_ID
    ):
        raise ValueError("R015 freeze manifest changed its planning rules.")
    expected_planning_evidence_count = PLANNING_BRANCH_COUNTS.index(
        int(selection["planning_branch_count"])
    ) + 1
    selected_filter = _require_mapping(selection.get("filter"), field="design filter")
    expected_filter_evidence_count = FILTER_CANDIDATE_ORDER.index(
        (
            int(selected_filter["particles_per_prototype"]),
            str(selected_filter["resampling_timing"]),
        )
    ) + 1
    for evidence_field, expected_count in (
        ("filter_evidence", expected_filter_evidence_count),
        ("planning_evidence", expected_planning_evidence_count),
    ):
        if selection.get(evidence_field) != selection_report.get(evidence_field):
            raise ValueError("R015 selection evidence list differs from its report.")
        evidence = _require_sequence(selection.get(evidence_field), field=evidence_field)
        if len(evidence) != expected_count:
            raise ValueError("R015 selection evidence file count is incomplete.")
        for raw_entry in evidence:
            entry = _require_mapping(raw_entry, field=f"{evidence_field} entry")
            if not _is_sha256(entry.get("sha256")) or file_sha256(
                entry.get("path")
            ) != entry.get("sha256"):
                raise ValueError("R015 selection evidence hash does not match.")
    checkpoints = _require_sequence(manifest.get("checkpoints"), field="checkpoints")
    if len(checkpoints) != 5:
        raise ValueError("R015 freeze manifest must bind five checkpoints.")
    for raw_checkpoint in checkpoints:
        checkpoint = _require_mapping(raw_checkpoint, field="checkpoint binding")
        if not _is_sha256(checkpoint.get("checkpoint_sha256")) or (
            checkpoint_artifact_sha256(checkpoint.get("path"))
            != checkpoint.get("checkpoint_sha256")
        ):
            raise ValueError("R015 checkpoint artifact hash does not match.")
        if not _is_sha256(checkpoint.get("model_weights_sha256")) or checkpoint.get(
            "model_weights_hash_domain"
        ) != FLAX_WEIGHTS_HASH_DOMAIN:
            raise ValueError("R015 checkpoint weight hash binding is missing.")
        params = load_flax_parameter_tree(
            checkpoint.get("path"),
            checkpoint_format=str(checkpoint.get("format")),
            parameter_tree_path=tuple(checkpoint.get("parameter_tree_path", ())),
        )
        if flax_weights_sha256(params) != checkpoint.get("model_weights_sha256"):
            raise ValueError("R015 checkpoint weight hash does not match its parameter tree.")
        if not _is_sha256(checkpoint.get("training_run_id")):
            raise ValueError("R015 checkpoint training run identity is missing.")
        if not _is_sha256(checkpoint.get("training_manifest_sha256")) or file_sha256(
            checkpoint.get("training_manifest_path")
        ) != checkpoint.get("training_manifest_sha256"):
            raise ValueError("R015 training manifest hash does not match.")
    for identity_field in (
        "checkpoint_sha256",
        "model_weights_sha256",
        "training_run_id",
        "artifact_id",
    ):
        values = [
            _require_mapping(value, field="checkpoint binding").get(identity_field)
            for value in checkpoints
        ]
        if len(set(values)) != 5:
            raise ValueError(f"R015 checkpoint {identity_field} values are not distinct.")
    checkpoint_contract = {
        _require_mapping(value, field="checkpoint binding").get("training_seed"): {
            "role": _require_mapping(value, field="checkpoint binding").get("role"),
            "family_id": _require_mapping(value, field="checkpoint binding").get(
                "family_id"
            ),
            "effective_environment_steps": _require_mapping(
                value, field="checkpoint binding"
            ).get("effective_environment_steps"),
        }
        for value in checkpoints
    }
    if checkpoint_contract != EXPECTED_FREEZE_CHECKPOINTS:
        raise ValueError("R015 frozen checkpoints differ from the signed production set.")
    required_groups = {
        "environment_sources": {"settings", "overcooked", "layouts", "common"},
        "semantic_artifacts": {
            "response_vocabulary",
            "probe_registry",
            "ego_history_contract",
            "response_projection",
            "filter_implementation",
            "planner_implementation",
            "continuation_controller",
        },
        "text_artifacts": {
            "audit_spec",
            "preregistration",
            "return_bound_derivation",
        },
        "execution_artifacts": set(EXECUTION_ARTIFACTS),
        "preregistration_artifacts": set(PREFLIGHT_PREREGISTRATION_ARTIFACTS),
        "protocol_artifacts": set(PROTOCOL_ARTIFACTS),
    }
    for group_name, required_names in required_groups.items():
        group = _require_mapping(manifest.get(group_name), field=group_name)
        if set(group) != required_names:
            raise ValueError(f"R015 {group_name} binding set is incomplete.")
        for raw_binding in group.values():
            binding = _require_mapping(raw_binding, field=f"{group_name} binding")
            if not _is_sha256(binding.get("sha256")) or file_sha256(
                binding.get("path")
            ) != binding.get("sha256"):
                raise ValueError(f"R015 {group_name} file hash does not match.")
    parsed_reward = parse_delivery_reward(reward_readback.get("source_path"))
    if dict(reward_readback) != parsed_reward or manifest["environment_sources"][
        "settings"
    ]["sha256"] != parsed_reward["source_sha256"]:
        raise ValueError("R015 delivery-reward readback differs from settings.py.")
    support = _require_mapping(manifest.get("partner_support"), field="partner_support")
    for path_field, hash_field in (
        ("registration_path", "registration_sha256"),
        ("report_path", "report_sha256"),
        ("episode_evidence_path", "episode_evidence_sha256"),
    ):
        if not _is_sha256(support.get(hash_field)) or file_sha256(
            support.get(path_field)
        ) != support.get(hash_field):
            raise ValueError("R015 partner support hash does not match.")
    if load_mapping(support.get("report_path")).get("support_complete") is not True:
        raise ValueError("R015 partner support is not complete.")
    if support.get("episode_evidence_rows") != 1600 or _count_nonempty_lines(
        support.get("episode_evidence_path")
    ) != 1600:
        raise ValueError("R015 partner support row count changed.")
    if len(_require_sequence(support.get("candidate_ids"), field="support candidate_ids")) != 4:
        raise ValueError("R015 partner support must bind exactly four candidates.")
    recomputed_support = _validate_partner_support_binding(
        registration_path=Path(str(support["registration_path"])),
        report_path=Path(str(support["report_path"])),
        episode_evidence_path=Path(str(support["episode_evidence_path"])),
        checkpoints=[
            _require_mapping(value, field="checkpoint binding")
            for value in checkpoints
        ],
    )
    if dict(support) != dict(recomputed_support):
        raise ValueError("R015 partner support binding changed after manifest assembly.")
    _validate_selected_design_evidence(
        selection_report,
        prototype_ids=tuple(support["candidate_ids"]),
    )
    pilot = _require_mapping(
        manifest.get("pilot_wiring_report"), field="pilot_wiring_report"
    )
    if dict(pilot) != {
        "status": "not_run_in_registered_sequence",
        "path": None,
        "sha256": None,
    }:
        raise ValueError("R015 pilot slot changed before the registered pilot run.")
    return {
        "schema_version": FREEZE_VALIDATION_SCHEMA,
        "ready_for_authorized_freeze_flip": True,
        "manifest_sha256": canonical_sha256(manifest),
    }


def preregistration_fill_values(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the mechanical values to copy into the still-unfrozen preregistration."""

    selection = _require_mapping(manifest.get("design_selection"), field="design_selection")
    filtering = _require_mapping(selection.get("filter"), field="design filter")
    checkpoints = _require_sequence(manifest.get("checkpoints"), field="checkpoints")
    ego = next(
        (
            _require_mapping(value, field="ego checkpoint")
            for value in checkpoints
            if _require_mapping(value, field="checkpoint").get("role") == "ego"
        ),
        None,
    )
    if ego is None:
        raise ValueError("R015 freeze manifest lacks the ego checkpoint.")
    delivery_reward = _finite(
        _require_mapping(
            manifest.get("delivery_reward_readback"),
            field="delivery_reward_readback",
        ).get("value"),
        field="delivery reward",
    )
    protocol_artifacts = _require_mapping(
        manifest.get("protocol_artifacts"), field="protocol_artifacts"
    )
    partner_support = _require_mapping(
        manifest.get("partner_support"), field="partner_support"
    )
    preregistration_artifacts = _require_mapping(
        manifest.get("preregistration_artifacts"), field="preregistration_artifacts"
    )
    artifact_updates: dict[str, Any] = {}
    for name, raw_binding in preregistration_artifacts.items():
        binding = _require_mapping(raw_binding, field=f"preregistration artifact {name}")
        artifact_updates[f"artifacts.{name}.path"] = binding["path"]
        artifact_updates[f"artifacts.{name}.sha256"] = binding["sha256"]
    return {
        "schema_version": PREREGISTRATION_FILL_SCHEMA,
        "freeze_status_after_copy": "static_registered_not_frozen",
        "scientific_readout_allowed": False,
        "field_updates": {
            "controller.hidden_state_filter.particles_per_prototype": filtering[
                "particles_per_prototype"
            ],
            "controller.hidden_state_filter.resampling_algorithm": filtering[
                "resampling_algorithm"
            ],
            "controller.hidden_state_filter.resampling_interval_environment_steps": (
                filtering["resampling_interval_environment_steps"]
            ),
            "controller.hidden_state_filter.resampling_timing": filtering[
                "resampling_timing"
            ],
            "controller.hidden_state_filter.resampling_ess_fraction_threshold": (
                filtering["resampling_ess_fraction_threshold"]
            ),
            "controller.planning.branches_per_candidate": _positive_int(
                selection.get("planning_branch_count"),
                field="planning branch count",
            ),
            "statistics.delivery_reward": delivery_reward,
            "statistics.practical_margin_fraction": PRACTICAL_MARGIN_FRACTION,
            "design_data.protocol_path": protocol_artifacts[
                "design_data_protocol"
            ]["path"],
            "design_data.protocol_sha256": protocol_artifacts[
                "design_data_protocol"
            ]["sha256"],
            "design_data.pilot_protocol_path": protocol_artifacts[
                "pilot_protocol"
            ]["path"],
            "design_data.pilot_protocol_sha256": protocol_artifacts[
                "pilot_protocol"
            ]["sha256"],
            "design_data.filter_selection_report_path": selection["report_path"],
            "design_data.filter_selection_report_sha256": selection[
                "report_sha256"
            ],
            "design_data.planning_selection_report_path": selection["report_path"],
            "design_data.planning_selection_report_sha256": selection[
                "report_sha256"
            ],
            "partner_support.registration_path": partner_support[
                "registration_path"
            ],
            "partner_support.registration_sha256": partner_support[
                "registration_sha256"
            ],
            "partner_support.report_path": partner_support["report_path"],
            "artifacts.ego_checkpoint.path": ego["path"],
            "artifacts.ego_checkpoint.sha256": ego["checkpoint_sha256"],
            "artifacts.ego_checkpoint.model_weights_sha256": ego[
                "model_weights_sha256"
            ],
            "artifacts.ego_checkpoint.training_manifest_path": ego[
                "training_manifest_path"
            ],
            "artifacts.ego_checkpoint.training_run_id": ego["training_run_id"],
            **artifact_updates,
        },
        "content_bindings": {
            "environment_sources": copy.deepcopy(manifest["environment_sources"]),
            "semantic_artifacts": copy.deepcopy(manifest["semantic_artifacts"]),
            "execution_artifacts": copy.deepcopy(manifest["execution_artifacts"]),
            "preregistration_artifacts": copy.deepcopy(
                manifest["preregistration_artifacts"]
            ),
            "protocol_artifacts": copy.deepcopy(manifest["protocol_artifacts"]),
            "text_artifacts": copy.deepcopy(manifest["text_artifacts"]),
            "partner_support": copy.deepcopy(manifest["partner_support"]),
            "checkpoints": copy.deepcopy(list(checkpoints)),
        },
        "type_b_signoff": copy.deepcopy(dict(manifest.get("type_b_signoff", {}))),
    }


def apply_preregistration_fill_values(
    preregistration: Mapping[str, Any],
    fill_values: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Copy only registered mechanical fields into an unfrozen preregistration."""

    if preregistration.get("schema_version") != "path_c_r015_preregistration_v2":
        raise ValueError("R015 preregistration has the wrong schema.")
    if preregistration.get("status") != "template_not_valid_for_runs" or (
        preregistration.get("scientific_readout_allowed") is not False
    ):
        raise ValueError("R015 preregistration honesty markers changed before fill.")
    experiment = _require_mapping(
        preregistration.get("experiment"), field="preregistration experiment"
    )
    if experiment.get("freeze_status") != "static_registered_not_frozen":
        raise ValueError("R015 preregistration is not in the fillable static state.")
    if fill_values.get("schema_version") != (
        PREREGISTRATION_FILL_SCHEMA
    ) or fill_values.get("freeze_status_after_copy") != (
        "static_registered_not_frozen"
    ) or fill_values.get("scientific_readout_allowed") is not False:
        raise ValueError("R015 preregistration fill values changed the honesty state.")

    result = copy.deepcopy(dict(preregistration))
    updates = _require_mapping(fill_values.get("field_updates"), field="field_updates")
    expected_update_paths = {
        "controller.hidden_state_filter.particles_per_prototype",
        "controller.hidden_state_filter.resampling_algorithm",
        "controller.hidden_state_filter.resampling_interval_environment_steps",
        "controller.hidden_state_filter.resampling_timing",
        "controller.hidden_state_filter.resampling_ess_fraction_threshold",
        "controller.planning.branches_per_candidate",
        "statistics.delivery_reward",
        "statistics.practical_margin_fraction",
        "design_data.protocol_path",
        "design_data.protocol_sha256",
        "design_data.pilot_protocol_path",
        "design_data.pilot_protocol_sha256",
        "design_data.filter_selection_report_path",
        "design_data.filter_selection_report_sha256",
        "design_data.planning_selection_report_path",
        "design_data.planning_selection_report_sha256",
        "partner_support.registration_path",
        "partner_support.registration_sha256",
        "partner_support.report_path",
        "artifacts.ego_checkpoint.path",
        "artifacts.ego_checkpoint.sha256",
        "artifacts.ego_checkpoint.model_weights_sha256",
        "artifacts.ego_checkpoint.training_manifest_path",
        "artifacts.ego_checkpoint.training_run_id",
        *(
            f"artifacts.{name}.{field}"
            for name in PREFLIGHT_PREREGISTRATION_ARTIFACTS
            for field in ("path", "sha256")
        ),
    }
    if set(updates) != expected_update_paths:
        raise ValueError("R015 preregistration fill is incomplete or changes extra fields.")
    for dotted_path, value in updates.items():
        if not isinstance(dotted_path, str) or not dotted_path:
            raise ValueError("R015 preregistration update path is invalid.")
        parts = dotted_path.split(".")
        cursor: Any = result
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(
                    f"R015 preregistration update path is absent: {dotted_path}"
                )
            cursor = cursor[part]
        if not isinstance(cursor, dict) or parts[-1] not in cursor:
            raise ValueError(
                f"R015 preregistration update path is absent: {dotted_path}"
            )
        cursor[parts[-1]] = copy.deepcopy(value)
    if result["experiment"]["freeze_status"] != "static_registered_not_frozen" or (
        result["scientific_readout_allowed"] is not False
    ):
        raise ValueError("R015 preregistration fill changed the freeze/readout state.")
    return result


def validate_prefilled_preregistration(
    preregistration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """确认试点前预登记只保留试点报告和未来正式输出槽位。"""

    if preregistration.get("schema_version") != "path_c_r015_preregistration_v2":
        raise ValueError("R015 preregistration has the wrong schema.")
    if preregistration.get("status") != "template_not_valid_for_runs" or (
        preregistration.get("scientific_readout_allowed") is not False
    ):
        raise ValueError("R015 pre-pilot preregistration changed its honesty markers.")
    experiment = _require_mapping(
        preregistration.get("experiment"), field="preregistration experiment"
    )
    if experiment.get("freeze_status") != "static_registered_not_frozen":
        raise ValueError("R015 preregistration cannot be formally frozen before pilot.")
    pending = _pending_paths(preregistration)
    allowed_prefixes = (
        "$.artifacts.pilot_wiring_report.",
        "$.formal_data.",
    )
    disallowed = [
        path for path in pending if not any(path.startswith(prefix) for prefix in allowed_prefixes)
    ]
    if disallowed:
        raise ValueError(
            "R015 pre-pilot preregistration retains unresolved pre-pilot value(s): "
            + ", ".join(disallowed)
        )
    pilot = _require_mapping(
        _require_mapping(preregistration.get("artifacts"), field="artifacts").get(
            "pilot_wiring_report"
        ),
        field="artifacts.pilot_wiring_report",
    )
    if pilot.get("path") != "pending" or pilot.get("sha256") != "pending":
        raise ValueError("R015 pilot report must remain pending until the pilot finishes.")
    upper = _require_mapping(
        preregistration.get("full_state_upper_bound"), field="full_state_upper_bound"
    )
    if dict(upper) != {
        "status": "not_provided",
        "value": None,
        "proof_artifact_path": None,
        "proof_artifact_sha256": None,
    }:
        raise ValueError("R015 full-state upper bound must remain explicitly not provided.")
    return {
        "schema_version": "path_c_r015_prefilled_preregistration_validation_v1",
        "ready_for_second_manifest_pass": True,
        "content_sha256": canonical_sha256(preregistration),
        "allowed_pending_paths": sorted(pending),
    }


def sign_ready_freeze_manifest(
    manifest: Mapping[str, Any],
    *,
    authorization_quote: str,
) -> Mapping[str, Any]:
    """把第二遍清单写成待授权翻转状态，不开启科学读取。"""

    quote = str(authorization_quote).strip()
    if not quote:
        raise ValueError("R015 Type-B authorization quote is empty.")
    if manifest.get("schema_version") != FREEZE_MANIFEST_SCHEMA or manifest.get(
        "freeze_status"
    ) != "pending_type_b_signature" or manifest.get(
        "scientific_readout_allowed"
    ) is not False:
        raise ValueError("Only a second-pass pending R015 manifest can be signed.")
    result = copy.deepcopy(dict(manifest))
    result["freeze_status"] = "ready_for_authorized_freeze_flip"
    result["type_b_signoff"] = {
        "alpha": PRACTICAL_MARGIN_FRACTION,
        "authorization_quote": quote,
        "authorization_quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        "status": "signed",
    }
    validate_ready_freeze_manifest(result)
    return result


def authorize_pilot_freeze_manifest(
    manifest: Mapping[str, Any],
    *,
    authorization_quote: str,
) -> Mapping[str, Any]:
    """显式授权试点用冻结清单；预登记和科学读取状态均不在此处翻转。"""

    validation = validate_ready_freeze_manifest(manifest)
    quote = str(authorization_quote).strip()
    if not quote or _require_mapping(
        manifest.get("type_b_signoff"), field="type_b_signoff"
    ).get("authorization_quote") != quote:
        raise ValueError("R015 pilot-freeze authorization must repeat the signed quote.")
    result = copy.deepcopy(dict(manifest))
    result["freeze_status"] = "frozen"
    result["scientific_readout_allowed"] = False
    result["pilot_freeze_authorization"] = {
        "authorization_quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        "ready_manifest_sha256": validation["manifest_sha256"],
        "scope": "design_frozen_for_registered_pilot_only",
    }
    return result


def validate_authorized_pilot_freeze_manifest(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """重建待授权清单并核对显式试点冻结记录。"""

    if manifest.get("schema_version") != FREEZE_MANIFEST_SCHEMA or manifest.get(
        "freeze_status"
    ) != "frozen" or manifest.get("scientific_readout_allowed") is not False:
        raise ValueError("R015 pilot requires an authorized v2 no-readout manifest.")
    authorization = _require_mapping(
        manifest.get("pilot_freeze_authorization"), field="pilot_freeze_authorization"
    )
    if authorization.get("scope") != "design_frozen_for_registered_pilot_only":
        raise ValueError("R015 pilot-freeze authorization has the wrong scope.")
    signoff = _require_mapping(manifest.get("type_b_signoff"), field="type_b_signoff")
    if authorization.get("authorization_quote_sha256") != signoff.get(
        "authorization_quote_sha256"
    ):
        raise ValueError("R015 pilot-freeze authorization changed the signed quote.")
    ready_copy = copy.deepcopy(dict(manifest))
    ready_copy.pop("pilot_freeze_authorization", None)
    ready_copy["freeze_status"] = "ready_for_authorized_freeze_flip"
    validation = validate_ready_freeze_manifest(ready_copy)
    if authorization.get("ready_manifest_sha256") != validation.get("manifest_sha256"):
        raise ValueError("R015 pilot-freeze authorization binds another ready manifest.")
    return {
        "schema_version": "path_c_r015_authorized_pilot_freeze_validation_v1",
        "authorized_for_registered_pilot": True,
        "frozen_manifest_sha256": canonical_sha256(manifest),
        "ready_manifest_sha256": validation["manifest_sha256"],
    }


def bind_completed_pilot_report(
    manifest: Mapping[str, Any],
    *,
    pilot_report_path: str | Path,
) -> Mapping[str, Any]:
    """试点完成后只补入报告路径与摘要，不改变任何科学选择。"""

    if manifest.get("schema_version") != FREEZE_MANIFEST_SCHEMA or manifest.get(
        "freeze_status"
    ) != "frozen" or manifest.get("scientific_readout_allowed") is not False:
        raise ValueError("R015 pilot report requires an authorized no-readout manifest.")
    pilot = _require_mapping(
        manifest.get("pilot_wiring_report"), field="pilot_wiring_report"
    )
    if dict(pilot) != {
        "status": "not_run_in_registered_sequence",
        "path": None,
        "sha256": None,
    }:
        raise ValueError("R015 pilot report slot was already consumed.")
    path = Path(pilot_report_path).resolve()
    report = load_mapping(path)
    if report.get("schema_version") != "path_c_r015_pilot_wiring_report_v2" or (
        report.get("scientific_readout_allowed") is not False
    ):
        raise ValueError("R015 pilot report has the wrong schema or readout role.")
    checks = _require_mapping(report.get("wiring_checks"), field="pilot wiring_checks")
    if set(checks) != PILOT_WIRING_CHECKS or any(
        value is not True for value in checks.values()
    ):
        raise ValueError("R015 pilot report does not pass all nine wiring checks.")
    result = copy.deepcopy(dict(manifest))
    result["pilot_wiring_report"] = {
        "status": "passed",
        "path": str(path),
        "sha256": file_sha256(path),
    }
    result["scientific_readout_allowed"] = False
    return result


def bind_completed_pilot_report_to_preregistration(
    preregistration: Mapping[str, Any],
    completed_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """把试点报告路径与摘要补入预登记，继续保持正式读取关闭。"""

    validate_prefilled_preregistration(preregistration)
    pilot = _require_mapping(
        completed_manifest.get("pilot_wiring_report"), field="pilot_wiring_report"
    )
    if completed_manifest.get("freeze_status") != "frozen" or (
        completed_manifest.get("scientific_readout_allowed") is not False
    ) or pilot.get("status") != "passed" or not _is_sha256(pilot.get("sha256")) or (
        file_sha256(pilot.get("path")) != pilot.get("sha256")
    ):
        raise ValueError("R015 completed manifest lacks a verified pilot report.")
    result = copy.deepcopy(dict(preregistration))
    result["artifacts"]["pilot_wiring_report"] = {
        "path": pilot["path"],
        "sha256": pilot["sha256"],
    }
    if result["experiment"]["freeze_status"] != "static_registered_not_frozen" or (
        result["status"] != "template_not_valid_for_runs"
    ) or result["scientific_readout_allowed"] is not False:
        raise ValueError("R015 pilot binding changed the formal readout state.")
    return result


def _validate_completed_pilot_manifest(
    completed_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """重验试点冻结授权、试点文件摘要和九项检查。"""

    if completed_manifest.get("schema_version") != FREEZE_MANIFEST_SCHEMA or (
        completed_manifest.get("freeze_status") != "frozen"
    ) or completed_manifest.get("scientific_readout_allowed") is not False:
        raise ValueError("R015 formal freeze requires a completed no-readout pilot manifest.")
    pilot = _require_mapping(
        completed_manifest.get("pilot_wiring_report"), field="pilot_wiring_report"
    )
    if pilot.get("status") != "passed" or not _is_sha256(pilot.get("sha256")) or (
        file_sha256(pilot.get("path")) != pilot.get("sha256")
    ):
        raise ValueError("R015 completed pilot binding is missing or changed.")
    report = load_mapping(pilot.get("path"))
    checks = _require_mapping(report.get("wiring_checks"), field="pilot wiring_checks")
    if report.get("schema_version") != "path_c_r015_pilot_wiring_report_v2" or (
        report.get("scientific_readout_allowed") is not False
    ) or set(checks) != PILOT_WIRING_CHECKS or any(
        value is not True for value in checks.values()
    ):
        raise ValueError("R015 completed pilot no longer passes all nine wiring checks.")
    authorization_copy = copy.deepcopy(dict(completed_manifest))
    authorization_copy["pilot_wiring_report"] = {
        "status": "not_run_in_registered_sequence",
        "path": None,
        "sha256": None,
    }
    authorization_validation = validate_authorized_pilot_freeze_manifest(
        authorization_copy
    )
    return {
        "pilot_binding": dict(pilot),
        "pilot_report": dict(report),
        "pilot_authorization_validation": authorization_validation,
    }


def authorize_formal_freeze_after_pilot(
    completed_manifest: Mapping[str, Any],
    pilot_bound_preregistration: Mapping[str, Any],
    *,
    authorization_quote: str,
    formal_dataset_path: str | Path,
    firing_count_checkpoint_paths: Sequence[str | Path],
    formal_view_record_path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """用新的显式授权完成试点后的正式封存，不读取任何正式结果。"""

    pilot_validation = _validate_completed_pilot_manifest(completed_manifest)
    quote = str(authorization_quote).strip()
    if not quote:
        raise ValueError("R015 formal-freeze authorization quote is empty.")
    previous_signoff = _require_mapping(
        completed_manifest.get("type_b_signoff"), field="type_b_signoff"
    )
    previous_quote = str(previous_signoff.get("authorization_quote", "")).strip()
    if quote == previous_quote or hashlib.sha256(quote.encode("utf-8")).hexdigest() == (
        _require_mapping(
            completed_manifest.get("pilot_freeze_authorization"),
            field="pilot_freeze_authorization",
        ).get("authorization_quote_sha256")
    ):
        raise ValueError("R015 formal freeze requires a new authorization quote.")

    manifest_preregistration_binding = _require_mapping(
        _require_mapping(
            completed_manifest.get("text_artifacts"), field="text_artifacts"
        ).get("preregistration"),
        field="text_artifacts.preregistration",
    )
    if not _is_sha256(manifest_preregistration_binding.get("sha256")) or (
        file_sha256(manifest_preregistration_binding.get("path"))
        != manifest_preregistration_binding.get("sha256")
    ):
        raise ValueError("R015 filled preregistration text binding changed after pilot.")
    prepilot_preregistration = load_mapping(
        manifest_preregistration_binding.get("path")
    )
    validate_prefilled_preregistration(prepilot_preregistration)
    expected_pilot_bound = copy.deepcopy(dict(prepilot_preregistration))
    expected_pilot_bound["artifacts"]["pilot_wiring_report"] = copy.deepcopy(
        pilot_validation["pilot_binding"]
    )
    expected_pilot_bound["artifacts"]["pilot_wiring_report"].pop("status", None)
    if dict(pilot_bound_preregistration) != expected_pilot_bound:
        raise ValueError("R015 preregistration scientific fields drifted after pilot.")

    if not str(formal_dataset_path).strip() or not str(formal_view_record_path).strip():
        raise ValueError("R015 formal dataset and view paths cannot be empty.")
    raw_checkpoint_paths = _require_sequence(
        firing_count_checkpoint_paths, field="firing_count_checkpoint_paths"
    )
    if any(not str(path).strip() for path in raw_checkpoint_paths):
        raise ValueError("R015 firing-count checkpoint paths cannot be empty.")
    dataset_path = Path(formal_dataset_path).resolve()
    checkpoint_paths = tuple(Path(path).resolve() for path in raw_checkpoint_paths)
    view_path = Path(formal_view_record_path).resolve()
    if len(checkpoint_paths) != 5 or len(set(checkpoint_paths)) != 5:
        raise ValueError("R015 formal freeze requires five distinct firing checkpoints.")
    all_output_paths = (dataset_path, *checkpoint_paths, view_path)
    if len(set(all_output_paths)) != len(all_output_paths):
        raise ValueError("R015 formal output and single-view paths must be distinct.")
    if view_path.exists():
        raise ValueError("R015 formal view was already consumed before formal freeze.")

    frozen_preregistration = copy.deepcopy(dict(pilot_bound_preregistration))
    frozen_preregistration["formal_data"] = {
        "dataset_path": str(dataset_path),
        "firing_count_checkpoint_paths": [str(path) for path in checkpoint_paths],
        "view_consumption_record_path": str(view_path),
    }
    frozen_preregistration["status"] = "frozen"
    frozen_preregistration["experiment"]["freeze_status"] = "frozen"
    frozen_preregistration["scientific_readout_allowed"] = True
    pending = _pending_paths(frozen_preregistration)
    if pending:
        raise ValueError(
            "R015 formal preregistration retains pending value(s): " + ", ".join(pending)
        )

    frozen_manifest = copy.deepcopy(dict(completed_manifest))
    frozen_manifest["freeze_status"] = "frozen"
    frozen_manifest["scientific_readout_allowed"] = True
    frozen_manifest["formal_data"] = copy.deepcopy(
        frozen_preregistration["formal_data"]
    )
    frozen_manifest["formal_preregistration_canonical_sha256"] = canonical_sha256(
        frozen_preregistration
    )
    frozen_manifest["formal_freeze_authorization"] = {
        "authorization_quote": quote,
        "authorization_quote_sha256": hashlib.sha256(
            quote.encode("utf-8")
        ).hexdigest(),
        "completed_pilot_manifest_sha256": canonical_sha256(completed_manifest),
        "pilot_report_sha256": pilot_validation["pilot_binding"]["sha256"],
        "scope": "formal_r015_execution_after_completed_pilot",
    }
    return frozen_manifest, frozen_preregistration


def freeze_inputs_with_preregistration_path(
    inputs: Mapping[str, Any],
    preregistration_path: str | Path,
) -> Mapping[str, Any]:
    """第二遍装配只替换预登记文本路径，其余输入保持逐字节相同。"""

    if inputs.get("schema_version") != FREEZE_INPUT_SCHEMA:
        raise ValueError("R015 freeze inputs have the wrong schema.")
    result = copy.deepcopy(dict(inputs))
    text_artifacts = dict(
        _require_mapping(result.get("text_artifacts"), field="text_artifacts")
    )
    if set(text_artifacts) != {"audit_spec", "preregistration", "return_bound_derivation"}:
        raise ValueError("R015 text artifact set is incomplete.")
    text_artifacts["preregistration"] = str(Path(preregistration_path).resolve())
    result["text_artifacts"] = text_artifacts
    return result


def _jsonable(value: Any) -> Any:
    """把远端执行证据转换为稳定的 JSON 值。"""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"R015 design evidence contains unsupported value {type(value)!r}.")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_complete_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    records: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"R015 design JSONL has an incomplete line {line_number}."
            ) from error
        if not isinstance(record, Mapping):
            raise ValueError("R015 design JSONL records must be mappings.")
        records.append(record)
    return records


def _load_recoverable_jsonl_prefix(
    path: Path,
) -> tuple[list[Mapping[str, Any]], bool]:
    """读取完整记录，并只允许最后一个未换行片段被原子批次重建。

    设计历史的真实来源是先完成原子写入的每原型批次文件；规范 JSONL 只是随后按
    登记顺序追加的索引。中间坏行仍然关闭运行，只有文件末尾因崩溃留下的片段可以
    在批次摘要和全部完整前缀逐项核对后删除。
    """

    if not path.exists():
        return [], False
    raw = path.read_bytes()
    final_newline = raw.rfind(b"\n")
    complete_bytes = raw[: final_newline + 1] if final_newline >= 0 else b""
    tail = raw[final_newline + 1 :]
    records: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(complete_bytes.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"R015 design JSONL has a corrupt completed line {line_number}."
            ) from error
        if not isinstance(record, Mapping):
            raise ValueError("R015 design JSONL records must be mappings.")
        records.append(record)
    if tail.strip():
        try:
            final_record = json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # 调用方只能在原子批次已经重建并核对完整前缀后执行截尾。
            return records, True
        if not isinstance(final_record, Mapping):
            raise ValueError("R015 design JSONL final record must be a mapping.")
        records.append(final_record)
    return records, bool(tail)


def _rewrite_jsonl_atomic(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".repair.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    _jsonable(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_jsonl_record(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _design_seed(root_seed: int, namespace: str, prototype_id: str, index: int) -> int:
    digest = canonical_sha256(
        [namespace, int(root_seed), str(prototype_id), int(index)]
    )
    return int(digest[-8:], 16)


def _forbidden_effect_field(value: Any, prefix: str = "$") -> str | None:
    forbidden = {"Delta_net", "Delta_response", "Delta_cost", "formal_effect_interval"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in forbidden:
                return f"{prefix}.{key}"
            nested = _forbidden_effect_field(child, f"{prefix}.{key}")
            if nested is not None:
                return nested
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            nested = _forbidden_effect_field(child, f"{prefix}[{index}]")
            if nested is not None:
                return nested
    return None


def _collect_design_history(
    *,
    backend: Any,
    prototype_id: str,
    episode_seed: int,
    episode_index: int,
    design_sequence_index: int,
    design_episode_id: str,
) -> Mapping[str, Any]:
    """兼容单回合调用；正式设计入口使用同一设备方法的固定批次。"""

    root_key = canonical_sha256(
        ["r015_design_passive_history_v1", design_episode_id, episode_seed]
    )
    collected = backend.collect_design_history_batch(
        adapter=backend.full_horizon_executor.adapter,
        partner_prototype_id=prototype_id,
        episode_seeds=(episode_seed,),
        passive_history_root_keys=(root_key,),
    )
    episodes = collected.get("episodes")
    if not isinstance(episodes, Sequence) or len(episodes) != 1:
        raise RuntimeError("R015 design-history device batch returned another shape.")
    return _design_history_record_from_device_episode(
        prototype_id=prototype_id,
        episode_seed=episode_seed,
        episode_index=episode_index,
        design_sequence_index=design_sequence_index,
        design_episode_id=design_episode_id,
        episode=episodes[0],
    )


def _design_history_record_from_device_episode(
    *,
    prototype_id: str,
    episode_seed: int,
    episode_index: int,
    design_sequence_index: int,
    design_episode_id: str,
    episode: Mapping[str, Any],
) -> Mapping[str, Any]:
    """把设备扫描输出恢复成既有官方历史 schema，不增加科学字段。"""

    from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
        OFFICIAL_ACTION_ORDER,
    )
    from experiments.overcooked_v2.path_c_r015_controller import OfficialHistoryV1

    initial_observation = np.asarray(
        episode.get("initial_official_local_observation")
    )
    next_observations = np.asarray(
        episode.get("official_local_observation_after_step")
    )
    action_indices = np.asarray(episode.get("ego_action_indices"))
    rewards = np.asarray(episode.get("raw_team_rewards"))
    boundaries = np.asarray(episode.get("episode_boundaries"), dtype=np.bool_)
    if next_observations.shape[0] != 400 or action_indices.shape != (400,) or (
        rewards.shape != (400,) or boundaries.shape != (400,)
    ):
        raise ValueError("R015 device history does not contain exactly 400 steps.")
    if not bool(boundaries[-1]):
        raise ValueError("R015 design history did not complete exactly 400 steps.")
    if any(
        isinstance(value, bool) or not 0 <= int(value) < len(OFFICIAL_ACTION_ORDER)
        for value in action_indices.tolist()
    ):
        raise ValueError("R015 design history contains an unknown ego action.")
    records: list[Mapping[str, Any]] = [
        {
            "official_local_observation": initial_observation,
            "episode_boundaries": [True],
        }
    ]
    for step in range(400):
        records.append(
            {
                "official_local_observation": next_observations[step],
                "ego_action_history": [
                    OFFICIAL_ACTION_ORDER[int(action_indices[step])]
                ],
                "raw_team_reward_history": [float(rewards[step])],
                "episode_boundaries": [bool(boundaries[step])],
            }
        )
    history = OfficialHistoryV1(tuple(records))
    return {
        "schema_version": "path_c_r015_design_history_v1",
        "scientific_readout_allowed": False,
        "design_episode_id": design_episode_id,
        "design_block_id": design_episode_id,
        "partner_prototype_id": prototype_id,
        "episode_index": episode_index,
        "design_sequence_index": design_sequence_index,
        "episode_seed": episode_seed,
        "collection_action_rule": "ego_seed100_official_categorical_no_probe_v1",
        "official_history": history.to_payload(),
    }


def _replay_filter_episode(
    *,
    executor: Any,
    history_record: Mapping[str, Any],
    particles_per_prototype: int,
    resampling_timing: str,
    stream_id: str,
    retain_consultation_states: bool = True,
) -> tuple[Mapping[str, Any], Mapping[int, Any]]:
    from experiments.overcooked_v2.path_c_r015_controller import (
        OfficialHistoryV1,
        derive_controller_key,
    )

    records = tuple(history_record["official_history"])
    initial_key = canonical_sha256(
        [
            stream_id,
            particles_per_prototype,
            resampling_timing,
            history_record["design_sequence_index"],
        ]
    )
    closed = False
    consultations: list[Mapping[str, Any]] = []
    states_by_step: dict[int, Any] = {}
    try:
        belief = executor.initialize_belief_v1_diagnostic(
            official_initial_observation=records[0]["official_local_observation"],
            particles_per_prototype=particles_per_prototype,
            resampling_timing=resampling_timing,
            initialization_key=initial_key,
        )
        replay_history = OfficialHistoryV1((records[0],))
        for environment_step, record in enumerate(records[1:], start=1):
            passive_record = {
                "official_local_observation": record["official_local_observation"],
                "ego_action_history": record["ego_action_history"],
                "raw_team_reward_history": record["raw_team_reward_history"],
                "episode_boundaries": record["episode_boundaries"],
            }
            belief = executor.update_belief_v1_diagnostic(
                belief,
                passive_record,
                key=derive_controller_key(
                    initial_key,
                    "design_filter_update",
                    environment_step,
                ),
                materialize_state_hashes=False,
            )
            replay_history = replay_history.append(record)
            if environment_step in CONSULTATION_STEPS:
                consultations.append(
                    {
                        "environment_step": environment_step,
                        "prototype_posterior": dict(belief.prototype_weights),
                        "pre_resample_ess_fraction_by_prototype": dict(
                            belief.last_pre_resample_ess_fraction_by_prototype
                        ),
                    }
                )
                if retain_consultation_states:
                    belief = executor.materialize_belief_state_hashes(belief)
                    states_by_step[environment_step] = (
                        copy.deepcopy(belief),
                        copy.deepcopy(replay_history),
                    )
    except ValueError as error:
        if "compatible support" not in str(error):
            raise
        closed = True
    return (
        {
            "filter_stream_id": stream_id,
            "closed_for_zero_support": closed,
            "consultations": consultations,
        },
        states_by_step,
    )


def _select_planning_histories(
    *,
    histories: Sequence[Mapping[str, Any]],
    selected_filter_evidence: Mapping[str, Any],
    planning_filter_stream_id: str,
) -> tuple[Mapping[str, Any], ...]:
    """按登记的伙伴交错顺序选出能提供前 200 个咨询点的完整历史。

    过滤候选证据已经包含每条历史在第一条过滤随机流上的关闭标志。规划只从没有关闭、
    且包含全部 20 个登记咨询点的历史取点；这一步不重新执行过滤器，也不读取规划值。
    """

    episodes = _require_sequence(
        selected_filter_evidence.get("episodes"),
        field="selected filter evidence episodes",
    )
    evidence_by_episode: dict[str, Mapping[str, Any]] = {}
    for raw_episode in episodes:
        episode = _require_mapping(raw_episode, field="selected filter episode")
        episode_id = str(episode.get("design_episode_id", ""))
        if not episode_id or episode_id in evidence_by_episode:
            raise ValueError("Selected filter evidence has duplicate episode ids.")
        evidence_by_episode[episode_id] = episode

    ordered = sorted(
        (_require_mapping(item, field="design history") for item in histories),
        key=lambda item: (
            _positive_int(item.get("episode_index"), field="design episode index"),
            str(item.get("partner_prototype_id", "")),
        ),
    )
    selected: list[Mapping[str, Any]] = []
    point_count = 0
    for history in ordered:
        episode_id = str(history.get("design_episode_id", ""))
        episode = evidence_by_episode.get(episode_id)
        if episode is None:
            raise ValueError("Selected filter evidence omitted a design history.")
        repeat_matches = [
            _require_mapping(value, field="selected filter repeat")
            for value in _require_sequence(
                episode.get("filter_repeats"),
                field="selected filter repeats",
            )
            if _require_mapping(value, field="selected filter repeat").get(
                "filter_stream_id"
            )
            == planning_filter_stream_id
        ]
        if len(repeat_matches) != 1:
            raise ValueError(
                "Selected filter evidence must contain exactly one planning stream."
            )
        repeat = repeat_matches[0]
        if bool(repeat.get("closed_for_zero_support")):
            continue
        consultation_steps = tuple(
            _positive_int(
                _require_mapping(value, field="selected filter consultation").get(
                    "environment_step"
                ),
                field="selected filter consultation step",
            )
            for value in _require_sequence(
                repeat.get("consultations"),
                field="selected filter consultations",
            )
        )
        if consultation_steps != CONSULTATION_STEPS:
            raise ValueError(
                "A planning-source history lacks the complete registered consultations."
            )
        selected.append(history)
        point_count += len(consultation_steps)
        if point_count >= PLANNING_MINIMUM_POINT_COUNT:
            break
    if point_count != PLANNING_MINIMUM_POINT_COUNT:
        raise ValueError("R015 design did not provide exactly 200 planning points.")
    return tuple(selected)


def _selected_filter_initialization_key(history: Mapping[str, Any]) -> str:
    """重建 v2 过滤器登记的、与真实回合 seed 无关的初始化根。"""

    return canonical_sha256(
        [
            FILTER_DEVICE_KEY_CONTRACT_ID,
            PLANNING_FILTER_STREAM_ID,
            _nonnegative_int(
                history.get("design_sequence_index"),
                field="selected-filter design sequence index",
            ),
        ]
    )


def _write_binary_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _checkpoint_pytree_sha256(value: Any, *, domain: str) -> str:
    """以树结构、dtype、shape 和连续原始字节摘要 checkpoint 状态。"""

    import jax

    digest = hashlib.sha256((domain + "\0").encode("utf-8"))
    structure = str(jax.tree_util.tree_structure(value)).encode("utf-8")
    digest.update(len(structure).to_bytes(8, "big"))
    digest.update(structure)
    for leaf in jax.tree_util.tree_leaves(value):
        array = np.ascontiguousarray(np.asarray(leaf))
        for framed in (
            str(array.dtype).encode("ascii"),
            ",".join(str(int(item)) for item in array.shape).encode("ascii"),
            array.view(np.uint8).tobytes(),
        ):
            digest.update(len(framed).to_bytes(8, "big"))
            digest.update(framed)
    return digest.hexdigest()


def _load_selected_filter_checkpoint_payload(
    *,
    path: Path,
    expected_sha256: str,
) -> Mapping[str, Any]:
    """只在文件摘要匹配清单后恢复本次运行自己写出的设备状态。"""

    if not _is_sha256(expected_sha256) or file_sha256(path) != expected_sha256:
        raise ValueError("A selected-filter checkpoint file changed after writing.")
    restored = pickle.loads(path.read_bytes())
    if not isinstance(restored, Mapping) or set(restored) != {
        "schema_version",
        "codec_id",
        "metadata",
        "device_filter_state",
        "continuation_states_by_member_id",
    }:
        raise ValueError("A selected-filter checkpoint has a malformed payload.")
    if restored.get("schema_version") != SELECTED_FILTER_CHECKPOINT_SCHEMA or (
        restored.get("codec_id") != SELECTED_FILTER_CHECKPOINT_CODEC_ID
    ):
        raise ValueError("A selected-filter checkpoint has the wrong schema or codec.")
    return restored


def _load_or_build_selected_filter_checkpoints_v2(
    *,
    executor: Any,
    selected_histories: Sequence[Mapping[str, Any]],
    particles_per_prototype: int,
    resampling_timing: str,
    selected_filter_evidence_path: Path,
    runtime_source_bindings: Mapping[str, Any],
    conditioned_opening_prefix_cache_binding_sha256: str,
    output_dir: Path,
) -> tuple[tuple[Any, ...], Mapping[str, Any], Path]:
    """生成或恢复机械选中后的 200 个完整 v2 咨询点。

    候选网格仍只保存后验摘要。只有机械选择已经完成后，本函数才为固定的第一条
    过滤流同步推进过滤状态和五个延续策略循环状态，并在登记咨询边界落盘。规划
    因而不会重新走旧的逐粒子 v1 路径，也不会把候选摘要冒充完整粒子云。
    """

    histories = tuple(selected_histories)
    if not histories or len(histories) * len(CONSULTATION_STEPS) != (
        PLANNING_MINIMUM_POINT_COUNT
    ):
        raise ValueError("Selected-filter checkpointing requires exactly 200 points.")
    checkpoint_dir = output_dir / "selected_filter_consultation_checkpoints_v2"
    manifest_path = checkpoint_dir / "manifest.json"
    history_bindings = [
        {
            "design_episode_id": str(history["design_episode_id"]),
            "episode_index": int(history["episode_index"]),
            "partner_prototype_id": str(history["partner_prototype_id"]),
            "design_sequence_index": int(history["design_sequence_index"]),
            "official_history_sha256": canonical_sha256(
                history["official_history"]
            ),
            "filter_initialization_key": _selected_filter_initialization_key(
                history
            ),
        }
        for history in histories
    ]
    fixed_binding = {
        "filter_algorithm_id": FILTER_ALGORITHM_ID,
        "filter_key_contract": FILTER_DEVICE_KEY_CONTRACT_ID,
        "filter_stream_id": PLANNING_FILTER_STREAM_ID,
        "particles_per_prototype": int(particles_per_prototype),
        "resampling_timing": str(resampling_timing),
        "consultation_steps": list(CONSULTATION_STEPS),
        "selected_filter_evidence_path": str(
            selected_filter_evidence_path.resolve()
        ),
        "selected_filter_evidence_sha256": file_sha256(
            selected_filter_evidence_path
        ),
        "conditioned_opening_prefix_cache_binding_sha256": (
            conditioned_opening_prefix_cache_binding_sha256
        ),
        "runtime_source_bindings": dict(runtime_source_bindings),
        "history_bindings": history_bindings,
    }

    if manifest_path.exists():
        manifest = load_mapping(manifest_path)
        if manifest.get("schema_version") != (
            SELECTED_FILTER_CHECKPOINT_MANIFEST_SCHEMA
        ) or manifest.get("scientific_readout_allowed") is not False or (
            manifest.get("fixed_binding") != fixed_binding
        ):
            raise ValueError("Selected-filter checkpoint resume changed its binding.")
    else:
        scan = executor.replay_selected_consultation_states_v2(
            history_records=histories,
            particles_per_prototype=int(particles_per_prototype),
            resampling_timing=str(resampling_timing),
            stream_id=PLANNING_FILTER_STREAM_ID,
            consultation_steps=CONSULTATION_STEPS,
            conditioned_opening_prefix_cache_binding_sha256=(
                conditioned_opening_prefix_cache_binding_sha256
            ),
        )
        raw_points = tuple(
            _require_mapping(value, field="selected-filter consultation state")
            for value in _require_sequence(
                scan.get("consultation_states"),
                field="selected-filter consultation states",
            )
        )
        expected_coordinates = tuple(
            (
                str(history["design_episode_id"]),
                int(environment_step),
            )
            for history in histories
            for environment_step in CONSULTATION_STEPS
        )
        actual_coordinates = tuple(
            (
                str(point.get("design_episode_id", "")),
                _positive_int(
                    point.get("environment_step"),
                    field="selected-filter checkpoint step",
                ),
            )
            for point in raw_points
        )
        if actual_coordinates != expected_coordinates:
            raise RuntimeError(
                "Selected-filter device scan changed the registered point order."
            )
        checkpoint_refs = []
        for point in raw_points:
            episode_id = str(point["design_episode_id"])
            environment_step = int(point["environment_step"])
            history = next(
                item for item in histories if item["design_episode_id"] == episode_id
            )
            history_prefix = tuple(history["official_history"])[
                : environment_step + 1
            ]
            metadata = {
                "design_episode_id": episode_id,
                "episode_index": int(history["episode_index"]),
                "partner_prototype_id": str(history["partner_prototype_id"]),
                "environment_step": environment_step,
                "official_history_prefix_sha256": canonical_sha256(
                    history_prefix
                ),
                "filter_initialization_key": _selected_filter_initialization_key(
                    history
                ),
                "filter_algorithm_id": FILTER_ALGORITHM_ID,
                "filter_key_contract": FILTER_DEVICE_KEY_CONTRACT_ID,
                "filter_stream_id": PLANNING_FILTER_STREAM_ID,
                "particles_per_prototype": int(particles_per_prototype),
                "resampling_timing": str(resampling_timing),
                "continuation_controller_id": CONTINUATION_CONTROLLER_ID,
                "device_filter_state_sha256": _checkpoint_pytree_sha256(
                    point["device_filter_state"],
                    domain="path_c_r015_selected_filter_device_state_v1",
                ),
                "continuation_states_sha256": _checkpoint_pytree_sha256(
                    point["continuation_states_by_member_id"],
                    domain="path_c_r015_selected_filter_continuation_states_v1",
                ),
            }
            payload = {
                "schema_version": SELECTED_FILTER_CHECKPOINT_SCHEMA,
                "codec_id": SELECTED_FILTER_CHECKPOINT_CODEC_ID,
                "metadata": metadata,
                "device_filter_state": point["device_filter_state"],
                "continuation_states_by_member_id": point[
                    "continuation_states_by_member_id"
                ],
            }
            encoded = pickle.dumps(payload, protocol=5)
            checkpoint_path = checkpoint_dir / (
                f"{episode_id}_step_{environment_step:03d}.pkl"
            )
            if checkpoint_path.exists():
                orphan = _load_selected_filter_checkpoint_payload(
                    path=checkpoint_path,
                    expected_sha256=file_sha256(checkpoint_path),
                )
                orphan_metadata = _require_mapping(
                    orphan.get("metadata"),
                    field="orphan selected-filter checkpoint metadata",
                )
                if dict(orphan_metadata) != metadata or (
                    _checkpoint_pytree_sha256(
                        orphan["device_filter_state"],
                        domain="path_c_r015_selected_filter_device_state_v1",
                    )
                    != metadata["device_filter_state_sha256"]
                ) or (
                    _checkpoint_pytree_sha256(
                        orphan["continuation_states_by_member_id"],
                        domain=(
                            "path_c_r015_selected_filter_continuation_states_v1"
                        ),
                    )
                    != metadata["continuation_states_sha256"]
                ):
                    raise ValueError(
                        "An orphan selected-filter checkpoint changed its state."
                    )
            else:
                _write_binary_atomic(checkpoint_path, encoded)
            checkpoint_refs.append(
                {
                    **metadata,
                    "path": str(checkpoint_path.resolve()),
                    "sha256": file_sha256(checkpoint_path),
                }
            )
        manifest = {
            "schema_version": SELECTED_FILTER_CHECKPOINT_MANIFEST_SCHEMA,
            "scientific_readout_allowed": False,
            "fixed_binding": fixed_binding,
            "point_count": len(checkpoint_refs),
            "ordered_checkpoint_refs": checkpoint_refs,
            "device_execution": dict(
                _require_mapping(
                    scan.get("device_execution"),
                    field="selected-filter checkpoint device execution",
                )
            ),
        }
        _write_json_atomic(manifest_path, manifest)

    checkpoint_device_execution = _require_mapping(
        manifest.get("device_execution"),
        field="selected-filter checkpoint device execution",
    )
    opening_cache_mode = checkpoint_device_execution.get(
        "selected_replay_opening_cache_mode"
    )
    if (
        checkpoint_device_execution.get("filter_algorithm_id")
        != FILTER_ALGORITHM_ID
        or checkpoint_device_execution.get("filter_key_contract")
        != FILTER_DEVICE_KEY_CONTRACT_ID
        or checkpoint_device_execution.get(
            "requested_conditioned_opening_prefix_cache_binding_sha256"
        )
        != conditioned_opening_prefix_cache_binding_sha256
        or checkpoint_device_execution.get(
            "host_sync_inside_environment_loop"
        )
        is not False
        or opening_cache_mode
        not in {
            "reused_complete_prefix",
            "cold_rebuild_from_registered_roots",
        }
    ):
        raise ValueError(
            "Selected-filter checkpoint execution changed its filter or cache contract."
        )
    if opening_cache_mode == "reused_complete_prefix":
        if checkpoint_device_execution.get(
            "conditioned_opening_prefix_cache_binding_sha256"
        ) != conditioned_opening_prefix_cache_binding_sha256 or (
            checkpoint_device_execution.get(
                "conditioned_opening_prefix_particles_used_per_prototype"
            )
            != int(particles_per_prototype)
        ):
            raise ValueError(
                "Selected-filter checkpoint did not fully reuse its bound prefix."
            )
    elif checkpoint_device_execution.get(
        "conditioned_opening_prefix_cache_binding_sha256"
    ) is not None or checkpoint_device_execution.get(
        "conditioned_opening_prefix_particles_used_per_prototype"
    ) != 0:
        raise ValueError(
            "Selected-filter cold rebuild mixed cached and regenerated particles."
        )

    refs = tuple(
        _require_mapping(value, field="selected-filter checkpoint reference")
        for value in _require_sequence(
            manifest.get("ordered_checkpoint_refs"),
            field="selected-filter checkpoint references",
        )
    )
    if manifest.get("point_count") != PLANNING_MINIMUM_POINT_COUNT or len(refs) != (
        PLANNING_MINIMUM_POINT_COUNT
    ):
        raise ValueError("Selected-filter checkpoint manifest is incomplete.")
    history_by_episode = {
        str(history["design_episode_id"]): history for history in histories
    }
    point_sources = []
    for ref in refs:
        episode_id = str(ref.get("design_episode_id", ""))
        environment_step = _positive_int(
            ref.get("environment_step"), field="selected-filter restored step"
        )
        history = history_by_episode.get(episode_id)
        if history is None:
            raise ValueError("Selected-filter checkpoint refers to another history.")
        checkpoint_path = Path(str(ref.get("path", ""))).resolve()
        restored = _load_selected_filter_checkpoint_payload(
            path=checkpoint_path,
            expected_sha256=str(ref.get("sha256", "")),
        )
        metadata = _require_mapping(
            restored.get("metadata"), field="selected-filter checkpoint metadata"
        )
        expected_metadata = {key: ref[key] for key in metadata}
        if dict(metadata) != expected_metadata or metadata.get(
            "official_history_prefix_sha256"
        ) != canonical_sha256(
            tuple(history["official_history"])[: environment_step + 1]
        ):
            raise ValueError("Selected-filter checkpoint metadata changed.")
        if _checkpoint_pytree_sha256(
            restored["device_filter_state"],
            domain="path_c_r015_selected_filter_device_state_v1",
        ) != metadata.get("device_filter_state_sha256") or (
            _checkpoint_pytree_sha256(
                restored["continuation_states_by_member_id"],
                domain="path_c_r015_selected_filter_continuation_states_v1",
            )
            != metadata.get("continuation_states_sha256")
        ):
            raise ValueError("Selected-filter checkpoint state digest changed.")
        belief, replay_history = executor.materialize_selected_consultation_state_v2(
            device_filter_state=restored["device_filter_state"],
            continuation_states_by_member_id=restored[
                "continuation_states_by_member_id"
            ],
            resampling_timing=str(resampling_timing),
            official_history_records=tuple(history["official_history"])[
                : environment_step + 1
            ],
        )
        point_sources.append(
            (
                int(history["episode_index"]),
                str(history["partner_prototype_id"]),
                environment_step,
                history,
                (belief, replay_history, str(ref["sha256"])),
            )
        )
    if tuple(item[:3] for item in point_sources) != tuple(
        sorted(item[:3] for item in point_sources)
    ):
        raise ValueError("Selected-filter checkpoints changed the point order.")
    return tuple(point_sources), manifest, manifest_path


def _replay_filter_episode_group(
    *,
    executor: Any,
    history_stream_pairs: Sequence[tuple[Mapping[str, Any], str]],
    particles_per_prototype: int,
    resampling_timing: str,
) -> tuple[Mapping[str, Any], ...]:
    """批量推进多条独立设计历史；每条历史仍单独更新权重和重采样。"""

    from experiments.overcooked_v2.path_c_r015_controller import derive_controller_key

    contexts = []
    for history_record, stream_id in history_stream_pairs:
        records = tuple(history_record["official_history"])
        initial_key = canonical_sha256(
            [
                stream_id,
                particles_per_prototype,
                resampling_timing,
                history_record["design_sequence_index"],
            ]
        )
        belief = None
        closed = False
        try:
            belief = executor.initialize_belief_v1_diagnostic(
                official_initial_observation=records[0]["official_local_observation"],
                particles_per_prototype=particles_per_prototype,
                resampling_timing=resampling_timing,
                initialization_key=initial_key,
            )
        except ValueError as error:
            if "compatible support" not in str(error):
                raise
            closed = True
        contexts.append(
            {
                "records": records,
                "initial_key": initial_key,
                "stream_id": stream_id,
                "belief": belief,
                "closed": closed,
                "consultations": [],
            }
        )
    for environment_step in range(1, 401):
        active_indices = [
            index for index, context in enumerate(contexts) if not context["closed"]
        ]
        if not active_indices:
            break
        passive_records = []
        update_keys = []
        for index in active_indices:
            context = contexts[index]
            record = context["records"][environment_step]
            passive_records.append(
                {
                    "official_local_observation": record["official_local_observation"],
                    "ego_action_history": record["ego_action_history"],
                    "raw_team_reward_history": record["raw_team_reward_history"],
                    "episode_boundaries": record["episode_boundaries"],
                }
            )
            update_keys.append(
                derive_controller_key(
                    context["initial_key"],
                    "design_filter_update",
                    environment_step,
                )
            )
        updated = executor.update_beliefs_v1_diagnostic(
            tuple(contexts[index]["belief"] for index in active_indices),
            tuple(passive_records),
            keys=tuple(update_keys),
            materialize_state_hashes=False,
            close_zero_support=True,
        )
        for index, belief in zip(active_indices, updated):
            context = contexts[index]
            if belief is None:
                context["closed"] = True
                context["belief"] = None
                continue
            context["belief"] = belief
            if environment_step in CONSULTATION_STEPS:
                context["consultations"].append(
                    {
                        "environment_step": environment_step,
                        "prototype_posterior": dict(belief.prototype_weights),
                        "pre_resample_ess_fraction_by_prototype": dict(
                            belief.last_pre_resample_ess_fraction_by_prototype
                        ),
                    }
                )
    return tuple(
        {
            "filter_stream_id": context["stream_id"],
            "closed_for_zero_support": bool(context["closed"]),
            "consultations": context["consultations"],
        }
        for context in contexts
    )


def _array_sha256(value: Any, *, domain: str) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256((domain + "\0").encode("utf-8"))
    for framed in (
        str(array.dtype).encode("ascii"),
        ",".join(str(int(item)) for item in array.shape).encode("ascii"),
        array.view(np.uint8).tobytes(),
    ):
        digest.update(len(framed).to_bytes(8, "big"))
        digest.update(framed)
    return digest.hexdigest()


def _arrays_bitwise_equal(left: Any, right: Any) -> bool:
    left_array = np.ascontiguousarray(np.asarray(left))
    right_array = np.ascontiguousarray(np.asarray(right))
    return (
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and left_array.view(np.uint8).tobytes()
        == right_array.view(np.uint8).tobytes()
    )


def _first_bitwise_mismatch(left: Any, right: Any) -> Mapping[str, Any] | None:
    """只记录首个设备等价差异的位置和原始位型，不记录设计统计。"""

    left_array = np.ascontiguousarray(np.asarray(left))
    right_array = np.ascontiguousarray(np.asarray(right))
    if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
        return {
            "left_dtype": str(left_array.dtype),
            "right_dtype": str(right_array.dtype),
            "left_shape": list(left_array.shape),
            "right_shape": list(right_array.shape),
        }
    left_bytes = left_array.view(np.uint8).reshape(
        left_array.shape + (left_array.dtype.itemsize,)
    )
    right_bytes = right_array.view(np.uint8).reshape(
        right_array.shape + (right_array.dtype.itemsize,)
    )
    unequal = np.any(left_bytes != right_bytes, axis=-1)
    mismatch_locations = np.argwhere(unequal)
    if mismatch_locations.size == 0:
        return None
    index = tuple(int(item) for item in mismatch_locations[0])
    left_scalar = np.ascontiguousarray(left_array[index])
    right_scalar = np.ascontiguousarray(right_array[index])
    result: dict[str, Any] = {
        "index": list(index),
        "environment_step": index[0] + 1,
        "left_bytes_hex": left_scalar.view(np.uint8).tobytes().hex(),
        "right_bytes_hex": right_scalar.view(np.uint8).tobytes().hex(),
    }
    if left_array.dtype.kind == "f":
        result["left_float_hex"] = float(left_array[index]).hex()
        result["right_float_hex"] = float(right_array[index]).hex()
    return result


def _reference_filter_trace(
    *,
    executor: Any,
    history_record: Mapping[str, Any],
    particles_per_prototype: int,
    resampling_timing: str,
    stream_id: str,
) -> Mapping[str, Any]:
    """保留已验证逐步路径，供设备扫描在读取设计统计前逐位核对。"""

    from experiments.overcooked_v2.path_c_r015_controller import derive_controller_key

    records = tuple(history_record["official_history"])
    initial_key = canonical_sha256(
        [
            stream_id,
            particles_per_prototype,
            resampling_timing,
            history_record["design_sequence_index"],
        ]
    )
    belief = executor.initialize_belief_v1_diagnostic(
        official_initial_observation=records[0]["official_local_observation"],
        particles_per_prototype=particles_per_prototype,
        resampling_timing=resampling_timing,
        initialization_key=initial_key,
    )
    closed = False
    previous_ess = np.ones((4,), dtype=np.float64)
    weight_trace = []
    posterior_trace = []
    ess_trace = []
    closed_trace = []
    for environment_step, record in enumerate(records[1:], start=1):
        if not closed:
            passive_record = {
                "official_local_observation": record["official_local_observation"],
                "ego_action_history": record["ego_action_history"],
                "raw_team_reward_history": record["raw_team_reward_history"],
                "episode_boundaries": record["episode_boundaries"],
            }
            try:
                belief = executor.update_belief_v1_diagnostic(
                    belief,
                    passive_record,
                    key=derive_controller_key(
                        initial_key,
                        "design_filter_update",
                        environment_step,
                    ),
                    materialize_state_hashes=False,
                )
                previous_ess = np.asarray(
                    [
                        belief.last_pre_resample_ess_fraction_by_prototype[value]
                        for value in belief.prototype_ids
                    ],
                    dtype=np.float64,
                )
            except ValueError as error:
                if "compatible support" not in str(error):
                    raise
                closed = True
        weight_trace.append(
            np.asarray(
                [particle.weight for particle in belief.particles],
                dtype=np.float64,
            ).reshape((4, particles_per_prototype))
        )
        posterior_trace.append(
            np.asarray(
                [belief.prototype_weights[value] for value in belief.prototype_ids],
                dtype=np.float64,
            )
        )
        ess_trace.append(previous_ess.copy())
        closed_trace.append(closed)
    return {
        "particle_weights": np.stack(weight_trace),
        "posterior": np.stack(posterior_trace),
        "ess": np.stack(ess_trace),
        "closed": np.asarray(closed_trace, dtype=np.bool_),
    }


def _reference_filter_group_trace(
    *,
    executor: Any,
    history_stream_pairs: Sequence[tuple[Mapping[str, Any], str]],
    particles_per_prototype: int,
    resampling_timing: str,
    environment_step_limit: int = 400,
    return_state_diagnostics: bool = False,
) -> Mapping[str, Any]:
    """按既有逐条路径隔离推进车道，避免无关历史改变策略批宽。"""

    from experiments.overcooked_v2.path_c_r015_controller import derive_controller_key
    from experiments.overcooked_v2.path_c_r015_full_horizon import (
        R015_PARTICLE_LINEAGE_ID,
    )

    contexts = []
    for history_record, stream_id in history_stream_pairs:
        records = tuple(history_record["official_history"])
        initial_key = canonical_sha256(
            [
                stream_id,
                particles_per_prototype,
                resampling_timing,
                history_record["design_sequence_index"],
            ]
        )
        belief = executor.initialize_belief_v1_diagnostic(
            official_initial_observation=records[0]["official_local_observation"],
            particles_per_prototype=particles_per_prototype,
            resampling_timing=resampling_timing,
            initialization_key=initial_key,
        )
        contexts.append(
            {
                "records": records,
                "initial_key": initial_key,
                "belief": belief,
                "closed": False,
                "previous_ess": np.ones((4,), dtype=np.float64),
            }
        )
    weight_trace = []
    posterior_trace = []
    ess_trace = []
    closed_trace = []
    partner_action_trace = []
    resampling_due_trace = []
    selected_source_index_trace = []
    partner_recurrent_state_trace = []
    agent_0_observation_trace = []
    partner_recurrent_state_before_trace = []
    agent_0_observation_before_trace = []
    for environment_step in range(1, int(environment_step_limit) + 1):
        active_indices = [
            index for index, context in enumerate(contexts) if not context["closed"]
        ]
        passive_records = []
        update_keys = []
        source_index_by_lineage = {}
        for index in active_indices:
            context = contexts[index]
            record = context["records"][environment_step]
            passive_records.append(
                {
                    "official_local_observation": record[
                        "official_local_observation"
                    ],
                    "ego_action_history": record["ego_action_history"],
                    "raw_team_reward_history": record["raw_team_reward_history"],
                    "episode_boundaries": record["episode_boundaries"],
                }
            )
            update_key = derive_controller_key(
                context["initial_key"],
                "design_filter_update",
                environment_step,
            )
            update_keys.append(update_key)
            lineage_by_prototype = {
                prototype_id: {}
                for prototype_id in context["belief"].prototype_ids
            }
            for particle_index, particle in enumerate(
                context["belief"].particles
            ):
                transition_key = derive_controller_key(
                    update_key,
                    "filter_transition",
                    particle.prototype_id,
                    particle_index,
                )
                lineage = canonical_sha256(
                    {
                        "schema_version": R015_PARTICLE_LINEAGE_ID,
                        "parent_state_sha256": particle.state_sha256,
                        "transition_key": transition_key,
                    }
                )
                local_index = particle_index % particles_per_prototype
                lineage_by_prototype[particle.prototype_id][lineage] = local_index
            source_index_by_lineage[index] = lineage_by_prototype
        step_partner_actions = np.full(
            (len(contexts), 4, particles_per_prototype),
            -1,
            dtype=np.int32,
        )
        step_resampling_due = np.zeros(
            (len(contexts), 4),
            dtype=np.bool_,
        )
        step_selected_source = np.full(
            (len(contexts), 4, particles_per_prototype),
            -1,
            dtype=np.int32,
        )
        if active_indices:
            captured_outputs = []
            captured_inputs = []
            original_batch_runner = (
                executor.production_backend.run_frozen_trajectory_batch
            )

            def capture_batch_outputs(*args: Any, **kwargs: Any) -> Any:
                captured_inputs.extend(tuple(kwargs["kernels"]))
                outputs = original_batch_runner(*args, **kwargs)
                captured_outputs.extend(outputs)
                return outputs

            executor.production_backend.run_frozen_trajectory_batch = (
                capture_batch_outputs
            )
            try:
                updated = executor.update_beliefs_v1_diagnostic(
                    tuple(contexts[index]["belief"] for index in active_indices),
                    tuple(passive_records),
                    keys=tuple(update_keys),
                    materialize_state_hashes=False,
                    close_zero_support=True,
                )
            finally:
                executor.production_backend.run_frozen_trajectory_batch = (
                    original_batch_runner
                )
            expected_outputs = len(active_indices) * 4 * particles_per_prototype
            if len(captured_outputs) != expected_outputs:
                raise RuntimeError(
                    "R015 grouped reference captured an unexpected transition count."
                )
            captured_actions = np.asarray(
                [
                    int(np.asarray(output["partner_actions"]).reshape(-1)[0])
                    for output in captured_outputs
                ],
                dtype=np.int32,
            ).reshape((len(active_indices), 4, particles_per_prototype))
            step_partner_actions[active_indices] = captured_actions
            if return_state_diagnostics:
                first_state = np.asarray(
                    captured_outputs[0]["partner_recurrent_state"]
                )
                first_observation = np.asarray(
                    captured_outputs[0]["raw_observation"]["agent_0"]
                )
                first_state_before = np.asarray(
                    captured_inputs[0].partner_recurrent_state
                )
                first_observation_before = np.asarray(
                    captured_inputs[0].snapshot.raw_obs["agent_0"]
                )
                step_partner_state = np.zeros(
                    (
                        len(contexts),
                        4,
                        particles_per_prototype,
                    )
                    + first_state.shape[1:],
                    dtype=first_state.dtype,
                )
                step_agent_0_observation = np.zeros(
                    (
                        len(contexts),
                        4,
                        particles_per_prototype,
                    )
                    + first_observation.shape,
                    dtype=first_observation.dtype,
                )
                step_partner_state_before = np.zeros(
                    (
                        len(contexts),
                        4,
                        particles_per_prototype,
                    )
                    + first_state_before.shape[1:],
                    dtype=first_state_before.dtype,
                )
                step_agent_0_observation_before = np.zeros(
                    (
                        len(contexts),
                        4,
                        particles_per_prototype,
                    )
                    + first_observation_before.shape,
                    dtype=first_observation_before.dtype,
                )
                outputs_per_lane = 4 * particles_per_prototype
                for local_index, context_index in enumerate(active_indices):
                    start = local_index * outputs_per_lane
                    stop = start + outputs_per_lane
                    lane_outputs = captured_outputs[start:stop]
                    step_partner_state[context_index] = np.concatenate(
                        [
                            np.asarray(output["partner_recurrent_state"])
                            for output in lane_outputs
                        ],
                        axis=0,
                    ).reshape(
                        (4, particles_per_prototype)
                        + first_state.shape[1:]
                    )
                    step_agent_0_observation[context_index] = np.stack(
                        [
                            np.asarray(output["raw_observation"]["agent_0"])
                            for output in lane_outputs
                        ]
                    ).reshape(
                        (4, particles_per_prototype)
                        + first_observation.shape
                    )
                    lane_inputs = captured_inputs[start:stop]
                    step_partner_state_before[context_index] = np.concatenate(
                        [
                            np.asarray(kernel.partner_recurrent_state)
                            for kernel in lane_inputs
                        ],
                        axis=0,
                    ).reshape(
                        (4, particles_per_prototype)
                        + first_state_before.shape[1:]
                    )
                    step_agent_0_observation_before[context_index] = np.stack(
                        [
                            np.asarray(kernel.snapshot.raw_obs["agent_0"])
                            for kernel in lane_inputs
                        ]
                    ).reshape(
                        (4, particles_per_prototype)
                        + first_observation_before.shape
                    )
                partner_recurrent_state_trace.append(step_partner_state)
                agent_0_observation_trace.append(step_agent_0_observation)
                partner_recurrent_state_before_trace.append(
                    step_partner_state_before
                )
                agent_0_observation_before_trace.append(
                    step_agent_0_observation_before
                )
            for index, belief in zip(active_indices, updated):
                context = contexts[index]
                if belief is None:
                    context["closed"] = True
                    continue
                context["belief"] = belief
                context["previous_ess"] = np.asarray(
                    [
                        belief.last_pre_resample_ess_fraction_by_prototype[value]
                        for value in belief.prototype_ids
                    ],
                    dtype=np.float64,
                )
                for prototype_index, prototype_id in enumerate(
                    belief.prototype_ids
                ):
                    step_resampling_due[index, prototype_index] = (
                        prototype_id in belief.last_resampled_prototypes
                    )
                    output_particles = [
                        particle
                        for particle in belief.particles
                        if particle.prototype_id == prototype_id
                    ]
                    lineage_map = source_index_by_lineage[index][prototype_id]
                    step_selected_source[index, prototype_index] = np.asarray(
                        [
                            lineage_map[particle.state_sha256]
                            for particle in output_particles
                        ],
                        dtype=np.int32,
                    )
        weight_trace.append(
            np.stack(
                [
                    np.asarray(
                        [particle.weight for particle in context["belief"].particles],
                        dtype=np.float64,
                    ).reshape((4, particles_per_prototype))
                    for context in contexts
                ]
            )
        )
        posterior_trace.append(
            np.stack(
                [
                    np.asarray(
                        [
                            context["belief"].prototype_weights[value]
                            for value in context["belief"].prototype_ids
                        ],
                        dtype=np.float64,
                    )
                    for context in contexts
                ]
            )
        )
        ess_trace.append(
            np.stack([context["previous_ess"] for context in contexts])
        )
        closed_trace.append(
            np.asarray(
                [context["closed"] for context in contexts],
                dtype=np.bool_,
            )
        )
        partner_action_trace.append(step_partner_actions)
        resampling_due_trace.append(step_resampling_due)
        selected_source_index_trace.append(step_selected_source)
    result = {
        "particle_weights": np.stack(weight_trace),
        "posterior": np.stack(posterior_trace),
        "ess": np.stack(ess_trace),
        "closed": np.stack(closed_trace),
        "partner_actions": np.stack(partner_action_trace),
        "resampling_due": np.stack(resampling_due_trace),
        "selected_source_indices": np.stack(selected_source_index_trace),
    }
    if return_state_diagnostics:
        result["partner_recurrent_state"] = np.stack(
            partner_recurrent_state_trace
        )
        result["agent_0_observation"] = np.stack(
            agent_0_observation_trace
        )
        result["partner_recurrent_state_before"] = np.stack(
            partner_recurrent_state_before_trace
        )
        result["agent_0_observation_before"] = np.stack(
            agent_0_observation_before_trace
        )
    return result


def run_r015_filter_device_equivalence(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """只复现旧 v1 设备扫描证明；新版过滤器另用精确枚举参照核查。"""

    if config.get("_execution_authorized") is not True:
        raise PermissionError("R015 filter equivalence requires recorded authorization.")
    protocol_path = Path(str(config.get("_protocol_path", ""))).resolve()
    output_dir = Path(str(config.get("_output_dir", ""))).resolve()
    proof_path = Path(str(config.get("_equivalence_output_path", ""))).resolve()
    if not protocol_path.is_file() or not output_dir.is_dir() or not str(
        config.get("_equivalence_output_path", "")
    ):
        raise ValueError("R015 filter equivalence requires protocol, data, and output paths.")
    histories_path = output_dir / "design_histories.jsonl"
    histories_sha256 = file_sha256(histories_path)
    if histories_sha256 != SUPERSEDED_V1_DESIGN_HISTORIES_SHA256:
        raise ValueError("R015 fixed design-history digest changed before equivalence.")
    histories = _load_complete_jsonl(histories_path)
    if len(histories) != 80:
        raise ValueError("R015 equivalence requires the fixed 80 design histories.")
    protocol = load_r015_design_protocol(protocol_path)

    from experiments.overcooked_v2.official import r015_runtime_bridge
    from experiments.overcooked_v2.path_c_pool_admission import (
        R015PartnerSupportSpec,
        load_r015_partner_support_config,
    )

    support = _require_mapping(protocol.get("support"), field="support")
    support_path = Path(str(support["partner_support_registration"]))
    if not support_path.is_absolute():
        support_path = (protocol_path.parent / support_path).resolve()
    backend = r015_runtime_bridge.build_official_r015_production_backend(
        protocol,
        R015PartnerSupportSpec.from_mapping(
            load_r015_partner_support_config(support_path)
        ),
        config_base_path=protocol_path.parent,
    )
    stream_ids = (
        "r015_filter_repeat_0_20260715_v1",
        "r015_filter_repeat_1_20260715_v1",
    )
    particles_per_prototype, resampling_timing = (
        FILTER_DEVICE_EQUIVALENCE_CANDIDATE
    )
    diagnostic_steps_raw = config.get("_filter_device_diagnostic_steps")
    diagnostic_only = diagnostic_steps_raw is not None
    compared_steps = (
        400
        if diagnostic_steps_raw is None
        else _positive_int(
            diagnostic_steps_raw,
            field="filter device diagnostic steps",
        )
    )
    if compared_steps > 400:
        raise ValueError("R015 filter device comparison cannot exceed 400 steps.")
    pairs = [
        (history, stream_id)
        for history in histories[:2]
        for stream_id in stream_ids
    ]
    device = backend.full_horizon_executor.replay_filter_candidate_device_v1(
        history_stream_pairs=pairs,
        particles_per_prototype=particles_per_prototype,
        resampling_timing=resampling_timing,
        return_particle_weight_trace=True,
        environment_step_limit=compared_steps,
        return_state_diagnostics=diagnostic_only,
    )
    reference_arrays = _reference_filter_group_trace(
        executor=backend.full_horizon_executor,
        history_stream_pairs=pairs,
        particles_per_prototype=particles_per_prototype,
        resampling_timing=resampling_timing,
        environment_step_limit=compared_steps,
        return_state_diagnostics=diagnostic_only,
    )
    device_arrays = {
        "particle_weights": np.asarray(device["particle_weight_trace"]),
        "posterior": np.asarray(device["posterior_trace"]),
        "ess": np.asarray(device["ess_trace"]),
        "closed": np.asarray(device["closed_trace"]),
    }
    reference_diagnostics = {
        "partner_actions": np.asarray(reference_arrays.pop("partner_actions")),
        "resampling_due": np.asarray(reference_arrays.pop("resampling_due")),
        "selected_source_indices": np.asarray(
            reference_arrays.pop("selected_source_indices")
        ),
    }
    if diagnostic_only:
        reference_diagnostics.update(
            {
                "partner_recurrent_state": np.asarray(
                    reference_arrays.pop("partner_recurrent_state")
                ),
                "agent_0_observation": np.asarray(
                    reference_arrays.pop("agent_0_observation")
                ),
                "partner_recurrent_state_before": np.asarray(
                    reference_arrays.pop("partner_recurrent_state_before")
                ),
                "agent_0_observation_before": np.asarray(
                    reference_arrays.pop("agent_0_observation_before")
                ),
            }
        )
    device_diagnostics = {
        "partner_actions": np.asarray(device["partner_action_trace"]),
        "resampling_due": np.asarray(device["resampling_due_trace"]),
        "selected_source_indices": np.asarray(
            device["selected_source_index_trace"]
        ),
    }
    if diagnostic_only:
        device_diagnostics.update(
            {
                "partner_recurrent_state": np.asarray(
                    device["partner_recurrent_state_trace"]
                ),
                "agent_0_observation": np.asarray(
                    device["agent_0_observation_trace"]
                ),
                "partner_recurrent_state_before": np.asarray(
                    device["partner_recurrent_state_before_trace"]
                ),
                "agent_0_observation_before": np.asarray(
                    device["agent_0_observation_before_trace"]
                ),
            }
        )
    equality = {
        name: _arrays_bitwise_equal(
            reference_arrays[name],
            device_arrays[name],
        )
        for name in reference_arrays
    }
    proof = {
        "schema_version": (
            FILTER_DEVICE_PREFIX_DIAGNOSTIC_SCHEMA
            if diagnostic_only
            else FILTER_DEVICE_EQUIVALENCE_SCHEMA_V1
        ),
        "diagnostic_only": diagnostic_only,
        "scientific_readout_allowed": False,
        "execution_id": "r015_filter_jit_scan_vmap_v1",
        "design_histories_sha256": histories_sha256,
        "history_count": 2,
        "filter_stream_count_per_history": 2,
        "candidate": {
            "particles_per_prototype": particles_per_prototype,
            "resampling_timing": resampling_timing,
        },
        "compared_environment_steps_per_lane": compared_steps,
        "particle_weights_bitwise_equal": equality["particle_weights"],
        "prototype_posterior_bitwise_equal": equality["posterior"],
        "ess_bitwise_equal": equality["ess"],
        "closure_flags_bitwise_equal": equality["closed"],
        "reference_sha256": {
            name: _array_sha256(value, domain=f"r015_filter_reference_{name}_v1")
            for name, value in reference_arrays.items()
        },
        "device_sha256": {
            name: _array_sha256(value, domain=f"r015_filter_reference_{name}_v1")
            for name, value in device_arrays.items()
        },
        "device_throughput": dict(device["throughput"]),
        "diagnostic_bitwise_equal": {
            name: _arrays_bitwise_equal(
                reference_diagnostics[name],
                device_diagnostics[name],
            )
            for name in reference_diagnostics
        },
        "passed": all(equality.values()),
    }
    if not proof["passed"]:
        proof["first_bitwise_mismatch_by_quantity"] = {
            name: _first_bitwise_mismatch(
                reference_arrays[name],
                device_arrays[name],
            )
            for name in reference_arrays
            if not equality[name]
        }
        proof["first_bitwise_mismatch_by_diagnostic"] = {
            name: _first_bitwise_mismatch(
                reference_diagnostics[name],
                device_diagnostics[name],
            )
            for name in reference_diagnostics
            if not proof["diagnostic_bitwise_equal"][name]
        }
    _write_json_atomic(proof_path, proof)
    if not proof["passed"] and not diagnostic_only:
        raise RuntimeError("R015 device filter failed the bitwise equivalence gate.")
    return {**proof, "proof_path": str(proof_path), "proof_sha256": file_sha256(proof_path)}


def _precompute_planning_length_buckets_v2(
    *,
    point_sources: Sequence[tuple[Any, ...]],
    doubled_branch_count: int,
    verified_branch_cache: Mapping[str, Sequence[Mapping[str, Any]]],
    executor: Any,
    probe_scripts: Sequence[Any],
    schedule_type: Any,
    request_type: Any,
) -> Mapping[str, Any]:
    """按剩余回合长度合并 200 个咨询点的新增样本。"""

    requests_by_remaining: dict[int, list[Any]] = {}
    for source in point_sources:
        episode_index, prototype_id, environment_step, history, state = source
        del episode_index, prototype_id
        belief, replay_history = state[:2]
        checkpoint_sha256 = (
            str(state[2])
            if len(state) >= 3
            else canonical_sha256(
                {
                    "schema_version": (
                        "path_c_r015_in_process_filter_checkpoint_v2"
                    ),
                    "design_episode_id": history["design_episode_id"],
                    "environment_step": environment_step,
                    "particle_state_sha256": [
                        particle.state_sha256 for particle in belief.particles
                    ],
                }
            )
        )
        consultation_id = canonical_sha256(
            [history["design_episode_id"], environment_step]
        )
        planning_key = _six_coordinate_random_key(
            audit_unit_id=history["design_block_id"],
            partner_prototype_id=history["partner_prototype_id"],
            episode_seed=int(history["episode_seed"]),
            purpose="value_planning",
            environment_step=environment_step,
        )
        schedule = schedule_type.build(
            belief=belief,
            planning_key=planning_key,
            sample_count=doubled_branch_count,
        )
        cached_slots = {
            int(branch["canonical_slot_16"])
            for branch in verified_branch_cache.get(consultation_id, ())
        }
        missing_samples = tuple(
            sample
            for sample in schedule.samples
            if sample.canonical_slot_16 not in cached_slots
        )
        if not missing_samples:
            continue
        remaining_steps = 400 - int(environment_step)
        requests_by_remaining.setdefault(remaining_steps, []).append(
            request_type(
                consultation_id=consultation_id,
                filter_checkpoint_sha256=checkpoint_sha256,
                belief=belief,
                history=replay_history,
                samples=missing_samples,
                remaining_steps=remaining_steps,
            )
        )

    compiled_batch_calls = 0
    active_batch_sizes: list[int] = []
    bucket_reports: list[Mapping[str, Any]] = []
    wall_seconds = 0.0
    true_environment_transitions = 0
    computed_environment_transitions_including_padding = 0
    branch_head_particle_transitions = 0
    for remaining_steps in sorted(requests_by_remaining, reverse=True):
        result = executor.rollout_planning_length_bucket_v2(
            requests=tuple(requests_by_remaining[remaining_steps]),
            scripts=tuple(probe_scripts),
            remaining_steps=remaining_steps,
        )
        device = dict(result.device_execution)
        if device.get("host_sync_inside_environment_loop") is not False:
            raise RuntimeError("R015 length bucket synchronized inside its scan.")
        calls = int(device.get("compiled_batch_calls", 0))
        sizes = tuple(int(value) for value in device.get("active_batch_sizes", ()))
        if calls != 4 or len(sizes) != 4 or any(value <= 0 for value in sizes):
            raise RuntimeError("R015 length bucket changed its four device stages.")
        compiled_batch_calls += calls
        active_batch_sizes.extend(sizes)
        wall_seconds += float(device["wall_seconds"])
        true_environment_transitions += int(
            device["true_environment_transitions"]
        )
        computed_environment_transitions_including_padding += int(
            device["computed_environment_transitions_including_padding"]
        )
        branch_head_particle_transitions += int(
            device["branch_head_particle_transitions"]
        )
        bucket_reports.append(device)
    return {
        "compiled_batch_calls": compiled_batch_calls,
        "active_batch_sizes": tuple(active_batch_sizes),
        "wall_seconds": wall_seconds,
        "true_environment_transitions": true_environment_transitions,
        "computed_environment_transitions_including_padding": (
            computed_environment_transitions_including_padding
        ),
        "branch_head_particle_transitions": branch_head_particle_transitions,
        "length_bucket_reports": tuple(bucket_reports),
        "host_sync_inside_environment_loop": False,
    }


def run_r015_design(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """生成登记设计数据并机械选择过滤器数值和规划分支数。"""

    if config.get("_execution_authorized") is not True:
        raise PermissionError("R015 design execution requires the recorded authorization.")
    forbidden_effect = _forbidden_effect_field(config)
    if forbidden_effect is not None:
        raise ValueError(
            "R015 design input contains a forbidden effect field at "
            + forbidden_effect
        )
    protocol_path = Path(str(config.get("_protocol_path", ""))).resolve()
    output_dir = Path(str(config.get("_output_dir", ""))).resolve()
    if not protocol_path.is_file() or not str(config.get("_output_dir", "")):
        raise ValueError("R015 design execution requires protocol and output paths.")
    protocol = load_r015_design_protocol(protocol_path)
    if _require_mapping(
        protocol.get("execution_boundary"), field="execution_boundary"
    ).get("current_execution_authorized") is not False:
        raise ValueError("The canonical R015 design protocol authorization marker changed.")

    from experiments.overcooked_v2.official import r015_runtime_bridge
    from experiments.overcooked_v2.path_c_pool_admission import (
        R015PartnerSupportSpec,
        load_r015_partner_support_config,
    )
    from experiments.overcooked_v2.path_c_r015_controller import (
        NestedHiddenStateBranchScheduleV1,
        R015SequentialPlannerV1,
        default_r015_probe_scripts,
    )
    from experiments.overcooked_v2.path_c_r015_full_horizon import (
        R015PlanningPointBatchRequestV2,
    )

    support = _require_mapping(protocol.get("support"), field="support")
    support_path = Path(str(support["partner_support_registration"]))
    if not support_path.is_absolute():
        support_path = (protocol_path.parent / support_path).resolve()
    support_config = load_r015_partner_support_config(support_path)
    support_spec = R015PartnerSupportSpec.from_mapping(support_config)
    backend = r015_runtime_bridge.build_official_r015_production_backend(
        protocol,
        support_spec,
        config_base_path=protocol_path.parent,
    )

    histories_path = output_dir / "design_histories.jsonl"
    existing, histories_need_tail_repair = _load_recoverable_jsonl_prefix(
        histories_path
    )
    prototype_ids = tuple(str(item) for item in support["partner_prototype_ids"])
    expected_coordinates = [
        (prototype_id, episode_index)
        for episode_index in range(1, 21)
        for prototype_id in prototype_ids
    ]
    existing_coordinates = [
        (str(item.get("partner_prototype_id")), int(item.get("episode_index", -1)))
        for item in existing
    ]
    if existing_coordinates != expected_coordinates[: len(existing_coordinates)]:
        raise ValueError("R015 design resume records changed the frozen order.")
    if [int(item.get("design_sequence_index", -1)) for item in existing] != list(
        range(1, len(existing) + 1)
    ):
        raise ValueError("R015 design resume changed identity-free filter indices.")
    isolation = _require_mapping(protocol.get("role_isolation"), field="role_isolation")
    expected_history_inputs: list[Mapping[str, Any]] = []
    for sequence_offset, (prototype_id, episode_index) in enumerate(
        expected_coordinates,
        start=1,
    ):
        episode_seed = _design_seed(
            int(isolation["design_root_seed"]),
            str(isolation["design_episode_namespace"]),
            prototype_id,
            episode_index,
        )
        episode_id = canonical_sha256(
            ["r015_design_episode_v2", prototype_id, episode_index, episode_seed]
        )
        expected_history_inputs.append(
            {
                "partner_prototype_id": prototype_id,
                "episode_index": episode_index,
                "design_sequence_index": sequence_offset,
                "episode_seed": episode_seed,
                "design_episode_id": episode_id,
                "passive_history_root_key": canonical_sha256(
                    [
                        "r015_design_passive_history_v1",
                        episode_id,
                        episode_seed,
                    ]
                ),
            }
        )

    # 每个伙伴原型的 20 个回合构成一个固定车道批次。同一个已编译程序接收
    # 不同 checkpoint 参数，因此正常的完整收集只有一次编译和四次设备调用。
    # 批次文件先原子落盘；若进程在追加规范 JSONL 时中断，恢复直接复用该批次。
    design_source_sha256 = file_sha256(Path(__file__).resolve())
    bridge_source_path = Path(str(r015_runtime_bridge.__file__)).resolve()
    bridge_source_sha256 = file_sha256(bridge_source_path)
    batch_dir = output_dir / "design_history_batches"
    batch_payloads: list[Mapping[str, Any]] = []
    records_by_episode_id: dict[str, Mapping[str, Any]] = {}
    for prototype_position, prototype_id in enumerate(prototype_ids, start=1):
        batch_inputs = tuple(
            item
            for item in expected_history_inputs
            if item["partner_prototype_id"] == prototype_id
        )
        if len(batch_inputs) != 20:
            raise ValueError("R015 design-history partner batch is not 20 episodes.")
        batch_binding = {
            "execution_id": "r015_design_history_jit_scan_vmap_v1",
            "collection_action_rule": (
                "ego_seed100_official_categorical_no_probe_v1"
            ),
            "protocol_path": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "design_source_sha256": design_source_sha256,
            "runtime_bridge_source_sha256": bridge_source_sha256,
            "partner_prototype_id": prototype_id,
            "inputs": [dict(item) for item in batch_inputs],
        }
        batch_binding_sha256 = canonical_sha256(batch_binding)
        batch_path = batch_dir / f"partner_{prototype_position:02d}.json"
        if batch_path.is_file():
            batch = load_mapping(batch_path)
            if batch.get("schema_version") != (
                "path_c_r015_design_history_batch_v1"
            ) or batch.get("scientific_readout_allowed") is not False or (
                batch.get("batch_binding") != batch_binding
            ) or batch.get("batch_binding_sha256") != batch_binding_sha256:
                raise ValueError("R015 design-history batch changed its source or inputs.")
            batch_records = _require_sequence(
                batch.get("records"), field="design-history batch records"
            )
            if batch.get("records_sha256") != canonical_sha256(batch_records):
                raise ValueError("R015 design-history batch records changed after writing.")
        else:
            collected = backend.collect_design_history_batch(
                adapter=backend.full_horizon_executor.adapter,
                partner_prototype_id=prototype_id,
                episode_seeds=tuple(
                    int(item["episode_seed"]) for item in batch_inputs
                ),
                passive_history_root_keys=tuple(
                    str(item["passive_history_root_key"])
                    for item in batch_inputs
                ),
            )
            device_episodes = _require_sequence(
                collected.get("episodes"), field="device design histories"
            )
            if len(device_episodes) != len(batch_inputs):
                raise RuntimeError("R015 device history batch returned another width.")
            batch_records = [
                _design_history_record_from_device_episode(
                    prototype_id=str(item["partner_prototype_id"]),
                    episode_seed=int(item["episode_seed"]),
                    episode_index=int(item["episode_index"]),
                    design_sequence_index=int(item["design_sequence_index"]),
                    design_episode_id=str(item["design_episode_id"]),
                    episode=_require_mapping(
                        episode, field="device design-history episode"
                    ),
                )
                for item, episode in zip(batch_inputs, device_episodes)
            ]
            throughput = dict(
                _require_mapping(
                    collected.get("throughput"),
                    field="design-history device throughput",
                )
            )
            if throughput.get("execution_id") != (
                "r015_design_history_jit_scan_vmap_v1"
            ) or throughput.get("true_environment_transitions") != 8000 or (
                throughput.get("compiled_batch_calls") != 1
            ) or throughput.get("active_lane_batch_width") != 20 or (
                throughput.get("host_sync_inside_environment_loop") is not False
            ):
                raise RuntimeError("R015 design-history throughput accounting changed.")
            batch = {
                "schema_version": "path_c_r015_design_history_batch_v1",
                "scientific_readout_allowed": False,
                "batch_binding": batch_binding,
                "batch_binding_sha256": batch_binding_sha256,
                "records": batch_records,
                "records_sha256": canonical_sha256(batch_records),
                "throughput": throughput,
            }
            _write_json_atomic(batch_path, batch)
        if len(batch_records) != 20:
            raise ValueError("R015 design-history batch record count changed.")
        for record in batch_records:
            episode_id = str(record.get("design_episode_id", ""))
            if not episode_id or episode_id in records_by_episode_id:
                raise ValueError("R015 design-history batch duplicated an episode.")
            records_by_episode_id[episode_id] = record
        batch_payloads.append({**dict(batch), "batch_path": str(batch_path)})

    ordered_records = [
        records_by_episode_id[str(item["design_episode_id"])]
        for item in expected_history_inputs
    ]
    normalized_existing = [_jsonable(item) for item in existing]
    normalized_expected_prefix = [
        _jsonable(item) for item in ordered_records[: len(existing)]
    ]
    if normalized_existing != normalized_expected_prefix:
        raise ValueError("R015 design resume records differ from the device batch.")
    if histories_need_tail_repair:
        # 只有原子批次已经重建、且所有完整记录与登记前缀逐项一致后，才修复末尾。
        _rewrite_jsonl_atomic(histories_path, existing)
    for record in ordered_records[len(existing) :]:
        _append_jsonl_record(histories_path, record)
    histories = _load_complete_jsonl(histories_path)
    if len(histories) != 80:
        raise ValueError("R015 design history count is not 80.")
    if file_sha256(histories_path) == SUPERSEDED_V1_DESIGN_HISTORIES_SHA256:
        raise ValueError("R015 v2 design cannot reuse the superseded v1 histories.")

    batch_throughputs = [
        _require_mapping(item.get("throughput"), field="history batch throughput")
        for item in batch_payloads
    ]
    total_history_wall_seconds = sum(
        _finite(item.get("wall_seconds"), field="history batch wall seconds")
        for item in batch_throughputs
    )
    total_history_transitions = sum(
        _positive_int(
            item.get("true_environment_transitions"),
            field="history batch transitions",
        )
        for item in batch_throughputs
    )
    history_collection_report = {
        "schema_version": "path_c_r015_design_history_collection_report_v1",
        "scientific_readout_allowed": False,
        "execution_id": "r015_design_history_jit_scan_vmap_v1",
        "collection_action_rule": "ego_seed100_official_categorical_no_probe_v1",
        "design_episode_namespace": str(isolation["design_episode_namespace"]),
        "design_root_seed": int(isolation["design_root_seed"]),
        "canonical_episode_order_sha256": canonical_sha256(
            [dict(item) for item in expected_history_inputs]
        ),
        "history_count": len(histories),
        "environment_steps_per_history": 400,
        "design_histories_path": str(histories_path),
        "design_histories_sha256": file_sha256(histories_path),
        "batch_files": [
            {
                "path": str(item["batch_path"]),
                "sha256": file_sha256(str(item["batch_path"])),
                "partner_prototype_id": item["batch_binding"][
                    "partner_prototype_id"
                ],
                "episode_count": len(item["records"]),
            }
            for item in batch_payloads
        ],
        "throughput": {
            "true_environment_transitions": total_history_transitions,
            "compiled_batch_calls": sum(
                int(item["compiled_batch_calls"]) for item in batch_throughputs
            ),
            "jit_compilations": sum(
                int(item["jit_compilations"]) for item in batch_throughputs
            ),
            "active_lane_batch_widths": [
                int(item["active_lane_batch_width"])
                for item in batch_throughputs
            ],
            "wall_seconds": total_history_wall_seconds,
            "true_transitions_per_second": (
                total_history_transitions / total_history_wall_seconds
                if total_history_wall_seconds > 0.0
                else None
            ),
            "host_sync_inside_environment_loop": False,
        },
    }
    history_collection_report_path = output_dir / (
        "design_history_collection_report.json"
    )
    _write_json_atomic(
        history_collection_report_path, history_collection_report
    )
    stream_ids = tuple(str(item) for item in isolation["filter_repeat_namespaces"])
    opening_cache_binding = build_filter_opening_prefix_cache_binding(
        protocol_path=protocol_path,
        protocol=protocol,
        histories_path=histories_path,
        histories=histories,
        stream_ids=stream_ids,
    )
    opening_cache_binding_sha256 = filter_opening_prefix_cache_binding_sha256(
        opening_cache_binding
    )

    requested_filter_candidate = config.get("_filter_candidate_only")
    requested_candidate: tuple[int, str] | None = None
    if requested_filter_candidate is not None:
        if (
            not isinstance(requested_filter_candidate, Sequence)
            or isinstance(requested_filter_candidate, (str, bytes, bytearray))
            or len(requested_filter_candidate) != 2
        ):
            raise ValueError("R015 filter-only execution requires one registered candidate.")
        requested_candidate = (
            int(requested_filter_candidate[0]),
            str(requested_filter_candidate[1]),
        )
        if requested_candidate not in FILTER_CANDIDATE_ORDER:
            raise ValueError("R015 filter-only execution requested an unknown candidate.")

    def candidate_evidence_path(candidate: tuple[int, str]) -> Path:
        particles_per_prototype, timing = candidate
        return output_dir / f"filter_{particles_per_prototype}_{timing}.json"

    def read_candidate_evidence(
        candidate: tuple[int, str],
        evidence_path: Path,
    ) -> Mapping[str, Any]:
        existing_evidence = load_mapping(evidence_path)
        existing_result = evaluate_filter_candidate(
            existing_evidence,
            prototype_ids=prototype_ids,
        )
        validate_filter_opening_prefix_cache_binding(
            _require_mapping(
                existing_evidence.get("conditioned_opening_prefix_cache_binding"),
                field="existing filter opening-cache binding",
            ),
            expected=opening_cache_binding,
        )
        particles_per_prototype, timing = candidate
        if (
            int(existing_result["particles_per_prototype"])
            != particles_per_prototype
            or str(existing_result["resampling_timing"]) != timing
            or [
                str(item.get("design_episode_id"))
                for item in _require_sequence(
                    existing_evidence.get("episodes"),
                    field="existing filter episodes",
                )
            ]
            != [
                str(item["design_episode_id"])
                for item in histories[: int(existing_result["processed_episode_count"])]
            ]
        ):
            raise ValueError("R015 filter resume evidence changed its candidate or data.")
        return existing_result

    existing_filter_results: dict[tuple[int, str], Mapping[str, Any]] = {}
    missing_predecessor = False
    first_pass_seen = False
    for candidate in FILTER_CANDIDATE_ORDER:
        evidence_path = candidate_evidence_path(candidate)
        if not evidence_path.exists():
            missing_predecessor = True
            continue
        if missing_predecessor:
            raise ValueError(
                "R015 filter evidence is not a contiguous FILTER_CANDIDATE_ORDER prefix."
            )
        if first_pass_seen:
            raise ValueError(
                "R015 filter evidence exists after the first passing candidate."
            )
        result = read_candidate_evidence(candidate, evidence_path)
        existing_filter_results[candidate] = result
        first_pass_seen = bool(result["passed"])

    if requested_candidate is None:
        filter_candidate_order = FILTER_CANDIDATE_ORDER
    else:
        requested_index = FILTER_CANDIDATE_ORDER.index(requested_candidate)
        if requested_candidate not in existing_filter_results:
            if first_pass_seen:
                raise ValueError(
                    "R015 filter-only execution cannot continue after the first pass."
                )
            if requested_index != len(existing_filter_results):
                raise ValueError(
                    "R015 filter-only execution must request the next ascending candidate."
                )
        filter_candidate_order = (requested_candidate,)

    filter_paths: list[Path] = []
    filter_results: list[Mapping[str, Any]] = []
    for particles_per_prototype, timing in filter_candidate_order:
        candidate = (particles_per_prototype, timing)
        evidence_path = candidate_evidence_path(candidate)
        if evidence_path.exists():
            existing_result = existing_filter_results[candidate]
            filter_paths.append(evidence_path)
            filter_results.append(existing_result)
            if existing_result["passed"]:
                break
            continue
        grouped_pairs = [
            (history, stream_id)
            for history in histories
            for stream_id in stream_ids
        ]
        candidate_started = time.perf_counter()
        scanned = backend.full_horizon_executor.replay_filter_candidate_device(
            history_stream_pairs=grouped_pairs,
            particles_per_prototype=particles_per_prototype,
            resampling_timing=timing,
            return_particle_weight_trace=False,
            conditioned_opening_prefix_cache_binding=opening_cache_binding,
            stop_at_irreversible_s1_failure=True,
        )
        candidate_progress = _require_mapping(
            scanned.get("candidate_progress"),
            field="device filter candidate progress",
        )
        progress_status = str(candidate_progress.get("status", ""))
        processed_episode_count = _positive_int(
            candidate_progress.get("processed_episode_count"),
            field="device filter processed episode count",
        )
        zero_support_close_count = _nonnegative_int(
            candidate_progress.get("zero_support_close_count"),
            field="device filter zero-support close count",
        )
        if candidate_progress.get("total_episode_count") != (
            FILTER_DESIGN_EPISODE_COUNT
        ) or candidate_progress.get("processed_lane_count") != (
            processed_episode_count * FILTER_REPEAT_COUNT_PER_EPISODE
        ) or candidate_progress.get("stop_candidate_at_closed_episode_count") != (
            FILTER_S1_EARLY_STOP_CLOSED_EPISODE_COUNT
        ):
            raise RuntimeError("R015 device filter changed the irreversible S1 rule.")
        if progress_status == FILTER_CANDIDATE_S1_STOP_STATUS:
            if not s1_irreversible_failure(
                zero_support_close_count,
                total_episode_count=FILTER_DESIGN_EPISODE_COUNT,
                threshold=S1_MAXIMUM_CLOSE_RATE,
            ):
                raise RuntimeError("R015 device filter stopped before S1 was irreversible.")
        elif progress_status != FILTER_CANDIDATE_COMPLETE_STATUS:
            raise RuntimeError("R015 device filter returned an unknown progress status.")
        repeat_by_coordinate = {}
        for (history, stream_id), repeat in zip(
            grouped_pairs, scanned["lane_results"]
        ):
            if repeat["design_episode_id"] != history["design_episode_id"] or (
                repeat["filter_stream_id"] != stream_id
            ):
                raise RuntimeError("R015 device filter changed lane order.")
            repeat_by_coordinate[(history["design_episode_id"], stream_id)] = {
                "filter_stream_id": repeat["filter_stream_id"],
                "filter_initialization_key": repeat["filter_initialization_key"],
                "closed_for_zero_support": repeat["closed_for_zero_support"],
                "consultations": (
                    repeat["consultations"]
                    if progress_status == FILTER_CANDIDATE_COMPLETE_STATUS
                    else []
                ),
            }
        candidate_wall_seconds = time.perf_counter() - candidate_started
        throughput = dict(scanned["throughput"])
        device_wall_seconds = float(throughput["wall_seconds"])
        true_transitions = int(
            throughput["true_particle_environment_transitions"]
        )
        throughput["device_compile_and_execute_wall_seconds"] = device_wall_seconds
        throughput["wall_seconds"] = candidate_wall_seconds
        throughput["true_transitions_per_second"] = (
            true_transitions / candidate_wall_seconds
            if candidate_wall_seconds > 0.0
            else None
        )
        evidence_episodes = []
        for history in histories[:processed_episode_count]:
            repeats = [
                repeat_by_coordinate[(history["design_episode_id"], stream_id)]
                for stream_id in stream_ids
            ]
            closed_flags = [
                bool(repeat["closed_for_zero_support"]) for repeat in repeats
            ]
            evidence_episodes.append(
                {
                    "design_episode_id": history["design_episode_id"],
                    "partner_prototype_id": history["partner_prototype_id"],
                    "episode_seed": history["episode_seed"],
                    "design_sequence_index": history["design_sequence_index"],
                    "closed_for_zero_support": any(closed_flags),
                    "filter_repeats": repeats,
                }
            )
        evidence = {
            "schema_version": FILTER_EVIDENCE_SCHEMA,
            "scientific_readout_allowed": False,
            "particles_per_prototype": particles_per_prototype,
            "filter_algorithm_id": FILTER_ALGORITHM_ID,
            "filter_key_contract": FILTER_DEVICE_KEY_CONTRACT_ID,
            "resampling_algorithm": FILTER_RESAMPLING_ALGORITHM_ID,
            "resampling_timing": timing,
            "conditioned_opening_prefix_cache_binding": opening_cache_binding,
            "conditioned_opening_prefix_sha256_by_particle_count": dict(
                _require_mapping(
                    scanned.get(
                        "conditioned_opening_prefix_sha256_by_particle_count"
                    ),
                    field="device conditioned-opening prefix digests",
                )
            ),
            "conditioned_opening_attempt_sha256_by_particle_count": dict(
                _require_mapping(
                    scanned.get(
                        "conditioned_opening_attempt_sha256_by_particle_count"
                    ),
                    field="device conditioned-opening attempt digests",
                )
            ),
            "candidate_progress": dict(candidate_progress),
            "metric_computation": (
                {"s1": "computed", "s2": "computed", "s3": "computed"}
                if progress_status == FILTER_CANDIDATE_COMPLETE_STATUS
                else {
                    "s1": "failed_from_fixed_denominator_lower_bound",
                    "s2": "not_computed_due_to_irreversible_s1_failure",
                    "s3": "not_computed_due_to_irreversible_s1_failure",
                }
            ),
            "throughput": {
                "execution_id": FILTER_DEVICE_EXECUTION_ID,
                "conditioned_opening_prefix_cache_binding_sha256": (
                    opening_cache_binding_sha256
                ),
                **throughput,
            },
            "episodes": evidence_episodes,
        }
        candidate_result = evaluate_filter_candidate(
            evidence,
            prototype_ids=prototype_ids,
        )
        evidence = {**evidence, "candidate_result": dict(candidate_result)}
        _write_json_atomic(evidence_path, evidence)
        filter_paths.append(evidence_path)
        filter_results.append(candidate_result)
        if candidate_result["passed"]:
            for higher_candidate in FILTER_CANDIDATE_ORDER[
                FILTER_CANDIDATE_ORDER.index(candidate) + 1 :
            ]:
                if candidate_evidence_path(higher_candidate).exists():
                    raise ValueError(
                        "R015 filter evidence exists after the first passing candidate."
                    )
            break

    if requested_filter_candidate is not None:
        evidence_path = filter_paths[0]
        return {
            "schema_version": "path_c_r015_filter_candidate_execution_v2",
            "scientific_readout_allowed": False,
            "filter_evidence_path": str(evidence_path),
            "filter_evidence_sha256": file_sha256(evidence_path),
            "conditioned_opening_prefix_cache_binding_sha256": (
                opening_cache_binding_sha256
            ),
            "conditioned_opening_prefix_sha256_by_particle_count": dict(
                _require_mapping(
                    filter_results[0].get(
                        "conditioned_opening_prefix_sha256_by_particle_count"
                    ),
                    field="filter candidate conditioned-opening prefix digests",
                )
            ),
            "conditioned_opening_attempt_sha256_by_particle_count": dict(
                _require_mapping(
                    filter_results[0].get(
                        "conditioned_opening_attempt_sha256_by_particle_count"
                    ),
                    field="filter candidate conditioned-opening attempt digests",
                )
            ),
        }

    filter_selection = select_filter_candidate(filter_results)
    planning_paths: list[Path] = []
    if filter_selection["status"] == "selected":
        selected = filter_selection["selected_candidate"]
        particles = int(selected["particles_per_prototype"])
        timing = str(selected["resampling_timing"])
        runtime_source_paths = {
            "design_runner": Path(__file__).resolve(),
            "controller": Path(__file__).with_name(
                "path_c_r015_controller.py"
            ).resolve(),
            "full_horizon_executor": Path(__file__).with_name(
                "path_c_r015_full_horizon.py"
            ).resolve(),
            "runtime_bridge": Path(__file__).parent
            / "official"
            / "r015_runtime_bridge.py",
        }
        runtime_source_bindings = {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in runtime_source_paths.items()
        }
        selected_candidate_key = (particles, timing)
        selected_filter_evidence_path = candidate_evidence_path(
            selected_candidate_key
        )
        selected_filter_evidence = load_mapping(selected_filter_evidence_path)
        evaluate_filter_candidate(
            selected_filter_evidence,
            prototype_ids=prototype_ids,
        )
        planning_source_histories = _select_planning_histories(
            histories=histories,
            selected_filter_evidence=selected_filter_evidence,
            planning_filter_stream_id=PLANNING_FILTER_STREAM_ID,
        )
        (
            point_sources,
            selected_filter_checkpoint_manifest,
            selected_filter_checkpoint_manifest_path,
        ) = _load_or_build_selected_filter_checkpoints_v2(
            executor=backend.full_horizon_executor,
            selected_histories=planning_source_histories,
            particles_per_prototype=particles,
            resampling_timing=timing,
            selected_filter_evidence_path=selected_filter_evidence_path,
            runtime_source_bindings=runtime_source_bindings,
            conditioned_opening_prefix_cache_binding_sha256=(
                opening_cache_binding_sha256
            ),
            output_dir=output_dir,
        )
        selected_filter_checkpoint_manifest_binding = {
            "path": str(selected_filter_checkpoint_manifest_path.resolve()),
            "sha256": file_sha256(selected_filter_checkpoint_manifest_path),
            "point_count": int(
                selected_filter_checkpoint_manifest["point_count"]
            ),
        }
        released_opening_cache_lane_count = (
            backend.full_horizon_executor.release_conditioned_opening_prefix_cache(
                opening_cache_binding_sha256
            )
        )
        if not 0 <= released_opening_cache_lane_count <= (
            len(histories) * len(stream_ids)
        ):
            raise RuntimeError("R015 opening-prefix cache release changed the lane count.")
        verified_branch_cache: dict[str, tuple[Mapping[str, Any], ...]] = {}
        planning_results: list[Mapping[str, Any]] = []
        for branch_count in PLANNING_BRANCH_COUNTS:
            evidence_path = output_dir / (
                f"planning_r{branch_count}_vs_{2 * branch_count}.json"
            )
            if evidence_path.exists():
                existing_evidence = load_mapping(evidence_path)
                if existing_evidence.get("runtime_source_bindings") != (
                    runtime_source_bindings
                ):
                    raise ValueError(
                        "R015 planning resume changed its runtime source closure."
                    )
                candidate_result = evaluate_planning_candidate(existing_evidence)
                existing_points = _require_sequence(
                    existing_evidence.get("consultation_points"),
                    field="existing planning points",
                )
                if len(existing_points) != len(point_sources):
                    raise ValueError("R015 planning resume changed its point set.")
                for source, raw_point in zip(point_sources, existing_points):
                    _, _, environment_step, history, state = source
                    belief, _ = state[:2]
                    consultation_id = canonical_sha256(
                        [history["design_episode_id"], environment_step]
                    )
                    point = _require_mapping(raw_point, field="existing planning point")
                    if point.get("consultation_id") != consultation_id:
                        raise ValueError("R015 planning resume changed point order.")
                    if point.get("filter_checkpoint_sha256") != state[2]:
                        raise ValueError(
                            "R015 planning resume changed its filter checkpoint."
                        )
                    planning_key = _six_coordinate_random_key(
                        audit_unit_id=history["design_block_id"],
                        partner_prototype_id=history["partner_prototype_id"],
                        episode_seed=int(history["episode_seed"]),
                        purpose="value_planning",
                        environment_step=environment_step,
                    )
                    schedule = NestedHiddenStateBranchScheduleV1.build(
                        belief=belief,
                        planning_key=planning_key,
                        sample_count=2 * branch_count,
                    )
                    branches = tuple(
                        _require_sequence(
                            point.get("doubled_r_planning_branches"),
                            field="existing doubled planning branches",
                        )
                    )
                    expected_sources = {
                        sample.canonical_slot_16: (
                            sample.source_particle_index,
                            sample.source_particle.state_sha256,
                            sample.common_random_key,
                        )
                        for sample in schedule.samples
                    }
                    actual_sources = {
                        int(branch["canonical_slot_16"]): (
                            int(branch["source_particle_index"]),
                            str(branch["source_particle_state_sha256"]),
                            str(branch["common_random_key"]),
                        )
                        for branch in branches
                    }
                    if actual_sources != expected_sources:
                        raise ValueError(
                            "R015 planning resume changed its particle cloud or random key."
                        )
                    verified_branch_cache[consultation_id] = branches
                planning_paths.append(evidence_path)
                planning_results.append(candidate_result)
                if candidate_result["passed"]:
                    for higher in PLANNING_BRANCH_COUNTS[
                        PLANNING_BRANCH_COUNTS.index(branch_count) + 1 :
                    ]:
                        if (output_dir / f"planning_r{higher}_vs_{2 * higher}.json").exists():
                            raise ValueError(
                                "R015 planning contains evidence after the first pass."
                            )
                    break
                continue
            points = []
            candidate_started = time.perf_counter()
            batch_execution = _precompute_planning_length_buckets_v2(
                point_sources=point_sources,
                doubled_branch_count=2 * branch_count,
                verified_branch_cache=verified_branch_cache,
                executor=backend.full_horizon_executor,
                probe_scripts=default_r015_probe_scripts(),
                schedule_type=NestedHiddenStateBranchScheduleV1,
                request_type=R015PlanningPointBatchRequestV2,
            )
            candidate_compiled_batch_calls = int(
                batch_execution["compiled_batch_calls"]
            )
            candidate_active_batch_sizes = list(
                batch_execution["active_batch_sizes"]
            )
            for episode_index, prototype_id, environment_step, history, state in point_sources:
                belief, replay_history = state[:2]
                consultation_id = canonical_sha256(
                    [history["design_episode_id"], environment_step]
                )
                planning_key = _six_coordinate_random_key(
                    audit_unit_id=history["design_block_id"],
                    partner_prototype_id=prototype_id,
                    episode_seed=int(history["episode_seed"]),
                    purpose="value_planning",
                    environment_step=environment_step,
                )
                score_key = _six_coordinate_random_key(
                    audit_unit_id=history["design_block_id"],
                    partner_prototype_id=prototype_id,
                    episode_seed=int(history["episode_seed"]),
                    purpose="score_estimation",
                    environment_step=environment_step,
                )
                point_started = time.perf_counter()
                doubled_result = R015SequentialPlannerV1(
                    probe_scripts=default_r015_probe_scripts(),
                    branches_per_candidate=2 * branch_count,
                ).evaluate(
                    history=replay_history,
                    belief=belief,
                    environment_step=environment_step,
                    planning_key=planning_key,
                    score_key=score_key,
                    executor=backend.full_horizon_executor,
                    cached_planning_branches=verified_branch_cache.get(
                        consultation_id, ()
                    ),
                )
                r_result = R015SequentialPlannerV1(
                    probe_scripts=default_r015_probe_scripts(),
                    branches_per_candidate=branch_count,
                ).evaluate(
                    history=replay_history,
                    belief=belief,
                    environment_step=environment_step,
                    planning_key=planning_key,
                    score_key=score_key,
                    executor=backend.full_horizon_executor,
                    cached_planning_branches=doubled_result.planning_branches,
                )
                point_wall_seconds = time.perf_counter() - point_started
                verified_branch_cache[consultation_id] = tuple(
                    doubled_result.planning_branches
                )
                r_cost = dict(r_result.cost_accounting)
                doubled_cost = dict(doubled_result.cost_accounting)
                r_device = dict(r_result.device_execution)
                doubled_device = dict(doubled_result.device_execution)
                r_calls = int(r_device.get("compiled_batch_calls", 0))
                r_sizes = list(r_device.get("active_batch_sizes", ()))
                r_sync = r_device.get(
                    "host_sync_inside_environment_loop", False
                )
                if r_calls != 0 or r_sizes or r_sync is not False:
                    raise RuntimeError(
                        "The reconstructed R estimate performed new device work."
                    )
                point_calls = int(doubled_device.get("compiled_batch_calls", 0))
                point_sizes = tuple(doubled_device.get("active_batch_sizes", ()))
                point_sync = doubled_device.get(
                    "host_sync_inside_environment_loop", False
                )
                if point_calls != 0 or point_sizes or point_sync is not False:
                    raise RuntimeError(
                        "R015 planning attributed length-bucket device work to one point."
                    )
                points.append(
                    {
                        "consultation_id": consultation_id,
                        "design_block_id": history["design_block_id"],
                        "design_episode_id": history["design_episode_id"],
                        "partner_prototype_id": prototype_id,
                        "episode_index": episode_index,
                        "episode_seed": int(history["episode_seed"]),
                        "environment_step": environment_step,
                        "filter_checkpoint_sha256": state[2],
                        "planning_random_key": planning_key,
                        "score_random_key": score_key,
                        "r_decision": {
                            "masked_reference_action": r_result.masked_reference_action_id,
                            "probe_decision": r_result.selected_for_safety_probe_id or "no_probe",
                        },
                        "doubled_r_decision": {
                            "masked_reference_action": doubled_result.masked_reference_action_id,
                            "probe_decision": doubled_result.selected_for_safety_probe_id or "no_probe",
                        },
                        "r_planning_branches": list(r_result.planning_branches),
                        "doubled_r_planning_branches": list(
                            doubled_result.planning_branches
                        ),
                        "r_cost": r_cost,
                        "doubled_r_cost": doubled_cost,
                    }
                )
            candidate_wall_seconds = time.perf_counter() - candidate_started
            total_new_logical = sum(
                int(point["doubled_r_cost"]["new_real_environment_transitions"])
                for point in points
            )
            total_branch_head_particles = sum(
                int(point["doubled_r_cost"]["new_branch_head_particle_transitions"])
                for point in points
            )
            if total_new_logical != int(
                batch_execution["true_environment_transitions"]
            ) or total_branch_head_particles != int(
                batch_execution["branch_head_particle_transitions"]
            ):
                raise RuntimeError(
                    "R015 length-bucket execution changed the registered semantic cost."
                )
            evidence = {
                "schema_version": PLANNING_EVIDENCE_SCHEMA,
                "scientific_readout_allowed": False,
                "branch_count": branch_count,
                "doubled_branch_count": 2 * branch_count,
                "common_random_numbers": True,
                "full_remaining_episode": True,
                "registered_probe_count": 6,
                "prototype_count": 4,
                "consultation_point_sampling_rule_id": PLANNING_POINT_SAMPLING_RULE_ID,
                "planning_filter_stream_id": PLANNING_FILTER_STREAM_ID,
                "branch_sampling_rule_id": BRANCH_SAMPLING_RULE_ID,
                "branch_belief_rule_id": BRANCH_BELIEF_RULE_ID,
                "grid_evaluation_rule_id": GRID_EVALUATION_RULE_ID,
                "nested_branch_counts": list(NESTED_BRANCH_COUNTS),
                "runtime_source_bindings": runtime_source_bindings,
                "selected_filter_checkpoint_manifest": (
                    selected_filter_checkpoint_manifest_binding
                ),
                "total_particle_count": 4 * particles,
                "new_real_environment_transitions": total_new_logical,
                "computed_environment_transitions_including_padding": int(
                    batch_execution[
                        "computed_environment_transitions_including_padding"
                    ]
                ),
                "new_branch_head_particle_transitions": total_branch_head_particles,
                "compiled_batch_calls": candidate_compiled_batch_calls,
                "active_batch_sizes": candidate_active_batch_sizes,
                "length_bucket_device_execution": list(
                    batch_execution["length_bucket_reports"]
                ),
                "host_sync_inside_environment_loop": False,
                "wall_seconds": candidate_wall_seconds,
                "new_real_transition_throughput_per_second": (
                    total_new_logical / candidate_wall_seconds
                    if candidate_wall_seconds > 0.0
                    else None
                ),
                "consultation_points": points,
            }
            _write_json_atomic(evidence_path, evidence)
            planning_paths.append(evidence_path)
            candidate_result = evaluate_planning_candidate(evidence)
            planning_results.append(candidate_result)
            if candidate_result["passed"]:
                for higher in PLANNING_BRANCH_COUNTS[
                    PLANNING_BRANCH_COUNTS.index(branch_count) + 1 :
                ]:
                    if (output_dir / f"planning_r{higher}_vs_{2 * higher}.json").exists():
                        raise ValueError(
                            "R015 planning contains evidence after the first pass."
                        )
                break

    else:
        released_opening_cache_lane_count = (
            backend.full_horizon_executor.release_conditioned_opening_prefix_cache(
                opening_cache_binding_sha256
            )
        )
        if not 0 <= released_opening_cache_lane_count <= (
            len(histories) * len(stream_ids)
        ):
            raise RuntimeError("R015 opening-prefix cache release changed the lane count.")

    report = build_design_selection_report(
        protocol_path=protocol_path,
        filter_evidence_paths=filter_paths,
        planning_evidence_paths=planning_paths,
    )
    report_path = output_dir / "design_selection_report.json"
    _write_json_atomic(report_path, report)
    return {**report, "selection_report_path": str(report_path)}
