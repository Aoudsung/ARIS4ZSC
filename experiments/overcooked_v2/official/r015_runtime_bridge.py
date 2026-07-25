"""R015 运行侧桥接；训练清单绑定的官方适配器保持逐字节不变。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
    OFFICIAL_ACTION_ORDER,
    OCV2R015OfficialRolloutBackendV1,
    OvercookedV2ExperimentsNetworkAdapter,
    _LoadedPolicy,
    _load_candidate_policy,
)
from experiments.overcooked_v2.path_c_flax_policy import OfficialFlaxPolicy
from experiments.overcooked_v2.path_c_official_artifact import (
    FLAX_WEIGHTS_HASH_DOMAIN,
    checkpoint_artifact_sha256,
    flax_weights_sha256,
    validate_official_artifact_manifest,
)
from experiments.overcooked_v2.path_c_pool_admission import (
    R015CandidateSpec,
    R015PartnerSupportSpec,
)
from experiments.overcooked_v2.path_c_r015_controller import (
    ContinuationLibraryStatesV1,
    ContinuationMemberStepV1,
    FullHorizonBranchExecutor,
    FullHorizonRolloutV1,
    HiddenStateParticleV1,
    MAPPrototypeCommittedCookV1,
    OfficialHistoryV1,
    PairedProbeRolloutV1,
    PlanningBatchRolloutsV1,
    PlanningBranchSampleV1,
    ProbeScriptV1,
    R015_CONTINUATION_CONTROLLER_ID,
    R015_FUTURE_RANDOM_DERIVATION_ID,
    StratifiedParticleBeliefV1,
    derive_controller_key,
)
from experiments.overcooked_v2.path_c_seed import derive_ocv2_execution_seed


R015_FILTER_ALGORITHM_ID_V2 = (
    "official_history_fully_adapted_particle_filter_v2"
)
R015_FILTER_KEY_CONTRACT_V2 = "r015_filter_device_fold_in_keys_v2"
R015_FILTER_DEVICE_EXECUTION_ID_V2 = "r015_filter_jit_scan_vmap_v2"
R015_FILTER_RESET_PROPOSAL_MULTIPLIER = 256


def systematic_resample_indices_logarithmic(
    cumulative_weights: Any,
    positions: Any,
) -> Any:
    """按 ``count(cdf < position)`` 求系统重采样祖先，空间复杂度为 O(P)。

    ``cumulative_weights`` 和 ``positions`` 的最后一维都为每个原型的粒子槽。
    二分查找只保留上下界，不构造粒子数乘粒子数的比较矩阵。
    """

    import jax
    import jax.numpy as jnp

    cumulative = jnp.asarray(cumulative_weights)
    targets = jnp.asarray(positions)
    if cumulative.ndim < 1 or targets.shape != cumulative.shape:
        raise ValueError("R015 systematic resampling arrays must share one shape.")
    particle_count = cumulative.shape[-1]
    if particle_count <= 0:
        raise ValueError("R015 systematic resampling requires particles.")
    if particle_count == 1:
        return jnp.zeros_like(targets, dtype=jnp.int32)

    lower = jnp.zeros_like(targets, dtype=jnp.int32)
    upper = jnp.full_like(targets, particle_count - 1, dtype=jnp.int32)
    comparison_count = (particle_count - 1).bit_length()

    def narrow(_iteration: int, bounds: tuple[Any, Any]) -> tuple[Any, Any]:
        current_lower, current_upper = bounds
        midpoint = (current_lower + current_upper) // 2
        midpoint_value = jnp.take_along_axis(
            cumulative,
            midpoint,
            axis=-1,
        )
        move_right = midpoint_value < targets
        return (
            jnp.where(move_right, midpoint + 1, current_lower),
            jnp.where(move_right, current_upper, midpoint),
        )

    result, _ = jax.lax.fori_loop(
        0,
        comparison_count,
        narrow,
        (lower, upper),
    )
    return result


def _r015_sha256_words(value: str) -> tuple[int, ...]:
    """把完整 SHA-256 文本拆成八个设备 ``fold_in`` 坐标。"""

    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("R015 filter key must be one SHA-256 value.")
    try:
        return tuple(int(value[offset : offset + 8], 16) for offset in range(0, 64, 8))
    except ValueError as error:
        raise ValueError("R015 filter key must be one SHA-256 value.") from error


def _r015_device_key_from_words(words: Any) -> Any:
    """在设备上折入 SHA-256 的全部 256 位，不把键截成宿主 seed。"""

    import jax
    import jax.numpy as jnp

    values = jnp.asarray(words, dtype=jnp.uint32)
    key = jax.random.PRNGKey(values[0])
    for index in range(1, 8):
        key = jax.random.fold_in(key, values[index])
    return key


def _jax_key(value: Any) -> Any:
    """把登记的 SHA-256 随机键转换成官方策略接受的 JAX 键。"""

    if not isinstance(value, str) or len(value) != 64:
        return value
    try:
        seed = int(value[:16], 16) & 0xFFFFFFFF
    except ValueError as error:
        raise ValueError("R015 official policy key must be SHA-256.") from error
    import jax

    return jax.random.PRNGKey(seed)


@dataclass(frozen=True)
class _CompiledR015ActorAdapter:
    """缓存官方网络并编译逐步调用；不改变参数、动作规则或循环状态。"""

    stable_adapter: OvercookedV2ExperimentsNetworkAdapter
    _compiled_apply: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import jax
        import jax.numpy as jnp

        # 旧路径的粒子权重由 Python ``float`` 逐项更新，实际为二进制 64 位；环境、
        # 网络参数和动作计算仍保持 32 位。下方只在权重与 ESS 小段使用局部 64 位
        # 上下文，避免改变官方环境的整数类型或策略张量精度。

        network = self.stable_adapter._network()

        def apply_actor(
            params: Mapping[str, Any],
            recurrent_state: Any,
            observation: Any,
            episode_start: Any,
        ) -> tuple[Any, Any]:
            observation_array = jnp.asarray(observation)
            start_array = jnp.asarray(episode_start, dtype=jnp.bool_)
            single = observation_array.ndim == 3
            if single:
                observation_array = observation_array[jnp.newaxis, ...]
                start_array = start_array.reshape((1,))
            if observation_array.ndim != 4 or start_array.ndim != 1 or (
                observation_array.shape[0] != start_array.shape[0]
            ):
                raise ValueError(
                    "Official actor expects one batch of local observations."
                )
            next_state, distribution, _ = network.apply(
                params,
                recurrent_state,
                (
                    observation_array[jnp.newaxis, ...],
                    start_array[jnp.newaxis, ...],
                ),
            )
            logits = distribution.logits[0]
            if logits.shape[-1] != len(OFFICIAL_ACTION_ORDER):
                raise ValueError("Official actor must expose six logits.")
            if single:
                logits = logits[0]
            return next_state, logits

        object.__setattr__(self, "_compiled_apply", jax.jit(apply_actor))

    def initial_state(self, batch_size: int) -> Any:
        return self.stable_adapter.initial_state(batch_size)

    def apply_actor(
        self,
        params: Mapping[str, Any],
        recurrent_state: Any,
        observation: Any,
        episode_start: Any,
    ) -> tuple[Any, Any]:
        return self._compiled_apply(
            params,
            recurrent_state,
            observation,
            episode_start,
        )


def _load_r015_policy(candidate: R015CandidateSpec) -> _LoadedPolicy:
    """先走稳定适配器的来源核验，再只替换运行侧的编译调用。"""

    loaded = _load_candidate_policy(candidate)
    compiled_adapter = _CompiledR015ActorAdapter(loaded.adapter)
    compiled_policy = OfficialFlaxPolicy(
        params=loaded.policy.params,
        network=compiled_adapter,
        expected_model_weights_sha256=(
            loaded.policy.expected_model_weights_sha256
        ),
        action_rule=loaded.policy.action_rule,
        temperature=loaded.policy.temperature,
    )
    return _LoadedPolicy(
        policy=compiled_policy,
        adapter=loaded.adapter,
        config=loaded.config,
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_checkpoint_identity(
    candidate: R015CandidateSpec,
    loaded: _LoadedPolicy,
) -> Mapping[str, Any]:
    """从实际文件和已加载参数重建一个正式运行成员的来源身份。"""

    provenance = validate_official_artifact_manifest(
        candidate.training_config_path,
        candidate.training_manifest_path,
        expected_checkpoint_path=candidate.checkpoint_path,
        params=loaded.policy.params,
    )
    return {
        "artifact_id": candidate.candidate_id,
        "training_seed": int(candidate.training_seed),
        "path": str(Path(candidate.checkpoint_path).resolve()),
        "checkpoint_sha256": str(provenance["checkpoint_sha256"]),
        "model_weights_hash_domain": FLAX_WEIGHTS_HASH_DOMAIN,
        "model_weights_sha256": str(provenance["model_weights_sha256"]),
        "training_run_id": str(provenance["training_run_id"]),
        "training_manifest_path": str(
            Path(candidate.training_manifest_path).resolve()
        ),
        "training_manifest_sha256": str(provenance["training_manifest_sha256"]),
    }


def _resolve_r015_candidate_artifact_paths(
    raw_candidate: Mapping[str, Any],
    *,
    config_base_path: str | Path,
) -> dict[str, Any]:
    """按声明该候选的配置目录解析三项正式制品路径。"""

    base = Path(config_base_path).expanduser().resolve()
    candidate = dict(raw_candidate)
    for field in (
        "checkpoint_path",
        "training_config_path",
        "training_manifest_path",
    ):
        text = str(candidate.get(field, "")).strip()
        if not text:
            raise ValueError(f"R015 candidate lacks {field}.")
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = base / path
        candidate[field] = str(path.resolve())
    return candidate


def _immutable_jax_parameter_snapshot(
    params: Mapping[str, Any],
) -> tuple[Any, tuple[Any, ...]] | None:
    """保存不可变 JAX 参数叶节点的结构和对象引用，不复制设备数据。

    JAX 数组不能原地改写。只要树结构相同且每个叶节点仍是同一对象，启动时
    已计算的 Flax 权重摘要仍然成立；若树或任一叶节点被替换，调用方必须重新
    计算完整摘要。含 NumPy 等可变叶节点的树不使用这条快速路径。
    """

    import jax

    leaves, tree_structure = jax.tree_util.tree_flatten(params)
    if not leaves or any(not isinstance(leaf, jax.Array) for leaf in leaves):
        return None
    return tree_structure, tuple(leaves)


def _same_immutable_jax_parameter_snapshot(
    previous: tuple[Any, tuple[Any, ...]] | None,
    current: tuple[Any, tuple[Any, ...]] | None,
) -> bool:
    if previous is None or current is None or previous[0] != current[0]:
        return False
    return len(previous[1]) == len(current[1]) and all(
        old is new for old, new in zip(previous[1], current[1])
    )


@dataclass
class OCV2R015ProductionBackendV1:
    """把五个官方策略接到同一延续控制器和完整回合执行器。"""

    policies: Mapping[str, _LoadedPolicy]
    prototype_ids: tuple[str, ...]
    baseline_member_id: str
    full_horizon_executor: FullHorizonBranchExecutor | None = None
    checkpoint_identities: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict,
        repr=False,
    )
    _frozen_batch_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _filter_scan_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _online_execution_scan_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _design_history_scan_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _verified_parameter_snapshots: dict[
        str, tuple[Any, tuple[Any, ...]] | None
    ] = field(default_factory=dict, init=False, repr=False)
    _verified_parameter_hashes: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )
    _shared_actor_apply: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        expected_members = {*self.prototype_ids, self.baseline_member_id}
        if len(self.prototype_ids) != 4 or len(set(self.prototype_ids)) != 4:
            raise ValueError("R015 production backend requires four prototypes.")
        if set(self.policies) != expected_members:
            raise ValueError("R015 production policy library must contain five members.")
        self.continuation_controller = MAPPrototypeCommittedCookV1(
            prototype_ids=self.prototype_ids,
            baseline_member_id=self.baseline_member_id,
            member_actor=self._advance_continuation_member,
        )

    def verify_frozen_checkpoint_identities(
        self,
        freeze_manifest: Mapping[str, Any],
        *,
        refresh_files: bool,
    ) -> Mapping[str, Any]:
        """核对正式执行实际加载的五成员与冻结清单逐项相同。

        启动时 ``refresh_files`` 必须为真并重新计算 checkpoint 与训练清单摘要。
        每块独立重放前核对不可变 JAX 参数树仍由同一组叶节点组成；若结构或叶节点
        被替换，就重新计算完整 Flax 摘要。含可变叶节点的树每次都重算。磁盘文件
        即使随后改变，也不会替换本进程已经加载的参数。
        """

        if not isinstance(freeze_manifest, Mapping):
            raise TypeError("R015 frozen checkpoint identity requires one manifest.")
        raw_records = freeze_manifest.get("checkpoints")
        if not isinstance(raw_records, Sequence) or isinstance(
            raw_records, (str, bytes, bytearray)
        ) or len(raw_records) != 5:
            raise ValueError("R015 frozen manifest must bind exactly five checkpoints.")
        records: dict[str, Mapping[str, Any]] = {}
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise TypeError("R015 frozen checkpoint record must be a mapping.")
            artifact_id = str(raw_record.get("artifact_id", ""))
            if not artifact_id or artifact_id in records:
                raise ValueError("R015 frozen checkpoint artifact identities are invalid.")
            records[artifact_id] = raw_record
        expected_ids = set(self.policies)
        if set(self.checkpoint_identities) != expected_ids or set(records) != expected_ids:
            raise ValueError("R015 frozen manifest differs from the loaded five-member library.")

        checked: dict[str, Mapping[str, Any]] = {}
        verified_snapshots: dict[
            str, tuple[Any, tuple[Any, ...]] | None
        ] = {}
        verified_hashes: dict[str, str] = {}
        for artifact_id in sorted(expected_ids):
            identity = self.checkpoint_identities[artifact_id]
            record = records[artifact_id]
            checkpoint_path = Path(str(identity.get("path", ""))).resolve()
            manifest_path = Path(
                str(identity.get("training_manifest_path", ""))
            ).resolve()
            if refresh_files:
                actual_checkpoint_sha256 = checkpoint_artifact_sha256(
                    checkpoint_path
                )
                actual_manifest_sha256 = _file_sha256(manifest_path)
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                actual_training_run_id = str(
                    manifest_payload.get("training_run_id", "")
                )
            else:
                actual_checkpoint_sha256 = str(
                    identity.get("checkpoint_sha256", "")
                )
                actual_manifest_sha256 = str(
                    identity.get("training_manifest_sha256", "")
                )
                actual_training_run_id = str(identity.get("training_run_id", ""))
            params = self.policies[artifact_id].policy.params
            current_snapshot = _immutable_jax_parameter_snapshot(params)
            cached_snapshot = self._verified_parameter_snapshots.get(artifact_id)
            cached_hash = self._verified_parameter_hashes.get(artifact_id)
            if (
                not refresh_files
                and cached_hash is not None
                and _same_immutable_jax_parameter_snapshot(
                    cached_snapshot,
                    current_snapshot,
                )
            ):
                actual_weights_sha256 = cached_hash
            else:
                actual_weights_sha256 = flax_weights_sha256(params)
            actual = {
                "artifact_id": artifact_id,
                "training_seed": int(identity.get("training_seed", -1)),
                "path": str(checkpoint_path),
                "checkpoint_sha256": actual_checkpoint_sha256,
                "model_weights_hash_domain": FLAX_WEIGHTS_HASH_DOMAIN,
                "model_weights_sha256": actual_weights_sha256,
                "training_run_id": actual_training_run_id,
                "training_manifest_path": str(manifest_path),
                "training_manifest_sha256": actual_manifest_sha256,
            }
            if any(identity.get(field) != value for field, value in actual.items()):
                raise ValueError("R015 loaded checkpoint identity changed after verification.")
            for field, value in actual.items():
                if field in record and record.get(field) != value:
                    raise ValueError(
                        "R015 loaded checkpoint differs from its frozen identity."
                    )
            required_frozen_fields = {
                "artifact_id",
                "training_seed",
                "path",
                "checkpoint_sha256",
                "model_weights_hash_domain",
                "model_weights_sha256",
                "training_run_id",
                "training_manifest_path",
                "training_manifest_sha256",
            }
            if not required_frozen_fields.issubset(record):
                raise ValueError("R015 frozen checkpoint identity is incomplete.")
            checked[artifact_id] = actual
            verified_snapshots[artifact_id] = current_snapshot
            verified_hashes[artifact_id] = actual_weights_sha256
        # 只有五个成员全部通过后才更新快速核验状态，避免一次失败留下部分缓存。
        self._verified_parameter_snapshots = verified_snapshots
        self._verified_parameter_hashes = verified_hashes
        return checked

    def _compiled_actor_boundary(
        self,
        params: Mapping[str, Any],
        recurrent_state: Any,
        observation: Any,
    ) -> tuple[Any, Any]:
        """Use one non-inlined policy kernel in every R015 execution graph."""

        if self._shared_actor_apply is None:
            import jax
            import jax.numpy as jnp

            network = next(iter(self.policies.values())).adapter._network()

            def apply_actor(
                current_params: Mapping[str, Any],
                current_state: Any,
                current_observation: Any,
            ) -> tuple[Any, Any]:
                next_state, distribution, _ = network.apply(
                    current_params,
                    current_state,
                    (
                        current_observation[jnp.newaxis, ...],
                        jnp.zeros(
                            (1, current_observation.shape[0]),
                            dtype=jnp.bool_,
                        ),
                    ),
                )
                return next_state, distribution.logits[0]

            self._shared_actor_apply = jax.jit(
                apply_actor,
                inline=False,
            )
        return self._shared_actor_apply(
            params,
            recurrent_state,
            observation,
        )

    def _compiled_actor_boundary_with_start(
        self,
        member_id: str,
        recurrent_state: Any,
        observation: Any,
        episode_start: Any,
    ) -> tuple[Any, Any]:
        """在设备批次内推进一个官方成员，并保留回合开始标记。"""

        loaded = self.policies[member_id]
        return loaded.policy.network.apply_actor(
            loaded.policy.params,
            recurrent_state,
            observation,
            episode_start,
        )

    @property
    def continuation_controller_id(self) -> str:
        return R015_CONTINUATION_CONTROLLER_ID

    @property
    def admission_backend(self) -> OCV2R015OfficialRolloutBackendV1:
        """准入只使用四伙伴；主体 checkpoint 不得进入支持候选。"""

        return OCV2R015OfficialRolloutBackendV1(
            policies={prototype_id: self.policies[prototype_id] for prototype_id in self.prototype_ids}
        )

    def initial_continuation_states(self) -> ContinuationLibraryStatesV1:
        return ContinuationLibraryStatesV1(
            {
                member_id: self.policies[member_id].policy.initial_state(1)
                for member_id in self.continuation_controller.member_ids
            }
        )

    def initial_partner_state(self, prototype_id: str) -> Any:
        if prototype_id not in self.prototype_ids:
            raise ValueError("R015 requested an unregistered partner prototype.")
        return self.policies[prototype_id].policy.initial_state(1)

    def act_policy_member(
        self,
        member_id: str,
        observation: Any,
        recurrent_state: Any,
        random_key: Any,
        *,
        episode_start: bool,
    ) -> ContinuationMemberStepV1:
        if member_id not in self.policies:
            raise ValueError("R015 requested an unknown official policy member.")
        step = self.policies[member_id].policy.act(
            observation,
            bool(episode_start),
            recurrent_state,
            _jax_key(random_key),
        )
        action_array = np.asarray(step.action)
        if action_array.size != 1:
            raise ValueError("R015 official policy action must be scalar.")
        action_index = int(action_array.reshape(-1)[0])
        if not 0 <= action_index < len(OFFICIAL_ACTION_ORDER):
            raise ValueError("R015 official policy returned an unknown action.")
        return ContinuationMemberStepV1(
            action_id=OFFICIAL_ACTION_ORDER[action_index],
            next_recurrent_state=step.next_recurrent_state,
        )

    def collect_design_history_batch(
        self,
        *,
        adapter: Any,
        partner_prototype_id: str,
        episode_seeds: Sequence[int],
        passive_history_root_keys: Sequence[str],
    ) -> Mapping[str, Any]:
        """在一次设备扫描中收集同一伙伴原型的一批 400 步官方历史。

        该方法只替换原先逐环境步返回宿主的执行骨架。环境 reset、每步环境键、
        seed-100 主体与伙伴动作键都逐项复现旧路径；历史字段、回合顺序和随机数
        命名空间不变。设备只在完整批次结束后向宿主返回一次。
        """

        raw_seeds = tuple(episode_seeds)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in raw_seeds
        ):
            raise TypeError("R015 design-history episode seeds must be integers.")
        seeds = tuple(int(value) for value in raw_seeds)
        root_keys = tuple(str(value) for value in passive_history_root_keys)
        lane_count = len(seeds)
        if lane_count <= 0 or len(root_keys) != lane_count:
            raise ValueError("R015 design-history batch inputs have different lengths.")
        if partner_prototype_id not in self.prototype_ids:
            raise ValueError("R015 design history requested an unknown partner prototype.")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in root_keys
        ):
            raise ValueError("R015 design-history action roots must be SHA-256 values.")
        if self.full_horizon_executor is None or adapter is not (
            self.full_horizon_executor.adapter
        ):
            raise ValueError("R015 design histories require the production environment.")
        if int(adapter.max_steps) != 400:
            raise ValueError("R015 design histories require exactly 400 steps.")

        batch_started = time.perf_counter()
        import jax
        import jax.numpy as jnp

        baseline = self.policies[self.baseline_member_id]
        partner = self.policies[partner_prototype_id]
        network = baseline.adapter._network()

        # 旧收集器用 canonical SHA-256 键的前 64 位再截成 uint32。这里先在
        # 宿主构造相同 400 个键，只把固定张量送进一次编译调用；循环内部不再
        # 执行 Python 或设备到宿主同步。
        ego_action_seeds = np.asarray(
            [
                [
                    int(
                        derive_controller_key(root_key, "ego_action", step)[:16],
                        16,
                    )
                    & 0xFFFFFFFF
                    for step in range(400)
                ]
                for root_key in root_keys
            ],
            dtype=np.uint32,
        )
        partner_action_seeds = np.asarray(
            [
                [
                    int(
                        derive_controller_key(root_key, "partner_action", step)[
                            :16
                        ],
                        16,
                    )
                    & 0xFFFFFFFF
                    for step in range(400)
                ]
                for root_key in root_keys
            ],
            dtype=np.uint32,
        )
        # scan 以时间为第一维；每一列对应一条登记设计历史。
        action_key_inputs = (
            jnp.asarray(np.swapaxes(ego_action_seeds, 0, 1)),
            jnp.asarray(np.swapaxes(partner_action_seeds, 0, 1)),
        )
        execution_seeds = jnp.asarray(
            [derive_ocv2_execution_seed(value) for value in seeds],
            dtype=jnp.uint32,
        )
        initial_ego_state = baseline.policy.initial_state(lane_count)
        initial_partner_state = partner.policy.initial_state(lane_count)

        def apply_actor(
            params: Mapping[str, Any],
            recurrent_state: Any,
            observation: Any,
            episode_start: Any,
        ) -> tuple[Any, Any]:
            next_state, distribution, _ = network.apply(
                params,
                recurrent_state,
                (
                    observation[jnp.newaxis, ...],
                    episode_start[jnp.newaxis, ...],
                ),
            )
            logits = distribution.logits[0]
            if logits.shape[-1] != len(OFFICIAL_ACTION_ORDER):
                raise ValueError("Official actor must expose six logits.")
            return next_state, logits

        def compiled_batch(
            ego_params: Mapping[str, Any],
            partner_params: Mapping[str, Any],
            current_execution_seeds: Any,
            current_ego_state: Any,
            current_partner_state: Any,
            current_action_keys: Any,
        ) -> tuple[Any, Any]:
            reset_roots = jax.vmap(jax.random.PRNGKey)(current_execution_seeds)
            reset_pairs = jax.vmap(jax.random.split)(reset_roots)
            snapshot_keys = reset_pairs[:, 0]
            initial_observation, initial_environment_state = jax.vmap(
                adapter.env.reset
            )(reset_pairs[:, 1])

            def scan_step(carry: Any, step_input: Any):
                (
                    observation,
                    environment_state,
                    environment_keys,
                    ego_state,
                    partner_state,
                    step_index,
                ) = carry
                ego_seeds, partner_seeds = step_input
                ego_keys = jax.vmap(jax.random.PRNGKey)(ego_seeds)
                partner_keys = jax.vmap(jax.random.PRNGKey)(partner_seeds)
                environment_pairs = jax.vmap(jax.random.split)(environment_keys)
                episode_start = jnp.full(
                    (lane_count,), step_index == 0, dtype=jnp.bool_
                )
                next_ego_state, ego_logits = apply_actor(
                    ego_params,
                    ego_state,
                    observation["agent_1"],
                    episode_start,
                )
                next_partner_state, partner_logits = apply_actor(
                    partner_params,
                    partner_state,
                    observation["agent_0"],
                    episode_start,
                )
                ego_actions = jax.vmap(jax.random.categorical)(
                    ego_keys, ego_logits
                )
                partner_actions = jax.vmap(jax.random.categorical)(
                    partner_keys, partner_logits
                )
                next_observation, next_environment_state, rewards, dones, _ = (
                    jax.vmap(adapter.env.step_env)(
                        environment_pairs[:, 1],
                        environment_state,
                        {
                            "agent_0": partner_actions,
                            "agent_1": ego_actions,
                        },
                    )
                )
                trace = {
                    "official_local_observation": next_observation["agent_1"],
                    "ego_action_index": ego_actions,
                    "raw_team_reward_agent_0": rewards["agent_0"],
                    "raw_team_reward_agent_1": rewards["agent_1"],
                    "episode_boundary": dones["__all__"],
                }
                return (
                    next_observation,
                    next_environment_state,
                    environment_pairs[:, 0],
                    next_ego_state,
                    next_partner_state,
                    step_index + 1,
                ), trace

            final, trace = jax.lax.scan(
                scan_step,
                (
                    initial_observation,
                    initial_environment_state,
                    snapshot_keys,
                    current_ego_state,
                    current_partner_state,
                    jnp.asarray(0, dtype=jnp.int32),
                ),
                current_action_keys,
                length=400,
            )
            return initial_observation["agent_1"], trace

        observation_shape = tuple(
            int(value) for value in adapter.env.observation_space().shape
        )
        cache_key = (
            "design_history_scan_v1",
            lane_count,
            observation_shape,
        )
        compiled = self._design_history_scan_cache.get(cache_key)
        jit_compilations = 0
        if compiled is None:
            compiled = jax.jit(compiled_batch)
            self._design_history_scan_cache[cache_key] = compiled
            jit_compilations = 1
        device_started = time.perf_counter()
        initial_observation, trace = compiled(
            baseline.policy.params,
            partner.policy.params,
            execution_seeds,
            initial_ego_state,
            initial_partner_state,
            action_key_inputs,
        )
        initial_host, trace_host = jax.device_get((initial_observation, trace))
        device_wall_seconds = time.perf_counter() - device_started

        rewards_0 = np.asarray(trace_host["raw_team_reward_agent_0"])
        rewards_1 = np.asarray(trace_host["raw_team_reward_agent_1"])
        boundaries = np.asarray(trace_host["episode_boundary"], dtype=np.bool_)
        if not np.array_equal(rewards_0, rewards_1):
            raise ValueError("R015 design history lacks one shared team reward.")
        if boundaries.shape != (400, lane_count) or not bool(
            np.all(boundaries[-1])
        ):
            raise ValueError("R015 design histories did not complete 400 steps.")

        next_observations = np.asarray(
            trace_host["official_local_observation"]
        )
        ego_actions = np.asarray(trace_host["ego_action_index"])
        episodes = []
        for lane in range(lane_count):
            episodes.append(
                {
                    "initial_official_local_observation": np.asarray(
                        initial_host[lane]
                    ),
                    "official_local_observation_after_step": np.asarray(
                        next_observations[:, lane]
                    ),
                    "ego_action_indices": np.asarray(ego_actions[:, lane]),
                    "raw_team_rewards": np.asarray(rewards_0[:, lane]),
                    "episode_boundaries": np.asarray(boundaries[:, lane]),
                }
            )
        true_transitions = lane_count * 400
        wall_seconds = time.perf_counter() - batch_started
        return {
            "episodes": tuple(episodes),
            "throughput": {
                "execution_id": "r015_design_history_jit_scan_vmap_v1",
                "true_environment_transitions": true_transitions,
                "compiled_batch_calls": 1,
                "jit_compilations": jit_compilations,
                "active_lane_batch_width": lane_count,
                "environment_steps_per_lane": 400,
                "device_compile_and_execute_wall_seconds": device_wall_seconds,
                "wall_seconds": wall_seconds,
                "true_transitions_per_second": (
                    true_transitions / wall_seconds if wall_seconds > 0.0 else None
                ),
                "host_sync_inside_environment_loop": False,
            },
        }

    @staticmethod
    def stack_online_filter_states_v2(states: Sequence[Mapping[str, Any]]) -> Any:
        """把独立在线过滤状态堆成车道首维，不复制过滤语义。"""

        state_tuple = tuple(states)
        if not state_tuple:
            raise ValueError("R015 online filter stacking requires states.")
        import jax
        import jax.numpy as jnp

        structure = jax.tree_util.tree_structure(state_tuple[0])
        if any(jax.tree_util.tree_structure(state) != structure for state in state_tuple):
            raise ValueError("R015 online filter states have different structures.")
        leaves_by_state = tuple(
            jax.tree_util.tree_leaves(state) for state in state_tuple
        )
        if any(
            not hasattr(leaf, "ndim") or leaf.ndim < 1 or int(leaf.shape[0]) != 1
            for leaves in leaves_by_state
            for leaf in leaves
        ):
            raise ValueError(
                "Each R015 persistent online filter state must contain one lane."
            )
        return jax.tree_util.tree_map(
            lambda *items: jnp.concatenate(
                tuple(jnp.asarray(item) for item in items), axis=0
            ),
            *state_tuple,
        )

    @staticmethod
    def unstack_online_filter_states_v2(state: Mapping[str, Any]) -> tuple[Any, ...]:
        """在一个设备段结束后按原车道顺序拆回在线过滤状态。"""

        import jax

        leaves = jax.tree_util.tree_leaves(state)
        if not leaves or np.asarray(leaves[0]).ndim < 1:
            raise ValueError("R015 batched online filter state lacks a lane axis.")
        lane_count = int(np.asarray(leaves[0]).shape[0])
        if lane_count <= 0 or any(
            np.asarray(leaf).ndim < 1
            or int(np.asarray(leaf).shape[0]) != lane_count
            for leaf in leaves
        ):
            raise ValueError("R015 online filter state changed its lane axis.")
        return tuple(
            jax.tree_util.tree_map(
                lambda leaf: np.asarray(leaf)[lane : lane + 1], state
            )
            for lane in range(lane_count)
        )

    def run_online_trajectory_batch(
        self,
        *,
        adapter: Any,
        kernels: Sequence[Any],
        filter_state: Mapping[str, Any],
        branch_keys: Sequence[str],
        total_steps: int,
        key_offset: int,
        forced_first_actions: Sequence[str | None],
        resampling_timing: str,
        active_steps: Sequence[int] | None = None,
        explicit_step_keys: Sequence[Sequence[str]] | None = None,
        return_recurrent_state_trace: bool = True,
    ) -> Mapping[str, Any]:
        """在一个设备扫描中推进真实环境、在线过滤和五个延续成员。

        一个调用中的所有车道必须使用同一个真实伙伴原型。安全检查按原型分成
        四个批次，因此伙伴网络只计算一次，而不是让四个伙伴网络在每条车道上
        重复前向。``active_steps`` 用固定长度扫描的活动掩码承载末批短车道；
        环境步骤之间没有设备到宿主的同步。
        """

        lane_count = len(tuple(kernels))
        if lane_count <= 0 or not (
            lane_count == len(branch_keys) == len(forced_first_actions)
        ):
            raise ValueError("R015 online execution inputs have different lane counts.")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or (
            total_steps <= 0
        ):
            raise ValueError("R015 online execution requires a positive segment length.")
        if resampling_timing not in {
            "adaptive_ess_below_half_v1",
            "every_environment_step_v1",
        }:
            raise ValueError("R015 online execution uses an unregistered timing.")
        partner_ids = {kernel.partner_prototype_id for kernel in kernels}
        if len(partner_ids) != 1:
            raise ValueError(
                "One R015 online batch must contain one true partner prototype."
            )
        partner_id = next(iter(partner_ids))
        if partner_id not in self.prototype_ids:
            raise ValueError("R015 online execution contains an unknown partner.")
        if active_steps is None:
            active_lengths = (total_steps,) * lane_count
        else:
            active_lengths = tuple(int(value) for value in active_steps)
            if len(active_lengths) != lane_count or any(
                value <= 0 or value > total_steps for value in active_lengths
            ):
                raise ValueError("R015 online execution has invalid active lengths.")
        if explicit_step_keys is not None:
            normalized_explicit = tuple(tuple(row) for row in explicit_step_keys)
            if len(normalized_explicit) != lane_count or any(
                len(row) != total_steps for row in normalized_explicit
            ):
                raise ValueError("R015 explicit execution keys changed the batch shape.")
        else:
            normalized_explicit = tuple(
                tuple(
                    derive_controller_key(
                        branch_key,
                        "future_environment",
                        key_offset + offset,
                    )
                    for offset in range(total_steps)
                )
                for branch_key in branch_keys
            )
        if any(
            not isinstance(value, str) or len(value) != 64
            for row in normalized_explicit
            for value in row
        ):
            raise ValueError("R015 online execution step keys must be SHA-256 values.")

        required_filter_methods = (
            "online_filter_prototype_masses_v2",
            "_advance_online_filter_v2_device",
        )
        if any(not callable(getattr(self, name, None)) for name in required_filter_methods):
            raise RuntimeError(
                "R015 online execution requires the shared v2 filter device program."
            )

        import jax
        import jax.numpy as jnp
        from jaxmarl.environments.overcooked_v2.common import (
            Actions,
            DynamicObject,
            StaticObject,
        )

        member_ids = tuple(self.continuation_controller.member_ids)
        member_index = {member_id: index for index, member_id in enumerate(member_ids)}
        baseline_index = member_index[self.baseline_member_id]

        def concatenate_rows(values: Sequence[Any]) -> Any:
            return jax.tree_util.tree_map(
                lambda *items: jnp.concatenate(
                    tuple(jnp.asarray(item) for item in items), axis=0
                ),
                *values,
            )

        environment_state = jax.tree_util.tree_map(
            lambda *items: jnp.stack(tuple(jnp.asarray(item) for item in items)),
            *(kernel.snapshot.state for kernel in kernels),
        )
        observation = {
            agent_id: jnp.stack(
                tuple(
                    jnp.asarray(kernel.snapshot.raw_obs[agent_id])
                    for kernel in kernels
                )
            )
            for agent_id in ("agent_0", "agent_1")
        }
        partner_state = concatenate_rows(
            [kernel.partner_recurrent_state for kernel in kernels]
        )
        continuation_states = tuple(
            concatenate_rows(
                [
                    kernel.continuation_states.by_member_id[member_id]
                    for kernel in kernels
                ]
            )
            for member_id in member_ids
        )
        snapshot_keys = jnp.stack(
            tuple(jnp.asarray(kernel.snapshot.key) for kernel in kernels)
        )
        initial_environment_steps = jnp.asarray(
            [int(kernel.environment_step) for kernel in kernels], dtype=jnp.int32
        )
        filter_update_counts = np.asarray(filter_state["update_count"])
        expected_environment_steps = np.asarray(
            [int(kernel.environment_step) for kernel in kernels], dtype=np.int32
        )
        if filter_update_counts.shape != (lane_count,) or not np.array_equal(
            filter_update_counts.astype(np.int32), expected_environment_steps
        ):
            raise ValueError(
                "R015 true kernels and persistent filters are on different steps."
            )
        forced_indices = jnp.asarray(
            [
                -1 if value is None else OFFICIAL_ACTION_ORDER.index(value)
                for value in forced_first_actions
            ],
            dtype=jnp.int32,
        )
        active_lengths_array = jnp.asarray(active_lengths, dtype=jnp.int32)

        environment_keys = []
        next_snapshot_keys = []
        partner_keys = []
        member_keys = {member_id: [] for member_id in member_ids}
        for offset in range(total_steps):
            environment_column = []
            next_column = []
            partner_column = []
            member_column = {member_id: [] for member_id in member_ids}
            for lane in range(lane_count):
                step_key = normalized_explicit[lane][offset]
                next_key, environment_key = jax.random.split(_jax_key(step_key))
                environment_column.append(environment_key)
                next_column.append(next_key)
                partner_column.append(
                    _jax_key(derive_controller_key(step_key, "partner_action"))
                )
                for member_id in member_ids:
                    member_column[member_id].append(
                        _jax_key(
                            derive_controller_key(
                                step_key,
                                "continuation",
                                member_id,
                            )
                        )
                    )
            environment_keys.append(jnp.stack(environment_column))
            next_snapshot_keys.append(jnp.stack(next_column))
            partner_keys.append(jnp.stack(partner_column))
            for member_id in member_ids:
                member_keys[member_id].append(jnp.stack(member_column[member_id]))
        environment_key_array = jnp.stack(environment_keys)
        next_snapshot_key_array = jnp.stack(next_snapshot_keys)
        partner_key_array = jnp.stack(partner_keys)
        member_key_array = tuple(
            jnp.stack(member_keys[member_id]) for member_id in member_ids
        )

        def select_lanes(mask: Any, new: Any, old: Any) -> Any:
            def select_leaf(next_value: Any, current_value: Any) -> Any:
                next_array = jnp.asarray(next_value)
                current_array = jnp.asarray(current_value)
                shape = (mask.shape[0],) + (1,) * (next_array.ndim - 1)
                return jnp.where(mask.reshape(shape), next_array, current_array)

            return jax.tree_util.tree_map(select_leaf, new, old)

        def choose_member_rows(stacked: Any, indices: Any) -> Any:
            return jax.tree_util.tree_map(
                lambda value: jax.vmap(lambda row, index: value[index, row])(
                    jnp.arange(indices.shape[0]), indices
                ),
                stacked,
            )

        def routed_member_indices(current_filter_state: Any) -> Any:
            posterior = self.online_filter_prototype_masses_v2(
                current_filter_state
            )
            maxima = jnp.max(posterior, axis=-1, keepdims=True)
            tied = posterior == maxima
            unique = jnp.sum(tied, axis=-1) == 1
            prototype_choice = jnp.argmax(posterior, axis=-1)
            return jnp.where(unique, prototype_choice, baseline_index), posterior

        def wrong_delivery(
            current_state: Any,
            next_state: Any,
            current_partner_action: Any,
            current_ego_action: Any,
        ) -> Any:
            action_array = jnp.stack(
                (current_partner_action, current_ego_action), axis=0
            )
            forward = jax.vmap(lambda agent: agent.get_fwd_pos())(
                current_state.agents
            )
            cells = current_state.grid[forward.y, forward.x]
            deliveries = (
                (action_array == int(Actions.interact))
                & (cells[:, 0] == int(StaticObject.GOAL))
                & ((current_state.agents.inventory & int(DynamicObject.COOKED)) != 0)
            )
            return jnp.sum(deliveries) > next_state.new_correct_delivery

        def compiled_scan(
            initial_carry: Any,
            all_inputs: Any,
            current_forced_indices: Any,
            current_active_lengths: Any,
            current_initial_environment_steps: Any,
        ):
            def scan_step(carry: Any, scan_input: Any):
                (
                    env_state,
                    obs,
                    current_partner_state,
                    current_members,
                    current_filter_state,
                    current_snapshot_keys,
                    terminated,
                ) = carry
                (
                    step_index,
                    environment_key,
                    next_snapshot_key,
                    current_partner_key,
                    current_member_keys,
                ) = scan_input
                active = (step_index < current_active_lengths) & (~terminated)
                episode_start = (
                    current_initial_environment_steps + step_index
                ) == 0
                member_indices, posterior_before = routed_member_indices(
                    current_filter_state
                )
                next_partner_state, partner_logits = (
                    self._compiled_actor_boundary_with_start(
                        partner_id,
                        current_partner_state,
                        obs["agent_0"],
                        episode_start,
                    )
                )
                partner_actions = jax.vmap(jax.random.categorical)(
                    current_partner_key,
                    partner_logits,
                )
                next_members = []
                logits_by_member = []
                for member_id, member_state in zip(member_ids, current_members):
                    next_state, logits = self._compiled_actor_boundary_with_start(
                        member_id,
                        member_state,
                        obs["agent_1"],
                        episode_start,
                    )
                    next_members.append(next_state)
                    logits_by_member.append(logits)
                stacked_logits = jnp.stack(tuple(logits_by_member))
                selected_logits = choose_member_rows(stacked_logits, member_indices)
                stacked_member_keys = jnp.stack(current_member_keys)
                selected_member_keys = choose_member_rows(
                    stacked_member_keys,
                    member_indices,
                )
                ego_actions = jax.vmap(jax.random.categorical)(
                    selected_member_keys,
                    selected_logits,
                )
                ego_actions = jnp.where(
                    (step_index == 0) & (current_forced_indices >= 0),
                    current_forced_indices,
                    ego_actions,
                )
                actions = {"agent_0": partner_actions, "agent_1": ego_actions}
                next_obs, next_env_state, rewards, dones, _ = jax.vmap(
                    adapter.env.step_env
                )(environment_key, env_state, actions)
                official_step_records = {
                    "official_local_observation": next_obs["agent_1"],
                    "ego_action_index": ego_actions,
                    "raw_team_reward": rewards["agent_0"],
                    "episode_boundary": dones["__all__"],
                }
                advanced = self._advance_online_filter_v2_device(
                    adapter=adapter,
                    filter_state=current_filter_state,
                    official_step_records=official_step_records,
                    environment_step=(
                        current_initial_environment_steps + step_index + 1
                    ),
                    resampling_timing=resampling_timing,
                    return_diagnostics=True,
                )
                advanced_filter_state = advanced["filter_state"]
                posterior_after = advanced["posterior"]
                wrong = jax.vmap(wrong_delivery)(
                    env_state,
                    next_env_state,
                    partner_actions,
                    ego_actions,
                )
                selected_env_state = select_lanes(active, next_env_state, env_state)
                selected_obs = select_lanes(active, next_obs, obs)
                selected_partner_state = select_lanes(
                    active,
                    next_partner_state,
                    current_partner_state,
                )
                selected_members = tuple(
                    select_lanes(active, next_state, current_state)
                    for next_state, current_state in zip(
                        next_members,
                        current_members,
                    )
                )
                selected_filter_state = select_lanes(
                    active,
                    advanced_filter_state,
                    current_filter_state,
                )
                selected_snapshot_keys = select_lanes(
                    active,
                    next_snapshot_key,
                    current_snapshot_keys,
                )
                selected_done = active & dones["__all__"]
                selected_filter_closed = active & advanced["closed"]
                next_terminated = terminated | selected_done | selected_filter_closed
                trace = {
                    "active": active,
                    "partner_actions": jnp.where(active, partner_actions, -1),
                    "ego_actions": jnp.where(active, ego_actions, -1),
                    "raw_team_rewards": jnp.where(
                        active, rewards["agent_0"], 0.0
                    ),
                    "done": selected_done,
                    "wrong_delivery": active & wrong,
                    "agent_1_observation_before": obs["agent_1"],
                    "agent_1_observation_after": selected_obs["agent_1"],
                    "prototype_posterior_before": posterior_before,
                    "prototype_posterior_after": jnp.where(
                        active[:, None], posterior_after, posterior_before
                    ),
                    "selected_member_indices": member_indices,
                    "filter_closed": selected_filter_closed,
                    "filter_pre_resample_ess_fraction": advanced[
                        "pre_resample_ess_fraction"
                    ],
                    "filter_resampling_due": advanced["resampling_due"],
                    "filter_particle_environment_transitions": jnp.where(
                        active,
                        advanced.get(
                            "particle_environment_transition_count",
                            jnp.asarray(0, dtype=jnp.int32),
                        )
                        // active.shape[0],
                        0,
                    ),
                    "computed_filter_particle_environment_transitions": (
                        advanced.get(
                            "particle_environment_transition_count",
                            jnp.asarray(0, dtype=jnp.int32),
                        )
                    ),
                }
                if return_recurrent_state_trace:
                    # before 状态就是 scan carry：第一步来自输入，后续步骤等于
                    # 上一步 after。执行器由这条等式重建 before 摘要，不把同一
                    # 大型循环状态在设备输出中复制第二份。
                    trace.update(
                        {
                            "partner_state_after": selected_partner_state,
                            "continuation_states_after": selected_members,
                        }
                    )
                return (
                    selected_env_state,
                    selected_obs,
                    selected_partner_state,
                    selected_members,
                    selected_filter_state,
                    selected_snapshot_keys,
                    next_terminated,
                ), trace

            return jax.lax.scan(scan_step, initial_carry, all_inputs)

        scan_inputs = (
            jnp.arange(total_steps, dtype=jnp.int32),
            environment_key_array,
            next_snapshot_key_array,
            partner_key_array,
            member_key_array,
        )
        cache_key = (
            "online_execution_v2",
            lane_count,
            total_steps,
            partner_id,
            resampling_timing,
            bool(return_recurrent_state_trace),
        )
        compiled = self._online_execution_scan_cache.get(cache_key)
        compilation_count = 0
        if compiled is None:
            compiled = jax.jit(compiled_scan)
            self._online_execution_scan_cache[cache_key] = compiled
            compilation_count = 1
        started = time.perf_counter()
        final, trace = compiled(
            (
                environment_state,
                observation,
                partner_state,
                continuation_states,
                filter_state,
                snapshot_keys,
                jnp.zeros((lane_count,), dtype=jnp.bool_),
            ),
            scan_inputs,
            forced_indices,
            active_lengths_array,
            initial_environment_steps,
        )
        host_final, host_trace = jax.device_get((final, trace))
        wall_seconds = time.perf_counter() - started
        (
            final_environment_state,
            final_observation,
            final_partner_state,
            final_continuation_states,
            final_filter_state,
            final_snapshot_keys,
            final_terminated,
        ) = host_final
        active_transitions = int(np.sum(np.asarray(host_trace["active"])))
        computed_transitions = int(lane_count * total_steps)
        filter_particle_transitions = int(
            np.sum(
                np.asarray(
                    host_trace["filter_particle_environment_transitions"]
                )
            )
        )
        computed_filter_particle_transitions = int(
            np.sum(
                np.asarray(
                    host_trace[
                        "computed_filter_particle_environment_transitions"
                    ]
                )
            )
        )
        return {
            "final_environment_state": final_environment_state,
            "final_observation": final_observation,
            "final_partner_state": final_partner_state,
            "final_continuation_states": final_continuation_states,
            "final_filter_state": final_filter_state,
            "final_snapshot_keys": final_snapshot_keys,
            "terminated": final_terminated,
            "trace": host_trace,
            "step_keys": normalized_explicit,
            "throughput": {
                "true_environment_transitions": active_transitions,
                "computed_environment_transitions_including_masked_padding": (
                    computed_transitions
                ),
                "filter_particle_environment_transitions": (
                    filter_particle_transitions
                ),
                "computed_filter_particle_environment_transitions": (
                    computed_filter_particle_transitions
                ),
                "compiled_batch_calls": 1,
                "jit_compilations": compilation_count,
                "active_lane_batch_width": lane_count,
                "active_step_counts": active_lengths,
                "wall_seconds": wall_seconds,
                "true_transitions_per_second": (
                    active_transitions / wall_seconds
                    if wall_seconds > 0.0
                    else None
                ),
                "computed_transitions_per_second": (
                    computed_transitions / wall_seconds
                    if wall_seconds > 0.0
                    else None
                ),
                "host_sync_inside_environment_loop": False,
            },
        }

    def run_safety_trajectory_batch(
        self,
        *,
        adapter: Any,
        kernels: Sequence[Any],
        filter_state: Mapping[str, Any],
        branch_keys: Sequence[str],
        remaining_steps: int,
        probe_id: str,
        resampling_timing: str,
        repetitions: int = 279,
    ) -> Mapping[str, Any]:
        """运行一个支持原型的一批完整安全后缀。

        外层按 4096 个父粒子槽切分 279 次检查；本入口只接收一个微批，并在
        一次设备扫描中走完其全部剩余步骤。
        """

        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or not 1 <= repetitions <= 279
            or len(kernels) != repetitions
        ):
            raise ValueError(
                "One R015 safety microbatch must contain from 1 through 279 lanes."
            )
        if probe_id not in OFFICIAL_ACTION_ORDER:
            raise ValueError("R015 safety received an unknown probe action.")
        result = dict(
            self.run_online_trajectory_batch(
                adapter=adapter,
                kernels=kernels,
                filter_state=filter_state,
                branch_keys=branch_keys,
                total_steps=remaining_steps,
                key_offset=0,
                forced_first_actions=(probe_id,) * repetitions,
                resampling_timing=resampling_timing,
                active_steps=(remaining_steps,) * repetitions,
                return_recurrent_state_trace=False,
            )
        )
        result["batch_purpose"] = "safety_wrong_delivery_full_suffix_v1"
        result["prototype_id"] = kernels[0].partner_prototype_id
        result["repetitions"] = repetitions
        return result

    def _advance_continuation_member(
        self,
        member_id: str,
        history: OfficialHistoryV1,
        recurrent_state: Any,
        random_key: Any,
    ) -> ContinuationMemberStepV1:
        record = history.records[-1]
        raw_boundary = record.get("episode_boundaries", False)
        if isinstance(raw_boundary, Sequence) and not isinstance(
            raw_boundary, (str, bytes, bytearray)
        ):
            if not raw_boundary:
                raise ValueError("R015 official history lacks an episode boundary.")
            episode_start = bool(raw_boundary[-1])
        else:
            episode_start = bool(raw_boundary)
        return self.act_policy_member(
            member_id,
            record["official_local_observation"],
            recurrent_state,
            random_key,
            episode_start=episode_start,
        )

    def run_frozen_trajectory_batch(
        self,
        *,
        adapter: Any,
        kernels: Sequence[Any],
        branch_keys: Sequence[str],
        total_steps: int,
        key_offset: int,
        forced_first_actions: Sequence[str | None],
        committed_member_ids: Sequence[str],
        key_contract: str = "planning_future_v1",
        active_steps: Sequence[int] | None = None,
        explicit_step_keys: Sequence[Sequence[str]] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        """在设备内批量推进固定成员的完整规划后缀。

        ``active_steps`` 让不同剩余长度复用一个固定形状设备程序；超出活动长度
        的扫描位置不改变状态，也不进入返回轨迹。``explicit_step_keys`` 由规划层
        一次生成同一隐藏样本的登记未来随机流，六个探查头和十二个后缀只引用
        同一字节串，不再重复执行宿主 SHA-256 派生。
        """

        if not kernels or not (
            len(kernels)
            == len(branch_keys)
            == len(forced_first_actions)
            == len(committed_member_ids)
        ):
            raise ValueError("R015 batch planning inputs have different lengths.")
        if total_steps <= 0:
            raise ValueError("R015 batch planning requires at least one step.")
        if key_contract not in {"planning_future_v1", "filter_transition_v1"}:
            raise ValueError("R015 batch planning key contract is unregistered.")
        if key_contract == "filter_transition_v1" and total_steps != 1:
            raise ValueError("A filter transition batch must contain exactly one step.")
        lane_count = len(kernels)
        if active_steps is None:
            active_lengths = (total_steps,) * lane_count
        else:
            active_lengths = tuple(int(value) for value in active_steps)
            if len(active_lengths) != lane_count or any(
                value <= 0 or value > total_steps for value in active_lengths
            ):
                raise ValueError("R015 frozen planning has invalid active lengths.")
        if explicit_step_keys is None:
            normalized_step_keys = tuple(
                (
                    (str(branch_key),)
                    if key_contract == "filter_transition_v1"
                    else tuple(
                        derive_controller_key(
                            str(branch_key),
                            "future_environment",
                            key_offset + offset,
                        )
                        for offset in range(total_steps)
                    )
                )
                for branch_key in branch_keys
            )
        else:
            normalized_step_keys = tuple(
                tuple(str(value) for value in row) for row in explicit_step_keys
            )
            if len(normalized_step_keys) != lane_count or any(
                len(row) != total_steps for row in normalized_step_keys
            ):
                raise ValueError("R015 explicit planning keys changed batch shape.")
        if any(
            not isinstance(value, str) or len(value) != 64
            for row in normalized_step_keys
            for value in row
        ):
            raise ValueError("R015 frozen planning keys must be SHA-256 values.")
        if key_contract == "filter_transition_v1" and any(
            row != (str(branch_key),)
            for row, branch_key in zip(normalized_step_keys, branch_keys)
        ):
            raise ValueError("R015 filter transition changed its registered root key.")
        import jax
        import jax.numpy as jnp

        member_ids = tuple(self.continuation_controller.member_ids)
        member_index = {member_id: index for index, member_id in enumerate(member_ids)}
        prototype_index = {
            prototype_id: index for index, prototype_id in enumerate(self.prototype_ids)
        }
        if any(value not in member_index for value in committed_member_ids):
            raise ValueError("R015 batch planning committed an unknown member.")
        if any(kernel.partner_prototype_id not in prototype_index for kernel in kernels):
            raise ValueError("R015 batch planning contains an unknown partner.")

        # 规划样本的规范顺序由抽样合同决定，返回证据也必须保持该顺序；但设备
        # 程序内部没有跨车道运算，因此可先按伙伴原型稳定排列，结束后再恢复。
        # 这样，同一批宽下只按四个原型各有多少车道区分编译程序，不再为每一种
        # 原型排列重新编译。每条车道携带的环境状态、策略状态、动作和随机键都
        # 一起移动，所以抽样、随机键与轨迹逐字节不变。
        original_partner_index_values = tuple(
            prototype_index[kernel.partner_prototype_id] for kernel in kernels
        )
        execution_order = tuple(
            sorted(
                range(lane_count),
                key=lambda row: (original_partner_index_values[row], row),
            )
        )
        original_lane_by_execution_lane = execution_order

        def reorder_lanes(values: Sequence[Any]) -> tuple[Any, ...]:
            return tuple(values[row] for row in execution_order)

        kernels = reorder_lanes(kernels)
        branch_keys = reorder_lanes(branch_keys)
        forced_first_actions = reorder_lanes(forced_first_actions)
        committed_member_ids = reorder_lanes(committed_member_ids)
        active_lengths = reorder_lanes(active_lengths)
        normalized_step_keys = reorder_lanes(normalized_step_keys)

        continuation_params = tuple(
            self.policies[member_id].policy.params for member_id in member_ids
        )
        partner_params = tuple(
            self.policies[prototype_id].policy.params
            for prototype_id in self.prototype_ids
        )

        def concatenate_rows(values: Sequence[Any]) -> Any:
            return jax.tree_util.tree_map(
                lambda *items: jnp.concatenate(
                    tuple(jnp.asarray(item) for item in items), axis=0
                ),
                *values,
            )

        environment_state = jax.tree_util.tree_map(
            lambda *items: jnp.stack(tuple(jnp.asarray(item) for item in items)),
            *(kernel.snapshot.state for kernel in kernels),
        )
        observation = {
            agent_id: jnp.stack(
                tuple(
                    jnp.asarray(kernel.snapshot.raw_obs[agent_id])
                    for kernel in kernels
                )
            )
            for agent_id in ("agent_0", "agent_1")
        }
        partner_state = concatenate_rows(
            [kernel.partner_recurrent_state for kernel in kernels]
        )
        continuation_states = tuple(
            concatenate_rows(
                [
                    kernel.continuation_states.by_member_id[member_id]
                    for kernel in kernels
                ]
            )
            for member_id in member_ids
        )
        partner_index_values = tuple(
            prototype_index[kernel.partner_prototype_id] for kernel in kernels
        )
        partner_lane_counts = tuple(
            partner_index_values.count(prototype_id)
            for prototype_id in range(len(self.prototype_ids))
        )
        partner_rows_by_prototype = tuple(
            np.asarray(
                [
                    row
                    for row, selected_index in enumerate(partner_index_values)
                    if selected_index == prototype_id
                ],
                dtype=np.int32,
            )
            for prototype_id in range(len(self.prototype_ids))
        )
        assigned_partner_rows = sorted(
            int(row)
            for rows in partner_rows_by_prototype
            for row in np.asarray(rows).tolist()
        )
        if assigned_partner_rows != list(range(lane_count)):
            raise RuntimeError(
                "R015 partner prototype groups must partition every planning lane once."
            )
        committed_indices = jnp.asarray(
            [member_index[value] for value in committed_member_ids], dtype=jnp.int32
        )
        forced_indices = jnp.asarray(
            [
                -1 if value is None else OFFICIAL_ACTION_ORDER.index(value)
                for value in forced_first_actions
            ],
            dtype=jnp.int32,
        )

        # 同一隐藏样本的根键在六个探查头和十二个后缀车道重复出现。先按
        # 实际 step-key 字节去重，再由一个 inverse-index 张量恢复规范车道顺序。
        # 这只消除重复宿主 SHA-256；每条车道实际使用的随机键逐字节不变。
        unique_step_keys: list[str] = []
        unique_index_by_key: dict[str, int] = {}
        inverse_indices = np.empty((total_steps, lane_count), dtype=np.int32)
        for lane, row in enumerate(normalized_step_keys):
            for offset, step_key in enumerate(row):
                index = unique_index_by_key.get(step_key)
                if index is None:
                    index = len(unique_step_keys)
                    unique_index_by_key[step_key] = index
                    unique_step_keys.append(step_key)
                inverse_indices[offset, lane] = index

        unique_environment_keys = []
        unique_snapshot_next_keys = []
        unique_partner_action_keys = []
        unique_continuation_action_keys = {
            member_id: [] for member_id in member_ids
        }
        continuation_key_purpose = (
            "continuation_member"
            if key_contract == "filter_transition_v1"
            else "continuation"
        )
        for step_key in unique_step_keys:
            environment_root_key = (
                derive_controller_key(step_key, "particle_environment")
                if key_contract == "filter_transition_v1"
                else step_key
            )
            next_key, environment_key = jax.random.split(
                _jax_key(environment_root_key)
            )
            unique_environment_keys.append(environment_key)
            unique_snapshot_next_keys.append(next_key)
            unique_partner_action_keys.append(
                _jax_key(derive_controller_key(step_key, "partner_action"))
            )
            for member_id in member_ids:
                unique_continuation_action_keys[member_id].append(
                    _jax_key(
                        derive_controller_key(
                            step_key,
                            continuation_key_purpose,
                            member_id,
                        )
                    )
                )
        inverse = jnp.asarray(inverse_indices, dtype=jnp.int32)
        environment_keys_array = jnp.stack(unique_environment_keys)[inverse]
        next_keys_array = jnp.stack(unique_snapshot_next_keys)[inverse]
        partner_keys_array = jnp.stack(unique_partner_action_keys)[inverse]
        continuation_keys_array = tuple(
            jnp.stack(unique_continuation_action_keys[member_id])[inverse]
            for member_id in member_ids
        )
        step_indices = jnp.arange(total_steps, dtype=jnp.int32)
        active_lengths_array = jnp.asarray(active_lengths, dtype=jnp.int32)

        def actor_apply(params: Mapping[str, Any], state: Any, obs: Any):
            """Apply one official policy to every active planning slot at once."""

            return self._compiled_actor_boundary(
                params,
                state,
                obs,
            )

        def choose_rows(stacked: Any, indices: Any) -> Any:
            return jax.tree_util.tree_map(
                lambda value: jax.vmap(lambda row, index: value[index, row])(
                    jnp.arange(indices.shape[0]), indices
                ),
                stacked,
            )

        def compiled_scan(
            initial_carry: Any,
            all_scan_inputs: Any,
            current_committed_indices: Any,
            current_forced_indices: Any,
            current_active_lengths: Any,
        ):
            def scan_step(carry: Any, scan_input: Any):
                env_state, obs, current_partner_state, current_members = carry
                (
                    step_index,
                    environment_key,
                    partner_key,
                    member_keys,
                ) = scan_input
                # Every planning lane contains exactly one realized partner
                # prototype.  Run only that prototype's actor on its static row
                # subset, then scatter back to canonical lane order.  The five
                # continuation members below still all advance in parallel.
                next_partner_state = current_partner_state
                partner_logits = jnp.zeros(
                    (lane_count, len(OFFICIAL_ACTION_ORDER)), dtype=jnp.float32
                )
                for params, host_rows in zip(
                    partner_params, partner_rows_by_prototype
                ):
                    if len(host_rows) == 0:
                        continue
                    rows = jnp.asarray(host_rows, dtype=jnp.int32)
                    selected_state = jax.tree_util.tree_map(
                        lambda value: value[rows], current_partner_state
                    )
                    selected_observation = obs["agent_0"][rows]
                    selected_next_state, selected_logits = actor_apply(
                        params,
                        selected_state,
                        selected_observation,
                    )
                    next_partner_state = jax.tree_util.tree_map(
                        lambda complete, selected: complete.at[rows].set(selected),
                        next_partner_state,
                        selected_next_state,
                    )
                    partner_logits = partner_logits.at[rows].set(selected_logits)
                partner_actions = jax.vmap(jax.random.categorical)(
                    partner_key, partner_logits
                )
                next_members = []
                member_logits = []
                for params, state in zip(continuation_params, current_members):
                    next_state, logits = actor_apply(params, state, obs["agent_1"])
                    next_members.append(next_state)
                    member_logits.append(logits)
                member_logits_stack = jnp.stack(tuple(member_logits))
                committed_logits = choose_rows(
                    member_logits_stack, current_committed_indices
                )
                continuation_key_stack = jnp.stack(member_keys)
                committed_keys = choose_rows(
                    continuation_key_stack, current_committed_indices
                )
                ego_actions = jax.vmap(jax.random.categorical)(
                    committed_keys, committed_logits
                )
                ego_actions = jnp.where(
                    (step_index == 0) & (current_forced_indices >= 0),
                    current_forced_indices,
                    ego_actions,
                )
                actions = {"agent_0": partner_actions, "agent_1": ego_actions}
                next_obs, next_env_state, rewards, dones, _ = jax.vmap(
                    adapter.env.step_env
                )(environment_key, env_state, actions)
                candidate_carry = (
                    next_env_state,
                    next_obs,
                    next_partner_state,
                    tuple(next_members),
                )
                active = step_index < current_active_lengths

                def keep_inactive(candidate: Any, current: Any) -> Any:
                    return jax.tree_util.tree_map(
                        lambda next_value, current_value: jnp.where(
                            active.reshape(
                                (active.shape[0],)
                                + (1,) * (jnp.asarray(next_value).ndim - 1)
                            ),
                            next_value,
                            current_value,
                        ),
                        candidate,
                        current,
                    )

                return keep_inactive(candidate_carry, carry), {
                    "partner_actions": partner_actions,
                    "ego_actions": ego_actions,
                    "raw_team_rewards": rewards["agent_0"],
                    "done": dones["__all__"],
                    "agent_0_observation": next_obs["agent_0"],
                    "agent_1_observation": next_obs["agent_1"],
                }

            return jax.lax.scan(scan_step, initial_carry, all_scan_inputs)

        scan_inputs = (
            step_indices,
            environment_keys_array,
            partner_keys_array,
            tuple(continuation_keys_array),
        )
        cache_key = (
            len(kernels),
            total_steps,
            key_contract,
            partner_lane_counts,
        )
        compiled = self._frozen_batch_cache.get(cache_key)
        if compiled is None:
            compiled = jax.jit(compiled_scan)
            self._frozen_batch_cache[cache_key] = compiled
        final, trace = compiled(
            (environment_state, observation, partner_state, continuation_states),
            scan_inputs,
            committed_indices,
            forced_indices,
            active_lengths_array,
        )
        final_env_state, final_obs, final_partner_state, final_members = final
        final_key_indices = active_lengths_array - 1
        final_keys = next_keys_array[
            final_key_indices,
            jnp.arange(lane_count, dtype=jnp.int32),
        ]
        host = jax.device_get(
            (
                final_env_state,
                final_obs,
                final_partner_state,
                final_members,
                trace,
                final_keys,
            )
        )
        (
            final_env_state_host,
            final_obs_host,
            final_partner_state_host,
            final_members_host,
            trace_host,
            final_keys_host,
        ) = host
        unique_branch_root_count = len(set(str(value) for value in branch_keys))
        future_random_key_derivation = {
            "unique_root_count": unique_branch_root_count,
            "requested_lane_count": lane_count,
            "derived_step_key_count": len(unique_step_keys),
            "requested_step_key_slot_count": lane_count * total_steps,
            "reused_lane_count": lane_count - unique_branch_root_count,
            "reused_step_key_slot_count": (
                lane_count * total_steps - len(unique_step_keys)
            ),
            "derivation_contract_id": R015_FUTURE_RANDOM_DERIVATION_ID,
        }
        results: list[Mapping[str, Any] | None] = [None] * lane_count
        for row, kernel in enumerate(kernels):
            active_length = int(active_lengths[row])
            slice_row = lambda value: jax.tree_util.tree_map(
                lambda item: np.asarray(item[row : row + 1]), value
            )
            original_lane = original_lane_by_execution_lane[row]
            results[original_lane] = {
                "environment_state": jax.tree_util.tree_map(
                    lambda item: np.asarray(item[row]), final_env_state_host
                ),
                "raw_observation": {
                    agent_id: np.asarray(final_obs_host[agent_id][row])
                    for agent_id in ("agent_0", "agent_1")
                },
                "snapshot_key": np.asarray(final_keys_host[row]),
                "partner_recurrent_state": slice_row(final_partner_state_host),
                "continuation_states": {
                    member_id: slice_row(final_members_host[index])
                    for index, member_id in enumerate(member_ids)
                },
                "raw_team_rewards": np.asarray(
                    trace_host["raw_team_rewards"][:active_length, row]
                ),
                "partner_actions": np.asarray(
                    trace_host["partner_actions"][:active_length, row]
                ),
                "ego_actions": np.asarray(
                    trace_host["ego_actions"][:active_length, row]
                ),
                "done": np.asarray(trace_host["done"][:active_length, row]),
                "agent_1_observation_sequence": np.asarray(
                    trace_host["agent_1_observation"][:active_length, row]
                ),
                "future_random_keys": tuple(
                    normalized_step_keys[row][:active_length]
                ),
                "future_random_key_derivation": dict(future_random_key_derivation),
            }
        if any(result is None for result in results):
            raise RuntimeError("R015 planning did not restore every canonical lane.")
        return [result for result in results if result is not None]

    @staticmethod
    def _replace_filter_recipe_v2(environment_state: Any, recipe: Any) -> Any:
        """只替换登记环境状态中的配方；其余动态状态保持不变。"""

        if hasattr(environment_state, "replace"):
            return environment_state.replace(recipe=recipe)
        if hasattr(environment_state, "_replace"):
            return environment_state._replace(recipe=recipe)
        raise TypeError(
            "Official OvercookedV2 state does not expose a recipe replacement interface."
        )

    def _initialize_online_filter_v2_device(
        self,
        *,
        adapter: Any,
        opening_observations: Any,
        root_key_words: Any,
        particles_per_prototype: int,
        initial_partner_states: Sequence[Any],
        cached_prefix_state: Mapping[str, Any] | None = None,
        cached_particles_per_prototype: int = 0,
    ) -> Mapping[str, Any]:
        """在设备上条件化开局，按原始提议顺序保留前 P 个相容状态。"""

        import jax
        import jax.numpy as jnp

        lane_count = int(opening_observations.shape[0])
        prototype_count = len(self.prototype_ids)
        particle_count = int(particles_per_prototype)
        if prototype_count != 4:
            raise ValueError("R015 v2 filter requires four prototypes.")
        if particle_count <= 0:
            raise ValueError("R015 v2 filter requires positive P.")
        cached_count = int(cached_particles_per_prototype)
        if cached_count < 0 or cached_count > particle_count:
            raise ValueError("R015 cached opening prefix has an invalid length.")
        if (cached_prefix_state is None) != (cached_count == 0):
            raise ValueError("R015 cached opening prefix state/count disagree.")
        if cached_count == particle_count:
            reused = dict(cached_prefix_state)
            reused["root_key_words"] = root_key_words
            reused["update_count"] = jnp.zeros((lane_count,), dtype=jnp.int32)
            return reused

        roots = jax.vmap(_r015_device_key_from_words)(root_key_words)
        local_slots = jnp.arange(particle_count, dtype=jnp.uint32)
        prototype_slots = jnp.arange(prototype_count, dtype=jnp.uint32)
        if cached_count > 0:
            proposal_starts = jnp.asarray(
                cached_prefix_state["opening_proposal_counts"],
                dtype=jnp.uint32,
            )
        else:
            proposal_starts = jnp.zeros(
                (lane_count, prototype_count),
                dtype=jnp.uint32,
            )

        def proposal_key(root: Any, prototype_index: Any, proposal_index: Any) -> Any:
            key = jax.random.fold_in(root, jnp.uint32(0x01500101))
            key = jax.random.fold_in(key, prototype_index)
            return jax.random.fold_in(key, proposal_index)

        def proposal_batch(batch_index: Any) -> tuple[Any, Any, Any, Any]:
            proposal_indices = (
                proposal_starts[..., None]
                + jnp.asarray(batch_index, dtype=jnp.uint32)
                * jnp.uint32(particle_count)
                + local_slots.reshape((1, 1, particle_count))
            )

            def lane_keys(root: Any, lane_proposals: Any) -> Any:
                return jax.vmap(
                    lambda prototype_index, prototype_proposals: jax.vmap(
                        lambda proposal_index: proposal_key(
                            root,
                            prototype_index,
                            proposal_index,
                        )
                    )(prototype_proposals)
                )(prototype_slots, lane_proposals)

            keys = jax.vmap(lane_keys)(roots, proposal_indices)
            flat_keys = keys.reshape((-1, 2))
            split_keys = jax.vmap(jax.random.split)(flat_keys)
            observation, environment_state = jax.vmap(adapter.env.reset)(
                split_keys[:, 1]
            )

            def to_grid(value: Any) -> Any:
                return jax.tree_util.tree_map(
                    lambda item: item.reshape(
                        (lane_count, prototype_count, particle_count)
                        + item.shape[1:]
                    ),
                    value,
                )

            next_snapshot_keys = split_keys[:, 0].reshape(
                (lane_count, prototype_count, particle_count, 2)
            )
            return (
                to_grid(observation),
                to_grid(environment_state),
                next_snapshot_keys,
                proposal_indices.astype(jnp.int32),
            )

        def scatter_ordered(buffer: Any, candidate: Any, destinations: Any) -> Any:
            group_count = lane_count * prototype_count

            def scatter_leaf(left: Any, right: Any) -> Any:
                suffix = left.shape[3:]
                left_groups = left.reshape((group_count, particle_count) + suffix)
                right_groups = right.reshape((group_count, particle_count) + suffix)
                destination_groups = destinations.reshape(
                    (group_count, particle_count)
                )

                def scatter_group(
                    left_group: Any,
                    right_group: Any,
                    destination_group: Any,
                ) -> Any:
                    return left_group.at[destination_group].set(
                        right_group,
                        mode="drop",
                    )

                result = jax.vmap(scatter_group)(
                    left_groups,
                    right_groups,
                    destination_groups,
                )
                return result.reshape(left.shape)

            return jax.tree_util.tree_map(scatter_leaf, buffer, candidate)

        def accept_candidates(
            carry: Any,
            batch_index: Any,
            candidate_observation: Any,
            candidate_state: Any,
            candidate_keys: Any,
            candidate_proposal_indices: Any,
        ) -> Any:
            (
                state_buffer,
                observation_buffer,
                key_buffer,
                source_indices,
                accepted_counts,
            ) = carry
            expected = opening_observations[:, None, None, ...]
            observation_axes = tuple(
                range(3, candidate_observation["agent_1"].ndim)
            )
            compatible = jnp.all(
                candidate_observation["agent_1"] == expected,
                axis=observation_axes,
            ) & (
                candidate_proposal_indices
                < R015_FILTER_RESET_PROPOSAL_MULTIPLIER * particle_count
            )
            rank = jnp.cumsum(compatible.astype(jnp.int32), axis=-1) - 1
            destinations = accepted_counts[..., None] + rank
            valid = compatible & (destinations < particle_count)
            safe_destinations = jnp.where(valid, destinations, particle_count)
            state_buffer = scatter_ordered(
                state_buffer,
                candidate_state,
                safe_destinations,
            )
            observation_buffer = scatter_ordered(
                observation_buffer,
                candidate_observation,
                safe_destinations,
            )
            key_buffer = scatter_ordered(
                key_buffer,
                candidate_keys,
                safe_destinations,
            )
            source_indices = scatter_ordered(
                source_indices,
                candidate_proposal_indices,
                safe_destinations,
            )
            accepted_counts = jnp.minimum(
                particle_count,
                accepted_counts + jnp.sum(compatible, axis=-1, dtype=jnp.int32),
            )
            return (
                state_buffer,
                observation_buffer,
                key_buffer,
                source_indices,
                accepted_counts,
            )

        def accept_batch(carry: Any, batch_index: Any) -> Any:
            def generate_and_accept(current: Any) -> Any:
                (
                    candidate_observation,
                    candidate_state,
                    candidate_keys,
                    candidate_proposal_indices,
                ) = proposal_batch(batch_index)
                return accept_candidates(
                    current,
                    batch_index,
                    candidate_observation,
                    candidate_state,
                    candidate_keys,
                    candidate_proposal_indices,
                )

            return jax.lax.cond(
                jnp.all(carry[-1] >= particle_count),
                lambda current: current,
                generate_and_accept,
                carry,
            )

        (
            first_observation,
            first_state,
            first_keys,
            first_proposal_indices,
        ) = proposal_batch(jnp.asarray(0, dtype=jnp.int32))
        if cached_count > 0:
            def extend_grid(cached_value: Any, candidate_value: Any) -> Any:
                return jax.tree_util.tree_map(
                    lambda prefix, candidate: jnp.concatenate(
                        (
                            prefix,
                            candidate[:, :, cached_count:, ...],
                        ),
                        axis=2,
                    ),
                    cached_value,
                    candidate_value,
                )

            initial_state_buffer = extend_grid(
                cached_prefix_state["environment_state"],
                first_state,
            )
            initial_observation_buffer = extend_grid(
                cached_prefix_state["observation"],
                first_observation,
            )
            initial_key_buffer = extend_grid(
                cached_prefix_state["snapshot_keys"],
                first_keys,
            )
            empty_sources = jnp.concatenate(
                (
                    jnp.asarray(
                        cached_prefix_state["opening_accepted_source_indices"],
                        dtype=jnp.int32,
                    ),
                    jnp.full(
                        (
                            lane_count,
                            prototype_count,
                            particle_count - cached_count,
                        ),
                        -1,
                        dtype=jnp.int32,
                    ),
                ),
                axis=2,
            )
            empty_counts = jnp.full(
                (lane_count, prototype_count),
                cached_count,
                dtype=jnp.int32,
            )
        else:
            initial_state_buffer = first_state
            initial_observation_buffer = first_observation
            initial_key_buffer = first_keys
            empty_sources = jnp.full(
                (lane_count, prototype_count, particle_count),
                -1,
                dtype=jnp.int32,
            )
            empty_counts = jnp.zeros(
                (lane_count, prototype_count),
                dtype=jnp.int32,
            )
        initial = accept_candidates(
            (
                initial_state_buffer,
                initial_observation_buffer,
                initial_key_buffer,
                empty_sources,
                empty_counts,
            ),
            jnp.asarray(0, dtype=jnp.int32),
            first_observation,
            first_state,
            first_keys,
            first_proposal_indices,
        )
        (
            selected_state,
            selected_observation,
            selected_keys,
            accepted_source_indices,
            accepted_counts,
        ) = jax.lax.fori_loop(
            1,
            R015_FILTER_RESET_PROPOSAL_MULTIPLIER,
            accept_batch,
            initial,
        )
        opening_closed = jnp.any(accepted_counts < particle_count, axis=-1)
        proposal_cap = R015_FILTER_RESET_PROPOSAL_MULTIPLIER * particle_count
        proposal_counts = jnp.where(
            accepted_counts == particle_count,
            accepted_source_indices[..., -1] + 1,
            proposal_cap,
        )
        with jax.experimental.enable_x64():
            prototype_masses = jnp.full(
                (lane_count, prototype_count),
                1.0 / prototype_count,
                dtype=jnp.float64,
            )
            within_weights = jnp.full(
                (lane_count, prototype_count, particle_count),
                1.0 / particle_count,
                dtype=jnp.float64,
            )
            pre_resample_ess = jnp.ones(
                (lane_count, prototype_count),
                dtype=jnp.float64,
            )
        partner_state_grid = tuple(
            jax.tree_util.tree_map(
                lambda item: item.reshape(
                    (lane_count, particle_count) + item.shape[1:]
                ),
                state,
            )
            for state in initial_partner_states
        )
        empty_particle_records = jnp.full(
            (lane_count, prototype_count, particle_count),
            -1,
            dtype=jnp.int32,
        )
        return {
            "environment_state": selected_state,
            "observation": selected_observation,
            "snapshot_keys": selected_keys,
            "partner_recurrent_states": partner_state_grid,
            "prototype_masses": prototype_masses,
            "within_prototype_weights": within_weights,
            "pre_resample_ess_fraction": pre_resample_ess,
            "closed": opening_closed,
            "root_key_words": root_key_words,
            "opening_accepted_counts": accepted_counts,
            "opening_proposal_counts": proposal_counts,
            "opening_accepted_source_indices": accepted_source_indices,
            "lineage_indices": accepted_source_indices,
            "last_predicted_response_tokens": empty_particle_records,
            "last_partner_actions": empty_particle_records,
            "last_recipe_outcomes": empty_particle_records,
            "last_conditional_successor_indices": empty_particle_records,
            "update_count": jnp.zeros((lane_count,), dtype=jnp.int32),
        }

    def _advance_online_filter_v2_device(
        self,
        *,
        adapter: Any,
        filter_state: Mapping[str, Any],
        official_step_records: Mapping[str, Any],
        environment_step: Any,
        resampling_timing: str,
        update_keys: Any | None = None,
        return_diagnostics: bool = True,
    ) -> Mapping[str, Any]:
        """共享的纯设备一步核；可直接放入外层 ``jax.lax.scan``。"""

        import jax
        import jax.numpy as jnp
        from jaxmarl.environments.overcooked_v2.common import DynamicObject
        from experiments.overcooked_v2.path_c_response_probe import (
            REGISTERED_LOCAL_RESPONSE_SPEC,
        )

        if resampling_timing not in {
            "adaptive_ess_below_half_v1",
            "every_environment_step_v1",
        }:
            raise ValueError("R015 v2 filter uses an unregistered timing.")
        if not hasattr(adapter.env, "possible_recipes") or not callable(
            getattr(adapter.env, "get_obs", None)
        ):
            raise TypeError(
                "Official environment must expose possible_recipes and get_obs."
            )

        within = jnp.asarray(filter_state["within_prototype_weights"])
        prototype_masses = jnp.asarray(filter_state["prototype_masses"])
        lane_count, prototype_count, particle_count = within.shape
        action_count = len(OFFICIAL_ACTION_ORDER)
        possible_recipes = jnp.asarray(adapter.env.possible_recipes)
        encoded_possible_recipes = jax.vmap(
            DynamicObject.get_recipe_encoding
        )(possible_recipes)
        recipe_count = int(possible_recipes.shape[0])
        sample_recipe_on_delivery = getattr(
            adapter.env,
            "sample_recipe_on_delivery",
            None,
        )
        if sample_recipe_on_delivery is None:
            raise TypeError(
                "Official environment must expose sample_recipe_on_delivery."
            )
        if prototype_count != 4 or action_count != 6 or recipe_count <= 0:
            raise ValueError("R015 v2 filter has an invalid finite outcome space.")
        if update_keys is not None:
            raise ValueError(
                "R015 v2 innovations derive only from persistent filter roots."
            )

        environment_grid = filter_state["environment_state"]
        observation_grid = filter_state["observation"]
        observation_channels = int(observation_grid["agent_1"].shape[-1])
        if observation_channels < 27 or (observation_channels - 27) % 4:
            raise ValueError(
                "R015 response tokens require the official default observation."
            )
        ingredient_count = (observation_channels - 27) // 4
        other_agent_position_channel = ingredient_count + 7
        first_non_agent_channel = 2 * (ingredient_count + 7)
        visible_response_token = int(
            REGISTERED_LOCAL_RESPONSE_SPEC.encode(
                response_class="visible",
                latency_steps=1,
            )
        )
        unseen_response_token = int(
            REGISTERED_LOCAL_RESPONSE_SPEC.encode(
                response_class="unseen",
                latency_steps=1,
            )
        )
        local_change_response_token = int(
            REGISTERED_LOCAL_RESPONSE_SPEC.encode(
                response_class="local_non_agent_change",
                latency_steps=1,
            )
        )
        previous_partner_states = tuple(filter_state["partner_recurrent_states"])
        if len(previous_partner_states) != prototype_count:
            raise ValueError("R015 v2 filter lost one prototype actor state.")

        next_partner_states = []
        logits_by_prototype = []
        for prototype_index, prototype_id in enumerate(self.prototype_ids):
            prototype_observation = observation_grid["agent_0"][
                :, prototype_index, ...
            ].reshape(
                (lane_count * particle_count,)
                + observation_grid["agent_0"].shape[3:]
            )
            flat_previous_state = jax.tree_util.tree_map(
                lambda item: item.reshape(
                    (lane_count * particle_count,) + item.shape[2:]
                ),
                previous_partner_states[prototype_index],
            )
            next_state, logits = self._compiled_actor_boundary(
                self.policies[prototype_id].policy.params,
                flat_previous_state,
                prototype_observation,
            )
            next_partner_states.append(next_state)
            logits_by_prototype.append(
                logits.reshape((lane_count, particle_count, action_count))
            )
        logits = jnp.stack(tuple(logits_by_prototype), axis=1)
        finite_logits = jnp.all(jnp.isfinite(logits), axis=(-1, -2))

        total_parent_slots = lane_count * prototype_count * particle_count
        prototype_indices = jnp.tile(
            jnp.repeat(jnp.arange(prototype_count, dtype=jnp.uint32), particle_count),
            lane_count,
        )
        particle_indices = jnp.tile(
            jnp.tile(jnp.arange(particle_count, dtype=jnp.uint32), prototype_count),
            lane_count,
        )
        lineage_indices = jnp.asarray(filter_state["lineage_indices"])
        flat_lineage_indices = lineage_indices.reshape((-1,)).astype(jnp.uint32)
        lane_steps = jnp.asarray(environment_step, dtype=jnp.uint32)
        if lane_steps.ndim == 0:
            lane_steps = jnp.broadcast_to(lane_steps, (lane_count,))
        if lane_steps.shape != (lane_count,):
            raise ValueError("R015 v2 environment steps must be scalar or [lane].")
        filter_roots = jax.vmap(_r015_device_key_from_words)(
            filter_state["root_key_words"]
        )
        lane_filter_roots = jnp.repeat(
            filter_roots,
            prototype_count * particle_count,
            axis=0,
        )
        parent_steps = jnp.repeat(
            lane_steps,
            prototype_count * particle_count,
            axis=0,
        )

        def filter_innovation_key(
            root: Any,
            prototype_index: Any,
            accepted_proposal_index: Any,
            step_value: Any,
            purpose: Any,
            outcome: Any,
            child_slot: Any,
        ) -> Any:
            """依登记次序折入原型、来源、步骤、用途、结果与子槽。"""

            key = jax.random.fold_in(root, prototype_index)
            key = jax.random.fold_in(key, accepted_proposal_index)
            key = jax.random.fold_in(key, step_value)
            key = jax.random.fold_in(key, purpose)
            key = jax.random.fold_in(key, outcome)
            return jax.random.fold_in(key, child_slot)

        actions = jnp.arange(action_count, dtype=jnp.uint32)
        action_keys = jax.vmap(
            lambda root, prototype_index, lineage_index, step_value, particle_index: jax.vmap(
                lambda action: filter_innovation_key(
                    root,
                    prototype_index,
                    lineage_index,
                    step_value,
                    jnp.uint32(0x01500201),
                    action,
                    particle_index,
                )
            )(actions)
        )(
            lane_filter_roots,
            prototype_indices,
            flat_lineage_indices,
            parent_steps,
            particle_indices,
        )
        flat_action_keys = action_keys.reshape((-1, 2))
        action_key_pairs = jax.vmap(jax.random.split)(flat_action_keys)
        next_action_snapshot_keys = action_key_pairs[:, 0]

        def repeat_actions(value: Any) -> Any:
            return jax.tree_util.tree_map(
                lambda item: jnp.repeat(
                    item.reshape((total_parent_slots,) + item.shape[3:]),
                    action_count,
                    axis=0,
                ),
                value,
            )

        expanded_environment = repeat_actions(environment_grid)
        partner_actions = jnp.tile(
            jnp.arange(action_count, dtype=jnp.int32),
            total_parent_slots,
        )
        ego_actions = jnp.broadcast_to(
            jnp.asarray(official_step_records["ego_action_index"], dtype=jnp.int32)[
                :, None, None
            ],
            (lane_count, prototype_count, particle_count),
        ).reshape((-1,))
        ego_actions = jnp.repeat(ego_actions, action_count, axis=0)
        (
            base_observation,
            base_environment_state,
            base_rewards,
            base_dones,
            _base_info,
        ) = jax.vmap(adapter.env.step_env)(
            action_key_pairs[:, 1],
            expanded_environment,
            {"agent_0": partner_actions, "agent_1": ego_actions},
        )

        base_transition_count = total_parent_slots * action_count
        outcome_slot_count = base_transition_count * recipe_count
        repeated_environment = jax.tree_util.tree_map(
            lambda item: jnp.repeat(item, recipe_count, axis=0),
            base_environment_state,
        )
        base_correct_delivery = jnp.asarray(
            base_environment_state.new_correct_delivery,
            dtype=jnp.bool_,
        )
        expanded_randomized_delivery = jnp.repeat(
            base_correct_delivery,
            recipe_count,
            axis=0,
        ) & jnp.asarray(
            sample_recipe_on_delivery,
            dtype=jnp.bool_,
        )
        repeated_current_recipe = jnp.repeat(
            jnp.asarray(base_environment_state.recipe),
            recipe_count,
            axis=0,
        )
        tiled_recipes = jnp.tile(
            encoded_possible_recipes,
            (base_transition_count,)
            + (1,) * (encoded_possible_recipes.ndim - 1),
        )
        recipe_mask_shape = (outcome_slot_count,) + (1,) * (
            repeated_current_recipe.ndim - 1
        )
        outcome_recipes = jnp.where(
            expanded_randomized_delivery.reshape(recipe_mask_shape),
            tiled_recipes,
            repeated_current_recipe,
        )
        outcome_environment = self._replace_filter_recipe_v2(
            repeated_environment,
            outcome_recipes,
        )
        outcome_observation = jax.vmap(adapter.env.get_obs)(outcome_environment)

        base_reward = jnp.repeat(base_rewards["agent_0"], recipe_count, axis=0)
        base_boundary = jnp.repeat(base_dones["__all__"], recipe_count, axis=0)
        outcome_indices = jnp.tile(
            jnp.arange(recipe_count, dtype=jnp.int32),
            base_transition_count,
        )
        outcome_probability = jnp.where(
            expanded_randomized_delivery,
            jnp.asarray(1.0 / recipe_count, dtype=jnp.float64),
            jnp.where(
                outcome_indices == 0,
                jnp.asarray(1.0, dtype=jnp.float64),
                jnp.asarray(0.0, dtype=jnp.float64),
            ),
        ).reshape(
            (
                lane_count,
                prototype_count,
                particle_count,
                action_count,
                recipe_count,
            )
        )
        delivery_outcome_mask = expanded_randomized_delivery.reshape(
            (
                lane_count,
                prototype_count,
                particle_count,
                action_count,
                recipe_count,
            )
        )
        non_delivery_outcome_mask = (
            (~expanded_randomized_delivery) & (outcome_indices == 0)
        ).reshape(delivery_outcome_mask.shape)
        result_quality_multiplicity = jnp.sum(
            outcome_probability > 0.0,
            axis=-1,
            dtype=jnp.int32,
        )
        result_quality_multiplicity_is_one = result_quality_multiplicity == 1

        expected_observation = jnp.asarray(
            official_step_records["official_local_observation"]
        )[:, None, None, None, None, ...]
        outcome_observation_grid = jax.tree_util.tree_map(
            lambda item: item.reshape(
                (
                    lane_count,
                    prototype_count,
                    particle_count,
                    action_count,
                    recipe_count,
                )
                + item.shape[1:]
            ),
            outcome_observation,
        )
        observation_axes = tuple(
            range(5, outcome_observation_grid["agent_1"].ndim)
        )
        observation_match = jnp.all(
            outcome_observation_grid["agent_1"] == expected_observation,
            axis=observation_axes,
        )
        reward_grid = base_reward.reshape(
            (lane_count, prototype_count, particle_count, action_count, recipe_count)
        )
        boundary_grid = base_boundary.reshape(reward_grid.shape)
        reward_match = reward_grid == jnp.asarray(
            official_step_records["raw_team_reward"]
        )[:, None, None, None, None]
        boundary_match = boundary_grid == jnp.asarray(
            official_step_records["episode_boundary"], dtype=jnp.bool_
        )[:, None, None, None, None]
        compatible = observation_match & reward_match & boundary_match

        with jax.experimental.enable_x64():
            log_action_probability = jax.nn.log_softmax(
                logits.astype(jnp.float64),
                axis=-1,
            )
            log_outcome_probability = jnp.where(
                outcome_probability > 0.0,
                jnp.log(outcome_probability),
                -jnp.inf,
            )
            log_joint_probability = (
                log_action_probability[..., None] + log_outcome_probability
            )
            compatible_log_probability = jnp.where(
                compatible,
                log_joint_probability,
                -jnp.inf,
            )
            log_predictive_likelihood = jax.scipy.special.logsumexp(
                compatible_log_probability,
                axis=(-2, -1),
            )
            action_outcome_likelihood = jnp.exp(log_predictive_likelihood)
            compatible_joint_probability = jnp.where(
                compatible,
                jnp.exp(log_joint_probability),
                0.0,
            )
            log_within = jnp.where(within > 0.0, jnp.log(within), -jnp.inf)
            log_particle_predictive = (
                log_within + log_predictive_likelihood
            )
            log_predictive_mass = jax.scipy.special.logsumexp(
                log_particle_predictive,
                axis=-1,
            )
            predictive_mass = jnp.exp(log_predictive_mass)
            zero_support = ~jnp.isfinite(log_predictive_mass)
            normalized_predictive = jnp.exp(
                log_particle_predictive - log_predictive_mass[..., None]
            )
            pre_resample_ess = 1.0 / (
                particle_count
                * jnp.sum(normalized_predictive * normalized_predictive, axis=-1)
            )
            log_prototype_mass = jnp.where(
                prototype_masses > 0.0,
                jnp.log(prototype_masses),
                -jnp.inf,
            ) + log_predictive_mass
            log_total_mass = jax.scipy.special.logsumexp(
                log_prototype_mass,
                axis=-1,
                keepdims=True,
            )
            candidate_prototype_masses = jnp.exp(
                log_prototype_mass - log_total_mass
            )

        flat_conditional = jnp.where(
            compatible,
            jnp.exp(
                compatible_log_probability
                - log_predictive_likelihood[..., None, None]
            ),
            0.0,
        ).reshape(
            (lane_count, prototype_count, particle_count, action_count * recipe_count)
        )
        conditional_cdf = jnp.cumsum(flat_conditional, axis=-1)
        conditional_cdf = conditional_cdf.at[..., -1].set(1.0)

        conditional_keys = jax.vmap(
            lambda root, prototype_index, lineage_index, step_value, particle_index: filter_innovation_key(
                root,
                prototype_index,
                lineage_index,
                step_value,
                jnp.uint32(0x01500202),
                jnp.uint32(0),
                particle_index,
            )
        )(
            lane_filter_roots,
            prototype_indices,
            flat_lineage_indices,
            parent_steps,
            particle_indices,
        ).reshape((lane_count, prototype_count, particle_count, 2))
        with jax.experimental.enable_x64():
            conditional_uniform = jax.vmap(
                jax.vmap(
                    jax.vmap(
                        lambda key: jax.random.uniform(
                            key,
                            shape=(),
                            minval=0.0,
                            maxval=1.0,
                            dtype=jnp.float64,
                        )
                    )
                )
            )(conditional_keys)
        selected_outcome_slot = jnp.sum(
            conditional_cdf < conditional_uniform[..., None],
            axis=-1,
            dtype=jnp.int32,
        )

        def outcome_to_grid(value: Any) -> Any:
            return jax.tree_util.tree_map(
                lambda item: item.reshape(
                    (
                        lane_count,
                        prototype_count,
                        particle_count,
                        action_count * recipe_count,
                    )
                    + item.shape[1:]
                ),
                value,
            )

        outcome_environment_grid = outcome_to_grid(outcome_environment)
        selected_observation_grid = outcome_to_grid(outcome_observation)

        def gather_outcome(value: Any) -> Any:
            def gather_leaf(item: Any) -> Any:
                suffix = item.shape[4:]
                index = selected_outcome_slot.reshape(
                    selected_outcome_slot.shape + (1,) * (len(suffix) + 1)
                )
                index = jnp.broadcast_to(
                    index,
                    selected_outcome_slot.shape + (1,) + suffix,
                )
                return jnp.take_along_axis(item, index, axis=3).squeeze(axis=3)

            return jax.tree_util.tree_map(gather_leaf, value)

        selected_environment = gather_outcome(outcome_environment_grid)
        selected_observation = gather_outcome(selected_observation_grid)
        selected_partner_actions = selected_outcome_slot // recipe_count
        selected_recipe_outcomes = selected_outcome_slot % recipe_count
        previous_ego_observation = observation_grid["agent_1"]
        selected_ego_observation = selected_observation["agent_1"]
        partner_visible = jnp.any(
            selected_ego_observation[..., other_agent_position_channel] > 0,
            axis=(-2, -1),
        )
        center_y = int(previous_ego_observation.shape[-3]) // 2
        center_x = int(previous_ego_observation.shape[-2]) // 2
        local_non_agent_change = jnp.any(
            previous_ego_observation[
                ...,
                center_y - 1 : center_y + 2,
                center_x - 1 : center_x + 2,
                first_non_agent_channel:,
            ]
            != selected_ego_observation[
                ...,
                center_y - 1 : center_y + 2,
                center_x - 1 : center_x + 2,
                first_non_agent_channel:,
            ],
            axis=(-3, -2, -1),
        )
        selected_response_tokens = jnp.where(
            local_non_agent_change,
            jnp.asarray(local_change_response_token, dtype=jnp.int32),
            jnp.where(
                partner_visible,
                jnp.asarray(visible_response_token, dtype=jnp.int32),
                jnp.asarray(unseen_response_token, dtype=jnp.int32),
            ),
        )
        outcome_snapshot_keys = jnp.repeat(
            next_action_snapshot_keys,
            recipe_count,
            axis=0,
        ).reshape(
            (
                lane_count,
                prototype_count,
                particle_count,
                action_count * recipe_count,
                2,
            )
        )
        selected_snapshot_keys = gather_outcome(outcome_snapshot_keys)

        next_closed = (
            jnp.asarray(filter_state["closed"])
            | jnp.any(zero_support, axis=-1)
            | jnp.any(~finite_logits, axis=-1)
        )
        if resampling_timing == "every_environment_step_v1":
            due = jnp.ones(
                (lane_count, prototype_count), dtype=jnp.bool_
            )
        else:
            due = pre_resample_ess < 0.5
        effective_due = due & ~next_closed[:, None]

        phases = jax.vmap(
            lambda root, step_value: jax.vmap(
                lambda index: filter_innovation_key(
                    root,
                    index,
                    jnp.uint32(0),
                    step_value,
                    jnp.uint32(0x01500203),
                    jnp.uint32(0),
                    jnp.uint32(0),
                )
            )(
                jnp.arange(prototype_count, dtype=jnp.uint32)
            )
        )(filter_roots, lane_steps)
        with jax.experimental.enable_x64():
            phase = jax.vmap(jax.vmap(lambda key: jax.random.uniform(
                key,
                shape=(),
                minval=0.0,
                maxval=1.0,
                dtype=jnp.float64,
            )))(phases)
            cumulative = jnp.cumsum(normalized_predictive, axis=-1)
            cumulative = cumulative.at[..., -1].set(1.0)
            positions = (
                phase[..., None]
                + jnp.arange(particle_count, dtype=jnp.float64)
            ) / particle_count
        ancestor_indices = systematic_resample_indices_logarithmic(
            cumulative,
            positions,
        )

        def gather_particles(value: Any) -> Any:
            def gather_leaf(item: Any) -> Any:
                suffix = item.shape[3:]
                index = ancestor_indices.reshape(
                    ancestor_indices.shape + (1,) * len(suffix)
                )
                index = jnp.broadcast_to(
                    index,
                    ancestor_indices.shape + suffix,
                )
                return jnp.take_along_axis(item, index, axis=2)

            return jax.tree_util.tree_map(gather_leaf, value)

        def select_due(original: Any, resampled: Any) -> Any:
            return jax.tree_util.tree_map(
                lambda left, right: jnp.where(
                    effective_due.reshape(
                        effective_due.shape + (1,) * (left.ndim - 2)
                    ),
                    right,
                    left,
                ),
                original,
                resampled,
            )

        candidate_environment = select_due(
            selected_environment,
            gather_particles(selected_environment),
        )
        candidate_observation = select_due(
            selected_observation,
            gather_particles(selected_observation),
        )
        candidate_snapshot_keys = select_due(
            selected_snapshot_keys,
            gather_particles(selected_snapshot_keys),
        )
        candidate_lineage_indices = select_due(
            lineage_indices,
            gather_particles(lineage_indices),
        )
        candidate_response_tokens = select_due(
            selected_response_tokens,
            gather_particles(selected_response_tokens),
        )
        candidate_partner_actions = select_due(
            selected_partner_actions,
            gather_particles(selected_partner_actions),
        )
        candidate_recipe_outcomes = select_due(
            selected_recipe_outcomes,
            gather_particles(selected_recipe_outcomes),
        )
        candidate_successor_indices = select_due(
            selected_outcome_slot,
            gather_particles(selected_outcome_slot),
        )

        next_partner_tuple = []
        for prototype_index, next_partner_state in enumerate(next_partner_states):
            partner_grid = jax.tree_util.tree_map(
                lambda item: item.reshape(
                    (lane_count, particle_count) + item.shape[1:]
                ),
                next_partner_state,
            )
            source = ancestor_indices[:, prototype_index, :]

            def gather_partner_leaf(item: Any) -> Any:
                suffix = item.shape[2:]
                index = source.reshape(source.shape + (1,) * len(suffix))
                index = jnp.broadcast_to(index, source.shape + suffix)
                return jnp.take_along_axis(item, index, axis=1)

            resampled_partner = jax.tree_util.tree_map(
                gather_partner_leaf,
                partner_grid,
            )
            chosen_partner = jax.tree_util.tree_map(
                lambda left, right: jnp.where(
                    effective_due[:, prototype_index].reshape(
                        (lane_count,)
                        + (1,) * (left.ndim - 1)
                    ),
                    right,
                    left,
                ),
                partner_grid,
                resampled_partner,
            )
            previous_partner = previous_partner_states[prototype_index]
            chosen_partner = jax.tree_util.tree_map(
                lambda candidate, previous: jnp.where(
                    next_closed.reshape(
                        (lane_count,) + (1,) * (previous.ndim - 1)
                    ),
                    previous,
                    candidate,
                ),
                chosen_partner,
                previous_partner,
            )
            next_partner_tuple.append(chosen_partner)

        def freeze_closed_grid(previous: Any, candidate: Any) -> Any:
            return jax.tree_util.tree_map(
                lambda left, right: jnp.where(
                    next_closed.reshape(
                        (lane_count,) + (1,) * (left.ndim - 1)
                    ),
                    left,
                    right,
                ),
                previous,
                candidate,
            )

        with jax.experimental.enable_x64():
            next_within = jnp.where(
                effective_due[..., None],
                jnp.asarray(1.0 / particle_count, dtype=jnp.float64),
                normalized_predictive,
            )
            next_within = jnp.where(
                next_closed[:, None, None],
                within,
                next_within,
            )
            next_prototype_masses = jnp.where(
                next_closed[:, None],
                prototype_masses,
                candidate_prototype_masses,
            )
            next_ess = jnp.where(
                next_closed[:, None],
                jnp.asarray(filter_state["pre_resample_ess_fraction"]),
                pre_resample_ess,
            )
        next_response_tokens = freeze_closed_grid(
            filter_state["last_predicted_response_tokens"],
            candidate_response_tokens,
        )
        next_partner_action_records = freeze_closed_grid(
            filter_state["last_partner_actions"],
            candidate_partner_actions,
        )
        next_recipe_outcome_records = freeze_closed_grid(
            filter_state["last_recipe_outcomes"],
            candidate_recipe_outcomes,
        )
        next_successor_records = freeze_closed_grid(
            filter_state["last_conditional_successor_indices"],
            candidate_successor_indices,
        )
        next_filter_state = dict(filter_state)
        next_filter_state.update(
            {
                "environment_state": freeze_closed_grid(
                    environment_grid,
                    candidate_environment,
                ),
                "observation": freeze_closed_grid(
                    observation_grid,
                    candidate_observation,
                ),
                "snapshot_keys": freeze_closed_grid(
                    filter_state["snapshot_keys"],
                    candidate_snapshot_keys,
                ),
                "lineage_indices": freeze_closed_grid(
                    lineage_indices,
                    candidate_lineage_indices,
                ),
                "partner_recurrent_states": tuple(next_partner_tuple),
                "prototype_masses": next_prototype_masses,
                "within_prototype_weights": next_within,
                "pre_resample_ess_fraction": next_ess,
                "closed": next_closed,
                "last_predicted_response_tokens": next_response_tokens,
                "last_partner_actions": next_partner_action_records,
                "last_recipe_outcomes": next_recipe_outcome_records,
                "last_conditional_successor_indices": next_successor_records,
                "update_count": jnp.asarray(filter_state["update_count"])
                + (~jnp.asarray(filter_state["closed"])).astype(jnp.int32),
            }
        )
        direct_indices = jnp.broadcast_to(
            jnp.arange(particle_count, dtype=jnp.int32).reshape((1, 1, -1)),
            ancestor_indices.shape,
        )
        selected_sources = jnp.where(
            next_closed[:, None, None],
            -1,
            jnp.where(effective_due[..., None], ancestor_indices, direct_indices),
        )
        result = {
            "filter_state": next_filter_state,
            "posterior": next_prototype_masses,
            "pre_resample_ess_fraction": next_ess,
            "closed": next_closed,
            "resampling_due": effective_due,
            "ancestor_indices": selected_sources,
            "partner_actions": jnp.where(
                next_closed[:, None, None], -1, candidate_partner_actions
            ),
            "predicted_response_tokens": jnp.where(
                next_closed[:, None, None], -1, candidate_response_tokens
            ),
        }
        if return_diagnostics:
            result.update(
                {
                    "predictive_likelihood": action_outcome_likelihood,
                    "predictive_mass_by_prototype": predictive_mass,
                    "action_marginal_compatible_probability": jnp.sum(
                        outcome_probability * compatible.astype(jnp.float64),
                        axis=-1,
                    ),
                    "outcome_compatible": compatible,
                    "delivery_outcome_mask": delivery_outcome_mask,
                    "non_delivery_outcome_mask": non_delivery_outcome_mask,
                    "result_quality_multiplicity": result_quality_multiplicity,
                    "non_delivery_result_quality_multiplicity_is_one": jnp.all(
                        jnp.where(
                            jnp.any(non_delivery_outcome_mask, axis=-1),
                            result_quality_multiplicity_is_one,
                            True,
                        ),
                        axis=(-2, -1),
                    ),
                    "conditional_successor_indices": jnp.where(
                        next_closed[:, None, None],
                        -1,
                        candidate_successor_indices,
                    ),
                    "conditional_partner_actions": jnp.where(
                        next_closed[:, None, None],
                        -1,
                        candidate_partner_actions,
                    ),
                    "conditional_recipe_outcomes": jnp.where(
                        next_closed[:, None, None],
                        -1,
                        candidate_recipe_outcomes,
                    ),
                    "nonfinite_logits": ~finite_logits,
                    "particle_environment_transition_count": jnp.asarray(
                        base_transition_count,
                        dtype=jnp.int64,
                    ),
                    "finite_outcome_check_count": jnp.asarray(
                        outcome_slot_count,
                        dtype=jnp.int64,
                    ),
                }
            )
        return result

    def initialize_online_filter_v2(
        self,
        *,
        adapter: Any,
        official_opening_records: Sequence[Mapping[str, Any]],
        initialization_keys: Sequence[str],
        particles_per_prototype: int,
    ) -> Mapping[str, Any]:
        """宿主边界只准备登记输入；条件提议和选择全部留在设备。"""

        if not official_opening_records or len(official_opening_records) != len(
            initialization_keys
        ):
            raise ValueError("R015 v2 opening inputs have different lane counts.")
        import jax
        import jax.numpy as jnp

        opening = jnp.asarray(
            np.stack(
                tuple(
                    np.asarray(record["official_local_observation"])
                    for record in official_opening_records
                )
            )
        )
        root_words = jnp.asarray(
            np.asarray(
                tuple(_r015_sha256_words(value) for value in initialization_keys),
                dtype=np.uint32,
            )
        )
        lane_count = len(official_opening_records)
        prototype_actor_slot_count = lane_count * particles_per_prototype
        initial_partner_states = []
        for prototype_id in self.prototype_ids:
            policy = self.policies[prototype_id].policy
            initial_partner_states.append(
                policy.initial_state(prototype_actor_slot_count)
            )

        cache_key = ("initialize_v2", lane_count, int(particles_per_prototype))
        compiled = self._filter_scan_cache.get(cache_key)
        if compiled is None:
            compiled = jax.jit(
                lambda opening_value, words, states: self._initialize_online_filter_v2_device(
                    adapter=adapter,
                    opening_observations=opening_value,
                    root_key_words=words,
                    particles_per_prototype=int(particles_per_prototype),
                    initial_partner_states=states,
                )
            )
            self._filter_scan_cache[cache_key] = compiled
        return compiled(opening, root_words, tuple(initial_partner_states))

    def advance_online_filter_v2(
        self,
        *,
        adapter: Any,
        filter_state: Mapping[str, Any],
        official_step_records: Mapping[str, Any],
        environment_step: Any,
        resampling_timing: str,
        update_keys: Any | None = None,
        return_diagnostics: bool = True,
    ) -> Mapping[str, Any]:
        """公开的一步包装；外层设备扫描应直接调用同一个纯设备核。"""

        return self._advance_online_filter_v2_device(
            adapter=adapter,
            filter_state=filter_state,
            official_step_records=official_step_records,
            environment_step=environment_step,
            resampling_timing=resampling_timing,
            update_keys=update_keys,
            return_diagnostics=return_diagnostics,
        )

    def run_planning_branch_head_batch_v2(
        self,
        *,
        adapter: Any,
        filter_state: Mapping[str, Any],
        official_step_records: Mapping[str, Any],
        environment_step: Any,
        resampling_timing: str,
        response_tokens: Any,
    ) -> Mapping[str, Any]:
        """一次编译调用生成屏蔽回应和使用回应的分支头信念。"""

        import jax
        import jax.numpy as jnp
        from experiments.overcooked_v2.path_c_response_probe import (
            REGISTERED_LOCAL_RESPONSE_SPEC,
        )

        masses = jnp.asarray(filter_state["prototype_masses"])
        if masses.ndim != 2:
            raise ValueError("R015 planning branch heads require a lane axis.")
        lane_count = int(masses.shape[0])
        particle_count = int(
            jnp.asarray(filter_state["within_prototype_weights"]).shape[-1]
        )
        token_array = np.asarray(response_tokens, dtype=np.int32)
        registered_tokens = set(REGISTERED_LOCAL_RESPONSE_SPEC.token_ids.values())
        if any(int(value) not in registered_tokens for value in token_array.flat):
            raise ValueError("R015 planning received an unknown response token.")
        tokens = jnp.asarray(token_array, dtype=jnp.int32)
        if tokens.shape != (lane_count,):
            raise ValueError("R015 planning response tokens changed lane shape.")
        records = {
            "official_local_observation": jnp.asarray(
                official_step_records["official_local_observation"]
            ),
            "ego_action_index": jnp.asarray(
                official_step_records["ego_action_index"], dtype=jnp.int32
            ),
            "raw_team_reward": jnp.asarray(
                official_step_records["raw_team_reward"]
            ),
            "episode_boundary": jnp.asarray(
                official_step_records["episode_boundary"], dtype=jnp.bool_
            ),
        }
        steps = jnp.asarray(environment_step, dtype=jnp.int32)
        if steps.ndim == 0:
            steps = jnp.broadcast_to(steps, (lane_count,))
        if steps.shape != (lane_count,):
            raise ValueError("R015 planning branch-head steps changed lane shape.")
        member_ids = tuple(self.continuation_controller.member_ids)
        prototype_member_indices = jnp.asarray(
            tuple(member_ids.index(value) for value in self.prototype_ids),
            dtype=jnp.int32,
        )
        baseline_member_index = jnp.asarray(
            member_ids.index(self.baseline_member_id), dtype=jnp.int32
        )

        def compiled_branch_head(
            current_state: Mapping[str, Any],
            current_records: Mapping[str, Any],
            current_steps: Any,
            observed_tokens: Any,
        ) -> Any:
            normalized_tokens = jax.vmap(
                lambda value: jnp.asarray(value, dtype=jnp.int32)
            )(observed_tokens)
            masked_result = self._advance_online_filter_v2_device(
                adapter=adapter,
                filter_state=current_state,
                official_step_records=current_records,
                environment_step=current_steps,
                resampling_timing=resampling_timing,
                return_diagnostics=False,
            )
            masked_state = masked_result["filter_state"]
            used_state = self.reweight_online_filter_v2_response(
                filter_state=masked_state,
                response_tokens=normalized_tokens,
            )

            def committed_member_indices(state: Mapping[str, Any]) -> Any:
                posterior = jnp.asarray(state["prototype_masses"])
                maxima = jnp.max(posterior, axis=-1, keepdims=True)
                tied = posterior == maxima
                unique = jnp.sum(tied, axis=-1) == 1
                prototype_choice = prototype_member_indices[
                    jnp.argmax(posterior, axis=-1)
                ]
                baseline_index = baseline_member_index
                return jnp.where(unique, prototype_choice, baseline_index)

            response_match = (
                jnp.asarray(masked_state["last_predicted_response_tokens"])
                == normalized_tokens[..., None, None]
            )
            response_match_masses = jnp.sum(
                jnp.asarray(masked_state["prototype_masses"])[..., None]
                * jnp.asarray(masked_state["within_prototype_weights"])
                * response_match.astype(
                    jnp.asarray(masked_state["within_prototype_weights"]).dtype
                ),
                axis=-1,
            )
            summary = {
                "masked_prototype_masses": masked_state["prototype_masses"],
                "used_prototype_masses": used_state["prototype_masses"],
                "masked_closed": masked_state["closed"],
                "used_closed": used_state["closed"],
                "masked_committed_member_indices": committed_member_indices(
                    masked_state
                ),
                "used_committed_member_indices": committed_member_indices(
                    used_state
                ),
                "response_match_counts_by_prototype": jnp.sum(
                    response_match, axis=-1, dtype=jnp.int32
                ),
                "response_match_masses_by_prototype": response_match_masses,
            }
            return summary

        cache_key = (
            "planning_branch_head_v2",
            lane_count,
            particle_count,
            str(resampling_timing),
            id(adapter.env),
        )
        compiled = self._filter_scan_cache.get(cache_key)
        if compiled is None:
            compiled = jax.jit(compiled_branch_head)
            self._filter_scan_cache[cache_key] = compiled
        summary = compiled(
            filter_state,
            records,
            steps,
            tokens,
        )
        # 规划后缀只读取这个紧凑充分投影。一次性搬运八个小数组，避免外层
        # 为每个分支分别触发设备同步或拆回完整粒子云。
        host_summary = jax.device_get(summary)
        return {
            **host_summary,
            "continuation_member_ids": member_ids,
            "compiled_batch_calls": 1,
            "active_lane_count": lane_count,
            "host_sync_inside_environment_loop": False,
        }

    @staticmethod
    def online_filter_prototype_masses_v2(filter_state: Mapping[str, Any]) -> Any:
        """返回设备常驻的四原型质量，不读取任何隐藏真实身份。"""

        return filter_state["prototype_masses"]

    def materialize_online_filter_v2(
        self,
        *,
        filter_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """只在段边界把完整粒子云搬到宿主，供规划证据内容寻址。"""

        import jax

        host = jax.device_get(filter_state)
        prototype_masses = np.asarray(host["prototype_masses"])
        within = np.asarray(host["within_prototype_weights"])
        return {
            "environment_state": host["environment_state"],
            "raw_observation": host["observation"],
            "snapshot_keys": np.asarray(host["snapshot_keys"]),
            "partner_recurrent_states": host["partner_recurrent_states"],
            "prototype_masses": prototype_masses,
            "within_prototype_weights": within,
            "particle_total_weights": prototype_masses[..., None] * within,
            "pre_resample_ess_fraction": np.asarray(
                host["pre_resample_ess_fraction"]
            ),
            "closed": np.asarray(host["closed"], dtype=np.bool_),
            "opening_accepted_counts": np.asarray(
                host["opening_accepted_counts"]
            ),
            "opening_proposal_counts": np.asarray(
                host["opening_proposal_counts"]
            ),
            "opening_accepted_source_indices": np.asarray(
                host["opening_accepted_source_indices"]
            ),
            "last_predicted_response_tokens": np.asarray(
                host["last_predicted_response_tokens"], dtype=np.int32
            ),
            "last_partner_actions": np.asarray(
                host["last_partner_actions"], dtype=np.int32
            ),
            "last_recipe_outcomes": np.asarray(
                host["last_recipe_outcomes"], dtype=np.int32
            ),
            "last_conditional_successor_indices": np.asarray(
                host["last_conditional_successor_indices"], dtype=np.int32
            ),
            "update_count": np.asarray(host["update_count"]),
        }

    def materialize_online_filter_state_v2(
        self,
        *,
        adapter: Any,
        filter_state: Mapping[str, Any],
        template_belief: Any,
        continuation_states: Any,
        previous_official_observation: Any | None = None,
        predicted_raw_team_reward: float | None = None,
        predicted_done: bool | None = None,
    ) -> Any:
        """把段末设备过滤状态重新绑定为正式粒子后验。

        此导入只发生在真实执行段边界，避免稳定训练适配器反向依赖 R015；实际
        构造由完整回合模块的唯一实现完成，因此规划和执行不会各自维护一份转换。
        """

        from experiments.overcooked_v2.path_c_r015_full_horizon import (
            R015OnlineParticleBeliefV2,
            _materialize_online_particle_belief_v2,
        )

        if not isinstance(template_belief, R015OnlineParticleBeliefV2):
            raise TypeError("R015 v2 materialization requires its prior segment belief.")
        return _materialize_online_particle_belief_v2(
            production_backend=self,
            adapter=adapter,
            filter_state=filter_state,
            continuation_states=continuation_states,
            resampling_timing=template_belief.resampling_timing,
            template_belief=template_belief,
            previous_official_observation=previous_official_observation,
            predicted_raw_team_reward=predicted_raw_team_reward,
            predicted_done=predicted_done,
        )

    def reweight_online_filter_v2_response(
        self,
        *,
        filter_state: Mapping[str, Any],
        response_tokens: Any | None = None,
        prototype_masses: Any | None = None,
        within_prototype_weights: Any | None = None,
        particle_total_weights: Any | None = None,
    ) -> Mapping[str, Any]:
        """用登记回应记号重加权；隐藏状态和循环状态逐字节保持。"""

        import jax.numpy as jnp

        current_masses = jnp.asarray(filter_state["prototype_masses"])
        current_within = jnp.asarray(filter_state["within_prototype_weights"])
        external_weight_form = any(
            value is not None
            for value in (
                prototype_masses,
                within_prototype_weights,
                particle_total_weights,
            )
        )
        if response_tokens is not None and external_weight_form:
            raise ValueError(
                "R015 response tokens cannot be combined with caller-supplied weights."
            )
        if response_tokens is not None:
            observed = jnp.asarray(response_tokens, dtype=jnp.int32)
            lane_shape = current_masses.shape[:-1]
            if observed.ndim == 0:
                observed = jnp.broadcast_to(observed, lane_shape)
            if observed.shape != lane_shape:
                raise ValueError("R015 response-token lanes changed shape.")
            predicted = jnp.asarray(
                filter_state["last_predicted_response_tokens"],
                dtype=jnp.int32,
            )
            if predicted.shape != current_within.shape:
                raise ValueError("R015 predicted response tokens changed particle shape.")
            matched_total = (
                current_masses[..., None]
                * current_within
                * (predicted == observed[..., None, None]).astype(
                    current_within.dtype
                )
            )
            matched_prototype_mass = jnp.sum(matched_total, axis=-1)
            finite_support = jnp.all(
                jnp.isfinite(matched_prototype_mass), axis=-1
            )
            positive_support = jnp.all(
                matched_prototype_mass > 0.0, axis=-1
            )
            lane_supported = finite_support & positive_support
            matched_total_mass = jnp.sum(
                matched_prototype_mass, axis=-1, keepdims=True
            )
            candidate_masses = matched_prototype_mass / jnp.where(
                matched_total_mass > 0.0,
                matched_total_mass,
                1.0,
            )
            candidate_within = matched_total / jnp.where(
                matched_prototype_mass[..., None] > 0.0,
                matched_prototype_mass[..., None],
                1.0,
            )
            masses = jnp.where(
                lane_supported[..., None],
                candidate_masses,
                current_masses,
            )
            within = jnp.where(
                lane_supported[..., None, None],
                candidate_within,
                current_within,
            )
            zero_support = ~lane_supported
        elif particle_total_weights is not None:
            if prototype_masses is not None or within_prototype_weights is not None:
                raise ValueError("R015 response reweighting received two weight forms.")
            total = jnp.asarray(particle_total_weights, dtype=current_within.dtype)
            masses = jnp.sum(total, axis=-1)
            total_mass = jnp.sum(masses, axis=-1, keepdims=True)
            normalized_total = total / jnp.where(
                total_mass[..., None] > 0.0,
                total_mass[..., None],
                1.0,
            )
            masses = jnp.sum(normalized_total, axis=-1)
            within = normalized_total / jnp.where(
                masses[..., None] > 0.0,
                masses[..., None],
                1.0,
            )
        else:
            masses = (
                current_masses
                if prototype_masses is None
                else jnp.asarray(prototype_masses, dtype=current_masses.dtype)
            )
            within = (
                current_within
                if within_prototype_weights is None
                else jnp.asarray(
                    within_prototype_weights,
                    dtype=current_within.dtype,
                )
            )
            masses = masses / jnp.where(
                jnp.sum(masses, axis=-1, keepdims=True) > 0.0,
                jnp.sum(masses, axis=-1, keepdims=True),
                1.0,
            )
            within = within / jnp.where(
                jnp.sum(within, axis=-1, keepdims=True) > 0.0,
                jnp.sum(within, axis=-1, keepdims=True),
                1.0,
            )
        if response_tokens is None:
            zero_support = (
                jnp.sum(masses, axis=-1) <= 0.0
            ) | jnp.any(
                (masses <= 0.0) | (jnp.sum(within, axis=-1) <= 0.0),
                axis=-1,
            )
        updated = dict(filter_state)
        updated["prototype_masses"] = masses
        updated["within_prototype_weights"] = within
        updated["closed"] = jnp.asarray(filter_state["closed"]) | zero_support
        return updated

    def run_offline_filter_scan_v2(
        self,
        *,
        adapter: Any,
        official_histories: Sequence[Sequence[Mapping[str, Any]]],
        initialization_keys: Sequence[str],
        particles_per_prototype: int,
        resampling_timing: str,
        return_particle_weight_trace: bool = False,
        environment_step_limit: int = 400,
        return_state_diagnostics: bool = False,
        conditioned_opening_prefix_cache: dict[str, Any] | None = None,
        conditioned_opening_prefix_cache_binding_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        """用共享 v2 一步核在一个编译扫描中重放完整官方历史。"""

        if not official_histories or len(official_histories) != len(
            initialization_keys
        ):
            raise ValueError("R015 v2 filter inputs have different lane counts.")
        if particles_per_prototype <= 0:
            raise ValueError("R015 v2 filter particle count must be positive.")
        if resampling_timing not in {
            "adaptive_ess_below_half_v1",
            "every_environment_step_v1",
        }:
            raise ValueError("R015 v2 filter uses an unregistered timing.")
        if (
            isinstance(environment_step_limit, bool)
            or not isinstance(environment_step_limit, int)
            or not 1 <= environment_step_limit <= 400
        ):
            raise ValueError("R015 v2 filter step limit must lie in 1,...,400.")
        if return_state_diagnostics and not return_particle_weight_trace:
            raise ValueError("R015 v2 state diagnostics require the particle trace.")
        if (conditioned_opening_prefix_cache is None) != (
            conditioned_opening_prefix_cache_binding_sha256 is None
        ):
            raise ValueError("R015 v2 opening cache requires its binding SHA-256.")
        if conditioned_opening_prefix_cache is not None and (
            not isinstance(conditioned_opening_prefix_cache, dict)
            or not isinstance(conditioned_opening_prefix_cache_binding_sha256, str)
            or len(conditioned_opening_prefix_cache_binding_sha256) != 64
        ):
            raise ValueError("R015 v2 opening cache binding is malformed.")

        from experiments.overcooked_v2.path_c_r015_runtime import OCV2_ACTION_INDEX

        import jax
        import jax.numpy as jnp

        histories = tuple(tuple(value) for value in official_histories)
        if any(len(value) != 401 for value in histories):
            raise ValueError("R015 v2 filter requires one opening record and 400 steps.")
        lane_count = len(histories)
        step_count = int(environment_step_limit)
        opening = np.stack(
            tuple(
                np.asarray(history[0]["official_local_observation"])
                for history in histories
            )
        )
        expected_observations = np.stack(
            tuple(
                np.stack(
                    tuple(
                        np.asarray(record["official_local_observation"])
                        for record in history[1 : step_count + 1]
                    )
                )
                for history in histories
            )
        )
        ego_actions = np.asarray(
            tuple(
                tuple(
                    OCV2_ACTION_INDEX[str(record["ego_action_history"][0])]
                    for record in history[1 : step_count + 1]
                )
                for history in histories
            ),
            dtype=np.int32,
        )
        rewards = np.asarray(
            tuple(
                tuple(
                    float(record["raw_team_reward_history"][0])
                    for record in history[1 : step_count + 1]
                )
                for history in histories
            ),
            dtype=np.float32,
        )
        boundaries = np.asarray(
            tuple(
                tuple(
                    bool(record["episode_boundaries"][0])
                    for record in history[1 : step_count + 1]
                )
                for history in histories
            ),
            dtype=np.bool_,
        )
        root_words = np.asarray(
            tuple(_r015_sha256_words(value) for value in initialization_keys),
            dtype=np.uint32,
        )

        prototype_actor_slot_count = lane_count * particles_per_prototype
        initial_partner_states = []
        for prototype_id in self.prototype_ids:
            policy = self.policies[prototype_id].policy
            initial_partner_states.append(
                policy.initial_state(prototype_actor_slot_count)
            )

        def slice_cached_opening_payload(
            value: Mapping[str, Any],
            count: int,
        ) -> Mapping[str, Any]:
            """只取条件化开局规范序列的前 ``count`` 个粒子槽。"""

            accepted_counts = np.asarray(value["opening_accepted_counts"])
            if count <= 0 or accepted_counts.shape != (1, 4) or np.any(
                accepted_counts < count
            ):
                raise ValueError(
                    "R015 opening cache cannot expose an incomplete particle prefix."
                )
            payload = dict(value)
            payload["environment_state"] = jax.tree_util.tree_map(
                lambda item: np.asarray(item)[:, :, :count, ...],
                value["environment_state"],
            )
            payload["observation"] = jax.tree_util.tree_map(
                lambda item: np.asarray(item)[:, :, :count, ...],
                value["observation"],
            )
            for field_name in (
                "snapshot_keys",
                "opening_accepted_source_indices",
                "lineage_indices",
                "last_predicted_response_tokens",
                "last_partner_actions",
                "last_recipe_outcomes",
                "last_conditional_successor_indices",
            ):
                payload[field_name] = np.asarray(value[field_name])[
                    :, :, :count, ...
                ]
            payload["partner_recurrent_states"] = tuple(
                jax.tree_util.tree_map(
                    lambda item: np.asarray(item)[:, :count, ...],
                    state,
                )
                for state in value["partner_recurrent_states"]
            )
            payload["prototype_masses"] = np.full(
                (1, 4), 0.25, dtype=np.float64
            )
            payload["within_prototype_weights"] = np.full(
                (1, 4, count), 1.0 / count, dtype=np.float64
            )
            payload["pre_resample_ess_fraction"] = np.ones(
                (1, 4), dtype=np.float64
            )
            payload["closed"] = np.zeros((1,), dtype=np.bool_)
            payload["opening_accepted_counts"] = np.full(
                (1, 4), count, dtype=np.int32
            )
            payload["opening_proposal_counts"] = np.asarray(
                payload["opening_accepted_source_indices"]
            )[..., -1] + 1
            payload["update_count"] = np.zeros((1,), dtype=np.int32)
            return payload

        cached_particles = 0
        cached_opening_state: Mapping[str, Any] | None = None
        cached_proposal_counts_before = np.zeros((lane_count, 4), dtype=np.int32)
        if conditioned_opening_prefix_cache is not None:
            entries = tuple(
                conditioned_opening_prefix_cache.get(key)
                for key in initialization_keys
            )
            if any(entry is not None and not isinstance(entry, Mapping) for entry in entries):
                raise TypeError("R015 opening cache contains a malformed lane entry.")
            present = tuple(entry for entry in entries if isinstance(entry, Mapping))
            if present:
                if any(
                    entry.get("binding_sha256")
                    != conditioned_opening_prefix_cache_binding_sha256
                    for entry in present
                ):
                    raise RuntimeError("R015 opening cache binding SHA-256 changed.")
                if any(
                    isinstance(
                        entry.get("materialized_particles_per_prototype"), bool
                    )
                    or not isinstance(
                        entry.get("materialized_particles_per_prototype"), int
                    )
                    or not 0
                    <= int(entry["materialized_particles_per_prototype"])
                    <= 256
                    or (
                        int(entry["materialized_particles_per_prototype"]) > 0
                        and not isinstance(entry.get("payload"), Mapping)
                    )
                    for entry in present
                ):
                    raise RuntimeError(
                        "R015 opening cache contains an invalid reusable prefix."
                    )
                # A device batch can reuse a prefix only when every lane owns a
                # complete prefix.  A missing or zero-length lane makes the whole
                # batch rebuild from its registered root keys; successful lanes
                # remain cached for a later homogeneous batch.
                if len(present) == lane_count:
                    realized_counts = tuple(
                        int(entry.get("materialized_particles_per_prototype", 0))
                        for entry in present
                    )
                    reusable_count = min(realized_counts, default=0)
                    cached_particles = min(
                        reusable_count,
                        int(particles_per_prototype),
                    )
                    if cached_particles > 0:
                        lane_payloads = tuple(
                            slice_cached_opening_payload(
                                entry["payload"], cached_particles
                            )
                            for entry in present
                        )
                        cached_opening_state = jax.tree_util.tree_map(
                            lambda *items: np.concatenate(
                                tuple(np.asarray(item) for item in items),
                                axis=0,
                            ),
                            *lane_payloads,
                        )
                        cached_proposal_counts_before = np.concatenate(
                            tuple(
                                np.asarray(payload["opening_proposal_counts"])
                                for payload in lane_payloads
                            ),
                            axis=0,
                        )

        def compiled_filter_scan(
            opening_value: Any,
            observation_values: Any,
            action_values: Any,
            reward_values: Any,
            boundary_values: Any,
            word_values: Any,
            partner_states: Any,
            cached_state: Any,
        ) -> Any:
            state = self._initialize_online_filter_v2_device(
                adapter=adapter,
                opening_observations=opening_value,
                root_key_words=word_values,
                particles_per_prototype=int(particles_per_prototype),
                initial_partner_states=partner_states,
                cached_prefix_state=cached_state,
                cached_particles_per_prototype=cached_particles,
            )
            def scan_step(carry: Any, step_input: Any) -> Any:
                (
                    step_index,
                    expected_observation,
                    ego_action,
                    raw_reward,
                    boundary,
                ) = step_input

                advanced = self._advance_online_filter_v2_device(
                    adapter=adapter,
                    filter_state=carry,
                    official_step_records={
                        "official_local_observation": expected_observation,
                        "ego_action_index": ego_action,
                        "raw_team_reward": raw_reward,
                        "episode_boundary": boundary,
                    },
                    environment_step=step_index,
                    resampling_timing=resampling_timing,
                    return_diagnostics=bool(return_particle_weight_trace),
                )
                next_state = advanced["filter_state"]
                trace = {
                    "posterior": advanced["posterior"],
                    "ess": advanced["pre_resample_ess_fraction"],
                    "closed": advanced["closed"],
                }
                if return_particle_weight_trace:
                    trace.update(
                        {
                            "particle_weights": advanced["posterior"][..., None]
                            * next_state["within_prototype_weights"],
                            "partner_actions": advanced["partner_actions"],
                            "resampling_due": advanced["resampling_due"],
                            "selected_source_indices": advanced[
                                "ancestor_indices"
                            ],
                        }
                    )
                    if return_state_diagnostics:
                        def flatten_partner(states: Any) -> Any:
                            return jax.tree_util.tree_map(
                                lambda *items: jnp.stack(
                                    tuple(
                                        item
                                        for item in items
                                    ),
                                    axis=1,
                                ).reshape(
                                    (lane_count * 4 * particles_per_prototype,)
                                    + items[0].shape[2:]
                                ),
                                *states,
                            )

                        trace.update(
                            {
                                "partner_recurrent_state_before": flatten_partner(
                                    carry["partner_recurrent_states"]
                                ),
                                "agent_0_observation_before": carry["observation"][
                                    "agent_0"
                                ].reshape(
                                    (lane_count * 4 * particles_per_prototype,)
                                    + carry["observation"]["agent_0"].shape[3:]
                                ),
                                "partner_recurrent_state": flatten_partner(
                                    next_state["partner_recurrent_states"]
                                ),
                                "agent_0_observation": next_state["observation"][
                                    "agent_0"
                                ].reshape(
                                    (lane_count * 4 * particles_per_prototype,)
                                    + next_state["observation"]["agent_0"].shape[3:]
                                ),
                            }
                        )
                return next_state, trace

            scan_inputs = (
                jnp.arange(1, step_count + 1, dtype=jnp.int32),
                jnp.swapaxes(observation_values, 0, 1),
                jnp.swapaxes(action_values, 0, 1),
                jnp.swapaxes(reward_values, 0, 1),
                jnp.swapaxes(boundary_values, 0, 1),
            )
            final_state, trace = jax.lax.scan(scan_step, state, scan_inputs)
            return final_state, trace, state

        cache_key = (
            "offline_v2",
            lane_count,
            int(particles_per_prototype),
            resampling_timing,
            bool(return_particle_weight_trace),
            step_count,
            bool(return_state_diagnostics),
            cached_particles,
        )
        compiled = self._filter_scan_cache.get(cache_key)
        compilation_count = 0
        if compiled is None:
            compiled = jax.jit(compiled_filter_scan)
            self._filter_scan_cache[cache_key] = compiled
            compilation_count = 1
        started = time.perf_counter()
        final_state, trace, opening_state = compiled(
            jnp.asarray(opening),
            jnp.asarray(expected_observations),
            jnp.asarray(ego_actions),
            jnp.asarray(rewards),
            jnp.asarray(boundaries),
            jnp.asarray(root_words),
            tuple(initial_partner_states),
            cached_opening_state,
        )
        host_trace, host_opening, host_opening_state = jax.device_get(
            (
                trace,
                {
                    "opening_accepted_counts": final_state[
                        "opening_accepted_counts"
                    ],
                    "opening_proposal_counts": final_state[
                        "opening_proposal_counts"
                    ],
                    "opening_accepted_source_indices": final_state[
                        "opening_accepted_source_indices"
                    ],
                },
                opening_state,
            )
        )
        if conditioned_opening_prefix_cache is not None:
            import hashlib

            def lane_slice(value: Any, lane: int) -> Any:
                return jax.tree_util.tree_map(
                    lambda item: np.asarray(item[lane : lane + 1]),
                    value,
                )

            def prefix_payload(value: Mapping[str, Any], count: int) -> Mapping[str, Any]:
                accepted_counts = np.asarray(value["opening_accepted_counts"])
                if count <= 0 or accepted_counts.shape != (1, 4) or np.any(
                    accepted_counts < count
                ):
                    raise ValueError(
                        "R015 opening cache cannot store an incomplete particle prefix."
                    )
                payload = dict(value)
                payload["environment_state"] = jax.tree_util.tree_map(
                    lambda item: np.asarray(item)[:, :, :count, ...],
                    value["environment_state"],
                )
                payload["observation"] = jax.tree_util.tree_map(
                    lambda item: np.asarray(item)[:, :, :count, ...],
                    value["observation"],
                )
                payload["snapshot_keys"] = np.asarray(value["snapshot_keys"])[
                    :, :, :count, ...
                ]
                for field_name in (
                    "last_predicted_response_tokens",
                    "last_partner_actions",
                    "last_recipe_outcomes",
                    "last_conditional_successor_indices",
                    "lineage_indices",
                ):
                    payload[field_name] = np.asarray(value[field_name])[
                        :, :, :count, ...
                    ]
                payload["partner_recurrent_states"] = tuple(
                    jax.tree_util.tree_map(
                        lambda item: np.asarray(item)[:, :count, ...],
                        state,
                    )
                    for state in value["partner_recurrent_states"]
                )
                payload["within_prototype_weights"] = np.full(
                    (1, 4, count),
                    1.0 / count,
                    dtype=np.float64,
                )
                payload["prototype_masses"] = np.full(
                    (1, 4), 0.25, dtype=np.float64
                )
                payload["pre_resample_ess_fraction"] = np.ones(
                    (1, 4), dtype=np.float64
                )
                payload["closed"] = np.zeros((1,), dtype=np.bool_)
                payload["opening_accepted_source_indices"] = np.asarray(
                    value["opening_accepted_source_indices"]
                )[:, :, :count]
                payload["opening_accepted_counts"] = np.full(
                    (1, 4), count, dtype=np.int32
                )
                payload["opening_proposal_counts"] = np.asarray(
                    payload["opening_accepted_source_indices"]
                )[..., -1] + 1
                payload["update_count"] = np.zeros((1,), dtype=np.int32)
                return payload

            def prefix_digest(value: Mapping[str, Any], count: int) -> str:
                prefix = prefix_payload(value, count)
                digest = hashlib.sha256()
                digest.update(R015_FILTER_KEY_CONTRACT_V2.encode("ascii"))
                digest.update(str(count).encode("ascii"))
                core = (
                    prefix["environment_state"],
                    prefix["observation"],
                    prefix["snapshot_keys"],
                    prefix["partner_recurrent_states"],
                    prefix["opening_accepted_source_indices"],
                    prefix["last_predicted_response_tokens"],
                )
                for leaf in jax.tree_util.tree_leaves(core):
                    array = np.ascontiguousarray(np.asarray(leaf))
                    digest.update(str(array.dtype).encode("ascii"))
                    digest.update(repr(tuple(array.shape)).encode("ascii"))
                    digest.update(array.tobytes(order="C"))
                return digest.hexdigest()

            def opening_attempt_digest(
                value: Mapping[str, Any], requested_count: int
            ) -> str:
                """摘要一次尝试，但不把失败尝试冒充可复用前缀。"""

                digest = hashlib.sha256()
                digest.update(b"path_c_r015_conditioned_opening_attempt_v2")
                digest.update(R015_FILTER_KEY_CONTRACT_V2.encode("ascii"))
                digest.update(str(requested_count).encode("ascii"))
                core = (
                    value["environment_state"],
                    value["observation"],
                    value["snapshot_keys"],
                    value["partner_recurrent_states"],
                    value["opening_accepted_source_indices"],
                    value["opening_accepted_counts"],
                    value["opening_proposal_counts"],
                    value["closed"],
                )
                for leaf in jax.tree_util.tree_leaves(core):
                    array = np.ascontiguousarray(np.asarray(leaf))
                    digest.update(str(array.dtype).encode("ascii"))
                    digest.update(repr(tuple(array.shape)).encode("ascii"))
                    digest.update(array.tobytes(order="C"))
                return digest.hexdigest()

            for lane, initialization_key in enumerate(initialization_keys):
                lane_opening = lane_slice(host_opening_state, lane)
                previous = conditioned_opening_prefix_cache.get(initialization_key)
                previous_count = 0
                previous_digests: dict[str, str] = {}
                previous_attempt_digests: dict[str, str] = {}
                if isinstance(previous, Mapping):
                    previous_count = int(
                        previous.get("materialized_particles_per_prototype", 0)
                    )
                    previous_digests = {
                        str(key): str(value)
                        for key, value in dict(
                            previous.get("prefix_sha256_by_particle_count", {})
                        ).items()
                    }
                    previous_attempt_digests = {
                        str(key): str(value)
                        for key, value in dict(
                            previous.get(
                                "opening_attempt_sha256_by_particle_count", {}
                            )
                        ).items()
                    }
                accepted_counts = np.asarray(
                    lane_opening["opening_accepted_counts"], dtype=np.int64
                )
                completed_counts = tuple(
                    count
                    for count in (64, 128, 256)
                    if count <= int(particles_per_prototype)
                    and np.all(accepted_counts >= count)
                )
                completed_count = max(completed_counts, default=0)
                materialized = max(previous_count, completed_count)
                if completed_count >= materialized and materialized > 0:
                    stored_payload = prefix_payload(lane_opening, materialized)
                elif isinstance(previous, Mapping) and previous_count > 0:
                    stored_payload = previous.get("payload")
                else:
                    stored_payload = None
                for count in (64, 128, 256):
                    if count <= completed_count:
                        count_key = str(count)
                        realized_digest = prefix_digest(lane_opening, count)
                        previous_digest = previous_digests.get(count_key)
                        if previous_digest is not None and (
                            previous_digest != realized_digest
                        ):
                            raise RuntimeError(
                                "R015 opening prefix changed while extending its cache."
                            )
                        previous_digests[count_key] = realized_digest
                requested_key = str(int(particles_per_prototype))
                realized_attempt_digest = opening_attempt_digest(
                    lane_opening, int(particles_per_prototype)
                )
                previous_attempt_digest = previous_attempt_digests.get(requested_key)
                if previous_attempt_digest is not None and (
                    previous_attempt_digest != realized_attempt_digest
                ):
                    raise RuntimeError(
                        "R015 opening attempt changed for the same particle count."
                    )
                previous_attempt_digests[requested_key] = realized_attempt_digest
                conditioned_opening_prefix_cache[initialization_key] = {
                    "binding_sha256": (
                        conditioned_opening_prefix_cache_binding_sha256
                    ),
                    "materialized_particles_per_prototype": materialized,
                    "prefix_sha256_by_particle_count": previous_digests,
                    "opening_attempt_sha256_by_particle_count": (
                        previous_attempt_digests
                    ),
                    **({"payload": stored_payload} if materialized > 0 else {}),
                }
        wall_seconds = time.perf_counter() - started
        action_transitions = step_count * lane_count * 4 * particles_per_prototype * 6
        finite_outcome_checks = (
            action_transitions * int(np.asarray(adapter.env.possible_recipes).shape[0])
        )
        proposal_delta = np.maximum(
            0,
            np.asarray(host_opening["opening_proposal_counts"], dtype=np.int64)
            - cached_proposal_counts_before.astype(np.int64),
        )
        maximum_opening_proposal_count = int(np.max(proposal_delta))
        opening_batches = (
            maximum_opening_proposal_count + particles_per_prototype - 1
        ) // particles_per_prototype
        opening_proposals = (
            lane_count * 4 * opening_batches * particles_per_prototype
        )
        return {
            "filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
            "filter_key_contract": R015_FILTER_KEY_CONTRACT_V2,
            "device_execution_id": R015_FILTER_DEVICE_EXECUTION_ID_V2,
            "trace": host_trace,
            "opening_diagnostics": host_opening,
            "throughput": {
                "true_particle_environment_transitions": action_transitions,
                "finite_environment_outcome_checks": finite_outcome_checks,
                "opening_reset_proposals_evaluated": opening_proposals,
                "conditioned_opening_prefix_cache_contract_id": (
                    "r015_shared_conditioned_opening_256_particle_prefix_v1"
                ),
                "conditioned_opening_prefix_cache_binding_sha256": (
                    conditioned_opening_prefix_cache_binding_sha256
                ),
                "conditioned_opening_prefix_particles_used_per_prototype": int(
                    cached_particles
                ),
                "post_resampling_state_reuse_count": 0,
                "compiled_batch_calls": 1,
                "jit_compilations": compilation_count,
                "active_lane_count": lane_count,
                "active_particle_batch_width": lane_count
                * 4
                * particles_per_prototype,
                "policy_actor_lane_batch_width": prototype_actor_slot_count,
                "wall_seconds": wall_seconds,
                "true_transitions_per_second": (
                    action_transitions / wall_seconds if wall_seconds > 0.0 else None
                ),
                "host_sync_inside_environment_loop": False,
            },
        }

    def replay_selected_consultation_states_v2(
        self,
        *,
        adapter: Any,
        official_histories: Sequence[Sequence[Mapping[str, Any]]],
        initialization_keys: Sequence[str],
        particles_per_prototype: int,
        resampling_timing: str,
        consultation_steps: Sequence[int],
        conditioned_opening_prefix_cache: Mapping[str, Any] | None = None,
        conditioned_opening_prefix_cache_binding_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        """同步重放选中过滤器与五个延续成员，只返回登记咨询边界。"""

        from experiments.overcooked_v2.path_c_r015_runtime import OCV2_ACTION_INDEX

        import jax
        import jax.numpy as jnp

        histories = tuple(tuple(value) for value in official_histories)
        keys = tuple(str(value) for value in initialization_keys)
        steps = tuple(int(value) for value in consultation_steps)
        if not histories or len(histories) != len(keys):
            raise ValueError(
                "R015 selected-filter histories and initialization keys differ."
            )
        if any(len(history) != 401 for history in histories):
            raise ValueError("R015 selected-filter replay requires 400-step histories.")
        if steps != tuple(range(1, 97, 5)):
            raise ValueError("R015 selected-filter replay changed consultation steps.")
        if (
            isinstance(particles_per_prototype, bool)
            or int(particles_per_prototype) not in {64, 128, 256}
            or len(self.prototype_ids) != 4
        ):
            raise ValueError(
                "R015 selected-filter replay requires four prototypes and a registered particle count."
            )
        if resampling_timing not in {
            "adaptive_ess_below_half_v1",
            "every_environment_step_v1",
        }:
            raise ValueError("R015 selected-filter replay uses an unknown timing.")
        if (conditioned_opening_prefix_cache is None) != (
            conditioned_opening_prefix_cache_binding_sha256 is None
        ):
            raise ValueError(
                "R015 selected-filter opening cache requires its binding SHA-256."
            )
        if conditioned_opening_prefix_cache is not None and (
            not isinstance(conditioned_opening_prefix_cache, Mapping)
            or not isinstance(
                conditioned_opening_prefix_cache_binding_sha256, str
            )
            or len(conditioned_opening_prefix_cache_binding_sha256) != 64
        ):
            raise ValueError("R015 selected-filter opening cache is malformed.")

        maximum_step = steps[-1]
        lane_count = len(histories)
        opening = np.stack(
            tuple(
                np.asarray(history[0]["official_local_observation"])
                for history in histories
            )
        )
        member_input_observations = np.stack(
            tuple(
                np.stack(
                    tuple(
                        np.asarray(record["official_local_observation"])
                        for record in history[:maximum_step]
                    )
                )
                for history in histories
            )
        )
        next_observations = np.stack(
            tuple(
                np.stack(
                    tuple(
                        np.asarray(record["official_local_observation"])
                        for record in history[1 : maximum_step + 1]
                    )
                )
                for history in histories
            )
        )
        ego_actions = np.asarray(
            tuple(
                tuple(
                    OCV2_ACTION_INDEX[str(record["ego_action_history"][0])]
                    for record in history[1 : maximum_step + 1]
                )
                for history in histories
            ),
            dtype=np.int32,
        )
        rewards = np.asarray(
            tuple(
                tuple(
                    float(record["raw_team_reward_history"][0])
                    for record in history[1 : maximum_step + 1]
                )
                for history in histories
            ),
            dtype=np.float32,
        )
        boundaries = np.asarray(
            tuple(
                tuple(
                    bool(record["episode_boundaries"][0])
                    for record in history[1 : maximum_step + 1]
                )
                for history in histories
            ),
            dtype=np.bool_,
        )
        root_words = np.asarray(
            tuple(_r015_sha256_words(value) for value in keys),
            dtype=np.uint32,
        )

        def slice_opening_prefix(
            value: Mapping[str, Any],
            count: int,
        ) -> Mapping[str, Any]:
            accepted_counts = np.asarray(value["opening_accepted_counts"])
            if count <= 0 or accepted_counts.shape != (1, 4) or np.any(
                accepted_counts < count
            ):
                raise ValueError(
                    "R015 selected-filter cache cannot expose an incomplete prefix."
                )
            payload = dict(value)
            payload["environment_state"] = jax.tree_util.tree_map(
                lambda item: np.asarray(item)[:, :, :count, ...],
                value["environment_state"],
            )
            payload["observation"] = jax.tree_util.tree_map(
                lambda item: np.asarray(item)[:, :, :count, ...],
                value["observation"],
            )
            for field_name in (
                "snapshot_keys",
                "opening_accepted_source_indices",
                "lineage_indices",
                "last_predicted_response_tokens",
                "last_partner_actions",
                "last_recipe_outcomes",
                "last_conditional_successor_indices",
            ):
                payload[field_name] = np.asarray(value[field_name])[
                    :, :, :count, ...
                ]
            payload["partner_recurrent_states"] = tuple(
                jax.tree_util.tree_map(
                    lambda item: np.asarray(item)[:, :count, ...],
                    state,
                )
                for state in value["partner_recurrent_states"]
            )
            payload["prototype_masses"] = np.full(
                (1, 4), 0.25, dtype=np.float64
            )
            payload["within_prototype_weights"] = np.full(
                (1, 4, count), 1.0 / count, dtype=np.float64
            )
            payload["pre_resample_ess_fraction"] = np.ones(
                (1, 4), dtype=np.float64
            )
            payload["closed"] = np.zeros((1,), dtype=np.bool_)
            payload["opening_accepted_counts"] = np.full(
                (1, 4), count, dtype=np.int32
            )
            payload["opening_proposal_counts"] = np.asarray(
                payload["opening_accepted_source_indices"]
            )[..., -1] + 1
            payload["update_count"] = np.zeros((1,), dtype=np.int32)
            return payload

        cached_particles = 0
        cached_payloads: tuple[Mapping[str, Any], ...] | None = None
        if conditioned_opening_prefix_cache is not None:
            import hashlib

            entries = tuple(
                conditioned_opening_prefix_cache.get(key) for key in keys
            )
            present = tuple(entry for entry in entries if isinstance(entry, Mapping))
            if any(
                entry.get("binding_sha256")
                != conditioned_opening_prefix_cache_binding_sha256
                for entry in present
            ):
                raise ValueError(
                    "R015 selected-filter opening cache binding changed."
                )
            if any(
                isinstance(
                    entry.get("materialized_particles_per_prototype"), bool
                )
                or not isinstance(
                    entry.get("materialized_particles_per_prototype"), int
                )
                or not 0
                <= int(entry["materialized_particles_per_prototype"])
                <= 256
                for entry in present
            ):
                raise ValueError(
                    "R015 selected-filter opening cache has an invalid prefix length."
                )
            # Planning may consume only a complete selected-P prefix.  If any
            # lane is absent or incomplete, the whole batch deterministically
            # rebuilds from the registered roots instead of mixing cached and
            # fresh particle sequences.
            complete_cache = len(present) == lane_count and all(
                int(entry["materialized_particles_per_prototype"])
                >= int(particles_per_prototype)
                and isinstance(entry.get("payload"), Mapping)
                for entry in present
            )
            if complete_cache:
                cached_particles = int(particles_per_prototype)
                cached_payloads = tuple(
                    slice_opening_prefix(entry["payload"], cached_particles)
                    for entry in present
                )

            def opening_prefix_digest(value: Mapping[str, Any]) -> str:
                digest = hashlib.sha256()
                digest.update(R015_FILTER_KEY_CONTRACT_V2.encode("ascii"))
                digest.update(str(cached_particles).encode("ascii"))
                core = (
                    value["environment_state"],
                    value["observation"],
                    value["snapshot_keys"],
                    value["partner_recurrent_states"],
                    value["opening_accepted_source_indices"],
                    value["last_predicted_response_tokens"],
                )
                for leaf in jax.tree_util.tree_leaves(core):
                    array = np.ascontiguousarray(np.asarray(leaf))
                    digest.update(str(array.dtype).encode("ascii"))
                    digest.update(repr(tuple(array.shape)).encode("ascii"))
                    digest.update(array.tobytes(order="C"))
                return digest.hexdigest()

            if cached_payloads is not None:
                for entry, payload in zip(present, cached_payloads):
                    expected_digest = dict(
                        entry.get("prefix_sha256_by_particle_count", {})
                    ).get(str(cached_particles))
                    if not isinstance(expected_digest, str) or (
                        opening_prefix_digest(payload) != expected_digest
                    ):
                        raise ValueError(
                            "R015 selected-filter opening prefix digest changed."
                        )

        parent_slot_target = 4096
        lanes_per_batch = max(
            1,
            parent_slot_target // (4 * int(particles_per_prototype)),
        )
        member_ids = tuple(self.continuation_controller.member_ids)
        all_points: list[Mapping[str, Any]] = []
        compilation_count = 0
        compiled_call_count = 0
        total_wall_seconds = 0.0

        def slice_lane(value: Any, lane: int) -> Any:
            return jax.tree_util.tree_map(
                lambda item: np.ascontiguousarray(
                    np.asarray(item)[lane : lane + 1]
                ),
                value,
            )

        for batch_start in range(0, lane_count, lanes_per_batch):
            batch_stop = min(lane_count, batch_start + lanes_per_batch)
            batch_size = batch_stop - batch_start
            partner_actor_slots = batch_size * int(particles_per_prototype)
            initial_partner_states = tuple(
                self.policies[prototype_id].policy.initial_state(
                    partner_actor_slots
                )
                for prototype_id in self.prototype_ids
            )
            initial_member_states = tuple(
                self.policies[member_id].policy.initial_state(batch_size)
                for member_id in member_ids
            )
            if cached_payloads is None:
                batch_cached_state = None
            else:
                batch_cached_state = jax.tree_util.tree_map(
                    lambda *items: np.concatenate(
                        tuple(np.asarray(item) for item in items), axis=0
                    ),
                    *cached_payloads[batch_start:batch_stop],
                )

            def compiled_replay(
                opening_value: Any,
                member_observation_values: Any,
                next_observation_values: Any,
                action_values: Any,
                reward_values: Any,
                boundary_values: Any,
                word_values: Any,
                partner_states: Any,
                member_states: Any,
                cached_state: Any,
            ) -> Any:
                filter_value = self._initialize_online_filter_v2_device(
                    adapter=adapter,
                    opening_observations=opening_value,
                    root_key_words=word_values,
                    particles_per_prototype=int(particles_per_prototype),
                    initial_partner_states=partner_states,
                    cached_prefix_state=cached_state,
                    cached_particles_per_prototype=cached_particles,
                )
                carry = (filter_value, member_states)
                checkpoint_filters = []
                checkpoint_members = []
                previous_step = 0

                def scan_step(current: Any, step_input: Any) -> Any:
                    current_filter, current_members = current
                    (
                        step_index,
                        member_observation,
                        next_observation,
                        ego_action,
                        raw_reward,
                        boundary,
                    ) = step_input
                    episode_start = jnp.broadcast_to(
                        step_index == 1,
                        (batch_size,),
                    )
                    next_members = []
                    for member_id, member_state in zip(
                        member_ids, current_members
                    ):
                        next_member_state, _ = (
                            self._compiled_actor_boundary_with_start(
                                member_id,
                                member_state,
                                member_observation,
                                episode_start,
                            )
                        )
                        next_members.append(next_member_state)
                    advanced = self._advance_online_filter_v2_device(
                        adapter=adapter,
                        filter_state=current_filter,
                        official_step_records={
                            "official_local_observation": next_observation,
                            "ego_action_index": ego_action,
                            "raw_team_reward": raw_reward,
                            "episode_boundary": boundary,
                        },
                        environment_step=step_index,
                        resampling_timing=resampling_timing,
                        return_diagnostics=False,
                    )
                    return advanced["filter_state"], tuple(next_members)

                for consultation_step in steps:
                    segment = slice(previous_step, consultation_step)
                    segment_inputs = (
                        jnp.arange(
                            previous_step + 1,
                            consultation_step + 1,
                            dtype=jnp.int32,
                        ),
                        jnp.swapaxes(
                            member_observation_values[:, segment], 0, 1
                        ),
                        jnp.swapaxes(
                            next_observation_values[:, segment], 0, 1
                        ),
                        jnp.swapaxes(action_values[:, segment], 0, 1),
                        jnp.swapaxes(reward_values[:, segment], 0, 1),
                        jnp.swapaxes(boundary_values[:, segment], 0, 1),
                    )
                    carry = jax.lax.scan(
                        lambda value, item: (scan_step(value, item), None),
                        carry,
                        segment_inputs,
                    )[0]
                    checkpoint_filters.append(carry[0])
                    checkpoint_members.append(carry[1])
                    previous_step = consultation_step
                return tuple(checkpoint_filters), tuple(checkpoint_members)

            cache_key = (
                "selected_consultation_replay_v2",
                batch_size,
                int(particles_per_prototype),
                str(resampling_timing),
                steps,
                id(adapter.env),
                cached_particles,
            )
            compiled = self._filter_scan_cache.get(cache_key)
            if compiled is None:
                compiled = jax.jit(compiled_replay)
                self._filter_scan_cache[cache_key] = compiled
                compilation_count += 1
            started = time.perf_counter()
            device_filters, device_members = compiled(
                jnp.asarray(opening[batch_start:batch_stop]),
                jnp.asarray(
                    member_input_observations[batch_start:batch_stop]
                ),
                jnp.asarray(next_observations[batch_start:batch_stop]),
                jnp.asarray(ego_actions[batch_start:batch_stop]),
                jnp.asarray(rewards[batch_start:batch_stop]),
                jnp.asarray(boundaries[batch_start:batch_stop]),
                jnp.asarray(root_words[batch_start:batch_stop]),
                initial_partner_states,
                initial_member_states,
                batch_cached_state,
            )
            host_filters, host_members = jax.device_get(
                (device_filters, device_members)
            )
            total_wall_seconds += time.perf_counter() - started
            compiled_call_count += 1
            for local_lane in range(batch_size):
                global_lane = batch_start + local_lane
                for point_index, environment_step in enumerate(steps):
                    point_filter = host_filters[point_index]
                    point_members = host_members[point_index]
                    all_points.append(
                        {
                            "lane_index": global_lane,
                            "environment_step": environment_step,
                            "closed_for_zero_support": bool(
                                np.asarray(point_filter["closed"])[local_lane]
                            ),
                            "device_filter_state": slice_lane(
                                point_filter,
                                local_lane,
                            ),
                            "continuation_states_by_member_id": {
                                member_id: slice_lane(
                                    point_members[member_index],
                                    local_lane,
                                )
                                for member_index, member_id in enumerate(
                                    member_ids
                                )
                            },
                        }
                    )

        particle_transitions = (
            lane_count
            * maximum_step
            * 4
            * int(particles_per_prototype)
            * len(OFFICIAL_ACTION_ORDER)
        )
        return {
            "consultation_states": tuple(all_points),
            "device_execution": {
                "filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
                "filter_key_contract": R015_FILTER_KEY_CONTRACT_V2,
                "device_execution_id": R015_FILTER_DEVICE_EXECUTION_ID_V2,
                "compiled_batch_calls": compiled_call_count,
                "jit_compilations": compilation_count,
                "active_lane_count": lane_count,
                "parent_particle_slot_target_per_batch": parent_slot_target,
                "lanes_per_full_batch": lanes_per_batch,
                "consultation_point_count": len(all_points),
                "filter_particle_environment_transitions": (
                    particle_transitions
                ),
                "continuation_member_forwards": (
                    lane_count * maximum_step * len(member_ids)
                ),
                "wall_seconds": total_wall_seconds,
                "true_particle_transitions_per_second": (
                    particle_transitions / total_wall_seconds
                    if total_wall_seconds > 0.0
                    else None
                ),
                "host_sync_inside_environment_loop": False,
                "checkpoint_payload_host_serialization": (
                    "contiguous_numpy_lane_slice_v1"
                ),
                "conditioned_opening_prefix_cache_binding_sha256": (
                    conditioned_opening_prefix_cache_binding_sha256
                ),
                "conditioned_opening_prefix_particles_used_per_prototype": (
                    cached_particles
                ),
            },
        }

    def run_offline_filter_scan_v1_diagnostic(
        self,
        *,
        adapter: Any,
        official_histories: Sequence[Sequence[Mapping[str, Any]]],
        initialization_keys: Sequence[str],
        particles_per_prototype: int,
        resampling_timing: str,
        return_particle_weight_trace: bool = False,
        environment_step_limit: int = 400,
        return_state_diagnostics: bool = False,
    ) -> Mapping[str, Any]:
        """只为归档诊断重放第一版过滤器，不得进入第二版执行链。

        该旧实现会推进五个延续成员，并在宿主构造逐粒子逐步随机数；保留它只为读取
        2026-07-17 已归档失败的诊断语义。设计、试点和正式运行只允许调用
        ``run_offline_filter_scan_v2``。
        """

        if not official_histories or len(official_histories) != len(
            initialization_keys
        ):
            raise ValueError("R015 device filter inputs have different lane counts.")
        if particles_per_prototype <= 0:
            raise ValueError("R015 device filter particle count must be positive.")
        if resampling_timing not in {
            "adaptive_ess_below_half_v1",
            "every_environment_step_v1",
        }:
            raise ValueError("R015 device filter uses an unregistered timing.")
        if (
            isinstance(environment_step_limit, bool)
            or not isinstance(environment_step_limit, int)
            or not 1 <= environment_step_limit <= 400
        ):
            raise ValueError("R015 device filter step limit must lie in 1,...,400.")
        if return_state_diagnostics and not return_particle_weight_trace:
            raise ValueError("R015 state diagnostics require the equivalence trace.")

        from experiments.overcooked_v2.path_c_r015_full_horizon import (
            R015_FILTER_KEY_CONTRACT,
        )
        from experiments.overcooked_v2.path_c_r015_runtime import OCV2_ACTION_INDEX

        import jax
        import jax.numpy as jnp

        lane_count = len(official_histories)
        prototype_count = len(self.prototype_ids)
        particle_count = prototype_count * particles_per_prototype
        if prototype_count != 4:
            raise ValueError("R015 device filter requires four prototypes.")
        histories = tuple(tuple(value) for value in official_histories)
        if any(len(value) != 401 for value in histories):
            raise ValueError("R015 device filter requires one opening record and 400 steps.")
        step_count = int(environment_step_limit)

        observations = np.stack(
            [
                np.stack(
                    [
                        np.asarray(record["official_local_observation"])
                        for record in history[: step_count + 1]
                    ]
                )
                for history in histories
            ]
        )
        ego_actions = np.asarray(
            [
                [
                    OCV2_ACTION_INDEX[str(record["ego_action_history"][0])]
                    for record in history[1 : step_count + 1]
                ]
                for history in histories
            ],
            dtype=np.int32,
        )
        raw_rewards = np.asarray(
            [
                [
                    float(record["raw_team_reward_history"][0])
                    for record in history[1 : step_count + 1]
                ]
                for history in histories
            ],
            dtype=np.float32,
        )
        boundaries = np.asarray(
            [
                [
                    bool(record["episode_boundaries"][0])
                    for record in history[1 : step_count + 1]
                ]
                for history in histories
            ],
            dtype=np.bool_,
        )

        initial_execution_seeds = np.empty(
            (lane_count, prototype_count, particles_per_prototype), dtype=np.uint32
        )
        environment_seeds = np.empty(
            (step_count, lane_count, prototype_count, particles_per_prototype),
            dtype=np.uint32,
        )
        partner_seeds = np.empty_like(environment_seeds)
        resampling_phases = np.empty(
            (step_count, lane_count, prototype_count), dtype=np.float64
        )
        for lane, initialization_key in enumerate(initialization_keys):
            if not isinstance(initialization_key, str) or len(initialization_key) != 64:
                raise ValueError("R015 device filter initialization key must be SHA-256.")
            update_keys = tuple(
                derive_controller_key(
                    initialization_key,
                    "design_filter_update",
                    environment_step,
                )
                for environment_step in range(1, step_count + 1)
            )
            for prototype_index, prototype_id in enumerate(self.prototype_ids):
                for local_index in range(particles_per_prototype):
                    global_index = prototype_index * particles_per_prototype + local_index
                    particle_key = derive_controller_key(
                        initialization_key,
                        R015_FILTER_KEY_CONTRACT,
                        prototype_id,
                        local_index,
                    )
                    initial_execution_seeds[lane, prototype_index, local_index] = (
                        derive_ocv2_execution_seed(int(particle_key[:16], 16))
                    )
                    for environment_step, update_key in enumerate(
                        update_keys, start=1
                    ):
                        transition_key = derive_controller_key(
                            update_key,
                            "filter_transition",
                            prototype_id,
                            global_index,
                        )
                        environment_key = derive_controller_key(
                            transition_key, "particle_environment"
                        )
                        partner_key = derive_controller_key(
                            transition_key, "partner_action"
                        )
                        environment_seeds[
                            environment_step - 1, lane, prototype_index, local_index
                        ] = int(environment_key[:16], 16) & 0xFFFFFFFF
                        partner_seeds[
                            environment_step - 1, lane, prototype_index, local_index
                        ] = int(partner_key[:16], 16) & 0xFFFFFFFF
                for environment_step, update_key in enumerate(
                    update_keys, start=1
                ):
                    resample_key = derive_controller_key(
                        update_key,
                        "systematic_per_prototype_v1",
                        prototype_id,
                    )
                    resampling_phases[
                        environment_step - 1, lane, prototype_index
                    ] = int(resample_key[:16], 16) / float(16**16)

        prototype_indices = np.tile(
            np.repeat(np.arange(prototype_count, dtype=np.int32), particles_per_prototype),
            lane_count,
        )
        total_slots = lane_count * particle_count
        partner_initial_candidates = tuple(
            self.policies[prototype_id].policy.initial_state(total_slots)
            for prototype_id in self.prototype_ids
        )
        partner_params = tuple(
            self.policies[prototype_id].policy.params for prototype_id in self.prototype_ids
        )
        continuation_member_ids = tuple(
            self.continuation_controller.member_ids
        )
        continuation_params = tuple(
            self.policies[member_id].policy.params
            for member_id in continuation_member_ids
        )
        initial_continuation_states = tuple(
            self.policies[member_id].policy.initial_state(total_slots)
            for member_id in continuation_member_ids
        )

        def choose_rows(stacked: Any, indices: Any) -> Any:
            return jax.tree_util.tree_map(
                lambda value: value[
                    indices, jnp.arange(indices.shape[0], dtype=jnp.int32)
                ],
                stacked,
            )

        initial_partner_state = choose_rows(
            jax.tree_util.tree_map(
                lambda *items: jnp.stack(tuple(jnp.asarray(item) for item in items)),
                *partner_initial_candidates,
            ),
            jnp.asarray(prototype_indices),
        )

        def compiled_filter_scan(
            reset_seeds: Any,
            expected_observations: Any,
            forced_ego_actions: Any,
            expected_rewards: Any,
            expected_boundaries: Any,
            all_environment_seeds: Any,
            all_partner_seeds: Any,
            all_resampling_phases: Any,
            initial_partner: Any,
            initial_members: Any,
            initial_weight_numerator: Any,
            initial_ess: Any,
            zero_by_prototype: Any,
            one_by_prototype: Any,
            half_by_prototype: Any,
            particle_count_denominator: Any,
            systematic_offsets: Any,
        ) -> Any:
            flat_reset_seeds = reset_seeds.reshape((-1,))
            reset_roots = jax.vmap(jax.random.PRNGKey)(flat_reset_seeds)
            reset_pairs = jax.vmap(jax.random.split)(reset_roots)
            initial_observation, initial_environment_state = jax.vmap(
                adapter.env.reset
            )(reset_pairs[:, 1])
            slot_prototypes = jnp.asarray(prototype_indices)

            def actor_apply(params: Mapping[str, Any], state: Any, obs: Any):
                return self._compiled_actor_boundary(
                    params,
                    state,
                    obs,
                )

            def left_sum_last(value: Any) -> Any:
                """Match Python ``sum`` order along the final axis."""

                moved = jnp.moveaxis(value, -1, 0)

                def add_one(total: Any, item: Any):
                    next_total = total + item
                    return next_total, next_total

                total, _ = jax.lax.scan(
                    add_one,
                    moved[0] - moved[0],
                    moved,
                )
                return total

            def left_cumulative_last(value: Any) -> Any:
                moved = jnp.moveaxis(value, -1, 0)

                def add_one(total: Any, item: Any):
                    next_total = total + item
                    return next_total, next_total

                _, cumulative = jax.lax.scan(
                    add_one,
                    moved[0] - moved[0],
                    moved,
                )
                return jnp.moveaxis(cumulative, 0, -1)

            opening_expected = jnp.repeat(
                expected_observations[:, 0, ...], particle_count, axis=0
            )
            opening_match = jnp.all(
                initial_observation["agent_1"] == opening_expected,
                axis=tuple(range(1, initial_observation["agent_1"].ndim)),
            ).reshape((lane_count, prototype_count, particles_per_prototype))
            compatible_counts = jnp.sum(opening_match, axis=-1)
            opening_closed = jnp.any(compatible_counts <= 0, axis=-1)
            with jax.experimental.enable_x64():
                compatible_counts_float = compatible_counts.astype(jnp.float64)
                compatible_counts_column = compatible_counts_float[..., None]
                weights = jnp.where(
                    opening_match,
                    initial_weight_numerator
                    / jnp.maximum(
                        compatible_counts_column,
                        one_by_prototype[..., None],
                    ),
                    initial_weight_numerator - initial_weight_numerator,
                )
                last_ess = initial_ess

            def as_grid(value: Any) -> Any:
                return jax.tree_util.tree_map(
                    lambda item: item.reshape(
                        (lane_count, prototype_count, particles_per_prototype)
                        + item.shape[1:]
                    ),
                    value,
                )

            def as_flat(value: Any) -> Any:
                return jax.tree_util.tree_map(
                    lambda item: item.reshape((total_slots,) + item.shape[3:]),
                    value,
                )

            def gather_slots(value: Any, indices: Any) -> Any:
                def gather_leaf(item: Any) -> Any:
                    suffix = (1,) * (item.ndim - 3)
                    expanded = indices.reshape(indices.shape + suffix)
                    expanded = jnp.broadcast_to(
                        expanded, indices.shape + item.shape[3:]
                    )
                    return jnp.take_along_axis(item, expanded, axis=2)

                return jax.tree_util.tree_map(gather_leaf, value)

            def select_due(original: Any, resampled: Any, due: Any) -> Any:
                return jax.tree_util.tree_map(
                    lambda left, right: jnp.where(
                        due.reshape(due.shape + (1,) * (left.ndim - 2)),
                        right,
                        left,
                    ),
                    original,
                    resampled,
                )

            def freeze_closed_lanes(
                original: Any,
                candidate: Any,
                closed: Any,
            ) -> Any:
                """关闭后的车道仍占据固定批宽，但不再改变隐藏执行状态。"""

                closed_by_slot = jnp.repeat(closed, particle_count)
                return jax.tree_util.tree_map(
                    lambda left, right: jnp.where(
                        closed_by_slot.reshape(
                            closed_by_slot.shape
                            + (1,) * (left.ndim - closed_by_slot.ndim)
                        ),
                        left,
                        right,
                    ),
                    original,
                    candidate,
                )

            def update_weight_state(
                current_weights: Any,
                compatible: Any,
                closed: Any,
                previous_ess: Any,
                resampling_phase: Any,
            ) -> Any:
                """Reproduce the existing Python-float filter arithmetic only."""

                with jax.experimental.enable_x64():
                    expanded_weights = current_weights * compatible.astype(
                        current_weights.dtype
                    )
                    prototype_mass = left_sum_last(expanded_weights)
                    squared_mass = left_sum_last(
                        expanded_weights * expanded_weights
                    )
                    ess = jnp.where(
                        squared_mass > zero_by_prototype,
                        prototype_mass * prototype_mass / squared_mass,
                        zero_by_prototype,
                    ) / particle_count_denominator
                    just_closed = jnp.any(
                        prototype_mass <= zero_by_prototype,
                        axis=-1,
                    )
                    next_closed = closed | just_closed
                    total_mass = left_sum_last(prototype_mass)
                    safe_total = jnp.where(
                        total_mass > zero_by_prototype[:, 0],
                        total_mass,
                        one_by_prototype[:, 0],
                    )
                    normalized = expanded_weights / safe_total[:, None, None]
                    if resampling_timing == "every_environment_step_v1":
                        due = jnp.ones_like(prototype_mass, dtype=jnp.bool_)
                    else:
                        due = ess < half_by_prototype
                    normalized_within_prototype = expanded_weights / jnp.where(
                        prototype_mass[..., None]
                        > zero_by_prototype[..., None],
                        prototype_mass[..., None],
                        one_by_prototype[..., None],
                    )
                    normalized_cumulative = left_cumulative_last(
                        normalized_within_prototype
                    )
                    positions = (
                        resampling_phase[..., None]
                        / particle_count_denominator[..., None]
                        + systematic_offsets
                    )
                    resample_indices = jnp.sum(
                        normalized_cumulative[..., :, None]
                        < positions[..., None, :],
                        axis=-2,
                        dtype=jnp.int32,
                    )
                    resample_indices = jnp.minimum(
                        resample_indices,
                        particles_per_prototype - 1,
                    )
                    resampled_weights = jnp.broadcast_to(
                        prototype_mass[..., None]
                        / safe_total[:, None, None]
                        / particle_count_denominator[..., None],
                        normalized.shape,
                    )
                    next_weights = jnp.where(
                        due[..., None], resampled_weights, normalized
                    )
                    unusable = closed | just_closed
                    next_weights = jnp.where(
                        unusable[:, None, None], current_weights, next_weights
                    )
                    next_ess_output = jnp.where(
                        unusable[:, None], previous_ess, ess
                    )
                    posterior = left_sum_last(next_weights)
                return (
                    next_weights,
                    next_closed,
                    next_ess_output,
                    posterior,
                    due,
                    resample_indices,
                )

            def scan_step(carry: Any, step_input: Any):
                (
                    environment_state,
                    observation,
                    partner_state,
                    continuation_states,
                    current_weights,
                    closed,
                    previous_ess,
                ) = carry
                (
                    step_observation,
                    forced_action,
                    step_reward,
                    step_boundary,
                    environment_seed,
                    partner_seed,
                    resampling_phase,
                ) = step_input

                partner_candidates = tuple(
                    actor_apply(
                        params,
                        partner_state,
                        observation["agent_0"],
                    )
                    for params in partner_params
                )
                partner_state_stack = jax.tree_util.tree_map(
                    lambda *items: jnp.stack(items),
                    *(item[0] for item in partner_candidates),
                )
                partner_logits_stack = jnp.stack(
                    tuple(item[1] for item in partner_candidates)
                )
                next_partner_state = choose_rows(
                    partner_state_stack, slot_prototypes
                )
                partner_logits = choose_rows(
                    partner_logits_stack, slot_prototypes
                )
                flat_partner_seeds = partner_seed.reshape((-1,))
                partner_keys = jax.vmap(jax.random.PRNGKey)(flat_partner_seeds)
                partner_actions = jax.vmap(jax.random.categorical)(
                    partner_keys, partner_logits
                )
                next_continuation_states = []
                continuation_logit_checksums = []
                for params, member_state in zip(
                    continuation_params,
                    continuation_states,
                ):
                    next_member_state, member_logits = actor_apply(
                        params,
                        member_state,
                        observation["agent_1"],
                    )
                    next_continuation_states.append(next_member_state)
                    continuation_logit_checksums.append(
                        jnp.sum(member_logits, dtype=jnp.float32)
                    )
                ego_action_grid = jnp.broadcast_to(
                    forced_action[:, None, None],
                    (lane_count, prototype_count, particles_per_prototype),
                )
                ego_actions = ego_action_grid.reshape((-1,))
                flat_environment_seeds = environment_seed.reshape((-1,))
                environment_roots = jax.vmap(jax.random.PRNGKey)(
                    flat_environment_seeds
                )
                environment_pairs = jax.vmap(jax.random.split)(environment_roots)
                next_observation, next_environment_state, rewards, dones, _ = jax.vmap(
                    adapter.env.step_env
                )(
                    environment_pairs[:, 1],
                    environment_state,
                    {"agent_0": partner_actions, "agent_1": ego_actions},
                )

                expected_observation = jnp.repeat(
                    step_observation, particle_count, axis=0
                )
                observation_match = jnp.all(
                    next_observation["agent_1"] == expected_observation,
                    axis=tuple(range(1, next_observation["agent_1"].ndim)),
                )
                reward_match = jnp.abs(
                    rewards["agent_0"]
                    - jnp.repeat(step_reward, particle_count, axis=0)
                ) <= jnp.asarray(1.0e-12, dtype=rewards["agent_0"].dtype)
                boundary_match = dones["__all__"] == jnp.repeat(
                    step_boundary, particle_count, axis=0
                )
                compatible = (
                    observation_match & reward_match & boundary_match
                ).reshape((lane_count, prototype_count, particles_per_prototype))
                (
                    next_weights,
                    next_closed,
                    next_ess_output,
                    posterior,
                    due,
                    resample_indices,
                ) = update_weight_state(
                    current_weights,
                    compatible,
                    closed,
                    previous_ess,
                    resampling_phase,
                )
                effective_due = due & ~next_closed[:, None]
                environment_grid = as_grid(next_environment_state)
                observation_grid = as_grid(next_observation)
                partner_grid = as_grid(next_partner_state)
                selected_environment = select_due(
                    environment_grid,
                    gather_slots(environment_grid, resample_indices),
                    effective_due,
                )
                selected_observation = select_due(
                    observation_grid,
                    gather_slots(observation_grid, resample_indices),
                    effective_due,
                )
                selected_partner = select_due(
                    partner_grid,
                    gather_slots(partner_grid, resample_indices),
                    effective_due,
                )
                selected_continuation_states = []
                for next_member_state in next_continuation_states:
                    member_grid = as_grid(next_member_state)
                    selected_continuation_states.append(
                        as_flat(
                            select_due(
                                member_grid,
                                gather_slots(member_grid, resample_indices),
                                effective_due,
                            )
                        )
                    )
                carried_environment = freeze_closed_lanes(
                    environment_state,
                    as_flat(selected_environment),
                    next_closed,
                )
                carried_observation = freeze_closed_lanes(
                    observation,
                    as_flat(selected_observation),
                    next_closed,
                )
                carried_partner = freeze_closed_lanes(
                    partner_state,
                    as_flat(selected_partner),
                    next_closed,
                )
                carried_continuation_states = tuple(
                    freeze_closed_lanes(
                        previous_member_state,
                        selected_member_state,
                        next_closed,
                    )
                    for previous_member_state, selected_member_state in zip(
                        continuation_states,
                        selected_continuation_states,
                    )
                )
                next_carry = (
                    carried_environment,
                    carried_observation,
                    carried_partner,
                    carried_continuation_states,
                    next_weights,
                    next_closed,
                    next_ess_output,
                )
                trace = {
                    "posterior": posterior,
                    "ess": next_ess_output,
                    "closed": next_closed,
                    "continuation_logit_checksums": jnp.stack(
                        tuple(continuation_logit_checksums)
                    ),
                }
                if return_particle_weight_trace:
                    trace["particle_weights"] = next_weights
                    trace["partner_actions"] = partner_actions.reshape(
                        (
                            lane_count,
                            prototype_count,
                            particles_per_prototype,
                        )
                    )
                    trace["resampling_due"] = effective_due
                    direct_source_indices = jnp.broadcast_to(
                        jnp.arange(
                            particles_per_prototype,
                            dtype=jnp.int32,
                        ).reshape((1, 1, particles_per_prototype)),
                        resample_indices.shape,
                    )
                    trace["selected_source_indices"] = jnp.where(
                        next_closed[:, None, None],
                        jnp.asarray(-1, dtype=jnp.int32),
                        jnp.where(
                            effective_due[..., None],
                            resample_indices,
                            direct_source_indices,
                        ),
                    )
                    if return_state_diagnostics:
                        trace["partner_recurrent_state_before"] = partner_state
                        trace["agent_0_observation_before"] = observation["agent_0"]
                        trace["partner_recurrent_state"] = carried_partner
                        trace["agent_0_observation"] = carried_observation["agent_0"]
                return next_carry, trace

            scan_inputs = (
                jnp.swapaxes(expected_observations[:, 1:, ...], 0, 1),
                jnp.swapaxes(forced_ego_actions, 0, 1),
                jnp.swapaxes(expected_rewards, 0, 1),
                jnp.swapaxes(expected_boundaries, 0, 1),
                all_environment_seeds,
                all_partner_seeds,
                all_resampling_phases,
            )
            _, trace = jax.lax.scan(
                scan_step,
                (
                    initial_environment_state,
                    initial_observation,
                    initial_partner,
                    initial_members,
                    weights,
                    opening_closed,
                    last_ess,
                ),
                scan_inputs,
            )
            return trace

        cache_key = (
            lane_count,
            particles_per_prototype,
            resampling_timing,
            bool(return_particle_weight_trace),
            step_count,
            bool(return_state_diagnostics),
        )
        compiled = self._filter_scan_cache.get(cache_key)
        compilation_count = 0
        if compiled is None:
            compiled = jax.jit(compiled_filter_scan)
            self._filter_scan_cache[cache_key] = compiled
            compilation_count = 1
        started = time.perf_counter()
        with jax.experimental.enable_x64():
            resampling_phase_array = jnp.asarray(resampling_phases)
            initial_weight_numerator = jnp.asarray(
                np.full(
                    (lane_count, prototype_count, 1),
                    0.25,
                    dtype=np.float64,
                )
            )
            initial_ess = jnp.asarray(
                np.ones(
                    (lane_count, prototype_count),
                    dtype=np.float64,
                )
            )
            zero_by_prototype = jnp.asarray(
                np.zeros(
                    (lane_count, prototype_count),
                    dtype=np.float64,
                )
            )
            one_by_prototype = jnp.asarray(
                np.ones(
                    (lane_count, prototype_count),
                    dtype=np.float64,
                )
            )
            half_by_prototype = jnp.asarray(
                np.full(
                    (lane_count, prototype_count),
                    0.5,
                    dtype=np.float64,
                )
            )
            particle_count_denominator = jnp.asarray(
                np.full(
                    (lane_count, prototype_count),
                    particles_per_prototype,
                    dtype=np.float64,
                )
            )
            systematic_offsets = jnp.asarray(
                (
                    np.arange(particles_per_prototype, dtype=np.float64)
                    / np.float64(particles_per_prototype)
                ).reshape((1, 1, particles_per_prototype))
            )
        trace = compiled(
            jnp.asarray(initial_execution_seeds),
            jnp.asarray(observations),
            jnp.asarray(ego_actions),
            jnp.asarray(raw_rewards),
            jnp.asarray(boundaries),
            jnp.asarray(environment_seeds),
            jnp.asarray(partner_seeds),
            resampling_phase_array,
            initial_partner_state,
            initial_continuation_states,
            initial_weight_numerator,
            initial_ess,
            zero_by_prototype,
            one_by_prototype,
            half_by_prototype,
            particle_count_denominator,
            systematic_offsets,
        )
        host_trace = jax.device_get(trace)
        wall_seconds = time.perf_counter() - started
        true_transitions = step_count * lane_count * particle_count
        return {
            "trace": host_trace,
            "throughput": {
                "true_particle_environment_transitions": true_transitions,
                "compiled_batch_calls": 1,
                "jit_compilations": compilation_count,
                "active_lane_count": lane_count,
                "active_particle_batch_width": lane_count * particle_count,
                "policy_actor_lane_batch_width": total_slots,
                "wall_seconds": wall_seconds,
                "true_transitions_per_second": (
                    true_transitions / wall_seconds if wall_seconds > 0.0 else None
                ),
                "host_sync_inside_environment_loop": False,
            },
        }

    def _executor(self) -> FullHorizonBranchExecutor:
        if self.full_horizon_executor is None:
            raise ValueError("R015 full-horizon executor is not configured.")
        if self.full_horizon_executor.continuation_controller_id != (
            self.continuation_controller_id
        ):
            raise ValueError("R015 planning and execution use different C(q,x).")
        return self.full_horizon_executor

    def candidate_status(
        self,
        *,
        history: OfficialHistoryV1,
        script: ProbeScriptV1,
        environment_step: int,
    ) -> tuple[bool, bool]:
        return self._executor().candidate_status(
            history=history,
            script=script,
            environment_step=environment_step,
        )

    def rollout_base(
        self,
        *,
        particle: HiddenStateParticleV1,
        belief: StratifiedParticleBeliefV1,
        history: OfficialHistoryV1,
        branch_key: str,
        remaining_steps: int,
    ) -> FullHorizonRolloutV1:
        return self._executor().rollout_base(
            particle=particle,
            belief=belief,
            history=history,
            branch_key=branch_key,
            remaining_steps=remaining_steps,
        )

    def rollout_probe_pair(
        self,
        *,
        particle: HiddenStateParticleV1,
        belief: StratifiedParticleBeliefV1,
        history: OfficialHistoryV1,
        script: ProbeScriptV1,
        branch_key: str,
        remaining_steps: int,
    ) -> PairedProbeRolloutV1:
        return self._executor().rollout_probe_pair(
            particle=particle,
            belief=belief,
            history=history,
            script=script,
            branch_key=branch_key,
            remaining_steps=remaining_steps,
        )

    def rollout_planning_batch(
        self,
        *,
        samples: Sequence[PlanningBranchSampleV1],
        belief: StratifiedParticleBeliefV1,
        history: OfficialHistoryV1,
        scripts: Sequence[ProbeScriptV1],
        remaining_steps: int,
    ) -> PlanningBatchRolloutsV1:
        return self._executor().rollout_planning_batch(
            samples=samples,
            belief=belief,
            history=history,
            scripts=scripts,
            remaining_steps=remaining_steps,
        )


def build_official_r015_production_backend(
    config: Mapping[str, Any],
    support_spec: R015PartnerSupportSpec,
    *,
    config_base_path: str | Path,
) -> OCV2R015ProductionBackendV1:
    """从已核验的四伙伴和 seed-100 主体构建 R015 生产后端。"""

    support = config.get("support")
    raw_ego = support.get("ego_candidate") if isinstance(support, Mapping) else None
    if not isinstance(raw_ego, Mapping):
        raise ValueError("R015 production config requires the registered ego candidate.")
    ego_candidate = R015CandidateSpec.from_mapping(
        _resolve_r015_candidate_artifact_paths(
            raw_ego,
            config_base_path=config_base_path,
        )
    )
    policies: dict[str, _LoadedPolicy] = {}
    checkpoint_identities: dict[str, Mapping[str, Any]] = {}
    for candidate in support_spec.candidates:
        loaded = _load_r015_policy(candidate)
        policies[candidate.candidate_id] = loaded
        checkpoint_identities[candidate.candidate_id] = (
            _validated_checkpoint_identity(candidate, loaded)
        )
    if ego_candidate.candidate_id in policies:
        raise ValueError("R015 ego baseline duplicates a partner prototype.")
    ego_loaded = _load_r015_policy(ego_candidate)
    policies[ego_candidate.candidate_id] = ego_loaded
    checkpoint_identities[ego_candidate.candidate_id] = (
        _validated_checkpoint_identity(ego_candidate, ego_loaded)
    )
    backend = OCV2R015ProductionBackendV1(
        policies=policies,
        prototype_ids=tuple(
            candidate.candidate_id for candidate in support_spec.candidates
        ),
        baseline_member_id=ego_candidate.candidate_id,
        checkpoint_identities=checkpoint_identities,
    )
    injected = config.get("_r015_full_horizon_executor")
    if injected is None:
        from experiments.overcooked_v2.path_c_r015_full_horizon import (
            build_r015_full_horizon_executor,
        )

        injected = build_r015_full_horizon_executor(config, backend)
    if getattr(injected, "continuation_controller_id", None) != (
        R015_CONTINUATION_CONTROLLER_ID
    ):
        raise ValueError("R015 production executor changed C(q,x).")
    backend.full_horizon_executor = injected
    return backend


__all__ = [
    "OCV2R015ProductionBackendV1",
    "build_official_r015_production_backend",
]
