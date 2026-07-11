from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.overcooked_v2.partner_pool import option_distribution
from experiments.overcooked_v2.path_c_audit_battery import ProbeScriptV1
from experiments.overcooked_v2.path_c_belief_audit import (
    FullHiddenStateV1,
    NamedRNGStreamsV1,
    OuterReplicaDrawV1,
    RNGKeyScheduleV1,
    fork_from_snapshot,
    replay_from_snapshot,
)
from experiments.overcooked_v2.path_c_ocv2_audit_bridge import (
    OCV2_AUDIT_BRIDGE_VERSION,
    OCV2_ROLLOUT_HOOK_CONTRACT_VERSION,
    CanonicalJSONStateCodecV1,
    EncodedOCV2AuditCaptureInputsV1,
    ExactPartnerPosteriorBridgeV1,
    ExactRegisteredPartnerKernelV1,
    FrozenPartnerRegistryV1,
    JaxPytreeStateCodecV1,
    NativeOCV2AdapterRolloutHooks,
    OCV2AuditContentAddressedStoreV1,
    OCV2AuditHarvestRecordV1,
    OCV2AuditRolloutTraceV1,
    OCV2SnapshotForkRunner,
    PartnerEvidenceObservationV1,
    PartnerRuntimeBranchV1,
    RegisteredPartnerStateCodecV1,
)
from experiments.overcooked_v2.path_c_sequence import EgoEvidenceSpecV1


def _rng_schedule() -> RNGKeyScheduleV1:
    return RNGKeyScheduleV1.from_original_seeds(
        manifest_seed=101,
        jax=11,
        numpy=12,
        python=13,
        torch=14,
    )


def _probe() -> ProbeScriptV1:
    return ProbeScriptV1(
        script_id="probe-0",
        option_ids=(0,),
        sampling_probability=1.0,
    )


def _evidence_spec() -> EgoEvidenceSpecV1:
    return EgoEvidenceSpecV1(
        observation_dim=1,
        num_primitive_actions=2,
        num_options=1,
        max_primitive_steps_per_decision=1,
        progress_event_dim=1,
        observation_schema="test-public-observation-v1",
        primitive_action_names=("stay", "interact"),
        option_names=("noop",),
        progress_event_names=("delivery",),
    )


def _evidence_observation() -> PartnerEvidenceObservationV1:
    spec = _evidence_spec()
    row = spec.encode_decision(
        observation=[0.0],
        ego_primitive_actions=[0],
        partner_primitive_actions=[1],
        ego_option_id=0,
        duration=1,
        reward=0.0,
        progress_events=[0.0],
        valid_actions=[True],
        terminated=False,
        truncated=False,
    )
    return PartnerEvidenceObservationV1.from_evidence_row(
        evidence_spec=spec,
        evidence_row=row.numpy(),
        option_choice_public_state_bytes=b"public-state",
        decision_public_state_bytes=b"public-state",
    )


def _capture() -> EncodedOCV2AuditCaptureInputsV1:
    codec = CanonicalJSONStateCodecV1()
    return EncodedOCV2AuditCaptureInputsV1.from_runtime(
        unit_id="audit-unit-0",
        episode_uid="audit-episode-0",
        decision_index=3,
        audit_role="locked_audit",
        harvest_run_id="audit-harvest-run-0",
        harvest_policy_id="registered-harvest-policy",
        inclusion_probability=0.5,
        env_state={"timestep": 3},
        raw_observation={"agent_0": [1.0], "agent_1": [2.0]},
        partner_state={"theta_id": "partner-a", "elapsed": 1},
        ego_state={"current_option": None},
        env_state_codec=codec,
        observation_codec=codec,
        partner_state_codec=codec,
        ego_state_codec=codec,
        rng_key_schedule=_rng_schedule(),
    )


def test_jax_pytree_codec_round_trip_is_bound_to_template_structure():
    template = {
        "grid": np.asarray([[1, 2], [3, 4]], dtype=np.int32),
        "agents": (
            np.asarray([0, 1], dtype=np.int16),
            np.asarray([True, False], dtype=bool),
        ),
    }
    codec = JaxPytreeStateCodecV1(template)
    payload = codec.encode(template)
    restored = codec.decode(payload)
    assert np.array_equal(restored["grid"], template["grid"])
    assert np.array_equal(restored["agents"][0], template["agents"][0])
    with pytest.raises(ValueError, match="pytree definition"):
        codec.encode({"different": np.asarray([1], dtype=np.int32)})


def test_registered_partner_codec_restores_embedded_public_state_pytree():
    public_state = {"counter": np.asarray([3], dtype=np.int32)}
    codec = RegisteredPartnerStateCodecV1(JaxPytreeStateCodecV1(public_state))
    state = {
        "schema_version": "path_c_scripted_partner_state_v1",
        "current_option": 0,
        "last_state": public_state,
        "last_primitive_action": 1,
        "option_runtime": None,
        "elapsed": 1,
        "bottleneck_alternate_phase": 0,
    }
    payload = codec.encode(state)
    restored = codec.decode(payload)
    np.testing.assert_array_equal(restored["last_state"]["counter"], [3])
    assert codec.encode(restored) == payload


def test_complete_evidence_accepts_truncated_terminal_row_without_valid_actions():
    spec = _evidence_spec()
    source = _evidence_observation()
    row = np.frombuffer(source.evidence_row_bytes, dtype=np.dtype("<f4")).copy()
    slices = spec.field_slices()
    row[slices["valid_actions"]] = 0.0
    row[slices["truncated"]] = 1.0
    observation = PartnerEvidenceObservationV1.from_evidence_row(
        evidence_spec=spec,
        evidence_row=row,
        option_choice_public_state_bytes=b"public-state",
        decision_public_state_bytes=b"public-state",
    )
    assert observation.evidence_row_bytes == row.tobytes(order="C")


def test_audit_capture_is_content_addressed_and_cannot_claim_training_provenance(
    tmp_path,
):
    capture = _capture()
    record = OCV2AuditHarvestRecordV1.from_capture(capture)
    manifest = record.to_manifest()
    assert manifest["source_kind"] == "audit_harvest"
    assert manifest["training_snapshot_reference"] is None
    assert record.snapshot_reference == f"sha256:{record.snapshot.sha256}"
    assert manifest["snapshot_manifest"]["snapshot_sha256"] == record.snapshot.sha256
    assert len(record.sha256) == 64
    bundle = record.to_bundle_bytes()
    restored = OCV2AuditHarvestRecordV1.from_bundle_bytes(bundle)
    assert restored == record
    tampered_bundle = bytearray(bundle)
    tampered_bundle[-1] ^= 1
    with pytest.raises(ValueError, match="hash mismatch"):
        OCV2AuditHarvestRecordV1.from_bundle_bytes(bytes(tampered_bundle))
    store = OCV2AuditContentAddressedStoreV1(tmp_path / "audit-store")
    stored_path = store.write(record)
    assert stored_path.name == f"{record.bundle_sha256}.ocv2audit"
    assert store.read(record.bundle_reference) == record

    with pytest.raises(ValueError, match="separate audit harvest"):
        replace(capture, source_kind="training")
    with pytest.raises(ValueError, match="cannot reference a training snapshot"):
        replace(record, training_snapshot_reference="sha256:training")


class _RecordingRolloutHooks:
    contract_version = OCV2_ROLLOUT_HOOK_CONTRACT_VERSION
    fresh_runtime_per_call = True
    shared_option_distribution = staticmethod(option_distribution)

    def __init__(self) -> None:
        self.calls = []

    def rollout_from(self, **kwargs):
        self.calls.append(kwargs)
        return OCV2AuditRolloutTraceV1(
            snapshot_sha256=kwargs["snapshot_sha256"],
            rng_keys=kwargs["rng_keys"],
            response_token_id=2,
            support_violation=False,
            trajectory_bytes=b"canonical-primitive-trajectory",
            primitive_steps=1,
        )


class _RestorableController:
    def __init__(self):
        self.state = {}

    def get_state(self):
        return dict(self.state)

    def set_state(self, state):
        self.state = dict(state)


def test_native_ocv2_hooks_restore_adapter_and_controller_state():
    from experiments.overcooked_v2.env_adapter import OCV2Adapter

    adapter = OCV2Adapter.__new__(OCV2Adapter)
    adapter.layout_name = "test-layout"
    adapter.max_steps = 12
    adapter.featurizer = None
    adapter.key = None
    adapter.state = None
    adapter.raw_obs = None
    adapter.obs = None

    def execute(**kwargs):
        restored = kwargs["adapter"].capture_state()
        assert tuple(map(int, np.asarray(restored.key).tolist())) == (1, 2)
        assert kwargs["partner"].get_state() == {"elapsed": 4}
        assert kwargs["ego"].get_state() == {"hidden": [1.0]}
        return OCV2AuditRolloutTraceV1(
            snapshot_sha256=kwargs["snapshot_sha256"],
            rng_keys=kwargs["rng_keys"],
            response_token_id=1,
            support_violation=False,
            trajectory_bytes=b"native-restored-trajectory",
            primitive_steps=1,
        )

    hooks = NativeOCV2AdapterRolloutHooks(
        adapter_factory=lambda: adapter,
        partner_factory=lambda _theta_id: _RestorableController(),
        ego_factory=_RestorableController,
        probe_executor=execute,
        restoration_contract_id="test-native-restore-v1",
    )
    keys = NamedRNGStreamsV1(
        jax=(1 << 32) | 2,
        numpy=3,
        python=4,
        torch=5,
        mode="fork",
        coordinate=("unit-0", "0", "probe-0", "0"),
    )
    result = hooks.rollout_from(
        snapshot_sha256="a" * 64,
        env_state={"counter": np.asarray(7, dtype=np.int32)},
        raw_observation={
            "agent_0": np.asarray([1], dtype=np.int32),
            "agent_1": np.asarray([2], dtype=np.int32),
        },
        partner_state={"elapsed": 4},
        ego_state={"hidden": [1.0]},
        theta_id="partner-a",
        probe_script=_probe(),
        rng_keys=keys,
    )
    assert result.snapshot_sha256 == "a" * 64


def test_ocv2_runner_separates_replay_and_fork_keys_and_replaces_full_hidden_state():
    capture = _capture()
    snapshot = capture.to_snapshot()
    codec = CanonicalJSONStateCodecV1()
    hooks = _RecordingRolloutHooks()
    runner = OCV2SnapshotForkRunner(
        env_state_codec=codec,
        observation_codec=codec,
        partner_state_codec=codec,
        ego_state_codec=codec,
        rollout_hooks=hooks,
    )
    before = snapshot.sha256
    replay = replay_from_snapshot(snapshot, _probe(), runner)
    assert replay.rng_keys.mode == "replay"
    assert hooks.calls[0]["theta_id"] is None
    assert hooks.calls[0]["partner_state"]["theta_id"] == "partner-a"

    hidden_payload = codec.encode({"theta_id": "partner-b", "elapsed": 7})
    hidden = FullHiddenStateV1(
        theta_id="partner-b",
        execution_state_key=hashlib.sha256(hidden_payload).hexdigest(),
        execution_state_bytes=hidden_payload,
    )
    draw = OuterReplicaDrawV1(
        audit_unit_id=snapshot.unit_id,
        outer_id=4,
        hidden_state=hidden,
        posterior_draw_probability=0.25,
        sampler_seed=23,
        posterior_mode="exact",
        rho_prune=0.0,
        source_episode_uid=snapshot.episode_uid,
    )
    fork = fork_from_snapshot(snapshot, draw, _probe(), runner, inner_id=2)
    assert fork.rng_keys.mode == "fork"
    assert fork.rng_keys.coordinate == (snapshot.unit_id, "4", "probe-0", "2")
    assert hooks.calls[1]["theta_id"] == "partner-b"
    assert hooks.calls[1]["partner_state"]["elapsed"] == 7
    for stream_name in ("jax", "numpy", "python", "torch"):
        assert fork.rng_keys.seed(stream_name) != replay.rng_keys.seed(stream_name)
    assert snapshot.sha256 == before
    assert replay.result.trajectory_sha256 != fork.result.trajectory_sha256

    forged_fork_keys = NamedRNGStreamsV1(
        jax=101,
        numpy=102,
        python=103,
        torch=104,
        mode="fork",
        coordinate=(snapshot.unit_id, "4", "probe-0", "2"),
    )
    with pytest.raises(ValueError, match="frozen derivation schedule"):
        runner.run(
            snapshot=snapshot,
            hidden_state=hidden,
            probe_script=_probe(),
            rng_keys=forged_fork_keys,
        )


def test_ocv2_runner_rejects_hooks_that_do_not_bind_shared_generation_policy():
    hooks = _RecordingRolloutHooks()
    hooks.shared_option_distribution = lambda *args, **kwargs: np.asarray([1.0])
    codec = CanonicalJSONStateCodecV1()
    with pytest.raises(ValueError, match="shared option_distribution"):
        OCV2SnapshotForkRunner(
            env_state_codec=codec,
            observation_codec=codec,
            partner_state_codec=codec,
            ego_state_codec=codec,
            rollout_hooks=hooks,
        )


@dataclass(frozen=True)
class _Option:
    id: int = 0
    kind: str = "noop"
    target_pos: None = None
    metadata: None = None


class _OneOptionLibrary:
    options = (_Option(),)

    @staticmethod
    def expected_cost(public_state, agent_id, option_id):
        del public_state, agent_id, option_id
        return 0.0


class _ObservedActionController:
    def __init__(self):
        self.state = {"count": 0}

    def get_state(self):
        return dict(self.state)

    def set_state(self, state):
        self.state = dict(state)

    def exact_observed_action_branches(self, state, observed_primitive_action):
        del state
        if int(observed_primitive_action) != 1:
            return ()
        return (({"count": int(self.state["count"]) + 1}, 0.4),)


def test_exact_registered_partner_kernel_binds_transition_and_likelihood():
    codec = CanonicalJSONStateCodecV1()
    option_library = _OneOptionLibrary()
    kernel = ExactRegisteredPartnerKernelV1(
        option_library=option_library,
        runtime_state_codec=codec,
        controller_factory=lambda _theta_id, _theta: _ObservedActionController(),
    )
    runtime_state = {"count": 0}
    runtime_bytes = codec.encode(runtime_state)
    evidence = _evidence_observation()
    evidence_row = np.frombuffer(
        evidence.evidence_row_bytes,
        dtype=np.dtype("<f4"),
    ).copy()
    public_state = _PublicStateCodec.decode(b"public-state")
    branches = tuple(kernel.advance(
        theta_id="partner-a",
        theta_specification=object(),
        runtime_state=runtime_state,
        runtime_state_bytes=runtime_bytes,
        option_choice_public_state=public_state,
        decision_public_state=public_state,
        public_state_path=(public_state, public_state),
        evidence_row=evidence_row,
        evidence_spec=_evidence_spec(),
        step_index=0,
    ))
    assert len(branches) == 1
    assert branches[0].conditional_probability == pytest.approx(1.0)
    probability = kernel.probability(
        theta_id="partner-a",
        theta_specification=object(),
        runtime_state=runtime_state,
        next_runtime_state_bytes=branches[0].state_bytes,
        option_choice_public_state=public_state,
        decision_public_state=public_state,
        public_state_path=(public_state, public_state),
        evidence_row=evidence_row,
        evidence_spec=_evidence_spec(),
        option_library=option_library,
        option_distribution_fn=option_distribution,
        step_index=0,
    )
    assert probability == pytest.approx(0.4)


class _PublicStateCodec:
    @staticmethod
    def encode(value):
        if not hasattr(value, "agents"):
            raise TypeError("expected public OCV2 state")
        return b"public-state"

    @staticmethod
    def decode(payload):
        if payload != b"public-state":
            raise ValueError("wrong public state payload")
        return SimpleNamespace(
            agents=SimpleNamespace(inventory=np.asarray([0, 0], dtype=np.int32))
        )


class _IdentityRuntimeTransition:
    contract_version = OCV2_AUDIT_BRIDGE_VERSION
    shared_option_distribution = staticmethod(option_distribution)

    @staticmethod
    def advance(**kwargs):
        return (
            PartnerRuntimeBranchV1(
                state_key=hashlib.sha256(kwargs["runtime_state_bytes"]).hexdigest(),
                state_bytes=kwargs["runtime_state_bytes"],
                conditional_probability=1.0,
            ),
        )


class _UnitEvidenceLikelihood:
    contract_version = OCV2_AUDIT_BRIDGE_VERSION
    shared_option_distribution = staticmethod(option_distribution)

    @staticmethod
    def probability(**kwargs):
        assert kwargs["evidence_row"].shape == (
            kwargs["evidence_spec"].evidence_dim,
        )
        probabilities = kwargs["option_distribution_fn"](
            kwargs["theta_specification"],
            kwargs["runtime_state"],
            kwargs["option_choice_public_state"],
            option_library=kwargs["option_library"],
        )
        return float(np.sum(probabilities))


class _UnboundRuntimeTransition(_IdentityRuntimeTransition):
    shared_option_distribution = staticmethod(
        lambda *_args, **_kwargs: np.asarray([1.0], dtype=np.float64)
    )


def test_exact_partner_bridge_requires_shared_distribution_for_both_hooks():
    with pytest.raises(ValueError, match="runtime transition must bind shared"):
        ExactPartnerPosteriorBridgeV1(
            registry=FrozenPartnerRegistryV1.from_registered("standard7"),
            option_library=_OneOptionLibrary(),
            option_library_sha256="a" * 64,
            ego_evidence_spec=_evidence_spec(),
            runtime_state_codec=CanonicalJSONStateCodecV1(),
            public_state_codec=_PublicStateCodec(),
            runtime_transition=_UnboundRuntimeTransition(),
            evidence_likelihood=_UnitEvidenceLikelihood(),
        )


def test_exact_partner_bridge_uses_registry_shared_distribution_and_full_u_sampler():
    registry = FrozenPartnerRegistryV1.from_registered("standard7")
    runtime_codec = CanonicalJSONStateCodecV1()
    bridge = ExactPartnerPosteriorBridgeV1(
        registry=registry,
        option_library=_OneOptionLibrary(),
        option_library_sha256="a" * 64,
        ego_evidence_spec=_evidence_spec(),
        runtime_state_codec=runtime_codec,
        public_state_codec=_PublicStateCodec(),
        runtime_transition=_IdentityRuntimeTransition(),
        evidence_likelihood=_UnitEvidenceLikelihood(),
    )
    runtime_payload = runtime_codec.encode(
        {
            "elapsed": 0,
            "bottleneck_alternate_phase": 0,
            "epsilon": 0.0,
            "valid_options": np.asarray([True], dtype=bool),
        }
    )
    theta_ids = tuple(entry.theta_id for entry in registry.entries)
    result = bridge.infer(
        initial_runtime_state_bytes_by_theta={
            theta_id: runtime_payload for theta_id in theta_ids
        },
        prior_by_theta={theta_id: 1.0 / len(theta_ids) for theta_id in theta_ids},
        observations=(_evidence_observation(),),
    )
    assert result.forward.mode == "exact"
    assert result.forward.prune_below == 0.0
    assert result.forward.rho_prune == 0.0
    assert len(result.forward.components) == len(registry.entries)
    assert result.partner_registry_sha256 == registry.sha256
    assert result.ego_evidence_spec_sha256 == _evidence_observation().evidence_spec_sha256

    draws = result.full_state_sampler().sample("audit-unit-0", 4, 31)
    assert len(draws) == 4
    assert all(draw.hidden_state.theta_id in theta_ids for draw in draws)
    assert all(draw.hidden_state.execution_state_bytes == runtime_payload for draw in draws)

    wrong_spec = replace(_evidence_observation(), evidence_spec_sha256="f" * 64)
    with pytest.raises(ValueError, match="does not bind this frozen EgoEvidenceSpecV1"):
        bridge.infer(
            initial_runtime_state_bytes_by_theta={
                theta_id: runtime_payload for theta_id in theta_ids
            },
            prior_by_theta={theta_id: 1.0 / len(theta_ids) for theta_id in theta_ids},
            observations=(wrong_spec,),
        )


class _MassLosingRuntimeTransition:
    contract_version = OCV2_AUDIT_BRIDGE_VERSION
    shared_option_distribution = staticmethod(option_distribution)

    @staticmethod
    def advance(**kwargs):
        return (
            PartnerRuntimeBranchV1(
                state_key=hashlib.sha256(kwargs["runtime_state_bytes"]).hexdigest(),
                state_bytes=kwargs["runtime_state_bytes"],
                conditional_probability=0.5,
            ),
        )


def test_exact_partner_bridge_rejects_runtime_transition_mass_loss():
    registry = FrozenPartnerRegistryV1.from_registered("standard7")
    runtime_codec = CanonicalJSONStateCodecV1()
    bridge = ExactPartnerPosteriorBridgeV1(
        registry=registry,
        option_library=_OneOptionLibrary(),
        option_library_sha256="b" * 64,
        ego_evidence_spec=_evidence_spec(),
        runtime_state_codec=runtime_codec,
        public_state_codec=_PublicStateCodec(),
        runtime_transition=_MassLosingRuntimeTransition(),
        evidence_likelihood=_UnitEvidenceLikelihood(),
    )
    runtime_payload = runtime_codec.encode(
        {"valid_options": [True], "elapsed": 0, "bottleneck_alternate_phase": 0}
    )
    theta_ids = tuple(entry.theta_id for entry in registry.entries)
    with pytest.raises(ValueError, match="must sum to one"):
        bridge.infer(
            initial_runtime_state_bytes_by_theta={
                theta_id: runtime_payload for theta_id in theta_ids
            },
            prior_by_theta={theta_id: 1.0 / len(theta_ids) for theta_id in theta_ids},
            observations=(_evidence_observation(),),
        )
