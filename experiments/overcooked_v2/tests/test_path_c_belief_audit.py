from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib

import numpy as np
import pytest

from experiments.overcooked_v2.path_c_audit_battery import (
    FrozenAuditBatteryV1,
    ProbeScriptV1,
)
from experiments.overcooked_v2.path_c_belief_audit import (
    CategoricalPosteriorFullStateSampler,
    CategoricalRolloutResultV1,
    ForwardBranchV1,
    ForwardHypothesisV1,
    FrozenInstrumentProbeV1,
    FullHiddenStateV1,
    HistoryHashCollisionError,
    InstrumentKernelCellV1,
    NamedRNGStreamsV1,
    RNGKeyScheduleV1,
    SparseForwardStateV1,
    SnapshotV1,
    Tier2HistoryCandidateV1,
    Tier2MatchedFullStateSampler,
    build_tier2_exact_history_index,
    build_instrument_cell_registry_v2,
    build_instrument_evidence_from_kernel_records_v2,
    build_instrument_measurement_v1,
    assess_tier_agreement_equivalence,
    capture_snapshot_v1,
    collect_paired_battery_kernel_records,
    compute_alignment_bounds,
    estimate_finite_categorical_cell,
    diagnose_synthetic_outer_sampling,
    first_inner_tokens_by_probe,
    fork_from_snapshot,
    make_exact_history_key,
    replay_from_snapshot,
    recompute_instrument_measurement_from_evidence_v2,
    sparse_state_merging_forward_recursion,
    verify_exact_history_match,
    verify_history_digest_and_bytes,
    verify_snapshot_runner_purity,
)


def _schedule() -> RNGKeyScheduleV1:
    return RNGKeyScheduleV1.from_original_seeds(
        manifest_seed=123,
        jax=11,
        numpy=22,
        python=33,
        torch=44,
    )


def _snapshot() -> SnapshotV1:
    return SnapshotV1(
        unit_id="unit-0",
        episode_uid="episode-0",
        decision_index=3,
        env_state_bytes=b"env-state",
        raw_observation_bytes=b"raw-observation",
        partner_state_bytes=b"partner-state",
        ego_state_bytes=b"ego-state",
        rng_key_schedule=_schedule(),
    )


def test_jax_key_words_preserve_all_frozen_64_bit_seed_material():
    streams = NamedRNGStreamsV1(
        jax=0x123456789ABCDEF0,
        numpy=1,
        python=2,
        torch=3,
        mode="replay",
        coordinate=("generation",),
    )
    assert streams.jax_key_words_uint32() == (0x12345678, 0x9ABCDEF0)


def _snapshot_for_unit(unit_id: str, episode_uid: str) -> SnapshotV1:
    return SnapshotV1(
        unit_id=unit_id,
        episode_uid=episode_uid,
        decision_index=3,
        env_state_bytes=f"env:{unit_id}".encode("utf-8"),
        raw_observation_bytes=f"observation:{unit_id}".encode("utf-8"),
        partner_state_bytes=f"partner:{unit_id}".encode("utf-8"),
        ego_state_bytes=f"ego:{unit_id}".encode("utf-8"),
        rng_key_schedule=_schedule(),
    )


def _identity_transition(state, _observation, _step_index):
    return (
        ForwardBranchV1(
            state_key=state.state_key,
            state_bytes=state.state_bytes,
            probability=1.0,
        ),
    )


def _exact_posterior():
    return sparse_state_merging_forward_recursion(
        (
            ForwardHypothesisV1("theta-a", "state-a", b"state-a", 0.7),
            ForwardHypothesisV1("theta-b", "state-b", b"state-b", 0.3),
        ),
        ("observation",),
        _identity_transition,
        mode="exact",
    )


def _battery() -> FrozenAuditBatteryV1:
    return FrozenAuditBatteryV1(
        battery_id="paired-test",
        battery_version="v1",
        T_probe=2,
        L_inner=2,
        core_scripts=(
            ProbeScriptV1("probe-a", (1, 2), 0.5),
            ProbeScriptV1("probe-b", (3, 4), 0.5),
        ),
    )


class _DeterministicRunner:
    def run(self, *, snapshot, hidden_state, probe_script, rng_keys):
        hidden_key = "replay" if hidden_state is None else hidden_state.execution_state_key
        payload = (
            f"{snapshot.sha256}:{hidden_key}:{probe_script.script_id}:"
            f"{rng_keys.mode}:{rng_keys.seed('numpy')}"
        ).encode("utf-8")
        token = int(hashlib.sha256(payload).digest()[0] % 4)
        return CategoricalRolloutResultV1(
            response_token_id=token,
            support_violation=False,
            trajectory_sha256=hashlib.sha256(payload).hexdigest(),
        )


class _RegisteredCellTokenRunner:
    def run(self, *, snapshot, hidden_state, probe_script, rng_keys):
        del hidden_state
        token = 0 if snapshot.unit_id.startswith("value-a") else 1
        payload = (
            f"{snapshot.sha256}:{snapshot.unit_id}:{probe_script.script_id}:"
            f"{rng_keys.coordinate}:{token}"
        ).encode("utf-8")
        return CategoricalRolloutResultV1(
            response_token_id=token,
            support_violation=False,
            trajectory_sha256=hashlib.sha256(payload).hexdigest(),
        )


def _instrument_evidence_from_kernel_records(*, outer_replicas: int = 256):
    posterior = _exact_posterior()
    battery = _battery()
    sampler = CategoricalPosteriorFullStateSampler(posterior)
    cells = []
    for cell_id, value_class_id, unit_id in (
        ("a-left", "a", "value-a-left"),
        ("a-right", "a", "value-a-right"),
        ("b", "b", "value-b"),
    ):
        snapshot = _snapshot_for_unit(unit_id, f"episode:{unit_id}")
        sampler_base_seed = 103
        records = collect_paired_battery_kernel_records(
            snapshot,
            sampler.sample(
                unit_id,
                outer_replicas,
                seed=sampler_base_seed,
            ),
            battery,
            _RegisteredCellTokenRunner(),
            role="locked_audit",
        )
        cells.append(
            InstrumentKernelCellV1.from_posterior(
                cell_id=cell_id,
                value_class_id=value_class_id,
                audit_unit_id=unit_id,
                snapshot_sha256=snapshot.sha256,
                probe_id="probe-a",
                sampler_base_seed=sampler_base_seed,
                outer_id_start=0,
                posterior=posterior,
                kernel_records=records,
            )
        )
    frozen_probes = tuple(
        FrozenInstrumentProbeV1(
            probe_id=script.script_id,
            sampling_probability=script.sampling_probability,
        )
        for script in battery.core_scripts
    )
    registry = build_instrument_cell_registry_v2(
        cells,
        response_vocabulary_sha256="a" * 64,
        battery_sha256=battery.sha256,
        frozen_probes=frozen_probes,
        summary_id="summary-v1",
        inner_forks_L_inner=battery.L_inner,
        support_violation_token_id=3,
    )
    return build_instrument_evidence_from_kernel_records_v2(
        cells,
        response_vocabulary_sha256="a" * 64,
        vocabulary_size_q=4,
        confidence_delta=0.05,
        exact_or_approximate="exact",
        rho_prune=0.0,
        posterior_bias_bound=0.0,
        reset_bias_bound=0.0,
        battery_sha256=battery.sha256,
        frozen_probes=frozen_probes,
        summary_id="summary-v1",
        inner_forks_L_inner=battery.L_inner,
        support_violation_token_id=3,
        frozen_cell_registry_sha256=registry["sha256"],
    )


class _BytesCodec:
    def encode(self, value):
        return str(value).encode("utf-8")

    def decode(self, payload):
        return payload.decode("utf-8")


def test_snapshot_capture_encodes_every_fork_complete_component() -> None:
    snapshot = capture_snapshot_v1(
        unit_id="unit-captured",
        episode_uid="episode-captured",
        decision_index=2,
        env_state="env",
        raw_observation="observation",
        partner_state="partner",
        ego_state="ego",
        state_codec=_BytesCodec(),
        observation_codec=_BytesCodec(),
        rng_key_schedule=_schedule(),
    )
    assert snapshot.env_state_bytes == b"env"
    assert snapshot.raw_observation_bytes == b"observation"
    assert snapshot.partner_state_bytes == b"partner"
    assert snapshot.ego_state_bytes == b"ego"


def test_snapshot_runner_is_pure_for_an_identical_fork_coordinate() -> None:
    snapshot = _snapshot()
    draw = CategoricalPosteriorFullStateSampler(_exact_posterior()).sample(
        "unit-0", 1, seed=5
    )[0]
    assert verify_snapshot_runner_purity(
        snapshot,
        draw,
        _battery().core_scripts[0],
        _DeterministicRunner(),
        inner_id=0,
    ) is True


def test_snapshot_is_immutable_and_binds_original_replay_keys():
    snapshot = _snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.decision_index = 4

    copied = snapshot.snapshot_copy()
    assert copied == snapshot
    assert copied is not snapshot
    assert copied.sha256 == snapshot.sha256
    assert snapshot.to_manifest()["snapshot_sha256"] == snapshot.sha256
    assert snapshot.rng_key_schedule.replay_keys().to_payload() == {
        "jax": 11,
        "numpy": 22,
        "python": 33,
        "torch": 44,
        "mode": "replay",
        "schedule_version": "path_c_rng_key_schedule_v1",
        "coordinate": ["generation"],
    }


def test_replay_and_fork_rng_streams_are_separate_and_deterministic():
    schedule = _schedule()
    first = schedule.fork_keys(
        unit_id="unit-0",
        outer_id=2,
        probe_id="probe-a",
        inner_id=1,
    )
    repeated = schedule.fork_keys(
        unit_id="unit-0",
        outer_id=2,
        probe_id="probe-a",
        inner_id=1,
    )
    changed_inner = schedule.fork_keys(
        unit_id="unit-0",
        outer_id=2,
        probe_id="probe-a",
        inner_id=2,
    )

    assert first == repeated
    assert first != changed_inner
    assert first.mode == "fork"
    for stream_name in ("jax", "numpy", "python", "torch"):
        assert first.seed(stream_name) != schedule.replay_keys().seed(stream_name)


def test_sparse_forward_recursion_merges_states_and_exact_mode_never_prunes():
    def branching_transition(state, _observation, _step_index):
        return (
            ForwardBranchV1("merged", b"merged", 0.25),
            ForwardBranchV1("merged", b"merged", 0.75),
        )

    result = sparse_state_merging_forward_recursion(
        (
            ForwardHypothesisV1("theta-a", "start", b"start", 0.5),
            ForwardHypothesisV1("theta-b", "start", b"start", 0.5),
        ),
        ("observation",),
        branching_transition,
        mode="exact",
    )

    assert result.rho_prune == 0.0
    assert len(result.components) == 2
    assert result.steps[0].emitted_branches == 4
    assert result.steps[0].hypotheses_after_merge == 2
    assert result.steps[0].merged_branch_count == 2
    with pytest.raises(ValueError, match="forbids"):
        sparse_state_merging_forward_recursion(
            (ForwardHypothesisV1("theta", "start", b"start", 1.0),),
            ("observation",),
            _identity_transition,
            mode="exact",
            prune_below=1.0e-6,
        )


def test_sparse_forward_recursion_matches_tiny_horizon_brute_force():
    def transition(state, _observation, step_index):
        if step_index == 0:
            return (
                ForwardBranchV1("left", b"left", 0.25),
                ForwardBranchV1("right", b"right", 0.75),
            )
        return (
            ForwardBranchV1("merged", b"merged", 0.5),
            ForwardBranchV1(f"unique-{state.state_key}", state.state_bytes, 0.5),
        )

    initial = (ForwardHypothesisV1("theta", "start", b"start", 1.0),)
    result = sparse_state_merging_forward_recursion(
        initial,
        ("obs-0", "obs-1"),
        transition,
        mode="exact",
    )

    paths = [("theta", "start", b"start", 1.0)]
    for step_index, observation in enumerate(("obs-0", "obs-1")):
        expanded = []
        for theta_id, state_key, state_bytes, mass in paths:
            state = SparseForwardStateV1(theta_id, state_key, state_bytes)
            for branch in transition(state, observation, step_index):
                expanded.append(
                    (
                        theta_id,
                        branch.state_key,
                        branch.state_bytes,
                        mass * branch.probability,
                    )
                )
        paths = expanded
    brute: dict[tuple[str, str], float] = {}
    for theta_id, state_key, _state_bytes, mass in paths:
        brute[(theta_id, state_key)] = brute.get((theta_id, state_key), 0.0) + mass
    observed = {
        (item.hidden_state.theta_id, item.hidden_state.execution_state_key):
        item.posterior_probability
        for item in result.components
    }
    assert observed == pytest.approx(brute)

def test_approximate_forward_recursion_records_pruned_posterior_mass():
    result = sparse_state_merging_forward_recursion(
        (
            ForwardHypothesisV1("theta-major", "major", b"major", 0.99),
            ForwardHypothesisV1("theta-minor", "minor", b"minor", 0.01),
        ),
        ("observation",),
        _identity_transition,
        mode="approximate",
        prune_below=0.05,
    )

    assert len(result.components) == 1
    assert result.components[0].hidden_state.theta_id == "theta-major"
    assert result.steps[0].pruned_hypotheses == 1
    assert result.rho_prune == pytest.approx(0.01)


def test_full_state_sampler_draws_complete_outer_states_with_replacement():
    sampler = CategoricalPosteriorFullStateSampler(_exact_posterior())
    draws = sampler.sample("unit-0", 12, seed=91, outer_id_start=5)

    assert [draw.outer_id for draw in draws] == list(range(5, 17))
    assert all(draw.audit_unit_id == "unit-0" for draw in draws)
    assert all(draw.posterior_mode == "exact" and draw.rho_prune == 0.0 for draw in draws)
    assert all(draw.hidden_state.execution_state_bytes for draw in draws)
    assert {draw.hidden_state.theta_id for draw in draws}.issubset({"theta-a", "theta-b"})


def test_full_state_sampler_coordinates_are_chunk_stable_and_unit_specific():
    sampler = CategoricalPosteriorFullStateSampler(_exact_posterior())
    whole = sampler.sample("unit-0", 8, seed=91, outer_id_start=0)
    chunked = (
        *sampler.sample("unit-0", 3, seed=91, outer_id_start=0),
        *sampler.sample("unit-0", 5, seed=91, outer_id_start=3),
    )
    other_unit = sampler.sample("unit-1", 8, seed=91, outer_id_start=0)

    assert whole == chunked
    assert [draw.sampler_seed for draw in whole] != [
        draw.sampler_seed for draw in other_unit
    ]


def test_replay_uses_original_keys_and_fork_uses_fresh_keys():
    snapshot = _snapshot()
    script = _battery().core_scripts[0]
    runner = _DeterministicRunner()
    draw = CategoricalPosteriorFullStateSampler(_exact_posterior()).sample(
        "unit-0", 1, seed=7
    )[0]

    replay = replay_from_snapshot(snapshot, script, runner)
    fork = fork_from_snapshot(snapshot, draw, script, runner, inner_id=0)

    assert replay.rng_keys.mode == "replay"
    assert replay.rng_keys == snapshot.rng_key_schedule.replay_keys()
    assert fork.rng_keys.mode == "fork"
    assert fork.rng_keys != replay.rng_keys


def test_kernel_table_keeps_one_row_per_outer_replica_and_pairs_all_probes():
    snapshot = _snapshot()
    battery = _battery()
    draws = CategoricalPosteriorFullStateSampler(_exact_posterior()).sample(
        "unit-0", 4, seed=19
    )
    records = collect_paired_battery_kernel_records(
        snapshot,
        draws,
        battery,
        _DeterministicRunner(),
        role="locked_audit",
    )

    assert len(records) == len(draws)
    assert len({record.outer_cluster_id for record in records}) == len(draws)
    assert all(len(record.probe_responses) == 2 for record in records)
    assert all(record.hidden_state_sha256 for record in records)
    assert all(
        len(probe.responses) == battery.L_inner
        for record in records
        for probe in record.probe_responses
    )
    fork_coordinates = {
        response.rng_keys.coordinate
        for record in records
        for probe in record.probe_responses
        for response in probe.responses
    }
    assert len(fork_coordinates) == len(draws) * 2 * battery.L_inner
    assert len(first_inner_tokens_by_probe(records, "probe-a")) == len(draws)


def test_exact_history_key_uses_raw_discrete_bytes_and_joint_actions():
    states = (
        {
            "grid": np.asarray([[1, 2], [3, 4]], dtype=np.int16),
            "inventory": np.asarray([0, 1], dtype=np.uint8),
        },
        {
            "grid": np.asarray([[1, 2], [3, 5]], dtype=np.int16),
            "inventory": np.asarray([1, 1], dtype=np.uint8),
        },
    )
    key = make_exact_history_key(
        states,
        (0, 1),
        (2, 3),
        field_order=("grid", "inventory"),
    )
    same = make_exact_history_key(
        states,
        (0, 1),
        (2, 3),
        field_order=("grid", "inventory"),
    )
    different_action = make_exact_history_key(
        states,
        (0, 1),
        (2, 4),
        field_order=("grid", "inventory"),
    )

    assert verify_exact_history_match(key, same) is True
    assert verify_exact_history_match(key, different_action) is False
    with pytest.raises(ValueError, match="discrete"):
        make_exact_history_key(
            ({"grid": np.asarray([0.25], dtype=np.float32)},),
            (0,),
            (0,),
            field_order=("grid",),
        )
    with pytest.raises(HistoryHashCollisionError):
        verify_history_digest_and_bytes("0" * 64, b"left", "0" * 64, b"right")


def test_tier2_index_enforces_episode_uniqueness_and_positivity():
    key = make_exact_history_key(
        ({"grid": np.asarray([1, 2], dtype=np.int8)},),
        (0,),
        (1,),
        field_order=("grid",),
    )
    hidden = FullHiddenStateV1("theta", "state", b"state")
    first = Tier2HistoryCandidateV1(
        episode_uid="episode-a",
        decision_index=1,
        history_key=key,
        snapshot_reference=f"sha256:{hashlib.sha256(b'snapshot-a').hexdigest()}",
        hidden_state=hidden,
        harvest_policy_id="policy-a",
        inclusion_probability=0.5,
    )
    second = Tier2HistoryCandidateV1(
        episode_uid="episode-b",
        decision_index=2,
        history_key=key,
        snapshot_reference=f"sha256:{hashlib.sha256(b'snapshot-b').hexdigest()}",
        hidden_state=hidden,
        harvest_policy_id="policy-b",
        inclusion_probability=0.75,
    )
    index = build_tier2_exact_history_index(
        (first, second),
        minimum_inclusion_probability=0.25,
        minimum_unique_episodes=2,
    )

    group = index.lookup(key)
    assert group is not None
    assert len(group.members) == 2
    assert group.minimum_inclusion_probability == 0.5
    assert group.inverse_probability_effective_sample_size <= 2.0
    with pytest.raises(ValueError, match="equal inclusion"):
        Tier2MatchedFullStateSampler(group)
    with pytest.raises(ValueError, match="at most one"):
        build_tier2_exact_history_index(
            (first, first),
            minimum_inclusion_probability=0.25,
        )
    with pytest.raises(ValueError, match="positivity floor"):
        build_tier2_exact_history_index(
            (first,),
            minimum_inclusion_probability=0.75,
        )


def test_tier2_equal_propensity_group_is_a_full_state_outer_sampler():
    key = make_exact_history_key(
        ({"grid": np.asarray([1, 2], dtype=np.int8)},),
        (0,),
        (1,),
        field_order=("grid",),
    )
    candidates = tuple(
        Tier2HistoryCandidateV1(
            episode_uid=f"episode-{index}",
            decision_index=index,
            history_key=key,
            snapshot_reference=(
                "sha256:"
                + hashlib.sha256(f"snapshot-{index}".encode("utf-8")).hexdigest()
            ),
            hidden_state=FullHiddenStateV1(
                f"theta-{index}",
                f"state-{index}",
                f"state-{index}".encode("utf-8"),
            ),
            harvest_policy_id="policy",
            inclusion_probability=0.5,
        )
        for index in range(2)
    )
    index = build_tier2_exact_history_index(
        candidates,
        minimum_inclusion_probability=0.5,
        minimum_unique_episodes=2,
    )
    group = index.lookup(key)
    assert group is not None
    sampler = Tier2MatchedFullStateSampler(group)
    draws = sampler.sample("unit-0", 2, seed=12)

    assert len(draws) == 2
    assert all(draw.posterior_mode == "exact" for draw in draws)
    assert all(draw.source_episode_uid in {"episode-0", "episode-1"} for draw in draws)
    assert len({draw.source_episode_uid for draw in draws}) == len(draws)
    assert draws == (
        *sampler.sample("unit-0", 1, seed=12, outer_id_start=0),
        *sampler.sample("unit-0", 1, seed=12, outer_id_start=1),
    )
    with pytest.raises(ValueError, match="without replacement"):
        sampler.sample("unit-0", 3, seed=12)


def test_cell_specific_bounds_use_frozen_q_and_bound_before_max_min():
    common = {
        "probe_id": "probe-a",
        "summary_id": "summary-v1",
        "vocabulary_size_q": 2,
        "simultaneous_cell_count": 3,
        "confidence_delta": 0.05,
    }
    within_left = estimate_finite_categorical_cell(
        cell_id="class-a-left",
        value_class_id="class-a",
        response_token_ids=[0] * 2000,
        **common,
    )
    within_right = estimate_finite_categorical_cell(
        cell_id="class-a-right",
        value_class_id="class-a",
        response_token_ids=[0] * 1980 + [1] * 20,
        **common,
    )
    between = estimate_finite_categorical_cell(
        cell_id="class-b",
        value_class_id="class-b",
        response_token_ids=[1] * 2000,
        **common,
    )
    bounds = compute_alignment_bounds((within_left, within_right, between))

    assert bounds.alpha_upper == max(
        item.upper_bound for item in bounds.within_class_bounds
    )
    assert bounds.beta_lower == min(
        item.lower_bound for item in bounds.between_class_bounds
    )
    assert bounds.beta_lower > bounds.alpha_upper
    assert bounds.instrument_valid is True
    measurement = build_instrument_measurement_v1(
        bounds,
        exact_or_approximate="exact",
        rho_prune=0.0,
        minimum_outer_replicas=2000,
        response_vocabulary_sha256="a" * 64,
    )
    assert measurement["alpha_upper"] == bounds.alpha_upper
    assert measurement["beta_lower"] == bounds.beta_lower
    assert measurement["rho_prune"] == 0.0
    assert "instrument_valid" not in measurement
    with pytest.raises(ValueError, match="outside"):
        estimate_finite_categorical_cell(
            cell_id="bad",
            value_class_id="bad",
            response_token_ids=(0, 2),
            **common,
        )


def test_instrument_measurement_is_recomputed_from_unique_outer_rows():
    evidence = _instrument_evidence_from_kernel_records()

    assert evidence["schema_version"] == "path_c_instrument_evidence_v2"
    assert "validity_checks" not in evidence
    assert all("kernel_records" in cell for cell in evidence["cells"])
    assert all("response_token_ids" not in cell for cell in evidence["cells"])
    measurement = recompute_instrument_measurement_from_evidence_v2(evidence)

    assert measurement["minimum_outer_replicas"] == 256
    assert measurement["minimum_effective_outer_sample_size"] == 256.0
    assert measurement["sampling_design"] == (
        "iid_full_posterior_with_replacement_v1"
    )
    assert measurement["sampling_source"] == (
        "categorical_full_state_posterior_v1"
    )
    assert measurement["battery_sha256"] == _battery().sha256
    assert measurement["kernel_records_sha256"] == evidence[
        "kernel_records_sha256"
    ]
    assert set(measurement["validity_checks"]) == {
        "embedded_kernel_records_schema_valid",
        "battery_snapshot_audit_unit_probe_binding_valid",
        "iid_full_posterior_with_replacement_valid",
        "sampler_seed_schedule_valid",
        "posterior_support_draws_valid",
        "outer_cluster_and_fork_coordinates_unique",
        "no_primary_support_violations",
    }
    assert all(measurement["validity_checks"].values())
    assert measurement["beta_lower"] > measurement["alpha_upper"]
    assert all(
        item["effective_outer_sample_size"] == item["outer_replicas_M"]
        for item in measurement["cell_diagnostics"]
    )
    assert all(
        0.0 < item["observed_support_fraction"] <= 1.0
        and 0.0 < item["observed_posterior_mass"] <= 1.0
        for item in measurement["cell_diagnostics"]
    )


def test_instrument_evidence_rejects_self_registered_cell_tampering():
    evidence = _instrument_evidence_from_kernel_records()
    tampered = copy.deepcopy(evidence)
    tampered["cells"][0]["value_class_id"] = "post-hoc-value-class"
    with pytest.raises(ValueError, match="cell registry SHA-256"):
        recompute_instrument_measurement_from_evidence_v2(tampered)


def test_support_violation_cannot_enter_an_ordinary_response_cell():
    evidence = _instrument_evidence_from_kernel_records()
    tampered = copy.deepcopy(evidence)
    response = tampered["cells"][0]["kernel_records"][0]["probe_responses"][0][
        "responses"
    ][0]
    response["support_violation"] = True
    with pytest.raises(ValueError, match="frozen special response token"):
        recompute_instrument_measurement_from_evidence_v2(tampered)


def test_instrument_evidence_rejects_legacy_tokens_and_self_reported_checks():
    legacy = {
        "schema_version": "path_c_instrument_evidence_v1",
        "response_vocabulary_sha256": "a" * 64,
        "vocabulary_size_q": 2,
        "simultaneous_cell_count": 3,
        "confidence_delta": 0.05,
        "exact_or_approximate": "exact",
        "rho_prune": 0.0,
        "posterior_bias_bound": 0.0,
        "reset_bias_bound": 0.0,
        "validity_checks": {"full_state_sampling": True},
        "cells": [],
    }

    with pytest.raises(ValueError, match="key mismatch"):
        recompute_instrument_measurement_from_evidence_v2(legacy)

    forged_tokens = _instrument_evidence_from_kernel_records(outer_replicas=16)
    forged_tokens["cells"][0]["response_token_ids"] = [0] * 16
    with pytest.raises(ValueError, match="key mismatch"):
        recompute_instrument_measurement_from_evidence_v2(forged_tokens)


def test_instrument_evidence_rejects_tampered_embedded_token_and_sampler_seed():
    evidence = _instrument_evidence_from_kernel_records(outer_replicas=32)
    tampered_token = copy.deepcopy(evidence)
    tampered_token["cells"][0]["kernel_records"][0]["probe_responses"][0][
        "responses"
    ][0]["response_token_id"] = 3
    with pytest.raises(ValueError, match="kernel-record SHA-256"):
        recompute_instrument_measurement_from_evidence_v2(tampered_token)

    tampered_seed = copy.deepcopy(evidence)
    tampered_seed["cells"][0]["kernel_records"][0]["sampler_seed"] += 1
    with pytest.raises(ValueError, match="sampler seed"):
        recompute_instrument_measurement_from_evidence_v2(tampered_seed)


def test_primary_instrument_rejects_nonposterior_source_and_no_replacement_design():
    evidence = _instrument_evidence_from_kernel_records(outer_replicas=32)
    matched_history = copy.deepcopy(evidence)
    matched_history["cells"][0]["kernel_records"][0][
        "source_episode_uid"
    ] = "matched-episode"
    with pytest.raises(ValueError, match="episode sources"):
        recompute_instrument_measurement_from_evidence_v2(matched_history)

    without_replacement = copy.deepcopy(evidence)
    without_replacement["sampling_design"] = "without_replacement"
    with pytest.raises(ValueError, match="with replacement"):
        recompute_instrument_measurement_from_evidence_v2(without_replacement)


def test_instrument_evidence_binds_battery_snapshot_probe_and_support_draw():
    evidence = _instrument_evidence_from_kernel_records(outer_replicas=32)
    changed_battery = copy.deepcopy(evidence)
    changed_battery["cells"][0]["kernel_records"][0]["battery_sha256"] = (
        "b" * 64
    )
    with pytest.raises(ValueError, match="battery hash"):
        recompute_instrument_measurement_from_evidence_v2(changed_battery)

    changed_snapshot = copy.deepcopy(evidence)
    changed_snapshot["cells"][0]["kernel_records"][0]["snapshot_sha256"] = (
        "b" * 64
    )
    with pytest.raises(ValueError, match="snapshot"):
        recompute_instrument_measurement_from_evidence_v2(changed_snapshot)

    changed_probe = copy.deepcopy(evidence)
    changed_probe["cells"][0]["kernel_records"][0]["probe_responses"][0][
        "probe_id"
    ] = "not-frozen"
    with pytest.raises(ValueError, match="probe set"):
        recompute_instrument_measurement_from_evidence_v2(changed_probe)

    changed_support_draw = copy.deepcopy(evidence)
    record = changed_support_draw["cells"][0]["kernel_records"][0]
    record["theta_id"] = "forged-theta"
    with pytest.raises(ValueError, match="deterministic posterior draw"):
        recompute_instrument_measurement_from_evidence_v2(changed_support_draw)

    changed_support_bytes = copy.deepcopy(evidence)
    changed_support_bytes["cells"][0]["posterior_support"][0][
        "execution_state_bytes_hex"
    ] = b"forged-state".hex()
    with pytest.raises(ValueError, match="complete state bytes"):
        recompute_instrument_measurement_from_evidence_v2(changed_support_bytes)

    changed_mode = copy.deepcopy(evidence)
    changed_mode["cells"][0]["kernel_records"][0][
        "posterior_mode"
    ] = "approximate"
    with pytest.raises(ValueError, match="posterior mode"):
        recompute_instrument_measurement_from_evidence_v2(changed_mode)


def test_full_state_outer_sampling_converges_while_naive_single_state_is_biased():
    diagnostic = diagnose_synthetic_outer_sampling(
        posterior_weights=(0.5, 0.5),
        response_probabilities_by_full_state=((1.0, 0.0), (0.0, 1.0)),
        outer_replicas_M=4096,
        repetitions=32,
        failure_threshold=0.25,
        seed=17,
    )
    assert diagnostic.target_distribution == pytest.approx((0.5, 0.5))
    assert diagnostic.oracle_point_mass_index == 0
    assert diagnostic.oracle_point_mass_tv == pytest.approx(0.5)
    assert diagnostic.expected_naive_single_draw_tv == pytest.approx(0.5)
    assert diagnostic.naive_failure_probability == pytest.approx(1.0)
    assert diagnostic.full_u_tv_quantile_95 < 0.03


def test_tier_agreement_is_an_interval_equivalence_test_not_point_equality():
    equivalent = assess_tier_agreement_equivalence(
        0.01,
        (-0.02, 0.04),
        equivalence_bound=0.05,
    )
    inconclusive = assess_tier_agreement_equivalence(
        0.01,
        (-0.02, 0.06),
        equivalence_bound=0.05,
    )
    assert equivalent.equivalent is True
    assert inconclusive.equivalent is False
