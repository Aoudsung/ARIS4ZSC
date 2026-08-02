from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from experiments.overcooked_v2.deployment import (  # noqa: E402
    Deployment,
    deployment_action,
)
from src.path_c.anchor_replay import (  # noqa: E402
    AnchorAuditManifest,
    AnchorAuditRecord,
    AnchorReplayItem,
    AnchorTrainingReplay,
)
from src.path_c.anchor_sampling import anchor_preflight_world_count  # noqa: E402
from src.path_c.calibration import empty_calibration  # noqa: E402
from src.path_c.base_distillation import _carry_leaf_max_error  # noqa: E402
from src.path_c.fallback import (  # noqa: E402
    DeploymentTier,
    FallbackArtifactIdentity,
    assert_all_formal_seeds,
    deployment_tier,
)
from src.path_c.gradient_routing import (  # noqa: E402
    project_response_gradient,
    scale_base_policy_gradients,
    tree_inner_product,
    tree_l2_norm,
)
from src.path_c.model import (  # noqa: E402
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.partner_episode import (  # noqa: E402
    GeneratorAdmission,
    PartnerEpisodeBatch,
    rollback_unqualified_generator,
    source_logit_distillation_loss,
    validate_complete_episode_batch,
)
from src.path_c.policy_epoch import (  # noqa: E402
    TargetPolicyEpoch,
    clone_epoch_target_for_live_control,
    combined_policy_fingerprint,
)
from src.path_c.qualification import (  # noqa: E402
    GateDecision,
    SignalQualification,
)
from src.path_c.signal_audit import fixture_contract_outcome  # noqa: E402
from src.path_c.training import (  # noqa: E402
    conditional_entropy_noncollapse_penalty,
)
from src.path_c.types import ModelOutput  # noqa: E402


def _epoch(identifier: int = 0, suffix: str = "a") -> TargetPolicyEpoch:
    return TargetPolicyEpoch(
        epoch_id=identifier,
        actor_fingerprint=f"actor-{suffix}",
        belief_fingerprint=f"belief-{suffix}",
        raw_q_fingerprint=f"raw-q-{suffix}",
        target_params={"frozen": jnp.asarray([identifier], dtype=jnp.float32)},
        started_update=identifier * 16 + 1,
    )


def test_training_and_audit_anchor_preflight_have_distinct_registered_shapes() -> None:
    config = SimpleNamespace(
        anchors=SimpleNamespace(
            selected_ordinary=24,
            selected_matched_code=24,
            audit_ordinary_states=32,
            audit_matched_code_states=32,
        )
    )
    assert anchor_preflight_world_count(config, mode="training") == 72
    assert anchor_preflight_world_count(config, mode="audit") == 64
    with pytest.raises(ValueError, match="training or audit"):
        anchor_preflight_world_count(config, mode="unknown")


def test_audit_manifest_enforces_the_run_registered_split() -> None:
    manifest = AnchorAuditManifest(
        expected_fit_replicas=1,
        expected_evaluation_replicas=1,
        expected_continuation_horizon=8,
    )
    record = AnchorAuditRecord(
        milestone="C0",
        artifact_path="audit.json",
        artifact_fingerprint="a" * 64,
        random_domain="audit/fixture",
        fit_replicas=1,
        evaluation_replicas=1,
        continuation_horizon=8,
    )
    assert manifest.add(record).records == (record,)
    with pytest.raises(ValueError, match="registered split"):
        manifest.add(replace(record, evaluation_replicas=2))
    with pytest.raises(ValueError, match="registered horizon"):
        manifest.add(replace(record, continuation_horizon=9))


def test_gate_zero_carry_comparison_supports_boolean_episode_boundaries() -> None:
    assert float(_carry_leaf_max_error(
        jnp.asarray([True, False]), jnp.asarray([True, False])
    )) == 0.0
    assert float(_carry_leaf_max_error(
        jnp.asarray([True, False]), jnp.asarray([False, False])
    )) == 1.0
    assert float(_carry_leaf_max_error(
        jnp.asarray([1.0, 2.0]), jnp.asarray([1.0, 2.25])
    )) == pytest.approx(0.25)


def _replay_item(epoch: TargetPolicyEpoch) -> AnchorReplayItem:
    return AnchorReplayItem(
        target_policy_epoch_id=epoch.epoch_id,
        target_policy_fingerprint=combined_policy_fingerprint(epoch),
        partner_source="external",
        partner_run_id="partner-a",
        uniform_or_opportunity="uniform",
        fit_return_mean=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        fit_return_standard_error=(0.1,) * 6,
        collection_step=16,
    )


def _decision(passed: bool, label: str) -> GateDecision:
    return GateDecision(
        passed=passed,
        statistic=1.0 if passed else -1.0,
        lower_bound=0.5 if passed else -1.5,
        upper_bound=1.5 if passed else -0.5,
        sample_count=4,
        random_domain=f"fixture/{label}",
        artifact_fingerprint=(label * 64)[:64],
    )


def test_raw_q_replay_rejects_mixed_continuation_policy_epochs() -> None:
    first = _epoch(0, "first")
    second = _epoch(1, "second")
    replay = AnchorTrainingReplay(epoch_id=0).add((_replay_item(first),))
    assert replay.sample(batch_size=1, key=3, current_epoch=first)
    with pytest.raises(ValueError, match="Old target-policy epoch"):
        replay.sample(batch_size=1, key=3, current_epoch=second)
    with pytest.raises(ValueError, match="Old target-policy epoch"):
        replay.add((_replay_item(second),))


def test_donated_live_target_cannot_delete_frozen_epoch_snapshot() -> None:
    epoch = TargetPolicyEpoch(
        epoch_id=0,
        actor_fingerprint="actor",
        belief_fingerprint="belief",
        raw_q_fingerprint="raw-q",
        target_params={"layer": jnp.arange(128, dtype=jnp.float32)},
        started_update=1,
    )
    live_target = clone_epoch_target_for_live_control(epoch)
    donated_update = jax.jit(
        lambda tree: jax.tree_util.tree_map(lambda value: value + 1.0, tree),
        donate_argnums=(0,),
    )
    updated = donated_update(live_target)
    jax.block_until_ready(updated)
    np.testing.assert_array_equal(
        np.asarray(epoch.target_params["layer"]),
        np.arange(128, dtype=np.float32),
    )


def test_pre_evidence_hidden_code_cannot_change_online_posterior() -> None:
    observation_shape = (5, 5, 39)
    model = build_model(
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=16,
        belief_hidden_dim=16,
        latent_dim=4,
        mixture_components=2,
        belief_embedding_dim=16,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        response_hidden_dim=16,
        modulation_rank=2,
        action_embedding_dim=4,
        log_variance_minimum=-5.0,
        log_variance_maximum=2.0,
    )
    state = initial_policy_state(
        batch_size=2,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=16,
        belief_hidden_dim=16,
        latent_dim=4,
        mixture_components=2,
    )
    observation = jnp.zeros((2,) + observation_shape, dtype=jnp.int32)
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(7),
        example_state=state,
        example_observation=observation,
        partner_code_dim=4,
    )
    # Hidden code labels are deliberately absent from this API.  Replaying the
    # exact same legal history therefore has to be byte-identical.
    _, first = model.apply(
        {"params": params}, state, observation, jnp.zeros((2,)), method=model.step
    )
    _, second = model.apply(
        {"params": params}, state, observation, jnp.zeros((2,)), method=model.step
    )
    np.testing.assert_array_equal(first.mixture_logits, second.mixture_logits)
    np.testing.assert_array_equal(first.mixture_means, second.mixture_means)

    # Legal post-evidence observations may differ and are the only mechanism
    # through which the online posterior is allowed to separate.
    visible = observation.at[1, 2, 2, 11].set(1)
    _, post = model.apply(
        {"params": params}, state, visible, jnp.zeros((2,)), method=model.step
    )
    assert float(jnp.max(jnp.abs(post.mixture_means[1] - first.mixture_means[1]))) > 0.0


def test_generator_source_distillation_and_failed_admission_rollback() -> None:
    source = jnp.asarray([[[2.0, 0.0, -1.0]]], dtype=jnp.float32)
    mask = jnp.ones((1, 1), dtype=jnp.float32)
    assert float(source_logit_distillation_loss(source, source, mask)) == pytest.approx(
        0.0, abs=1.0e-7
    )
    admission = GeneratorAdmission(
        competence_passed=False,
        signature_variance_passed=True,
        oracle_code_lift_passed=True,
        stable_action_regions=2,
        trust_region_kl=0.01,
        behavior_kl=0.01,
        interpolation_uniform_kl=0.1,
        interpolation_uniform_kl_minimum=0.01,
        mean_noninferiority_lcb=-100.0,
        cvar_noninferiority_lcb=-100.0,
    )
    qualified = {"w": jnp.asarray([1.0, 2.0])}
    candidate = {"w": jnp.asarray([9.0, 9.0])}
    restored = rollback_unqualified_generator(candidate, qualified, admission)
    np.testing.assert_array_equal(restored["w"], qualified["w"])


def _episode_batch() -> PartnerEpisodeBatch:
    time, count, code_dim = 400, 2, 3
    valid = np.zeros((time, count), dtype=bool)
    valid[:11] = True
    done = np.zeros((time, count), dtype=bool)
    done[10] = True
    codes = np.broadcast_to(
        np.asarray([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])[None],
        (time, count, code_dim),
    ).copy()
    return PartnerEpisodeBatch(
        observations=np.zeros((time, count, 5, 5, 39), dtype=np.float32),
        actions=np.zeros((time, count), dtype=np.int32),
        raw_rewards=np.zeros((time, count), dtype=np.float32),
        official_shaped_rewards=np.zeros((time, count), dtype=np.float32),
        dones=done,
        codes=codes,
        behavior_log_probabilities=np.zeros((time, count), dtype=np.float32),
        initial_carries=np.zeros((count, 4), dtype=np.float32),
        parameter_version_ids=np.full((time, count), 17, dtype=np.int32),
        valid_mask=valid,
    )


def test_generator_episode_records_do_not_cross_done_code_or_version() -> None:
    batch = _episode_batch()
    validate_complete_episode_batch(batch)
    contaminated_code = batch._replace(codes=np.asarray(batch.codes).copy())
    contaminated_code.codes[5, 0, 0] = 99.0
    with pytest.raises(ValueError, match="code changed"):
        validate_complete_episode_batch(contaminated_code)
    contaminated_done = batch._replace(valid_mask=np.asarray(batch.valid_mask).copy())
    contaminated_done.valid_mask[11, 0] = True
    with pytest.raises(ValueError, match="post-done"):
        validate_complete_episode_batch(contaminated_done)


@pytest.mark.parametrize(
    ("relevant", "identifiable", "calibrated", "expected"),
    (
        (False, False, False, {"C1": False, "C3": False, "C5": False}),
        (True, False, False, {"C1": True, "C3": False, "C5": False}),
        (True, True, True, {"C1": True, "C3": True, "C5": True}),
    ),
)
def test_signal_contract_fixtures(relevant, identifiable, calibrated, expected) -> None:
    assert fixture_contract_outcome(
        decision_relevant=relevant,
        history_identifiable=identifiable,
        regret_calibrated=calibrated,
    ) == expected


def test_pcgrad_cannot_cancel_decision_gradient_and_is_norm_capped() -> None:
    decision = {"belief": jnp.asarray([1.0, 0.0])}
    response = {"belief": jnp.asarray([-3.0, 4.0])}
    projected, metrics = project_response_gradient(response, decision)
    assert float(tree_inner_product(projected, decision)) >= -1.0e-7
    assert float(tree_l2_norm(projected)) <= float(tree_l2_norm(decision)) + 1.0e-7
    assert float(metrics["pre_projection_dot"]) < 0.0


def test_c0_freezes_only_task_trunk_while_base_actor_can_fine_tune() -> None:
    gradients = {
        "task_encoder": {"kernel": jnp.ones((2,))},
        "universal_actor": {
            "base_dense": {"kernel": jnp.ones((2,))},
            "context_residual": {"kernel": jnp.ones((2,))},
        },
    }
    routed = scale_base_policy_gradients(
        gradients,
        task_trunk_scale=jnp.asarray(0.0),
        base_actor_scale=jnp.asarray(1.0),
    )
    np.testing.assert_array_equal(routed["task_encoder"]["kernel"], 0.0)
    np.testing.assert_array_equal(
        routed["universal_actor"]["base_dense"]["kernel"], 1.0
    )
    np.testing.assert_array_equal(
        routed["universal_actor"]["context_residual"]["kernel"], 1.0
    )


def test_conditional_entropy_penalty_is_relative_to_base_not_fixed() -> None:
    deterministic = jnp.asarray([[20.0, -20.0, -20.0]])
    mask = jnp.ones((1,))
    assert float(
        conditional_entropy_noncollapse_penalty(
            deterministic, deterministic, mask, tolerance=0.1
        )
    ) == pytest.approx(0.0, abs=1.0e-7)
    uniform = jnp.zeros((1, 3))
    assert float(
        conditional_entropy_noncollapse_penalty(
            uniform, deterministic, mask, tolerance=0.1
        )
    ) > 0.0


def test_every_formal_seed_survives_qualification_failure_as_fallback() -> None:
    qualification = SignalQualification({"C0": _decision(False, "C0")})
    assert deployment_tier(qualification) == DeploymentTier.OWNER_SP_FALLBACK
    artifacts = {
        seed: FallbackArtifactIdentity(
            seed_index=seed,
            tier=deployment_tier(qualification),
            qualification_fingerprint=qualification.fingerprint,
            source_fingerprint=f"source-{seed}",
        ).to_mapping()
        for seed in range(10)
    }
    assert_all_formal_seeds(artifacts)


def test_full_deployment_calls_calibrated_gate_and_always_on_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def step(self):
            raise AssertionError("Flax method marker only")

        def apply(self, variables, state, observation, gate, method=None):
            del variables, observation, gate, method
            batch = state.previous_action.shape[0]
            output = ModelOutput(
                task_features=jnp.zeros((batch, 2)),
                belief_embedding=jnp.zeros((batch, 2)),
                mixture_logits=jnp.zeros((batch, 1)),
                mixture_means=jnp.zeros((batch, 1, 2)),
                mixture_log_variances=jnp.zeros((batch, 1, 2)),
                support_score=jnp.ones((batch,)),
                base_logits=jnp.asarray([[4.0, 0.0]] * batch),
                residual_logits=jnp.asarray([[-8.0, 8.0]] * batch),
                gate=jnp.ones((batch,)),
                execution_logits=jnp.asarray([[-4.0, 8.0]] * batch),
                state_value=jnp.zeros((batch,)),
                raw_q1=jnp.asarray([[0.0, 10.0]] * batch),
                raw_q2=jnp.asarray([[0.0, 10.0]] * batch),
                action_values=jnp.asarray([[0.0, 10.0]] * batch),
            )
            return state, output

    called = {"count": 0}

    def gate(predicted_gain, support_score, calibration):
        del predicted_gain, support_score, calibration
        called["count"] += 1
        return jnp.ones((1,), dtype=jnp.float32)

    monkeypatch.setattr("experiments.overcooked_v2.deployment.hard_adaptation_gate", gate)
    state = SimpleNamespace(previous_action=jnp.zeros((1,), dtype=jnp.int32))
    config = SimpleNamespace(
        calibration=SimpleNamespace(enable_hard_gate_at_evaluation=True)
    )
    deployment = Deployment(
        ego_run_id="fixture",
        config=config,
        model=FakeModel(),
        params={},
        calibration=empty_calibration(latent_dim=2, alpha=0.05),
        deployment_tier=DeploymentTier.CALIBRATED_FULL_ACTIVE,
        always_on_ablation=False,
    )
    _, _, output, _ = deployment_action(
        deployment=deployment,
        state=state,
        observation=jnp.zeros((1, 5, 5, 39)),
        keys=jax.random.split(jax.random.PRNGKey(0), 1),
    )
    assert called["count"] == 1
    np.testing.assert_array_equal(output.gate, 1.0)

    deployment = replace(deployment, always_on_ablation=True)
    _, _, output, _ = deployment_action(
        deployment=deployment,
        state=state,
        observation=jnp.zeros((1, 5, 5, 39)),
        keys=jax.random.split(jax.random.PRNGKey(1), 1),
    )
    assert called["count"] == 1
    np.testing.assert_array_equal(output.gate, 1.0)
