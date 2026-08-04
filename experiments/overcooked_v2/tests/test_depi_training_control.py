"""DEPI compiled training-control tests.

Specification entries covered (docs/METHOD_SPEC.md):
- §3.3 single combined gradient step (toy ``apply_training_core_update``).
- §3.4 combined_policy_kl early stop aborts the remaining minibatches.
- Nonfinite updates abort the scan (official safeguard retained).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.compiled_kernels import CompiledCallable  # noqa: E402
import src.path_c.training as training  # noqa: E402
from src.path_c.types import RolloutBatch, TrainingCoreState  # noqa: E402


def _batch(time_count: int = 2, lane_count: int = 4) -> RolloutBatch:
    transition = jnp.arange(time_count * lane_count, dtype=jnp.float32).reshape(
        (time_count, lane_count)
    ) / 7.0
    states = jnp.zeros((time_count + 1, lane_count), dtype=jnp.float32)
    return RolloutBatch(
        observations=states,
        response_next_observations=transition,
        previous_actions=states.astype(jnp.int32),
        episode_starts=states.astype(jnp.bool_),
        action_keys=jnp.zeros((time_count + 1, lane_count, 2), dtype=jnp.uint32),
        context_dropout_masks=states.astype(jnp.bool_),
        actions=transition.astype(jnp.int32),
        rewards=transition,
        official_shaped_rewards=jnp.zeros_like(transition),
        official_shaping_factors=jnp.ones_like(transition),
        shaped_rewards=transition,
        dones=jnp.zeros_like(transition, dtype=jnp.bool_),
        old_log_probabilities=jnp.zeros_like(transition),
        old_values=states,
        behavior_probabilities=jnp.full_like(transition, 0.5),
        ppo_mask=jnp.ones_like(transition),
        partner_sources=jnp.zeros((time_count, lane_count), dtype=jnp.int32),
        partner_members=jnp.zeros((time_count, lane_count), dtype=jnp.int32),
        partner_family_ids=jnp.zeros((time_count, lane_count), dtype=jnp.int32),
        partner_checkpoint_stages=jnp.zeros(
            (time_count, lane_count), dtype=jnp.float32
        ),
        partner_run_ids=jnp.zeros((time_count, lane_count), dtype=jnp.int32),
        initial_policy_state=jnp.zeros((lane_count, 1), dtype=jnp.float32),
        initial_target_policy_state=jnp.zeros((lane_count, 1), dtype=jnp.float32),
    )


def _metrics(
    value: object, *, kl_stop: object = 0.0, nonfinite: object = 0.0
) -> dict[str, object]:
    """Exact metric key set emitted by ``apply_training_core_update`` (§3.3).

    The scan's skip branch must return the identical pytree structure, so the
    toy update has to emit exactly these keys.
    """

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return {
        "actor_loss": value,
        "approx_kl": zero,
        "clip_fraction": zero,
        "response_total_loss": zero,
        "response_visibility_loss": zero,
        "response_position_loss": zero,
        "response_direction_loss": zero,
        "response_inventory_loss": zero,
        "response_event_loss": zero,
        "signature_loss": zero,
        "signature_huber_loss": zero,
        "signature_hinge_loss": zero,
        "decision_policy_loss": zero,
        "decision_supervision_confidence": zero,
        "separation_loss": zero,
        "capability_consistency_loss": zero,
        "anchor_effective_sample_size": zero,
        "anchor_use_count_max": zero,
        "anchor_auxiliary_actual_total_weight": zero,
        "auxiliary_gradient_norm": zero,
        "ppo_total_loss": zero,
        "total_loss": value,
        "value_loss": zero,
        "entropy": zero,
        "mean_raw_reward": zero,
        "mean_shaped_reward": zero,
        "mean_posterior_entropy": zero,
        "context_dropout_fraction": zero,
        "method_variant_code": zero,
        "combined_policy_kl": zero,
        "optimizer_applied": 1.0 - jnp.asarray(nonfinite, dtype=jnp.float32),
        "nonfinite_update": jnp.asarray(nonfinite, dtype=jnp.float32),
        "kl_early_stop": jnp.asarray(kl_stop, dtype=jnp.float32),
    }


def test_cuda_scan_preserves_all_minibatch_updates_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def toy_update(*, core, batch, **unused):
        increment = jnp.mean(batch.rewards)
        next_core = core._replace(params={"weight": core.params["weight"] + increment})
        return next_core, _metrics(next_core.params["weight"])

    monkeypatch.setattr(training, "apply_training_core_update", toy_update)
    core = TrainingCoreState(
        params={"weight": jnp.asarray(0.0)},
        target_params={"weight": jnp.asarray(9.0)},
        ppo_optimizer_state=jnp.asarray(0),
    )
    schedule = jnp.asarray([[0, 3], [2, 1], [1, 0], [3, 2]], dtype=jnp.int32)
    result, metrics = jax.jit(
        lambda value: training.scan_training_updates(
            model=None,
            core=value,
            optimizer=None,
            batch=_batch(),
            schedule=schedule,
            config=SimpleNamespace(ppo=SimpleNamespace()),
        )
    )(core)
    expected = sum(
        float(np.mean(np.asarray(training.slice_rollout_lanes(_batch(), row).rewards)))
        for row in np.asarray(schedule)
    )
    assert float(result.params["weight"]) == pytest.approx(expected)
    assert int(np.asarray(jnp.sum(metrics["optimizer_applied"]))) == 4
    assert np.asarray(metrics["training_aborted"]).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_combined_policy_kl_early_stop_aborts_remaining_minibatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # METHOD_SPEC §3.4: once combined_policy_kl exceeds the frozen 0.04 gate
    # (emitted here as kl_early_stop=1), no further minibatch is applied.
    def toy_update(*, core, **unused):
        value = core.params["weight"] + 1.0
        kl_stop = value >= 2.0
        next_core = core._replace(params={"weight": value})
        return next_core, _metrics(value, kl_stop=kl_stop)

    monkeypatch.setattr(training, "apply_training_core_update", toy_update)
    core = TrainingCoreState(
        params={"weight": jnp.asarray(0.0)},
        target_params={"weight": jnp.asarray(0.0)},
        ppo_optimizer_state=jnp.asarray(0),
    )
    result, metrics = training.scan_training_updates(
        model=None,
        core=core,
        optimizer=None,
        batch=_batch(),
        schedule=jnp.asarray([[0, 1], [2, 3], [0, 2]], dtype=jnp.int32),
        config=SimpleNamespace(ppo=SimpleNamespace()),
    )
    assert float(result.params["weight"]) == 2.0
    assert np.asarray(metrics["optimizer_applied"]).tolist() == [1.0, 1.0, 0.0]
    assert np.asarray(metrics["kl_early_stop"]).tolist() == [0.0, 1.0, 0.0]
    assert np.asarray(metrics["training_aborted"]).tolist() == [0.0, 0.0, 0.0]


def test_any_kl_stop_event_reduces_the_next_update_epoch_count() -> None:
    next_epochs, fraction, occurred = training.adapt_effective_update_epochs(
        jnp.asarray([0.0, 1.0, 0.0, 0.0]),
        current_epochs=4,
        maximum_epochs=4,
    )
    assert occurred
    assert fraction == pytest.approx(0.25)
    assert next_epochs == 3


def test_clean_update_recovers_one_epoch_without_exceeding_the_maximum() -> None:
    next_epochs, fraction, occurred = training.adapt_effective_update_epochs(
        jnp.zeros((4,)), current_epochs=2, maximum_epochs=4
    )
    assert not occurred
    assert fraction == 0.0
    assert next_epochs == 3
    capped, _, _ = training.adapt_effective_update_epochs(
        jnp.zeros((4,)), current_epochs=4, maximum_epochs=4
    )
    assert capped == 4


def test_nonfinite_minibatch_aborts_the_remaining_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def toy_update(*, core, **unused):
        next_value = core.params["weight"] + 1.0
        failed = next_value >= 2.0
        next_core = jax.lax.cond(
            failed,
            lambda _: core,
            lambda _: core._replace(params={"weight": next_value}),
            operand=None,
        )
        return next_core, _metrics(next_value, nonfinite=failed)

    monkeypatch.setattr(training, "apply_training_core_update", toy_update)
    core = TrainingCoreState(
        params={"weight": jnp.asarray(0.0)},
        target_params={"weight": jnp.asarray(0.0)},
        ppo_optimizer_state=jnp.asarray(0),
    )
    result, metrics = training.scan_training_updates(
        model=None,
        core=core,
        optimizer=None,
        batch=_batch(),
        schedule=jnp.asarray([[0, 1], [2, 3], [0, 2]], dtype=jnp.int32),
        config=SimpleNamespace(ppo=SimpleNamespace()),
    )
    assert float(result.params["weight"]) == 1.0
    assert np.asarray(metrics["optimizer_applied"]).tolist() == [1.0, 0.0, 0.0]
    assert np.asarray(metrics["training_aborted"]).tolist() == [0.0, 1.0, 0.0]


def test_compiled_callable_reuses_one_shape_signature() -> None:
    compiled = CompiledCallable("add", lambda left, right: left + right)
    np.testing.assert_allclose(np.asarray(compiled(jnp.ones((3,)), jnp.ones((3,)))), 2.0)
    assert compiled.last_call_compiled
    np.testing.assert_allclose(np.asarray(compiled(jnp.zeros((3,)), jnp.ones((3,)))), 1.0)
    assert not compiled.last_call_compiled
    assert len(compiled.metadata()) == 1


def test_context_dropout_mask_is_a_replayed_batch_field() -> None:
    batch = _batch()
    changed = batch._replace(
        context_dropout_masks=batch.context_dropout_masks.at[0, 1].set(True)
    )
    sliced = training.slice_rollout_lanes(changed, jnp.asarray([1, 3]))
    assert bool(np.asarray(sliced.context_dropout_masks[0, 0]))
    assert sliced.context_dropout_masks.shape == (3, 2)


def test_global_anchor_payload_is_active_only_once_per_outer_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def toy_update(*, core, auxiliary_active, **unused):
        return core, _metrics(jnp.asarray(auxiliary_active, dtype=jnp.float32))

    monkeypatch.setattr(training, "apply_training_core_update", toy_update)
    core = TrainingCoreState(
        params={"weight": jnp.asarray(0.0)},
        target_params={"weight": jnp.asarray(0.0)},
        ppo_optimizer_state=jnp.asarray(0),
    )
    _, metrics = training.scan_training_updates(
        model=None,
        core=core,
        optimizer=None,
        batch=_batch(),
        schedule=jnp.asarray([[0, 1], [2, 3], [0, 2]], dtype=jnp.int32),
        config=SimpleNamespace(ppo=SimpleNamespace()),
        anchors="global-anchor-payload",
    )
    assert np.asarray(metrics["actor_loss"]).tolist() == [1.0, 0.0, 0.0]
