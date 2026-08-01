from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

from src.path_c.compiled_kernels import CompiledCallable  # noqa: E402
import src.path_c.training as training  # noqa: E402
from src.path_c.types import LossBundle, RolloutBatch, TrainingCoreState  # noqa: E402


def _batch(time_count: int = 2, lane_count: int = 4) -> RolloutBatch:
    transition = jnp.arange(time_count * lane_count, dtype=jnp.float32).reshape(
        (time_count, lane_count)
    ) / 7.0
    states = jnp.zeros((time_count + 1, lane_count), dtype=jnp.float32)
    return RolloutBatch(
        observations=states,
        response_next_observations=transition,
        previous_actions=states.astype(jnp.int32),
        previous_rewards=states,
        episode_starts=states.astype(jnp.bool_),
        action_keys=jnp.zeros((time_count + 1, lane_count, 2), dtype=jnp.uint32),
        gate_overrides=states,
        actions=transition.astype(jnp.int32),
        rewards=transition,
        official_shaped_rewards=jnp.zeros_like(transition),
        official_shaping_factors=jnp.ones_like(transition),
        decision_regret_shaping=jnp.zeros_like(transition),
        shaped_rewards=transition,
        dones=jnp.zeros_like(transition, dtype=jnp.bool_),
        old_log_probabilities=jnp.zeros_like(transition),
        old_values=states,
        ppo_mask=jnp.ones_like(transition),
        partner_codes=jnp.zeros((time_count, lane_count, 1), dtype=jnp.float32),
        partner_sources=jnp.zeros((time_count, lane_count), dtype=jnp.int32),
        partner_run_ids=jnp.zeros((time_count, lane_count), dtype=jnp.int32),
        initial_policy_state=jnp.zeros((lane_count, 1), dtype=jnp.float32),
    )


def _assert_tree_close(left, right) -> None:
    for actual, expected in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
        )


def test_cuda_scan_matches_sequential_minibatch_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixture_loss(*, params, batch, anchors=None, **unused):
        anchor_term = (
            jnp.asarray(0.0, dtype=jnp.float32)
            if anchors is None
            else jnp.asarray(anchors, dtype=jnp.float32)
        )
        residual = (
            params["weight"] * jnp.mean(batch.rewards)
            + params["bias"]
            + 0.125 * anchor_term
        )
        total = jnp.square(residual) + 0.01 * jnp.square(params["weight"])
        return LossBundle(
            total=total,
            metrics={"total_loss": total, "weight": params["weight"]},
        )

    monkeypatch.setattr(training, "compute_loss", fixture_loss)
    params = {
        "weight": jnp.asarray(0.75, dtype=jnp.float32),
        "bias": jnp.asarray(-0.2, dtype=jnp.float32),
    }
    optimizer = optax.adam(2.5e-4, eps=1.0e-5)
    initial = TrainingCoreState(
        params=params,
        target_params=params,
        optimizer_state=optimizer.init(params),
    )
    config = SimpleNamespace(ppo=SimpleNamespace(polyak_coefficient=0.03))
    batch = _batch()
    schedule = jnp.asarray(
        [[0, 3], [2, 1], [1, 0], [3, 2]], dtype=jnp.int32
    )

    reference = initial
    reference_metrics = []
    for indexes in np.asarray(schedule):
        reference, metrics = training.apply_training_core_update(
            model=None,
            core=reference,
            optimizer=optimizer,
            batch=training.slice_rollout_lanes(batch, indexes),
            config=config,
        )
        reference_metrics.append(metrics)
    expected_metrics = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values), *reference_metrics
    )

    scanned, observed_metrics = jax.jit(
        lambda core: training.scan_training_updates(
            model=None,
            core=core,
            optimizer=optimizer,
            batch=batch,
            schedule=schedule,
            config=config,
        )
    )(initial)
    _assert_tree_close(scanned, reference)
    _assert_tree_close(observed_metrics, expected_metrics)


def test_anchor_first_plus_normal_scan_matches_registered_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixture_loss(*, params, batch, anchors=None, **unused):
        anchor_term = 0.0 if anchors is None else anchors
        residual = params["weight"] * jnp.mean(batch.rewards) + anchor_term
        total = jnp.square(residual)
        return LossBundle(total=total, metrics={"total_loss": total})

    monkeypatch.setattr(training, "compute_loss", fixture_loss)
    params = {"weight": jnp.asarray(0.4, dtype=jnp.float32)}
    optimizer = optax.adam(1.0e-3, eps=1.0e-5)
    initial = TrainingCoreState(params, params, optimizer.init(params))
    config = SimpleNamespace(ppo=SimpleNamespace(polyak_coefficient=0.1))
    batch = _batch()
    schedule = jnp.asarray([[0, 1], [2, 3], [1, 3]], dtype=jnp.int32)
    anchor = jnp.asarray(0.75, dtype=jnp.float32)

    reference, first_reference = training.apply_training_core_update(
        model=None,
        core=initial,
        optimizer=optimizer,
        batch=training.slice_rollout_lanes(batch, schedule[0]),
        config=config,
        anchors=anchor,
    )
    remaining_reference = []
    for indexes in np.asarray(schedule[1:]):
        reference, metrics = training.apply_training_core_update(
            model=None,
            core=reference,
            optimizer=optimizer,
            batch=training.slice_rollout_lanes(batch, indexes),
            config=config,
        )
        remaining_reference.append(metrics)

    first_kernel = jax.jit(
        lambda core: training.apply_anchor_first_update(
            model=None,
            core=core,
            optimizer=optimizer,
            batch=batch,
            lane_indexes=schedule[0],
            config=config,
            anchors=anchor,
            quotient_pairs=None,
        )
    )
    scan_kernel = jax.jit(
        lambda core: training.scan_training_updates(
            model=None,
            core=core,
            optimizer=optimizer,
            batch=batch,
            schedule=schedule[1:],
            config=config,
        )
    )
    observed, first_observed = first_kernel(initial)
    observed, remaining_observed = scan_kernel(observed)
    _assert_tree_close(observed, reference)
    _assert_tree_close(first_observed, first_reference)
    expected_remaining = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values), *remaining_reference
    )
    _assert_tree_close(remaining_observed, expected_remaining)


def test_compiled_callable_reuses_abstract_signature() -> None:
    kernel = CompiledCallable("fixture", lambda value: value * 3.0 + 1.0)
    first = kernel(jnp.ones((4,), dtype=jnp.float32))
    assert kernel.last_call_compiled is True
    second = kernel(jnp.arange(4, dtype=jnp.float32))
    assert kernel.last_call_compiled is False
    np.testing.assert_allclose(np.asarray(first), np.full((4,), 4.0))
    np.testing.assert_allclose(np.asarray(second), np.arange(4) * 3.0 + 1.0)
    metadata = kernel.metadata()
    assert len(metadata) == 1
    assert metadata[0]["executable_fingerprint"]


def test_donated_training_core_uses_distinct_online_and_target_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixture_loss(*, params, batch, **unused):
        total = jnp.square(params["weight"] * jnp.mean(batch.rewards))
        return LossBundle(total=total, metrics={"total_loss": total})

    monkeypatch.setattr(training, "compute_loss", fixture_loss)
    params = {"weight": jnp.asarray(0.75, dtype=jnp.float32)}
    target = jax.tree_util.tree_map(jnp.copy, params)
    assert (
        params["weight"].unsafe_buffer_pointer()
        != target["weight"].unsafe_buffer_pointer()
    )
    optimizer = optax.adam(2.5e-4, eps=1.0e-5)
    core = TrainingCoreState(
        params=params,
        target_params=target,
        optimizer_state=optimizer.init(params),
    )
    config = SimpleNamespace(ppo=SimpleNamespace(polyak_coefficient=0.03))
    batch = _batch()
    schedule = jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32)
    kernel = CompiledCallable(
        "donated_training_core_fixture",
        lambda value, current_batch, current_schedule: (
            training.scan_training_updates(
                model=None,
                core=value,
                optimizer=optimizer,
                batch=current_batch,
                schedule=current_schedule,
                config=config,
            )
        ),
        donate_argnums=(0,),
    )
    completed, metrics = kernel(core, batch, schedule)
    for leaf in jax.tree_util.tree_leaves(completed):
        leaf.block_until_ready()
    assert np.isfinite(np.asarray(metrics["total_loss"])).all()
