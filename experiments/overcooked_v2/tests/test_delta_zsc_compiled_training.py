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
    initial = jnp.zeros((lane_count, 1), dtype=jnp.float32)
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
        initial_policy_state=initial,
        initial_target_policy_state=initial,
    )


def _metrics(value):
    names = (
        "approx_kl", "base_approx_kl", "conditional_approx_kl",
        "clip_fraction", "actor_loss", "total_loss", "value_loss",
        "entropy", "information_bottleneck", "mean_raw_reward",
        "mean_shaped_reward", "mean_support_score", "mean_gate",
        "base_entropy", "conditional_entropy",
        "conditional_entropy_noncollapse", "conditional_to_base_kl",
        "conditional_residual_rms",
        "nonfinite_update",
    )
    metrics = {name: value for name in names}
    metrics["nonfinite_update"] = jnp.asarray(0.0, dtype=jnp.float32)
    return metrics


def test_cuda_scan_preserves_sequential_minibatch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def toy_update(*, core, batch, **unused):
        increment = jnp.mean(batch.rewards)
        params = {"weight": core.params["weight"] + increment}
        value = params["weight"]
        return core._replace(params=params), {
            **_metrics(value),
            "optimizer_applied": jnp.asarray(1.0, dtype=jnp.float32),
        }

    monkeypatch.setattr(training, "apply_training_core_update", toy_update)
    initial = TrainingCoreState(
        params={"weight": jnp.asarray(0.0)},
        target_params={"weight": jnp.asarray(9.0)},
        ppo_optimizer_state=jnp.asarray(0),
    )
    batch = _batch()
    schedule = jnp.asarray([[0, 3], [2, 1], [1, 0], [3, 2]], dtype=jnp.int32)
    config = SimpleNamespace(ppo=SimpleNamespace(max_approx_kl=1.0e9))
    expected = initial
    expected_values = []
    for indexes in np.asarray(schedule):
        expected, metrics = toy_update(
            core=expected, batch=training.slice_rollout_lanes(batch, indexes)
        )
        expected_values.append(metrics["total_loss"])
    observed, metrics = jax.jit(
        lambda core: training.scan_training_updates(
            model=None,
            core=core,
            optimizer=None,
            batch=batch,
            schedule=schedule,
            config=config,
        )
    )(initial)
    np.testing.assert_allclose(
        np.asarray(observed.params["weight"]),
        np.asarray(expected.params["weight"]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(metrics["total_loss"]), np.asarray(expected_values), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(observed.target_params["weight"]), 9.0, atol=0.0
    )


def test_kl_stop_makes_remaining_scan_steps_exact_noops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def toy_update(*, core, **unused):
        value = core.params["weight"] + 1.0
        return core._replace(params={"weight": value}), {
            **_metrics(value),
            "base_approx_kl": value,
            "conditional_approx_kl": jnp.asarray(0.0),
            "optimizer_applied": jnp.asarray(1.0),
        }

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
        config=SimpleNamespace(ppo=SimpleNamespace(max_approx_kl=0.5)),
    )
    assert float(result.params["weight"]) == 1.0
    np.testing.assert_array_equal(np.asarray(metrics["optimizer_applied"]), [1, 0, 0])


def test_compiled_callable_reuses_abstract_signature() -> None:
    kernel = CompiledCallable("fixture", lambda value: value * 3.0 + 1.0)
    first = kernel(jnp.ones((4,), dtype=jnp.float32))
    assert kernel.last_call_compiled is True
    second = kernel(jnp.arange(4, dtype=jnp.float32))
    assert kernel.last_call_compiled is False
    np.testing.assert_allclose(np.asarray(first), np.full((4,), 4.0))
    np.testing.assert_allclose(np.asarray(second), np.arange(4) * 3.0 + 1.0)


def test_zero_initialized_residual_hinge_has_finite_zero_gradient() -> None:
    """Regression: r3 residuals start at zero and must not poison PPO."""

    residual = jnp.zeros((3, 4, 6), dtype=jnp.float32)

    def penalty(value):
        rms = training.stable_root_mean_square(value)
        return jnp.maximum(rms - 1.0, 0.0)

    value, gradient = jax.value_and_grad(penalty)(residual)
    assert float(value) == 0.0
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros(residual.shape))
    assert np.all(np.isfinite(np.asarray(gradient)))
