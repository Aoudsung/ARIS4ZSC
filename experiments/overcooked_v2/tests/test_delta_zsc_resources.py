from __future__ import annotations

import pytest

from src.path_c.resources import (
    ResourceLedger,
    aggregate_resource_ledgers,
    gpu_hours_for_wall_seconds,
    measure_policy_inference_latency_ms,
)


def test_resource_ledger_rejects_hidden_or_inconsistent_total() -> None:
    ledger = ResourceLedger(ego_policy_steps=10, counterfactual_steps=20)
    payload = dict(ledger.to_mapping())
    assert ResourceLedger.from_mapping(payload) == ledger
    payload["total_training_simulator_steps"] += 1
    with pytest.raises(ValueError, match="total"):
        ResourceLedger.from_mapping(payload)


def test_resource_aggregation_sums_cost_but_not_model_parameters() -> None:
    combined = aggregate_resource_ledgers(
        (
            ResourceLedger(
                ego_policy_steps=10,
                gpu_hours=1.0,
                peak_memory_bytes=3,
                deployable_parameters=100,
                training_only_parameters=50,
            ),
            ResourceLedger(
                ego_policy_steps=10,
                gpu_hours=2.0,
                peak_memory_bytes=7,
                deployable_parameters=100,
                training_only_parameters=50,
            ),
        )
    )
    assert combined.ego_policy_steps == 20
    assert combined.gpu_hours == pytest.approx(3.0)
    assert combined.peak_memory_bytes == 7
    assert combined.deployable_parameters == 100
    assert combined.training_only_parameters == 50


def test_gpu_hours_are_device_hours_not_wall_hours() -> None:
    assert gpu_hours_for_wall_seconds(3_600.0, device_count=4) == pytest.approx(4.0)
    assert gpu_hours_for_wall_seconds(3_600.0, device_count=0) == 0.0
    with pytest.raises(ValueError):
        gpu_hours_for_wall_seconds(-1.0, device_count=1)


def test_latency_measurement_is_positive_for_compiled_recurrent_policy() -> None:
    import jax.numpy as jnp

    class Policy:
        def init_hstate(self, batch_size: int):
            return jnp.zeros((batch_size,), dtype=jnp.float32)

        def compute_action(self, obs, done, hstate, key):
            del done, key
            next_state = hstate + jnp.sum(obs)
            return jnp.asarray(0, dtype=jnp.int32), next_state

    observed = measure_policy_inference_latency_ms(
        Policy(), jnp.ones((2,), dtype=jnp.float32), warmup_steps=1, measurement_steps=2
    )
    assert observed > 0.0
