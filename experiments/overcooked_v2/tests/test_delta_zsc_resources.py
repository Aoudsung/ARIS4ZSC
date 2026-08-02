from __future__ import annotations

import pytest

from src.path_c.resources import (
    ResourceLedger,
    _cuda_major_from_ptxas_output,
    aggregate_resource_ledgers,
    gpu_hours_for_wall_seconds,
    measure_policy_inference_latency_ms,
    require_single_cuda_worker,
)


def test_cuda_ptxas_version_parser_requires_cuda_12_or_newer() -> None:
    assert _cuda_major_from_ptxas_output(
        "Cuda compilation tools, release 12.9, V12.9.86"
    ) == 12
    assert _cuda_major_from_ptxas_output("Build cuda_11.8.r11.8/compiler") == 11
    with pytest.raises(RuntimeError, match="parse"):
        _cuda_major_from_ptxas_output("unknown compiler")


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


def _registered_cuda_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "CUDA_VISIBLE_DEVICES": "5",
        "JAX_PLATFORMS": "cuda",
        "DELTA_PHYSICAL_GPU_INDEX": "5",
        "DELTA_PHYSICAL_GPU_UUID": "GPU-test",
        "DELTA_GPU_NAME": "Test CUDA GPU",
        "DELTA_GPU_TOTAL_MEMORY_MIB": "46068",
        "DELTA_GPU_START_MEMORY_USED_MIB": "12",
        "DELTA_GPU_START_UTILIZATION_PERCENT": "0",
        "DELTA_GPU_VOLATILE_UNCORRECTABLE_ECC": "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_formal_cuda_worker_requires_actual_single_jax_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax

    class Device:
        id = 0
        platform = "gpu"
        device_kind = "Test CUDA GPU"

    _registered_cuda_environment(monkeypatch)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda: [Device()])
    observed = require_single_cuda_worker()
    assert observed["jax_backend"] == "gpu"
    assert observed["jax_device_count"] == 1
    assert observed["dispatcher_registration"]["physical_index"] == "5"
    assert observed["dispatcher_registration"]["volatile_uncorrectable_ecc"] == 0


def test_formal_cuda_worker_rejects_cpu_fallback_and_unhealthy_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax

    class Device:
        id = 0
        platform = "cpu"
        device_kind = "CPU"

    _registered_cuda_environment(monkeypatch)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(jax, "devices", lambda: [Device()])
    with pytest.raises(RuntimeError, match="CUDA backend"):
        require_single_cuda_worker()

    monkeypatch.setenv("DELTA_GPU_VOLATILE_UNCORRECTABLE_ECC", "1")
    with pytest.raises(RuntimeError, match="ECC"):
        require_single_cuda_worker()
