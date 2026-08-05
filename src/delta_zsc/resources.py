"""Transparent simulator, accelerator, and deployment resource accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ResourceLedger:
    ego_policy_steps: int = 0
    anchor_continuation_steps: int = 0
    upstream_partner_steps: int = 0
    evaluation_steps: int = 0
    calibration_steps: int = 0
    intervention_steps: int = 0
    final_decision_audit_steps: int = 0
    training_gpu_hours: float = 0.0
    shared_gpu_hours: float = 0.0
    measurement_gpu_hours: float = 0.0
    training_wall_clock_hours: float = 0.0
    shared_wall_clock_hours: float = 0.0
    measurement_wall_clock_hours: float = 0.0
    peak_memory_bytes: int = 0
    deployable_parameters: int = 0
    training_only_parameters: int = 0
    inference_latency_ms: float = 0.0

    @property
    def marginal_training_simulator_steps(self) -> int:
        return int(self.ego_policy_steps + self.anchor_continuation_steps)

    @property
    def shared_training_simulator_steps(self) -> int:
        return int(self.upstream_partner_steps)

    @property
    def total_training_simulator_steps(self) -> int:
        return int(
            self.marginal_training_simulator_steps
            + self.shared_training_simulator_steps
        )

    @property
    def post_training_measurement_steps(self) -> int:
        return int(
            self.evaluation_steps
            + self.calibration_steps
            + self.intervention_steps
            + self.final_decision_audit_steps
        )

    @property
    def gpu_hours(self) -> float:
        return float(
            self.training_gpu_hours
            + self.shared_gpu_hours
            + self.measurement_gpu_hours
        )

    @property
    def wall_clock_hours(self) -> float:
        return float(
            self.training_wall_clock_hours
            + self.shared_wall_clock_hours
            + self.measurement_wall_clock_hours
        )

    def plus(self, **increments: Any) -> "ResourceLedger":
        unknown = set(increments) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown resource ledger fields: {sorted(unknown)}")
        values = {}
        for name in self.__dataclass_fields__:
            current = getattr(self, name)
            values[name] = current + increments.get(name, 0)
        return replace(self, **values)

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ResourceLedger":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Resource ledger schema differs.")
        return cls(**payload)


def parameter_count(tree: Any) -> int:
    import jax
    import numpy as np

    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(tree)))


def peak_device_memory_bytes() -> int:
    try:
        import jax

        values = []
        for device in jax.devices():
            stats = device.memory_stats() or {}
            for key in ("peak_bytes_in_use", "peak_pool_bytes", "bytes_in_use"):
                if key in stats:
                    values.append(int(stats[key]))
        return max(values, default=0)
    except Exception:
        return 0


__all__ = ["ResourceLedger", "parameter_count", "peak_device_memory_bytes"]


def gpu_device_count() -> int:
    try:
        import jax

        return sum(str(device.platform).lower() == "gpu" for device in jax.devices())
    except Exception:
        return 0


def gpu_hours_for_wall_seconds(
    wall_seconds: float, *, device_count: int | None = None
) -> float:
    seconds = float(wall_seconds)
    devices = gpu_device_count() if device_count is None else int(device_count)
    if seconds < 0.0 or devices < 0:
        raise ValueError("Wall time and device count must be non-negative.")
    return seconds * devices / 3600.0


def measure_policy_inference_latency_ms(
    policy: Any,
    observation: Any,
    *,
    warmup_steps: int = 20,
    measurement_steps: int = 100,
) -> float:
    """Median synchronized batch-one policy latency after compilation."""

    import statistics
    import time
    import jax
    import jax.numpy as jnp

    if min(int(warmup_steps), int(measurement_steps)) <= 0:
        raise ValueError("Latency sample counts must be positive.")
    state = policy.init_hstate(1)
    obs = jnp.asarray(observation)
    done = jnp.asarray(False)

    def one(current: Any, key: Any):
        return policy.compute_action(obs, done, current, key)

    compiled = jax.jit(one)
    keys = jax.random.split(
        jax.random.PRNGKey(0), int(warmup_steps) + int(measurement_steps)
    )
    for index in range(int(warmup_steps)):
        action, state = compiled(state, keys[index])
        jax.block_until_ready(action)
    values = []
    for index in range(int(measurement_steps)):
        started = time.perf_counter_ns()
        action, state = compiled(state, keys[int(warmup_steps) + index])
        jax.block_until_ready(action)
        values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return float(statistics.median(values))


__all__ = [
    "ResourceLedger",
    "gpu_device_count",
    "gpu_hours_for_wall_seconds",
    "measure_policy_inference_latency_ms",
    "parameter_count",
    "peak_device_memory_bytes",
]
