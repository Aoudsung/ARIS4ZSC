"""Resource accounting helpers shared by CETR experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import time
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ResourceLedger:
    """Auditable simulator, compute, memory, and deployment measurements."""

    ego_policy_steps: int = 0
    upstream_partner_steps: int = 0
    evaluation_steps: int = 0
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
    def total_training_simulator_steps(self) -> int:
        return int(self.ego_policy_steps) + int(self.upstream_partner_steps)

    @property
    def marginal_training_simulator_steps(self) -> int:
        return int(self.ego_policy_steps)

    @property
    def shared_training_simulator_steps(self) -> int:
        return int(self.upstream_partner_steps)

    @property
    def post_training_measurement_steps(self) -> int:
        return int(self.evaluation_steps)

    @property
    def gpu_hours(self) -> float:
        return float(self.training_gpu_hours) + float(self.shared_gpu_hours)

    @property
    def wall_clock_hours(self) -> float:
        return float(self.training_wall_clock_hours) + float(self.shared_wall_clock_hours)

    def plus(self, **increments: Any) -> "ResourceLedger":
        names = {field.name for field in fields(self)}
        unknown = set(increments) - names
        if unknown:
            raise ValueError(f"Unknown resource fields: {sorted(unknown)!r}")
        values = asdict(self)
        for name, increment in increments.items():
            values[name] = values[name] + increment
        return replace(self, **values)

    def to_mapping(self) -> Mapping[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ResourceLedger":
        names = {field.name for field in fields(cls)}
        if not isinstance(payload, Mapping) or set(payload) != names:
            raise ValueError("Resource ledger fields differ.")
        integer_fields = {
            "ego_policy_steps",
            "upstream_partner_steps",
            "evaluation_steps",
            "peak_memory_bytes",
            "deployable_parameters",
            "training_only_parameters",
        }
        values = {
            name: (int(payload[name]) if name in integer_fields else float(payload[name]))
            for name in names
        }
        return cls(**values)


def parameter_count(tree: Any) -> int:
    """Count scalar leaves in a JAX parameter tree."""

    import jax

    return sum(
        int(getattr(leaf, "size", 0))
        for leaf in jax.tree_util.tree_leaves(tree)
        if leaf is not None
    )


def gpu_device_count() -> int:
    """Return the number of visible GPU devices reported by JAX."""

    import jax

    return sum(1 for device in jax.devices() if device.platform == "gpu")


def gpu_hours_for_wall_seconds(seconds: float) -> float:
    """Convert wall-clock seconds to visible-GPU hours."""

    return float(seconds) * float(gpu_device_count()) / 3600.0


def peak_device_memory_bytes() -> int:
    """Read the largest peak allocation reported by visible GPU devices."""

    import jax

    peak = 0
    for device in jax.devices():
        if device.platform != "gpu":
            continue
        stats = device.memory_stats()
        if not stats:
            continue
        peak = max(
            peak,
            int(stats.get("peak_bytes_in_use", 0)),
            int(stats.get("bytes_in_use", 0)),
        )
    return peak


def _block_until_ready(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()


def measure_policy_inference_latency_ms(
    policy: Any,
    observation: Any,
    *,
    warmup_iterations: int = 2,
    measured_iterations: int = 10,
) -> float:
    """Measure synchronous policy inference latency on one fixed observation."""

    import jax
    import jax.numpy as jnp

    warmups = int(warmup_iterations)
    measurements = int(measured_iterations)
    if warmups < 0 or measurements <= 0:
        raise ValueError("Inference measurement counts are invalid.")
    keys = jax.random.split(
        jax.random.PRNGKey(0), warmups + measurements
    )
    state = policy.init_hstate(1, keys[0])
    done = jnp.zeros((1,), dtype=jnp.bool_)
    for key in keys[:warmups]:
        action, state = policy.compute_action(observation, done, state, key)
        _block_until_ready((action, state))

    started = time.perf_counter()
    for key in keys[warmups:]:
        action, state = policy.compute_action(observation, done, state, key)
        _block_until_ready((action, state))
    elapsed = time.perf_counter() - started
    return 1000.0 * elapsed / float(measurements)


__all__ = [
    "ResourceLedger",
    "gpu_device_count",
    "gpu_hours_for_wall_seconds",
    "measure_policy_inference_latency_ms",
    "parameter_count",
    "peak_device_memory_bytes",
]
