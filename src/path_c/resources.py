"""Auditable simulator, hardware, and parameter accounting.

The ledger deliberately separates the deployed ego trajectory from every
training-only source of environment interaction.  No caller may record a
negative count or hide an auxiliary source inside ``ego_policy_steps``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ResourceLedger:
    ego_policy_steps: int = 0
    partner_training_steps: int = 0
    counterfactual_steps: int = 0
    state_collection_steps: int = 0
    calibration_steps: int = 0
    evaluation_steps: int = 0
    # r3 attribution fields.  These partition the aggregate categories above
    # and are therefore not added a second time to the total.
    base_distillation_steps: int = 0
    partner_source_training_steps: int = 0
    generator_training_steps: int = 0
    training_anchor_steps: int = 0
    audit_anchor_steps: int = 0
    gpu_hours: float = 0.0
    peak_memory_bytes: int = 0
    deployable_parameters: int = 0
    training_only_parameters: int = 0
    inference_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "ego_policy_steps",
            "partner_training_steps",
            "counterfactual_steps",
            "state_collection_steps",
            "calibration_steps",
            "evaluation_steps",
            "base_distillation_steps",
            "partner_source_training_steps",
            "generator_training_steps",
            "training_anchor_steps",
            "audit_anchor_steps",
            "peak_memory_bytes",
            "deployable_parameters",
            "training_only_parameters",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"Resource field {name} must be a non-negative integer.")
        for name in ("gpu_hours", "inference_latency_ms"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"Resource field {name} must be non-negative.")
        if self.generator_training_steps + self.partner_source_training_steps > self.partner_training_steps:
            raise ValueError("Partner-source and generator attributions exceed partner training.")
        if self.training_anchor_steps + self.audit_anchor_steps > self.counterfactual_steps:
            raise ValueError("Training/audit anchor attributions exceed counterfactual steps.")
        if self.base_distillation_steps > self.state_collection_steps:
            raise ValueError("Base-distillation attribution exceeds state collection.")

    @property
    def total_training_simulator_steps(self) -> int:
        return int(
            self.ego_policy_steps
            + self.partner_training_steps
            + self.counterfactual_steps
            + self.state_collection_steps
            + self.calibration_steps
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            **asdict(self),
            "total_training_simulator_steps": self.total_training_simulator_steps,
        }

    def plus(self, **increments: int | float) -> "ResourceLedger":
        values = asdict(self)
        unknown = set(increments) - set(values)
        if unknown:
            raise ValueError(f"Unknown resource fields: {sorted(unknown)}")
        for name, value in increments.items():
            values[name] += value
        return ResourceLedger(**values)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ResourceLedger":
        fields = set(cls.__dataclass_fields__)
        accepted = fields | {"total_training_simulator_steps"}
        if not isinstance(payload, Mapping) or set(payload) != accepted:
            raise ValueError("Resource ledger fields differ from the registered schema.")
        ledger = cls(**{name: payload[name] for name in fields})
        if int(payload["total_training_simulator_steps"]) != ledger.total_training_simulator_steps:
            raise ValueError("Resource ledger total is inconsistent with its components.")
        return ledger


def aggregate_resource_ledgers(ledgers: Sequence[ResourceLedger]) -> ResourceLedger:
    """Aggregate independent runs without inflating per-deployment quantities."""

    values = tuple(ledgers)
    if not values:
        raise ValueError("At least one resource ledger is required.")
    deployable = {item.deployable_parameters for item in values if item.deployable_parameters}
    if len(deployable) > 1:
        raise ValueError("A method's deployable parameter counts are inconsistent.")
    training_only = {
        item.training_only_parameters for item in values if item.training_only_parameters
    }
    if len(training_only) > 1:
        raise ValueError("A method's training-only parameter counts are inconsistent.")
    return ResourceLedger(
        ego_policy_steps=sum(item.ego_policy_steps for item in values),
        partner_training_steps=sum(item.partner_training_steps for item in values),
        counterfactual_steps=sum(item.counterfactual_steps for item in values),
        state_collection_steps=sum(item.state_collection_steps for item in values),
        calibration_steps=sum(item.calibration_steps for item in values),
        evaluation_steps=sum(item.evaluation_steps for item in values),
        base_distillation_steps=sum(item.base_distillation_steps for item in values),
        partner_source_training_steps=sum(item.partner_source_training_steps for item in values),
        generator_training_steps=sum(item.generator_training_steps for item in values),
        training_anchor_steps=sum(item.training_anchor_steps for item in values),
        audit_anchor_steps=sum(item.audit_anchor_steps for item in values),
        gpu_hours=sum(item.gpu_hours for item in values),
        peak_memory_bytes=max(item.peak_memory_bytes for item in values),
        deployable_parameters=(0 if not deployable else next(iter(deployable))),
        training_only_parameters=(
            0 if not training_only else next(iter(training_only))
        ),
        inference_latency_ms=max(item.inference_latency_ms for item in values),
    )


def parameter_count(tree: Any) -> int:
    """Count scalar values in a JAX/NumPy parameter tree."""

    import jax

    return int(sum(int(getattr(leaf, "size", 1)) for leaf in jax.tree_util.tree_leaves(tree)))


def peak_device_memory_bytes() -> int:
    """Best-effort peak accelerator allocation from JAX device counters."""

    try:
        import jax

        values = []
        for device in jax.devices():
            stats = device.memory_stats() or {}
            for name in ("peak_bytes_in_use", "peak_pool_bytes", "bytes_in_use"):
                if name in stats:
                    values.append(int(stats[name]))
        return max(values, default=0)
    except Exception:
        return 0


def gpu_device_count() -> int:
    """Return the number of visible JAX GPU devices used by this process.

    GPU-hours are accelerator-device hours, not wall-clock hours.  CPU-only
    mechanical tests therefore report zero GPU-hours rather than relabelling
    CPU time as accelerator use.
    """

    try:
        import jax

        return sum(str(device.platform).lower() == "gpu" for device in jax.devices())
    except Exception:
        return 0


def _cuda_major_from_ptxas_output(output: str) -> int:
    match = re.search(
        r"(?:release\s+|V|cuda[_ -]?)(\d+)(?:\.\d+)?",
        str(output),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError("Could not parse the registered ptxas CUDA version.")
    return int(match.group(1))


def configure_bundled_cuda_toolchain() -> Mapping[str, Any]:
    """Prepend the CUDA-12 ptxas shipped with the registered Python runtime.

    The server's global PATH may expose an older CUDA toolkit even when JAX is
    installed with its CUDA-12 plugin.  Formal workers must use the ptxas from
    ``nvidia-cuda-nvcc-cu12`` and fail closed if it is absent or older than
    CUDA 12.  This changes only compiler discovery, never model computation.
    """

    override = os.environ.get("DELTA_CUDA_PTXAS", "").strip()
    if override:
        candidate = Path(override).resolve()
    else:
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
        locations = () if spec is None else tuple(spec.submodule_search_locations or ())
        candidates = tuple(Path(location) / "bin" / "ptxas" for location in locations)
        candidate = next((path.resolve() for path in candidates if path.is_file()), None)
        if candidate is None:
            raise RuntimeError(
                "Formal CUDA execution requires bundled nvidia-cuda-nvcc-cu12 ptxas."
            )
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RuntimeError(f"Registered CUDA ptxas is not executable: {candidate}")
    completed = subprocess.run(
        (str(candidate), "--version"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version_text = "\n".join((completed.stdout, completed.stderr)).strip()
    cuda_major = _cuda_major_from_ptxas_output(version_text)
    if cuda_major < 12:
        raise RuntimeError(
            f"Formal JAX CUDA plugin requires CUDA-12 ptxas; observed major {cuda_major}."
        )
    binary_directory = str(candidate.parent)
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = binary_directory + (os.pathsep + existing if existing else "")
    return {
        "ptxas_path": str(candidate),
        "cuda_major": cuda_major,
        "version_output": version_text,
        "source": "nvidia-cuda-nvcc-cu12",
    }


def require_single_cuda_worker() -> Mapping[str, Any]:
    """Fail closed unless a formal worker is bound to one healthy CUDA GPU.

    The dispatcher-provided facts are checked before JAX is initialized.  The
    JAX runtime is then checked independently, so environment variables alone
    can never be mistaken for evidence that computation is actually executing
    on CUDA.  Formal entry points call this before creating a run identity or
    any training output.
    """

    required_environment = {
        "physical_index": "DELTA_PHYSICAL_GPU_INDEX",
        "uuid": "DELTA_PHYSICAL_GPU_UUID",
        "name": "DELTA_GPU_NAME",
        "total_memory_mib": "DELTA_GPU_TOTAL_MEMORY_MIB",
        "start_memory_used_mib": "DELTA_GPU_START_MEMORY_USED_MIB",
        "start_utilization_percent": "DELTA_GPU_START_UTILIZATION_PERCENT",
        "volatile_uncorrectable_ecc": "DELTA_GPU_VOLATILE_UNCORRECTABLE_ECC",
    }
    registered = {
        name: os.environ.get(environment_name, "").strip()
        for name, environment_name in required_environment.items()
    }
    missing = sorted(name for name, value in registered.items() if not value)
    if missing:
        raise RuntimeError(
            "Formal CUDA worker metadata is incomplete: " + ", ".join(missing)
        )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        raise RuntimeError(
            "Formal execution requires CUDA_VISIBLE_DEVICES to select exactly "
            "one physical GPU."
        )
    if visible != registered["physical_index"]:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES differs from the registered physical GPU index."
        )
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if platforms != "cuda":
        raise RuntimeError("Formal workers require JAX_PLATFORMS=cuda.")

    try:
        total_memory_mib = int(registered["total_memory_mib"])
        start_memory_used_mib = int(registered["start_memory_used_mib"])
        start_utilization_percent = int(registered["start_utilization_percent"])
        volatile_uncorrectable_ecc = int(
            registered["volatile_uncorrectable_ecc"]
        )
    except ValueError as error:
        raise RuntimeError("Formal CUDA worker metadata is not integral.") from error
    if total_memory_mib <= 0:
        raise RuntimeError("Registered CUDA total memory must be positive.")
    if not 0 <= start_memory_used_mib <= 1_024:
        raise RuntimeError(
            "Formal CUDA worker started above the registered 1 GiB memory limit."
        )
    if not 0 <= start_utilization_percent <= 10:
        raise RuntimeError(
            "Formal CUDA worker started above the registered 10% utilization limit."
        )
    if volatile_uncorrectable_ecc != 0:
        raise RuntimeError("Formal CUDA worker has volatile uncorrectable ECC errors.")

    import jax

    backend = str(jax.default_backend()).lower()
    devices = tuple(jax.devices())
    if backend != "gpu":
        raise RuntimeError(
            f"Formal worker did not initialize the JAX CUDA backend: {backend!r}."
        )
    if len(devices) != 1 or str(devices[0].platform).lower() != "gpu":
        raise RuntimeError(
            "Formal worker must expose exactly one JAX CUDA device; observed "
            f"{len(devices)} devices."
        )
    device = devices[0]
    return {
        "cuda_visible_devices": visible,
        "jax_platforms": platforms,
        "jax_backend": backend,
        "jax_device_count": 1,
        "jax_device": {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        },
        "dispatcher_registration": {
            **registered,
            "total_memory_mib": total_memory_mib,
            "start_memory_used_mib": start_memory_used_mib,
            "start_utilization_percent": start_utilization_percent,
            "volatile_uncorrectable_ecc": volatile_uncorrectable_ecc,
        },
    }


def gpu_hours_for_wall_seconds(
    wall_seconds: float, *, device_count: int | None = None
) -> float:
    seconds = float(wall_seconds)
    visible = gpu_device_count()
    # The DELTA and upstream/calibration paths place one unsharded computation
    # on JAX's default device. Multi-device Official trainers pass their device
    # count explicitly.
    devices = (1 if visible else 0) if device_count is None else int(device_count)
    if seconds < 0.0 or devices < 0:
        raise ValueError("Wall time and GPU device count must be non-negative.")
    return seconds * devices / 3600.0


def measure_policy_inference_latency_ms(
    policy: Any,
    observation: Any,
    *,
    warmup_steps: int = 20,
    measurement_steps: int = 100,
) -> float:
    """Measure median compiled batch-one stochastic policy latency.

    Every recorded step is explicitly synchronized.  The measurement includes
    recurrent-state update and stochastic action sampling, but excludes the
    environment transition and one-time compilation.  This is the deployment
    quantity registered by the formal resource table.
    """

    if int(warmup_steps) < 1 or int(measurement_steps) < 1:
        raise ValueError("Latency warmup and measurement counts must be positive.")

    import jax
    import jax.numpy as jnp

    state = policy.init_hstate(1)
    obs = jnp.asarray(observation)
    done = jnp.asarray(False, dtype=jnp.bool_)

    def one(current: Any, key: Any) -> tuple[Any, Any]:
        action, next_state = policy.compute_action(obs, done, current, key)
        return next_state, action

    compiled = jax.jit(one)
    keys = jax.random.split(
        jax.random.PRNGKey(0), int(warmup_steps) + int(measurement_steps)
    )
    for index in range(int(warmup_steps)):
        state, action = compiled(state, keys[index])
        jax.block_until_ready(action)

    timings = []
    offset = int(warmup_steps)
    for index in range(int(measurement_steps)):
        started = time.perf_counter_ns()
        state, action = compiled(state, keys[offset + index])
        jax.block_until_ready(action)
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return float(statistics.median(timings))


def official_main_steps(*, num_envs: int, rollout_length: int, updates: int) -> int:
    if min(int(num_envs), int(rollout_length), int(updates)) <= 0:
        raise ValueError("Official step-count factors must be positive.")
    return int(num_envs) * int(rollout_length) * int(updates)


def delta_anchor_attempted_steps(
    *,
    trigger_count: int,
    ordinary_worlds: int,
    matched_worlds: int,
    action_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    continuation_horizon: int,
) -> int:
    factors = (
        trigger_count,
        ordinary_worlds + matched_worlds,
        action_count,
        fit_replicas + evaluation_replicas,
        continuation_horizon,
    )
    if any(int(value) < 0 for value in factors):
        raise ValueError("Anchor budget factors cannot be negative.")
    result = 1
    for value in factors:
        result *= int(value)
    return result


def r3_training_anchor_attempted_steps(
    *,
    trigger_count: int,
    ordinary_candidates: int,
    matched_code_candidates: int,
    selected_ordinary: int,
    selected_matched_code: int,
    action_count: int,
    pilot_replicas: int,
    fit_replicas: int,
    continuation_horizon: int,
    probe_steps: int,
) -> int:
    """Exact attempted transitions for r3's pilot/fit training anchors."""

    values = (
        trigger_count,
        ordinary_candidates,
        matched_code_candidates,
        selected_ordinary,
        selected_matched_code,
        action_count,
        pilot_replicas,
        fit_replicas,
        continuation_horizon,
        probe_steps,
    )
    if any(int(value) < 0 for value in values):
        raise ValueError("Training-anchor budget factors cannot be negative.")
    per_trigger = (
        int(ordinary_candidates)
        * int(action_count)
        * int(pilot_replicas)
        * int(continuation_horizon)
        + 2
        * int(matched_code_candidates)
        * int(action_count)
        * int(pilot_replicas)
        * int(continuation_horizon)
        + (int(selected_ordinary) + 2 * int(selected_matched_code))
        * int(action_count)
        * int(fit_replicas)
        * int(continuation_horizon)
        + 2 * int(matched_code_candidates) * int(probe_steps)
    )
    return int(trigger_count) * per_trigger


def r3_audit_anchor_attempted_steps(
    *,
    milestone_count: int,
    ordinary_states: int,
    matched_code_pairs: int,
    action_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    continuation_horizon: int,
    probe_steps: int,
) -> int:
    """Exact attempted transitions for independent r3 audit anchors."""

    values = (
        milestone_count,
        ordinary_states,
        matched_code_pairs,
        action_count,
        fit_replicas,
        evaluation_replicas,
        continuation_horizon,
        probe_steps,
    )
    if any(int(value) < 0 for value in values):
        raise ValueError("Audit-anchor budget factors cannot be negative.")
    worlds = int(ordinary_states) + 2 * int(matched_code_pairs)
    continuation = (
        worlds
        * int(action_count)
        * (int(fit_replicas) + int(evaluation_replicas))
        * int(continuation_horizon)
    )
    evidence = 2 * int(matched_code_pairs) * int(probe_steps)
    return int(milestone_count) * (continuation + evidence)


__all__ = [
    "ResourceLedger",
    "aggregate_resource_ledgers",
    "delta_anchor_attempted_steps",
    "r3_audit_anchor_attempted_steps",
    "r3_training_anchor_attempted_steps",
    "gpu_device_count",
    "gpu_hours_for_wall_seconds",
    "configure_bundled_cuda_toolchain",
    "measure_policy_inference_latency_ms",
    "official_main_steps",
    "parameter_count",
    "peak_device_memory_bytes",
    "require_single_cuda_worker",
]
