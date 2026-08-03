"""Stable CUDA executable boundaries for the DEPI foundation update.

The active default path compiles only rollout, target-context and the
combined four-loss PPO scan (METHOD_SPEC §3.3).  The legacy V6 kernels
(detached raw-Q retrace, counterfactual anchor heads, separate response
head updates, belief-objective gradient collection and decision-regret
shaping) are disconnected from the default path by task #10's §5–§7
boundary: their slots are ``None`` unless a future task re-enables them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from .runner import (
    DECISION_REGRET_STATE_CHUNK_SIZE,
    collect_rollout,
    target_context_sequence,
)
from .training import scan_training_updates


class CompiledMemoryLimitError(RuntimeError):
    pass


def _argument_signature(arguments: tuple[Any, ...]) -> str:
    import jax

    leaves, structure = jax.tree_util.tree_flatten(arguments)
    payload = {
        "structure": str(structure),
        "leaves": [
            {
                "shape": list(getattr(leaf, "shape", ())),
                "dtype": str(getattr(leaf, "dtype", type(leaf).__name__)),
                "weak_type": bool(getattr(leaf, "weak_type", False)),
            }
            for leaf in leaves
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _memory_analysis_bytes(compiled: Any) -> Mapping[str, int]:
    try:
        analysis = compiled.memory_analysis()
    except Exception:
        return {}
    values: dict[str, int] = {}
    for name in (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    ):
        value = getattr(analysis, name, None)
        if value is not None:
            values[name] = int(value)
    if values:
        values["conservative_device_bytes"] = max(
            0,
            values.get("argument_size_in_bytes", 0)
            + values.get("output_size_in_bytes", 0)
            + values.get("temp_size_in_bytes", 0)
            - values.get("alias_size_in_bytes", 0),
        )
    return values


@dataclass
class CompiledCallable:
    name: str
    function: Callable[..., Any]
    donate_argnums: tuple[int, ...] = ()
    memory_limit_bytes: int | None = None
    _jitted: Any = field(init=False, repr=False)
    _compiled: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _metadata: dict[str, Mapping[str, Any]] = field(default_factory=dict, init=False)
    last_call_compiled: bool = field(default=False, init=False)
    last_signature: str | None = field(default=None, init=False)
    selected_microbatch_size: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        import jax

        self._jitted = jax.jit(self.function, donate_argnums=self.donate_argnums)

    def executable(self, *arguments: Any) -> Any:
        signature = _argument_signature(arguments)
        self.last_signature = signature
        self.last_call_compiled = signature not in self._compiled
        if self.last_call_compiled:
            compiled = self._jitted.lower(*arguments).compile()
            memory = dict(_memory_analysis_bytes(compiled))
            conservative = memory.get("conservative_device_bytes")
            if (
                self.memory_limit_bytes is not None
                and conservative is not None
                and conservative > int(self.memory_limit_bytes)
            ):
                raise CompiledMemoryLimitError(
                    f"{self.name} requires {conservative} device bytes; "
                    f"limit={self.memory_limit_bytes}."
                )
            try:
                executable = compiled.runtime_executable()
                fingerprint = executable.fingerprint
                fingerprint = fingerprint() if callable(fingerprint) else fingerprint
                fingerprint = fingerprint.hex() if isinstance(fingerprint, bytes) else str(fingerprint)
            except Exception:
                fingerprint = hashlib.sha256(
                    f"{self.name}:{signature}".encode()
                ).hexdigest()
            self._compiled[signature] = compiled
            self._metadata[signature] = {
                "name": self.name,
                "argument_signature": signature,
                "executable_fingerprint": fingerprint,
                "memory_analysis": memory,
            }
        return self._compiled[signature]

    def __call__(self, *arguments: Any) -> Any:
        return self.executable(*arguments)(*arguments)

    def metadata(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._metadata[key] for key in sorted(self._metadata))


@dataclass(frozen=True)
class CompiledTrainingKernels:
    rollout_minimal: CompiledCallable
    rollout_anchor_full: CompiledCallable
    rollout_support: CompiledCallable
    target_context_sequence: CompiledCallable
    ppo_scan: CompiledCallable
    # Legacy V6 kernels; disconnected until §5–§7 lands (task #13 boundary).
    regret_chunk: CompiledCallable | None = None
    regret_finalize: CompiledCallable | None = None
    raw_q_retrace_update: CompiledCallable | None = None
    anchor_head_update: CompiledCallable | None = None
    response_head_update: CompiledCallable | None = None
    belief_rollout_objectives: CompiledCallable | None = None
    belief_anchor_objectives: CompiledCallable | None = None
    belief_apply: CompiledCallable | None = None

    def metadata(self) -> Mapping[str, Any]:
        return {
            value.name: list(value.metadata())
            for value in self.__dict__.values()
            if isinstance(value, CompiledCallable)
        }


def configure_persistent_compilation_cache(
    *,
    repository_commit: str,
    official_commit: str,
    config_fingerprint: str,
    model_structure_fingerprint: str | None = None,
    cache_root: str | Path | None = None,
) -> Mapping[str, Any]:
    import jax

    device = jax.devices()[0]
    identity = {
        "repository_commit": str(repository_commit),
        "official_commit": str(official_commit),
        "config_fingerprint": str(config_fingerprint),
        "model_structure_fingerprint": str(model_structure_fingerprint or ""),
        "jax_version": str(jax.__version__),
        "device_kind": str(device.device_kind),
        "platform": str(device.platform),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = (
        Path(cache_root).expanduser()
        if cache_root is not None
        else Path.home() / ".cache" / "delta_zsc" / "jax_compilation"
    )
    directory = root / digest
    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    try:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception:
        pass
    return {**identity, "cache_directory": str(directory), "cache_identity": digest}


def build_training_kernels(
    *,
    environment: Any,
    model: Any,
    model_config: Any,
    partner_functions: Any,
    config: Any,
    ppo_optimizer: Any,
) -> CompiledTrainingKernels:
    """Build the finite family of fixed-shape DEPI CUDA programs.

    Only the active §1–§4 kernels are compiled.  All legacy V6 auxiliary
    kernels are returned as ``None`` so that importing this module never
    touches the deprecated belief/raw-Q/response-head objectives.
    """

    def rollout(mode: str, length: int) -> Callable[..., Any]:
        def execute(
            runner_state: Any,
            params: Any,
            target_params: Any,
            partner_parameters: Any,
            context_dropout_probability_value: Any,
            context_dropout_root: Any,
            official_shaping_factor: Any,
        ) -> Any:
            return collect_rollout(
                state=runner_state,
                length=length,
                environment=environment,
                model=model,
                params=params,
                target_params=target_params,
                model_config=model_config,
                partner_functions=partner_functions,
                partner_parameters=partner_parameters,
                context_dropout_probability_value=context_dropout_probability_value,
                context_dropout_root=context_dropout_root,
                official_shaping_factor=official_shaping_factor,
                record_mode=mode,
            )

        return execute

    def context(target_params: Any, batch: Any) -> Any:
        return target_context_sequence(model=model, target_params=target_params, batch=batch)

    def ppo_scan(
        core: Any,
        batch: Any,
        schedule: Any,
        anchors: Any = None,
        separation_terms: Any = None,
    ) -> Any:
        # ``anchors`` / ``separation_terms``: §5 anchor supervision payload;
        # anchor-batch-global, shared by every scan step.
        return scan_training_updates(
            model=model,
            core=core,
            optimizer=ppo_optimizer,
            batch=batch,
            schedule=schedule,
            config=config,
            anchors=anchors,
            separation_terms=separation_terms,
        )

    return CompiledTrainingKernels(
        rollout_minimal=CompiledCallable(
            "rollout_minimal", rollout("minimal", config.training.rollout_length)
        ),
        rollout_anchor_full=CompiledCallable(
            "rollout_anchor_full", rollout("anchor_full", config.training.rollout_length)
        ),
        rollout_support=CompiledCallable(
            "rollout_support", rollout("support", config.environment.episode_steps)
        ),
        target_context_sequence=CompiledCallable("target_context_sequence", context),
        # The frozen EMA target is consumed again by dense raw-Q immediately
        # after PPO.  Donating the containing core would invalidate those
        # aliased buffers on CUDA, so this boundary deliberately does not
        # donate its input tree.
        ppo_scan=CompiledCallable("ppo_scan", ppo_scan),
    )


def build_anchor_chunk_kernel(
    *, functions: Any, config: Any, memory_limit_bytes: int
) -> CompiledCallable:
    from .counterfactual_anchor import collect_counterfactual_anchors

    def execute(
        runtime: Any,
        anchor_ids: Any,
        root_keys: Any,
        world: Any,
        rollout_flat_indexes: Any,
        policy_states: Any,
        observations: Any,
        partner_codes: Any,
        partner_sources: Any,
        partner_run_ids: Any,
    ) -> Any:
        return collect_counterfactual_anchors(
            anchor_ids=anchor_ids,
            root_keys=root_keys,
            world=world,
            rollout_flat_indexes=rollout_flat_indexes,
            policy_states=policy_states,
            observations=observations,
            partner_codes=partner_codes,
            partner_sources=partner_sources,
            partner_run_ids=partner_run_ids,
            functions=functions,
            action_count=6,
            fit_replicas=config.anchors.fit_replicas,
            evaluation_replicas=0,
            continuation_horizon=config.anchors.continuation_horizon,
            discount=config.ppo.gamma,
            runtime=runtime,
        )

    return CompiledCallable(
        "anchor_continuation_chunk",
        execute,
        memory_limit_bytes=int(memory_limit_bytes),
    )


def attach_chunked_regret_with_kernels(
    *,
    kernels: CompiledTrainingKernels,
    batch: Any,
    target_params: Any,
    key: Any,
    action_range_ema: Any,
    effective_steps: Any,
    state_chunk_size: int = DECISION_REGRET_STATE_CHUNK_SIZE,
) -> tuple[Any, Mapping[str, Any]]:
    """Legacy V6 decision-regret shaping; disconnected from the DEPI path."""

    if kernels.regret_chunk is None or kernels.regret_finalize is None:
        raise RuntimeError(
            "Decision-regret shaping stays disabled until the §5–§7 geometry "
            "fixes land (METHOD_SPEC §7.3; boundary of task #10)."
        )
    import jax
    import jax.numpy as jnp

    context = kernels.target_context_sequence(target_params, batch)
    features = jnp.asarray(context.task_features)
    prefix = features.shape[:-1]
    state_count = int(math.prod(prefix))
    chunk_size = int(state_chunk_size)
    padded_count = ((state_count + chunk_size - 1) // chunk_size) * chunk_size
    padding = padded_count - state_count
    keys = jnp.concatenate(
        (
            jax.random.split(key, state_count),
            jnp.zeros((padding, 2), dtype=jnp.uint32),
        ),
        axis=0,
    )

    def flatten_pad(value: Any) -> Any:
        array = jnp.asarray(value)
        flat = array.reshape((state_count,) + array.shape[len(prefix):])
        return jnp.pad(flat, ((0, padding),) + ((0, 0),) * (flat.ndim - 1))

    flat = (
        flatten_pad(context.task_features),
        flatten_pad(context.belief_mean),
        flatten_pad(context.belief_log_standard_deviation),
    )
    regret_chunks, range_chunks = [], []
    for start in range(0, padded_count, chunk_size):
        stop = start + chunk_size
        regret, action_range = kernels.regret_chunk(
            target_params,
            flat[0][start:stop],
            flat[1][start:stop],
            flat[2][start:stop],
            keys[start:stop],
        )
        regret_chunks.append(regret)
        range_chunks.append(action_range)
    regrets = jnp.concatenate(regret_chunks)[:state_count].reshape(prefix)
    ranges = jnp.concatenate(range_chunks)[:state_count].reshape(prefix)
    return kernels.regret_finalize(
        batch, regrets, ranges, action_range_ema, effective_steps
    )


__all__ = [
    "CompiledCallable",
    "CompiledMemoryLimitError",
    "CompiledTrainingKernels",
    "attach_chunked_regret_with_kernels",
    "build_anchor_chunk_kernel",
    "build_training_kernels",
    "configure_persistent_compilation_cache",
]
