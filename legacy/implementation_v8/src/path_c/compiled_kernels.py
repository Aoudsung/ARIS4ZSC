"""Stable CUDA executable boundaries for the active DEPI update."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .runner import collect_rollout, target_context_sequence
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
    rollout_tail: CompiledCallable
    rollout_support: CompiledCallable
    target_context_sequence: CompiledCallable
    ppo_scan: CompiledCallable

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
        else Path.home() / ".cache" / "depi" / "jax_compilation"
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

    Only active rollout, context, combined-update, and real-continuation
    programs are represented by this object.
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

    vector_block_steps = int(config.environment.num_envs) * int(
        config.training.rollout_length
    )
    tail_transition_count = int(config.training.environment_steps) % vector_block_steps
    tail_length = (
        int(config.training.rollout_length)
        if tail_transition_count == 0
        else tail_transition_count // int(config.environment.num_envs)
    )
    if tail_length <= 0:
        raise ValueError("The configured training budget must contain whole vector steps.")

    return CompiledTrainingKernels(
        rollout_minimal=CompiledCallable(
            "rollout_minimal", rollout("minimal", config.training.rollout_length)
        ),
        rollout_anchor_full=CompiledCallable(
            "rollout_anchor_full", rollout("anchor_full", config.training.rollout_length)
        ),
        # Total-budget R0/B0/B1 controls can end on a partial vector rollout. It
        # remains a fixed-shape executable and is never used for B2 anchors.
        rollout_tail=CompiledCallable(
            "rollout_tail", rollout("minimal", tail_length)
        ),
        rollout_support=CompiledCallable(
            "rollout_support", rollout("support", config.environment.episode_steps)
        ),
        target_context_sequence=CompiledCallable("target_context_sequence", context),
        # The frozen EMA target is consumed again by the anchor evaluator after
        # PPO. Donating the containing core would invalidate aliased buffers on
        # CUDA, so this boundary deliberately does not donate its input tree.
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
        partner_sources: Any,
        partner_members: Any,
        partner_family_ids: Any,
        partner_checkpoint_stages: Any,
        partner_run_ids: Any,
    ) -> Any:
        return collect_counterfactual_anchors(
            anchor_ids=anchor_ids,
            root_keys=root_keys,
            world=world,
            rollout_flat_indexes=rollout_flat_indexes,
            policy_states=policy_states,
            observations=observations,
            partner_sources=partner_sources,
            partner_members=partner_members,
            partner_family_ids=partner_family_ids,
            partner_checkpoint_stages=partner_checkpoint_stages,
            partner_run_ids=partner_run_ids,
            functions=functions,
            action_count=6,
            fit_replicas=config.anchors.fit_replicas,
            evaluation_replicas=config.anchors.evaluation_replicas,
            continuation_horizon=config.anchors.continuation_horizon,
            discount=config.ppo.gamma,
            runtime=runtime,
        )

    return CompiledCallable(
        "anchor_continuation_chunk",
        execute,
        memory_limit_bytes=int(memory_limit_bytes),
    )


__all__ = [
    "CompiledCallable",
    "CompiledMemoryLimitError",
    "CompiledTrainingKernels",
    "build_anchor_chunk_kernel",
    "build_training_kernels",
    "configure_persistent_compilation_cache",
]
