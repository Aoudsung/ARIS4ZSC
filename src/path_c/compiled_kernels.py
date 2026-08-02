"""Stable CUDA executable boundaries for DELTA-ZSC training.

The helpers in this module change dispatch and materialization only.  Model
parameters, random keys, minibatch order, optimizer state transitions, losses,
and simulator budgets remain inputs to the same mathematical operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .runner import (
    DECISION_REGRET_STATE_CHUNK_SIZE,
    collect_rollout,
    decision_regret_chunk,
    finalize_decision_regret_shaping,
    target_context_sequence,
)
from .gradient_routing import (
    keep_owned_gradients,
    route_response_with_pcgrad,
)
from .raw_q import (
    anchor_all_action_loss,
    q_policy_coupling_loss,
    rollout_retrace_loss,
)
from .response_targets import response_auxiliary_objective
from .training import scan_training_updates
from .types import AuxiliaryCoreState


class CompiledMemoryLimitError(RuntimeError):
    """Raised before execution when XLA's conservative memory bound is too high."""


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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


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
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
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
    """Compile once per abstract signature and expose auditable metadata."""

    name: str
    function: Callable[..., Any]
    donate_argnums: tuple[int, ...] = ()
    memory_limit_bytes: int | None = None
    _jitted: Any = field(init=False, repr=False)
    _compiled: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _metadata: dict[str, Mapping[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    last_call_compiled: bool = field(default=False, init=False)
    last_signature: str | None = field(default=None, init=False)
    selected_microbatch_size: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        import jax

        self._jitted = jax.jit(
            self.function,
            donate_argnums=self.donate_argnums,
        )

    def executable(self, *arguments: Any) -> Any:
        signature = _argument_signature(arguments)
        self.last_signature = signature
        self.last_call_compiled = signature not in self._compiled
        if self.last_call_compiled:
            compiled = self._jitted.lower(*arguments).compile()
            try:
                runtime = compiled.runtime_executable()
                fingerprint_value = getattr(runtime, "fingerprint")
                if callable(fingerprint_value):
                    fingerprint_value = fingerprint_value()
                if isinstance(fingerprint_value, bytes):
                    fingerprint_value = fingerprint_value.hex()
                if not fingerprint_value:
                    raise ValueError("empty runtime executable fingerprint")
                fingerprint = str(fingerprint_value)
            except Exception:
                # The code commit and structural signature are recorded beside
                # this fallback.  Avoid materializing multi-gigabyte HLO text
                # merely to hash it on runtimes without executable fingerprints.
                fingerprint = hashlib.sha256(
                    f"{self.name}:{signature}".encode("utf-8")
                ).hexdigest()
            memory = dict(_memory_analysis_bytes(compiled))
            metadata = {
                "name": self.name,
                "argument_signature": signature,
                "executable_fingerprint": fingerprint,
                "memory_analysis": memory,
            }
            conservative = memory.get("conservative_device_bytes")
            if (
                self.memory_limit_bytes is not None
                and conservative is not None
                and conservative > int(self.memory_limit_bytes)
            ):
                raise CompiledMemoryLimitError(
                    f"{self.name} requires conservatively {conservative} bytes; "
                    f"limit={int(self.memory_limit_bytes)}."
                )
            self._compiled[signature] = compiled
            self._metadata[signature] = metadata
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
    regret_chunk: CompiledCallable
    regret_finalize: CompiledCallable
    ppo_normal_scan: CompiledCallable
    raw_q_auxiliary_scan: CompiledCallable
    response_auxiliary_scan: CompiledCallable

    def metadata(self) -> Mapping[str, Any]:
        return {
            item.name: list(item.metadata())
            for item in (
                self.rollout_minimal,
                self.rollout_anchor_full,
                self.rollout_support,
                self.target_context_sequence,
                self.regret_chunk,
                self.regret_finalize,
                self.ppo_normal_scan,
                self.raw_q_auxiliary_scan,
                self.response_auxiliary_scan,
            )
        }


def configure_persistent_compilation_cache(
    *,
    repository_commit: str,
    official_commit: str,
    config_fingerprint: str,
    model_structure_fingerprint: str | None = None,
    cache_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Enable a repository-external persistent cache before first compilation."""

    import jax

    device = jax.devices()[0]
    device_kind = str(device.device_kind)
    identity = {
        "repository_commit": str(repository_commit),
        "official_commit": str(official_commit),
        "config_fingerprint": str(config_fingerprint),
        "model_structure_fingerprint": str(model_structure_fingerprint or ""),
        "jax_version": str(jax.__version__),
        "device_kind": device_kind,
        "platform": str(device.platform),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
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
    return {
        **identity,
        "cache_directory": str(directory),
        "cache_identity": digest,
    }


def build_training_kernels(
    *,
    environment: Any,
    model: Any,
    model_config: Any,
    partner_functions: Any,
    config: Any,
    optimizer: Any,
    raw_q_optimizer: Any,
    response_optimizer: Any,
) -> CompiledTrainingKernels:
    """Build one static executable family for one partner-archive structure."""

    def rollout(mode: str, length: int) -> Callable[..., Any]:
        def execute(
            runner_state: Any,
            params: Any,
            target_params: Any,
            partner_parameters: Any,
            gate_values: Any,
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
                gate_values=gate_values,
                official_shaping_factor=official_shaping_factor,
                record_mode=mode,
            )

        return execute

    def context(target_params: Any, batch: Any) -> Any:
        return target_context_sequence(
            model=model,
            target_params=target_params,
            batch=batch,
        )

    def regret(
        target_params: Any,
        task_features: Any,
        mixture_logits: Any,
        mixture_means: Any,
        mixture_log_variances: Any,
        sample_keys: Any,
    ) -> Any:
        return decision_regret_chunk(
            model=model,
            target_params=target_params,
            task_features=task_features,
            mixture_logits=mixture_logits,
            mixture_means=mixture_means,
            mixture_log_variances=mixture_log_variances,
            sample_keys=sample_keys,
            posterior_particles=config.model.posterior_particles,
        )

    def finalize(batch: Any, regrets: Any, weight: Any) -> Any:
        return finalize_decision_regret_shaping(
            batch=batch,
            regrets=regrets,
            gamma=config.ppo.gamma,
            weight=weight,
        )

    def normal_scan(
        core: Any,
        batch: Any,
        schedule: Any,
        task_trunk_scale: Any,
        base_actor_scale: Any,
    ) -> Any:
        return scan_training_updates(
            model=model,
            core=core,
            optimizer=optimizer,
            batch=batch,
            schedule=schedule,
            config=config,
            task_trunk_scale=task_trunk_scale,
            base_actor_scale=base_actor_scale,
        )

    def raw_q_scan(
        core: AuxiliaryCoreState,
        target_params: Any,
        batch: Any,
        anchors: Any,
        anchor_enabled: Any,
        q_policy_enabled: Any,
        calibration_error: Any,
        minimum_margin: Any,
    ) -> tuple[AuxiliaryCoreState, Mapping[str, Any], Any]:
        import jax
        import jax.numpy as jnp
        import optax

        def one(current: AuxiliaryCoreState, unused: Any) -> tuple[Any, Any]:
            del unused

            def decision_objective(candidate: Any) -> tuple[Any, Mapping[str, Any]]:
                retrace, retrace_metrics = rollout_retrace_loss(
                    model=model,
                    params=candidate,
                    target_params=target_params,
                    batch=batch,
                    gamma=config.ppo.gamma,
                )
                anchor, anchor_metrics = anchor_all_action_loss(
                    model=model, params=candidate, anchors=anchors
                )
                total = (
                    float(config.loss.raw_q_weight) * retrace
                    + jnp.asarray(anchor_enabled, dtype=jnp.float32) * anchor
                )
                return total, {
                    **retrace_metrics,
                    **anchor_metrics,
                    "raw_q_decision_total": total,
                }

            def policy_objective(candidate: Any) -> tuple[Any, Mapping[str, Any]]:
                q_policy, q_policy_metrics = q_policy_coupling_loss(
                    model=model,
                    params=candidate,
                    batch=batch,
                    temperature=config.loss.q_policy_temperature,
                    calibration_error=calibration_error,
                    minimum_margin=minimum_margin,
                )
                total = (
                    jnp.asarray(q_policy_enabled, dtype=jnp.float32)
                    * float(config.loss.q_policy_weight)
                    * q_policy
                )
                return total, {
                    **q_policy_metrics,
                    "q_policy_auxiliary_total": total,
                }

            (decision_loss, decision_metrics), decision_gradients = jax.value_and_grad(
                decision_objective, has_aux=True
            )(current.params)
            (policy_loss, policy_metrics), policy_gradients = jax.value_and_grad(
                policy_objective, has_aux=True
            )(current.params)
            decision_gradients = keep_owned_gradients(
                decision_gradients, loss_name="raw_q"
            )
            policy_gradients = keep_owned_gradients(
                policy_gradients, loss_name="q_policy"
            )
            from .gradient_routing import sum_gradient_trees

            gradients = sum_gradient_trees(decision_gradients, policy_gradients)
            updates, optimizer_state = raw_q_optimizer.update(
                gradients, current.raw_q_optimizer_state, current.params
            )
            params = optax.apply_updates(current.params, updates)
            updated = current._replace(
                params=params,
                raw_q_optimizer_state=optimizer_state,
                raw_q_optimizer_step=current.raw_q_optimizer_step + 1,
            )
            metrics = {
                **decision_metrics,
                **policy_metrics,
                "raw_q_auxiliary_total": decision_loss + policy_loss,
            }
            return updated, (metrics, decision_gradients)

        final, (metrics, decision_gradients) = jax.lax.scan(
            one,
            core,
            xs=None,
            length=int(config.loss.raw_q_updates_per_outer_update),
        )
        latest_decision_gradients = jax.tree_util.tree_map(
            lambda value: value[-1], decision_gradients
        )
        return final, metrics, latest_decision_gradients

    def response_scan(
        core: AuxiliaryCoreState,
        batch: Any,
        decision_gradients: Any,
    ) -> tuple[AuxiliaryCoreState, Mapping[str, Any]]:
        import jax
        import optax

        def one(current: AuxiliaryCoreState, unused: Any) -> tuple[Any, Any]:
            del unused

            def objective(candidate: Any) -> tuple[Any, Mapping[str, Any]]:
                return response_auxiliary_objective(
                    model=model, params=candidate, batch=batch
                )

            (unused_loss, metrics), gradients = jax.value_and_grad(
                objective, has_aux=True
            )(current.params)
            del unused_loss
            gradients = keep_owned_gradients(gradients, loss_name="response")
            gradients, routing = route_response_with_pcgrad(
                gradients, decision_gradients
            )
            updates, optimizer_state = response_optimizer.update(
                gradients, current.response_optimizer_state, current.params
            )
            params = optax.apply_updates(current.params, updates)
            return current._replace(
                params=params,
                response_optimizer_state=optimizer_state,
                response_optimizer_step=current.response_optimizer_step + 1,
            ), {**metrics, **routing}

        return jax.lax.scan(
            one,
            core,
            xs=None,
            length=int(config.loss.response_updates_per_outer_update),
        )

    return CompiledTrainingKernels(
        rollout_minimal=CompiledCallable(
            "rollout_minimal",
            rollout("minimal", int(config.training.rollout_length)),
        ),
        rollout_anchor_full=CompiledCallable(
            "rollout_anchor_full",
            rollout("anchor_full", int(config.training.rollout_length)),
        ),
        rollout_support=CompiledCallable(
            "rollout_support",
            rollout("support", int(config.environment.episode_steps)),
        ),
        target_context_sequence=CompiledCallable("target_context_sequence", context),
        regret_chunk=CompiledCallable("regret_chunk", regret),
        regret_finalize=CompiledCallable("regret_finalize", finalize),
        ppo_normal_scan=CompiledCallable(
            "ppo_normal_scan", normal_scan, donate_argnums=(0,)
        ),
        raw_q_auxiliary_scan=CompiledCallable(
            "raw_q_auxiliary_scan", raw_q_scan, donate_argnums=(0,)
        ),
        response_auxiliary_scan=CompiledCallable(
            "response_auxiliary_scan", response_scan, donate_argnums=(0,)
        ),
    )


def build_anchor_chunk_kernel(
    *,
    functions: Any,
    config: Any,
    memory_limit_bytes: int,
    mode: str = "training",
) -> CompiledCallable:
    """Build the fixed-shape continuation executable with dynamic parameters."""

    from .counterfactual_anchor import collect_counterfactual_anchors

    if mode == "training":
        fit_replicas = int(config.anchors.fit_replicas)
        evaluation_replicas = 0
        continuation_horizon = int(config.anchors.continuation_horizon)
        kernel_name = "training_anchor_continuation_chunk"
    elif mode == "audit":
        fit_replicas = int(config.anchors.audit_fit_replicas)
        evaluation_replicas = int(config.anchors.audit_evaluation_replicas)
        continuation_horizon = int(config.anchors.audit_continuation_horizon)
        kernel_name = "audit_anchor_continuation_chunk"
    else:
        raise ValueError("Anchor kernel mode must be training or audit.")

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
            fit_replicas=fit_replicas,
            evaluation_replicas=evaluation_replicas,
            continuation_horizon=continuation_horizon,
            runtime=runtime,
        )

    return CompiledCallable(
        kernel_name,
        execute,
        memory_limit_bytes=int(memory_limit_bytes),
    )


def attach_chunked_regret_with_kernels(
    *,
    kernels: CompiledTrainingKernels,
    batch: Any,
    target_params: Any,
    key: Any,
    weight: float = 0.0,
    state_chunk_size: int = DECISION_REGRET_STATE_CHUNK_SIZE,
) -> tuple[Any, Mapping[str, Any]]:
    """Dispatch a fixed critic chunk 17 times for the formal 257x256 shape."""

    import jax
    import jax.numpy as jnp

    context = kernels.target_context_sequence(target_params, batch)
    features = jnp.asarray(context.task_features)
    prefix = features.shape[:-1]
    state_count = int(math.prod(prefix))
    chunk_size = int(state_chunk_size)
    padded_count = ((state_count + chunk_size - 1) // chunk_size) * chunk_size
    padding = padded_count - state_count
    real_keys = jax.random.split(key, state_count)
    padded_keys = jnp.concatenate(
        (real_keys, jnp.zeros((padding, 2), dtype=real_keys.dtype)), axis=0
    )

    def flatten_pad(value: Any) -> Any:
        array = jnp.asarray(value)
        flat = array.reshape((state_count,) + array.shape[len(prefix) :])
        if padding:
            flat = jnp.concatenate(
                (
                    flat,
                    jnp.zeros((padding,) + flat.shape[1:], dtype=flat.dtype),
                ),
                axis=0,
            )
        return flat

    flat_values = (
        flatten_pad(context.task_features),
        flatten_pad(context.mixture_logits),
        flatten_pad(context.mixture_means),
        flatten_pad(context.mixture_log_variances),
    )
    chunks = []
    for start in range(0, padded_count, chunk_size):
        stop = start + chunk_size
        chunks.append(
            kernels.regret_chunk(
                target_params,
                flat_values[0][start:stop],
                flat_values[1][start:stop],
                flat_values[2][start:stop],
                flat_values[3][start:stop],
                padded_keys[start:stop],
            )
        )
    regrets = jnp.concatenate(chunks, axis=0)[:state_count].reshape(prefix)
    return kernels.regret_finalize(
        batch, regrets, jnp.asarray(weight, dtype=jnp.float32)
    )


def safe_kernel_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


__all__ = [
    "CompiledCallable",
    "CompiledMemoryLimitError",
    "CompiledTrainingKernels",
    "attach_chunked_regret_with_kernels",
    "build_anchor_chunk_kernel",
    "build_training_kernels",
    "configure_persistent_compilation_cache",
]
