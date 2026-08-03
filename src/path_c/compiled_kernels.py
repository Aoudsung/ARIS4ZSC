"""Stable CUDA executable boundaries for the V6 end-to-end update."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from .gradient_routing import keep_owned_gradients
from .raw_q import (
    anchor_all_action_loss,
    decision_equivalence_loss,
    rollout_retrace_loss,
)
from .response_targets import response_auxiliary_objective
from .runner import (
    DECISION_REGRET_STATE_CHUNK_SIZE,
    collect_rollout,
    decision_regret_chunk,
    finalize_decision_regret_shaping,
    target_context_sequence,
)
from .training import belief_objective_gradients, scan_training_updates
from .types import TrainingCoreState


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
    regret_chunk: CompiledCallable
    regret_finalize: CompiledCallable
    ppo_scan: CompiledCallable
    raw_q_retrace_update: CompiledCallable
    anchor_head_update: CompiledCallable
    response_head_update: CompiledCallable
    belief_rollout_objectives: CompiledCallable
    belief_anchor_objectives: CompiledCallable
    belief_apply: CompiledCallable

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


def _apply_optimizer(
    *, params: Any, optimizer_state: Any, gradients: Any, optimizer: Any
) -> tuple[Any, Any]:
    import optax

    updates, next_optimizer_state = optimizer.update(
        gradients, optimizer_state, params
    )
    return optax.apply_updates(params, updates), next_optimizer_state


def build_training_kernels(
    *,
    environment: Any,
    model: Any,
    model_config: Any,
    partner_functions: Any,
    config: Any,
    ppo_optimizer: Any,
    raw_q_optimizer: Any,
    response_optimizer: Any,
    belief_optimizer: Any,
) -> CompiledTrainingKernels:
    """Build a finite family of fixed-shape V6 CUDA programs."""

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

    def regret(
        target_params: Any,
        task_features: Any,
        belief_mean: Any,
        belief_log_standard_deviation: Any,
        sample_keys: Any,
    ) -> Any:
        return decision_regret_chunk(
            model=model,
            target_params=target_params,
            task_features=task_features,
            belief_mean=belief_mean,
            belief_log_standard_deviation=belief_log_standard_deviation,
            sample_keys=sample_keys,
            posterior_particles=config.model.posterior_particles,
        )

    def finalize(
        batch: Any, regrets: Any, ranges: Any, action_range_ema: Any, effective_steps: Any
    ) -> Any:
        import jax

        progress = effective_steps / float(max(config.training.environment_steps, 1))
        weight = float(config.loss.decision_regret_weight_maximum) * jax.nn.sigmoid(
            (progress - float(config.loss.decision_regret_schedule_midpoint))
            / float(config.loss.decision_regret_schedule_temperature)
        )
        return finalize_decision_regret_shaping(
            batch=batch,
            regrets=regrets,
            action_ranges=ranges,
            action_range_ema=action_range_ema,
            gamma=config.ppo.gamma,
            weight=weight,
        )

    def ppo_scan(core: Any, batch: Any, schedule: Any) -> Any:
        return scan_training_updates(
            model=model,
            core=core,
            optimizer=ppo_optimizer,
            batch=batch,
            schedule=schedule,
            config=config,
        )

    def retrace_update(
        params: Any, target_params: Any, optimizer_state: Any, batch: Any
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        import jax

        def objective(candidate: Any):
            loss, metrics = rollout_retrace_loss(
                model=model,
                params=candidate,
                target_params=target_params,
                batch=batch,
                gamma=config.ppo.gamma,
            )
            return float(config.loss.raw_q_weight) * loss, metrics

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        gradients = keep_owned_gradients(gradients, loss_name="raw_q")
        next_params, next_state = _apply_optimizer(
            params=params,
            optimizer_state=optimizer_state,
            gradients=gradients,
            optimizer=raw_q_optimizer,
        )
        return next_params, next_state, {**metrics, "raw_q_total": loss}

    def anchor_update(
        params: Any,
        optimizer_state: Any,
        anchors: Any,
        replay_weights: Any,
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        import jax

        def objective(candidate: Any):
            loss, metrics = anchor_all_action_loss(
                model=model,
                params=candidate,
                anchors=anchors,
                replay_weights=replay_weights,
            )
            return float(config.loss.counterfactual_weight) * loss, metrics

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        gradients = keep_owned_gradients(gradients, loss_name="counterfactual")
        next_params, next_state = _apply_optimizer(
            params=params,
            optimizer_state=optimizer_state,
            gradients=gradients,
            optimizer=raw_q_optimizer,
        )
        return next_params, next_state, {**metrics, "counterfactual_total": loss}

    def response_update(
        params: Any, optimizer_state: Any, batch: Any
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        import jax

        def objective(candidate: Any):
            loss, metrics = response_auxiliary_objective(
                model=model, params=candidate, batch=batch
            )
            return float(config.loss.response_weight) * loss, metrics

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        gradients = keep_owned_gradients(gradients, loss_name="response")
        next_params, next_state = _apply_optimizer(
            params=params,
            optimizer_state=optimizer_state,
            gradients=gradients,
            optimizer=response_optimizer,
        )
        return next_params, next_state, {**metrics, "response_total": loss}

    def belief_rollout_gradients(
        params: Any, target_params: Any, batch: Any
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        import jax

        values, gradients = belief_objective_gradients(
            model=model, params=params, batch=batch, config=config
        )

        def raw_objective(candidate: Any):
            return rollout_retrace_loss(
                model=model,
                params=candidate,
                target_params=target_params,
                batch=batch,
                gamma=config.ppo.gamma,
            )[0]

        def response_objective(candidate: Any):
            return response_auxiliary_objective(
                model=model, params=candidate, batch=batch
            )[0]

        for name, objective in (
            ("raw_q", raw_objective),
            ("response", response_objective),
        ):
            values[name], gradients[name] = jax.value_and_grad(objective)(params)
        return values, gradients

    def belief_anchor_gradients(
        params: Any, anchors: Any, replay_weights: Any, advantage_scale: Any
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        import jax

        def cf_objective(candidate: Any):
            return anchor_all_action_loss(
                model=model,
                params=candidate,
                anchors=anchors,
                replay_weights=replay_weights,
            )[0]

        def de_objective(candidate: Any):
            return decision_equivalence_loss(
                model=model,
                params=candidate,
                anchors=anchors,
                advantage_scale=advantage_scale,
                replay_weights=replay_weights,
            )[0]

        values: dict[str, Any] = {}
        gradients: dict[str, Any] = {}
        for name, objective in (
            ("counterfactual", cf_objective),
            ("decision_equivalence", de_objective),
        ):
            values[name], gradients[name] = jax.value_and_grad(objective)(params)
        return values, gradients

    def apply_belief(
        params: Any, optimizer_state: Any, gradients: Any
    ) -> tuple[Any, Any]:
        from .gradient_routing import select_gradient_prefixes

        return _apply_optimizer(
            params=params,
            optimizer_state=optimizer_state,
            gradients=select_gradient_prefixes(gradients, ("belief_encoder",)),
            optimizer=belief_optimizer,
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
        regret_chunk=CompiledCallable("regret_chunk", regret),
        regret_finalize=CompiledCallable("regret_finalize", finalize),
        # The frozen EMA target is consumed again by dense raw-Q immediately
        # after PPO.  Donating the containing core would invalidate those
        # aliased buffers on CUDA, so this boundary deliberately does not
        # donate its input tree.
        ppo_scan=CompiledCallable("ppo_scan", ppo_scan),
        raw_q_retrace_update=CompiledCallable(
            "raw_q_retrace_update", retrace_update
        ),
        anchor_head_update=CompiledCallable("anchor_head_update", anchor_update),
        response_head_update=CompiledCallable("response_head_update", response_update),
        belief_rollout_objectives=CompiledCallable(
            "belief_rollout_objectives", belief_rollout_gradients
        ),
        belief_anchor_objectives=CompiledCallable(
            "belief_anchor_objectives", belief_anchor_gradients
        ),
        belief_apply=CompiledCallable("belief_apply", apply_belief),
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
