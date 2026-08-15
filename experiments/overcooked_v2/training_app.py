"""End-to-end Official OvercookedV2 training for unified DELTA-ZSC."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import yaml

from src.delta_zsc.anchors import (
    AnchorFunctions,
    make_anchor_batch_kernel,
)
from src.delta_zsc.config import (
    CHECKPOINT_SCHEMA_VERSION,
    FORMAL_PEAK_MEMORY_LIMIT_BYTES,
    METHOD_VERSION,
    OFFICIAL_ACTION_COUNT,
    load_config,
)
from src.delta_zsc.contrast import kendall_tau_b
from src.delta_zsc.losses import DECISION_METRIC_NAMES, latent_composite_loss
from src.delta_zsc.anchor_buffer import init_anchor_buffer
from src.delta_zsc.manifest import load_partner_manifest
from src.delta_zsc.model import DeltaModel, observe_after_transition
from src.delta_zsc.observation import INTERFACE_EVENT_CLASSES
from src.delta_zsc.official_initializer import (
    initialize_base_from_official,
    initializer_seed_index,
)
from src.delta_zsc.base_policy import trainable_base_params
from src.delta_zsc.optimizer import init_adam
from src.delta_zsc.partners import build_training_partner_pool, make_static_partner_functions
from src.delta_zsc.resources import ResourceLedger, parameter_count, peak_device_memory_bytes
from src.delta_zsc.semantic_initializer import (
    deterministic_simplex_initializer,
    load_semantic_initializer,
    validate_semantic_initializer_provenance,
)
from src.delta_zsc.runner import (
    initialize_runner,
    make_anchor_snapshot_rollout_kernel,
    make_compact_training_rollout_kernel,
)
from src.delta_zsc.storage import (
    ensure_run_identity,
    load_latest_checkpoint,
    save_checkpoint,
    write_json,
)
from src.delta_zsc.training import (
    environment_minibatch_schedule,
    init_latent_optimizer,
    make_training_update_kernel,
    polyak_update,
    slice_rollout_lanes,
    update_contrast_channel,
)
from src.delta_zsc.types import TrainState

from .deployment import export_deployment_bundle


def _training_key(seed_index: int) -> Any:
    import jax

    if int(seed_index) < 0:
        return jax.random.PRNGKey(0)
    return jax.random.split(jax.random.PRNGKey(42), 10)[int(seed_index)]


def _write_jsonl_batch(path: Path, payloads: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _host_converted(value: Any) -> Any:
    """Convert an already transferred JAX/NumPy tree to JSON values."""

    import jax

    def convert(item: Any) -> Any:
        array = np.asarray(item)
        if array.ndim == 0:
            return array.item()
        return array.tolist()

    return jax.tree_util.tree_map(convert, value)


def _same_parameters(left: Any, right: Any) -> bool:
    """Compare two parameter trees leaf by leaf."""

    import jax

    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.array_equal(np.asarray(jax.device_get(one)), np.asarray(jax.device_get(other)))
        for one, other in zip(left_leaves, right_leaves)
    )


def _parse_official_throughput(source: Path) -> Mapping[str, Any]:
    """Load a completed single-GPU Official parent training record.

    The pilot parent jobs predate ``ResourceLedger`` but saved their measured
    wall time next to Hydra's frozen training config.  Treat that pair as one
    auditable source: neither the configured budget nor an incomplete job is
    allowed to masquerade as measured upstream cost.
    """

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(
                f"Malformed Official throughput line {line_number}: {source}"
            )
        name, value = (part.strip() for part in line.split("=", 1))
        if not name or not value or name in values:
            raise ValueError(
                f"Invalid Official throughput field on line {line_number}: {source}"
            )
        values[name] = value

    required = {"status", "environment_steps", "wall_seconds"}
    missing = required - set(values)
    if missing:
        raise ValueError(
            f"Official throughput record lacks {sorted(missing)}: {source}"
        )
    if int(values["status"]) != 0:
        raise ValueError(
            f"Official parent training did not complete successfully: {source}"
        )
    steps = int(values["environment_steps"])
    wall_seconds = float(values["wall_seconds"])
    if steps <= 0 or wall_seconds <= 0.0 or not np.isfinite(wall_seconds):
        raise ValueError(f"Official parent cost must be positive: {source}")

    config_source = source.parent / "official_hydra" / ".hydra" / "config.yaml"
    if not config_source.is_file():
        raise FileNotFoundError(
            f"Official throughput record has no matching Hydra config: {config_source}"
        )
    config_payload = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if not isinstance(config_payload, Mapping):
        raise ValueError(f"Official Hydra config is not a mapping: {config_source}")
    model_payload = config_payload.get("model")
    if (
        not isinstance(model_payload, Mapping)
        or model_payload.get("TOTAL_TIMESTEPS") is None
    ):
        raise ValueError(
            f"Official Hydra config lacks model.TOTAL_TIMESTEPS: {config_source}"
        )
    configured_steps = int(round(float(model_payload["TOTAL_TIMESTEPS"])))
    if configured_steps != steps:
        raise ValueError(
            "Official throughput/config step mismatch: "
            f"measured={steps}, configured={configured_steps}, source={source}"
        )
    if int(config_payload.get("NUM_SEEDS", 0)) != 1:
        raise ValueError(
            f"Official parent cost record must describe one seed: {config_source}"
        )
    if config_payload.get("SEED") is None:
        raise ValueError(f"Official Hydra config lacks SEED: {config_source}")

    # Registered parent jobs are one-GPU jobs.  New launch scripts may record
    # ``gpu_device_count`` explicitly; the legacy pilot format omitted it.
    device_count = int(values.get("gpu_device_count", "1"))
    if device_count != 1:
        raise ValueError(f"Official parent cost must describe one GPU: {source}")
    return {
        "steps": steps,
        "gpu_hours": wall_seconds / 3600.0,
        "wall_clock_hours": wall_seconds / 3600.0,
        "source": str(source),
        "source_kind": "official_throughput",
        "configured_seed": int(config_payload["SEED"]),
        "gpu_device_count": device_count,
    }


def _resolve_upstream_parent_cost(checkpoint: Path) -> Mapping[str, Any] | None:
    """Resolve explicit or legacy-Official cost evidence for one checkpoint."""

    checkpoint = checkpoint.resolve()
    directories = (checkpoint, *checkpoint.parents[:6])
    for directory in directories:
        for source in (
            directory / "resource_ledger.json",
            directory / "upstream_summary.json",
        ):
            if not source.is_file():
                continue
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"Upstream resource source is not a mapping: {source}")
            try:
                registered = ResourceLedger.from_mapping(payload)
            except (TypeError, ValueError):
                registered = None
            if registered is not None:
                steps = int(registered.total_training_simulator_steps)
                gpu_hours = float(registered.gpu_hours)
                wall_clock_hours = float(registered.wall_clock_hours)
                source_kind = "resource_ledger"
            else:
                steps = 0
                for name in (
                    "total_training_simulator_steps",
                    "upstream_partner_steps",
                    "effective_environment_steps",
                    "actual_timesteps",
                ):
                    if payload.get(name) is not None and int(payload[name]) > 0:
                        steps = int(payload[name])
                        break
                gpu_hours = float(
                    payload.get("gpu_hours", payload.get("training_gpu_hours", 0.0))
                )
                wall_clock_hours = float(
                    payload.get(
                        "wall_clock_hours",
                        payload.get("training_wall_clock_hours", 0.0),
                    )
                )
                source_kind = "legacy_upstream_summary"
            if steps <= 0:
                raise ValueError(
                    f"Upstream resource source has no positive step total: {source}"
                )
            return {
                "steps": steps,
                "gpu_hours": gpu_hours,
                "wall_clock_hours": wall_clock_hours,
                "source": str(source),
                "source_kind": source_kind,
            }

    throughput = next(
        (
            directory / "throughput.txt"
            for directory in directories
            if (directory / "throughput.txt").is_file()
        ),
        None,
    )
    return None if throughput is None else _parse_official_throughput(throughput)


def _upstream_partner_cost(
    members: tuple[Any, ...],
) -> tuple[int, float, float, list[Mapping[str, Any]]]:
    """Resolve each unique parent policy's explicit simulator cost.

    New DELTA upstream runs expose ``ResourceLedger``.  Existing Official
    checkpoints may carry an older ledger/summary.  A parent without either is
    recorded as ``missing`` and contributes nothing to the totals; no cost is
    ever invented for it.  Upstream accounting is reported, not enforced, so a
    parent trained outside ``train-baseline`` does not block the run.
    """

    total = 0
    gpu_hours = 0.0
    wall_clock_hours = 0.0
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for member in members:
        parent = str(member.parent_training_run_id)
        if parent in seen:
            continue
        seen.add(parent)
        checkpoint = Path(member.checkpoint).resolve()
        resolved = _resolve_upstream_parent_cost(checkpoint)
        if resolved is None:
            records.append(
                {
                    "parent_training_run_id": parent,
                    "status": "missing",
                    "checkpoint": str(checkpoint),
                }
            )
            continue
        steps = int(resolved["steps"])
        parent_gpu = float(resolved["gpu_hours"])
        parent_wall = float(resolved["wall_clock_hours"])
        total += steps
        gpu_hours += parent_gpu
        wall_clock_hours += parent_wall
        records.append(
            {
                "parent_training_run_id": parent,
                "status": "counted",
                "steps": steps,
                "gpu_hours": parent_gpu,
                "wall_clock_hours": parent_wall,
                **dict(resolved),
            }
        )
    return total, gpu_hours, wall_clock_hours, records


def _reconcile_upstream_resource_cost(
    payload: Mapping[str, Any],
    *,
    steps: int,
    gpu_hours: float,
    wall_clock_hours: float,
) -> Mapping[str, Any]:
    """Replace shared parent cost after loading a legacy/resumed state."""

    ledger = ResourceLedger.from_mapping(payload)
    values = dict(ledger.to_mapping())
    values.update(
        upstream_partner_steps=int(steps),
        shared_gpu_hours=float(gpu_hours),
        shared_wall_clock_hours=float(wall_clock_hours),
    )
    return ResourceLedger.from_mapping(values).to_mapping()


def _anchor_functions(
    *, model: DeltaModel, partner_functions: Any, partner_parameters: Any, environment: Any
) -> AnchorFunctions:
    import jax

    def ego_step(base: Any, latent: Any, state: Any, observation: Any, keys: Any):
        # The CRN continuation must run the same belief-conditioned actor the
        # anchor is meant to score.  Continuing under a posterior-blind policy
        # makes Q_z(s,a) the value of "take a, then abandon whatever protocol a
        # committed to", which is nearly identical across actions and carries
        # no ranking signal.
        next_state, output = model.step(
            base,
            latent,
            state,
            observation,
            compute_latent=True,
            compute_decision=False,
            execute_adaptation=False,
        )
        action = jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(
            keys, output.base_policy_logits
        )
        return next_state, action

    def ego_value(base: Any, latent: Any, state: Any, observation: Any):
        _, output = model.step(
            base,
            latent,
            state,
            observation,
            compute_latent=True,
            compute_decision=False,
            execute_adaptation=False,
        )
        return output.value

    def ego_observe(state: Any, action: Any, done: Any):
        return observe_after_transition(state, action=action, done=done)

    def partner_step_fixed(
        state: Any, observation: Any, episode_start: Any, keys: Any
    ):
        # The anchor world stores the actual snapshot episode-start flag.  This
        # is essential for states sampled at an episode boundary: partner
        # recurrent policies must reset exactly as they did in the legal
        # rollout, rather than being silently forced to continue an old carry.
        action, next_state, context, unused_value = partner_functions.step(
            partner_parameters, state, observation, episode_start, keys
        )
        del unused_value
        return action, next_state, context

    def partner_observe(
        state: Any,
        context: Any,
        observation: Any,
        action: Any,
        reward: Any,
        done: Any,
        next_observation: Any,
    ):
        # The frozen Official recurrent partner is already advanced by
        # ``partner_step``.  Its observe hook only prepares the next episode's
        # member/carry, which is unobservable after a terminal anchor branch.
        del context, observation, action, reward, done, next_observation
        return state

    return AnchorFunctions(
        ego_step=ego_step,
        ego_value=ego_value,
        ego_observe=ego_observe,
        partner_step=partner_step_fixed,
        partner_observe=partner_observe,
        environment_step=getattr(
            environment,
            "step_anchor_terminal_with_keys",
            environment.step_with_keys,
        ),
    )


def _measure_inference_latency_ms(
    *, model: DeltaModel, base_params: Any, latent_params: Any
) -> float:
    """Median synchronized batch-one deployment-step latency."""

    import jax
    import jax.numpy as jnp

    state = model.initial_state(1)
    observation = jnp.zeros((1,) + tuple(model.observation_shape), dtype=jnp.float32)
    step = jax.jit(lambda current, frame: model.step(
        base_params, latent_params, current, frame
    ))
    for _ in range(20):
        state, output = step(state, observation)
        jax.block_until_ready(output.policy_logits)
    values = []
    for _ in range(100):
        started = time.perf_counter()
        state, output = step(state, observation)
        jax.block_until_ready(output.policy_logits)
        values.append((time.perf_counter() - started) * 1_000.0)
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _component_event_js_by_anchor(component_event_logits: Any) -> Any:
    """Reduce ``[anchor, probe, component, event]`` logits to ``[anchor]`` JS.

    The component axis is exchangeable and the probe axis is an experimental
    action axis.  Final-audit eligibility is anchor-level, so both axes must be
    reduced in that order before applying an anchor mask.
    """

    import jax.nn as jnn
    import jax.numpy as jnp

    probability = jnn.softmax(component_event_logits, axis=-1)
    component_mean = jnp.mean(probability, axis=-2, keepdims=True)
    js_by_probe = jnp.mean(
        jnp.sum(
            probability
            * (
                jnp.log(jnp.maximum(probability, 1.0e-30))
                - jnp.log(jnp.maximum(component_mean, 1.0e-30))
            ),
            axis=-1,
        ),
        axis=-1,
    )
    return jnp.mean(js_by_probe, axis=-1)


def _final_decision_audit(
    *,
    model: DeltaModel,
    base_params: Any,
    latent_params: Any,
    batch: Any,
    anchors: Any,
) -> Mapping[str, Any]:
    """Report current and probe-successor decision quality plus active selectivity."""

    import jax.nn as jnn
    import jax.numpy as jnp

    from src.delta_zsc.mirror_policy import mirror_policy_logits

    _, output = model.sequence(
        base_params,
        latent_params,
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        compute_latent=True,
        compute_decision=True,
        execute_adaptation=True,
    )
    time = anchors.time_indexes
    lane = anchors.lane_indexes
    predicted = output.expected_decision_values[time, lane]
    target = anchors.evaluation_returns_by_action
    # Kendall tau_b, not Spearman over argsort(argsort(...)).  Every anchor row
    # of the pre-refactor measurement contained exact ties, and the double
    # argsort was ranking equal values by index order -- so the reported
    # correlation was partly an artefact of action numbering.  tau_b puts ties
    # in the denominator instead of breaking them.
    spearman = kendall_tau_b(predicted, target, jnp.ones_like(target, dtype=bool))
    chosen = jnp.argmax(predicted, axis=-1)
    oracle = jnp.argmax(target, axis=-1)
    rows = jnp.arange(chosen.shape[0])
    regret = target[rows, oracle] - target[rows, chosen]

    component_current = output.component_decision_means[time, lane]
    component_current_top = jnp.argmax(component_current, axis=-1)
    component_current_disagreement = jnp.mean(
        (
            component_current_top[..., :, None]
            != component_current_top[..., None, :]
        ).astype(jnp.float32)
    )

    successor = output.component_successor_decision_means[time, lane]
    belief = output.belief[time, lane]
    successor_expected = jnp.sum(belief[:, None, :, None] * successor, axis=-2)
    successor_selected = jnp.argmax(successor_expected, axis=-1)
    successor_target = anchors.probe_evaluation_returns_by_action
    successor_oracle = jnp.argmax(successor_target, axis=-1)
    successor_valid = jnp.all(anchors.probe_action_mask, axis=-1)
    probe_rows = jnp.arange(successor_selected.shape[0])[:, None]
    probe_indexes = jnp.arange(successor_selected.shape[1])[None, :]
    successor_regret = (
        successor_target[probe_rows, probe_indexes, successor_oracle]
        - successor_target[probe_rows, probe_indexes, successor_selected]
    )
    successor_denominator = jnp.maximum(jnp.sum(successor_valid), 1)
    successor_agreement = jnp.sum(
        successor_valid * (successor_selected == successor_oracle)
    ) / successor_denominator
    successor_mean_regret = jnp.sum(successor_valid * successor_regret) / (
        successor_denominator
    )
    successor_component_top = jnp.argmax(successor, axis=-1)
    successor_component_disagreement = jnp.mean(
        (
            successor_component_top[..., :, :, None]
            != successor_component_top[..., :, None, :]
        ).astype(jnp.float32)
    )

    anchor_voi = output.active_voi[time, lane]
    anchor_information_gain = output.active_information_gain[time, lane]
    anchor_adaptation_kl = output.adaptation_kl[time, lane]
    active_eligible = output.active_probe_eligible[time, lane].astype(jnp.float32)
    active_denominator = jnp.maximum(jnp.sum(active_eligible), 1.0)

    def active_mean(value: Any) -> Any:
        return jnp.sum(active_eligible * jnp.asarray(value, dtype=jnp.float32)) / (
            active_denominator
        )
    base_logits = output.base_policy_logits[time, lane]
    active_logits = output.policy_logits[time, lane]
    passive_logits, _, _ = mirror_policy_logits(
        base_logits,
        output.expected_decision_values[time, lane],
        kl_budget=model.config.method.adaptation_kl_budget,
    )
    active_probability = jnn.softmax(active_logits, axis=-1)
    passive_probability = jnn.softmax(passive_logits, axis=-1)
    active_passive_tv = 0.5 * jnp.sum(
        jnp.abs(active_probability - passive_probability), axis=-1
    )
    base_greedy = jnp.argmax(base_logits, axis=-1)
    active_greedy = jnp.argmax(active_logits, axis=-1)
    passive_greedy = jnp.argmax(passive_logits, axis=-1)

    posterior = jnp.maximum(output.belief[time, lane], 1.0e-30)
    prior = jnp.maximum(output.predictive_belief[time, lane], 1.0e-30)
    posterior_entropy = -jnp.sum(posterior * jnp.log(posterior), axis=-1)
    filter_kl = jnp.sum(posterior * (jnp.log(posterior) - jnp.log(prior)), axis=-1)

    probe_component_event_js_by_anchor = _component_event_js_by_anchor(
        output.probe_response_prediction.interface_event_logits[time, lane]
    )
    voi_spread = jnp.max(anchor_voi, axis=-1) - jnp.min(anchor_voi, axis=-1)
    information_spread = jnp.max(anchor_information_gain, axis=-1) - jnp.min(
        anchor_information_gain, axis=-1
    )
    eligible_voi = jnp.where(active_eligible[:, None] > 0.0, anchor_voi, jnp.inf)
    eligible_minimum_voi = jnp.where(
        jnp.sum(active_eligible) > 0.0, jnp.min(eligible_voi), 0.0
    )

    return {
        "schema_version": 4,
        "mean_kendall_tau_b": float(jnp.mean(spearman)),
        "top_action_agreement": float(
            jnp.mean((chosen == oracle).astype(jnp.float32))
        ),
        "mean_empirical_action_regret": float(jnp.mean(regret)),
        "current_component_top_action_disagreement": float(
            component_current_disagreement
        ),
        "successor_top_action_agreement": float(successor_agreement),
        "successor_mean_empirical_action_regret": float(successor_mean_regret),
        "successor_component_top_action_disagreement": float(
            successor_component_disagreement
        ),
        "mean_action_voi": float(
            jnp.sum(active_eligible[:, None] * anchor_voi)
            / (active_denominator * anchor_voi.shape[-1])
        ),
        "mean_max_action_voi": float(
            active_mean(jnp.max(anchor_voi, axis=-1))
        ),
        "mean_voi_action_spread": float(active_mean(voi_spread)),
        "minimum_exact_voi": float(eligible_minimum_voi),
        "exact_voi_negative_fraction": float(
            jnp.sum(
                active_eligible[:, None]
                * (anchor_voi < 0.0).astype(jnp.float32)
            )
            / (active_denominator * anchor_voi.shape[-1])
        ),
        "mean_information_gain": float(
            jnp.sum(active_eligible[:, None] * anchor_information_gain)
            / (active_denominator * anchor_information_gain.shape[-1])
        ),
        "mean_information_gain_action_spread": float(
            active_mean(information_spread)
        ),
        "mean_active_passive_policy_tv": float(active_mean(active_passive_tv)),
        "active_passive_greedy_disagreement": float(
            active_mean((active_greedy != passive_greedy).astype(jnp.float32))
        ),
        "base_active_greedy_disagreement": float(
            active_mean((base_greedy != active_greedy).astype(jnp.float32))
        ),
        "mean_adaptation_kl": float(active_mean(anchor_adaptation_kl)),
        "mean_posterior_entropy": float(jnp.mean(posterior_entropy)),
        "mean_filter_kl": float(jnp.mean(filter_kl)),
        "probe_component_event_js": float(
            active_mean(probe_component_event_js_by_anchor)
        ),
        "active_probe_eligible_count": int(jnp.sum(active_eligible)),
        "anchor_count": int(chosen.shape[0]),
        "valid_probe_count": int(jnp.sum(successor_valid)),
        "report_only": True,
    }


def _make_non_anchor_block_kernel(
    *,
    rollout_kernel: Any,
    update_kernel: Any,
    loop_key: Any,
    block_length: int,
    rollout_steps: int,
    shaping_horizon: int,
    environment_count: int,
    minibatches_per_epoch: int,
    update_epochs: int,
) -> Any:
    """Compile consecutive ordinary updates into one device transaction."""

    import jax
    import jax.numpy as jnp

    length = int(block_length)
    if length <= 0:
        raise ValueError("Training block length must be positive.")

    @jax.jit
    def kernel(
        runner_state: Any,
        base_params: Any,
        latent_params: Any,
        target_latent_params: Any,
        base_optimizer_state: Any,
        latent_optimizer_state: Any,
        anchor_buffer: Any,
        start_update: Any,
        start_environment_steps: Any,
    ):
        initial = (
            runner_state,
            base_params,
            latent_params,
            target_latent_params,
            base_optimizer_state,
            latent_optimizer_state,
        )

        def one(carry: Any, offset: Any):
            (
                runner,
                base,
                latent,
                target,
                base_optimizer,
                latent_optimizer,
            ) = carry
            update_index = jnp.asarray(start_update, dtype=jnp.int32) + offset
            environment_steps = (
                jnp.asarray(start_environment_steps, dtype=jnp.int32)
                + offset * int(rollout_steps)
            )
            shaping = jnp.maximum(
                1.0
                - environment_steps.astype(jnp.float32)
                / jnp.asarray(shaping_horizon, dtype=jnp.float32),
                0.0,
            )
            runner, batch, unused_snapshots = rollout_kernel(
                runner,
                base,
                latent,
                shaping,
                loop_key,
            )
            del unused_snapshots
            schedule = environment_minibatch_schedule(
                jax.random.fold_in(loop_key, update_index),
                environment_count=environment_count,
                minibatches_per_epoch=minibatches_per_epoch,
                update_epochs=update_epochs,
            )
            (
                base,
                latent,
                target,
                base_optimizer,
                latent_optimizer,
                metrics,
            ) = update_kernel(
                base,
                latent,
                target,
                base_optimizer,
                latent_optimizer,
                batch,
                schedule,
            )
            return (
                runner,
                base,
                latent,
                target,
                base_optimizer,
                latent_optimizer,
            ), metrics

        core, metrics = jax.lax.scan(
            one, initial, jnp.arange(length, dtype=jnp.int32)
        )
        # Anchor-free blocks never touch the buffer; it rides through so both
        # kernels return the same state tuple.
        return core + (anchor_buffer,), metrics

    return kernel


def _make_anchor_update_kernel(
    *,
    rollout_kernel: Any,
    anchor_kernel: Any,
    update_kernel: Any,
    loop_key: Any,
    rollout_steps: int,
    shaping_horizon: int,
    environment_count: int,
    minibatches_per_epoch: int,
    update_epochs: int,
    gamma: float,
    fresh_states: int,
    replay_states: int,
    model: Any,
) -> Any:
    """Compile snapshot rollout, CRN continuations, and update as one kernel."""

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.anchor_buffer import (
        merge_snapshots,
        push_anchor_buffer,
        sample_anchor_worlds,
    )

    # Deliberately *not* jitted as a whole.  The rollout, the anchor batch and
    # the training update are each already a compiled executable; wrapping them
    # in one more jit makes XLA compile their union as a single HLO module, and
    # that module's backend compilation was measured at over thirty-five
    # minutes -- against five and a half minutes for the entire equivalent v4
    # run.  Composing the three compiled kernels from the host instead costs a
    # couple of round-trips on the one update in ~130 that carries an anchor.
    def kernel(
        runner_state: Any,
        base_params: Any,
        latent_params: Any,
        target_latent_params: Any,
        base_optimizer_state: Any,
        latent_optimizer_state: Any,
        anchor_buffer: Any,
        update_index: Any,
        environment_steps: Any,
    ):
        shaping = jnp.maximum(
            1.0
            - jnp.asarray(environment_steps, dtype=jnp.float32)
            / jnp.asarray(shaping_horizon, dtype=jnp.float32),
            0.0,
        )
        anchor_key = jax.random.fold_in(
            loop_key, 100_000 + jnp.asarray(update_index, dtype=jnp.int32)
        )
        index_key, root_key = jax.random.split(anchor_key)
        runner, batch, snapshots = rollout_kernel(
            runner_state,
            base_params,
            latent_params,
            shaping,
            index_key,
        )
        # Replay reuses the *search* for separable worlds, never a measurement:
        # every continuation below is re-run from the stored world with the
        # parameters in force now.
        measured = snapshots
        if replay_states:
            replay_key, root_key = jax.random.split(root_key)
            drawn, drawn_version = sample_anchor_worlds(
                anchor_buffer, replay_key, replay_states
            )
            measured = merge_snapshots(snapshots, drawn, fresh_states)
            del drawn_version
        anchors = anchor_kernel(
            root_key,
            measured,
            base_params,
            latent_params,
            jnp.asarray(gamma, dtype=jnp.float32),
        )
        next_buffer = anchor_buffer
        if replay_states:
            next_buffer = push_anchor_buffer(anchor_buffer, snapshots, update_index)
        schedule = environment_minibatch_schedule(
            jax.random.fold_in(loop_key, update_index),
            environment_count=environment_count,
            minibatches_per_epoch=minibatches_per_epoch,
            update_epochs=update_epochs,
        )
        # The ordinary module commits response/TD first, then PPO, and the
        # sparse contrast is applied afterward as its own small executable.
        # Its target still uses collection-time base_params: anchors and replay
        # must describe the same continuation policy.  Fusing both executables
        # put two 256-step recurrent scans in one HLO module that XLA could not
        # finish compiling.
        collection_base_params = base_params
        collection_latent_params = latent_params
        (
            base_params,
            latent_params,
            target_latent_params,
            base_optimizer_state,
            latent_optimizer_state,
            metrics,
        ) = update_kernel(
            base_params,
            latent_params,
            target_latent_params,
            base_optimizer_state,
            latent_optimizer_state,
            batch,
            schedule,
        )
        xp_start = int(environment_count) // 2
        xp_batch = slice_rollout_lanes(
            batch,
            jnp.arange(xp_start, int(environment_count), dtype=jnp.int32),
        )
        xp_anchors = anchors._replace(
            lane_indexes=anchors.lane_indexes - xp_start
        )
        latent_params, contrast_state, contrast_metrics = update_contrast_channel(
            model=model,
            latent_params=latent_params,
            base_params=collection_base_params,
            feature_latent_params=collection_latent_params,
            optimizer_state=latent_optimizer_state["contrast"],
            batch=xp_batch,
            anchors=xp_anchors,
        )
        latent_optimizer_state = {
            **latent_optimizer_state,
            "contrast": contrast_state,
        }
        # Contrast corrects the online critic after its ordinary Polyak commit.
        # Fold that sparse correction into the slow target before the next TD
        # update; otherwise each anchor correction is omitted from the target
        # until a later ordinary update happens to copy it indirectly.
        target_latent_params = polyak_update(
            target_latent_params, latent_params
        )
        metrics = {
            **metrics,
            "latent": {**metrics["latent"], **contrast_metrics},
        }
        return (
            runner,
            base_params,
            latent_params,
            target_latent_params,
            base_optimizer_state,
            latent_optimizer_state,
            next_buffer,
        ), metrics

    return kernel


def run_training(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    started = time.perf_counter()
    config = load_config(args.config, run_kind=args.run_kind)
    seed_index = int(args.seed_index)
    engineering_preflight = bool(getattr(args, "engineering_preflight", False))
    if config.run_kind == "formal" and seed_index not in range(10):
        if not (engineering_preflight and seed_index == -1):
            raise ValueError("Formal training seed index must be one of 0..9.")
    if config.run_kind == "formal" or bool(getattr(args, "require_cuda", False)):
        devices = [device for device in jax.devices() if device.platform == "gpu"]
        if len(devices) != 1:
            raise RuntimeError("Formal DELTA requires exactly one visible CUDA device.")
    from .official_adapter import FrozenPartnerPool, VectorEnvironment, validate_official_runtime

    official_runtime = (
        validate_official_runtime() if config.run_kind == "formal" else None
    )
    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(getattr(args, "skip_manifest_file_check", False)),
    )
    members = build_training_partner_pool(config, manifest)
    (
        upstream_steps,
        upstream_gpu_hours,
        upstream_wall_clock_hours,
        upstream_records,
    ) = _upstream_partner_cost(members)
    frozen_pool = FrozenPartnerPool.from_checkpoints(
        [member.checkpoint for member in members],
        parent_training_run_ids=[member.parent_training_run_id for member in members],
    )
    partner_functions = make_static_partner_functions(
        pool=frozen_pool,
        probabilities=np.asarray([member.probability for member in members], dtype=np.float32),
        run_ids=np.arange(len(members), dtype=np.int32),
    )
    partner_parameters = None
    environment = VectorEnvironment.create(config)
    model = DeltaModel(config, environment.observation_shape, OFFICIAL_ACTION_COUNT)
    initializer_path_value = getattr(args, "semantic_initializer", None)
    semantic_variant = config.method_variant in {
        "response_only",
        "delta_passive",
        "delta_active",
        "delta_active_blind_actor",
    }
    require_fitted_initializer = (
        config.run_kind in {"development", "formal"} and semantic_variant
    )
    if initializer_path_value:
        initializer_path = Path(initializer_path_value).resolve()
        semantic_initializer = load_semantic_initializer(
            initializer_path,
            component_count=config.method.latent_components,
            event_count=INTERFACE_EVENT_CLASSES,
        )
    else:
        initializer_path = None
        if require_fitted_initializer:
            raise RuntimeError(
                "Development/formal DELTA v6 semantic variants require "
                "--semantic-initializer built from the lineage-disjoint "
                "calibration panel."
            )
        semantic_initializer = deterministic_simplex_initializer(
            config.method.latent_components, INTERFACE_EVENT_CLASSES
        )
    validate_semantic_initializer_provenance(
        semantic_initializer,
        method=METHOD_VERSION,
        layout=config.environment.layout,
        component_count=config.method.latent_components,
        event_classes=INTERFACE_EVENT_CLASSES,
        protocol_version=config.official_protocol.protocol_version,
        official_source_commit=config.official_protocol.source_commit,
        training_parent_ids={member.parent_training_run_id for member in members},
        expected_calibration_run_ids=(
            row.run_id for row in manifest.by_role("calibration")
        ),
        require_fitted=require_fitted_initializer,
    )
    root = _training_key(int(args.seed_index))
    init_key, runner_key, loop_key = jax.random.split(root, 3)
    initialized_base_params, initialized_latent_params = model.init_parameters(
        init_key, semantic_initializer=semantic_initializer
    )
    sp_initializer = getattr(args, "sp_initializer", None)
    if not sp_initializer:
        raise ValueError("DELTA v6 requires --sp-initializer.")
    initializer_seed = initializer_seed_index(sp_initializer)
    expected_initializer_seed = 0 if seed_index == -1 else seed_index
    if initializer_seed != expected_initializer_seed:
        raise ValueError(
            "--sp-initializer must be the seed-matched run-<seed>/ckpt_final; "
            "engineering seed -1 uses run-0."
        )
    from .official_adapter import validate_official_partner_checkpoint

    validate_official_partner_checkpoint(
        sp_initializer,
        config=config,
        algorithm="rnn-sp",
        seed_index=expected_initializer_seed,
    )
    output = Path(args.output).resolve()
    identity = {
        "stage": "train",
        "method": METHOD_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sp_initializer": str(sp_initializer) if sp_initializer else None,
        "ego_run_id": str(args.ego_run_id),
        "seed_index": int(args.seed_index),
        "run_kind": config.run_kind,
        "layout": config.environment.layout,
        "method_variant": config.method_variant,
        "config": config.to_mapping(),
        "partner_manifest": {
            "path": str(manifest_path),
            "run_ids": [row.run_id for row in manifest.runs],
        },
        "official_runtime": official_runtime,
        "observation_shape": list(environment.observation_shape),
        "action_count": OFFICIAL_ACTION_COUNT,
        "semantic_initializer": {
            "path": None if initializer_path is None else str(initializer_path),
            "artifact": semantic_initializer.to_mapping(),
        },
    }
    ensure_run_identity(output, identity)
    write_json(
        output / "partner_pool.json",
        {
            "sampling": "mechanism_then_family_then_stage_then_run_uniform",
            "members": [
                {
                    "run_id": member.run_id,
                    "mechanism": member.mechanism,
                    "hyperparameter_family": member.hyperparameter_family,
                    "checkpoint_stage": member.checkpoint_stage,
                    "probability": member.probability,
                }
                for member in members
            ],
        },
    )
    write_json(output / "upstream_cost_manifest.json", {"parents": upstream_records})
    ledger = ResourceLedger(
        upstream_partner_steps=upstream_steps,
        shared_gpu_hours=upstream_gpu_hours,
        shared_wall_clock_hours=upstream_wall_clock_hours,
        deployable_parameters=(
            parameter_count(initialized_base_params)
            + parameter_count(initialized_latent_params)
        ),
    )
    checkpoint_identity = {
        "method": METHOD_VERSION,
        "config": config.to_mapping(),
        "partner_manifest_run_ids": [row.run_id for row in manifest.runs],
        "seed_index": int(args.seed_index),
        "sp_initializer": str(sp_initializer),
        "semantic_initializer": semantic_initializer.to_mapping(),
    }
    if bool(getattr(args, "resume", False)):
        restored = load_latest_checkpoint(
            output / "checkpoints", expected_identity=checkpoint_identity
        )
        if restored is None:
            raise FileNotFoundError("--resume requested but no checkpoint exists.")
        _, state = restored
        state = state._replace(
            resource_ledger=_reconcile_upstream_resource_cost(
                state.resource_ledger,
                steps=upstream_steps,
                gpu_hours=upstream_gpu_hours,
                wall_clock_hours=upstream_wall_clock_hours,
            )
        )
    else:
        initialized_base_params = initialize_base_from_official(
            initialized_base_params, sp_initializer
        )
        base_params = initialized_base_params
        latent_params = initialized_latent_params
        base_optimizer_state = init_adam(trainable_base_params(base_params))
        latent_optimizer_state = init_latent_optimizer(latent_params)
        # The target starts as an exact copy: at update zero the slow critic and
        # the online critic are the same estimate, and Polyak separates them.
        target_latent_params = jax.tree_util.tree_map(jnp.asarray, latent_params)
        runner = initialize_runner(
            environment=environment,
            model=model,
            partner_functions=partner_functions,
            random_key=runner_key,
        )
        state = TrainState(
            base_params=base_params,
            latent_params=latent_params,
            target_latent_params=target_latent_params,
            base_optimizer_state=base_optimizer_state,
            latent_optimizer_state=latent_optimizer_state,
            # Filled once the snapshot kernel exists and its shapes are known.
            anchor_buffer=None,
            runner_state=runner,
            update_count=jnp.asarray(0, dtype=jnp.int32),
            effective_environment_steps=jnp.asarray(0, dtype=jnp.int32),
            resource_ledger=ledger.to_mapping(),
        )
    rollout_steps = config.environment.num_envs * config.training.rollout_length
    total_updates = config.training.environment_steps // rollout_steps
    maximum_updates = getattr(args, "maximum_updates", None)
    if maximum_updates is not None:
        total_updates = min(total_updates, int(maximum_updates))
    anchor_functions = _anchor_functions(
        model=model,
        partner_functions=partner_functions,
        partner_parameters=partner_parameters,
        environment=environment,
    )
    compact_rollout_kernel = make_compact_training_rollout_kernel(
        environment=environment,
        model=model,
        partner_functions=partner_functions,
        partner_parameters=partner_parameters,
        length=config.training.rollout_length,
    )
    host_update_count = int(np.asarray(state.update_count))
    host_environment_steps = int(np.asarray(state.effective_environment_steps))
    shaping_horizon = max(int(config.upstream.reward_shaping_horizon), 1)
    checkpoint_updates = max(
        int(config.training.checkpoint_interval_environment_steps) // rollout_steps,
        1,
    )
    anchor_enabled = bool(
        config.anchors.enabled
        and config.method_variant
        in {"delta_passive", "delta_active", "delta_active_blind_actor"}
    )
    anchor_updates = (
        int(config.anchors.interval_environment_steps) // rollout_steps
        if anchor_enabled
        else None
    )
    force_first_anchor = bool(getattr(args, "force_anchor", False))
    total_optimizer_steps = (
        (config.training.environment_steps // rollout_steps)
        * config.training.minibatches_per_epoch
        * config.ppo.update_epochs
    )
    no_anchor_update_kernel = make_training_update_kernel(
        model=model,
        total_optimizer_steps=total_optimizer_steps,
        with_anchors=False,
    )
    anchor_update_kernel = None
    anchor_steps_per_trigger = 0
    if anchor_enabled:
        anchor_rollout_kernel = make_anchor_snapshot_rollout_kernel(
            environment=environment,
            model=model,
            partner_functions=partner_functions,
            partner_parameters=partner_parameters,
            length=config.training.rollout_length,
            # The rollout produces the wide pilot candidate set; the anchor
            # kernel is what narrows it to the measured states.
            states_per_trigger=config.anchors.pilot_states,
        )
        compiled_anchor_batch = make_anchor_batch_kernel(
            functions=anchor_functions,
            action_count=OFFICIAL_ACTION_COUNT,
            fit_replicas=config.anchors.fit_replicas,
            evaluation_replicas=config.anchors.evaluation_replicas,
            horizon=config.method.continuation_horizon,
            collect_successor=(
                config.method_variant
                in {"delta_active", "delta_active_blind_actor"}
            ),
            states_per_trigger=config.anchors.states_per_trigger,
            pilot_replicas=config.anchors.pilot_replicas,
        )
        # Deliberately the anchor-free kernel: see the comment in
        # _make_anchor_update_kernel.
        with_anchor_update_kernel = make_training_update_kernel(
            model=model,
            total_optimizer_steps=total_optimizer_steps,
            with_anchors=False,
        )
        anchor_update_kernel = _make_anchor_update_kernel(
            rollout_kernel=anchor_rollout_kernel,
            anchor_kernel=compiled_anchor_batch,
            update_kernel=with_anchor_update_kernel,
            loop_key=loop_key,
            rollout_steps=rollout_steps,
            shaping_horizon=shaping_horizon,
            environment_count=config.environment.num_envs,
            minibatches_per_epoch=config.training.minibatches_per_epoch,
            update_epochs=config.ppo.update_epochs,
            gamma=config.ppo.gamma,
            fresh_states=(
                config.anchors.pilot_states - config.anchors.replay_states
            ),
            replay_states=config.anchors.replay_states,
            model=model,
        )
        if config.anchors.replay_states and not bool(getattr(args, "resume", False)):
            # Shapes only: eval_shape traces the snapshot kernel without
            # running a rollout, so the buffer is allocated from the real
            # structure rather than a hand-copied one that could drift.
            structure = jax.eval_shape(
                anchor_rollout_kernel,
                state.runner_state,
                state.base_params,
                state.latent_params,
                jnp.asarray(0.0, dtype=jnp.float32),
                jax.random.PRNGKey(0),
            )[2]
            state = state._replace(
                anchor_buffer=init_anchor_buffer(
                    jax.tree_util.tree_map(
                        lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype),
                        structure,
                    ),
                    config.anchors.buffer_triggers,
                )
            )
        anchor_replicas = (
            config.anchors.fit_replicas + config.anchors.evaluation_replicas
        )
        current_anchor_steps = (
            config.anchors.states_per_trigger
            * OFFICIAL_ACTION_COUNT
            * anchor_replicas
            * config.method.continuation_horizon
        )
        successor_anchor_steps = 0
        if config.method_variant in {"delta_active", "delta_active_blind_actor"}:
            successor_anchor_steps = (
                config.anchors.states_per_trigger
                * OFFICIAL_ACTION_COUNT
                * anchor_replicas
                * (
                    2
                    + OFFICIAL_ACTION_COUNT
                    * config.method.continuation_horizon
                )
            )
        pilot_anchor_steps = (
            config.anchors.pilot_states
            * OFFICIAL_ACTION_COUNT
            * config.anchors.pilot_replicas
            * config.method.continuation_horizon
        )
        anchor_steps_per_trigger = (
            pilot_anchor_steps + current_anchor_steps + successor_anchor_steps
        )

    def is_anchor_update(update_index: int) -> bool:
        return bool(
            anchor_enabled
            and (
                (update_index + 1) % int(anchor_updates) == 0
                or (force_first_anchor and update_index == 0)
            )
        )

    # Plan every executable shape before the first training dispatch.  Normal
    # development/formal runs need one cadence shape and at most one shorter
    # final block.
    planned_lengths: set[int] = set()
    planned_cursor = host_update_count
    while planned_cursor < total_updates:
        if is_anchor_update(planned_cursor):
            planned_cursor += 1
            continue
        next_checkpoint = (
            (planned_cursor // checkpoint_updates) + 1
        ) * checkpoint_updates
        next_anchor = total_updates
        if anchor_enabled:
            candidate = planned_cursor
            while candidate < total_updates and not is_anchor_update(candidate):
                candidate += 1
            next_anchor = candidate
        planned_end = min(total_updates, next_checkpoint, next_anchor)
        if planned_end <= planned_cursor:
            raise AssertionError("Training block planner made no progress.")
        planned_lengths.add(planned_end - planned_cursor)
        planned_cursor = planned_end
    block_kernels = {
        length: _make_non_anchor_block_kernel(
            rollout_kernel=compact_rollout_kernel,
            update_kernel=no_anchor_update_kernel,
            loop_key=loop_key,
            block_length=length,
            rollout_steps=rollout_steps,
            shaping_horizon=shaping_horizon,
            environment_count=config.environment.num_envs,
            minibatches_per_epoch=config.training.minibatches_per_epoch,
            update_epochs=config.ppo.update_epochs,
        )
        for length in sorted(planned_lengths)
    }

    # Pending entries hold whole device blocks.  They are transferred together
    # only at an anchor/checkpoint/log boundary.
    pending_metrics: list[tuple[int, int, bool, Any]] = []
    io_futures: list[Future[Any]] = []
    io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="delta-io")

    def reap_io() -> None:
        for future in tuple(io_futures):
            if future.done():
                future.result()
                io_futures.remove(future)

    def flush_boundary(*, checkpoint_step: int | None = None) -> None:
        nonlocal pending_metrics
        if not pending_metrics and checkpoint_step is None:
            return
        reap_io()
        metric_trees = [entry[3] for entry in pending_metrics]
        if checkpoint_step is None:
            hosted_metrics = jax.device_get(metric_trees)
            hosted_state = None
        else:
            hosted_metrics, hosted_state = jax.device_get((metric_trees, state))
        payloads: list[Mapping[str, Any]] = []
        failed = False
        for (start, count, anchor_trigger, _), segment in zip(
            pending_metrics, hosted_metrics, strict=True
        ):
            for offset in range(count):
                metrics = jax.tree_util.tree_map(
                    lambda value: np.asarray(value)[offset], segment
                )
                converted = _host_converted(metrics)
                update_number = start + offset + 1
                environment_steps = update_number * rollout_steps
                shaping = max(
                    1.0
                    - (environment_steps - rollout_steps) / float(shaping_horizon),
                    0.0,
                )
                payloads.append(
                    {
                        "update": update_number,
                        "environment_steps": environment_steps,
                        "anchor_trigger": anchor_trigger,
                        "official_shaping_factor": shaping,
                        "metrics": converted,
                    }
                )
                failed = failed or (
                    float(converted["ppo"]["base_update_applied"]) < 1.0
                    or float(converted["ppo"]["base_nonfinite_update"]) > 0.0
                    or float(converted["latent"]["latent_nonfinite_update"]) > 0.0
                )
        pending_metrics = []
        if payloads:
            io_futures.append(
                io_executor.submit(
                    _write_jsonl_batch,
                    output / "records" / "metrics.jsonl",
                    payloads,
                )
            )
        if checkpoint_step is not None:
            io_futures.append(
                io_executor.submit(
                    save_checkpoint,
                    output / "checkpoints",
                    step=checkpoint_step,
                    state=hosted_state,
                    identity=checkpoint_identity,
                )
            )
        if failed:
            for future in io_futures:
                future.result()
            raise FloatingPointError(
                "A DELTA optimizer transaction produced NaN/Inf and was rolled back."
            )

    update = host_update_count
    while update < total_updates:
        if is_anchor_update(update):
            assert anchor_update_kernel is not None
            core, metrics = anchor_update_kernel(
                state.runner_state,
                state.base_params,
                state.latent_params,
                state.target_latent_params,
                state.base_optimizer_state,
                state.latent_optimizer_state,
                state.anchor_buffer,
                update,
                host_environment_steps,
            )
            count = 1
            anchor_trigger = True
            anchor_cost = anchor_steps_per_trigger
            metrics = jax.tree_util.tree_map(lambda value: value[None], metrics)
        else:
            next_checkpoint = (
                (update // checkpoint_updates) + 1
            ) * checkpoint_updates
            next_anchor = total_updates
            if anchor_enabled:
                candidate = update
                while candidate < total_updates and not is_anchor_update(candidate):
                    candidate += 1
                next_anchor = candidate
            block_end = min(total_updates, next_checkpoint, next_anchor)
            count = block_end - update
            if count <= 0:
                raise AssertionError("Training block execution made no progress.")
            core, metrics = block_kernels[count](
                state.runner_state,
                state.base_params,
                state.latent_params,
                state.target_latent_params,
                state.base_optimizer_state,
                state.latent_optimizer_state,
                state.anchor_buffer,
                update,
                host_environment_steps,
            )
            anchor_trigger = False
            anchor_cost = 0
        (
            runner,
            base_params,
            latent_params,
            target_latent_params,
            base_optimizer_state,
            latent_optimizer_state,
            anchor_buffer,
        ) = core
        next_update = update + count
        next_steps = host_environment_steps + count * rollout_steps
        ledger = ResourceLedger.from_mapping(state.resource_ledger).plus(
            ego_policy_steps=count * rollout_steps,
            anchor_continuation_steps=anchor_cost,
        )
        state = TrainState(
            base_params=base_params,
            latent_params=latent_params,
            target_latent_params=target_latent_params,
            base_optimizer_state=base_optimizer_state,
            latent_optimizer_state=latent_optimizer_state,
            anchor_buffer=anchor_buffer,
            runner_state=runner,
            update_count=next_update,
            effective_environment_steps=next_steps,
            resource_ledger=ledger.to_mapping(),
        )
        pending_metrics.append((update, count, anchor_trigger, metrics))
        update = next_update
        host_environment_steps = next_steps
        checkpoint_due = (
            update % checkpoint_updates == 0
            and next_steps < config.training.environment_steps
        )
        if anchor_trigger or checkpoint_due:
            flush_boundary(
                checkpoint_step=next_steps if checkpoint_due else None
            )
    flush_boundary()
    for future in io_futures:
        future.result()
    io_executor.shutdown(wait=True)
    elapsed = time.perf_counter() - started
    ledger = ResourceLedger.from_mapping(state.resource_ledger).plus(
        training_wall_clock_hours=elapsed / 3600.0,
        training_gpu_hours=(
            elapsed / 3600.0
            if any(device.platform == "gpu" for device in jax.devices())
            else 0.0
        ),
    )
    ledger = ResourceLedger(
        **{
            **{name: getattr(ledger, name) for name in ResourceLedger.__dataclass_fields__},
            "peak_memory_bytes": max(
                int(ledger.peak_memory_bytes), peak_device_memory_bytes()
            ),
        }
    )
    state = state._replace(resource_ledger=ledger.to_mapping())
    if (
        config.run_kind == "formal"
        and ledger.peak_memory_bytes >= FORMAL_PEAK_MEMORY_LIMIT_BYTES
    ):
        write_json(output / "resource_ledger.json", ledger.to_mapping())
        raise RuntimeError("Formal DELTA peak device memory must remain below 40,000 MiB.")

    deployment_directory = output / "final_deployment"
    if deployment_directory.exists() and any(deployment_directory.iterdir()):
        from .deployment import load_deployment

        existing = load_deployment(deployment_directory)
        if not _same_parameters(
            existing.base_params, state.base_params
        ) or not _same_parameters(existing.latent_params, state.latent_params):
            raise RuntimeError("Existing final deployment differs from the final checkpoint.")
    else:
        export_deployment_bundle(
            deployment_directory,
            ego_run_id=str(args.ego_run_id),
            config=config,
            observation_shape=environment.observation_shape,
            base_params=state.base_params,
            latent_params=state.latent_params,
            source_training_run=output,
            semantic_initializer=semantic_initializer.to_mapping(include_bias=False),
        )

    # Fresh report-only decision audit.  It never alters learned parameters and
    # is idempotent across resume of a completed run.
    audit_path = output / "final_decision_audit.json"
    final_environment_steps = int(np.asarray(state.effective_environment_steps))
    audit_complete = False
    if audit_path.is_file():
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_complete = (
            int(existing_audit.get("environment_steps", -1)) == final_environment_steps
        )
    if anchor_enabled and not audit_complete:
        audit_started = time.perf_counter()
        audit_key = jax.random.fold_in(loop_key, 999_999)
        audit_index_key, audit_root_key = jax.random.split(audit_key)
        audit_runner, audit_batch, audit_snapshots = anchor_rollout_kernel(
            state.runner_state,
            state.base_params,
            state.latent_params,
            jnp.asarray(0.0, dtype=jnp.float32),
            audit_index_key,
        )
        del audit_runner
        audit_anchors = compiled_anchor_batch(
            audit_root_key,
            audit_snapshots,
            state.base_params,
            state.latent_params,
            jnp.asarray(config.ppo.gamma, dtype=jnp.float32),
        )
        # Decision-side audit on a freshly measured anchor batch.  The oracle
        # entries are computed against the evaluation replicas, which no loss
        # reads, and ``anchor_oracle_repeatability`` bounds every other ordering
        # number in this file: when the fit and evaluation halves of the same
        # anchor disagree about the best action, nothing measured against the
        # fit half is skill.
        def decision_metrics(latent, base, batch, anchors):
            metrics = latent_composite_loss(
                model, latent, base, batch, anchors
            ).metrics
            return {name: metrics[name] for name in DECISION_METRIC_NAMES}

        decision_audit = jax.jit(decision_metrics)(
            state.latent_params,
            state.base_params,
            audit_batch,
            audit_anchors,
        )
        decision_audit = jax.tree_util.tree_map(
            jax.block_until_ready, decision_audit
        )
        audit_replicas = (
            config.anchors.fit_replicas + config.anchors.evaluation_replicas
        )
        audit_current_steps = (
            config.anchors.states_per_trigger
            * OFFICIAL_ACTION_COUNT
            * audit_replicas
            * config.method.continuation_horizon
        )
        audit_successor_steps = 0
        if config.method_variant in {"delta_active", "delta_active_blind_actor"}:
            audit_successor_steps = (
                config.anchors.states_per_trigger
                * OFFICIAL_ACTION_COUNT
                * audit_replicas
                * (
                    2
                    + OFFICIAL_ACTION_COUNT
                    * config.method.continuation_horizon
                )
            )
        audit_pilot_steps = (
            config.anchors.pilot_states
            * OFFICIAL_ACTION_COUNT
            * config.anchors.pilot_replicas
            * config.method.continuation_horizon
        )
        audit_steps = (
            config.environment.num_envs * config.training.rollout_length
            + audit_pilot_steps
            + audit_current_steps
            + audit_successor_steps
        )
        audit_elapsed = time.perf_counter() - audit_started
        ledger = ledger.plus(
            final_decision_audit_steps=audit_steps,
            measurement_wall_clock_hours=audit_elapsed / 3600.0,
            measurement_gpu_hours=(
                audit_elapsed / 3600.0
                if any(device.platform == "gpu" for device in jax.devices())
                else 0.0
            ),
        )
        state = state._replace(resource_ledger=ledger.to_mapping())
        write_json(
            audit_path,
            {
                "version": 5,
                "artifact_type": "delta_v6_final_decision_audit",
                **_final_decision_audit(
                    model=model,
                    base_params=state.base_params,
                    latent_params=state.latent_params,
                    batch=audit_batch,
                    anchors=audit_anchors,
                ),
                "decision_channel": _host_converted(decision_audit),
                "environment_steps": final_environment_steps,
            },
        )

    measured_latency = _measure_inference_latency_ms(
        model=model,
        base_params=state.base_params,
        latent_params=state.latent_params,
    )
    current_ledger = ResourceLedger.from_mapping(state.resource_ledger)
    if current_ledger.inference_latency_ms == 0.0:
        current_ledger = ResourceLedger(
            **{
                **{
                    name: getattr(current_ledger, name)
                    for name in ResourceLedger.__dataclass_fields__
                },
                "inference_latency_ms": measured_latency,
            }
        )
        state = state._replace(resource_ledger=current_ledger.to_mapping())

    save_checkpoint(
        output / "checkpoints",
        step=int(np.asarray(state.effective_environment_steps)),
        state=state,
        identity=checkpoint_identity,
    )
    write_json(output / "resource_ledger.json", ResourceLedger.from_mapping(state.resource_ledger).to_mapping())


__all__ = ["run_cuda_preflight", "run_training"]


def run_cuda_preflight(args: argparse.Namespace) -> None:
    """Execute one registered real-partner mechanical update and persistence round trip."""
    if int(args.seed_index) != -1:
        raise ValueError("CUDA preflight uses the engineering seed index -1.")
    args.maximum_updates = 1
    args.force_anchor = True
    args.require_cuda = True
    args.resume = False
    args.engineering_preflight = True
    run_training(args)
    output = Path(args.output).resolve()
    deployment = output / "final_deployment"
    checkpoint = output / "checkpoints" / "latest.json"
    audit = output / "final_decision_audit.json"
    ledger = output / "resource_ledger.json"
    for path in (deployment / "deployment_bundle.json", checkpoint, audit, ledger):
        if not path.is_file():
            raise RuntimeError(f"CUDA preflight artifact is missing: {path}")
    write_json(
        output / "cuda_preflight_report.json",
        {
            "passed": True,
            "method": METHOD_VERSION,
            "config": str(Path(args.config).resolve()),
            "partner_manifest": str(Path(args.partner_manifest).resolve()),
            "deployment": str(deployment),
            "checkpoint_descriptor": str(checkpoint),
            "final_decision_audit": str(audit),
            "resource_ledger": str(ledger),
        },
    )
