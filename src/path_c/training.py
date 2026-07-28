"""Path C V4.4 transition batches, Bellman assignment, and updates."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .method import (
    CodebookState,
    nearest_codes,
    slot_bayes_update,
    solve_temperature_for_target_kl,
    target_response_signatures,
    uniform_slot_log_belief,
    update_codebook,
)
from .model import model_forward


class ModelFunctions(NamedTuple):
    heads: Any
    official_sequence: Callable[..., tuple[Any, Any, Any, Any]]


class TransitionBatch(NamedTuple):
    """All loss inputs.  Partner identity is deliberately absent."""

    observations: Any
    response_next_observations: Any
    episode_start: Any
    previous_actions: Any
    previous_team_rewards: Any
    initial_official_carry: Any
    initial_control_carry: Any
    reference_logits: Any
    execution_logits: Any
    generic_execution_logits: Any
    behavior_logits: Any
    posterior_scores: Any
    generic_scores: Any
    slot_log_beliefs: Any
    actions: Any
    rewards: Any
    dones: Any
    response_codes: Any


class TrainState(NamedTuple):
    online_params: Any
    target_params: Any
    bellman_optimizer_state: Any
    outcome_optimizer_state: Any
    codebook: CodebookState
    runner_state: Any
    random_key: Any
    update_count: Any


class OptimizerBundle(NamedTuple):
    bellman_optimizer: Any
    outcome_optimizer: Any
    bellman_state: Any
    outcome_state: Any


class TrainingUpdate(NamedTuple):
    params: Any
    target_params: Any
    bellman_optimizer_state: Any
    outcome_optimizer_state: Any
    metrics: Mapping[str, Any]


class FrozenAssignments(NamedTuple):
    responsibilities: Any
    responsibility_energies: Any
    bootstrap_mask: Any
    bellman_targets: Any
    one_step_bellman_targets: Any
    stale_bellman_targets: Any
    importance_ratio_mean: Any
    trace_coefficient_mean: Any
    stale_current_beliefs: Any
    response_signature_targets: Any
    response_code_targets: Any
    next_q_use_targets: Any
    next_q_mask_targets: Any
    next_reference_logits_targets: Any
    td_energies: Any
    response_energies: Any
    reward_energies: Any
    next_q_use_energies: Any
    next_q_mask_energies: Any
    next_reference_energy: Any


class LossBundle(NamedTuple):
    total: Any
    metrics: Mapping[str, Any]


def gather_actions(values: Any, actions: Any, *, action_axis: int = -1) -> Any:
    import jax.numpy as jnp

    source = jnp.asarray(values)
    indexes = jnp.asarray(actions, dtype=jnp.int32)
    axis = action_axis if action_axis >= 0 else source.ndim + action_axis
    shape = list(indexes.shape)
    while len(shape) < source.ndim:
        shape.insert(axis, 1)
    return jnp.squeeze(
        jnp.take_along_axis(source, indexes.reshape(tuple(shape)), axis=axis),
        axis=axis,
    )


def huber(error: Any, delta: float = 1.0) -> Any:
    import jax.numpy as jnp

    absolute = jnp.abs(jnp.asarray(error))
    quadratic = jnp.minimum(absolute, float(delta))
    return 0.5 * jnp.square(quadratic) + float(delta) * (
        absolute - quadratic
    )


def gaussian_negative_log_likelihood(
    value: Any, mean: Any, log_standard_deviation: Any
) -> Any:
    import jax.numpy as jnp

    log_std = jnp.asarray(log_standard_deviation)
    normalized = (jnp.asarray(value) - jnp.asarray(mean)) * jnp.exp(-log_std)
    return 0.5 * jnp.square(normalized) + log_std


def bellman_targets(
    *,
    rewards: Any,
    dones: Any,
    next_execution_probabilities: Any,
    target_next_q_values: Any,
    gamma: float,
) -> Any:
    import jax.numpy as jnp

    target_q = jnp.min(jnp.asarray(target_next_q_values), axis=-3)
    expected = jnp.einsum(
        "...a,...ma->...m",
        jnp.asarray(next_execution_probabilities),
        target_q,
    )
    return jnp.asarray(rewards)[..., None] + float(gamma) * (
        1.0 - jnp.asarray(dones, dtype=jnp.float32)
    )[..., None] * expected


def retrace_targets(
    *,
    rewards: Any,
    dones: Any,
    actions: Any,
    behavior_logits: Any,
    target_execution_probabilities: Any,
    target_q_values: Any,
    gamma: float,
    trace_lambda: float,
    importance_ratio_clip: float,
) -> tuple[Any, Any, Any]:
    """Full-episode Retrace targets for off-policy recurrent action values.

    ``target_q_values`` and ``target_execution_probabilities`` include the final
    bootstrap state and therefore have length ``T+1``.  The returned target has
    shape ``[T,B,M]``.  Importance ratios are computed from the action actually
    sampled from the recorded behavior distribution; clipped trace coefficients
    propagate sparse reward information backward without changing the deployed
    target policy.
    """

    import jax
    import jax.numpy as jnp

    reward = jnp.asarray(rewards, dtype=jnp.float32)
    done = jnp.asarray(dones, dtype=jnp.bool_)
    action = jnp.asarray(actions, dtype=jnp.int32)
    behavior_log = jax.nn.log_softmax(
        jnp.asarray(behavior_logits, dtype=jnp.float32), axis=-1
    )
    target_probability = jnp.asarray(
        target_execution_probabilities, dtype=jnp.float32
    )
    q_values = jnp.min(jnp.asarray(target_q_values, dtype=jnp.float32), axis=-3)
    if target_probability.shape[0] != reward.shape[0] + 1:
        raise ValueError("Retrace target policies must include T+1 states.")
    if q_values.shape[0] != reward.shape[0] + 1:
        raise ValueError("Retrace Q values must include T+1 states.")
    if action.shape != reward.shape or done.shape != reward.shape:
        raise ValueError("Retrace actions and done flags must align with rewards.")
    if behavior_log.shape[:-1] != reward.shape:
        raise ValueError("Behavior logits must align with the reward prefix axes.")
    if target_probability.shape[1:-1] != reward.shape[1:]:
        raise ValueError("Retrace target-policy prefixes do not align with rewards.")
    if q_values.shape[1:-2] != reward.shape[1:]:
        raise ValueError("Retrace Q prefixes do not align with rewards.")
    if q_values.shape[-1] != target_probability.shape[-1]:
        raise ValueError("Retrace policy and Q action axes differ.")
    if not 0.0 <= float(trace_lambda) <= 1.0:
        raise ValueError("Retrace lambda must lie in [0, 1].")
    if not 0.0 < float(importance_ratio_clip) <= 1.0:
        raise ValueError("Retrace importance clip must lie in (0, 1].")

    selected_q = gather_actions(q_values[:-1], action, action_axis=-1)
    next_value = jnp.sum(
        target_probability[1:, ..., None, :] * q_values[1:], axis=-1
    )
    delta = reward[..., None] + float(gamma) * (
        1.0 - done.astype(jnp.float32)
    )[..., None] * next_value - selected_q

    selected_target_probability = gather_actions(
        target_probability[:-1], action, action_axis=-1
    )
    selected_behavior_probability = jnp.exp(
        gather_actions(behavior_log, action, action_axis=-1)
    )
    ratio = selected_target_probability / jnp.maximum(
        selected_behavior_probability, 1.0e-12
    )
    coefficient = float(trace_lambda) * jnp.minimum(
        ratio, float(importance_ratio_clip)
    )
    next_coefficient = jnp.concatenate(
        (coefficient[1:], jnp.zeros_like(coefficient[:1])), axis=0
    )

    def backward(correction_next: Any, values: tuple[Any, ...]) -> tuple[Any, Any]:
        q_at_time, delta_at_time, terminal, c_next = values
        target = q_at_time + delta_at_time + float(gamma) * (
            1.0 - terminal.astype(jnp.float32)
        )[..., None] * c_next[..., None] * correction_next
        return target - q_at_time, target

    unused_correction, reversed_target = jax.lax.scan(
        backward,
        jnp.zeros_like(selected_q[-1]),
        (
            selected_q[::-1],
            delta[::-1],
            done[::-1],
            next_coefficient[::-1],
        ),
    )
    del unused_correction
    return (
        jax.lax.stop_gradient(reversed_target[::-1]),
        jax.lax.stop_gradient(ratio),
        jax.lax.stop_gradient(coefficient),
    )


def _td_evidence_items(*, q_values: Any, actions: Any, targets: Any) -> Any:
    import jax.numpy as jnp

    selected = gather_actions(q_values, actions, action_axis=-1)
    return jnp.mean(
        huber(selected - jnp.asarray(targets)[..., None, :]), axis=-2
    )


def _selected_outcome_items(
    *,
    response_logits: Any,
    reward_mean: Any,
    reward_log_standard_deviation: Any,
    next_q_use_mean: Any,
    next_q_use_log_standard_deviation: Any,
    next_q_mask_mean: Any,
    next_q_mask_log_standard_deviation: Any,
    next_reference_logits_mean: Any,
    response_codes: Any,
    rewards: Any,
    next_q_use_targets: Any,
    next_q_mask_targets: Any,
    next_reference_logits_targets: Any,
    actions: Any,
) -> Mapping[str, Any]:
    """Per-transition physical-model evidence.

    Slot-specific items have shape ``[T,B,M]``.  The next-reference item has
    shape ``[T,B]`` and is never used to assign a latent slot.
    """

    import jax
    import jax.numpy as jnp

    action_response_logits = gather_actions(
        response_logits, actions, action_axis=-2
    )
    response_target = jnp.asarray(response_codes, dtype=jnp.int32)
    response_nll = -jnp.take_along_axis(
        jax.nn.log_softmax(action_response_logits, axis=-1),
        response_target[..., None, None],
        axis=-1,
    )[..., 0]

    selected_reward_mean = gather_actions(reward_mean, actions, action_axis=-1)
    selected_reward_log_std = gather_actions(
        reward_log_standard_deviation, actions, action_axis=-1
    )
    reward_nll = gaussian_negative_log_likelihood(
        jnp.asarray(rewards)[..., None],
        selected_reward_mean,
        selected_reward_log_std,
    )

    def q_items(mean: Any, log_std: Any, target: Any) -> Any:
        selected_action_mean = gather_actions(mean, actions, action_axis=-3)
        selected_action_std = gather_actions(log_std, actions, action_axis=-3)
        selected_mean = gather_actions(
            selected_action_mean, response_codes, action_axis=-2
        )
        selected_std = gather_actions(
            selected_action_std, response_codes, action_axis=-2
        )
        items = gaussian_negative_log_likelihood(
            jnp.asarray(target), selected_mean, selected_std
        )
        # [T,B,E,M,U] -> [T,B,M]
        return jnp.mean(items, axis=(-3, -1))

    use_nll = q_items(
        next_q_use_mean,
        next_q_use_log_standard_deviation,
        next_q_use_targets,
    )
    mask_nll = q_items(
        next_q_mask_mean,
        next_q_mask_log_standard_deviation,
        next_q_mask_targets,
    )

    selected_reference = gather_actions(
        next_reference_logits_mean, actions, action_axis=-3
    )
    selected_reference = gather_actions(
        selected_reference, response_codes, action_axis=-2
    )
    reference_error = jnp.mean(
        jnp.square(
            selected_reference - jnp.asarray(next_reference_logits_targets)
        ),
        axis=-1,
    )
    return {
        "response": response_nll,
        "reward": reward_nll,
        "next_q_use": use_nll,
        "next_q_mask": mask_nll,
        "next_reference": reference_error,
    }


def episode_responsibility_evidence(
    *,
    q_values: Any,
    actions: Any,
    targets: Any,
    temperature: float,
    availability_mask: Any | None = None,
) -> tuple[Any, Any]:
    """Assign latent slots from complete-episode Bellman evidence only."""

    import jax
    import jax.numpy as jnp

    energies = jnp.sum(
        _td_evidence_items(q_values=q_values, actions=actions, targets=targets),
        axis=0,
    )
    available = (
        jnp.ones_like(energies, dtype=jnp.bool_)
        if availability_mask is None
        else jnp.asarray(availability_mask, dtype=jnp.bool_)
    )
    if available.shape != energies.shape:
        raise ValueError("Responsibility availability must be [environment,slot].")
    minimum = jnp.min(
        jnp.where(available, energies, jnp.inf), axis=-1, keepdims=True
    )
    logits = jnp.where(
        available,
        -(energies - minimum) / float(temperature),
        -jnp.inf,
    )
    return (
        jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1)),
        jax.lax.stop_gradient(energies),
    )


def sample_bootstrap_mask(
    key: Any,
    *,
    environment_count: int,
    slot_count: int,
    probability: float,
) -> Any:
    import jax
    import jax.numpy as jnp

    mask_key, required_slot_key = jax.random.split(key)
    mask = jax.random.bernoulli(
        mask_key, float(probability), (environment_count, slot_count)
    )
    required_slot = jax.random.randint(
        required_slot_key, (environment_count,), 0, slot_count
    )
    required_mask = jax.nn.one_hot(
        required_slot, slot_count, dtype=jnp.bool_
    )
    return jnp.where(
        jnp.any(mask, axis=-1, keepdims=True), mask, required_mask
    )


def bellman_loss(
    *,
    q_values: Any,
    actions: Any,
    targets: Any,
    stopped_responsibilities: Any,
    bootstrap_mask: Any,
    metric_prefix: str = "bellman",
) -> LossBundle:
    import jax
    import jax.numpy as jnp

    selected = gather_actions(q_values, actions, action_axis=-1)
    per_item = huber(selected - jnp.asarray(targets)[..., None, :])
    weights = jax.lax.stop_gradient(
        jnp.asarray(stopped_responsibilities)
        * jnp.asarray(bootstrap_mask, dtype=jnp.float32)
    )
    time_weights = weights[None, :, None, :]
    denominator = q_values.shape[0] * 2.0 * jnp.sum(weights) + 1.0e-8
    loss = jnp.sum(per_item * time_weights) / denominator
    return LossBundle(
        total=loss,
        metrics={
            metric_prefix: loss,
            f"{metric_prefix}_mean_absolute_td_error": jnp.sum(
                jnp.abs(selected - jnp.asarray(targets)[..., None, :])
                * time_weights
            )
            / denominator,
        },
    )


def outcome_loss(
    *,
    response_logits: Any,
    reward_mean: Any,
    reward_log_standard_deviation: Any,
    next_q_use_mean: Any,
    next_q_use_log_standard_deviation: Any,
    next_q_mask_mean: Any,
    next_q_mask_log_standard_deviation: Any,
    next_reference_logits_mean: Any,
    response_codes: Any,
    rewards: Any,
    next_q_use_targets: Any,
    next_q_mask_targets: Any,
    next_reference_logits_targets: Any,
    actions: Any,
    stopped_responsibilities: Any,
    bootstrap_mask: Any | None = None,
) -> LossBundle:
    import jax
    import jax.numpy as jnp

    parts = _selected_outcome_items(
        response_logits=response_logits,
        reward_mean=reward_mean,
        reward_log_standard_deviation=reward_log_standard_deviation,
        next_q_use_mean=next_q_use_mean,
        next_q_use_log_standard_deviation=(
            next_q_use_log_standard_deviation
        ),
        next_q_mask_mean=next_q_mask_mean,
        next_q_mask_log_standard_deviation=(
            next_q_mask_log_standard_deviation
        ),
        next_reference_logits_mean=next_reference_logits_mean,
        response_codes=response_codes,
        rewards=rewards,
        next_q_use_targets=next_q_use_targets,
        next_q_mask_targets=next_q_mask_targets,
        next_reference_logits_targets=next_reference_logits_targets,
        actions=actions,
    )
    weights = jnp.asarray(stopped_responsibilities)
    if bootstrap_mask is not None:
        weights = weights * jnp.asarray(bootstrap_mask, dtype=jnp.float32)
    weights = jax.lax.stop_gradient(weights)[None, ...]
    denominator = response_logits.shape[0] * jnp.sum(weights) + 1.0e-8

    slot_terms = (
        parts["response"]
        + parts["reward"]
        + parts["next_q_use"]
        + parts["next_q_mask"]
    )
    fitted_slots = jnp.sum(slot_terms * weights) / denominator
    reference_loss = jnp.mean(parts["next_reference"])
    total = fitted_slots + reference_loss
    return LossBundle(
        total=total,
        metrics={
            "slot_outcome": fitted_slots,
            "response_nll": jnp.sum(parts["response"] * weights)
            / denominator,
            "reward_nll": jnp.sum(parts["reward"] * weights) / denominator,
            "next_q_use_nll": jnp.sum(parts["next_q_use"] * weights)
            / denominator,
            "next_q_mask_nll": jnp.sum(parts["next_q_mask"] * weights)
            / denominator,
            "next_reference_mse": reference_loss,
        },
    )


def response_encoder_loss(
    *,
    predicted_signatures: Any,
    target_signatures: Any,
    target_codes: Any,
    codebook_embeddings: Any,
) -> LossBundle:
    import jax
    import jax.numpy as jnp

    predicted = jnp.asarray(predicted_signatures)
    target = jax.lax.stop_gradient(jnp.asarray(target_signatures))
    codebook = jax.lax.stop_gradient(jnp.asarray(codebook_embeddings))
    logits = -jnp.sum(jnp.square(predicted[..., None, :] - codebook), axis=-1)
    targets = jnp.asarray(target_codes, dtype=jnp.int32)
    valid = targets < codebook.shape[0]
    safe_targets = jnp.where(valid, targets, 0)
    items = -jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1),
        safe_targets[..., None],
        axis=-1,
    )[..., 0]
    denominator = jnp.sum(valid.astype(jnp.float32)) + 1.0e-8
    classification = jnp.sum(items * valid.astype(jnp.float32)) / denominator
    commitment = jnp.sum(
        jnp.mean(jnp.square(predicted - target), axis=-1)
        * valid.astype(jnp.float32)
    ) / denominator
    return LossBundle(
        total=classification + commitment,
        metrics={
            "response_encoder_classification": classification,
            "response_encoder_commitment": commitment,
        },
    )


def _head_labels(params: Mapping[str, Any], *, loss: str) -> Any:
    import jax

    if loss == "bellman":
        selected = {"control_memory"}
    elif loss == "outcome":
        selected = {"outcome", "response_encoder"}
    else:
        raise ValueError("Unknown loss partition.")

    labels = {}
    for name, subtree in params.items():
        if loss == "bellman" and name == "q_heads":
            labels[name] = {
                estimator_name: jax.tree_util.tree_map(
                    lambda unused, label=(
                        "train"
                        if str(estimator_name).startswith("LearnedEstimator_")
                        else "frozen"
                    ): label,
                    estimator_params,
                )
                for estimator_name, estimator_params in subtree.items()
            }
        else:
            label = "train" if name in selected else "frozen"
            labels[name] = jax.tree_util.tree_map(
                lambda unused, current=label: current, subtree
            )
    return labels


def make_optimizers(
    params: Mapping[str, Any],
    *,
    bellman_learning_rate: float,
    outcome_learning_rate: float,
    gradient_clip_norm: float,
) -> OptimizerBundle:
    import jax
    import optax

    bellman_labels = {
        "official": jax.tree_util.tree_map(
            lambda unused: "train", params["official"]
        ),
        "heads": _head_labels(params["heads"], loss="bellman"),
    }
    outcome_labels = {
        "official": jax.tree_util.tree_map(
            lambda unused: "frozen", params["official"]
        ),
        "heads": _head_labels(params["heads"], loss="outcome"),
    }

    def transform(rate: float, labels: Any) -> Any:
        return optax.multi_transform(
            {
                "train": optax.chain(
                    optax.clip_by_global_norm(float(gradient_clip_norm)),
                    optax.adam(float(rate)),
                ),
                "frozen": optax.set_to_zero(),
            },
            labels,
        )

    bellman = transform(bellman_learning_rate, bellman_labels)
    outcome = transform(outcome_learning_rate, outcome_labels)
    return OptimizerBundle(
        bellman_optimizer=bellman,
        outcome_optimizer=outcome,
        bellman_state=bellman.init(params),
        outcome_state=outcome.init(params),
    )


def apply_model_sequence(
    *,
    functions: ModelFunctions,
    params: Mapping[str, Any],
    batch: TransitionBatch,
) -> Mapping[str, Any]:
    unused_final_carry, official_features, unused_logits, unused_values = (
        functions.official_sequence(
            params["official"],
            batch.initial_official_carry,
            batch.observations,
            batch.episode_start,
        )
    )
    del unused_final_carry, unused_logits, unused_values
    unused_control_carry, output = functions.heads.apply(
        {"params": params["heads"]},
        batch.initial_control_carry,
        official_features,
        batch.previous_actions,
        batch.previous_team_rewards,
        batch.episode_start,
        batch.slot_log_beliefs,
        method=functions.heads.sequence,
    )
    del unused_control_carry
    return output


def apply_heads_from_features(
    *,
    functions: ModelFunctions,
    params: Mapping[str, Any],
    features: Any,
    slot_log_beliefs: Any,
) -> Mapping[str, Any]:
    return functions.heads.apply(
        {"params": params["heads"]},
        features,
        slot_log_beliefs,
        method=functions.heads.from_features,
    )


def slice_environment_lanes(
    batch: TransitionBatch, indexes: Any
) -> TransitionBatch:
    return TransitionBatch(
        **{
            name: (
                value[indexes]
                if name in {"initial_official_carry", "initial_control_carry"}
                else value[:, indexes]
            )
            for name, value in batch._asdict().items()
        }
    )


def _forward(
    *,
    raw_output: Mapping[str, Any],
    reference_logits: Any,
    slot_log_belief: Any,
    temperature: Any,
    generic_temperature: Any,
    gamma: float,
    uncertainty_penalty: float,
) -> Any:
    return model_forward(
        raw_output=raw_output,
        reference_logits=reference_logits,
        slot_log_belief=slot_log_belief,
        temperature=temperature,
        generic_temperature=generic_temperature,
        deployment_mode="posterior_use",
        gamma=gamma,
        uncertainty_penalty=uncertainty_penalty,
    )


def _branch_targets(
    *,
    functions: ModelFunctions,
    target_params: Mapping[str, Any],
    target_output: Mapping[str, Any],
    batch: TransitionBatch,
) -> tuple[Any, Any, Any]:
    """Exact next-Q vectors for updated and one-response-stale beliefs."""

    import jax
    import jax.numpy as jnp

    use_q = jnp.asarray(target_output["q_values"][1:])
    mask_raw = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_heads_from_features(
            functions=functions,
            params=target_params,
            features=target_output["features"][1:],
            slot_log_beliefs=batch.slot_log_beliefs[:-1],
        ),
    )
    mask_q = jnp.asarray(mask_raw["q_values"])
    done = jnp.asarray(batch.dones, dtype=jnp.bool_)
    use_q = jnp.where(done[..., None, None, None], 0.0, use_q)
    mask_q = jnp.where(done[..., None, None, None], 0.0, mask_q)
    next_reference = jax.nn.log_softmax(batch.reference_logits[1:], axis=-1)
    next_reference = next_reference - jnp.mean(
        next_reference, axis=-1, keepdims=True
    )
    next_reference = jnp.where(done[..., None], 0.0, next_reference)
    return use_q, mask_q, next_reference


def prepare_frozen_assignments(
    *,
    functions: ModelFunctions,
    target_params: Mapping[str, Any],
    batch: TransitionBatch,
    codebook_embeddings: Any,
    key: Any,
    temperature: Any,
    generic_temperature: Any,
    gamma: float,
    responsibility_temperature: float,
    bootstrap_probability: float,
    retrace_lambda: float,
    importance_ratio_clip: float,
    uncertainty_penalty: float,
    terminal_response: int,
    bootstrap_mask: Any | None = None,
) -> FrozenAssignments:
    """Run a target-network E-step for complete environment lanes."""

    import jax
    import jax.numpy as jnp

    target = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_model_sequence(functions=functions, params=target_params, batch=batch),
    )
    use_q_targets, mask_q_targets, reference_targets = _branch_targets(
        functions=functions,
        target_params=target_params,
        target_output=target,
        batch=batch,
    )
    all_forward = _forward(
        raw_output=target,
        reference_logits=batch.reference_logits,
        slot_log_belief=batch.slot_log_beliefs,
        temperature=temperature,
        generic_temperature=generic_temperature,
        gamma=gamma,
        uncertainty_penalty=uncertainty_penalty,
    )
    target_execution_probabilities = jax.nn.softmax(
        all_forward.execution_logits, axis=-1
    )
    one_step_targets = jax.lax.stop_gradient(
        bellman_targets(
            rewards=batch.rewards,
            dones=batch.dones,
            next_execution_probabilities=target_execution_probabilities[1:],
            target_next_q_values=target["q_values"][1:],
            gamma=gamma,
        )
    )
    targets, importance_ratios, trace_coefficients = retrace_targets(
        rewards=batch.rewards,
        dones=batch.dones,
        actions=batch.actions,
        behavior_logits=batch.behavior_logits,
        target_execution_probabilities=target_execution_probabilities,
        target_q_values=target["q_values"],
        gamma=gamma,
        trace_lambda=retrace_lambda,
        importance_ratio_clip=importance_ratio_clip,
    )

    slot_count = int(batch.slot_log_beliefs.shape[-1])
    uniform_current = uniform_slot_log_belief(
        target["features"][:-1].shape[:-1], slot_count
    )
    uniform_next = uniform_slot_log_belief(
        target["features"][1:].shape[:-1], slot_count
    )
    current_response_heads = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_heads_from_features(
            functions=functions,
            params=target_params,
            features=target["features"][:-1],
            slot_log_beliefs=uniform_current,
        ),
    )
    next_response_heads = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_heads_from_features(
            functions=functions,
            params=target_params,
            features=target["features"][1:],
            slot_log_beliefs=uniform_next,
        ),
    )
    signatures = jax.lax.stop_gradient(
        target_response_signatures(
            current_centered_advantages=current_response_heads[
                "centered_advantages"
            ],
            next_centered_advantages=next_response_heads[
                "centered_advantages"
            ],
        )
    )
    code_targets, unused_quantization_error = nearest_codes(
        signatures, codebook_embeddings
    )
    del unused_quantization_error
    code_targets = jax.lax.stop_gradient(
        jnp.where(batch.dones, int(terminal_response), code_targets)
    )

    stale_current = batch.slot_log_beliefs[:-2]
    stale_heads = apply_heads_from_features(
        functions=functions,
        params=target_params,
        features=target["features"][1:-1],
        slot_log_beliefs=stale_current,
    )
    stale_next_belief = slot_bayes_update(
        slot_log_belief=stale_current,
        slot_response_probabilities=stale_heads["response_probabilities"],
        action=batch.actions[1:],
        response_code=batch.response_codes[1:],
    )
    stale_next_heads = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_heads_from_features(
            functions=functions,
            params=target_params,
            features=target["features"][2:],
            slot_log_beliefs=stale_next_belief,
        ),
    )
    stale_forward = _forward(
        raw_output=stale_next_heads,
        reference_logits=batch.reference_logits[2:],
        slot_log_belief=stale_next_belief,
        temperature=temperature,
        generic_temperature=generic_temperature,
        gamma=gamma,
        uncertainty_penalty=uncertainty_penalty,
    )
    stale_targets = jax.lax.stop_gradient(
        bellman_targets(
            rewards=batch.rewards[1:],
            dones=batch.dones[1:],
            next_execution_probabilities=jax.nn.softmax(
                stale_forward.execution_logits, axis=-1
            ),
            target_next_q_values=stale_next_heads["q_values"],
            gamma=gamma,
        )
    )

    environment_count = int(batch.actions.shape[1])
    if bootstrap_mask is None:
        mask = sample_bootstrap_mask(
            key,
            environment_count=environment_count,
            slot_count=slot_count,
            probability=bootstrap_probability,
        )
    else:
        mask = jnp.asarray(bootstrap_mask, dtype=jnp.bool_)
        if mask.shape != (environment_count, slot_count):
            raise ValueError(
                "Bootstrap mask must be [environment,slot]."
            )

    responsibilities, td_energies = episode_responsibility_evidence(
        q_values=target["q_values"][:-1],
        actions=batch.actions,
        targets=targets,
        temperature=responsibility_temperature,
        availability_mask=mask,
    )
    audit_items = _selected_outcome_items(
        response_logits=target["response_logits"][:-1],
        reward_mean=target["reward_mean"][:-1],
        reward_log_standard_deviation=target[
            "reward_log_standard_deviation"
        ][:-1],
        next_q_use_mean=target["next_q_use_mean"][:-1],
        next_q_use_log_standard_deviation=target[
            "next_q_use_log_standard_deviation"
        ][:-1],
        next_q_mask_mean=target["next_q_mask_mean"][:-1],
        next_q_mask_log_standard_deviation=target[
            "next_q_mask_log_standard_deviation"
        ][:-1],
        next_reference_logits_mean=target[
            "next_reference_logits_mean"
        ][:-1],
        response_codes=batch.response_codes,
        rewards=batch.rewards,
        next_q_use_targets=use_q_targets,
        next_q_mask_targets=mask_q_targets,
        next_reference_logits_targets=reference_targets,
        actions=batch.actions,
    )
    return FrozenAssignments(
        responsibilities=responsibilities,
        responsibility_energies=td_energies,
        bootstrap_mask=mask,
        bellman_targets=targets,
        one_step_bellman_targets=one_step_targets,
        stale_bellman_targets=stale_targets,
        importance_ratio_mean=jnp.mean(importance_ratios, axis=0),
        trace_coefficient_mean=jnp.mean(trace_coefficients, axis=0),
        stale_current_beliefs=jax.lax.stop_gradient(stale_current),
        response_signature_targets=signatures,
        response_code_targets=code_targets,
        next_q_use_targets=jax.lax.stop_gradient(use_q_targets),
        next_q_mask_targets=jax.lax.stop_gradient(mask_q_targets),
        next_reference_logits_targets=jax.lax.stop_gradient(reference_targets),
        td_energies=td_energies,
        response_energies=jnp.sum(audit_items["response"], axis=0),
        reward_energies=jnp.sum(audit_items["reward"], axis=0),
        next_q_use_energies=jnp.sum(audit_items["next_q_use"], axis=0),
        next_q_mask_energies=jnp.sum(audit_items["next_q_mask"], axis=0),
        next_reference_energy=jnp.sum(
            audit_items["next_reference"], axis=0
        ),
    )


def slice_assignments(
    assignments: FrozenAssignments, indexes: Any
) -> FrozenAssignments:
    import jax

    sliced = FrozenAssignments(
        responsibilities=assignments.responsibilities[indexes],
        responsibility_energies=assignments.responsibility_energies[indexes],
        bootstrap_mask=assignments.bootstrap_mask[indexes],
        bellman_targets=assignments.bellman_targets[:, indexes],
        one_step_bellman_targets=(
            assignments.one_step_bellman_targets[:, indexes]
        ),
        stale_bellman_targets=assignments.stale_bellman_targets[:, indexes],
        importance_ratio_mean=assignments.importance_ratio_mean[indexes],
        trace_coefficient_mean=assignments.trace_coefficient_mean[indexes],
        stale_current_beliefs=assignments.stale_current_beliefs[:, indexes],
        response_signature_targets=assignments.response_signature_targets[:, indexes],
        response_code_targets=assignments.response_code_targets[:, indexes],
        next_q_use_targets=assignments.next_q_use_targets[:, indexes],
        next_q_mask_targets=assignments.next_q_mask_targets[:, indexes],
        next_reference_logits_targets=(
            assignments.next_reference_logits_targets[:, indexes]
        ),
        td_energies=assignments.td_energies[indexes],
        response_energies=assignments.response_energies[indexes],
        reward_energies=assignments.reward_energies[indexes],
        next_q_use_energies=assignments.next_q_use_energies[indexes],
        next_q_mask_energies=assignments.next_q_mask_energies[indexes],
        next_reference_energy=assignments.next_reference_energy[indexes],
    )
    return jax.tree_util.tree_map(jax.lax.stop_gradient, sliced)


def environment_minibatch_schedule(
    key: Any,
    *,
    environment_count: int,
    minibatches_per_epoch: int,
    update_epochs: int,
) -> Any:
    import jax
    import jax.numpy as jnp

    if environment_count % minibatches_per_epoch:
        raise ValueError("Environment lanes must divide exactly into minibatches.")
    lane_count = environment_count // minibatches_per_epoch
    keys = jax.random.split(key, update_epochs)
    return jnp.stack(
        [
            jax.random.permutation(epoch_key, environment_count).reshape(
                (minibatches_per_epoch, lane_count)
            )
            for epoch_key in keys
        ]
    )


def _losses(
    *,
    functions: ModelFunctions,
    params: Mapping[str, Any],
    batch: TransitionBatch,
    assignments: FrozenAssignments,
    codebook_embeddings: Any,
) -> tuple[LossBundle, LossBundle]:
    output = apply_model_sequence(functions=functions, params=params, batch=batch)
    actual = bellman_loss(
        q_values=output["q_values"][:-1],
        actions=batch.actions,
        targets=assignments.bellman_targets,
        stopped_responsibilities=assignments.responsibilities,
        bootstrap_mask=assignments.bootstrap_mask,
        metric_prefix="bellman",
    )
    stale_output = apply_heads_from_features(
        functions=functions,
        params=params,
        features=output["features"][1:-1],
        slot_log_beliefs=assignments.stale_current_beliefs,
    )
    stale = bellman_loss(
        q_values=stale_output["q_values"],
        actions=batch.actions[1:],
        targets=assignments.stale_bellman_targets,
        stopped_responsibilities=assignments.responsibilities,
        bootstrap_mask=assignments.bootstrap_mask,
        metric_prefix="stale_belief_bellman",
    )
    bellman = LossBundle(
        total=actual.total + stale.total,
        metrics={**actual.metrics, **stale.metrics},
    )
    outcomes = outcome_loss(
        response_logits=output["response_logits"][:-1],
        reward_mean=output["reward_mean"][:-1],
        reward_log_standard_deviation=output[
            "reward_log_standard_deviation"
        ][:-1],
        next_q_use_mean=output["next_q_use_mean"][:-1],
        next_q_use_log_standard_deviation=output[
            "next_q_use_log_standard_deviation"
        ][:-1],
        next_q_mask_mean=output["next_q_mask_mean"][:-1],
        next_q_mask_log_standard_deviation=output[
            "next_q_mask_log_standard_deviation"
        ][:-1],
        next_reference_logits_mean=output[
            "next_reference_logits_mean"
        ][:-1],
        response_codes=batch.response_codes,
        rewards=batch.rewards,
        next_q_use_targets=assignments.next_q_use_targets,
        next_q_mask_targets=assignments.next_q_mask_targets,
        next_reference_logits_targets=(
            assignments.next_reference_logits_targets
        ),
        actions=batch.actions,
        stopped_responsibilities=assignments.responsibilities,
        bootstrap_mask=assignments.bootstrap_mask,
    )
    predicted_signature = functions.heads.apply(
        {"params": params["heads"]},
        batch.observations[:-1],
        batch.actions,
        batch.response_next_observations,
        batch.dones,
        method=functions.heads.encode_response,
    )
    encoder = response_encoder_loss(
        predicted_signatures=predicted_signature,
        target_signatures=assignments.response_signature_targets,
        target_codes=assignments.response_code_targets,
        codebook_embeddings=codebook_embeddings,
    )
    return bellman, LossBundle(
        total=outcomes.total + encoder.total,
        metrics={**outcomes.metrics, **encoder.metrics},
    )


def apply_minibatch_update(
    *,
    functions: ModelFunctions,
    params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    optimizer_bundle: OptimizerBundle,
    batch: TransitionBatch,
    assignments: FrozenAssignments,
    codebook_embeddings: Any,
    polyak_coefficient: float,
) -> TrainingUpdate:
    import jax
    import optax

    def bellman_function(candidate: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        loss, unused = _losses(
            functions=functions,
            params=candidate,
            batch=batch,
            assignments=assignments,
            codebook_embeddings=codebook_embeddings,
        )
        del unused
        return loss.total, loss.metrics

    (unused_total, bellman_metrics), gradients = jax.value_and_grad(
        bellman_function, has_aux=True
    )(params)
    del unused_total
    updates, bellman_state = optimizer_bundle.bellman_optimizer.update(
        gradients, optimizer_bundle.bellman_state, params
    )
    params = optax.apply_updates(params, updates)

    def outcome_function(candidate: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        unused, loss = _losses(
            functions=functions,
            params=candidate,
            batch=batch,
            assignments=assignments,
            codebook_embeddings=codebook_embeddings,
        )
        del unused
        return loss.total, loss.metrics

    (unused_total, outcome_metrics), gradients = jax.value_and_grad(
        outcome_function, has_aux=True
    )(params)
    del unused_total
    updates, outcome_state = optimizer_bundle.outcome_optimizer.update(
        gradients, optimizer_bundle.outcome_state, params
    )
    params = optax.apply_updates(params, updates)
    return TrainingUpdate(
        params=params,
        target_params=polyak_update(
            target_params, params, polyak_coefficient
        ),
        bellman_optimizer_state=bellman_state,
        outcome_optimizer_state=outcome_state,
        metrics={**bellman_metrics, **outcome_metrics},
    )


def apply_rollout_updates(
    *,
    functions: ModelFunctions,
    params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    optimizer_bundle: OptimizerBundle,
    batch: TransitionBatch,
    assignments: FrozenAssignments,
    codebook_embeddings: Any,
    schedule: Any,
    polyak_coefficient: float,
) -> TrainingUpdate:
    import jax
    import jax.numpy as jnp

    flat_schedule = jnp.asarray(schedule).reshape((-1, schedule.shape[-1]))
    initial = (
        params,
        target_params,
        optimizer_bundle.bellman_state,
        optimizer_bundle.outcome_state,
    )

    def update_one(carry: tuple[Any, ...], indexes: Any) -> tuple[Any, Any]:
        current_params, current_target, bellman_state, outcome_state = carry
        current_bundle = OptimizerBundle(
            bellman_optimizer=optimizer_bundle.bellman_optimizer,
            outcome_optimizer=optimizer_bundle.outcome_optimizer,
            bellman_state=bellman_state,
            outcome_state=outcome_state,
        )
        updated = apply_minibatch_update(
            functions=functions,
            params=current_params,
            target_params=current_target,
            optimizer_bundle=current_bundle,
            batch=slice_environment_lanes(batch, indexes),
            assignments=slice_assignments(assignments, indexes),
            codebook_embeddings=codebook_embeddings,
            polyak_coefficient=polyak_coefficient,
        )
        return (
            updated.params,
            updated.target_params,
            updated.bellman_optimizer_state,
            updated.outcome_optimizer_state,
        ), updated.metrics

    final, metrics = jax.lax.scan(update_one, initial, flat_schedule)
    return TrainingUpdate(
        params=final[0],
        target_params=final[1],
        bellman_optimizer_state=final[2],
        outcome_optimizer_state=final[3],
        metrics=jax.tree_util.tree_map(lambda value: jnp.mean(value), metrics),
    )


def polyak_update(target_params: Any, online_params: Any, coefficient: float) -> Any:
    import jax

    tau = float(coefficient)
    return jax.tree_util.tree_map(
        lambda target, online: (1.0 - tau) * target + tau * online,
        target_params,
        online_params,
    )


def recompute_policy_scores(
    *,
    functions: ModelFunctions,
    params: Mapping[str, Any],
    batch: TransitionBatch,
    temperature: Any,
    generic_temperature: Any,
    gamma: float,
    uncertainty_penalty: float,
) -> tuple[Any, Any]:
    """Recompute current online scores after the rollout's parameter updates."""

    output = apply_model_sequence(functions=functions, params=params, batch=batch)
    forward = model_forward(
        raw_output={name: value[:-1] for name, value in output.items()},
        reference_logits=batch.reference_logits[:-1],
        slot_log_belief=batch.slot_log_beliefs[:-1],
        temperature=temperature,
        generic_temperature=generic_temperature,
        deployment_mode="posterior_use",
        gamma=gamma,
        uncertainty_penalty=uncertainty_penalty,
    )
    return forward.j_use, forward.information_gain


def rollout_kl_means(batch: TransitionBatch) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    reference = jax.nn.log_softmax(batch.reference_logits[:-1], axis=-1)

    def mean_kl(logits: Any) -> Any:
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        probabilities = jnp.exp(log_probabilities)
        return jnp.mean(
            jnp.sum(
                probabilities * (log_probabilities - reference), axis=-1
            )
        )

    return mean_kl(batch.execution_logits), mean_kl(
        batch.generic_execution_logits
    )


def solve_policy_temperatures(
    policy_state: Any,
    *,
    batch: TransitionBatch,
    target_kl: float,
    minimum_temperature: float,
    maximum_temperature: float,
    iterations: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Replace the ineffective dual step with exact batchwise KL solves."""

    import jax.numpy as jnp

    posterior_temperature, posterior_kl = solve_temperature_for_target_kl(
        reference_logits=batch.reference_logits[:-1],
        score=batch.posterior_scores,
        target_kl=target_kl,
        minimum_temperature=minimum_temperature,
        maximum_temperature=maximum_temperature,
        iterations=iterations,
    )
    generic_temperature, generic_kl = solve_temperature_for_target_kl(
        reference_logits=batch.reference_logits[:-1],
        score=batch.generic_scores,
        target_kl=target_kl,
        minimum_temperature=minimum_temperature,
        maximum_temperature=maximum_temperature,
        iterations=iterations,
    )
    state = policy_state._replace(
        log_temperature=jnp.full_like(
            policy_state.log_temperature, jnp.log(posterior_temperature)
        ),
        generic_log_temperature=jnp.full_like(
            policy_state.generic_log_temperature, jnp.log(generic_temperature)
        ),
    )
    return state, {
        "temperature": posterior_temperature,
        "generic_temperature": generic_temperature,
        "posterior_solved_kl": posterior_kl,
        "generic_solved_kl": generic_kl,
    }


def update_codebook_from_assignments(
    codebook: CodebookState,
    *,
    assignments: FrozenAssignments,
    dones: Any,
    key: Any,
    decay: float,
    replacement_after_rollouts: int,
) -> CodebookState:
    return update_codebook(
        codebook,
        signatures=assignments.response_signature_targets,
        valid_mask=~dones,
        key=key,
        decay=decay,
        replacement_after_rollouts=replacement_after_rollouts,
    ).state


__all__ = [
    "FrozenAssignments",
    "LossBundle",
    "ModelFunctions",
    "OptimizerBundle",
    "TrainState",
    "TrainingUpdate",
    "TransitionBatch",
    "apply_minibatch_update",
    "apply_model_sequence",
    "apply_rollout_updates",
    "bellman_loss",
    "bellman_targets",
    "environment_minibatch_schedule",
    "episode_responsibility_evidence",
    "gaussian_negative_log_likelihood",
    "gather_actions",
    "huber",
    "make_optimizers",
    "outcome_loss",
    "polyak_update",
    "prepare_frozen_assignments",
    "recompute_policy_scores",
    "retrace_targets",
    "response_encoder_loss",
    "rollout_kl_means",
    "sample_bootstrap_mask",
    "solve_policy_temperatures",
    "update_codebook_from_assignments",
]
