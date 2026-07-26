"""Path C V4.2 transition batches, losses, and parameter updates."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .method import (
    CodebookState,
    nearest_codes,
    slot_bayes_update,
    target_response_signatures,
    uniform_slot_log_belief,
    update_codebook,
    update_log_temperature,
)
from .model import model_forward


class ModelFunctions(NamedTuple):
    """Public official forward function plus the Path C heads."""

    heads: Any
    official_sequence: Callable[..., tuple[Any, Any, Any, Any]]


class TransitionBatch(NamedTuple):
    """All tensors consumed by the losses; partner identity is absent."""

    observations: Any
    response_next_observations: Any
    episode_start: Any
    previous_actions: Any
    previous_team_rewards: Any
    initial_official_carry: Any
    reference_logits: Any
    execution_logits: Any
    generic_execution_logits: Any
    slot_log_beliefs: Any
    actions: Any
    rewards: Any
    dones: Any
    response_codes: Any


class TrainState(NamedTuple):
    """All mutable training state, including the runner state."""

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
    stale_bellman_targets: Any
    stale_current_beliefs: Any
    response_signature_targets: Any
    response_code_targets: Any
    continuation_use_targets: Any
    continuation_mask_targets: Any


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
    return 0.5 * jnp.square(quadratic) + float(delta) * (absolute - quadratic)


def gaussian_negative_log_likelihood(
    value: Any, mean: Any, log_standard_deviation: Any
) -> Any:
    import jax.numpy as jnp

    log_std = jnp.asarray(log_standard_deviation)
    normalized = (jnp.asarray(value) - jnp.asarray(mean)) * jnp.exp(-log_std)
    return 0.5 * jnp.square(normalized) + log_std


def raw_policy_continuation_targets(
    *, execution_probabilities: Any, target_q_values: Any, dones: Any
) -> Any:
    """Raw expected Q under the exact target execution distribution."""

    import jax.numpy as jnp

    probabilities = jnp.asarray(execution_probabilities, dtype=jnp.float32)
    q_values = jnp.asarray(target_q_values, dtype=jnp.float32)
    if q_values.shape[-3] != 2:
        raise ValueError("Continuation targets require two Q estimators.")
    if q_values.shape[:-3] != probabilities.shape[:-1]:
        raise ValueError("Policy and target Q prefix axes differ.")
    if q_values.shape[-1] != probabilities.shape[-1]:
        raise ValueError("Policy and target Q action axes differ.")
    expected = jnp.einsum("...a,...ema->...em", probabilities, q_values)
    return jnp.where(
        jnp.asarray(dones, dtype=jnp.bool_)[..., None, None],
        jnp.zeros_like(expected),
        expected,
    )


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
        "...a,...ma->...m", jnp.asarray(next_execution_probabilities), target_q
    )
    return jnp.asarray(rewards)[..., None] + float(gamma) * (
        1.0 - jnp.asarray(dones, dtype=jnp.float32)
    )[..., None] * expected


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
    continuation_use_mean: Any,
    continuation_use_log_standard_deviation: Any,
    continuation_mask_mean: Any,
    continuation_mask_log_standard_deviation: Any,
    response_codes: Any,
    rewards: Any,
    continuation_use_targets: Any,
    continuation_mask_targets: Any,
    actions: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Joint per-transition control evidence with shape [T,B,slot]."""

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

    def continuation_items(mean: Any, log_std: Any, target: Any) -> Any:
        selected_action_mean = gather_actions(mean, actions, action_axis=-2)
        selected_action_std = gather_actions(log_std, actions, action_axis=-2)
        selected_mean = gather_actions(
            selected_action_mean, response_codes, action_axis=-1
        )
        selected_std = gather_actions(
            selected_action_std, response_codes, action_axis=-1
        )
        return jnp.mean(
            gaussian_negative_log_likelihood(target, selected_mean, selected_std),
            axis=-2,
        )

    use_nll = continuation_items(
        continuation_use_mean,
        continuation_use_log_standard_deviation,
        continuation_use_targets,
    )
    mask_nll = continuation_items(
        continuation_mask_mean,
        continuation_mask_log_standard_deviation,
        continuation_mask_targets,
    )
    return response_nll + reward_nll + use_nll + mask_nll, {
        "response_nll_items": response_nll,
        "reward_nll_items": reward_nll,
        "continuation_use_nll_items": use_nll,
        "continuation_mask_nll_items": mask_nll,
    }


def episode_responsibility_evidence(
    *,
    q_values: Any,
    actions: Any,
    targets: Any,
    temperature: float,
    response_logits: Any | None = None,
    reward_mean: Any | None = None,
    reward_log_standard_deviation: Any | None = None,
    continuation_use_mean: Any | None = None,
    continuation_use_log_standard_deviation: Any | None = None,
    continuation_mask_mean: Any | None = None,
    continuation_mask_log_standard_deviation: Any | None = None,
    response_codes: Any | None = None,
    rewards: Any | None = None,
    continuation_use_targets: Any | None = None,
    continuation_mask_targets: Any | None = None,
    availability_mask: Any | None = None,
) -> tuple[Any, Any]:
    """Infer stopped slot responsibilities from full-episode evidence."""

    import jax
    import jax.numpy as jnp

    evidence_items = _td_evidence_items(
        q_values=q_values, actions=actions, targets=targets
    )
    optional = (
        response_logits,
        reward_mean,
        reward_log_standard_deviation,
        continuation_use_mean,
        continuation_use_log_standard_deviation,
        continuation_mask_mean,
        continuation_mask_log_standard_deviation,
        response_codes,
        rewards,
        continuation_use_targets,
        continuation_mask_targets,
    )
    if any(value is not None for value in optional):
        if any(value is None for value in optional):
            raise ValueError(
                "Joint responsibility evidence requires every outcome tensor."
            )
        outcome_items, unused = _selected_outcome_items(
            response_logits=response_logits,
            reward_mean=reward_mean,
            reward_log_standard_deviation=reward_log_standard_deviation,
            continuation_use_mean=continuation_use_mean,
            continuation_use_log_standard_deviation=(
                continuation_use_log_standard_deviation
            ),
            continuation_mask_mean=continuation_mask_mean,
            continuation_mask_log_standard_deviation=(
                continuation_mask_log_standard_deviation
            ),
            response_codes=response_codes,
            rewards=rewards,
            continuation_use_targets=continuation_use_targets,
            continuation_mask_targets=continuation_mask_targets,
            actions=actions,
        )
        del unused
        evidence_items = evidence_items + outcome_items
    energies = jnp.sum(evidence_items, axis=0)
    available = (
        jnp.ones_like(energies, dtype=jnp.bool_)
        if availability_mask is None
        else jnp.asarray(availability_mask, dtype=jnp.bool_)
    )
    if available.shape != energies.shape:
        raise ValueError("Responsibility availability must be [environment, slot].")
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
    """Sample independent estimator membership conditioned on a nonempty row."""

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
    continuation_use_mean: Any,
    continuation_use_log_standard_deviation: Any,
    continuation_mask_mean: Any,
    continuation_mask_log_standard_deviation: Any,
    response_codes: Any,
    rewards: Any,
    continuation_use_targets: Any,
    continuation_mask_targets: Any,
    actions: Any,
    stopped_responsibilities: Any,
    bootstrap_mask: Any | None = None,
    terminal_response: int = 15,
) -> LossBundle:
    import jax
    import jax.numpy as jnp

    terms, parts = _selected_outcome_items(
        response_logits=response_logits,
        reward_mean=reward_mean,
        reward_log_standard_deviation=reward_log_standard_deviation,
        continuation_use_mean=continuation_use_mean,
        continuation_use_log_standard_deviation=(
            continuation_use_log_standard_deviation
        ),
        continuation_mask_mean=continuation_mask_mean,
        continuation_mask_log_standard_deviation=(
            continuation_mask_log_standard_deviation
        ),
        response_codes=response_codes,
        rewards=rewards,
        continuation_use_targets=continuation_use_targets,
        continuation_mask_targets=continuation_mask_targets,
        actions=actions,
    )
    weights = jnp.asarray(stopped_responsibilities)
    if bootstrap_mask is not None:
        weights = weights * jnp.asarray(bootstrap_mask, dtype=jnp.float32)
    weights = jax.lax.stop_gradient(weights)[None, ...]
    denominator = response_logits.shape[0] * jnp.sum(weights) + 1.0e-8
    fitted = jnp.sum(terms * weights) / denominator
    terminal_use = continuation_use_mean[..., int(terminal_response)]
    terminal_mask = continuation_mask_mean[..., int(terminal_response)]
    terminal_penalty = 0.5 * (
        jnp.mean(jnp.square(terminal_use))
        + jnp.mean(jnp.square(terminal_mask))
    )
    return LossBundle(
        total=fitted + terminal_penalty,
        metrics={
            "slot_outcome": fitted,
            "response_nll": jnp.sum(parts["response_nll_items"] * weights)
            / denominator,
            "reward_nll": jnp.sum(parts["reward_nll_items"] * weights)
            / denominator,
            "continuation_use_nll": jnp.sum(
                parts["continuation_use_nll_items"] * weights
            )
            / denominator,
            "continuation_mask_nll": jnp.sum(
                parts["continuation_mask_nll_items"] * weights
            )
            / denominator,
            "terminal_continuation_penalty": terminal_penalty,
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
        jax.nn.log_softmax(logits, axis=-1), safe_targets[..., None], axis=-1
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
    """Partition by mathematical consumer, not by checkpoint provenance."""

    import jax

    if loss == "bellman":
        selected = {
            "previous_action_embedding",
            "previous_action_to_hidden",
            "previous_reward_to_hidden",
        }
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
    """Train the official recurrent representation only through Bellman loss."""

    import jax
    import optax

    bellman_labels = {
        "official": jax.tree_util.tree_map(lambda unused: "train", params["official"]),
        "heads": _head_labels(params["heads"], loss="bellman"),
    }
    outcome_labels = {
        "official": jax.tree_util.tree_map(lambda unused: "frozen", params["official"]),
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
    unused_final_carry, features, unused_logits, unused_values = (
        functions.official_sequence(
            params["official"],
            batch.initial_official_carry,
            batch.observations,
            batch.episode_start,
        )
    )
    del unused_final_carry, unused_logits, unused_values
    return functions.heads.apply(
        {"params": params["heads"]},
        features,
        batch.previous_actions,
        batch.previous_team_rewards,
        batch.slot_log_beliefs,
    )


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
                if name == "initial_official_carry"
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
) -> Any:
    return model_forward(
        raw_output=raw_output,
        reference_logits=reference_logits,
        slot_log_belief=slot_log_belief,
        temperature=temperature,
        generic_temperature=generic_temperature,
        deployment_mode="posterior_use",
        gamma=gamma,
    )


def _branch_targets(
    *,
    functions: ModelFunctions,
    target_params: Mapping[str, Any],
    target_output: Mapping[str, Any],
    batch: TransitionBatch,
    temperature: Any,
    generic_temperature: Any,
    gamma: float,
) -> tuple[Any, Any]:
    """Compute use and response-mask targets from the same physical transition."""

    import jax
    import jax.numpy as jnp

    use_raw = {name: value[1:] for name, value in target_output.items()}
    use_forward = _forward(
        raw_output=use_raw,
        reference_logits=batch.reference_logits[1:],
        slot_log_belief=batch.slot_log_beliefs[1:],
        temperature=temperature,
        generic_temperature=generic_temperature,
        gamma=gamma,
    )
    mask_raw = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_heads_from_features(
            functions=functions,
            params=target_params,
            features=target_output["features"][1:],
            slot_log_beliefs=batch.slot_log_beliefs[:-1],
        ),
    )
    mask_forward = _forward(
        raw_output=mask_raw,
        reference_logits=batch.reference_logits[1:],
        slot_log_belief=batch.slot_log_beliefs[:-1],
        temperature=temperature,
        generic_temperature=generic_temperature,
        gamma=gamma,
    )
    use_probabilities = jax.nn.softmax(use_forward.execution_logits, axis=-1)
    mask_probabilities = jax.nn.softmax(mask_forward.execution_logits, axis=-1)
    use_targets = raw_policy_continuation_targets(
        execution_probabilities=use_probabilities,
        target_q_values=use_raw["q_values"],
        dones=batch.dones,
    )
    mask_targets = raw_policy_continuation_targets(
        execution_probabilities=mask_probabilities,
        target_q_values=mask_raw["q_values"],
        dones=batch.dones,
    )
    return jnp.asarray(use_targets), jnp.asarray(mask_targets)


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
    terminal_response: int,
    bootstrap_mask: Any | None = None,
) -> FrozenAssignments:
    """Infer one stopped responsibility vector per complete environment lane."""

    import jax
    import jax.numpy as jnp

    target = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_model_sequence(functions=functions, params=target_params, batch=batch),
    )
    use_targets, mask_targets = _branch_targets(
        functions=functions,
        target_params=target_params,
        target_output=target,
        batch=batch,
        temperature=temperature,
        generic_temperature=generic_temperature,
        gamma=gamma,
    )
    next_forward = _forward(
        raw_output={name: value[1:] for name, value in target.items()},
        reference_logits=batch.reference_logits[1:],
        slot_log_belief=batch.slot_log_beliefs[1:],
        temperature=temperature,
        generic_temperature=generic_temperature,
        gamma=gamma,
    )
    next_probabilities = jax.nn.softmax(
        next_forward.execution_logits, axis=-1
    )
    targets = jax.lax.stop_gradient(
        bellman_targets(
            rewards=batch.rewards,
            dones=batch.dones,
            next_execution_probabilities=next_probabilities,
            target_next_q_values=target["q_values"][1:],
            gamma=gamma,
        )
    )

    slot_count = int(batch.slot_log_beliefs.shape[-1])
    uniform_belief_for_response_targets = uniform_slot_log_belief(
        target["features"][1:].shape[:-1], slot_count
    )
    response_target_heads = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        apply_heads_from_features(
            functions=functions,
            params=target_params,
            features=target["features"][1:],
            slot_log_beliefs=uniform_belief_for_response_targets,
        ),
    )
    signatures = jax.lax.stop_gradient(
        target_response_signatures(
            target_centered_advantages=response_target_heads[
                "centered_advantages"
            ]
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
                "Bootstrap mask must have one row per environment lane "
                "and one column per partner slot."
            )
    responsibilities, energies = episode_responsibility_evidence(
        q_values=target["q_values"][:-1],
        actions=batch.actions,
        targets=targets,
        temperature=responsibility_temperature,
        response_logits=target["response_logits"][:-1],
        reward_mean=target["reward_mean"][:-1],
        reward_log_standard_deviation=target[
            "reward_log_standard_deviation"
        ][:-1],
        continuation_use_mean=target["continuation_use_mean"][:-1],
        continuation_use_log_standard_deviation=target[
            "continuation_use_log_standard_deviation"
        ][:-1],
        continuation_mask_mean=target["continuation_mask_mean"][:-1],
        continuation_mask_log_standard_deviation=target[
            "continuation_mask_log_standard_deviation"
        ][:-1],
        response_codes=batch.response_codes,
        rewards=batch.rewards,
        continuation_use_targets=use_targets,
        continuation_mask_targets=mask_targets,
        availability_mask=mask,
    )
    return FrozenAssignments(
        responsibilities=responsibilities,
        responsibility_energies=energies,
        bootstrap_mask=mask,
        bellman_targets=targets,
        stale_bellman_targets=stale_targets,
        stale_current_beliefs=jax.lax.stop_gradient(stale_current),
        response_signature_targets=signatures,
        response_code_targets=code_targets,
        continuation_use_targets=jax.lax.stop_gradient(use_targets),
        continuation_mask_targets=jax.lax.stop_gradient(mask_targets),
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
        stale_bellman_targets=assignments.stale_bellman_targets[:, indexes],
        stale_current_beliefs=assignments.stale_current_beliefs[:, indexes],
        response_signature_targets=assignments.response_signature_targets[:, indexes],
        response_code_targets=assignments.response_code_targets[:, indexes],
        continuation_use_targets=assignments.continuation_use_targets[:, indexes],
        continuation_mask_targets=assignments.continuation_mask_targets[:, indexes],
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
        continuation_use_mean=output["continuation_use_mean"][:-1],
        continuation_use_log_standard_deviation=output[
            "continuation_use_log_standard_deviation"
        ][:-1],
        continuation_mask_mean=output["continuation_mask_mean"][:-1],
        continuation_mask_log_standard_deviation=output[
            "continuation_mask_log_standard_deviation"
        ][:-1],
        response_codes=batch.response_codes,
        rewards=batch.rewards,
        continuation_use_targets=assignments.continuation_use_targets,
        continuation_mask_targets=assignments.continuation_mask_targets,
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


def rollout_kl_means(batch: TransitionBatch) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    reference = jax.nn.log_softmax(batch.reference_logits[:-1], axis=-1)

    def mean_kl(logits: Any) -> Any:
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        probabilities = jnp.exp(log_probabilities)
        return jnp.mean(
            jnp.sum(probabilities * (log_probabilities - reference), axis=-1)
        )

    return mean_kl(batch.execution_logits), mean_kl(
        batch.generic_execution_logits
    )


def update_policy_temperatures(
    policy_state: Any,
    *,
    posterior_mean_kl: Any,
    generic_mean_kl: Any,
    target_kl: float,
    learning_rate: float,
    minimum_temperature: float,
    maximum_temperature: float,
) -> Any:
    import jax.numpy as jnp

    posterior_log_temperature = update_log_temperature(
        log_temperature=jnp.ravel(policy_state.log_temperature)[0],
        mean_kl=posterior_mean_kl,
        target_kl=target_kl,
        learning_rate=learning_rate,
        minimum_temperature=minimum_temperature,
        maximum_temperature=maximum_temperature,
    )
    generic_log_temperature = update_log_temperature(
        log_temperature=jnp.ravel(policy_state.generic_log_temperature)[0],
        mean_kl=generic_mean_kl,
        target_kl=target_kl,
        learning_rate=learning_rate,
        minimum_temperature=minimum_temperature,
        maximum_temperature=maximum_temperature,
    )
    return policy_state._replace(
        log_temperature=jnp.full_like(
            policy_state.log_temperature, posterior_log_temperature
        ),
        generic_log_temperature=jnp.full_like(
            policy_state.generic_log_temperature, generic_log_temperature
        ),
    )


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
    "make_optimizers",
    "outcome_loss",
    "polyak_update",
    "prepare_frozen_assignments",
    "raw_policy_continuation_targets",
    "response_encoder_loss",
    "rollout_kl_means",
    "sample_bootstrap_mask",
    "update_codebook_from_assignments",
    "update_policy_temperatures",
]
