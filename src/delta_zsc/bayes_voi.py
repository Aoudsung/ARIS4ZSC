"""Rao-Blackwellized low-discrepancy Bayesian value-of-information.

For each candidate ego action ``a`` and predictive source component ``k`` this
module integrates a complete teammate response under the learned factorized
emission.  Binary visibility is summed exactly.  Conditional on visibility,
position, direction, and inventory factors are integrated with a deterministic
multidimensional Halton rule; inventory-change is then summed exactly whenever
it is observable.  Every response outcome is scored under *all* latent
components and followed by an exact categorical Bayes update.

The resulting decision value is

    E_y[max_a' E_{z|y,a} Q_z(a')] - max_a' E_z Q_z(a'),

not an entropy bonus, expected cross-log-likelihood proxy, or learned probing
critic.  The Halton sample count controls numerical resolution only and never
rescales task reward.
"""

from __future__ import annotations

from typing import Any

from .transition import transition_matrix
from .types import VOIResult


def _first_primes(count: int) -> tuple[int, ...]:
    """Return the first ``count`` primes without an optional dependency."""

    result: list[int] = []
    candidate = 2
    while len(result) < int(count):
        prime = True
        divisor = 2
        while divisor * divisor <= candidate:
            if candidate % divisor == 0:
                prime = False
                break
            divisor += 1
        if prime:
            result.append(candidate)
        candidate += 1
    return tuple(result)


def _radical_inverse(index: int, base: int) -> float:
    inverse = 1.0 / float(base)
    factor = inverse
    value = 0.0
    current = int(index)
    while current:
        current, digit = divmod(current, int(base))
        value += float(digit) * factor
        factor *= inverse
    return value


def halton_points(sample_count: int, dimensions: int) -> Any:
    """Return a deterministic ``[sample, dimension]`` Halton design.

    The array is constructed at Python trace time and embedded as a JAX
    constant, so it remains compatible with ``jax.jit`` when the sizes are
    static.  Index zero is skipped because it is the all-zero corner.
    """

    import jax.numpy as jnp
    import numpy as np

    samples = int(sample_count)
    dims = int(dimensions)
    if samples <= 0 or dims <= 0:
        raise ValueError("Halton sample count and dimension must be positive.")
    bases = _first_primes(dims)
    values = np.empty((samples, dims), dtype=np.float32)
    for row in range(samples):
        for column, base in enumerate(bases):
            values[row, column] = _radical_inverse(row + 1, base)
    return jnp.clip(jnp.asarray(values), 1.0e-7, 1.0 - 1.0e-7)


def _sample_categorical(probability: Any, uniforms: Any) -> Any:
    """Inverse-CDF labels with source-component and sample axes retained."""

    import jax.numpy as jnp

    p = jnp.asarray(probability, dtype=jnp.float32)
    u = jnp.asarray(uniforms, dtype=jnp.float32)
    if u.ndim != 1:
        raise ValueError("Categorical Halton coordinate must be one-dimensional.")
    cdf = jnp.cumsum(p, axis=-1)
    shaped = u.reshape((1,) * (p.ndim - 1) + (u.shape[0], 1))
    return jnp.sum(shaped >= cdf[..., None, :], axis=-1).astype(jnp.int32)


def _sample_factor_categorical(probability: Any, uniforms: Any) -> Any:
    """Sample ``[..., source, sample, factor]`` categorical labels."""

    import jax.numpy as jnp

    p = jnp.asarray(probability, dtype=jnp.float32)
    u = jnp.asarray(uniforms, dtype=jnp.float32)
    if u.ndim != 2 or u.shape[1] != p.shape[-2]:
        raise ValueError("Inventory Halton coordinates do not match factor count.")
    cdf = jnp.cumsum(p, axis=-1)[..., None, :, :]
    shaped = u.reshape((1,) * (p.ndim - 2) + u.shape + (1,))
    return jnp.sum(shaped >= cdf, axis=-1).astype(jnp.int32)


def _bernoulli_outcome_log_probability(logits: Any, outcome: float) -> Any:
    """Log probability of one fixed Bernoulli outcome under all components."""

    import jax.nn as jnn

    return jnn.log_sigmoid(logits if float(outcome) == 1.0 else -logits)


def _categorical_sample_log_probability(logits: Any, samples: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    one_hot = jnn.one_hot(jnp.asarray(samples, dtype=jnp.int32), logits.shape[-1])
    return jnp.einsum("...ksc,...jc->...ksj", one_hot, logp)


def _factor_sample_log_probability(logits: Any, samples: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    one_hot = jnn.one_hot(jnp.asarray(samples, dtype=jnp.int32), logits.shape[-1])
    return jnp.einsum("...ksfc,...jfc->...ksj", one_hot, logp)


def _validate_inputs(
    belief: Any,
    prediction: Any,
    decision_means: Any,
) -> tuple[tuple[int, ...], int, int, int]:
    import jax.numpy as jnp

    b = jnp.asarray(belief)
    visibility = jnp.asarray(prediction.visibility_logit)
    means = jnp.asarray(decision_means)
    if b.ndim < 1 or visibility.ndim < 2:
        raise ValueError("VOI belief/prediction ranks are too small.")
    lead = tuple(b.shape[:-1])
    components = int(b.shape[-1])
    if tuple(visibility.shape[:-2]) != lead or int(visibility.shape[-1]) != components:
        raise ValueError("VOI response component axes differ from belief.")
    probes = int(visibility.shape[-2])
    if means.shape[-2] != components:
        raise ValueError("VOI decision component axis differs from belief.")
    shared_shape = lead + (components, means.shape[-1])
    probe_shape = lead + (probes, components, means.shape[-1])
    if tuple(means.shape) not in (shared_shape, probe_shape):
        raise ValueError("Decision means must be shared or probe-conditioned.")
    return lead, components, probes, int(means.shape[-1])


def _posterior_value_and_entropy(
    log_prior: Any,
    component_log_probability: Any,
    decision_means: Any,
    *,
    shared_decision_matrix: bool,
) -> tuple[Any, Any]:
    """Return posterior-optimal value and entropy for each source/sample row."""

    import jax.nn as jnn
    import jax.numpy as jnp

    posterior = jnn.softmax(
        jnp.asarray(log_prior)[..., None, None, None, :]
        + jnp.asarray(component_log_probability),
        axis=-1,
    )
    means = jnp.asarray(decision_means, dtype=jnp.float32)
    if shared_decision_matrix:
        posterior_action_values = jnp.einsum(
            "...pksj,...ja->...pksa", posterior, means
        )
    else:
        posterior_action_values = jnp.einsum(
            "...pksj,...pja->...pksa", posterior, means
        )
    value = jnp.max(posterior_action_values, axis=-1)
    entropy = -jnp.sum(
        posterior * jnp.log(jnp.maximum(posterior, 1.0e-30)), axis=-1
    )
    return value, entropy


def myopic_value_of_information_details(
    belief: Any,
    transition_logits: Any,
    response_prediction_by_action: Any,
    decision_means: Any,
    *,
    previous_visibility: Any,
    sample_count: int,
) -> VOIResult:
    """Integrate one-response Bayes decision value for every probe action.

    ``decision_means`` may be ``[..., K, A]`` (the registered local-stationarity
    surrogate) or ``[..., P, K, A]`` when an environment-specific model supplies
    probe-conditioned future utilities.  The standard path does not claim to
    solve long-horizon Bayes-adaptive planning; it measures whether the next
    legal response can improve the immediately available latent-conditioned
    action ordering.
    """

    import jax.nn as jnn
    import jax.numpy as jnp

    samples = int(sample_count)
    if samples < 2 or samples % 2:
        raise ValueError("VOI quadrature requires an even sample count of at least two.")

    lead, component_count, probe_count, _ = _validate_inputs(
        belief, response_prediction_by_action, decision_means
    )
    probability = jnp.asarray(belief, dtype=jnp.float32)
    probability = probability / jnp.maximum(
        jnp.sum(probability, axis=-1, keepdims=True), 1.0e-12
    )
    predictive = probability @ transition_matrix(transition_logits)
    log_prior = jnp.log(jnp.maximum(predictive, 1.0e-30))
    prediction = response_prediction_by_action
    means = jnp.asarray(decision_means, dtype=jnp.float32)
    shared_decision_matrix = means.ndim == probability.ndim + 1

    # Binary visibility is integrated exactly.  Only the visible branch needs
    # low-discrepancy integration over position/direction/inventory factors.
    source_visibility = jnn.sigmoid(prediction.visibility_logit)
    invisible_logp = _bernoulli_outcome_log_probability(
        prediction.visibility_logit, 0.0
    )[..., None, None, :]
    invisible_logp = jnp.broadcast_to(
        invisible_logp,
        lead + (probe_count, component_count, 1, component_count),
    )
    invisible_value, invisible_entropy = _posterior_value_and_entropy(
        log_prior,
        invisible_logp,
        means,
        shared_decision_matrix=shared_decision_matrix,
    )
    invisible_value = invisible_value[..., 0]
    invisible_entropy = invisible_entropy[..., 0]

    inventory_factors = int(prediction.inventory_logits.shape[-2])
    points = halton_points(samples, 2 + inventory_factors)
    position = _sample_categorical(
        jnn.softmax(prediction.relative_position_logits, axis=-1), points[:, 0]
    )
    direction = _sample_categorical(
        jnn.softmax(prediction.direction_logits, axis=-1), points[:, 1]
    )
    inventory = _sample_factor_categorical(
        jnn.softmax(prediction.inventory_logits, axis=-1),
        points[:, 2 : 2 + inventory_factors],
    )
    visible_logp = _bernoulli_outcome_log_probability(
        prediction.visibility_logit, 1.0
    )[..., None, None, :]
    visible_logp = visible_logp + (
        _categorical_sample_log_probability(
            prediction.relative_position_logits, position
        )
        + _categorical_sample_log_probability(
            prediction.direction_logits, direction
        )
        + _factor_sample_log_probability(prediction.inventory_logits, inventory)
    )

    # Inventory-change is a legal factor only when the partner was visible at
    # both ends.  In that branch it is summed exactly rather than sampled.
    event_zero_logp = visible_logp + _bernoulli_outcome_log_probability(
        prediction.inventory_change_logit, 0.0
    )[..., None, None, :]
    event_one_logp = visible_logp + _bernoulli_outcome_log_probability(
        prediction.inventory_change_logit, 1.0
    )[..., None, None, :]
    base_visible_value, base_visible_entropy = _posterior_value_and_entropy(
        log_prior,
        visible_logp,
        means,
        shared_decision_matrix=shared_decision_matrix,
    )
    event_zero_value, event_zero_entropy = _posterior_value_and_entropy(
        log_prior,
        event_zero_logp,
        means,
        shared_decision_matrix=shared_decision_matrix,
    )
    event_one_value, event_one_entropy = _posterior_value_and_entropy(
        log_prior,
        event_one_logp,
        means,
        shared_decision_matrix=shared_decision_matrix,
    )
    source_event = jnn.sigmoid(prediction.inventory_change_logit)[..., None]
    event_value = (
        (1.0 - source_event) * event_zero_value + source_event * event_one_value
    )
    event_entropy = (
        (1.0 - source_event) * event_zero_entropy
        + source_event * event_one_entropy
    )

    previous = jnp.asarray(previous_visibility, dtype=jnp.bool_)
    if tuple(previous.shape) == lead:
        previous = jnp.broadcast_to(previous[..., None], lead + (probe_count,))
    if tuple(previous.shape) != lead + (probe_count,):
        raise ValueError("Previous-visibility axes differ from probe actions.")
    visible_value = jnp.where(
        previous[..., None, None], event_value, base_visible_value
    )
    visible_entropy = jnp.where(
        previous[..., None, None], event_entropy, base_visible_entropy
    )

    full_visible_value = jnp.mean(visible_value, axis=-1)
    full_visible_entropy = jnp.mean(visible_entropy, axis=-1)
    half_count = samples // 2
    half_visible_value = jnp.mean(visible_value[..., :half_count], axis=-1)

    expected_given_source = (
        (1.0 - source_visibility) * invisible_value
        + source_visibility * full_visible_value
    )
    half_expected_given_source = (
        (1.0 - source_visibility) * invisible_value
        + source_visibility * half_visible_value
    )
    entropy_given_source = (
        (1.0 - source_visibility) * invisible_entropy
        + source_visibility * full_visible_entropy
    )
    source_weights = predictive[..., None, :]
    expected_posterior_value = jnp.sum(
        source_weights * expected_given_source, axis=-1
    )
    half_posterior_value = jnp.sum(
        source_weights * half_expected_given_source, axis=-1
    )
    expected_posterior_entropy = jnp.sum(
        source_weights * entropy_given_source, axis=-1
    )

    if shared_decision_matrix:
        prior_action_values = jnp.einsum("...j,...ja->...a", predictive, means)
        prior_value = jnp.broadcast_to(
            jnp.max(prior_action_values, axis=-1)[..., None], lead + (probe_count,)
        )
    else:
        prior_action_values = jnp.einsum("...j,...pja->...pa", predictive, means)
        prior_value = jnp.max(prior_action_values, axis=-1)

    predictive_entropy = -jnp.sum(
        predictive * jnp.log(jnp.maximum(predictive, 1.0e-30)), axis=-1
    )
    raw_value = expected_posterior_value - prior_value
    information_gain = predictive_entropy[..., None] - expected_posterior_entropy
    quadrature_error = jnp.abs(
        expected_posterior_value - half_posterior_value
    )
    return VOIResult(
        # Exact decision VOI is non-negative by convexity.  This clamp only
        # suppresses finite-quadrature negative noise; the raw value is stored.
        value=jnp.maximum(raw_value, 0.0),
        raw_value=raw_value,
        expected_posterior_value=expected_posterior_value,
        prior_value=prior_value,
        expected_posterior_entropy=expected_posterior_entropy,
        predictive_entropy=jnp.broadcast_to(
            predictive_entropy[..., None], lead + (probe_count,)
        ),
        expected_information_gain=jnp.maximum(information_gain, 0.0),
        quadrature_error_estimate=quadrature_error,
    )


def myopic_value_of_information(
    belief: Any,
    transition_logits: Any,
    response_prediction_by_action: Any,
    decision_means: Any,
    *,
    previous_visibility: Any,
    sample_count: int,
) -> Any:
    """Compatibility wrapper returning only the non-negative VOI estimate."""

    return myopic_value_of_information_details(
        belief,
        transition_logits,
        response_prediction_by_action,
        decision_means,
        previous_visibility=previous_visibility,
        sample_count=sample_count,
    ).value


__all__ = [
    "halton_points",
    "myopic_value_of_information",
    "myopic_value_of_information_details",
]
