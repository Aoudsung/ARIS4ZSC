"""Exact Bayesian value of the delayed compact probe-response marginal.

DELTA v4 enumerates ``(visibility, interface_available, interface_changed,
interface_event)`` exactly: two unavailable outcomes, two available/no-change
outcomes, and sixty-two structured-change outcomes.  The posterior is valued
with a probe-conditioned successor decision matrix, never the current-state
matrix.
"""

from __future__ import annotations

from typing import Any

from .observation import INTERFACE_EVENT_CLASSES
from .types import ProbeResponsePrediction, VOIResult


def _validate_inputs(
    belief: Any,
    prediction: ProbeResponsePrediction,
    successor_decision_means: Any,
) -> tuple[tuple[int, ...], int, int, int]:
    import jax.numpy as jnp

    probability = jnp.asarray(belief)
    event = jnp.asarray(prediction.interface_event_logits)
    means = jnp.asarray(successor_decision_means)
    lead = tuple(probability.shape[:-1])
    components = int(probability.shape[-1])
    if tuple(event.shape[:-3]) != lead or int(event.shape[-2]) != components:
        raise ValueError("Probe-response component axes differ from belief.")
    probes = int(event.shape[-3])
    if int(event.shape[-1]) != INTERFACE_EVENT_CLASSES:
        raise ValueError("Probe-response event support differs from 31 classes.")
    expected = lead + (probes, components, means.shape[-1])
    if tuple(means.shape) != expected:
        raise ValueError(
            "DELTA v4 requires probe-conditioned successor decision means."
        )
    shared_shape = lead + (probes,)
    for value in (
        prediction.visibility_logit,
        prediction.interface_availability_logit,
        prediction.interface_change_logit,
    ):
        if tuple(jnp.asarray(value).shape) != shared_shape:
            raise ValueError("Probe-response shared occurrence axes differ.")
    return lead, components, probes, int(means.shape[-1])


def compact_active_outcome_log_probabilities(
    prediction: ProbeResponsePrediction,
) -> Any:
    """Return normalized ``[..., probe, 66, K]`` component log probabilities."""

    import jax.nn as jnn
    import jax.numpy as jnp

    event = jnn.log_softmax(prediction.interface_event_logits, axis=-1)
    components = int(event.shape[-2])

    def expand(shared: Any) -> Any:
        value = jnp.asarray(shared, dtype=jnp.float32)
        return jnp.broadcast_to(value[..., None], value.shape + (components,))

    visibility_one = expand(jnn.log_sigmoid(prediction.visibility_logit))
    visibility_zero = expand(jnn.log_sigmoid(-prediction.visibility_logit))
    availability_one = expand(
        jnn.log_sigmoid(prediction.interface_availability_logit)
    )
    availability_zero = expand(
        jnn.log_sigmoid(-prediction.interface_availability_logit)
    )
    change_one = expand(jnn.log_sigmoid(prediction.interface_change_logit))
    change_zero = expand(jnn.log_sigmoid(-prediction.interface_change_logit))

    unavailable = jnp.stack(
        (visibility_zero + availability_zero, visibility_one + availability_zero),
        axis=-2,
    )
    no_change = jnp.stack(
        (
            visibility_zero + availability_one + change_zero,
            visibility_one + availability_one + change_zero,
        ),
        axis=-2,
    )
    event_outcome = jnp.swapaxes(event, -1, -2)
    changed_zero = (
        visibility_zero[..., None, :]
        + availability_one[..., None, :]
        + change_one[..., None, :]
        + event_outcome
    )
    changed_one = (
        visibility_one[..., None, :]
        + availability_one[..., None, :]
        + change_one[..., None, :]
        + event_outcome
    )
    changed = jnp.concatenate((changed_zero, changed_one), axis=-2)
    return jnp.concatenate((unavailable, no_change, changed), axis=-2)


def myopic_value_of_information_details(
    belief: Any,
    response_prediction_by_action: ProbeResponsePrediction,
    successor_decision_means: Any,
) -> VOIResult:
    """Exactly integrate one delayed response with successor-state utility."""

    import jax.nn as jnn
    import jax.numpy as jnp
    import jax.scipy as jsp

    lead, _, probes, _ = _validate_inputs(
        belief, response_prediction_by_action, successor_decision_means
    )
    probability = jnp.asarray(belief, dtype=jnp.float32)
    probability = probability / jnp.maximum(
        jnp.sum(probability, axis=-1, keepdims=True), 1.0e-30
    )
    outcome_component_logp = compact_active_outcome_log_probabilities(
        response_prediction_by_action
    )
    joint_logp = (
        jnp.log(jnp.maximum(probability, 1.0e-30))[..., None, None, :]
        + outcome_component_logp
    )
    outcome_logp = jsp.special.logsumexp(joint_logp, axis=-1)
    outcome_probability = jnp.exp(outcome_logp)
    posterior = jnn.softmax(joint_logp, axis=-1)
    means = jnp.asarray(successor_decision_means, dtype=jnp.float32)
    posterior_action_values = jnp.einsum(
        "...pok,...pka->...poa", posterior, means
    )
    prior_action_values = jnp.einsum("...k,...pka->...pa", probability, means)
    prior_value = jnp.max(prior_action_values, axis=-1)
    posterior_value = jnp.max(posterior_action_values, axis=-1)
    expected_posterior_value = jnp.sum(
        outcome_probability * posterior_value, axis=-1
    )
    posterior_entropy = -jnp.sum(
        posterior * jnp.log(jnp.maximum(posterior, 1.0e-30)), axis=-1
    )
    expected_posterior_entropy = jnp.sum(
        outcome_probability * posterior_entropy, axis=-1
    )
    predictive_entropy = -jnp.sum(
        probability * jnp.log(jnp.maximum(probability, 1.0e-30)), axis=-1
    )
    return VOIResult(
        value=expected_posterior_value - prior_value,
        expected_posterior_value=expected_posterior_value,
        prior_value=prior_value,
        expected_posterior_entropy=expected_posterior_entropy,
        predictive_entropy=jnp.broadcast_to(
            predictive_entropy[..., None], lead + (probes,)
        ),
        expected_information_gain=(
            predictive_entropy[..., None] - expected_posterior_entropy
        ),
    )


def myopic_value_of_information(
    belief: Any,
    response_prediction_by_action: ProbeResponsePrediction,
    successor_decision_means: Any,
) -> Any:
    return myopic_value_of_information_details(
        belief, response_prediction_by_action, successor_decision_means
    ).value


__all__ = [
    "compact_active_outcome_log_probabilities",
    "myopic_value_of_information",
    "myopic_value_of_information_details",
]
