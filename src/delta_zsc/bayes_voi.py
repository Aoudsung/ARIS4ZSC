"""Exact Bayesian value of the compact active-response marginal.

Passive filtering scores the complete direct geometry, aligned interface event,
and independently covered recipe response.  Active DELTA marginalizes the
direct conditional geometry and recipe factors and exactly enumerates
``(visibility, interface_available, interface_changed, interface_event)``:
two unavailable, two available/no-change, and sixty-two structured-change
outcomes, for 66 outcomes per probe action.
"""

from __future__ import annotations

from typing import Any

from .observation import INTERFACE_EVENT_CLASSES
from .transition import transition_matrix
from .types import VOIResult


def _validate_inputs(
    belief: Any, prediction: Any, decision_means: Any
) -> tuple[tuple[int, ...], int, int]:
    import jax.numpy as jnp

    probability = jnp.asarray(belief)
    visibility = jnp.asarray(prediction.direct.visibility_logit)
    means = jnp.asarray(decision_means)
    lead = tuple(probability.shape[:-1])
    components = int(probability.shape[-1])
    if tuple(visibility.shape[:-2]) != lead or int(visibility.shape[-1]) != components:
        raise ValueError("VOI response axes differ from belief.")
    probes = int(visibility.shape[-2])
    shared = lead + (components, means.shape[-1])
    conditioned = lead + (probes, components, means.shape[-1])
    if tuple(means.shape) not in (shared, conditioned):
        raise ValueError("Decision means must be shared or probe-conditioned.")
    return lead, components, probes


def compact_active_outcome_log_probabilities(prediction: Any) -> Any:
    """Return normalized ``[..., probe, 66, K]`` component log probabilities."""

    import jax.nn as jnn
    import jax.numpy as jnp

    direct = prediction.direct
    visibility_one = jnn.log_sigmoid(direct.visibility_logit)
    visibility_zero = jnn.log_sigmoid(-direct.visibility_logit)
    availability_one = jnn.log_sigmoid(prediction.interface_availability_logit)[..., None]
    availability_zero = jnn.log_sigmoid(-prediction.interface_availability_logit)[..., None]
    change_one = jnn.log_sigmoid(prediction.interface_change_logit)
    change_zero = jnn.log_sigmoid(-prediction.interface_change_logit)
    event = jnn.log_softmax(prediction.interface_event_logits, axis=-1)

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
        + availability_one[..., None]
        + change_one[..., None, :]
        + event_outcome
    )
    changed_one = (
        visibility_one[..., None, :]
        + availability_one[..., None]
        + change_one[..., None, :]
        + event_outcome
    )
    changed = jnp.concatenate((changed_zero, changed_one), axis=-2)
    # unavailable/no_change are [..., probe, 2, K]; changed has the same
    # outcome-before-component convention.
    return jnp.concatenate((unavailable, no_change, changed), axis=-2)


def myopic_value_of_information_details(
    belief: Any,
    transition_logits: Any,
    response_prediction_by_action: Any,
    decision_means: Any,
) -> VOIResult:
    """Exactly integrate the registered 66-outcome one-response decision value."""

    import jax.nn as jnn
    import jax.numpy as jnp
    import jax.scipy as jsp

    lead, _, probes = _validate_inputs(
        belief, response_prediction_by_action, decision_means
    )
    probability = jnp.asarray(belief, dtype=jnp.float32)
    probability = probability / jnp.sum(probability, axis=-1, keepdims=True)
    predictive = probability @ transition_matrix(transition_logits)
    outcome_component_logp = compact_active_outcome_log_probabilities(
        response_prediction_by_action
    )
    joint_logp = (
        jnp.log(jnp.maximum(predictive, 1.0e-30))[..., None, None, :]
        + outcome_component_logp
    )
    outcome_logp = jsp.special.logsumexp(joint_logp, axis=-1)
    outcome_probability = jnp.exp(outcome_logp)
    posterior = jnn.softmax(joint_logp, axis=-1)
    means = jnp.asarray(decision_means, dtype=jnp.float32)
    if means.ndim == probability.ndim + 1:
        posterior_action_values = jnp.einsum("...pok,...ka->...poa", posterior, means)
        prior_action_values = jnp.einsum("...k,...ka->...a", predictive, means)
        prior_value = jnp.broadcast_to(
            jnp.max(prior_action_values, axis=-1)[..., None], lead + (probes,)
        )
    else:
        posterior_action_values = jnp.einsum("...pok,...pka->...poa", posterior, means)
        prior_action_values = jnp.einsum("...k,...pka->...pa", predictive, means)
        prior_value = jnp.max(prior_action_values, axis=-1)
    posterior_value = jnp.max(posterior_action_values, axis=-1)
    expected_posterior_value = jnp.sum(outcome_probability * posterior_value, axis=-1)
    posterior_entropy = -jnp.sum(
        posterior * jnp.log(jnp.maximum(posterior, 1.0e-30)), axis=-1
    )
    expected_posterior_entropy = jnp.sum(
        outcome_probability * posterior_entropy, axis=-1
    )
    predictive_entropy = -jnp.sum(
        predictive * jnp.log(jnp.maximum(predictive, 1.0e-30)), axis=-1
    )
    value = expected_posterior_value - prior_value
    return VOIResult(
        value=value,
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
    transition_logits: Any,
    response_prediction_by_action: Any,
    decision_means: Any,
) -> Any:
    return myopic_value_of_information_details(
        belief, transition_logits, response_prediction_by_action, decision_means
    ).value


__all__ = [
    "compact_active_outcome_log_probabilities",
    "myopic_value_of_information",
    "myopic_value_of_information_details",
]
