"""Myopic Bayes value of information derived from the response model.

For each ego action the response distribution is coarsened to the three exact
observable outcomes

    invisible,
    visible without inventory change,
    visible with inventory change.

This is a deterministic marginal of the registered response model.  The code
enumerates all outcomes exactly, updates the mode belief by Bayes' rule, and
computes the expected improvement in the best posterior action value.  There is
no information-bonus coefficient; VOI is already measured in task-reward units.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class VOIResult(NamedTuple):
    value: Any
    predictive_belief: Any
    outcome_probability: Any
    posterior_by_outcome: Any


def coarse_response_probability(
    visibility_logits: Any, inventory_change_logits: Any
) -> Any:
    """Return p(outcome | z,a), shape ``[...,action,K,3]``."""

    import jax.nn as jnn
    import jax.numpy as jnp

    visibility = jnn.sigmoid(jnp.asarray(visibility_logits, dtype=jnp.float32))
    change = jnn.sigmoid(
        jnp.asarray(inventory_change_logits, dtype=jnp.float32)
    )
    if visibility.shape != change.shape:
        raise ValueError("Visibility and inventory-change logits must align.")
    invisible = 1.0 - visibility
    visible_stable = visibility * (1.0 - change)
    visible_change = visibility * change
    probabilities = jnp.stack(
        (invisible, visible_stable, visible_change), axis=-1
    )
    return probabilities / jnp.maximum(
        jnp.sum(probabilities, axis=-1, keepdims=True), 1.0e-12
    )


def myopic_value_of_information(
    *,
    belief: Any,
    transition: Any,
    component_action_values: Any,
    response_outcome_probability: Any,
) -> VOIResult:
    """Exact outcome enumeration under a frozen local decision geometry.

    Args:
        belief: ``[...,K]`` current mode posterior.
        transition: ``[K,K]`` learned physical-time transition matrix.
        component_action_values: ``[...,K,A]`` decision-emission means.
        response_outcome_probability: ``[...,A,K,Y]`` response probabilities.
    """

    import jax.numpy as jnp

    posterior = jnp.asarray(belief, dtype=jnp.float32)
    matrix = jnp.asarray(transition, dtype=jnp.float32)
    values = jnp.asarray(component_action_values, dtype=jnp.float32)
    response = jnp.asarray(response_outcome_probability, dtype=jnp.float32)
    if matrix.shape != (posterior.shape[-1], posterior.shape[-1]):
        raise ValueError("VOI transition and belief dimensions differ.")
    if values.shape[:-2] != posterior.shape[:-1] or values.shape[-2] != posterior.shape[-1]:
        raise ValueError("VOI component action values do not align with belief.")
    if response.shape[:-3] != posterior.shape[:-1]:
        raise ValueError("VOI response batch axes differ from belief.")
    if response.shape[-2] != posterior.shape[-1] or response.shape[-3] != values.shape[-1]:
        raise ValueError("VOI action/component axes differ.")

    predictive = posterior @ matrix
    # p(y | a) = sum_k p(k) p(y | k,a)
    outcome = jnp.sum(
        predictive[..., None, :, None] * response,
        axis=-2,
    )
    joint = predictive[..., None, :, None] * response
    posterior_outcome = joint / jnp.maximum(outcome[..., :, None, :], 1.0e-12)
    # [...,A,K,Y] -> [...,A,Y,K]
    posterior_outcome = jnp.moveaxis(posterior_outcome, -2, -1)
    # Under each posterior outcome, choose the best next ego action from the
    # same local decision-emission geometry.
    value_by_outcome_action = jnp.einsum(
        "...ayk,...kb->...ayb", posterior_outcome, values
    )
    best_by_outcome = jnp.max(value_by_outcome_action, axis=-1)
    expected_after = jnp.sum(outcome * best_by_outcome, axis=-1)
    predictive_action_values = jnp.sum(
        predictive[..., :, None] * values, axis=-2
    )
    before = jnp.max(predictive_action_values, axis=-1, keepdims=True)
    value = jnp.maximum(expected_after - before, 0.0)
    return VOIResult(
        value=value,
        predictive_belief=predictive,
        outcome_probability=outcome,
        posterior_by_outcome=posterior_outcome,
    )


__all__ = [
    "VOIResult",
    "coarse_response_probability",
    "myopic_value_of_information",
]
