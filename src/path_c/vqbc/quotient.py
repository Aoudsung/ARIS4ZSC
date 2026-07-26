"""Conservative value quotients and persistent slot-level belief updates.

A value quotient is a control view over latent Bellman experts. It must not
replace the latent posterior: experts can be action-equivalent in one public
context and control-distinct later. This module therefore certifies equivalence
only when a confidence upper bound is small, aggregates class views under the
current slot posterior, and performs Bayesian filtering at slot level.
"""

from __future__ import annotations

from typing import Any, NamedTuple


VALUE_EQUIVALENCE_TOLERANCE = 1.0e-3


class QuotientAggregation(NamedTuple):
    class_ids: Any
    class_mask: Any
    class_counts: Any
    class_belief: Any
    response_probabilities: Any
    reward_mean: Any
    next_q_mean: Any


def normalized_advantages(q_values: Any) -> Any:
    """Remove value offsets while retaining optimal actions and all gaps."""

    import jax.numpy as jnp

    values = jnp.asarray(q_values)
    return values - jnp.max(values, axis=-1, keepdims=True)


def centered_advantages(q_values: Any) -> Any:
    """Compatibility alias for the registered max-normalized signature."""

    return normalized_advantages(q_values)


def value_signatures(q_values: Any) -> tuple[Any, Any]:
    """Return twin-mean signatures and half-disagreement confidence radii."""

    import jax.numpy as jnp

    values = jnp.asarray(q_values)
    if values.shape[-3] != 2:
        raise ValueError("Value signatures require exactly two Q estimators.")
    signatures = normalized_advantages(values)
    signature = 0.5 * (
        signatures[..., 0, :, :] + signatures[..., 1, :, :]
    )
    radius = 0.5 * jnp.max(
        jnp.abs(signatures[..., 0, :, :] - signatures[..., 1, :, :]),
        axis=-1,
    )
    return signature, radius


def complete_link_class_ids(
    signatures: Any,
    radii: Any,
    *,
    tolerance: float = VALUE_EQUIVALENCE_TOLERANCE,
) -> Any:
    """Certify complete-link value equivalence in deterministic slot order.

    Two slots are compatible only when an upper confidence bound on their
    control-signature distance is no larger than ``tolerance``::

        ||A_i - A_j||_inf + radius_i + radius_j <= tolerance.

    Uncertainty preserves separate hypotheses; it never makes merging easier.
    """

    import jax
    import jax.numpy as jnp

    if not 0.0 <= float(tolerance) < float("inf"):
        raise ValueError("Value-quotient tolerance must be finite and non-negative.")
    signature_values = jnp.asarray(signatures)
    radius_values = jnp.asarray(radii)
    if (
        signature_values.ndim < 2
        or radius_values.shape != signature_values.shape[:-1]
    ):
        raise ValueError("Signatures and radii have incompatible shapes.")
    slot_count = int(signature_values.shape[-2])
    flat_signatures = signature_values.reshape(
        (-1, slot_count, signature_values.shape[-1])
    )
    flat_radii = radius_values.reshape((-1, slot_count))

    def one(signatures_at_state: Any, radii_at_state: Any) -> Any:
        distances = jnp.max(
            jnp.abs(
                signatures_at_state[:, None, :]
                - signatures_at_state[None, :, :]
            ),
            axis=-1,
        )
        upper_bound = (
            distances
            + radii_at_state[:, None]
            + radii_at_state[None, :]
        )
        compatible = upper_bound <= float(tolerance)
        initial = (
            jnp.full((slot_count,), -1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def assign(index: int, carry: tuple[Any, Any]) -> tuple[Any, Any]:
            class_ids, class_count = carry
            class_indexes = jnp.arange(slot_count, dtype=jnp.int32)
            member_matrix = class_ids[None, :] == class_indexes[:, None]
            earlier = jnp.arange(slot_count) < index
            members = member_matrix & earlier[None, :]
            nonempty = jnp.any(members, axis=-1)
            compatible_with_all = jnp.all(
                jnp.where(members, compatible[index][None, :], True),
                axis=-1,
            )
            eligible = (
                nonempty
                & compatible_with_all
                & (class_indexes < class_count)
            )
            first = jnp.min(
                jnp.where(
                    eligible,
                    class_indexes,
                    jnp.asarray(slot_count, dtype=jnp.int32),
                )
            )
            selected = jnp.where(first < slot_count, first, class_count)
            next_count = jnp.maximum(class_count, selected + 1)
            return class_ids.at[index].set(selected), next_count

        return jax.lax.fori_loop(0, slot_count, assign, initial)[0]

    flat_ids = jax.vmap(one)(flat_signatures, flat_radii)
    return flat_ids.reshape(signature_values.shape[:-2] + (slot_count,))


def class_membership(class_ids: Any, slot_count: int) -> tuple[Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    ids = jnp.asarray(class_ids, dtype=jnp.int32)
    membership = jax.nn.one_hot(ids, slot_count, dtype=jnp.float32)
    counts = jnp.sum(membership, axis=-2)
    mask = counts > 0.0
    return membership, counts, mask


def _normalized_slot_probabilities(slot_log_belief: Any) -> Any:
    import jax
    import jax.numpy as jnp

    values = jnp.asarray(slot_log_belief, dtype=jnp.float32)
    return jnp.exp(
        values - jax.scipy.special.logsumexp(values, axis=-1, keepdims=True)
    )


def class_sum(values: Any, membership: Any) -> Any:
    import jax.numpy as jnp

    source = jnp.asarray(values)
    members = jnp.asarray(membership)
    return jnp.einsum("...mc,...m->...c", members, source)


def class_mean(values: Any, membership: Any, counts: Any) -> Any:
    """Uniform class mean retained for diagnostics and legacy analysis."""

    import jax.numpy as jnp

    source = jnp.asarray(values)
    members = jnp.asarray(membership)
    count_values = jnp.asarray(counts)
    prefix_rank = members.ndim - 2
    slot_axis = prefix_rank
    flattened = source.reshape(
        source.shape[:prefix_rank] + (source.shape[slot_axis], -1)
    )
    reduced = jnp.einsum("...mc,...mZ->...cZ", members, flattened)
    safe_counts = jnp.maximum(count_values, 1.0)[..., :, None]
    result = reduced / safe_counts
    suffix = source.shape[slot_axis + 1 :]
    return result.reshape(result.shape[:-1] + suffix)


def _posterior_class_mean(
    values: Any,
    membership: Any,
    slot_probabilities: Any,
    class_belief: Any,
) -> Any:
    """Condition a slot-axis tensor on each class under current posterior."""

    import jax.numpy as jnp

    source = jnp.asarray(values)
    members = jnp.asarray(membership)
    slot_probs = jnp.asarray(slot_probabilities)
    class_probs = jnp.asarray(class_belief)
    prefix_rank = members.ndim - 2
    slot_axis = prefix_rank
    flattened = source.reshape(
        source.shape[:prefix_rank] + (source.shape[slot_axis], -1)
    )
    weighted_membership = members * slot_probs[..., :, None]
    reduced = jnp.einsum(
        "...mc,...mZ->...cZ", weighted_membership, flattened
    )
    safe = jnp.maximum(class_probs, 1.0e-12)[..., :, None]
    result = reduced / safe
    suffix = source.shape[slot_axis + 1 :]
    return result.reshape(result.shape[:-1] + suffix)


def aggregate_slots(
    *,
    class_ids: Any,
    slot_log_belief: Any,
    response_probabilities: Any,
    reward_mean: Any,
    next_q_mean: Any,
) -> QuotientAggregation:
    """Build posterior-conditioned class views without altering slot belief."""

    import jax.numpy as jnp

    response = jnp.asarray(response_probabilities)
    slot_count = int(response.shape[-3])
    membership, counts, mask = class_membership(class_ids, slot_count)
    slot_probabilities = _normalized_slot_probabilities(slot_log_belief)
    class_belief = class_sum(slot_probabilities, membership)
    class_belief = jnp.where(mask, class_belief, 0.0)
    class_belief = class_belief / jnp.maximum(
        jnp.sum(class_belief, axis=-1, keepdims=True), 1.0e-12
    )

    continuation_by_slot = jnp.moveaxis(jnp.asarray(next_q_mean), -5, -4)
    continuation_by_class = _posterior_class_mean(
        continuation_by_slot,
        membership,
        slot_probabilities,
        class_belief,
    )
    continuation_by_class = jnp.moveaxis(continuation_by_class, -4, -5)
    return QuotientAggregation(
        class_ids=class_ids,
        class_mask=mask,
        class_counts=counts,
        class_belief=class_belief,
        response_probabilities=_posterior_class_mean(
            response,
            membership,
            slot_probabilities,
            class_belief,
        ),
        reward_mean=_posterior_class_mean(
            reward_mean,
            membership,
            slot_probabilities,
            class_belief,
        ),
        next_q_mean=continuation_by_class,
    )


def slot_bayes_update(
    *,
    slot_log_belief: Any,
    slot_response_probabilities: Any,
    action: Any,
    response_code: Any,
    probability_floor: float = 1.0e-8,
) -> Any:
    """Update the persistent latent-expert posterior from one response."""

    import jax
    import jax.numpy as jnp

    if not 0.0 < float(probability_floor) < 1.0:
        raise ValueError("probability_floor must lie strictly between zero and one.")
    probabilities = jnp.asarray(slot_response_probabilities)
    log_prior = jnp.asarray(slot_log_belief, dtype=jnp.float32)
    action_index = jnp.asarray(action, dtype=jnp.int32)
    response_index = jnp.asarray(response_code, dtype=jnp.int32)
    selected_action = jnp.take_along_axis(
        probabilities,
        action_index[..., None, None, None],
        axis=-2,
    )[..., 0, :]
    likelihood = jnp.take_along_axis(
        selected_action,
        response_index[..., None, None],
        axis=-1,
    )[..., 0]
    log_likelihood = jnp.log(
        jnp.clip(likelihood, float(probability_floor), 1.0)
    )
    posterior = log_prior + log_likelihood
    return posterior - jax.scipy.special.logsumexp(
        posterior, axis=-1, keepdims=True
    )


def quotient_bayes_update(**unused: Any) -> Any:
    """Fail closed on the retired class-to-uniform-slot projection."""

    del unused
    raise RuntimeError(
        "quotient_bayes_update is retired: update persistent slot belief with "
        "slot_bayes_update and use value quotients only as a control view."
    )


__all__ = [
    "QuotientAggregation",
    "VALUE_EQUIVALENCE_TOLERANCE",
    "aggregate_slots",
    "centered_advantages",
    "class_mean",
    "class_membership",
    "class_sum",
    "complete_link_class_ids",
    "normalized_advantages",
    "quotient_bayes_update",
    "slot_bayes_update",
    "value_signatures",
]
