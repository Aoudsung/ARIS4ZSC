"""Conservative value quotients and persistent slot-level Bayesian filtering."""

from __future__ import annotations

from typing import Any, NamedTuple


VALUE_EQUIVALENCE_TOLERANCE = 1.0e-3
POSTERIOR_SUPPORT_FLOOR = 1.0e-3


class QuotientAggregation(NamedTuple):
    class_ids: Any
    class_mask: Any
    class_counts: Any
    class_belief: Any
    response_probabilities: Any
    reward_mean: Any


def normalized_advantages(q_values: Any) -> Any:
    import jax.numpy as jnp

    values = jnp.asarray(q_values)
    return values - jnp.max(values, axis=-1, keepdims=True)


def centered_advantages(q_values: Any) -> Any:
    return normalized_advantages(q_values)


def value_signatures(q_values: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(q_values)
    if values.shape[-3] != 2:
        raise ValueError("Value signatures require exactly two Q estimators.")
    signatures = normalized_advantages(values)
    signature = 0.5 * (signatures[..., 0, :, :] + signatures[..., 1, :, :])
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
    """Merge only when the upper confidence bound certifies equivalence."""

    import jax
    import jax.numpy as jnp

    if not 0.0 <= float(tolerance) < float("inf"):
        raise ValueError("Value-quotient tolerance must be finite and non-negative.")
    signature_values = jnp.asarray(signatures)
    radius_values = jnp.asarray(radii)
    if radius_values.shape != signature_values.shape[:-1]:
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
        compatible = (
            distances
            + radii_at_state[:, None]
            + radii_at_state[None, :]
        ) <= float(tolerance)
        initial = (
            jnp.full((slot_count,), -1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def assign(index: int, carry: tuple[Any, Any]) -> tuple[Any, Any]:
            class_ids, class_count = carry
            indexes = jnp.arange(slot_count, dtype=jnp.int32)
            members = (class_ids[None, :] == indexes[:, None]) & (
                jnp.arange(slot_count) < index
            )[None, :]
            eligible = (
                jnp.any(members, axis=-1)
                & jnp.all(
                    jnp.where(members, compatible[index][None, :], True),
                    axis=-1,
                )
                & (indexes < class_count)
            )
            first = jnp.min(
                jnp.where(eligible, indexes, jnp.asarray(slot_count, jnp.int32))
            )
            selected = jnp.where(first < slot_count, first, class_count)
            return class_ids.at[index].set(selected), jnp.maximum(
                class_count, selected + 1
            )

        return jax.lax.fori_loop(0, slot_count, assign, initial)[0]

    flat = jax.vmap(one)(flat_signatures, flat_radii)
    return flat.reshape(signature_values.shape[:-2] + (slot_count,))


def class_membership(class_ids: Any, slot_count: int) -> tuple[Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    membership = jax.nn.one_hot(
        jnp.asarray(class_ids, dtype=jnp.int32), slot_count, dtype=jnp.float32
    )
    counts = jnp.sum(membership, axis=-2)
    return membership, counts, counts > 0.0


def normalized_slot_probabilities(slot_log_belief: Any) -> Any:
    import jax
    import jax.numpy as jnp

    values = jnp.asarray(slot_log_belief, dtype=jnp.float32)
    return jnp.exp(
        values - jax.scipy.special.logsumexp(values, axis=-1, keepdims=True)
    )


def class_sum(values: Any, membership: Any) -> Any:
    import jax.numpy as jnp

    return jnp.einsum("...mc,...m->...c", membership, values)


def _posterior_class_mean(
    values: Any, membership: Any, slot_probabilities: Any, class_belief: Any
) -> Any:
    import jax.numpy as jnp

    source = jnp.asarray(values)
    prefix_rank = membership.ndim - 2
    slot_axis = prefix_rank
    flattened = source.reshape(
        source.shape[:prefix_rank] + (source.shape[slot_axis], -1)
    )
    weighted = membership * slot_probabilities[..., :, None]
    reduced = jnp.einsum("...mc,...mZ->...cZ", weighted, flattened)
    result = reduced / jnp.maximum(class_belief, 1.0e-12)[..., :, None]
    return result.reshape(result.shape[:-1] + source.shape[slot_axis + 1 :])


def aggregate_slots(
    *,
    class_ids: Any,
    slot_log_belief: Any,
    response_probabilities: Any,
    reward_mean: Any,
) -> QuotientAggregation:
    """Current-control class view; it never overwrites slot posterior."""

    import jax.numpy as jnp

    response = jnp.asarray(response_probabilities)
    slot_count = int(response.shape[-3])
    membership, counts, mask = class_membership(class_ids, slot_count)
    slot_probabilities = normalized_slot_probabilities(slot_log_belief)
    class_belief = class_sum(slot_probabilities, membership)
    class_belief = jnp.where(mask, class_belief, 0.0)
    class_belief = class_belief / jnp.maximum(
        jnp.sum(class_belief, axis=-1, keepdims=True), 1.0e-12
    )
    return QuotientAggregation(
        class_ids=class_ids,
        class_mask=mask,
        class_counts=counts,
        class_belief=class_belief,
        response_probabilities=_posterior_class_mean(
            response, membership, slot_probabilities, class_belief
        ),
        reward_mean=_posterior_class_mean(
            reward_mean, membership, slot_probabilities, class_belief
        ),
    )


def posterior_supported_class_count(
    *,
    class_ids: Any,
    slot_log_belief: Any,
    probability_floor: float = POSTERIOR_SUPPORT_FLOOR,
) -> Any:
    import jax.numpy as jnp

    floor = float(probability_floor)
    if not 0.0 <= floor < 1.0:
        raise ValueError("Posterior support floor must lie in [0, 1).")
    probabilities = normalized_slot_probabilities(slot_log_belief)
    slot_count = int(probabilities.shape[-1])
    membership, unused_counts, class_mask = class_membership(
        class_ids, slot_count
    )
    del unused_counts
    class_belief = class_sum(probabilities, membership)
    return jnp.sum(
        class_mask & (class_belief >= floor), axis=-1, dtype=jnp.int32
    )


supported_quotient_count = posterior_supported_class_count


def slot_bayes_update(
    *,
    slot_log_belief: Any,
    slot_response_probabilities: Any,
    action: Any,
    response_code: Any,
    probability_floor: float = 1.0e-8,
) -> Any:
    import jax
    import jax.numpy as jnp

    probabilities = jnp.asarray(slot_response_probabilities)
    action_index = jnp.asarray(action, dtype=jnp.int32)
    response_index = jnp.asarray(response_code, dtype=jnp.int32)
    selected_action = jnp.take_along_axis(
        probabilities, action_index[..., None, None, None], axis=-2
    )[..., 0, :]
    likelihood = jnp.take_along_axis(
        selected_action, response_index[..., None, None], axis=-1
    )[..., 0]
    posterior = jnp.asarray(slot_log_belief) + jnp.log(
        jnp.clip(likelihood, float(probability_floor), 1.0)
    )
    return posterior - jax.scipy.special.logsumexp(
        posterior, axis=-1, keepdims=True
    )


def quotient_bayes_update(**unused: Any) -> Any:
    del unused
    raise RuntimeError(
        "quotient_bayes_update is retired; preserve and update slot belief."
    )


__all__ = [
    "POSTERIOR_SUPPORT_FLOOR",
    "QuotientAggregation",
    "VALUE_EQUIVALENCE_TOLERANCE",
    "aggregate_slots",
    "centered_advantages",
    "class_membership",
    "class_sum",
    "complete_link_class_ids",
    "normalized_advantages",
    "normalized_slot_probabilities",
    "posterior_supported_class_count",
    "quotient_bayes_update",
    "slot_bayes_update",
    "supported_quotient_count",
    "value_signatures",
]
