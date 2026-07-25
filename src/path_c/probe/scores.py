"""Exact one-step neural approximation of the registered sequential score."""

from __future__ import annotations

from typing import Any, NamedTuple


class SequentialScores(NamedTuple):
    j_use: Any
    j_mask: Any
    v_base: Any
    v_mask: Any
    s_seq: Any


def _normalized_belief(log_belief: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    values = jnp.asarray(log_belief)
    if values.ndim != 2:
        raise ValueError("log_belief must have shape [batch, prototype].")
    return jnn.softmax(values, axis=-1)


def sequential_scores(
    *,
    log_belief: Any,
    response_probabilities: Any,
    reward_estimates: Any,
    next_values: Any,
    base_action: Any,
    candidate_mask: Any,
    gamma: float,
    probability_floor: float,
) -> SequentialScores:
    """Compute ``J_use``, ``J_mask``, ``V_mask`` and ``S_seq``.

    Shapes are response ``[B,K,A,Y]``, reward ``[B,K,A]``, and next values
    ``[B,K,A,Y,K]``.  The first prototype axis generates the transition; the
    final prototype axis selects the partner-conditioned value head.

    This is the trainable one-step neural approximation of the proposal's
    frozen continuation value.  It is a controller score, not the paired
    deployment effect later measured from complete evaluation episodes.
    """

    import jax.nn as jnn
    import jax.numpy as jnp

    probability = jnp.asarray(response_probabilities)
    reward = jnp.asarray(reward_estimates)
    continuation = jnp.asarray(next_values)
    if probability.ndim != 4 or reward.ndim != 3 or continuation.ndim != 5:
        raise ValueError("Sequential score tensors have invalid ranks.")
    batch, prototypes, actions, responses = probability.shape
    if reward.shape != (batch, prototypes, actions):
        raise ValueError("reward_estimates must have shape [B,K,A].")
    if continuation.shape != (batch, prototypes, actions, responses, prototypes):
        raise ValueError("next_values must have shape [B,K,A,Y,K].")
    belief = _normalized_belief(log_belief)
    floor = float(probability_floor)
    if not 0.0 < floor < 1.0:
        raise ValueError("probability_floor must lie strictly between zero and one.")
    probability = jnp.clip(probability, floor, 1.0)
    probability = probability / jnp.sum(probability, axis=-1, keepdims=True)

    # Bayes update for every candidate and possible response: [B,A,Y,K].
    log_prior = jnp.log(jnp.clip(belief, floor, 1.0))
    log_likelihood = jnp.log(jnp.transpose(probability, (0, 2, 3, 1)))
    posterior_use = jnn.softmax(log_prior[:, None, None, :] + log_likelihood, axis=-1)

    # Continuation for each generating prototype: [B,K,A,Y].
    use_continuation = jnp.einsum("bayv,bkayv->bkay", posterior_use, continuation)
    mask_continuation = jnp.einsum("bv,bkayv->bkay", belief, continuation)
    immediate = jnp.einsum("bk,bka->ba", belief, reward)
    expected_use = jnp.einsum("bk,bkay,bkay->ba", belief, probability, use_continuation)
    expected_mask = jnp.einsum("bk,bkay,bkay->ba", belief, probability, mask_continuation)
    j_use = immediate + float(gamma) * expected_use
    j_mask = immediate + float(gamma) * expected_mask

    base = jnp.asarray(base_action, dtype=jnp.int32)
    mask = jnp.asarray(candidate_mask, dtype=jnp.bool_)
    if base.shape != (batch,) or mask.shape != (batch, actions):
        raise ValueError("base_action and candidate_mask have invalid shapes.")
    v_base = jnp.take_along_axis(j_mask, base[:, None], axis=-1)[:, 0]
    safe_masked = jnp.where(mask, j_mask, -jnp.inf)
    best_safe = jnp.max(safe_masked, axis=-1)
    v_mask = jnp.maximum(v_base, best_safe)
    s_seq = jnp.where(mask, j_use - v_mask[:, None], -jnp.inf)
    return SequentialScores(j_use=j_use, j_mask=j_mask, v_base=v_base, v_mask=v_mask, s_seq=s_seq)


def response_information_scores(
    *, log_belief: Any, response_probabilities: Any, probability_floor: float
) -> Any:
    """Return the belief-weighted generalized Jensen-Shannon score per action."""

    import jax.numpy as jnp

    probability = jnp.asarray(response_probabilities)
    if probability.ndim != 4:
        raise ValueError("response_probabilities must have shape [B,K,A,Y].")
    belief = _normalized_belief(log_belief)
    floor = float(probability_floor)
    if not 0.0 < floor < 1.0:
        raise ValueError("probability_floor must lie strictly between zero and one.")
    probability = jnp.clip(probability, floor, 1.0)
    probability = probability / jnp.sum(probability, axis=-1, keepdims=True)
    mixture = jnp.einsum("bk,bkay->bay", belief, probability)
    mixture_entropy = -jnp.sum(mixture * jnp.log(jnp.clip(mixture, floor, 1.0)), axis=-1)
    component_entropy = -jnp.sum(probability * jnp.log(jnp.clip(probability, floor, 1.0)), axis=-1)
    expected_entropy = jnp.einsum("bk,bka->ba", belief, component_entropy)
    return jnp.maximum(mixture_entropy - expected_entropy, 0.0)
