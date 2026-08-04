"""Exact filtering and one joint response-decision likelihood."""

from __future__ import annotations

from typing import Any, NamedTuple


class FilterStep(NamedTuple):
    predictive: Any
    posterior: Any
    log_evidence: Any


class SequenceLikelihood(NamedTuple):
    total_nll: Any
    response_nll: Any
    decision_nll: Any
    final_posterior: Any
    posterior_sequence: Any
    response_count: Any
    decision_count: Any


def transition_matrix(transition_logits: Any) -> Any:
    """Return a strictly positive row-stochastic learned transition matrix."""

    import jax.nn as jnn
    import jax.numpy as jnp

    logits = jnp.asarray(transition_logits, dtype=jnp.float32)
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Transition logits must be a square KxK matrix.")
    return jnn.softmax(logits, axis=-1)


def initial_belief(batch_shape: tuple[int, ...], component_count: int) -> Any:
    import jax.numpy as jnp

    count = int(component_count)
    if count < 2:
        raise ValueError("A coordination-mode model requires at least two components.")
    return jnp.full(
        tuple(int(value) for value in batch_shape) + (count,),
        1.0 / float(count),
        dtype=jnp.float32,
    )


def filter_step(
    previous_belief: Any,
    transition: Any,
    response_log_likelihood: Any,
    *,
    episode_start: Any = False,
) -> FilterStep:
    """One physical-time Chapman-Kolmogorov/Bayes update.

    Missing response factors are represented by zero log likelihood inside the
    emission model.  The state transition is never disabled merely because the
    partner is occluded; negative visibility remains legitimate evidence.
    """

    from jax.scipy.special import logsumexp
    import jax.numpy as jnp

    previous = jnp.asarray(previous_belief, dtype=jnp.float32)
    matrix = jnp.asarray(transition, dtype=jnp.float32)
    likelihood = jnp.asarray(response_log_likelihood, dtype=jnp.float32)
    if matrix.shape != (previous.shape[-1], previous.shape[-1]):
        raise ValueError("Belief and transition component counts differ.")
    if likelihood.shape != previous.shape:
        raise ValueError("Response likelihood must have one value per component.")
    start = jnp.asarray(episode_start, dtype=jnp.bool_)
    prior = jnp.where(
        start[..., None],
        jnp.full_like(previous, 1.0 / float(previous.shape[-1])),
        previous,
    )
    predictive = prior @ matrix
    log_joint = jnp.log(jnp.maximum(predictive, 1.0e-30)) + likelihood
    log_evidence = logsumexp(log_joint, axis=-1)
    posterior = jnp.exp(log_joint - log_evidence[..., None])
    return FilterStep(predictive, posterior, log_evidence)


def decision_emission_log_probability(
    mean: Any,
    log_scale: Any,
    target: Any,
    standard_error: Any,
    action_mask: Any,
) -> Any:
    """Heteroscedastic Gaussian log p(A_hat | z,x,r,u).

    Replica standard errors are added to the learned emission variance, so
    noisy anchors are downweighted by probability theory rather than by an
    independent confidence heuristic or temperature.
    """

    import jax.nn as jnn
    import jax.numpy as jnp

    predicted = jnp.asarray(mean, dtype=jnp.float32)
    learned_scale = jnn.softplus(jnp.asarray(log_scale, dtype=jnp.float32)) + 1.0e-4
    observed = jnp.asarray(target, dtype=jnp.float32)
    observed = observed - jnp.mean(observed, axis=-1, keepdims=True)
    error_scale = jnp.asarray(standard_error, dtype=jnp.float32)
    mask = jnp.asarray(action_mask, dtype=jnp.float32)
    if predicted.shape[:-2] != observed.shape[:-1] or predicted.shape[-1] != observed.shape[-1]:
        raise ValueError("Decision emission and target action axes differ.")
    if learned_scale.shape != predicted.shape:
        raise ValueError("Decision mean and scale shapes differ.")
    if error_scale.shape != observed.shape or mask.shape != observed.shape:
        raise ValueError("Decision uncertainty/mask shapes differ from targets.")
    total_variance = jnp.square(learned_scale) + jnp.square(error_scale[..., None, :]) + 1.0e-6
    residual = observed[..., None, :] - predicted
    log_probability = -0.5 * (
        jnp.square(residual) / total_variance
        + jnp.log(total_variance)
        + jnp.log(2.0 * jnp.pi)
    )
    return jnp.sum(mask[..., None, :] * log_probability, axis=-1)


def sequence_log_likelihood(
    *,
    transition: Any,
    response_log_likelihood: Any,
    episode_starts: Any,
    valid_mask: Any,
    decision_log_likelihood: Any | None = None,
    decision_mask: Any | None = None,
) -> SequenceLikelihood:
    """Forward algorithm for the single joint latent-variable objective.

    Response observations occur at every valid transition.  Privileged
    counterfactual decision observations occur only where ``decision_mask`` is
    true.  Their likelihood multiplies the response likelihood under the same
    latent component; there is no comparator, pseudo-label, or separation loss.
    """

    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    response = jnp.asarray(response_log_likelihood, dtype=jnp.float32)
    starts = jnp.asarray(episode_starts, dtype=jnp.bool_)
    valid = jnp.asarray(valid_mask, dtype=jnp.float32)
    if response.ndim < 2 or response.shape[:-1] != starts.shape or starts.shape != valid.shape:
        raise ValueError("Sequence likelihood time/batch axes differ.")
    count = response.shape[-1]
    matrix = jnp.asarray(transition, dtype=jnp.float32)
    if matrix.shape != (count, count):
        raise ValueError("Sequence transition matrix has the wrong shape.")
    log_transition = jnp.log(jnp.maximum(matrix, 1.0e-30))
    log_uniform = jnp.full((count,), -jnp.log(float(count)), dtype=jnp.float32)

    if decision_log_likelihood is None:
        decision = jnp.zeros_like(response)
        decision_valid = jnp.zeros_like(valid)
    else:
        decision = jnp.asarray(decision_log_likelihood, dtype=jnp.float32)
        decision_valid = jnp.asarray(decision_mask, dtype=jnp.float32)
        if decision.shape != response.shape or decision_valid.shape != valid.shape:
            raise ValueError("Decision evidence axes differ from response evidence.")

    batch_shape = response.shape[1:-1]
    initial = jnp.broadcast_to(log_uniform, batch_shape + (count,))

    def one(log_previous: Any, items: tuple[Any, Any, Any, Any, Any]):
        response_t, decision_t, decision_valid_t, start_t, valid_t = items
        prior = jnp.where(start_t[..., None], log_uniform, log_previous)
        log_predictive = logsumexp(
            prior[..., :, None] + log_transition,
            axis=-2,
        )
        response_joint = log_predictive + response_t
        response_evidence = logsumexp(response_joint, axis=-1)
        joint = response_joint + decision_valid_t[..., None] * decision_t
        joint_evidence = logsumexp(joint, axis=-1)
        posterior = joint - joint_evidence[..., None]
        posterior = jnp.where(valid_t[..., None] > 0.0, posterior, prior)
        response_term = -valid_t * response_evidence
        decision_term = -valid_t * decision_valid_t * (
            joint_evidence - response_evidence
        )
        return posterior, (
            posterior,
            response_term,
            decision_term,
        )

    final_log_posterior, (posterior_log, response_terms, decision_terms) = jax.lax.scan(
        one,
        initial,
        (response, decision, decision_valid, starts, valid),
    )
    response_count = jnp.sum(valid)
    decision_count = jnp.sum(valid * decision_valid)
    response_nll = jnp.sum(response_terms) / jnp.maximum(response_count, 1.0)
    decision_nll = jnp.sum(decision_terms) / jnp.maximum(decision_count, 1.0)
    total_observations = response_count + decision_count
    total_nll = (jnp.sum(response_terms) + jnp.sum(decision_terms)) / jnp.maximum(
        total_observations, 1.0
    )
    return SequenceLikelihood(
        total_nll=total_nll,
        response_nll=response_nll,
        decision_nll=decision_nll,
        final_posterior=jnp.exp(final_log_posterior),
        posterior_sequence=jnp.exp(posterior_log),
        response_count=response_count,
        decision_count=decision_count,
    )


__all__ = [
    "FilterStep",
    "SequenceLikelihood",
    "decision_emission_log_probability",
    "filter_step",
    "initial_belief",
    "sequence_log_likelihood",
    "transition_matrix",
]
