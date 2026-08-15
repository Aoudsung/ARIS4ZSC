"""Analytic KL-bounded policy improvement."""

from __future__ import annotations

from typing import Any


def categorical_kl_from_logits(left_logits: Any, right_logits: Any) -> Any:
    import jax.nn
    import jax.numpy as jnp

    left_logp = jax.nn.log_softmax(left_logits, axis=-1)
    right_logp = jax.nn.log_softmax(right_logits, axis=-1)
    probability = jnp.exp(left_logp)
    return jnp.sum(probability * (left_logp - right_logp), axis=-1)


def mirror_policy_logits(
    base_logits: Any,
    action_values: Any,
    *,
    kl_budget: float,
    iterations: int = 48,
) -> tuple[Any, Any, Any]:
    """Solve max_pi E_pi[Q] subject to KL(pi||pi0)<=delta.

    For a fixed dual temperature eta the solution is
    ``pi_eta proportional pi0 * exp(Q/eta)``.  A deterministic bisection finds
    eta; no actor-supervision weight or temperature is exposed.
    """

    import jax
    import jax.nn
    import jax.numpy as jnp

    base = jnp.asarray(base_logits, dtype=jnp.float32)
    values = jnp.asarray(action_values, dtype=jnp.float32)
    values = values - jnp.mean(values, axis=-1, keepdims=True)
    delta = float(kl_budget)
    if delta < 0.0:
        raise ValueError("KL budget must be non-negative.")
    base_logp = jax.nn.log_softmax(base, axis=-1)

    def candidate(log_eta: Any) -> tuple[Any, Any]:
        eta = jnp.exp(log_eta)
        logits = base_logp + values / eta[..., None]
        logp = jax.nn.log_softmax(logits, axis=-1)
        probability = jnp.exp(logp)
        kl = jnp.sum(probability * (logp - base_logp), axis=-1)
        return logits, kl

    lead = values.shape[:-1]
    low = jnp.full(lead, -9.0, dtype=jnp.float32)
    high = jnp.full(lead, 12.0, dtype=jnp.float32)

    def one(_: int, bounds: tuple[Any, Any]) -> tuple[Any, Any]:
        lo, hi = bounds
        middle = 0.5 * (lo + hi)
        _, kl = candidate(middle)
        # KL decreases monotonically as eta grows.
        return jnp.where(kl > delta, middle, lo), jnp.where(kl > delta, hi, middle)

    low, high = jax.lax.fori_loop(0, int(iterations), one, (low, high))
    # The dual temperature is an analytic constraint solution, not a learned
    # pathway.  Stopping its gradient follows the envelope theorem and avoids
    # back-propagating through 48 bisection iterations.
    solved_log_eta = jax.lax.stop_gradient(high)
    logits, kl = candidate(solved_log_eta)
    no_signal = jnp.max(values, axis=-1) - jnp.min(values, axis=-1) < 1.0e-7
    inactive = no_signal | (delta <= 0.0)
    final_logits = jnp.where(inactive[..., None], base, logits)
    final_kl = jnp.where(inactive, 0.0, kl)
    eta = jnp.where(inactive, jnp.inf, jnp.exp(solved_log_eta))
    return final_logits, final_kl, eta


def project_policy_logits(
    reference_logits: Any,
    candidate_logits: Any,
    *,
    kl_budget: float,
    iterations: int = 48,
) -> tuple[Any, Any, Any]:
    """Project arbitrary candidate logits into the reference forward-KL ball.

    The exponential interpolation

    ``pi_alpha proportional pi_ref * exp(alpha * (log pi_cand-log pi_ref))``

    traces the e-geodesic from the immutable reference to the candidate.  A
    deterministic bisection chooses the largest ``alpha in [0,1]`` whose
    ``KL(pi_alpha || pi_ref)`` does not exceed the registered budget.
    """

    import jax
    import jax.nn
    import jax.numpy as jnp

    reference = jnp.asarray(reference_logits, dtype=jnp.float32)
    candidate = jnp.asarray(candidate_logits, dtype=jnp.float32)
    delta = float(kl_budget)
    if delta < 0.0:
        raise ValueError("KL budget must be non-negative.")
    reference_logp = jax.nn.log_softmax(reference, axis=-1)
    candidate_logp = jax.nn.log_softmax(candidate, axis=-1)
    direction = candidate_logp - reference_logp

    def at(alpha: Any) -> tuple[Any, Any]:
        logits = reference_logp + alpha[..., None] * direction
        logp = jax.nn.log_softmax(logits, axis=-1)
        probability = jnp.exp(logp)
        kl = jnp.sum(probability * (logp - reference_logp), axis=-1)
        return logits, kl

    candidate_kl = categorical_kl_from_logits(candidate, reference)
    lead = candidate_kl.shape
    low = jnp.zeros(lead, dtype=jnp.float32)
    high = jnp.ones(lead, dtype=jnp.float32)

    def one(_: int, bounds: tuple[Any, Any]) -> tuple[Any, Any]:
        lo, hi = bounds
        middle = 0.5 * (lo + hi)
        _, kl = at(middle)
        return jnp.where(kl <= delta, middle, lo), jnp.where(kl <= delta, hi, middle)

    low, high = jax.lax.fori_loop(0, int(iterations), one, (low, high))
    alpha = jnp.where(candidate_kl <= delta, 1.0, low)
    alpha = jax.lax.stop_gradient(alpha)
    logits, kl = at(alpha)
    inactive = delta <= 0.0
    logits = jnp.where(inactive, reference, logits)
    kl = jnp.where(inactive, 0.0, kl)
    alpha = jnp.where(inactive, 0.0, alpha)
    # Logits are defined only up to an additive constant.  Preserve the
    # candidate's convention when it is already feasible so diagnostic modes
    # and zero-residual identity are exact, not merely distribution-equivalent.
    logits = jnp.where((candidate_kl <= delta)[..., None], candidate, logits)
    return logits, kl, alpha




MIRROR_UNCERTAINTY_PENALTY = 1.0
"""One ensemble standard deviation, as a registered constant rather than a knob.

The natural reading of ``beta = 1`` is "improve against the advantage you can
show at one sigma of ensemble disagreement".  It is deliberately not a config
key: a tunable value here would recreate the weighted-objective search the
refactor removed, and there is no held-out quantity that could tune it without
reading the confirmatory panel.
"""


def robust_mirror_policy_logits(
    base_logits: Any,
    advantage_mean: Any,
    advantage_dispersion: Any,
    *,
    kl_budget: float,
    uncertainty_penalty: float,
    iterations: int = 48,
) -> tuple[Any, Any, Any]:
    """Move as far as the evidence supports, never further than the budget.

    ``mirror_policy_logits`` solves for the temperature that makes the KL
    *equal* the budget, so it is invariant to positive rescaling of the
    advantage: a contrast of 1e-4 and a contrast of 10 produce the same policy.
    The previous version of this function shrank the advantage by its
    dispersion and then handed the result to that same solver, which promptly
    rescaled the shrinkage away.  Measured on a trained deployment, the two
    solvers spent bit-identical KL in **100%** of states -- 0.04000 both -- even
    though the conservative bound had zeroed 16% of the advantages.  The
    deployment therefore pushed a competent base policy a full 0.04 KL every
    step on an advantage whose median magnitude was 0.014 and whose measured
    resolvable-pair fraction was 5-15%.

    The temperature now has a floor set by the advantage's own uncertainty:

        eta = max(eta_budget, beta * dispersion)

    At ``eta = beta * sigma`` the exponent is the advantage's z-score divided
    by ``beta``, so the policy moves in proportion to how many standard
    deviations of evidence there are.  Where the advantage is well determined
    the budget temperature is the larger of the two and behaviour is unchanged;
    where it is noise, the step shrinks toward the base policy on its own.  The
    KL constraint still holds -- the realised KL can only fall below the
    budget, never rise above it.
    """

    import jax
    import jax.nn
    import jax.numpy as jnp

    base = jnp.asarray(base_logits, dtype=jnp.float32)
    mean = jnp.asarray(advantage_mean, dtype=jnp.float32)
    dispersion = jnp.asarray(advantage_dispersion, dtype=jnp.float32)
    beta = jnp.asarray(uncertainty_penalty, dtype=jnp.float32)

    # The budget temperature, from the unmodified dual solve.
    _, _, eta_budget = mirror_policy_logits(
        base, mean, kl_budget=kl_budget, iterations=iterations
    )

    # One evidence scale per state: the advantage is a vector, the temperature
    # is a scalar, and a per-action temperature would not be a temperature.
    evidence = beta * jnp.mean(dispersion, axis=-1)
    eta = jnp.maximum(eta_budget, evidence)

    centred = mean - jnp.mean(mean, axis=-1, keepdims=True)
    base_logp = jax.nn.log_softmax(base, axis=-1)
    logits = base_logp + centred / jnp.maximum(eta, 1.0e-30)[..., None]
    logp = jax.nn.log_softmax(logits, axis=-1)
    kl = jnp.sum(jnp.exp(logp) * (logp - base_logp), axis=-1)

    no_signal = jnp.max(centred, axis=-1) - jnp.min(centred, axis=-1) < 1.0e-7
    inactive = no_signal | (float(kl_budget) <= 0.0) | ~jnp.isfinite(eta_budget)
    return (
        jnp.where(inactive[..., None], base, logits),
        jnp.where(inactive, 0.0, kl),
        jnp.where(inactive, jnp.inf, eta),
    )


__all__ = [
    "categorical_kl_from_logits",
    "mirror_policy_logits",
    "project_policy_logits",
    "robust_mirror_policy_logits",
]
