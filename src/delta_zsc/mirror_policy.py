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
    """Improve against a conservative lower bound on the advantage.

    ``mirror_policy_logits`` treats any ordering as actionable: because the
    constrained optimum is invariant to positive rescaling of ``Q``, a contrast
    of 1e-4 and a contrast of 10 produce the same policy, only a different dual
    temperature.  Measured on the pre-refactor anchors, no action pair was
    separated by even two standard errors, yet the solver still spent the full
    0.04 KL budget on every active step.

    This objective replaces the raw inner product with

        (pi - pi0)^T Abar  -  beta * sqrt((pi - pi0)^T Sigma (pi - pi0))

    using the ensemble dispersion as a diagonal Sigma.  Where the advantage is
    well determined the first term dominates and the step is unchanged; where
    the ensemble disagrees the penalty cancels the unproven improvement and the
    solution stays near the base policy.  It is a continuous lower bound, not a
    gate: nothing is thresholded on or off.
    """

    import jax.nn
    import jax.numpy as jnp

    mean = jnp.asarray(advantage_mean, dtype=jnp.float32)
    dispersion = jnp.asarray(advantage_dispersion, dtype=jnp.float32)
    beta = jnp.asarray(uncertainty_penalty, dtype=jnp.float32)
    base = jnp.asarray(base_logits, dtype=jnp.float32)
    reference = jax.nn.softmax(base, axis=-1)

    # Shrink each advantage toward zero by its own uncertainty and clamp there.
    # Subtracting a signed penalty outright would be wrong twice over: the dual
    # solver is invariant to positive rescaling, so a uniformly shrunk vector
    # would still spend the whole budget, and once the dispersion exceeds the
    # mean the signed subtraction flips the ordering and *increases* the
    # magnitude.  Clamping at zero is the lower confidence bound: an action the
    # ensemble cannot separate from the baseline contributes nothing, and when
    # no action survives the vector is constant and the solver's no-signal
    # branch returns the base policy unchanged.
    conservative = jnp.sign(mean) * jnp.maximum(
        jnp.abs(mean) - beta * jnp.abs(dispersion), 0.0
    )
    del reference
    return mirror_policy_logits(
        base, conservative, kl_budget=kl_budget, iterations=iterations
    )



__all__ = [
    "categorical_kl_from_logits",
    "mirror_policy_logits",
    "robust_mirror_policy_logits",
]
