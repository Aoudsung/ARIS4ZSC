"""Posterior-calibration scoring primitives (METHOD_SPEC §2.4).

The held-out calibration protocol scores the posterior-predictive response
model on run-disjoint, family-held-out episodes (root-seed offset 2000; no
training gradient ever touches this set).  Registered proper scoring rules:

* primary   -- posterior-predictive **log score** (per-step NLL, nats);
* secondary -- **Brier score** of the interaction_change event;
* coverage  -- empirical coverage of the 90% highest-probability categorical
  prediction sets, registered band [0.85, 0.95].

Pass thresholds (fail triggers SCIENTIFIC_SPEC Φ5): log score beats the
uniform-prior-mixture and no-history baselines by >= 0.02 nats/step each;
coverage in band; event Brier <= 0.9 x prior-baseline Brier.

This module deliberately contains no deployment gate or conformal safety
wrapper.  Posterior calibration is a read-only scientific diagnostic with its
own artifact identity.
"""

from __future__ import annotations

from typing import Any


def hungarian_component_alignment(
    component_signatures: Any, empirical_prototypes: Any
) -> tuple[Any, Any]:
    """Permutation-invariant alignment for visualization only.

    Rows are exchangeable model components and columns are empirical
    continuation-signature prototypes.  Calibration scores never use the
    returned indexes.
    """

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    components = np.asarray(component_signatures, dtype=np.float64)
    prototypes = np.asarray(empirical_prototypes, dtype=np.float64)
    if components.ndim < 2 or prototypes.ndim != components.ndim:
        raise ValueError("Component signatures and prototypes need matching ranks.")
    if components.shape[0] != prototypes.shape[0] or components.shape[1:] != prototypes.shape[1:]:
        raise ValueError("Alignment requires equal component/prototype shapes.")
    flat_components = components.reshape((components.shape[0], -1))
    flat_prototypes = prototypes.reshape((prototypes.shape[0], -1))
    cost = np.mean(
        np.square(flat_components[:, None, :] - flat_prototypes[None, :, :]),
        axis=-1,
    )
    component_indexes, prototype_indexes = linear_sum_assignment(cost)
    order = component_indexes[np.argsort(prototype_indexes)]
    return order.astype(np.int32), components[order]


def per_step_log_score(mixture_log_probabilities: Any, mask: Any | None = None) -> Any:
    """Primary proper score: mean per-step posterior-predictive NLL (nats)."""

    import jax.numpy as jnp

    nll = -jnp.asarray(mixture_log_probabilities, dtype=jnp.float32)
    if mask is None:
        return jnp.mean(nll)
    weight = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(weight * nll) / jnp.maximum(jnp.sum(weight), 1.0)


def event_brier_score(
    event_probabilities: Any, event_labels: Any, mask: Any | None = None
) -> Any:
    """Secondary proper score: Brier of interaction_change predictions."""

    import jax.numpy as jnp

    probability = jnp.asarray(event_probabilities, dtype=jnp.float32)
    label = jnp.asarray(event_labels, dtype=jnp.float32)
    items = jnp.square(probability - label)
    if mask is None:
        return jnp.mean(items)
    weight = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(weight * items) / jnp.maximum(jnp.sum(weight), 1.0)


def posterior_predictive_probabilities(
    posterior_probabilities: Any, component_logits: Any
) -> Any:
    """Marginal class probabilities under an exchangeable component posterior."""

    import jax
    import jax.numpy as jnp

    posterior = jnp.asarray(posterior_probabilities, dtype=jnp.float32)
    logits = jnp.asarray(component_logits, dtype=jnp.float32)
    if logits.shape[:-2] != posterior.shape[:-1] or logits.shape[-2] != posterior.shape[-1]:
        raise ValueError("Posterior and component-logit axes do not align.")
    return jnp.sum(
        posterior[..., :, None] * jax.nn.softmax(logits, axis=-1), axis=-2
    )


def uniform_prior_log_score(
    component_log_probabilities: Any, mask: Any | None = None
) -> Any:
    """Registered baseline 1: uniform prior mixture log score (§2.4)."""

    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    component = jnp.asarray(component_log_probabilities, dtype=jnp.float32)
    uniform_log = logsumexp(component, axis=-1) - jnp.log(component.shape[-1])
    return per_step_log_score(uniform_log, mask)


def highest_probability_set_coverage(
    component_probabilities: Any,
    labels: Any,
    *,
    credibility: float = 0.90,
    mask: Any | None = None,
) -> Any:
    """Empirical coverage of the smallest set with >= ``credibility`` mass.

    ``component_probabilities`` has shape (..., K) with the class axis last;
    the set is built greedily from the most probable classes (registered 90%
    highest-probability sets, §2.4 coverage indicator).
    """

    import jax.numpy as jnp

    probabilities = jnp.asarray(component_probabilities, dtype=jnp.float32)
    label = jnp.asarray(labels, dtype=jnp.int32)
    order = jnp.argsort(-probabilities, axis=-1)
    sorted_probs = jnp.take_along_axis(probabilities, order, axis=-1)
    cumulative = jnp.cumsum(sorted_probs, axis=-1)
    included = cumulative - sorted_probs < float(credibility)
    label_rank = jnp.argmax(order == label[..., None], axis=-1)
    in_set = jnp.take_along_axis(
        included, label_rank[..., None], axis=-1
    )[..., 0]
    if mask is None:
        return jnp.mean(in_set.astype(jnp.float32))
    weight = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(weight * in_set.astype(jnp.float32)) / jnp.maximum(
        jnp.sum(weight), 1.0
    )


def calibration_pass_decision(
    *,
    model_log_score: float,
    uniform_baseline_log_score: float,
    no_history_baseline_log_score: float,
    position_coverage: float,
    direction_coverage: float,
    event_brier: float,
    prior_baseline_brier: float,
    log_score_margin: float = 0.02,
    coverage_low: float = 0.85,
    coverage_high: float = 0.95,
    brier_ratio: float = 0.90,
) -> bool:
    """Registered §2.4 gate; failure triggers SCIENTIFIC_SPEC Φ5 downgrade."""

    return bool(
        model_log_score
        <= float(uniform_baseline_log_score) - float(log_score_margin)
        and model_log_score
        <= float(no_history_baseline_log_score) - float(log_score_margin)
        and float(coverage_low) <= float(position_coverage) <= float(coverage_high)
        and float(coverage_low) <= float(direction_coverage) <= float(coverage_high)
        and float(event_brier) <= float(brier_ratio) * float(prior_baseline_brier)
    )


__all__ = [
    "calibration_pass_decision",
    "event_brier_score",
    "highest_probability_set_coverage",
    "hungarian_component_alignment",
    "per_step_log_score",
    "posterior_predictive_probabilities",
    "uniform_prior_log_score",
]
