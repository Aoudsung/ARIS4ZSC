"""Pairwise CRN action contrasts and their measurement precision.

The previous anchor target estimated six action means separately and then let a
Gaussian mixture explain the vector.  Measured on `postfix_matched/seed0`, that
target could not separate the best action from the second best on a single
anchor: the median best-second margin was 0.0004 against a median replica
standard error of 0.0031, and no anchor reached two standard errors.  Every
anchor row also contained exact ties, so `argmax` was assigning a winner by
index order and `argsort(argsort(...))` was assigning ranks to equal values.

Differencing *within* a replica instead uses the common random numbers the
anchor already pays for: the shared partner, environment noise and continuation
draw cancel, leaving the contrast the mirror step actually consumes.  The
standard error of that difference is then an honest measure of whether the pair
is resolvable at all, and it enters the loss as a fixed weight rather than as a
variance the model could shrink.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class PairwiseContrast(NamedTuple):
    """Same-replica action differences with their measurement precision.

    ``mean[..., a, b]`` estimates ``Q(a) - Q(b)``; ``standard_error`` is the
    standard error of that difference; ``valid`` marks pairs where both actions
    were legal and at least two replicas contributed.
    """

    mean: Any
    standard_error: Any
    valid: Any

    def precision_weights(self) -> Any:
        """Shrunk inverse-variance weights, zero on invalid pairs.

        The weight is ``1 / (variance + pooled)``, where ``pooled`` is the mean
        variance over the batch's valid pairs.  Plain ``1 / variance`` is wrong
        here in a way that matters: under CRN a pair of actions whose
        continuations re-merge produces *identical* returns in every replica, so
        the sample variance is exactly zero and the pair would receive
        essentially unbounded weight -- on the first measured batch, 1e6 against
        1e3 for a genuinely resolvable pair.  The loss would then be dominated by
        states asserting that two actions are exactly equal, which is the one
        thing the decision head must not be confident about.

        Eight replicas cannot certify infinite precision.  Adding the pooled
        variance is the standard shrinkage of a noisy per-pair variance toward
        the batch estimate: it bounds the weight above by ``1 / pooled``, leaves
        well-measured pairs at ``1 / variance`` asymptotically, and introduces no
        constant that anyone could tune -- ``pooled`` is computed from the same
        measurement.

        Nothing is thresholded away.  Unresolvable pairs still train "these two
        actions are close", which is the honest content of the measurement; they
        just no longer shout it.
        """

        import jax.numpy as jnp

        valid = jnp.asarray(self.valid, dtype=jnp.bool_)
        variance = jnp.square(jnp.asarray(self.standard_error, dtype=jnp.float32))
        variance = jnp.where(valid, variance, 0.0)
        pooled = jnp.sum(variance) / jnp.maximum(
            jnp.sum(valid.astype(jnp.float32)), 1.0
        )
        weights = 1.0 / jnp.maximum(variance + pooled, 1.0e-12)
        return jnp.where(valid, weights, 0.0)


def pairwise_contrasts_from_replicas(
    replica_returns_by_action: Any,
    action_mask: Any,
) -> PairwiseContrast:
    """Difference actions inside each replica, then average across replicas.

    ``replica_returns_by_action`` is ``[..., action, replica]``: replica ``r``
    of action ``a`` and replica ``r`` of action ``b`` share the CRN draw, so
    their difference removes the shared noise before averaging.
    """

    import jax.numpy as jnp

    returns = jnp.asarray(replica_returns_by_action, dtype=jnp.float32)
    replicas = int(returns.shape[-1])
    if replicas < 2:
        raise ValueError("Pairwise contrast standard errors need >= 2 replicas.")

    # [..., a, b, replica]
    differences = returns[..., :, None, :] - returns[..., None, :, :]
    mean = jnp.mean(differences, axis=-1)
    variance = jnp.var(differences, axis=-1, ddof=1)
    standard_error = jnp.sqrt(jnp.maximum(variance, 0.0) / float(replicas))

    legal = jnp.asarray(action_mask, dtype=jnp.bool_)
    both_legal = legal[..., :, None] & legal[..., None, :]
    off_diagonal = ~jnp.eye(int(returns.shape[-2]), dtype=jnp.bool_)
    valid = both_legal & off_diagonal
    return PairwiseContrast(
        mean=jnp.where(valid, mean, 0.0),
        standard_error=jnp.where(valid, standard_error, jnp.inf),
        valid=valid,
    )


def contrast_regression_loss(
    predicted_advantage: Any,
    contrast: PairwiseContrast,
    *,
    delta: float = 1.0,
) -> tuple[Any, Any]:
    """Fit predicted action differences to the measured ones.

    Only differences enter, so any state-level offset the critic carries is
    irrelevant here -- exactly the invariance the mirror step has.  Returns the
    weighted mean loss and the total weight, so callers can normalise across
    batches with different numbers of resolvable pairs.
    """

    import jax.numpy as jnp

    from .belief_value import huber

    advantage = jnp.asarray(predicted_advantage, dtype=jnp.float32)
    predicted = advantage[..., :, None] - advantage[..., None, :]
    weights = contrast.precision_weights()
    residual = predicted - jnp.asarray(contrast.mean, dtype=jnp.float32)
    weighted = weights * huber(residual, delta=delta)
    total_weight = jnp.sum(weights)
    return jnp.sum(weighted) / jnp.maximum(total_weight, 1.0e-30), total_weight


def resolvable_pair_statistics(
    contrast: PairwiseContrast, *, sigma_multiple: float = 2.0
) -> dict[str, Any]:
    """Report how much of the measurement is actually usable.

    Keys are unprefixed so callers can name the channel they came from.
    ``resolvable_fraction`` is the share of legal pairs whose measured
    difference exceeds ``sigma_multiple`` standard errors.  On the pre-refactor
    anchors this was zero, which is why every ordering metric sat at chance.
    """

    import jax.numpy as jnp

    valid = jnp.asarray(contrast.valid, dtype=jnp.bool_)
    magnitude = jnp.abs(jnp.asarray(contrast.mean, dtype=jnp.float32))
    error = jnp.asarray(contrast.standard_error, dtype=jnp.float32)
    resolvable = valid & (magnitude > float(sigma_multiple) * error)
    count = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    return {
        "pairs": jnp.sum(valid.astype(jnp.float32)),
        "resolvable_fraction": jnp.sum(resolvable.astype(jnp.float32)) / count,
        "mean_absolute": jnp.sum(magnitude * valid) / count,
        "mean_standard_error": jnp.sum(
            jnp.where(valid, error, 0.0)
        ) / count,
    }


def pairwise_sign_agreement(
    predicted_advantage: Any, contrast: PairwiseContrast, *, sigma_multiple: float = 2.0
) -> Any:
    """Sign agreement restricted to resolvable pairs.

    Top-action agreement over six actions cannot separate "the head is wrong"
    from "the target could not tell"; restricting to pairs the measurement
    resolves removes that confound.
    """

    import jax.numpy as jnp

    advantage = jnp.asarray(predicted_advantage, dtype=jnp.float32)
    predicted = advantage[..., :, None] - advantage[..., None, :]
    valid = jnp.asarray(contrast.valid, dtype=jnp.bool_)
    magnitude = jnp.abs(jnp.asarray(contrast.mean, dtype=jnp.float32))
    resolvable = valid & (
        magnitude > float(sigma_multiple) * jnp.asarray(contrast.standard_error)
    )
    agree = jnp.sign(predicted) == jnp.sign(jnp.asarray(contrast.mean))
    count = jnp.maximum(jnp.sum(resolvable.astype(jnp.float32)), 1.0)
    return jnp.sum((agree & resolvable).astype(jnp.float32)) / count


def kendall_tau_b(predicted: Any, measured: Any, mask: Any) -> Any:
    """Tie-corrected rank correlation between predicted and measured actions.

    The ``b`` variant is the one that matters here: every anchor row of the
    previous target contained exact ties, and Spearman over
    ``argsort(argsort(...))`` was silently ranking equal values by index order.
    Ties enter the denominator instead of being broken arbitrarily.
    """

    import jax.numpy as jnp

    left = jnp.asarray(predicted, dtype=jnp.float32)
    right = jnp.asarray(measured, dtype=jnp.float32)
    legal = jnp.asarray(mask, dtype=jnp.bool_)
    pair = legal[..., :, None] & legal[..., None, :]
    upper = jnp.triu(jnp.ones(left.shape[-1:] * 2, dtype=jnp.bool_), k=1)
    pair = pair & upper

    left_sign = jnp.sign(left[..., :, None] - left[..., None, :])
    right_sign = jnp.sign(right[..., :, None] - right[..., None, :])
    concordant = jnp.sum((pair & (left_sign * right_sign > 0)).astype(jnp.float32), axis=(-2, -1))
    discordant = jnp.sum((pair & (left_sign * right_sign < 0)).astype(jnp.float32), axis=(-2, -1))
    total = jnp.sum(pair.astype(jnp.float32), axis=(-2, -1))
    left_ties = jnp.sum((pair & (left_sign == 0)).astype(jnp.float32), axis=(-2, -1))
    right_ties = jnp.sum((pair & (right_sign == 0)).astype(jnp.float32), axis=(-2, -1))
    denominator = jnp.sqrt(
        jnp.maximum(total - left_ties, 0.0) * jnp.maximum(total - right_ties, 0.0)
    )
    return jnp.where(denominator > 0.0, (concordant - discordant) / jnp.maximum(denominator, 1.0e-30), 0.0)


def exact_tie_fraction(returns_by_action: Any, mask: Any) -> Any:
    """Share of legal action pairs whose measured returns are bit-identical.

    On ``postfix_matched/seed0`` this was 1.0 for every row, which is why the
    ordering metrics could not have exceeded chance regardless of the head.
    """

    import jax.numpy as jnp

    values = jnp.asarray(returns_by_action, dtype=jnp.float32)
    legal = jnp.asarray(mask, dtype=jnp.bool_)
    pair = legal[..., :, None] & legal[..., None, :]
    pair = pair & jnp.triu(jnp.ones(values.shape[-1:] * 2, dtype=jnp.bool_), k=1)
    tied = pair & (values[..., :, None] == values[..., None, :])
    return jnp.sum(tied.astype(jnp.float32)) / jnp.maximum(
        jnp.sum(pair.astype(jnp.float32)), 1.0
    )


__all__ = [
    "PairwiseContrast",
    "exact_tie_fraction",
    "kendall_tau_b",
    "contrast_regression_loss",
    "pairwise_contrasts_from_replicas",
    "pairwise_sign_agreement",
    "resolvable_pair_statistics",
]
