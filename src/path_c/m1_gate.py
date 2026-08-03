"""Extended M1 gate (METHOD_SPEC §6).

The action-ranking Spearman check is executed at both context paths:
(i) the posterior-mean context and (ii) every bootstrap history-encoder
member context (B=3), each evaluated against the shared CRN all-action
signature of the same anchor state.  The gate passes only if *both* paths
reach Spearman >= 0.8 on >= 90% of anchor states; a failing particle-era
substitute cannot unlock a formal run.

The readout is report-only; nothing here shapes rewards or gradients.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

M1_SPEARMAN_THRESHOLD = 0.8
M1_MINIMUM_ANCHOR_FRACTION = 0.9


class M1GateResult(NamedTuple):
    passed: Any
    path_passing_fractions: Any
    path_mean_spearman: Any
    spearman_threshold: Any
    minimum_anchor_fraction: Any


def spearman_rank_correlation(predicted: Any, reference: Any) -> Any:
    """Spearman rho as Pearson correlation of rank orders (no tie handling
    required: action values and signatures are continuous-valued)."""

    import jax.numpy as jnp

    left = jnp.asarray(predicted, dtype=jnp.float32)
    right = jnp.asarray(reference, dtype=jnp.float32)
    if left.shape != right.shape:
        raise ValueError("Spearman inputs must share shape.")
    left_ranks = jnp.argsort(jnp.argsort(left, axis=-1), axis=-1).astype(jnp.float32)
    right_ranks = jnp.argsort(jnp.argsort(right, axis=-1), axis=-1).astype(jnp.float32)
    left_centered = left_ranks - jnp.mean(left_ranks, axis=-1, keepdims=True)
    right_centered = right_ranks - jnp.mean(right_ranks, axis=-1, keepdims=True)
    numerator = jnp.sum(left_centered * right_centered, axis=-1)
    denominator = jnp.sqrt(
        jnp.sum(jnp.square(left_centered), axis=-1)
        * jnp.sum(jnp.square(right_centered), axis=-1)
        + 1.0e-12
    )
    return numerator / denominator


def _path_passing_fraction(
    spearman_values: Any,
    *,
    threshold: float,
    valid_mask: Any | None = None,
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(spearman_values, dtype=jnp.float32)
    mask = (
        jnp.asarray(valid_mask, dtype=jnp.float32)
        if valid_mask is not None
        else jnp.ones(values.shape, dtype=jnp.float32)
    )
    passing = ((values >= float(threshold)) * mask).astype(jnp.float32)
    total = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(passing) / total, jnp.sum(values * mask) / total


def m1_anchor_gate(
    predicted_action_values: Any,
    *,
    crn_signatures: Any,
    valid_mask: Any | None = None,
    spearman_threshold: float = M1_SPEARMAN_THRESHOLD,
    minimum_anchor_fraction: float = M1_MINIMUM_ANCHOR_FRACTION,
) -> tuple[Any, Any]:
    """Single-path gate: per-anchor-state Spearman of the action ranking
    between the context's Q values and the shared CRN signature."""

    rho = spearman_rank_correlation(predicted_action_values, crn_signatures)
    fraction, mean_rho = _path_passing_fraction(
        rho, threshold=spearman_threshold, valid_mask=valid_mask
    )
    return fraction >= float(minimum_anchor_fraction), fraction


def evaluate_extended_m1_gate(
    *,
    posterior_mean_action_values: Any,
    member_action_values: Any,
    crn_signatures: Any,
    valid_mask: Any | None = None,
    spearman_threshold: float = M1_SPEARMAN_THRESHOLD,
    minimum_anchor_fraction: float = M1_MINIMUM_ANCHOR_FRACTION,
) -> M1GateResult:
    """§6 extended gate: the posterior-mean path and *each* bootstrap member
    path must independently reach the passing fraction; pooling across
    members is forbidden."""

    import jax.numpy as jnp

    signatures = jnp.asarray(crn_signatures, dtype=jnp.float32)
    posterior_rho = spearman_rank_correlation(
        posterior_mean_action_values, signatures
    )
    posterior_fraction, posterior_mean_rho = _path_passing_fraction(
        posterior_rho, threshold=spearman_threshold, valid_mask=valid_mask
    )
    posterior_pass = posterior_fraction >= float(minimum_anchor_fraction)

    member_values = jnp.asarray(member_action_values, dtype=jnp.float32)
    member_signatures = jnp.broadcast_to(signatures[None], member_values.shape)
    member_rho = spearman_rank_correlation(member_values, member_signatures)
    if valid_mask is not None:
        member_mask = jnp.broadcast_to(
            jnp.asarray(valid_mask, dtype=jnp.float32)[None], member_values.shape[:-1]
        )
    else:
        member_mask = jnp.ones(member_values.shape[:-1], dtype=jnp.float32)
    passing = (member_rho >= float(spearman_threshold)).astype(jnp.float32)
    totals = jnp.maximum(jnp.sum(member_mask, axis=-1), 1.0)
    member_fractions = jnp.sum(passing * member_mask, axis=-1) / totals
    member_mean_rho = jnp.sum(member_rho * member_mask, axis=-1) / totals
    member_pass = jnp.all(member_fractions >= float(minimum_anchor_fraction))
    return M1GateResult(
        passed=jnp.asarray(posterior_pass) & jnp.asarray(member_pass),
        path_passing_fractions=jnp.concatenate(
            (jnp.asarray(posterior_fraction)[None], member_fractions)
        ),
        path_mean_spearman=jnp.concatenate(
            (jnp.asarray(posterior_mean_rho)[None], member_mean_rho)
        ),
        spearman_threshold=jnp.asarray(spearman_threshold, dtype=jnp.float32),
        minimum_anchor_fraction=jnp.asarray(
            minimum_anchor_fraction, dtype=jnp.float32
        ),
    )


def evaluate_m1_gate_on_anchor_batch(
    *,
    model: Any,
    params: Any,
    bootstrap_ensemble: Any,
    bootstrap_params: Any,
    anchors: Any,
    spearman_threshold: float = M1_SPEARMAN_THRESHOLD,
    minimum_anchor_fraction: float = M1_MINIMUM_ANCHOR_FRACTION,
) -> M1GateResult:
    """§6 gate over one collected anchor batch.

    Path (i) uses the posterior-mean context from the deployable model;
    path (ii) uses each B=3 bootstrap protocol-encoder member re-filtering
    the *same* real anchor history.  All hypotheses share the CRN all-action
    signature of their anchor state as the ranking reference, so every
    hypothesis carries a consistent continuation label.
    """

    import jax.numpy as jnp

    from .decision_geometry import centered_action_values
    from .protocol_encoder import initial_protocol_carry

    count = int(jnp.asarray(anchors.anchor_ids).shape[0])
    dropped = jnp.zeros((count,), dtype=jnp.bool_)
    _, posterior = model.apply(
        {"params": params},
        anchors.policy_states,
        anchors.observations,
        dropped,
        method=model.step,
    )
    posterior_summary = jnp.concatenate(
        (posterior.capability, posterior.protocol_embedding), axis=-1
    )
    posterior_values = model.apply(
        {"params": params},
        posterior.task_features,
        posterior_summary,
        method=model.action_values_from_features_and_context,
    )
    crn_signatures = centered_action_values(anchors.fit_returns_by_action)
    valid_mask = jnp.any(jnp.asarray(anchors.action_mask), axis=-1)

    component_matrix = params["protocol_component_embeddings"]["embedding"]
    protocol_hidden_dim = _protocol_hidden_dim(anchors.policy_states)
    member_count = int(getattr(bootstrap_ensemble, "member_count", 3))
    carries = tuple(
        initial_protocol_carry(count, protocol_hidden_dim)
        for _ in range(member_count)
    )
    previous_probabilities = jnp.full(
        (count, int(component_matrix.shape[0])),
        1.0 / float(component_matrix.shape[0]),
        dtype=jnp.float32,
    )
    inputs = (
        anchors.policy_states.previous_observation,
        anchors.observations,
        anchors.policy_states.previous_action,
        anchors.policy_states.episode_start,
        previous_probabilities,
    )
    _, member_probabilities = bootstrap_ensemble.apply(
        {"params": bootstrap_params}, carries, inputs, False
    )
    member_embeddings = jnp.einsum(
        "bsk,kd->bsd", member_probabilities, component_matrix
    )
    member_summaries = jnp.concatenate(
        (
            jnp.broadcast_to(
                posterior.capability[None], member_embeddings.shape
            ),
            member_embeddings,
        ),
        axis=-1,
    )
    member_values = model.apply(
        {"params": params},
        jnp.broadcast_to(posterior.task_features[None], member_summaries.shape[:-1] + posterior.task_features.shape[-1:]),
        member_summaries,
        method=model.action_values_from_features_and_context,
    )
    return evaluate_extended_m1_gate(
        posterior_mean_action_values=posterior_values,
        member_action_values=member_values,
        crn_signatures=crn_signatures,
        valid_mask=valid_mask,
        spearman_threshold=spearman_threshold,
        minimum_anchor_fraction=minimum_anchor_fraction,
    )


def _protocol_hidden_dim(policy_states: Any) -> int:
    """``policy_states.protocol_carry`` stores (gru_hidden, previous_probs)."""

    carry, _ = policy_states.protocol_carry
    return int(carry.shape[-1])


def m1_gate_report(result: M1GateResult) -> Mapping[str, Any]:
    """JSON-ready gate readout for run ledgers."""

    import numpy as np

    return {
        "m1_gate_passed": bool(np.asarray(result.passed)),
        "m1_path_passing_fractions": [
            float(value) for value in np.asarray(result.path_passing_fractions)
        ],
        "m1_path_mean_spearman": [
            float(value) for value in np.asarray(result.path_mean_spearman)
        ],
        "m1_spearman_threshold": float(np.asarray(result.spearman_threshold)),
        "m1_minimum_anchor_fraction": float(
            np.asarray(result.minimum_anchor_fraction)
        ),
    }


__all__ = [
    "M1_MINIMUM_ANCHOR_FRACTION",
    "M1_SPEARMAN_THRESHOLD",
    "M1GateResult",
    "evaluate_extended_m1_gate",
    "evaluate_m1_gate_on_anchor_batch",
    "m1_anchor_gate",
    "m1_gate_report",
    "spearman_rank_correlation",
]
