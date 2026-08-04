"""Permutation-invariant diagnostics for exchangeable response regimes."""

from __future__ import annotations

from itertools import permutations
from typing import Any, Mapping, Sequence

import numpy as np


def _matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 3:
        # Shared panel tensor [anchor, component, action] -> component rows.
        matrix = np.moveaxis(matrix, 1, 0).reshape((matrix.shape[1], -1))
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("Component signatures must be a K x feature matrix.")
    if matrix.shape[0] > 8 or not np.all(np.isfinite(matrix)):
        raise ValueError("Component signatures must be finite with 2 <= K <= 8.")
    return matrix


def best_permutation_alignment(
    reference: Any, candidate: Any
) -> tuple[tuple[int, ...], float]:
    """Return the minimum-MSE permutation without assigning component names."""

    left = _matrix(reference)
    right = _matrix(candidate)
    if left.shape != right.shape:
        raise ValueError("Aligned component signature shapes differ.")
    best_order: tuple[int, ...] | None = None
    best_error = float("inf")
    for order in permutations(range(left.shape[0])):
        error = float(np.mean(np.square(left - right[np.asarray(order)])))
        if error < best_error:
            best_error = error
            best_order = tuple(int(index) for index in order)
    if best_order is None:
        raise RuntimeError("No component permutation was evaluated.")
    return best_order, best_error


def permutation_aligned_component_stability(
    signatures_by_seed: Mapping[int, Any] | Sequence[Any],
) -> Mapping[str, Any]:
    """Align every seed to a deterministic medoid and report stability.

    The reference is the seed with the smallest total pairwise aligned MSE,
    avoiding an arbitrary privileged seed.  The score ``1/(1+RMSE)`` is
    bounded in (0,1] and is descriptive only.
    """

    if isinstance(signatures_by_seed, Mapping):
        ordered = sorted((int(seed), _matrix(value)) for seed, value in signatures_by_seed.items())
    else:
        ordered = [(index, _matrix(value)) for index, value in enumerate(signatures_by_seed)]
    if len(ordered) < 2:
        raise ValueError("Cross-seed stability requires at least two seeds.")
    shapes = {matrix.shape for _, matrix in ordered}
    if len(shapes) != 1:
        raise ValueError("Cross-seed component signature shapes differ.")

    pair_errors = np.zeros((len(ordered), len(ordered)), dtype=np.float64)
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            _, error = best_permutation_alignment(
                ordered[left][1], ordered[right][1]
            )
            pair_errors[left, right] = pair_errors[right, left] = error
    reference_index = int(np.argmin(np.sum(pair_errors, axis=1)))
    reference_seed, reference = ordered[reference_index]
    alignments = []
    aligned = []
    for seed, matrix in ordered:
        order, error = best_permutation_alignment(reference, matrix)
        aligned_matrix = matrix[np.asarray(order)]
        aligned.append(aligned_matrix)
        alignments.append(
            {
                "seed_index": seed,
                "permutation_to_reference": list(order),
                "mean_squared_error": error,
            }
        )
    aligned_stack = np.stack(aligned, axis=0)
    rmse = float(np.sqrt(np.mean(np.square(aligned_stack - reference[None]))))
    pair_values = pair_errors[np.triu_indices(len(ordered), k=1)]
    return {
        "reference_seed_index": reference_seed,
        "seed_count": len(ordered),
        "permutation_aligned_rmse": rmse,
        "permutation_aligned_stability": 1.0 / (1.0 + rmse),
        "mean_pairwise_aligned_mse": float(np.mean(pair_values)),
        "alignments": alignments,
    }


def component_intervention_summary(
    *,
    action_signatures: Any,
    policy_logits: Any,
    protocol_probabilities: Any,
    diagnostic_panel_ids: Any | None = None,
    diagnostic_panel_state_hashes: Any | None = None,
    component_embeddings: Any | None = None,
) -> Mapping[str, Any]:
    """Aggregate fresh-anchor one-hot intervention results for one run."""

    signatures = np.asarray(action_signatures, dtype=np.float64)
    logits = np.asarray(policy_logits, dtype=np.float64)
    posterior = np.asarray(protocol_probabilities, dtype=np.float64)
    if signatures.ndim != 3 or logits.shape != signatures.shape:
        raise ValueError("Intervention tensors must have shape anchor x K x action.")
    if posterior.shape != signatures.shape[:2]:
        raise ValueError("Intervention posterior axes differ.")
    if not all(np.all(np.isfinite(value)) for value in (signatures, logits, posterior)):
        raise ValueError("Intervention diagnostics contain non-finite values.")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probability = np.exp(shifted)
    probability /= np.sum(probability, axis=-1, keepdims=True)
    count = signatures.shape[1]
    pair_mask = ~np.eye(count, dtype=bool)
    signature_distance = np.sqrt(
        np.mean(
            np.square(signatures[:, :, None, :] - signatures[:, None, :, :]),
            axis=-1,
        )
    )
    actor_tv = 0.5 * np.sum(
        np.abs(probability[:, :, None, :] - probability[:, None, :, :]), axis=-1
    )
    top = np.argmax(probability, axis=-1)
    top_disagreement = top[:, :, None] != top[:, None, :]
    entropy = -np.sum(
        posterior * np.log(np.maximum(posterior, 1.0e-12)), axis=-1
    )
    utilization = np.mean(posterior, axis=0)
    usage_entropy = float(
        -np.sum(utilization * np.log(np.maximum(utilization, 1.0e-12)))
    )
    panel_ids = (
        [f"panel-{index}" for index in range(signatures.shape[0])]
        if diagnostic_panel_ids is None
        else [str(value) for value in np.asarray(diagnostic_panel_ids).reshape((-1,))]
    )
    state_hashes = (
        panel_ids
        if diagnostic_panel_state_hashes is None
        else [
            str(value)
            for value in np.asarray(diagnostic_panel_state_hashes).reshape((-1,))
        ]
    )
    if len(panel_ids) != signatures.shape[0] or len(state_hashes) != signatures.shape[0]:
        raise ValueError("Shared component diagnostic panel identities differ.")
    embedding = (
        np.ones((count, 1), dtype=np.float64)
        if component_embeddings is None
        else np.asarray(component_embeddings, dtype=np.float64)
    )
    if embedding.ndim != 2 or embedding.shape[0] != count or not np.all(
        np.isfinite(embedding)
    ):
        raise ValueError("Component embedding diagnostic shape differs.")
    embedding_norm = np.linalg.norm(embedding, axis=-1)
    return {
        "component_action_signatures": np.mean(signatures, axis=0).tolist(),
        "component_action_signatures_by_anchor": signatures.tolist(),
        "diagnostic_panel_ids": panel_ids,
        "diagnostic_panel_state_hashes": state_hashes,
        "component_one_hot_actor_probabilities": np.mean(probability, axis=0).tolist(),
        "component_utilization": utilization.tolist(),
        "mean_effective_component_count": float(np.mean(np.exp(entropy))),
        "minimum_component_utilization": float(
            np.min(np.mean(posterior, axis=0))
        ),
        "posterior_high_confidence_fraction": float(
            np.mean(np.max(posterior, axis=-1) > 0.98)
        ),
        "dominant_component_fraction": float(np.max(utilization)),
        "component_usage_entropy": usage_entropy,
        "component_embedding_norms": embedding_norm.tolist(),
        "component_embedding_minimum_norm": float(np.min(embedding_norm)),
        "component_embedding_maximum_norm": float(np.max(embedding_norm)),
        "mean_pairwise_action_signature_divergence": float(
            np.mean(signature_distance[:, pair_mask])
        ),
        "mean_pairwise_one_hot_actor_tv": float(np.mean(actor_tv[:, pair_mask])),
        "one_hot_top_action_disagreement_fraction": float(
            np.mean(top_disagreement[:, pair_mask])
        ),
    }


def validate_component_diagnostic_values(
    payload: Mapping[str, Any], *, component_count: int, action_count: int = 6
) -> None:
    """Fail closed on the complete registered component diagnostic surface."""

    count = int(component_count)
    signatures = np.asarray(payload.get("component_action_signatures"), dtype=np.float64)
    actor = np.asarray(
        payload.get("component_one_hot_actor_probabilities"), dtype=np.float64
    )
    utilization = np.asarray(payload.get("component_utilization"), dtype=np.float64)
    panel_signatures = np.asarray(
        payload.get("component_action_signatures_by_anchor"), dtype=np.float64
    )
    panel_ids = payload.get("diagnostic_panel_ids")
    panel_hashes = payload.get("diagnostic_panel_state_hashes")
    embedding_norms = np.asarray(
        payload.get("component_embedding_norms"), dtype=np.float64
    )
    if signatures.shape != (count, int(action_count)) or actor.shape != signatures.shape:
        raise ValueError("Component action diagnostic shapes differ.")
    if utilization.shape != (count,):
        raise ValueError("Component utilization shape differs.")
    if (
        panel_signatures.ndim != 3
        or panel_signatures.shape[1:] != (count, int(action_count))
        or not isinstance(panel_ids, list)
        or not isinstance(panel_hashes, list)
        or len(panel_ids) != panel_signatures.shape[0]
        or len(panel_hashes) != panel_signatures.shape[0]
        or embedding_norms.shape != (count,)
    ):
        raise ValueError("Shared component diagnostic panel differs.")
    if not all(
        np.all(np.isfinite(value))
        for value in (
            signatures,
            actor,
            utilization,
            panel_signatures,
            embedding_norms,
        )
    ):
        raise ValueError("Component diagnostic arrays contain non-finite values.")
    if (
        np.any(actor < 0.0)
        or not np.allclose(np.sum(actor, axis=-1), 1.0, atol=1.0e-5)
        or np.any(utilization < 0.0)
        or not np.isclose(np.sum(utilization), 1.0, atol=1.0e-5)
    ):
        raise ValueError("Component diagnostic probabilities are invalid.")
    nonnegative = (
        "mean_pairwise_response_divergence_all_actions",
        "mean_pairwise_action_signature_divergence",
        "mean_pairwise_one_hot_actor_tv",
    )
    for name in nonnegative:
        value = float(payload.get(name, float("nan")))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Component diagnostic {name} is invalid.")
    effective = float(payload.get("mean_effective_component_count", float("nan")))
    minimum = float(payload.get("minimum_component_utilization", float("nan")))
    high_confidence = float(
        payload.get("posterior_high_confidence_fraction", float("nan"))
    )
    dominant = float(payload.get("dominant_component_fraction", float("nan")))
    usage_entropy = float(payload.get("component_usage_entropy", float("nan")))
    disagreement = float(
        payload.get("one_hot_top_action_disagreement_fraction", float("nan"))
    )
    actor_tv = float(payload["mean_pairwise_one_hot_actor_tv"])
    minimum_norm = float(payload.get("component_embedding_minimum_norm", float("nan")))
    maximum_norm = float(payload.get("component_embedding_maximum_norm", float("nan")))
    if not (
        1.0 <= effective <= count + 1.0e-5
        and 0.0 <= minimum <= 1.0
        and 0.0 <= high_confidence <= 1.0
        and 0.0 <= dominant <= 1.0
        and 0.0 <= usage_entropy <= np.log(count) + 1.0e-5
        and 0.0 <= disagreement <= 1.0
        and 0.0 <= actor_tv <= 1.0 + 1.0e-5
        and minimum_norm > 0.0
        and maximum_norm >= minimum_norm
        and np.isclose(minimum_norm, np.min(embedding_norms), atol=1.0e-6)
        and np.isclose(maximum_norm, np.max(embedding_norms), atol=1.0e-6)
    ):
        raise ValueError("Component diagnostic scalar ranges differ.")


__all__ = [
    "best_permutation_alignment",
    "component_intervention_summary",
    "permutation_aligned_component_stability",
    "validate_component_diagnostic_values",
]
