"""Permutation-invariant diagnostics for exchangeable response regimes."""

from __future__ import annotations

from itertools import permutations
from typing import Any, Mapping, Sequence

import numpy as np


def _matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
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
    return {
        "component_action_signatures": np.mean(signatures, axis=0).tolist(),
        "component_one_hot_actor_probabilities": np.mean(probability, axis=0).tolist(),
        "component_utilization": np.mean(posterior, axis=0).tolist(),
        "mean_effective_component_count": float(np.mean(np.exp(entropy))),
        "minimum_component_utilization": float(
            np.min(np.mean(posterior, axis=0))
        ),
        "component_collapse_fraction": float(
            np.mean(np.max(posterior, axis=-1) > 0.98)
        ),
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
    if signatures.shape != (count, int(action_count)) or actor.shape != signatures.shape:
        raise ValueError("Component action diagnostic shapes differ.")
    if utilization.shape != (count,):
        raise ValueError("Component utilization shape differs.")
    if not all(np.all(np.isfinite(value)) for value in (signatures, actor, utilization)):
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
    collapse = float(payload.get("component_collapse_fraction", float("nan")))
    disagreement = float(
        payload.get("one_hot_top_action_disagreement_fraction", float("nan"))
    )
    actor_tv = float(payload["mean_pairwise_one_hot_actor_tv"])
    if not (
        1.0 <= effective <= count + 1.0e-5
        and 0.0 <= minimum <= 1.0
        and 0.0 <= collapse <= 1.0
        and 0.0 <= disagreement <= 1.0
        and 0.0 <= actor_tv <= 1.0 + 1.0e-5
    ):
        raise ValueError("Component diagnostic scalar ranges differ.")


__all__ = [
    "best_permutation_alignment",
    "component_intervention_summary",
    "permutation_aligned_component_stability",
    "validate_component_diagnostic_values",
]
