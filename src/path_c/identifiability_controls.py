"""Three identifiability controls (METHOD_SPEC §4).

All primitives are pure array functions: ``identifiability_app.py`` feeds them
precomputed task features, contexts and signatures so that every reading can
be reproduced offline from recorded checkpoints.

* §4.1 task-history leakage control -- structural isolation is enforced by the
  input wiring of ``model.py``; this module provides the audit readouts:
  ``task_repr_drift_under_shuffle`` and ``leakage_probe_accuracy``
  (registered target <= 0.55, chance 0.5; above -> SCIENTIFIC_SPEC Φ4).
* §4.2 history shuffle control -- ``history_shuffle_drop`` with paired
  bootstrap CI (9999 replications); drop <= 0 or CI crossing zero -> Φ2.
* §4.3 context swap control -- ``protocol_swap_causal_consistency``
  (registered target > 0.65, chance 0.5; <= 0.5 -> Φ3).
"""

from __future__ import annotations

from typing import Any

LEAKAGE_PROBE_EXCESS_THRESHOLD = 0.05
PROTOCOL_SWAP_CONSISTENCY_THRESHOLD = 0.65
BOOTSTRAP_REPLICATION = 9999
SHUFFLE_EPSILON_QUANTILE = 0.05


def task_representation_drift_under_shuffle(
    features_original: Any, features_shuffled: Any
) -> Any:
    """§4.1(a): mean relative drift ||Δx|| / ||x|| under partner-evidence shuffle."""

    import jax.numpy as jnp

    original = jnp.asarray(features_original, dtype=jnp.float32)
    shuffled = jnp.asarray(features_shuffled, dtype=jnp.float32)
    if original.shape != shuffled.shape:
        raise ValueError("Shuffled task features must match original shapes.")
    delta_norm = jnp.sqrt(jnp.sum(jnp.square(shuffled - original), axis=-1))
    base_norm = jnp.sqrt(jnp.sum(jnp.square(original), axis=-1))
    return jnp.mean(delta_norm / jnp.maximum(base_norm, 1.0e-8))


def balanced_linear_probe_accuracy(
    features: Any,
    run_ids: Any,
    *,
    fold_count: int = 5,
    ridge: float = 1.0e-3,
) -> Any:
    """§4.1(b): held-out balanced linear probe of partner run identity from x_t.

    Deterministic stratified k-fold; per-class balanced accuracy averaged over
    folds.  A task representation that leaks partner history scores far above
    the 0.5 chance level.
    """

    import numpy as np

    x = np.asarray(features, dtype=np.float64).reshape((len(features), -1))
    labels = np.asarray(run_ids)
    classes, encoded = np.unique(labels, return_inverse=True)
    class_count = classes.size
    if class_count < 2:
        raise ValueError("Leakage probe needs at least two partner runs.")
    fold_assignment = np.zeros_like(encoded)
    for class_index in range(class_count):
        indexes = np.flatnonzero(encoded == class_index)
        fold_assignment[indexes] = np.arange(indexes.size) % int(fold_count)
    predictions = np.zeros_like(encoded)
    one_hot = np.eye(class_count)[encoded]
    for fold in range(int(fold_count)):
        test = fold_assignment == fold
        train = ~test
        x_train, x_test = x[train], x[test]
        y_train = one_hot[train]
        gram = x_train.T @ x_train + float(ridge) * np.eye(x.shape[1])
        weights = np.linalg.solve(gram, x_train.T @ y_train)
        predictions[test] = np.argmax(x_test @ weights, axis=-1)
    balanced = []
    for class_index in range(class_count):
        member = encoded == class_index
        if np.any(member):
            balanced.append(float(np.mean(predictions[member] == encoded[member])))
    return float(np.mean(balanced))


def paired_bootstrap_drop(
    scores_normal: Any,
    scores_shuffled: Any,
    *,
    replications: int = BOOTSTRAP_REPLICATION,
    seed: int = 0,
) -> tuple[float, float, float]:
    """§4.2: drop = XP_normal - XP_shuffle with paired bootstrap CI."""

    import numpy as np

    normal = np.asarray(scores_normal, dtype=np.float64).reshape(-1)
    shuffled = np.asarray(scores_shuffled, dtype=np.float64).reshape(-1)
    if normal.shape != shuffled.shape or normal.size == 0:
        raise ValueError("Paired shuffle scores must be aligned and non-empty.")
    difference = normal - shuffled
    observed = float(np.mean(difference))
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, difference.size, size=(int(replications), difference.size))
    bootstrap_means = np.mean(difference[draws], axis=1)
    low = float(np.quantile(bootstrap_means, 0.005))
    high = float(np.quantile(bootstrap_means, 0.995))
    return observed, low, high


def history_shuffle_negative(drop: float, ci_low: float, ci_high: float) -> bool:
    """§4.2 verdict: drop <= 0 or CI crossing zero -> Φ2 negative."""

    return bool(drop <= 0.0 or (ci_low <= 0.0 <= ci_high))


def match_task_state_pairs(
    features_a: Any,
    features_b: Any,
    *,
    epsilon_quantile: float = SHUFFLE_EPSILON_QUANTILE,
) -> Any:
    """§4.3: for each row of A, nearest-neighbour index in B within epsilon.

    epsilon is the registered 5% quantile of the held-out pairwise distance
    distribution.  Returns (indexes_a, indexes_b) with unmatched rows dropped.
    """

    import numpy as np

    a = np.asarray(features_a, dtype=np.float64).reshape((len(features_a), -1))
    b = np.asarray(features_b, dtype=np.float64).reshape((len(features_b), -1))
    if a.size == 0 or b.size == 0:
        raise ValueError("Context-swap matching needs non-empty feature sets.")
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(a.shape[0]), nearest]
    pool = np.sort(nearest_distance)
    epsilon = float(np.quantile(pool, float(epsilon_quantile)))
    valid = nearest_distance <= epsilon
    return np.flatnonzero(valid).astype(np.int64), nearest[valid].astype(np.int64)


def match_different_partner_task_states(
    features: Any,
    partner_run_ids: Any,
    *,
    epsilon_quantile: float = SHUFFLE_EPSILON_QUANTILE,
) -> tuple[Any, Any, float, Any]:
    """Nearest task-state matches constrained to different partner runs.

    The epsilon is preregistered as the 5th percentile of the finite nearest
    different-run distances.  Returned source/donor indexes are therefore
    task matched without using partner family, latent component, or return
    labels.  The final array contains the retained source-to-donor distances.
    """

    import numpy as np

    values = np.asarray(features, dtype=np.float64).reshape((len(features), -1))
    run_ids = np.asarray(partner_run_ids)
    if values.shape[0] != run_ids.shape[0] or values.shape[0] < 2:
        raise ValueError("Task features and partner-run labels must align.")
    if np.unique(run_ids).size < 2:
        raise ValueError("Context swaps require at least two partner runs.")
    if not 0.0 < float(epsilon_quantile) <= 1.0:
        raise ValueError("Task-match epsilon quantile must lie in (0, 1].")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    admissible = run_ids[:, None] != run_ids[None, :]
    constrained = np.where(admissible, distances, np.inf)
    donor = np.argmin(constrained, axis=1)
    nearest = constrained[np.arange(values.shape[0]), donor]
    finite = np.isfinite(nearest)
    if not np.any(finite):
        raise ValueError("No finite different-partner task-state match exists.")
    epsilon = float(np.quantile(nearest[finite], float(epsilon_quantile)))
    valid = finite & (nearest <= epsilon)
    return (
        np.flatnonzero(valid).astype(np.int64),
        donor[valid].astype(np.int64),
        epsilon,
        nearest[valid].astype(np.float64),
    )


def protocol_swap_causal_consistency(
    *,
    swapped_logits: Any,
    original_logits: Any,
    signature_a: Any,
    signature_b: Any,
) -> float:
    """§4.3 readout: P(sign(Δlogit) agrees with the ΔA ordering) > 0.65.

    For each matched pair, the swap effect is
    ``Δlogit = logits(x_i with c_j) - logits(x_i with c_i)`` evaluated at the
    action where the CRN centered signature difference ``ΔA = A_i - A_j`` is
    largest in magnitude.  Consistency is the fraction of pairs whose sign
    matches sign(ΔA) at that action.
    """

    import numpy as np

    swapped = np.asarray(swapped_logits, dtype=np.float64)
    original = np.asarray(original_logits, dtype=np.float64)
    sig_a = np.asarray(signature_a, dtype=np.float64)
    sig_b = np.asarray(signature_b, dtype=np.float64)
    if not (swapped.shape == original.shape == sig_a.shape == sig_b.shape):
        raise ValueError("Context-swap tensors must share (pair, action) shape.")
    delta_logits = swapped - original
    delta_signature = sig_a - sig_b
    pivot = np.argmax(np.abs(delta_signature), axis=-1)
    rows = np.arange(delta_logits.shape[0])
    sign_logits = np.sign(delta_logits[rows, pivot])
    sign_signature = np.sign(delta_signature[rows, pivot])
    informative = sign_signature != 0.0
    if not np.any(informative):
        return 0.5
    return float(np.mean(sign_logits[informative] == sign_signature[informative]))


def continuation_swap_causal_consistency(
    *,
    swapped_returns_by_action: Any,
    original_returns_by_action: Any,
    source_signature: Any,
    target_signature: Any,
) -> float:
    """CRN continuation control for a real ``u`` or ``c`` intervention.

    At the action where target and source empirical signatures differ most,
    replacing the source context by the target context should move the real
    continuation return in the target-signature direction.  Inputs are raw
    simulator continuations, never policy logits.
    """

    import numpy as np

    swapped = np.asarray(swapped_returns_by_action, dtype=np.float64)
    original = np.asarray(original_returns_by_action, dtype=np.float64)
    source = np.asarray(source_signature, dtype=np.float64)
    target = np.asarray(target_signature, dtype=np.float64)
    if not (swapped.shape == original.shape == source.shape == target.shape):
        raise ValueError("Continuation-swap tensors must share (pair, action) shape.")
    direction = target - source
    pivot = np.argmax(np.abs(direction), axis=-1)
    rows = np.arange(direction.shape[0])
    expected = np.sign(direction[rows, pivot])
    observed = np.sign((swapped - original)[rows, pivot])
    informative = expected != 0.0
    if not np.any(informative):
        return 0.5
    return float(np.mean(observed[informative] == expected[informative]))


__all__ = [
    "BOOTSTRAP_REPLICATION",
    "LEAKAGE_PROBE_EXCESS_THRESHOLD",
    "PROTOCOL_SWAP_CONSISTENCY_THRESHOLD",
    "SHUFFLE_EPSILON_QUANTILE",
    "balanced_linear_probe_accuracy",
    "continuation_swap_causal_consistency",
    "history_shuffle_negative",
    "match_task_state_pairs",
    "match_different_partner_task_states",
    "paired_bootstrap_drop",
    "protocol_swap_causal_consistency",
    "task_representation_drift_under_shuffle",
]
