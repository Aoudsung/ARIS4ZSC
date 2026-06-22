import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        else:
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        ece += mask.sum() * abs(avg_confidence - avg_accuracy)
    return float(ece / len(probs)) if len(probs) > 0 else 0.0


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    alpha = (1 - ci) / 2
    boot_means = np.array([
        np.random.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return (
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.mean(values)),
        float(np.percentile(boot_means, 100 * (1 - alpha))),
    )


def pareto_frontier(costs: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(costs)
    pareto = []
    max_reward = -np.inf
    for idx in sorted_idx:
        if rewards[idx] > max_reward:
            pareto.append(idx)
            max_reward = rewards[idx]
    return np.array(pareto, dtype=int)
