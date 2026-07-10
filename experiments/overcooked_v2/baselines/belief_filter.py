from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HMMFilterConfig:
    mode_names: tuple[str, ...]
    transition_stay_prob: float = 0.98
    observation_smoothing: float = 1e-3


class BayesianHMMBeliefFilter:
    """Small discrete belief filter for Path C hard-baseline readouts.

    The filter is intentionally separate from the main method. It consumes a
    predeclared observation likelihood table over known mechanism names and
    returns posterior probabilities that downstream evaluation can residualize
    against the same public baseline as ARIS-Bellman.
    """

    def __init__(
        self,
        config: HMMFilterConfig,
        observation_likelihood: Mapping[str, Mapping[str, float]],
    ) -> None:
        if not config.mode_names:
            raise ValueError("HMMFilterConfig.mode_names must be non-empty.")
        self.config = config
        self.mode_names = tuple(str(name) for name in config.mode_names)
        self.mode_to_idx = {name: idx for idx, name in enumerate(self.mode_names)}
        self.likelihood = {
            str(obs): {
                str(mode): float(prob)
                for mode, prob in mode_probs.items()
            }
            for obs, mode_probs in observation_likelihood.items()
        }
        self.transition = self._transition_matrix(
            len(self.mode_names),
            float(config.transition_stay_prob),
        )
        self.belief = np.ones(len(self.mode_names), dtype=np.float64) / float(len(self.mode_names))

    def reset(self, prior: Sequence[float] | None = None) -> np.ndarray:
        if prior is None:
            self.belief = np.ones(len(self.mode_names), dtype=np.float64) / float(len(self.mode_names))
        else:
            arr = np.asarray(prior, dtype=np.float64)
            if arr.shape != (len(self.mode_names),):
                raise ValueError(
                    f"prior must have shape {(len(self.mode_names),)}, got {arr.shape}."
                )
            total = float(arr.sum())
            if total <= 0.0:
                raise ValueError("prior must have positive mass.")
            self.belief = arr / total
        return self.belief.copy()

    def update(self, observation: str) -> np.ndarray:
        predicted = self.transition.T @ self.belief
        likelihood = self._likelihood_vector(str(observation))
        posterior = predicted * likelihood
        total = float(posterior.sum())
        if total <= 0.0:
            posterior = np.ones_like(posterior) / float(posterior.size)
        else:
            posterior = posterior / total
        self.belief = posterior
        return self.belief.copy()

    def posterior(self) -> dict[str, float]:
        return {
            mode: float(self.belief[idx])
            for mode, idx in self.mode_to_idx.items()
        }

    @classmethod
    def fit_from_training_split(
        cls,
        config: HMMFilterConfig,
        observations: Sequence[str],
        mode_labels: Sequence[str],
    ) -> "BayesianHMMBeliefFilter":
        observations = np.asarray(observations).astype(str)
        modes = np.asarray(mode_labels).astype(str)
        if observations.shape != modes.shape or observations.ndim != 1:
            raise ValueError("Training observations and mode labels must be aligned vectors.")
        unknown_modes = sorted(set(modes.tolist()).difference(config.mode_names))
        if unknown_modes:
            raise ValueError(f"Unknown training mode labels: {unknown_modes}")
        smoothing = float(config.observation_smoothing)
        likelihood: dict[str, dict[str, float]] = {}
        for observation in sorted(np.unique(observations).tolist()):
            counts = {
                mode: float(np.count_nonzero((observations == observation) & (modes == mode)))
                + smoothing
                for mode in config.mode_names
            }
            total_by_mode = {
                mode: float(np.count_nonzero(modes == mode))
                + smoothing * max(1, len(np.unique(observations)))
                for mode in config.mode_names
            }
            likelihood[observation] = {
                mode: counts[mode] / total_by_mode[mode]
                for mode in config.mode_names
            }
        return cls(config, likelihood)

    def _likelihood_vector(self, observation: str) -> np.ndarray:
        probs = self.likelihood.get(observation, {})
        smooth = float(self.config.observation_smoothing)
        values = np.asarray(
            [float(probs.get(mode, smooth)) for mode in self.mode_names],
            dtype=np.float64,
        )
        return np.maximum(values, smooth)

    @staticmethod
    def _transition_matrix(n_modes: int, stay_prob: float) -> np.ndarray:
        if not 0.0 <= stay_prob <= 1.0:
            raise ValueError("transition_stay_prob must be in [0, 1].")
        if n_modes == 1:
            return np.ones((1, 1), dtype=np.float64)
        off = (1.0 - stay_prob) / float(n_modes - 1)
        mat = np.full((n_modes, n_modes), off, dtype=np.float64)
        np.fill_diagonal(mat, stay_prob)
        return mat


@dataclass(frozen=True)
class FittedBeliefValueBaseline:
    """Training-split HMM posterior plus a matched linear action-value head."""

    filter_config: HMMFilterConfig
    observation_likelihood: dict[str, dict[str, float]]
    value_weights: np.ndarray
    train_split_ids: tuple[str, ...]
    parameter_count: int

    @classmethod
    def fit(
        cls,
        *,
        filter_config: HMMFilterConfig,
        observations: Sequence[str],
        mode_labels: Sequence[str],
        episode_uids: Sequence[str],
        public_context: np.ndarray,
        target_action_values: np.ndarray,
        train_split_ids: Sequence[str],
        ridge: float = 1e-4,
    ) -> "FittedBeliefValueBaseline":
        observations_arr = np.asarray(observations).astype(str)
        modes_arr = np.asarray(mode_labels).astype(str)
        episodes_arr = np.asarray(episode_uids).astype(str)
        context = np.asarray(public_context, dtype=np.float64)
        targets = np.asarray(target_action_values, dtype=np.float64)
        n_rows = observations_arr.shape[0]
        if any(array.shape[0] != n_rows for array in (modes_arr, episodes_arr, context, targets)):
            raise ValueError("Belief baseline training arrays must have the same row count.")
        if context.ndim != 2 or targets.ndim != 2:
            raise ValueError("public_context and target_action_values must be matrices.")
        fitted_filter = BayesianHMMBeliefFilter.fit_from_training_split(
            filter_config,
            observations_arr,
            modes_arr,
        )
        posteriors = np.zeros((n_rows, len(filter_config.mode_names)), dtype=np.float64)
        previous_episode = None
        closed_episodes: set[str] = set()
        for index, (episode, observation) in enumerate(zip(episodes_arr, observations_arr, strict=True)):
            if episode != previous_episode:
                if episode in closed_episodes:
                    raise ValueError("Belief baseline episodes must be contiguous.")
                if previous_episode is not None:
                    closed_episodes.add(str(previous_episode))
                fitted_filter.reset()
                previous_episode = episode
            posteriors[index] = fitted_filter.update(str(observation))
        design = np.concatenate(
            [context, posteriors, np.ones((n_rows, 1), dtype=np.float64)],
            axis=1,
        )
        gram = design.T @ design + float(ridge) * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ targets)
        return cls(
            filter_config=filter_config,
            observation_likelihood={
                observation: dict(values)
                for observation, values in fitted_filter.likelihood.items()
            },
            value_weights=weights,
            train_split_ids=tuple(map(str, train_split_ids)),
            parameter_count=int(weights.size),
        )

    def predict(
        self,
        *,
        observations: Sequence[str],
        episode_uids: Sequence[str],
        public_context: np.ndarray,
        evaluation_split_id: str,
    ) -> np.ndarray:
        if str(evaluation_split_id) in set(self.train_split_ids):
            raise ValueError("Belief baseline evaluation split overlaps its training split.")
        observations_arr = np.asarray(observations).astype(str)
        episodes_arr = np.asarray(episode_uids).astype(str)
        context = np.asarray(public_context, dtype=np.float64)
        if observations_arr.shape[0] != episodes_arr.shape[0] or context.shape[0] != observations_arr.shape[0]:
            raise ValueError("Belief baseline prediction arrays must be aligned.")
        belief_filter = BayesianHMMBeliefFilter(
            self.filter_config,
            self.observation_likelihood,
        )
        posteriors = np.zeros(
            (observations_arr.shape[0], len(self.filter_config.mode_names)),
            dtype=np.float64,
        )
        previous_episode = None
        closed_episodes: set[str] = set()
        for index, (episode, observation) in enumerate(zip(episodes_arr, observations_arr, strict=True)):
            if episode != previous_episode:
                if episode in closed_episodes:
                    raise ValueError("Belief baseline episodes must be contiguous.")
                if previous_episode is not None:
                    closed_episodes.add(str(previous_episode))
                belief_filter.reset()
                previous_episode = episode
            posteriors[index] = belief_filter.update(str(observation))
        design = np.concatenate(
            [context, posteriors, np.ones((context.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        return design @ self.value_weights
