from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from experiments.overcooked_v2.path_c_protocol import (
    ActingTransition,
    PolicyAction,
    ProbeBudgetState,
)


@dataclass(frozen=True)
class HMMFilterConfig:
    mode_names: tuple[str, ...]
    transition_stay_prob: float = 0.98
    observation_smoothing: float = 1e-3

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.mode_names)
        if not names or any(not name.strip() for name in names):
            raise ValueError("HMMFilterConfig.mode_names must be non-empty names.")
        if len(set(names)) != len(names):
            raise ValueError("HMMFilterConfig.mode_names must be unique.")
        if not 0.0 <= float(self.transition_stay_prob) <= 1.0:
            raise ValueError("transition_stay_prob must be in [0, 1].")
        if (
            not math.isfinite(float(self.observation_smoothing))
            or float(self.observation_smoothing) <= 0.0
        ):
            raise ValueError("observation_smoothing must be finite and positive.")


@runtime_checkable
class OnlineModeBeliefFilter(Protocol):
    """Runtime interface shared by scripted, learned, and particle filters."""

    mode_names: tuple[str, ...]
    belief: np.ndarray
    inference_kind: str

    def reset(
        self,
        prior: Sequence[float] | None = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray: ...

    def update(self, observation: str) -> np.ndarray: ...

    def posterior(self) -> dict[str, float]: ...


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
        *,
        inference_kind: str = "scripted_hmm_filter",
    ) -> None:
        if str(inference_kind) not in {"scripted_hmm_filter", "learned_hmm_filter"}:
            raise ValueError("Bayesian HMM inference_kind is not registered.")
        self.config = config
        self.mode_names = tuple(str(name) for name in config.mode_names)
        self.mode_to_idx = {name: idx for idx, name in enumerate(self.mode_names)}
        self.inference_kind = str(inference_kind)
        self.likelihood = _validated_likelihood_table(
            observation_likelihood,
            mode_names=self.mode_names,
        )
        self.transition = self._transition_matrix(
            len(self.mode_names),
            float(config.transition_stay_prob),
        )
        self.belief = np.ones(len(self.mode_names), dtype=np.float64) / float(len(self.mode_names))

    def reset(
        self,
        prior: Sequence[float] | None = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        del seed
        self.belief = _normalized_prior(prior, n_modes=len(self.mode_names))
        return self.belief.copy()

    def update(self, observation: str) -> np.ndarray:
        predicted = self.transition.T @ self.belief
        likelihood = self._likelihood_vector(str(observation))
        posterior = predicted * likelihood
        total = float(posterior.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("The HMM posterior has no finite positive mass.")
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
        if observations.size == 0:
            raise ValueError("At least one training observation is required.")
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
        return cls(config, likelihood, inference_kind="learned_hmm_filter")

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
class ParticleFilterConfig:
    """Finite-state bootstrap particle-filter configuration."""

    mode_names: tuple[str, ...]
    particle_count: int = 1024
    transition_stay_prob: float = 0.98
    observation_smoothing: float = 1e-3

    def __post_init__(self) -> None:
        HMMFilterConfig(
            mode_names=tuple(self.mode_names),
            transition_stay_prob=float(self.transition_stay_prob),
            observation_smoothing=float(self.observation_smoothing),
        )
        if int(self.particle_count) <= 0:
            raise ValueError("particle_count must be positive.")

    @property
    def hmm_config(self) -> HMMFilterConfig:
        return HMMFilterConfig(
            mode_names=tuple(self.mode_names),
            transition_stay_prob=float(self.transition_stay_prob),
            observation_smoothing=float(self.observation_smoothing),
        )


class ParticleBeliefFilter:
    """Deployable bootstrap filter using only public response observations."""

    inference_kind = "particle_belief_filter"

    def __init__(
        self,
        config: ParticleFilterConfig,
        observation_likelihood: Mapping[str, Mapping[str, float]],
    ) -> None:
        self.config = config
        self.mode_names = tuple(str(name) for name in config.mode_names)
        self.mode_to_idx = {name: idx for idx, name in enumerate(self.mode_names)}
        self.likelihood = _validated_likelihood_table(
            observation_likelihood,
            mode_names=self.mode_names,
        )
        self.transition = BayesianHMMBeliefFilter._transition_matrix(
            len(self.mode_names),
            float(config.transition_stay_prob),
        )
        self._rng = np.random.default_rng(0)
        self.particles = np.zeros(int(config.particle_count), dtype=np.int64)
        self.belief = np.ones(len(self.mode_names), dtype=np.float64) / float(
            len(self.mode_names)
        )
        self.effective_sample_size = float(config.particle_count)
        self.reset(seed=0)

    def reset(
        self,
        prior: Sequence[float] | None = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        normalized = _normalized_prior(prior, n_modes=len(self.mode_names))
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self.particles = self._rng.choice(
            len(self.mode_names),
            size=int(self.config.particle_count),
            replace=True,
            p=normalized,
        ).astype(np.int64, copy=False)
        self.belief = normalized
        self.effective_sample_size = float(self.config.particle_count)
        return self.belief.copy()

    def update(self, observation: str) -> np.ndarray:
        transition_cdf = np.cumsum(self.transition[self.particles], axis=1)
        draws = self._rng.random(int(self.config.particle_count))
        predicted_particles = np.sum(
            draws[:, None] > transition_cdf,
            axis=1,
            dtype=np.int64,
        )
        predicted_particles = np.minimum(
            predicted_particles,
            len(self.mode_names) - 1,
        ).astype(np.int64, copy=False)
        likelihood = self._likelihood_vector(str(observation))[predicted_particles]
        weight_total = float(likelihood.sum())
        if not math.isfinite(weight_total) or weight_total <= 0.0:
            raise ValueError("The particle posterior has no finite positive mass.")
        weights = likelihood / weight_total
        self.effective_sample_size = float(1.0 / np.sum(weights * weights))
        posterior = np.bincount(
            predicted_particles,
            weights=weights,
            minlength=len(self.mode_names),
        ).astype(np.float64, copy=False)
        posterior_total = float(posterior.sum())
        if not math.isfinite(posterior_total) or posterior_total <= 0.0:
            raise ValueError("The particle mode posterior is invalid.")
        self.belief = posterior / posterior_total

        # Systematic resampling leaves the public posterior above untouched and
        # carries an equally weighted approximation into the next online step.
        positions = (
            self._rng.random() + np.arange(int(self.config.particle_count), dtype=np.float64)
        ) / float(self.config.particle_count)
        cumulative_weights = np.cumsum(weights)
        ancestor_indices = np.searchsorted(
            cumulative_weights,
            positions,
            side="right",
        )
        ancestor_indices = np.minimum(
            ancestor_indices,
            int(self.config.particle_count) - 1,
        )
        self.particles = predicted_particles[ancestor_indices].astype(np.int64, copy=False)
        return self.belief.copy()

    def posterior(self) -> dict[str, float]:
        return {
            mode: float(self.belief[index])
            for mode, index in self.mode_to_idx.items()
        }

    @classmethod
    def fit_from_training_split(
        cls,
        config: ParticleFilterConfig,
        observations: Sequence[str],
        mode_labels: Sequence[str],
    ) -> "ParticleBeliefFilter":
        learned = BayesianHMMBeliefFilter.fit_from_training_split(
            config.hmm_config,
            observations,
            mode_labels,
        )
        return cls(config, learned.likelihood)

    def _likelihood_vector(self, observation: str) -> np.ndarray:
        probabilities = self.likelihood.get(str(observation), {})
        smoothing = float(self.config.observation_smoothing)
        values = np.asarray(
            [float(probabilities.get(mode, smoothing)) for mode in self.mode_names],
            dtype=np.float64,
        )
        return np.maximum(values, smoothing)


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
            inference_kind="learned_hmm_filter",
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


class BeliefFilterActingPolicy:
    """Online belief-filter competitor under the common Path C acting protocol.

    This adapter is intentionally an acting policy rather than an offline readout.
    It updates its posterior after every observed transition and uses that posterior
    to choose the environment action under the same probe budget as the main method.
    """

    oracle_baseline = False

    def __init__(
        self,
        *,
        policy_name: str,
        belief_filter: OnlineModeBeliefFilter,
        action_values_by_mode: np.ndarray | None = None,
        public_context_weights: np.ndarray | None = None,
        public_context_key: str = "belief_public_context",
        observation_key: str = "belief_observation_token",
        benchmark_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if (action_values_by_mode is None) == (public_context_weights is None):
            raise ValueError(
                "Belief acting policy requires exactly one legacy mode table or "
                "state-conditioned public-context value head."
            )
        values = None
        weights = None
        if action_values_by_mode is not None:
            values = np.asarray(action_values_by_mode, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("action_values_by_mode must have shape [mode, action].")
            if values.shape[0] != len(belief_filter.mode_names):
                raise ValueError("Action-value mode dimension must match the belief filter.")
            if not np.all(np.isfinite(values)):
                raise ValueError("action_values_by_mode must be finite.")
        else:
            weights = np.asarray(public_context_weights, dtype=np.float64)
            if weights.ndim != 2 or weights.shape[0] <= len(belief_filter.mode_names) + 1:
                raise ValueError(
                    "public_context_weights must contain public context, belief, and intercept rows."
                )
            if not np.all(np.isfinite(weights)):
                raise ValueError("public_context_weights must be finite.")
        if not str(policy_name).strip():
            raise ValueError("policy_name must be non-empty.")
        self.policy_name = str(policy_name)
        self.filter = belief_filter
        self.action_values_by_mode = None if values is None else values.copy()
        self.public_context_weights = None if weights is None else weights.copy()
        self.public_context_key = str(public_context_key)
        if not self.public_context_key.strip():
            raise ValueError("public_context_key must be non-empty.")
        self._num_actions = int(
            values.shape[1] if values is not None else weights.shape[1]
        )
        self.observation_key = str(observation_key)
        self.benchmark_metadata = dict(benchmark_metadata or {})
        protected = {
            "probe_count",
            "environment_steps",
            "realized_probe_cost",
            "inference_kind",
            "oracle_baseline",
        }
        overlap = sorted(protected.intersection(self.benchmark_metadata))
        if overlap:
            raise ValueError(
                f"benchmark_metadata cannot override authoritative fields: {overlap}."
            )
        self._action_count = 0
        self._probe_count = 0
        self._environment_steps = 0
        self._realized_probe_cost = 0.0
        self._last_scores = np.zeros(self._num_actions, dtype=np.float64)

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None:
        del probe_budget_state
        self.filter.reset(seed=int(seed))
        self._action_count = 0
        self._probe_count = 0
        self._environment_steps = 0
        self._realized_probe_cost = 0.0
        self._last_scores.fill(0.0)

    def act(
        self,
        observation: Mapping[str, Any],
        probe_budget_state: ProbeBudgetState,
    ) -> PolicyAction:
        valid_actions = tuple(int(value) for value in observation.get("valid_action_ids", ()))
        if not valid_actions:
            raise ValueError("Belief acting policy requires non-empty valid_action_ids.")
        if len(set(valid_actions)) != len(valid_actions):
            raise ValueError("valid_action_ids must be unique.")
        if any(value < 0 or value >= self._num_actions for value in valid_actions):
            raise ValueError("valid_action_ids contain an action outside the value table.")
        probe_candidates = {
            int(value) for value in observation.get("probe_candidate_action_ids", ())
        }
        probe_only = {
            int(value) for value in observation.get("probe_only_action_ids", ())
        }
        valid_set = set(valid_actions)
        if not probe_candidates.issubset(valid_set):
            raise ValueError("probe_candidate_action_ids must be valid actions.")
        if not probe_only.issubset(probe_candidates):
            raise ValueError("probe_only_action_ids must be probe candidates.")
        eligible = tuple(
            value
            for value in valid_actions
            if probe_budget_state.remaining > 0 or value not in probe_only
        )
        if not eligible:
            raise ValueError("Probe budget masking removed every valid action.")
        belief = np.asarray(self.filter.belief, dtype=np.float64)
        if self.public_context_weights is None:
            if self.action_values_by_mode is None:
                raise RuntimeError("Belief acting value head was not configured.")
            scores = belief @ self.action_values_by_mode
        else:
            raw_context = observation.get(self.public_context_key)
            if raw_context is None:
                raise ValueError(
                    f"Belief acting observation lacks {self.public_context_key!r}."
                )
            context = np.asarray(raw_context, dtype=np.float64).reshape(-1)
            expected_context_dim = (
                self.public_context_weights.shape[0] - belief.size - 1
            )
            if context.shape != (expected_context_dim,) or not np.all(np.isfinite(context)):
                raise ValueError(
                    "Belief acting public context has the wrong shape or non-finite values."
                )
            design = np.concatenate([context, belief, np.ones(1, dtype=np.float64)])
            scores = design @ self.public_context_weights
        if probe_budget_state.remaining > 0:
            for action in probe_candidates:
                scores[int(action)] -= float(probe_budget_state.cost_per_probe)
        self._last_scores = scores.copy()
        selected = max(eligible, key=lambda action: (float(scores[action]), -int(action)))
        is_probe = int(selected) in probe_candidates and probe_budget_state.remaining > 0
        self._action_count += 1
        self._probe_count += int(is_probe)
        return PolicyAction(
            action_id=int(selected),
            is_probe=bool(is_probe),
            propensity=1.0,
            candidate_action_ids=eligible,
            candidate_scores=tuple(float(scores[action]) for action in eligible),
            estimated_probe_cost=(
                float(probe_budget_state.cost_per_probe) if is_probe else 0.0
            ),
            policy_kind="belief_filter_posterior_control",
        )

    def observe(self, transition: ActingTransition) -> None:
        token = transition.next_observation.get(self.observation_key)
        if token is None:
            raise ValueError(
                f"Belief acting transition lacks {self.observation_key!r}."
            )
        self.filter.update(str(token))
        self._environment_steps += int(transition.environment_steps)
        self._realized_probe_cost += float(transition.realized_probe_cost)

    def representation(self) -> np.ndarray:
        return np.asarray(self.filter.belief, dtype=np.float32).copy()

    def metrics(self) -> Mapping[str, Any]:
        metrics: dict[str, Any] = {
            **self.benchmark_metadata,
            "acting_protocol_version": "path_c_acting_policy_v1",
            "action_count": int(self._action_count),
            "probe_count": int(self._probe_count),
            "environment_steps": int(self._environment_steps),
            "realized_probe_cost": float(self._realized_probe_cost),
            "posterior_entropy": float(_categorical_entropy(self.filter.belief)),
            "inference_kind": str(self.filter.inference_kind),
            "oracle_baseline": False,
            "state_conditioned_action_values": self.public_context_weights is not None,
        }
        particle_count = getattr(getattr(self.filter, "config", None), "particle_count", None)
        effective_sample_size = getattr(self.filter, "effective_sample_size", None)
        if particle_count is not None:
            metrics["particle_count"] = int(particle_count)
        if effective_sample_size is not None:
            metrics["effective_sample_size"] = float(effective_sample_size)
        return metrics


def _categorical_entropy(probabilities: Sequence[float]) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    positive = probs > 0.0
    return float(-np.sum(probs[positive] * np.log(probs[positive])))


def _normalized_prior(
    prior: Sequence[float] | None,
    *,
    n_modes: int,
) -> np.ndarray:
    if int(n_modes) <= 0:
        raise ValueError("n_modes must be positive.")
    if prior is None:
        return np.ones(int(n_modes), dtype=np.float64) / float(n_modes)
    values = np.asarray(prior, dtype=np.float64)
    if values.shape != (int(n_modes),):
        raise ValueError(f"prior must have shape {(int(n_modes),)}, got {values.shape}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("prior must contain finite non-negative mass.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("prior must have positive mass.")
    return values / total


def _validated_likelihood_table(
    observation_likelihood: Mapping[str, Mapping[str, float]],
    *,
    mode_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    if not isinstance(observation_likelihood, Mapping):
        raise TypeError("observation_likelihood must be a mapping.")
    registered_modes = set(map(str, mode_names))
    validated: dict[str, dict[str, float]] = {}
    for observation, raw_probabilities in observation_likelihood.items():
        if not str(observation).strip():
            raise ValueError("Observation likelihood tokens must be non-empty.")
        if not isinstance(raw_probabilities, Mapping):
            raise TypeError("Each observation likelihood row must be a mapping.")
        unknown = set(map(str, raw_probabilities)).difference(registered_modes)
        if unknown:
            raise ValueError(f"Observation likelihood has unknown modes: {sorted(unknown)}")
        row: dict[str, float] = {}
        for mode, probability in raw_probabilities.items():
            value = float(probability)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("Observation likelihoods must be finite and non-negative.")
            row[str(mode)] = value
        validated[str(observation)] = row
    return validated
