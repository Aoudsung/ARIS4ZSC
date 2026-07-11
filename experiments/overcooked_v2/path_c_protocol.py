from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


PATH_C_ACTING_PROTOCOL_VERSION = "path_c_acting_policy_v1"
PATH_C_PRIMARY_ENDPOINT_VERSION = "path_c_primary_endpoint_v1"
DEPLOYABLE_STRONG_BASELINES = (
    "full_history_rnn",
    "rnn_residualized",
    "exact_belief_filter",
    "learned_hmm_filter",
    "particle_belief_filter",
)
FORMAL_ACTING_POLICY_NAMES = (
    "probing_ego",
    *DEPLOYABLE_STRONG_BASELINES,
    "random_probe",
    "no_probe",
    "direct_information",
)
FORMAL_LOCKED_ACTING_POLICY_NAMES = tuple(
    name for name in FORMAL_ACTING_POLICY_NAMES if name != "direct_information"
)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_value(value: Any) -> Any:
    """Convert acting metrics to a strict, deterministic JSON value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("Canonical acting logs require finite numeric metrics.")
        return converted
    if isinstance(value, np.ndarray):
        return _canonical_json_value(value.tolist())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Canonical acting-log metric keys must be strings.")
        return {
            str(key): _canonical_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(
        f"Acting-log metrics contain a non-JSON value of type {type(value).__name__}."
    )


def _require_exact_mapping_keys(
    payload: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    name: str,
) -> None:
    observed = set(payload)
    missing = sorted(set(expected).difference(observed))
    unknown = sorted(observed.difference(expected))
    if missing or unknown:
        raise ValueError(f"{name} key mismatch; missing={missing}, unknown={unknown}.")


@dataclass(frozen=True)
class ProbeBudgetState:
    """Immutable probe accounting passed to every acting policy."""

    budget: int
    probes_used: int = 0
    cost_per_probe: float = 0.0

    def __post_init__(self) -> None:
        for name in ("budget", "probes_used"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if int(self.budget) < 0:
            raise ValueError("Probe budget must be non-negative.")
        if not 0 <= int(self.probes_used) <= int(self.budget):
            raise ValueError("probes_used must be between zero and budget.")
        if not math.isfinite(float(self.cost_per_probe)) or float(self.cost_per_probe) < 0.0:
            raise ValueError("cost_per_probe must be finite and non-negative.")

    @property
    def remaining(self) -> int:
        return int(self.budget) - int(self.probes_used)

    @property
    def cumulative_cost(self) -> float:
        return float(self.probes_used) * float(self.cost_per_probe)

    def consume(self, *, is_probe: bool) -> "ProbeBudgetState":
        if not is_probe:
            return self
        if self.remaining <= 0:
            raise ValueError("A policy attempted a probe after exhausting its budget.")
        return ProbeBudgetState(
            budget=int(self.budget),
            probes_used=int(self.probes_used) + 1,
            cost_per_probe=float(self.cost_per_probe),
        )


@dataclass(frozen=True)
class PolicyAction:
    action_id: int
    is_probe: bool
    propensity: float
    candidate_action_ids: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    estimated_probe_cost: float
    policy_kind: str

    def __post_init__(self) -> None:
        if isinstance(self.action_id, bool) or not isinstance(self.action_id, Integral):
            raise TypeError("action_id must be an integer.")
        if type(self.is_probe) is not bool:
            raise TypeError("is_probe must be boolean.")
        if not str(self.policy_kind).strip():
            raise ValueError("policy_kind must be non-empty.")
        if not self.candidate_action_ids:
            raise ValueError("candidate_action_ids must be non-empty.")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in self.candidate_action_ids
        ):
            raise TypeError("candidate_action_ids must contain integers.")
        candidates = tuple(int(value) for value in self.candidate_action_ids)
        if int(self.action_id) not in candidates:
            raise ValueError("Selected action must be present in candidate_action_ids.")
        if len(candidates) != len(self.candidate_scores):
            raise ValueError("candidate_action_ids and candidate_scores must align.")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate_action_ids must be unique.")
        if (
            isinstance(self.propensity, bool)
            or not isinstance(self.propensity, (int, float, np.integer, np.floating))
            or not math.isfinite(float(self.propensity))
            or not 0.0 < float(self.propensity) <= 1.0
        ):
            raise ValueError("Action propensity must be in (0, 1].")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            for value in self.candidate_scores
        ):
            raise ValueError("candidate_scores must contain finite numeric values.")
        estimated_cost = float(self.estimated_probe_cost)
        if not math.isfinite(estimated_cost) or estimated_cost < 0.0:
            raise ValueError("estimated_probe_cost must be finite and non-negative.")
        if not bool(self.is_probe) and estimated_cost != 0.0:
            raise ValueError("A non-probe action must report zero estimated probe cost.")


@dataclass(frozen=True)
class ActingTransition:
    observation: Mapping[str, Any]
    action: PolicyAction
    reward: float
    next_observation: Mapping[str, Any]
    terminated: bool
    truncated: bool
    realized_probe_cost: float
    environment_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Mapping) or not isinstance(
            self.next_observation, Mapping
        ):
            raise TypeError("ActingTransition observations must be mappings.")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("ActingTransition termination flags must be boolean.")
        if self.terminated and self.truncated:
            raise ValueError("An acting transition cannot be both terminated and truncated.")
        if not math.isfinite(float(self.reward)):
            raise ValueError("ActingTransition.reward must be finite.")
        if not math.isfinite(float(self.realized_probe_cost)):
            raise ValueError("ActingTransition.realized_probe_cost must be finite.")
        if float(self.realized_probe_cost) < 0.0:
            raise ValueError("realized_probe_cost must be non-negative.")
        expected_cost = float(self.action.estimated_probe_cost) if self.action.is_probe else 0.0
        if not math.isclose(
            float(self.realized_probe_cost),
            expected_cost,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "realized_probe_cost must match the selected action's frozen probe cost."
            )
        if (
            isinstance(self.environment_steps, bool)
            or not isinstance(self.environment_steps, Integral)
            or int(self.environment_steps) <= 0
        ):
            raise ValueError("environment_steps must be positive.")


@runtime_checkable
class PathCActingPolicy(Protocol):
    """The common online protocol for the method and every acting baseline."""

    policy_name: str
    oracle_baseline: bool

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None: ...

    def act(
        self,
        observation: Mapping[str, Any],
        probe_budget_state: ProbeBudgetState,
    ) -> PolicyAction: ...

    def observe(self, transition: ActingTransition) -> None: ...

    def representation(self) -> np.ndarray: ...

    def metrics(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ActingEnvironmentStep:
    """One environment response under the common Path C acting protocol."""

    observation: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    environment_steps: int
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Mapping):
            raise TypeError("ActingEnvironmentStep.observation must be a mapping.")
        if not math.isfinite(float(self.reward)):
            raise ValueError("ActingEnvironmentStep.reward must be finite.")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("ActingEnvironmentStep termination flags must be boolean.")
        if self.terminated and self.truncated:
            raise ValueError("An environment step cannot be both terminated and truncated.")
        if (
            isinstance(self.environment_steps, bool)
            or not isinstance(self.environment_steps, Integral)
            or int(self.environment_steps) <= 0
        ):
            raise ValueError("ActingEnvironmentStep.environment_steps must be positive.")
        if not isinstance(self.info, Mapping):
            raise TypeError("ActingEnvironmentStep.info must be a mapping.")


@runtime_checkable
class PathCActingEnvironment(Protocol):
    """Environment side of the shared reset/step interaction contract."""

    def reset(self, *, seed: int) -> Mapping[str, Any]: ...

    def step(self, action_id: int) -> ActingEnvironmentStep: ...


@runtime_checkable
class PathCCostAwareActingEnvironment(Protocol):
    """Optional extension that records probe cost in the policy's evidence row."""

    def step_policy_action(
        self,
        action: PolicyAction,
        *,
        realized_probe_cost: float,
    ) -> ActingEnvironmentStep: ...


class CallableActingEnvironment:
    """Strict adapter for existing environments exposed through callables."""

    def __init__(self, *, reset_fn: Any, step_fn: Any) -> None:
        if not callable(reset_fn) or not callable(step_fn):
            raise TypeError("reset_fn and step_fn must be callable.")
        self._reset_fn = reset_fn
        self._step_fn = step_fn

    def reset(self, *, seed: int) -> Mapping[str, Any]:
        observation = self._reset_fn(seed=int(seed))
        if not isinstance(observation, Mapping):
            raise TypeError("The acting environment reset callable must return a mapping.")
        return observation

    def step(self, action_id: int) -> ActingEnvironmentStep:
        result = self._step_fn(int(action_id))
        if not isinstance(result, ActingEnvironmentStep):
            raise TypeError(
                "The acting environment step callable must return ActingEnvironmentStep."
            )
        return result


@dataclass(frozen=True)
class ActingBudgetContract:
    """Matched interaction and optimization schedule for benchmark policies."""

    environment_steps: int
    gradient_updates: int
    evaluation_episodes: int
    probe_budget_grid: tuple[int, ...]
    cost_per_probe: float

    def __post_init__(self) -> None:
        if min(
            int(self.environment_steps),
            int(self.gradient_updates),
            int(self.evaluation_episodes),
        ) <= 0:
            raise ValueError("Matched budget counts must be positive.")
        _validate_budget_grid(self.probe_budget_grid)
        if not math.isfinite(float(self.cost_per_probe)) or float(self.cost_per_probe) < 0.0:
            raise ValueError("cost_per_probe must be finite and non-negative.")

    @property
    def sha256(self) -> str:
        return _canonical_sha256({
            "environment_steps": int(self.environment_steps),
            "gradient_updates": int(self.gradient_updates),
            "evaluation_episodes": int(self.evaluation_episodes),
            "probe_budget_grid": list(map(int, self.probe_budget_grid)),
            "cost_per_probe": float(self.cost_per_probe),
        })


class ProbeBudgetController:
    """Stateful episode controller with immutable externally visible state."""

    def __init__(self, *, budget: int, cost_per_probe: float) -> None:
        self._initial = ProbeBudgetState(
            budget=int(budget),
            probes_used=0,
            cost_per_probe=float(cost_per_probe),
        )
        self._state = self._initial

    def reset(self) -> ProbeBudgetState:
        self._state = self._initial
        return self._state

    @property
    def state(self) -> ProbeBudgetState:
        return self._state

    def record(self, action: PolicyAction) -> ProbeBudgetState:
        expected_cost = float(self._state.cost_per_probe) if action.is_probe else 0.0
        if not math.isclose(
            float(action.estimated_probe_cost),
            expected_cost,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("PolicyAction uses a probe cost outside the assigned contract.")
        self._state = self._state.consume(is_probe=bool(action.is_probe))
        return self._state


class CallableActingPolicy:
    """Adapter used by the main recurrent ego and recurrent acting baselines."""

    oracle_baseline = False

    def __init__(
        self,
        *,
        policy_name: str,
        action_fn: Any,
        observe_fn: Any,
        representation_fn: Any,
        metrics_fn: Any,
        reset_fn: Any | None = None,
    ) -> None:
        for name, callback in (
            ("action_fn", action_fn),
            ("observe_fn", observe_fn),
            ("representation_fn", representation_fn),
            ("metrics_fn", metrics_fn),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable.")
        if reset_fn is not None and not callable(reset_fn):
            raise TypeError("reset_fn must be callable when supplied.")
        self.policy_name = str(policy_name)
        self._action_fn = action_fn
        self._observe_fn = observe_fn
        self._representation_fn = representation_fn
        self._metrics_fn = metrics_fn
        self._reset_fn = reset_fn
        self._seed: int | None = None

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None:
        self._seed = int(seed)
        reset_fn = self._reset_fn
        if reset_fn is None:
            reset_fn = getattr(self._action_fn, "reset", None)
        if callable(reset_fn):
            reset_fn(seed=int(seed), probe_budget_state=probe_budget_state)

    def act(
        self,
        observation: Mapping[str, Any],
        probe_budget_state: ProbeBudgetState,
    ) -> PolicyAction:
        action = self._action_fn(observation, probe_budget_state)
        if not isinstance(action, PolicyAction):
            raise TypeError("Callable acting policies must return PolicyAction.")
        return action

    def observe(self, transition: ActingTransition) -> None:
        self._observe_fn(transition)

    def representation(self) -> np.ndarray:
        return np.asarray(self._representation_fn(), dtype=np.float32).copy()

    def metrics(self) -> Mapping[str, Any]:
        return dict(self._metrics_fn())


class RandomProbeActingPolicy:
    """Matched-support random probe comparator with an explicit propensity."""

    oracle_baseline = False

    def __init__(self, *, exploitation_policy: PathCActingPolicy) -> None:
        self.policy_name = "random_probe"
        self._exploitation_policy = exploitation_policy
        self._rng = np.random.default_rng(0)
        self._probe_count = 0
        self._accounted_probe_count = 0
        self._environment_steps = 0
        self._realized_probe_cost = 0.0

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None:
        self._rng = np.random.default_rng(int(seed))
        self._probe_count = 0
        self._accounted_probe_count = 0
        self._environment_steps = 0
        self._realized_probe_cost = 0.0
        self._exploitation_policy.reset(
            seed=int(seed),
            probe_budget_state=probe_budget_state,
        )

    def act(
        self,
        observation: Mapping[str, Any],
        probe_budget_state: ProbeBudgetState,
    ) -> PolicyAction:
        candidates = tuple(int(value) for value in observation.get("probe_candidate_action_ids", ()))
        if probe_budget_state.remaining > 0 and candidates:
            selected = int(self._rng.choice(np.asarray(candidates, dtype=np.int64)))
            self._probe_count += 1
            return PolicyAction(
                action_id=selected,
                is_probe=True,
                propensity=1.0 / float(len(candidates)),
                candidate_action_ids=candidates,
                candidate_scores=tuple(0.0 for _ in candidates),
                estimated_probe_cost=float(probe_budget_state.cost_per_probe),
                policy_kind="random_probe_valid_support",
            )
        return self._exploitation_policy.act(observation, probe_budget_state)

    def observe(self, transition: ActingTransition) -> None:
        self._exploitation_policy.observe(transition)
        self._accounted_probe_count += int(transition.action.is_probe)
        self._environment_steps += int(transition.environment_steps)
        self._realized_probe_cost += float(transition.realized_probe_cost)

    def representation(self) -> np.ndarray:
        return self._exploitation_policy.representation()

    def metrics(self) -> Mapping[str, Any]:
        return {
            **dict(self._exploitation_policy.metrics()),
            "random_probe_count": int(self._probe_count),
            "probe_count": int(self._accounted_probe_count),
            "environment_steps": int(self._environment_steps),
            "realized_probe_cost": float(self._realized_probe_cost),
            "probe_support_policy": "matched_valid_action_support",
        }


class NoProbeActingPolicy:
    oracle_baseline = False

    def __init__(self, *, exploitation_policy: PathCActingPolicy) -> None:
        self.policy_name = "no_probe"
        self._exploitation_policy = exploitation_policy
        self._environment_steps = 0

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None:
        zero_budget = ProbeBudgetState(
            budget=0,
            probes_used=0,
            cost_per_probe=float(probe_budget_state.cost_per_probe),
        )
        self._environment_steps = 0
        self._exploitation_policy.reset(seed=int(seed), probe_budget_state=zero_budget)

    def act(
        self,
        observation: Mapping[str, Any],
        probe_budget_state: ProbeBudgetState,
    ) -> PolicyAction:
        zero_budget = ProbeBudgetState(
            budget=0,
            probes_used=0,
            cost_per_probe=float(probe_budget_state.cost_per_probe),
        )
        action = self._exploitation_policy.act(observation, zero_budget)
        if action.is_probe:
            raise ValueError("The no-probe comparator attempted to emit a probe.")
        return action

    def observe(self, transition: ActingTransition) -> None:
        if transition.action.is_probe or float(transition.realized_probe_cost) != 0.0:
            raise ValueError("The no-probe comparator observed a probe or probe cost.")
        self._exploitation_policy.observe(transition)
        self._environment_steps += int(transition.environment_steps)

    def representation(self) -> np.ndarray:
        return self._exploitation_policy.representation()

    def metrics(self) -> Mapping[str, Any]:
        return {
            **dict(self._exploitation_policy.metrics()),
            "probe_count": 0,
            "environment_steps": int(self._environment_steps),
            "realized_probe_cost": 0.0,
        }


class DirectInformationActingPolicy:
    """Synthetic-only oracle-design comparator using expected Jensen-Shannon gain."""

    oracle_baseline = True

    def __init__(
        self,
        *,
        response_probabilities: np.ndarray,
        exploitation_policy: PathCActingPolicy,
    ) -> None:
        probabilities = np.asarray(response_probabilities, dtype=np.float64)
        if probabilities.ndim != 3:
            raise ValueError(
                "response_probabilities must have shape [class, action, response]."
            )
        if any(int(size) <= 0 for size in probabilities.shape):
            raise ValueError("Every response-probability dimension must be positive.")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("Response probabilities must be finite and non-negative.")
        totals = probabilities.sum(axis=2, keepdims=True)
        if np.any(totals <= 0.0):
            raise ValueError("Every class/action response distribution needs positive mass.")
        self._response_probabilities = probabilities / totals
        self._exploitation_policy = exploitation_policy
        self.policy_name = "direct_information"
        self._posterior = np.ones(probabilities.shape[0], dtype=np.float64) / float(
            probabilities.shape[0]
        )
        self._last_action: int | None = None
        self._probe_count = 0
        self._environment_steps = 0
        self._realized_probe_cost = 0.0

    def reset(self, *, seed: int, probe_budget_state: ProbeBudgetState) -> None:
        self._posterior.fill(1.0 / float(self._posterior.size))
        self._last_action = None
        self._probe_count = 0
        self._environment_steps = 0
        self._realized_probe_cost = 0.0
        self._exploitation_policy.reset(seed=int(seed), probe_budget_state=probe_budget_state)

    def act(
        self,
        observation: Mapping[str, Any],
        probe_budget_state: ProbeBudgetState,
    ) -> PolicyAction:
        candidates = tuple(int(value) for value in observation.get("probe_candidate_action_ids", ()))
        if probe_budget_state.remaining <= 0 or not candidates:
            return self._exploitation_policy.act(observation, probe_budget_state)
        if any(action < 0 or action >= self._response_probabilities.shape[1] for action in candidates):
            raise ValueError("A direct-information candidate is outside the frozen action table.")
        scores = tuple(float(self._expected_js(action)) for action in candidates)
        best = max(scores)
        best_indices = [index for index, value in enumerate(scores) if value == best]
        chosen_index = int(best_indices[0])
        action_id = int(candidates[chosen_index])
        self._last_action = action_id
        return PolicyAction(
            action_id=action_id,
            is_probe=True,
            propensity=1.0,
            candidate_action_ids=candidates,
            candidate_scores=scores,
            estimated_probe_cost=float(probe_budget_state.cost_per_probe),
            policy_kind="direct_information_design_only_oracle",
        )

    def observe(self, transition: ActingTransition) -> None:
        response_token = transition.next_observation.get("response_token_id")
        if transition.action.is_probe and response_token is not None:
            action = int(transition.action.action_id)
            response = int(response_token)
            likelihood = self._response_probabilities[:, action, response]
            posterior = self._posterior * likelihood
            total = float(posterior.sum())
            if total > 0.0:
                self._posterior = posterior / total
        self._exploitation_policy.observe(transition)
        self._probe_count += int(transition.action.is_probe)
        self._environment_steps += int(transition.environment_steps)
        self._realized_probe_cost += float(transition.realized_probe_cost)

    def representation(self) -> np.ndarray:
        return self._posterior.astype(np.float32).copy()

    def metrics(self) -> Mapping[str, Any]:
        return {
            **dict(self._exploitation_policy.metrics()),
            "probe_count": int(self._probe_count),
            "environment_steps": int(self._environment_steps),
            "realized_probe_cost": float(self._realized_probe_cost),
            "oracle_baseline": True,
            "design_objective": "expected_jensen_shannon_separation",
        }

    def _expected_js(self, action: int) -> float:
        distributions = self._response_probabilities[:, int(action), :]
        mixture = np.sum(self._posterior[:, None] * distributions, axis=0)
        mixture = np.maximum(mixture, 1.0e-15)
        class_probs = np.maximum(distributions, 1.0e-15)
        kl = np.sum(class_probs * (np.log(class_probs) - np.log(mixture[None, :])), axis=1)
        return float(np.sum(self._posterior * kl))


@dataclass(frozen=True)
class ActingEpisodeSpec:
    """Frozen grouping and compute metadata for one acting episode."""

    split_role: str
    episode_uid: str
    split_group_id: str
    mechanism: str
    identity_group: str
    style_group: str
    layout_group: str
    seed_group: str
    seed: int
    probe_budget: int
    training_environment_steps: int
    gradient_updates: int
    evaluation_environment_step_limit: int
    evaluation_schedule_id: str

    def __post_init__(self) -> None:
        for name in (
            "episode_uid",
            "split_group_id",
            "mechanism",
            "identity_group",
            "style_group",
            "layout_group",
            "seed_group",
            "evaluation_schedule_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if self.split_role not in {"design", "locked_audit"}:
            raise ValueError("split_role must be design or locked_audit.")
        for name in (
            "seed",
            "probe_budget",
            "training_environment_steps",
            "gradient_updates",
            "evaluation_environment_step_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")
        if int(self.probe_budget) < 0:
            raise ValueError("probe_budget must be non-negative.")
        if int(self.training_environment_steps) <= 0:
            raise ValueError("training_environment_steps must be positive.")
        if int(self.gradient_updates) < 0:
            raise ValueError("gradient_updates must be non-negative.")
        if int(self.evaluation_environment_step_limit) <= 0:
            raise ValueError("evaluation_environment_step_limit must be positive.")


@dataclass(frozen=True)
class FormalActingBenchmarkContract:
    """Frozen interaction, training, split, and support contract for all factories."""

    split_manifest_sha256: str
    training_environment_steps: int
    gradient_updates: int
    evaluation_environment_step_limit: int
    evaluation_schedule_id: str
    probe_budget_grid: tuple[int, ...]
    probe_cost_per_use: float
    action_support_sha256: str
    effective_episode_floor: int
    effective_transition_floor: int
    schema_version: str = "path_c_formal_acting_benchmark_contract_v1"

    _MAPPING_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "split_manifest_sha256",
            "training_environment_steps",
            "gradient_updates",
            "evaluation_environment_step_limit",
            "evaluation_schedule_id",
            "probe_budget_grid",
            "probe_cost_per_use",
            "action_support_sha256",
            "effective_episode_floor",
            "effective_transition_floor",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != "path_c_formal_acting_benchmark_contract_v1":
            raise ValueError("Formal acting benchmark contract version changed.")
        for name in ("split_manifest_sha256", "action_support_sha256"):
            if not _is_sha256_digest(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
        object.__setattr__(
            self,
            "probe_budget_grid",
            tuple(self.probe_budget_grid),
        )
        for name in (
            "training_environment_steps",
            "gradient_updates",
            "evaluation_environment_step_limit",
            "effective_episode_floor",
            "effective_transition_floor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if int(self.training_environment_steps) <= 0:
            raise ValueError("training_environment_steps must be positive.")
        if int(self.gradient_updates) <= 0:
            raise ValueError("gradient_updates must be positive.")
        if int(self.evaluation_environment_step_limit) <= 0:
            raise ValueError("evaluation_environment_step_limit must be positive.")
        if int(self.effective_episode_floor) <= 0:
            raise ValueError("effective_episode_floor must be positive.")
        if int(self.effective_transition_floor) <= 0:
            raise ValueError("effective_transition_floor must be positive.")
        if not str(self.evaluation_schedule_id).strip():
            raise ValueError("evaluation_schedule_id must be non-empty.")
        _validate_budget_grid(self.probe_budget_grid)
        if not math.isfinite(float(self.probe_cost_per_use)) or float(
            self.probe_cost_per_use
        ) < 0.0:
            raise ValueError("probe_cost_per_use must be finite and non-negative.")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "FormalActingBenchmarkContract":
        if not isinstance(payload, Mapping):
            raise TypeError("Formal acting benchmark contract must be a mapping.")
        _require_exact_mapping_keys(
            payload,
            cls._MAPPING_KEYS,
            "formal acting benchmark contract",
        )
        return cls(
            schema_version=str(payload["schema_version"]),
            split_manifest_sha256=str(payload["split_manifest_sha256"]),
            training_environment_steps=payload["training_environment_steps"],
            gradient_updates=payload["gradient_updates"],
            evaluation_environment_step_limit=payload[
                "evaluation_environment_step_limit"
            ],
            evaluation_schedule_id=str(payload["evaluation_schedule_id"]),
            probe_budget_grid=tuple(payload["probe_budget_grid"]),
            probe_cost_per_use=float(payload["probe_cost_per_use"]),
            action_support_sha256=str(payload["action_support_sha256"]),
            effective_episode_floor=payload["effective_episode_floor"],
            effective_transition_floor=payload["effective_transition_floor"],
        )

    def validate_episode_spec(self, spec: ActingEpisodeSpec) -> None:
        observed = {
            "training_environment_steps": int(spec.training_environment_steps),
            "gradient_updates": int(spec.gradient_updates),
            "evaluation_environment_step_limit": int(
                spec.evaluation_environment_step_limit
            ),
            "evaluation_schedule_id": str(spec.evaluation_schedule_id),
        }
        expected = {
            "training_environment_steps": int(self.training_environment_steps),
            "gradient_updates": int(self.gradient_updates),
            "evaluation_environment_step_limit": int(
                self.evaluation_environment_step_limit
            ),
            "evaluation_schedule_id": str(self.evaluation_schedule_id),
        }
        if observed != expected:
            raise ValueError(
                "Acting episode spec differs from the frozen benchmark contract."
            )
        if int(spec.probe_budget) not in self.probe_budget_grid:
            raise ValueError("Acting episode probe budget is outside the frozen grid.")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_manifest_sha256": self.split_manifest_sha256,
            "training_environment_steps": int(self.training_environment_steps),
            "gradient_updates": int(self.gradient_updates),
            "evaluation_environment_step_limit": int(
                self.evaluation_environment_step_limit
            ),
            "evaluation_schedule_id": self.evaluation_schedule_id,
            "probe_budget_grid": list(map(int, self.probe_budget_grid)),
            "probe_cost_per_use": float(self.probe_cost_per_use),
            "action_support_sha256": self.action_support_sha256,
            "effective_episode_floor": int(self.effective_episode_floor),
            "effective_transition_floor": int(self.effective_transition_floor),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_mapping())


class FormalActingFactoryUnavailableError(RuntimeError):
    """Raised before rollout when any required formal policy factory is absent."""

    def __init__(self, missing_by_policy: Mapping[str, str]) -> None:
        self.missing_by_policy = {
            str(name): str(reason)
            for name, reason in sorted(missing_by_policy.items())
        }
        details = "; ".join(
            f"{name}: {reason}" for name, reason in self.missing_by_policy.items()
        )
        super().__init__(f"Formal acting policy factories are unavailable; {details}")


@dataclass(frozen=True)
class FormalActingPolicyFactoryEntry:
    """One explicit factory slot; unavailable slots remain visible and fail closed."""

    policy_name: str
    factory: Callable[
        [ActingEpisodeSpec, PathCActingEnvironment], PathCActingPolicy
    ] | None
    factory_id: str
    allowed_split_roles: tuple[str, ...]
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.policy_name not in FORMAL_ACTING_POLICY_NAMES:
            raise ValueError(f"Unknown formal acting policy {self.policy_name!r}.")
        if not str(self.factory_id).strip():
            raise ValueError("Formal acting factory_id must be non-empty.")
        roles = tuple(map(str, self.allowed_split_roles))
        if not roles or len(set(roles)) != len(roles) or any(
            role not in {"design", "locked_audit"} for role in roles
        ):
            raise ValueError("Formal acting factory split roles are invalid.")
        expected_roles = (
            ("design",)
            if self.policy_name == "direct_information"
            else ("design", "locked_audit")
        )
        if roles != expected_roles:
            raise ValueError(
                f"Formal acting policy {self.policy_name!r} must use roles {expected_roles}."
            )
        available = self.factory is not None
        if available == (self.unavailable_reason is not None):
            raise ValueError(
                "A formal factory slot needs exactly one of factory or unavailable_reason."
            )
        if available and not callable(self.factory):
            raise TypeError("Formal acting factory must be callable.")
        if self.unavailable_reason is not None and not self.unavailable_reason.strip():
            raise ValueError("Formal acting unavailable_reason must be non-empty.")

    @property
    def available(self) -> bool:
        return self.factory is not None

    def build(
        self,
        spec: ActingEpisodeSpec,
        environment: PathCActingEnvironment,
    ) -> PathCActingPolicy:
        if spec.split_role not in self.allowed_split_roles:
            raise ValueError(
                f"Policy {self.policy_name!r} is not allowed on {spec.split_role!r}."
            )
        if self.factory is None:
            raise FormalActingFactoryUnavailableError(
                {self.policy_name: str(self.unavailable_reason)}
            )
        policy = self.factory(spec, environment)
        if not isinstance(policy, PathCActingPolicy):
            raise TypeError(
                f"Factory {self.policy_name!r} did not return PathCActingPolicy."
            )
        if str(policy.policy_name) != self.policy_name:
            raise ValueError(
                f"Factory {self.policy_name!r} returned policy {policy.policy_name!r}."
            )
        return policy

    def to_manifest(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "factory_id": self.factory_id,
            "allowed_split_roles": list(self.allowed_split_roles),
            "available": bool(self.available),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class FormalActingPolicyFactoryRegistry:
    entries: tuple[FormalActingPolicyFactoryEntry, ...]
    benchmark_contract: FormalActingBenchmarkContract
    schema_version: str = "path_c_formal_acting_factory_registry_v1"

    def __post_init__(self) -> None:
        if self.schema_version != "path_c_formal_acting_factory_registry_v1":
            raise ValueError("Formal acting factory registry schema version changed.")
        if not isinstance(self.benchmark_contract, FormalActingBenchmarkContract):
            raise TypeError("Formal acting registry requires a benchmark contract.")
        names = tuple(entry.policy_name for entry in self.entries)
        if len(names) != len(set(names)):
            raise ValueError("Formal acting factory registry repeats a policy.")
        if set(names) != set(FORMAL_ACTING_POLICY_NAMES):
            missing = sorted(set(FORMAL_ACTING_POLICY_NAMES).difference(names))
            unknown = sorted(set(names).difference(FORMAL_ACTING_POLICY_NAMES))
            raise ValueError(
                f"Formal acting registry policy mismatch; missing={missing}, unknown={unknown}."
            )

    def factories_for_role(
        self,
        split_role: str,
    ) -> dict[str, FormalActingPolicyFactoryEntry]:
        required = (
            FORMAL_ACTING_POLICY_NAMES
            if split_role == "design"
            else FORMAL_LOCKED_ACTING_POLICY_NAMES
            if split_role == "locked_audit"
            else ()
        )
        if not required:
            raise ValueError("Formal acting split role must be design or locked_audit.")
        by_name = {entry.policy_name: entry for entry in self.entries}
        missing = {
            name: str(by_name[name].unavailable_reason)
            for name in required
            if not by_name[name].available
        }
        if missing:
            raise FormalActingFactoryUnavailableError(missing)
        return {name: by_name[name] for name in required}

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "benchmark_contract": self.benchmark_contract.to_mapping(),
            "benchmark_contract_sha256": self.benchmark_contract.sha256,
            "design_policy_names": list(FORMAL_ACTING_POLICY_NAMES),
            "locked_audit_policy_names": list(FORMAL_LOCKED_ACTING_POLICY_NAMES),
            "entries": [
                entry.to_manifest()
                for entry in sorted(self.entries, key=lambda item: item.policy_name)
            ],
        }
        return {**payload, "sha256": _canonical_sha256(payload)}


@dataclass(frozen=True)
class ActingStepLog:
    """Authoritative accounting for one act, debit, step, and observe cycle."""

    decision_index: int
    action: PolicyAction
    budget_before: ProbeBudgetState
    budget_after: ProbeBudgetState
    raw_reward: float
    realized_probe_cost: float
    environment_steps: int
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, Integral)
            or int(self.decision_index) < 0
        ):
            raise ValueError("decision_index must be non-negative.")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("ActingStepLog termination flags must be boolean.")
        if self.terminated and self.truncated:
            raise ValueError("An acting step cannot be both terminated and truncated.")
        if not math.isfinite(float(self.raw_reward)):
            raise ValueError("ActingStepLog.raw_reward must be finite.")
        if (
            not math.isfinite(float(self.realized_probe_cost))
            or float(self.realized_probe_cost) < 0.0
        ):
            raise ValueError("ActingStepLog.realized_probe_cost must be finite and non-negative.")
        if (
            isinstance(self.environment_steps, bool)
            or not isinstance(self.environment_steps, Integral)
            or int(self.environment_steps) <= 0
        ):
            raise ValueError("ActingStepLog.environment_steps must be positive.")
        if (
            int(self.budget_before.budget) != int(self.budget_after.budget)
            or not math.isclose(
                float(self.budget_before.cost_per_probe),
                float(self.budget_after.cost_per_probe),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("The probe budget contract changed inside an acting step.")
        expected_used = int(self.budget_before.probes_used) + int(self.action.is_probe)
        if int(self.budget_after.probes_used) != expected_used:
            raise ValueError("The acting step does not record its probe debit.")
        expected_cost = (
            float(self.budget_after.cumulative_cost)
            - float(self.budget_before.cumulative_cost)
        )
        if not math.isclose(
            float(self.realized_probe_cost),
            expected_cost,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("The acting step's realized probe cost does not match its debit.")
        if not self.action.is_probe and float(self.realized_probe_cost) != 0.0:
            raise ValueError("A non-probe acting step must realize zero probe cost.")

    @property
    def net_reward(self) -> float:
        return float(self.raw_reward) - float(self.realized_probe_cost)


@dataclass(frozen=True)
class ActingEpisodeLog:
    """Complete episode record used to construct one return-budget point."""

    policy_name: str
    spec: ActingEpisodeSpec
    probe_cost_per_use: float
    steps: tuple[ActingStepLog, ...]
    raw_return: float
    realized_probe_cost: float
    environment_steps: int
    final_budget_state: ProbeBudgetState
    policy_metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not str(self.policy_name).strip():
            raise ValueError("policy_name must be non-empty.")
        if not self.steps:
            raise ValueError("An acting episode must contain at least one environment step.")
        if not isinstance(self.policy_metrics, Mapping):
            raise TypeError("policy_metrics must be a mapping.")
        if (
            not math.isfinite(float(self.probe_cost_per_use))
            or float(self.probe_cost_per_use) < 0.0
        ):
            raise ValueError("probe_cost_per_use must be finite and non-negative.")
        if int(self.final_budget_state.budget) != int(self.spec.probe_budget):
            raise ValueError("The final budget state does not match the episode assignment.")
        if not math.isclose(
            float(self.final_budget_state.cost_per_probe),
            float(self.probe_cost_per_use),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("The final budget state changed the frozen probe price.")
        previous = ProbeBudgetState(
            budget=int(self.spec.probe_budget),
            probes_used=0,
            cost_per_probe=float(self.probe_cost_per_use),
        )
        for expected_index, step in enumerate(self.steps):
            if int(step.decision_index) != expected_index:
                raise ValueError("Acting step indices must be contiguous from zero.")
            if step.budget_before != previous:
                raise ValueError("Acting step budget states must form one contiguous ledger.")
            previous = step.budget_after
        if previous != self.final_budget_state:
            raise ValueError("The final budget state does not match the step ledger.")
        if not (self.steps[-1].terminated or self.steps[-1].truncated):
            raise ValueError("An acting episode log must end at termination or truncation.")
        expected_raw = float(sum(float(step.raw_reward) for step in self.steps))
        expected_cost = float(sum(float(step.realized_probe_cost) for step in self.steps))
        expected_environment_steps = int(sum(int(step.environment_steps) for step in self.steps))
        if not math.isclose(float(self.raw_return), expected_raw, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("raw_return does not match the acting step ledger.")
        if not math.isclose(
            float(self.realized_probe_cost),
            expected_cost,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("realized_probe_cost does not match the acting step ledger.")
        if int(self.environment_steps) != expected_environment_steps:
            raise ValueError("environment_steps does not match the acting step ledger.")
        if not math.isclose(
            expected_cost,
            float(self.final_budget_state.cumulative_cost),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Episode cost does not match the frozen per-use probe price.")

    @property
    def probes_used(self) -> int:
        return int(self.final_budget_state.probes_used)

    @property
    def net_return(self) -> float:
        return float(self.raw_return) - float(self.realized_probe_cost)

    def canonical_payload(self) -> dict[str, Any]:
        """Return the complete deterministic episode ledger used for hashing."""

        def budget_payload(state: ProbeBudgetState) -> dict[str, Any]:
            return {
                "budget": int(state.budget),
                "probes_used": int(state.probes_used),
                "cost_per_probe": float(state.cost_per_probe),
            }

        def action_payload(action: PolicyAction) -> dict[str, Any]:
            return {
                "action_id": int(action.action_id),
                "is_probe": bool(action.is_probe),
                "propensity": float(action.propensity),
                "candidate_action_ids": list(map(int, action.candidate_action_ids)),
                "candidate_scores": list(map(float, action.candidate_scores)),
                "estimated_probe_cost": float(action.estimated_probe_cost),
                "policy_kind": str(action.policy_kind),
            }

        return {
            "schema_version": "path_c_acting_episode_log_v1",
            "policy_name": str(self.policy_name),
            "spec": {
                "split_role": str(self.spec.split_role),
                "episode_uid": str(self.spec.episode_uid),
                "split_group_id": str(self.spec.split_group_id),
                "mechanism": str(self.spec.mechanism),
                "identity_group": str(self.spec.identity_group),
                "style_group": str(self.spec.style_group),
                "layout_group": str(self.spec.layout_group),
                "seed_group": str(self.spec.seed_group),
                "seed": int(self.spec.seed),
                "probe_budget": int(self.spec.probe_budget),
                "training_environment_steps": int(
                    self.spec.training_environment_steps
                ),
                "gradient_updates": int(self.spec.gradient_updates),
                "evaluation_environment_step_limit": int(
                    self.spec.evaluation_environment_step_limit
                ),
                "evaluation_schedule_id": str(self.spec.evaluation_schedule_id),
            },
            "probe_cost_per_use": float(self.probe_cost_per_use),
            "steps": [
                {
                    "decision_index": int(step.decision_index),
                    "action": action_payload(step.action),
                    "budget_before": budget_payload(step.budget_before),
                    "budget_after": budget_payload(step.budget_after),
                    "raw_reward": float(step.raw_reward),
                    "realized_probe_cost": float(step.realized_probe_cost),
                    "environment_steps": int(step.environment_steps),
                    "terminated": bool(step.terminated),
                    "truncated": bool(step.truncated),
                }
                for step in self.steps
            ],
            "raw_return": float(self.raw_return),
            "realized_probe_cost": float(self.realized_probe_cost),
            "environment_steps": int(self.environment_steps),
            "final_budget_state": budget_payload(self.final_budget_state),
            "policy_metrics": _canonical_json_value(self.policy_metrics),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ActingEpisodeLog":
        """Strictly reconstruct a hash-addressable episode accounting ledger."""

        if not isinstance(payload, Mapping):
            raise TypeError("Acting episode log must be a mapping.")
        _require_exact_mapping_keys(
            payload,
            {
                "schema_version",
                "policy_name",
                "spec",
                "probe_cost_per_use",
                "steps",
                "raw_return",
                "realized_probe_cost",
                "environment_steps",
                "final_budget_state",
                "policy_metrics",
            },
            "acting episode log",
        )
        if payload["schema_version"] != "path_c_acting_episode_log_v1":
            raise ValueError("Acting episode log has an unsupported schema version.")

        raw_spec = payload["spec"]
        if not isinstance(raw_spec, Mapping):
            raise TypeError("Acting episode spec must be a mapping.")
        _require_exact_mapping_keys(
            raw_spec,
            {
                "split_role",
                "episode_uid",
                "split_group_id",
                "mechanism",
                "identity_group",
                "style_group",
                "layout_group",
                "seed_group",
                "seed",
                "probe_budget",
                "training_environment_steps",
                "gradient_updates",
                "evaluation_environment_step_limit",
                "evaluation_schedule_id",
            },
            "acting episode spec",
        )
        spec = ActingEpisodeSpec(
            split_role=raw_spec["split_role"],
            episode_uid=raw_spec["episode_uid"],
            split_group_id=raw_spec["split_group_id"],
            mechanism=raw_spec["mechanism"],
            identity_group=raw_spec["identity_group"],
            style_group=raw_spec["style_group"],
            layout_group=raw_spec["layout_group"],
            seed_group=raw_spec["seed_group"],
            seed=raw_spec["seed"],
            probe_budget=raw_spec["probe_budget"],
            training_environment_steps=raw_spec["training_environment_steps"],
            gradient_updates=raw_spec["gradient_updates"],
            evaluation_environment_step_limit=raw_spec[
                "evaluation_environment_step_limit"
            ],
            evaluation_schedule_id=raw_spec["evaluation_schedule_id"],
        )

        def budget_from_mapping(raw: Any, name: str) -> ProbeBudgetState:
            if not isinstance(raw, Mapping):
                raise TypeError(f"{name} must be a mapping.")
            _require_exact_mapping_keys(
                raw,
                {"budget", "probes_used", "cost_per_probe"},
                name,
            )
            return ProbeBudgetState(
                budget=raw["budget"],
                probes_used=raw["probes_used"],
                cost_per_probe=raw["cost_per_probe"],
            )

        raw_steps = payload["steps"]
        if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
            raise TypeError("Acting episode steps must be a sequence.")
        steps: list[ActingStepLog] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise TypeError("Each acting episode step must be a mapping.")
            _require_exact_mapping_keys(
                raw_step,
                {
                    "decision_index",
                    "action",
                    "budget_before",
                    "budget_after",
                    "raw_reward",
                    "realized_probe_cost",
                    "environment_steps",
                    "terminated",
                    "truncated",
                },
                f"acting step {index}",
            )
            raw_action = raw_step["action"]
            if not isinstance(raw_action, Mapping):
                raise TypeError("Acting step action must be a mapping.")
            _require_exact_mapping_keys(
                raw_action,
                {
                    "action_id",
                    "is_probe",
                    "propensity",
                    "candidate_action_ids",
                    "candidate_scores",
                    "estimated_probe_cost",
                    "policy_kind",
                },
                f"acting step {index} action",
            )
            candidate_ids = raw_action["candidate_action_ids"]
            candidate_scores = raw_action["candidate_scores"]
            if isinstance(candidate_ids, (str, bytes)) or not isinstance(
                candidate_ids, Sequence
            ):
                raise TypeError("candidate_action_ids must be a sequence.")
            if isinstance(candidate_scores, (str, bytes)) or not isinstance(
                candidate_scores, Sequence
            ):
                raise TypeError("candidate_scores must be a sequence.")
            action = PolicyAction(
                action_id=raw_action["action_id"],
                is_probe=raw_action["is_probe"],
                propensity=raw_action["propensity"],
                candidate_action_ids=tuple(candidate_ids),
                candidate_scores=tuple(candidate_scores),
                estimated_probe_cost=raw_action["estimated_probe_cost"],
                policy_kind=raw_action["policy_kind"],
            )
            steps.append(
                ActingStepLog(
                    decision_index=raw_step["decision_index"],
                    action=action,
                    budget_before=budget_from_mapping(
                        raw_step["budget_before"], f"acting step {index} budget_before"
                    ),
                    budget_after=budget_from_mapping(
                        raw_step["budget_after"], f"acting step {index} budget_after"
                    ),
                    raw_reward=raw_step["raw_reward"],
                    realized_probe_cost=raw_step["realized_probe_cost"],
                    environment_steps=raw_step["environment_steps"],
                    terminated=raw_step["terminated"],
                    truncated=raw_step["truncated"],
                )
            )
        raw_metrics = payload["policy_metrics"]
        if not isinstance(raw_metrics, Mapping):
            raise TypeError("Acting episode policy_metrics must be a mapping.")
        log = cls(
            policy_name=payload["policy_name"],
            spec=spec,
            probe_cost_per_use=payload["probe_cost_per_use"],
            steps=tuple(steps),
            raw_return=payload["raw_return"],
            realized_probe_cost=payload["realized_probe_cost"],
            environment_steps=payload["environment_steps"],
            final_budget_state=budget_from_mapping(
                payload["final_budget_state"], "acting final budget state"
            ),
            policy_metrics=dict(raw_metrics),
        )
        if log.canonical_payload() != dict(payload):
            raise ValueError("Acting episode log is not in canonical form.")
        return log

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())

    def return_probe_budget_point(self) -> "ReturnProbeBudgetPoint":
        return ReturnProbeBudgetPoint(
            policy_name=str(self.policy_name),
            split_group_id=str(self.spec.split_group_id),
            mechanism=str(self.spec.mechanism),
            identity_group=str(self.spec.identity_group),
            style_group=str(self.spec.style_group),
            layout_group=str(self.spec.layout_group),
            seed_group=str(self.spec.seed_group),
            seed=int(self.spec.seed),
            budget=int(self.spec.probe_budget),
            raw_return=float(self.raw_return),
            probes_used=int(self.probes_used),
            realized_probe_cost=float(self.realized_probe_cost),
            probe_cost_per_use=float(self.probe_cost_per_use),
            episode_environment_steps=int(self.environment_steps),
            training_environment_steps=int(self.spec.training_environment_steps),
            gradient_updates=int(self.spec.gradient_updates),
            evaluation_environment_step_limit=int(
                self.spec.evaluation_environment_step_limit
            ),
            evaluation_schedule_id=str(self.spec.evaluation_schedule_id),
        )


class PathCActingRunner:
    """Execute every method and baseline through one accounting path."""

    def __init__(self, *, cost_per_probe: float, maximum_environment_steps: int) -> None:
        if not math.isfinite(float(cost_per_probe)) or float(cost_per_probe) < 0.0:
            raise ValueError("cost_per_probe must be finite and non-negative.")
        if int(maximum_environment_steps) <= 0:
            raise ValueError("maximum_environment_steps must be positive.")
        self.cost_per_probe = float(cost_per_probe)
        self.maximum_environment_steps = int(maximum_environment_steps)

    def run_episode(
        self,
        *,
        environment: PathCActingEnvironment,
        policy: PathCActingPolicy,
        spec: ActingEpisodeSpec,
    ) -> ActingEpisodeLog:
        if int(spec.evaluation_environment_step_limit) != self.maximum_environment_steps:
            raise ValueError(
                "Acting episode changed the frozen evaluation environment-step limit."
            )
        observation = environment.reset(seed=int(spec.seed))
        if not isinstance(observation, Mapping):
            raise TypeError("Path C acting environment reset must return a mapping.")
        controller = ProbeBudgetController(
            budget=int(spec.probe_budget),
            cost_per_probe=float(self.cost_per_probe),
        )
        policy.reset(seed=int(spec.seed), probe_budget_state=controller.state)
        step_logs: list[ActingStepLog] = []
        raw_return = 0.0
        realized_cost = 0.0
        environment_steps = 0
        while True:
            budget_before = controller.state
            action = policy.act(observation, budget_before)
            if not isinstance(action, PolicyAction):
                raise TypeError("Path C acting policies must return PolicyAction.")
            if not action.is_probe and float(action.estimated_probe_cost) != 0.0:
                raise ValueError("A non-probe action must report zero estimated probe cost.")

            # Debit first so an invalid cost or exhausted budget fails before the
            # environment can change state.
            budget_after = controller.record(action)
            step_cost = (
                float(budget_after.cumulative_cost)
                - float(budget_before.cumulative_cost)
            )
            if isinstance(environment, PathCCostAwareActingEnvironment):
                result = environment.step_policy_action(
                    action,
                    realized_probe_cost=float(step_cost),
                )
            else:
                result = environment.step(int(action.action_id))
            if not isinstance(result, ActingEnvironmentStep):
                raise TypeError("Path C acting environment step must return ActingEnvironmentStep.")
            next_environment_steps = environment_steps + int(result.environment_steps)
            if next_environment_steps > self.maximum_environment_steps:
                raise ValueError("The acting episode exceeded maximum_environment_steps.")
            transition = ActingTransition(
                observation=observation,
                action=action,
                reward=float(result.reward),
                next_observation=result.observation,
                terminated=bool(result.terminated),
                truncated=bool(result.truncated),
                realized_probe_cost=float(step_cost),
                environment_steps=int(result.environment_steps),
            )
            policy.observe(transition)
            step_logs.append(
                ActingStepLog(
                    decision_index=len(step_logs),
                    action=action,
                    budget_before=budget_before,
                    budget_after=budget_after,
                    raw_reward=float(result.reward),
                    realized_probe_cost=float(step_cost),
                    environment_steps=int(result.environment_steps),
                    terminated=bool(result.terminated),
                    truncated=bool(result.truncated),
                )
            )
            raw_return += float(result.reward)
            realized_cost += float(step_cost)
            environment_steps = next_environment_steps
            if result.terminated or result.truncated:
                break
            if environment_steps == self.maximum_environment_steps:
                raise ValueError(
                    "The acting episode reached maximum_environment_steps without ending."
                )
            observation = result.observation

        reported_metrics = policy.metrics()
        if not isinstance(reported_metrics, Mapping):
            raise TypeError("Path C acting policy metrics must be a mapping.")
        metrics = dict(reported_metrics)
        self._validate_reported_accounting(
            metrics,
            probes_used=int(controller.state.probes_used),
            realized_probe_cost=float(realized_cost),
            environment_steps=int(environment_steps),
        )
        return ActingEpisodeLog(
            policy_name=str(policy.policy_name),
            spec=spec,
            probe_cost_per_use=float(self.cost_per_probe),
            steps=tuple(step_logs),
            raw_return=float(raw_return),
            realized_probe_cost=float(realized_cost),
            environment_steps=int(environment_steps),
            final_budget_state=controller.state,
            policy_metrics=metrics,
        )

    @staticmethod
    def _validate_reported_accounting(
        metrics: Mapping[str, Any],
        *,
        probes_used: int,
        realized_probe_cost: float,
        environment_steps: int,
    ) -> None:
        expected = {
            "probe_count": int(probes_used),
            "realized_probe_cost": float(realized_probe_cost),
            "environment_steps": int(environment_steps),
        }
        for name, expected_value in expected.items():
            if name not in metrics:
                raise ValueError(f"Policy metrics must report authoritative {name}.")
            try:
                reported = float(metrics[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Policy metric {name} must be numeric.") from error
            if not math.isfinite(reported) or not math.isclose(
                reported,
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"Policy metric {name} disagrees with the acting ledger.")


@dataclass(frozen=True)
class ReturnProbeBudgetPoint:
    policy_name: str
    split_group_id: str
    mechanism: str
    identity_group: str
    style_group: str
    layout_group: str
    seed_group: str
    seed: int
    budget: int
    raw_return: float
    probes_used: int
    realized_probe_cost: float
    probe_cost_per_use: float
    episode_environment_steps: int
    training_environment_steps: int
    gradient_updates: int
    evaluation_environment_step_limit: int
    evaluation_schedule_id: str

    def __post_init__(self) -> None:
        for name in (
            "policy_name",
            "split_group_id",
            "mechanism",
            "identity_group",
            "style_group",
            "layout_group",
            "seed_group",
            "evaluation_schedule_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        for name in (
            "seed",
            "budget",
            "probes_used",
            "episode_environment_steps",
            "training_environment_steps",
            "gradient_updates",
            "evaluation_environment_step_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")
        if int(self.budget) < 0 or not 0 <= int(self.probes_used) <= int(self.budget):
            raise ValueError("Probe count must lie inside the assigned budget.")
        if not math.isfinite(float(self.raw_return)):
            raise ValueError("raw_return must be finite.")
        if (
            not math.isfinite(float(self.realized_probe_cost))
            or float(self.realized_probe_cost) < 0.0
        ):
            raise ValueError("realized_probe_cost must be finite and non-negative.")
        if (
            not math.isfinite(float(self.probe_cost_per_use))
            or float(self.probe_cost_per_use) < 0.0
        ):
            raise ValueError("probe_cost_per_use must be finite and non-negative.")
        expected_cost = int(self.probes_used) * float(self.probe_cost_per_use)
        if not math.isclose(
            float(self.realized_probe_cost),
            expected_cost,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "realized_probe_cost must equal probes_used times probe_cost_per_use."
            )
        if (
            int(self.episode_environment_steps) <= 0
            or int(self.training_environment_steps) <= 0
            or int(self.evaluation_environment_step_limit) <= 0
            or int(self.gradient_updates) < 0
        ):
            raise ValueError(
                "Episode/training/limit steps must be positive and updates non-negative."
            )

    @property
    def net_return(self) -> float:
        return float(self.raw_return) - float(self.realized_probe_cost)


@dataclass(frozen=True)
class ReturnProbeBudgetCurve:
    policy_name: str
    split_group_id: str
    mechanism: str
    identity_group: str
    style_group: str
    layout_group: str
    seed_group: str
    seed: int
    budgets: tuple[int, ...]
    normalized_net_returns: tuple[float, ...]
    auc: float
    normalization_lower: float
    normalization_upper: float
    probe_cost_per_use: float
    normalization_rule: str = "affine_without_clipping"

    def __post_init__(self) -> None:
        for name in (
            "policy_name",
            "split_group_id",
            "mechanism",
            "identity_group",
            "style_group",
            "layout_group",
            "seed_group",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        _validate_budget_grid(self.budgets)
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("Curve seed must be an integer.")
        if int(self.seed) < 0:
            raise ValueError("Curve seed must be non-negative.")
        if len(self.budgets) != len(self.normalized_net_returns):
            raise ValueError("Curve budgets and normalized net returns must align.")
        normalized = np.asarray(self.normalized_net_returns, dtype=np.float64)
        if not np.all(np.isfinite(normalized)) or not math.isfinite(float(self.auc)):
            raise ValueError("Curve values and AUC must be finite.")
        if (
            not math.isfinite(float(self.normalization_lower))
            or not math.isfinite(float(self.normalization_upper))
            or float(self.normalization_upper) <= float(self.normalization_lower)
        ):
            raise ValueError("Curve normalization requires finite upper > lower.")
        if (
            not math.isfinite(float(self.probe_cost_per_use))
            or float(self.probe_cost_per_use) < 0.0
        ):
            raise ValueError("Curve probe_cost_per_use must be finite and non-negative.")
        if self.normalization_rule != "affine_without_clipping":
            raise ValueError("Curve normalization rule is not registered.")
        grid = np.asarray(self.budgets, dtype=np.float64)
        width = float(grid[-1] - grid[0])
        expected_auc = (
            float(normalized[0])
            if width == 0.0
            else float(
                np.sum(
                    np.diff(grid) * 0.5 * (normalized[:-1] + normalized[1:]),
                    dtype=np.float64,
                )
                / width
            )
        )
        if not math.isclose(float(self.auc), expected_auc, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Curve AUC does not match its frozen budget points.")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "policy_name": str(self.policy_name),
            "split_group_id": str(self.split_group_id),
            "mechanism": str(self.mechanism),
            "identity_group": str(self.identity_group),
            "style_group": str(self.style_group),
            "layout_group": str(self.layout_group),
            "seed_group": str(self.seed_group),
            "seed": int(self.seed),
            "budgets": list(map(int, self.budgets)),
            "normalized_net_returns": list(
                map(float, self.normalized_net_returns)
            ),
            "auc": float(self.auc),
            "normalization_lower": float(self.normalization_lower),
            "normalization_upper": float(self.normalization_upper),
            "probe_cost_per_use": float(self.probe_cost_per_use),
            "normalization_rule": str(self.normalization_rule),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


@dataclass(frozen=True)
class BaselineSelection:
    selected_baseline: str
    design_auc_by_baseline: Mapping[str, float]
    candidate_order: tuple[str, ...]
    selection_role: str
    selection_rule: str
    probe_budget_grid: tuple[int, ...]
    normalization_rule: str
    normalization_lower: float
    normalization_upper: float
    probe_cost_per_use: float
    source_curve_set_sha256_by_baseline: Mapping[str, str]
    return_point_ledger_sha256: str
    design_split_group_ids: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        if self.selection_role != "design":
            raise ValueError("BaselineSelection must originate on the design split.")
        _validate_budget_grid(self.probe_budget_grid)
        if self.normalization_rule != "affine_without_clipping":
            raise ValueError("BaselineSelection changed the normalization rule.")
        if (
            not math.isfinite(float(self.normalization_lower))
            or not math.isfinite(float(self.normalization_upper))
            or float(self.normalization_upper) <= float(self.normalization_lower)
        ):
            raise ValueError("BaselineSelection normalization requires finite upper > lower.")
        if (
            not math.isfinite(float(self.probe_cost_per_use))
            or float(self.probe_cost_per_use) < 0.0
        ):
            raise ValueError("BaselineSelection probe cost must be finite and non-negative.")
        candidates = tuple(map(str, self.candidate_order))
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("BaselineSelection candidate order must be non-empty and unique.")
        if self.selected_baseline not in candidates:
            raise ValueError("BaselineSelection selected baseline is not a candidate.")
        if self.selection_rule != (
            "highest_mean_normalized_net_return_auc_then_preregistered_order"
        ):
            raise ValueError("BaselineSelection changed the frozen selection rule.")
        if set(self.design_auc_by_baseline) != set(candidates):
            raise ValueError("BaselineSelection AUCs do not match its candidate order.")
        if any(
            not math.isfinite(float(value))
            for value in self.design_auc_by_baseline.values()
        ):
            raise ValueError("BaselineSelection AUCs must be finite.")
        if set(self.source_curve_set_sha256_by_baseline) != set(candidates):
            raise ValueError("BaselineSelection curve hashes do not match its candidates.")
        if not all(
            _is_sha256_digest(value)
            for value in self.source_curve_set_sha256_by_baseline.values()
        ):
            raise ValueError("BaselineSelection source curve hashes must be SHA-256 digests.")
        if not _is_sha256_digest(self.return_point_ledger_sha256):
            raise ValueError("BaselineSelection must bind the return-point ledger SHA-256.")
        if (
            not self.design_split_group_ids
            or tuple(sorted(self.design_split_group_ids)) != self.design_split_group_ids
            or len(set(self.design_split_group_ids)) != len(self.design_split_group_ids)
            or any(not str(value).strip() for value in self.design_split_group_ids)
        ):
            raise ValueError(
                "BaselineSelection design split groups must be non-empty, unique, and sorted."
            )
        if not _is_sha256_digest(self.sha256):
            raise ValueError("BaselineSelection sha256 must be a SHA-256 digest.")


def baseline_selection_payload(selection: BaselineSelection) -> dict[str, Any]:
    """Return the canonical design-only selection artifact payload."""

    content = {
        "schema_version": "path_c_baseline_selection_v1",
        "selected_baseline": str(selection.selected_baseline),
        "design_auc_by_baseline": {
            str(name): float(value)
            for name, value in selection.design_auc_by_baseline.items()
        },
        "candidate_order": list(map(str, selection.candidate_order)),
        "selection_role": str(selection.selection_role),
        "selection_rule": str(selection.selection_rule),
        "probe_budget_grid": list(map(int, selection.probe_budget_grid)),
        "normalization_rule": str(selection.normalization_rule),
        "normalization_lower": float(selection.normalization_lower),
        "normalization_upper": float(selection.normalization_upper),
        "probe_cost_per_use": float(selection.probe_cost_per_use),
        "source_curve_set_sha256_by_baseline": {
            str(name): str(value)
            for name, value in selection.source_curve_set_sha256_by_baseline.items()
        },
        "return_point_ledger_sha256": str(selection.return_point_ledger_sha256),
        "design_split_group_ids": list(map(str, selection.design_split_group_ids)),
    }
    if _canonical_sha256(content) != str(selection.sha256):
        raise ValueError("BaselineSelection SHA-256 does not bind its canonical payload.")
    return {**content, "sha256": str(selection.sha256)}


def write_baseline_selection_artifact(
    requested_path: str | Path,
    selection: BaselineSelection,
    *,
    preregistration_sha256: str,
    resolved_path_c_sha256: str,
    semantic_bindings: Mapping[str, Any],
) -> tuple[Path, str]:
    """Write a bound version-3 measurement with semantic and file hashes."""

    path = Path(requested_path)
    if path.suffix.lower() != ".json":
        raise ValueError("Baseline selection artifact must be a JSON file.")
    if path.exists():
        raise FileExistsError("Baseline selection artifact is immutable and cannot be overwritten.")
    for name, value in (
        ("preregistration_sha256", preregistration_sha256),
        ("resolved_path_c_sha256", resolved_path_c_sha256),
    ):
        if len(str(value)) != 64 or any(
            character not in "0123456789abcdef" for character in str(value)
        ):
            raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    if not semantic_bindings:
        raise ValueError("semantic_bindings must be a non-empty mapping.")
    payload = {
        **baseline_selection_payload(selection),
        "measurement_schema_version": "path_c_measurement_v3",
        "preregistration_sha256": str(preregistration_sha256),
        "resolved_path_c_sha256": str(resolved_path_c_sha256),
        "semantic_bindings": dict(semantic_bindings),
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path, digest


@dataclass(frozen=True)
class PrimaryEndpointResult:
    schema_version: str
    available: bool
    endpoint_name: str
    split_role: str
    selected_baseline: str
    baseline_selection_sha256: str
    locked_return_point_ledger_sha256: str
    identity_shift_only: bool
    probe_budget_grid: tuple[int, ...]
    normalization_rule: str
    normalization_lower: float
    normalization_upper: float
    probe_cost_per_use: float
    estimate: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    cluster_unit: str
    clusters: int
    preregistered_margin: float

    @property
    def primary_effective(self) -> bool:
        return float(self.confidence_interval[0]) > float(self.preregistered_margin)


def build_return_probe_budget_curve(
    points: Sequence[ReturnProbeBudgetPoint],
    *,
    budget_grid: Sequence[int],
    normalization_lower: float,
    normalization_upper: float,
) -> ReturnProbeBudgetCurve:
    """Build one cross-identity curve using net probe-cost return."""

    grid = tuple(int(value) for value in budget_grid)
    _validate_budget_grid(grid)
    if not points:
        raise ValueError("At least one return point is required.")
    upper = float(normalization_upper)
    lower = float(normalization_lower)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("Return normalization requires finite upper > lower.")
    policy_names = {str(point.policy_name) for point in points}
    split_group_ids = {str(point.split_group_id) for point in points}
    mechanisms = {str(point.mechanism) for point in points}
    identity_groups = {str(point.identity_group) for point in points}
    style_groups = {str(point.style_group) for point in points}
    layout_groups = {str(point.layout_group) for point in points}
    seed_groups = {str(point.seed_group) for point in points}
    seeds = {int(point.seed) for point in points}
    if any(len(values) != 1 for values in (
        policy_names,
        split_group_ids,
        mechanisms,
        identity_groups,
        style_groups,
        layout_groups,
        seed_groups,
        seeds,
    )):
        raise ValueError("A curve must contain one policy/identity/layout/seed group.")
    by_budget: dict[int, list[float]] = {value: [] for value in grid}
    schedule_ids: set[str] = set()
    training_environment_steps: set[int] = set()
    gradient_updates: set[int] = set()
    evaluation_environment_step_limits: set[int] = set()
    probe_costs: set[float] = set()
    for point in points:
        budget = int(point.budget)
        if budget not in by_budget:
            raise ValueError(f"Unregistered probe budget {budget}.")
        if int(point.probes_used) > budget:
            raise ValueError("Observed probes_used exceeds the assigned budget.")
        if int(point.probes_used) < 0 or float(point.realized_probe_cost) < 0.0:
            raise ValueError("Probe count and realized probe cost must be non-negative.")
        if (
            int(point.episode_environment_steps) <= 0
            or int(point.training_environment_steps) <= 0
            or int(point.evaluation_environment_step_limit) <= 0
            or int(point.gradient_updates) < 0
        ):
            raise ValueError(
                "Episode/training/limit steps must be positive and updates non-negative."
            )
        by_budget[budget].append(float(point.net_return))
        schedule_ids.add(str(point.evaluation_schedule_id))
        training_environment_steps.add(int(point.training_environment_steps))
        evaluation_environment_step_limits.add(
            int(point.evaluation_environment_step_limit)
        )
        gradient_updates.add(int(point.gradient_updates))
        probe_costs.add(float(point.probe_cost_per_use))
    missing = [value for value, observations in by_budget.items() if not observations]
    if missing:
        raise ValueError(f"Curve is missing budget point(s): {missing}")
    if (
        len(schedule_ids) != 1
        or len(training_environment_steps) != 1
        or len(gradient_updates) != 1
        or len(evaluation_environment_step_limits) != 1
        or len(probe_costs) != 1
    ):
        raise ValueError("All curve points must share the matched evaluation/compute schedule.")
    means = np.asarray(
        [np.mean(by_budget[value], dtype=np.float64) for value in grid],
        dtype=np.float64,
    )
    normalized = (means - lower) / (upper - lower)
    width = float(grid[-1] - grid[0])
    if width == 0.0:
        auc = float(normalized[0])
    else:
        segment_widths = np.diff(np.asarray(grid, dtype=np.float64))
        segment_means = 0.5 * (normalized[:-1] + normalized[1:])
        auc = float(np.sum(segment_widths * segment_means, dtype=np.float64) / width)
    return ReturnProbeBudgetCurve(
        policy_name=next(iter(policy_names)),
        split_group_id=next(iter(split_group_ids)),
        mechanism=next(iter(mechanisms)),
        identity_group=next(iter(identity_groups)),
        style_group=next(iter(style_groups)),
        layout_group=next(iter(layout_groups)),
        seed_group=next(iter(seed_groups)),
        seed=next(iter(seeds)),
        budgets=grid,
        normalized_net_returns=tuple(map(float, normalized.tolist())),
        auc=auc,
        normalization_lower=lower,
        normalization_upper=upper,
        probe_cost_per_use=next(iter(probe_costs)),
        normalization_rule="affine_without_clipping",
    )


def select_strongest_baseline_on_design(
    curves: Sequence[ReturnProbeBudgetCurve],
    *,
    candidate_order: Sequence[str] = DEPLOYABLE_STRONG_BASELINES,
    split_role: str,
    return_point_ledger_sha256: str,
    design_split_group_ids: Sequence[str],
) -> BaselineSelection:
    """Select once on the design split; locked audit selection is rejected."""

    if str(split_role) != "design":
        raise ValueError("Strongest baseline selection is permitted only on the design split.")
    order = tuple(map(str, candidate_order))
    if not order or len(set(order)) != len(order):
        raise ValueError("candidate_order must be a non-empty unique sequence.")
    if not _is_sha256_digest(return_point_ledger_sha256):
        raise ValueError("Design selection requires a return-point ledger SHA-256.")
    registered_design_groups = tuple(sorted(map(str, design_split_group_ids)))
    if (
        not registered_design_groups
        or len(set(registered_design_groups)) != len(registered_design_groups)
        or any(not value.strip() for value in registered_design_groups)
    ):
        raise ValueError("Design selection requires unique registered primary split groups.")
    observed_design_groups = {
        str(curve.split_group_id)
        for curve in curves
        if curve.policy_name in set(order)
    }
    if observed_design_groups != set(registered_design_groups):
        raise ValueError(
            "Design baseline curves must exactly cover the registered primary layout groups."
        )
    aucs: dict[str, float] = {}
    curve_hashes: dict[str, str] = {}
    reference_curves: dict[
        tuple[str, str, str, str, str, int], ReturnProbeBudgetCurve
    ] | None = None
    for name in order:
        candidate_curves = [curve for curve in curves if curve.policy_name == name]
        if not candidate_curves:
            raise ValueError(f"Design split has no curve for baseline {name!r}.")
        keyed: dict[
            tuple[str, str, str, str, str, int], ReturnProbeBudgetCurve
        ] = {}
        for curve in candidate_curves:
            key = (
                str(curve.split_group_id),
                str(curve.mechanism),
                str(curve.identity_group),
                str(curve.style_group),
                str(curve.seed_group),
                int(curve.seed),
            )
            if key in keyed:
                raise ValueError(f"Design baseline {name!r} repeats curve key {key!r}.")
            keyed[key] = curve
        if reference_curves is None:
            reference_curves = keyed
        else:
            if set(keyed) != set(reference_curves):
                raise ValueError(
                    "Design baselines must use identical identity/layout/seed groups "
                    "and the same split, mechanism, and style groups."
                )
            for key, curve in keyed.items():
                reference = reference_curves[key]
                if (
                    curve.budgets != reference.budgets
                    or curve.layout_group != reference.layout_group
                    or curve.normalization_lower != reference.normalization_lower
                    or curve.normalization_upper != reference.normalization_upper
                    or curve.probe_cost_per_use != reference.probe_cost_per_use
                    or curve.normalization_rule != reference.normalization_rule
                ):
                    raise ValueError(
                        "Design baselines must share budget, cost, and normalization contracts."
                    )
        by_identity: dict[str, list[float]] = {}
        for curve in candidate_curves:
            by_identity.setdefault(str(curve.identity_group), []).append(float(curve.auc))
        aucs[name] = float(np.mean(np.asarray([
            np.mean(values, dtype=np.float64)
            for values in by_identity.values()
        ], dtype=np.float64)))
        curve_hashes[name] = _canonical_sha256({
            "policy_name": name,
            "curve_sha256s": sorted(curve.sha256 for curve in candidate_curves),
        })
    selected = max(order, key=lambda name: (aucs[name], -order.index(name)))
    if reference_curves is None:
        raise RuntimeError("Design baseline selection did not construct reference curves.")
    first_reference = next(iter(reference_curves.values()))
    payload = {
        "schema_version": "path_c_baseline_selection_v1",
        "selected_baseline": selected,
        "design_auc_by_baseline": aucs,
        "candidate_order": list(order),
        "selection_role": "design",
        "selection_rule": "highest_mean_normalized_net_return_auc_then_preregistered_order",
        "probe_budget_grid": list(map(int, first_reference.budgets)),
        "normalization_rule": str(first_reference.normalization_rule),
        "normalization_lower": float(first_reference.normalization_lower),
        "normalization_upper": float(first_reference.normalization_upper),
        "probe_cost_per_use": float(first_reference.probe_cost_per_use),
        "source_curve_set_sha256_by_baseline": curve_hashes,
        "return_point_ledger_sha256": str(return_point_ledger_sha256),
        "design_split_group_ids": list(registered_design_groups),
    }
    return BaselineSelection(
        selected_baseline=selected,
        design_auc_by_baseline=aucs,
        candidate_order=order,
        selection_role="design",
        selection_rule=payload["selection_rule"],
        probe_budget_grid=tuple(first_reference.budgets),
        normalization_rule=str(first_reference.normalization_rule),
        normalization_lower=float(first_reference.normalization_lower),
        normalization_upper=float(first_reference.normalization_upper),
        probe_cost_per_use=float(first_reference.probe_cost_per_use),
        source_curve_set_sha256_by_baseline=curve_hashes,
        return_point_ledger_sha256=str(return_point_ledger_sha256),
        design_split_group_ids=registered_design_groups,
        sha256=_canonical_sha256(payload),
    )


def estimate_locked_primary_endpoint(
    method_curves: Sequence[ReturnProbeBudgetCurve],
    baseline_curves: Sequence[ReturnProbeBudgetCurve],
    *,
    baseline_selection: BaselineSelection,
    locked_return_point_ledger_sha256: str,
    split_role: str,
    primary_split_group_ids: Sequence[str],
    preregistered_margin: float,
    confidence_level: float,
    bootstrap_iterations: int,
    seed: int,
) -> PrimaryEndpointResult:
    """Paired identity-cluster estimate of the locked primary AUC difference."""

    # Recompute the selection digest before using its chosen baseline. A manually
    # constructed or deserialized object cannot bypass the design-artifact binding.
    baseline_selection_payload(baseline_selection)
    if not _is_sha256_digest(locked_return_point_ledger_sha256):
        raise ValueError("Primary endpoint requires a locked return-point ledger SHA-256.")
    if str(split_role) != "locked_audit":
        raise ValueError("The primary endpoint is computed only on locked_audit.")
    registered_primary_groups = tuple(map(str, primary_split_group_ids))
    if (
        not registered_primary_groups
        or len(registered_primary_groups) != len(set(registered_primary_groups))
        or any(not value.strip() for value in registered_primary_groups)
    ):
        raise ValueError("Primary split-group ids must be a non-empty unique sequence.")
    if baseline_selection.selection_role != "design":
        raise ValueError("The baseline selection artifact must originate on design.")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")
    if int(bootstrap_iterations) <= 0:
        raise ValueError("bootstrap_iterations must be positive.")
    baseline_name = str(baseline_selection.selected_baseline)
    if not method_curves or any(
        curve.policy_name != "probing_ego" for curve in method_curves
    ):
        raise ValueError("Primary method curves must all belong to probing_ego.")
    method_map = _curves_by_paired_unit(method_curves)
    baseline_map = _curves_by_paired_unit(
        [curve for curve in baseline_curves if curve.policy_name == baseline_name]
    )
    if set(method_map) != set(baseline_map) or not method_map:
        raise ValueError("Method and selected baseline must share every locked identity cluster.")
    observed_primary_groups = {
        str(curve.split_group_id) for curve in method_map.values()
    }
    if observed_primary_groups != set(registered_primary_groups):
        raise ValueError(
            "Primary curves must exactly cover the split-manifest primary "
            "layout-stratum groups."
        )
    # Primary inference holds layout fixed within each matched cluster. Layout shift
    # is a separately labelled secondary endpoint.
    if any(method_map[key].layout_group != baseline_map[key].layout_group for key in method_map):
        raise ValueError("Primary identity-shift comparison must not confound layout shift.")
    for key in method_map:
        method_curve = method_map[key]
        baseline_curve = baseline_map[key]
        if (
            method_curve.budgets != baseline_curve.budgets
            or method_curve.normalization_lower != baseline_curve.normalization_lower
            or method_curve.normalization_upper != baseline_curve.normalization_upper
            or method_curve.probe_cost_per_use != baseline_curve.probe_cost_per_use
            or method_curve.normalization_rule != baseline_curve.normalization_rule
        ):
            raise ValueError(
                "Primary method and baseline curves must share budget and normalization."
            )
    reference_curve = method_map[sorted(method_map)[0]]
    for curve in (*method_map.values(), *baseline_map.values()):
        if (
            curve.budgets != reference_curve.budgets
            or curve.normalization_lower != reference_curve.normalization_lower
            or curve.normalization_upper != reference_curve.normalization_upper
            or curve.probe_cost_per_use != reference_curve.probe_cost_per_use
            or curve.normalization_rule != reference_curve.normalization_rule
        ):
            raise ValueError(
                "All primary identity clusters must share one frozen budget, cost, "
                "and normalization contract."
            )
    paired_differences: dict[str, list[float]] = {}
    for key in sorted(method_map):
        _split_group, _mechanism, identity_group, _style, _seed_group, _seed = key
        paired_differences.setdefault(identity_group, []).append(
            float(method_map[key].auc - baseline_map[key].auc)
        )
    differences = np.asarray([
        np.mean(paired_differences[identity], dtype=np.float64)
        for identity in sorted(paired_differences)
    ], dtype=np.float64)
    estimate = float(np.mean(differences))
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(bootstrap_iterations), dtype=np.float64)
    for index in range(int(bootstrap_iterations)):
        draw = rng.integers(0, differences.size, size=differences.size)
        boot[index] = float(np.mean(differences[draw]))
    tail = (1.0 - float(confidence_level)) / 2.0
    ci = tuple(map(float, np.quantile(boot, [tail, 1.0 - tail]).tolist()))
    return PrimaryEndpointResult(
        schema_version=PATH_C_PRIMARY_ENDPOINT_VERSION,
        available=True,
        endpoint_name="cross_identity_normalized_net_return_probe_budget_auc_difference",
        split_role="locked_audit",
        selected_baseline=baseline_name,
        baseline_selection_sha256=str(baseline_selection.sha256),
        locked_return_point_ledger_sha256=str(
            locked_return_point_ledger_sha256
        ),
        identity_shift_only=True,
        probe_budget_grid=tuple(reference_curve.budgets),
        normalization_rule=str(reference_curve.normalization_rule),
        normalization_lower=float(reference_curve.normalization_lower),
        normalization_upper=float(reference_curve.normalization_upper),
        probe_cost_per_use=float(reference_curve.probe_cost_per_use),
        estimate=estimate,
        confidence_interval=(ci[0], ci[1]),
        confidence_level=float(confidence_level),
        cluster_unit="identity_group",
        clusters=int(differences.size),
        preregistered_margin=float(preregistered_margin),
    )


def matched_policy_metrics(
    policy_metrics: Mapping[str, Mapping[str, Any]],
    *,
    required_policies: Sequence[str],
) -> dict[str, Any]:
    """Fail-closed fairness audit over interaction, compute, and latency fields."""

    required = tuple(map(str, required_policies))
    if not required or len(set(required)) != len(required):
        raise ValueError("required_policies must be a non-empty unique sequence.")
    missing = sorted(set(required).difference(policy_metrics))
    fields = (
        "training_environment_steps",
        "gradient_updates",
        "evaluation_environment_step_limit",
        "evaluation_schedule_id",
        "probe_budget_grid_sha256",
        "probe_cost_per_use",
        "action_support_sha256",
    )
    mismatches: dict[str, dict[str, Any]] = {}
    if not missing:
        reference = policy_metrics[required[0]]
        for field_name in fields:
            observed = {name: policy_metrics[name].get(field_name) for name in required}
            if any(value is None for value in observed.values()) or len({
                _json_comparable(value) for value in observed.values()
            }) != 1:
                mismatches[field_name] = observed
        probe_costs = {
            name: policy_metrics[name].get("probe_cost_per_use")
            for name in required
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in probe_costs.values()
        ):
            mismatches["probe_cost_per_use"] = probe_costs
    resource_fields = (
        "trainable_parameters",
        "training_flops",
        "wall_clock_seconds",
        "inference_latency_ms",
    )
    resources = {
        name: {field_name: policy_metrics.get(name, {}).get(field_name) for field_name in resource_fields}
        for name in required
    }
    return {
        "complete": not missing,
        "missing_policies": missing,
        "matched_fields": list(fields),
        "mismatches": mismatches,
        "resource_pareto_metrics": resources,
        "fairness_conformant": not missing and not mismatches,
    }


def estimate_audit_cost(
    *,
    audit_units: int,
    probes: int,
    outer_replicas_M: int,
    inner_forks_L_inner: int,
    horizon_T_probe: int,
    max_primitive_steps: int,
    maximum_primitive_step_budget: int,
    design_audit_units: int = 0,
    design_only_probes: int = 0,
) -> dict[str, Any]:
    """Static launch-cost estimate; it deliberately does not execute a rollout."""

    raw_values = {
        "audit_units": audit_units,
        "probes": probes,
        "outer_replicas_M": outer_replicas_M,
        "inner_forks_L_inner": inner_forks_L_inner,
        "horizon_T_probe": horizon_T_probe,
        "max_primitive_steps": max_primitive_steps,
        "maximum_primitive_step_budget": maximum_primitive_step_budget,
        "design_audit_units": design_audit_units,
        "design_only_probes": design_only_probes,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw_values.values()
    ):
        raise TypeError("Every cost-estimator dimension must be an integer.")
    values = {name: int(value) for name, value in raw_values.items()}
    positive_names = {
        "audit_units",
        "probes",
        "outer_replicas_M",
        "inner_forks_L_inner",
        "horizon_T_probe",
        "max_primitive_steps",
        "maximum_primitive_step_budget",
    }
    if any(values[name] <= 0 for name in positive_names):
        raise ValueError("Every cost-estimator dimension must be positive.")
    if values["design_audit_units"] < 0 or values["design_only_probes"] < 0:
        raise ValueError("Design-extension cost dimensions cannot be negative.")
    if (values["design_audit_units"] == 0) != (values["design_only_probes"] == 0):
        raise ValueError(
            "Design audit units and design-only probes must either both be zero "
            "or both be positive."
        )
    core_continuations = (
        values["audit_units"]
        * values["probes"]
        * values["outer_replicas_M"]
        * values["inner_forks_L_inner"]
    )
    design_extension_continuations = (
        values["design_audit_units"]
        * values["design_only_probes"]
        * values["outer_replicas_M"]
        * values["inner_forks_L_inner"]
    )
    continuations = core_continuations + design_extension_continuations
    option_steps = continuations * values["horizon_T_probe"]
    primitive_step_upper_bound = option_steps * values["max_primitive_steps"]
    return {
        "schema_version": "path_c_audit_cost_estimate_v1",
        **values,
        "core_continuations": core_continuations,
        "design_extension_continuations": design_extension_continuations,
        "continuations": continuations,
        "option_steps": option_steps,
        "primitive_step_upper_bound": primitive_step_upper_bound,
        "within_budget": bool(
            primitive_step_upper_bound <= values["maximum_primitive_step_budget"]
        ),
    }


def simulate_cluster_operating_characteristics(
    positive_cluster_effects: Sequence[float],
    null_cluster_effects: Sequence[float],
    *,
    clusters_per_trial: int,
    simulation_repetitions: int,
    bootstrap_iterations: int,
    alpha: float,
    efficacy_margin: float,
    equivalence_margin: float,
    seed: int,
) -> dict[str, Any]:
    """Estimate observed effects and positive/null/joint operating characteristics.

    The superiority readout uses the ordinary two-sided ``1 - alpha`` interval.
    Equivalence uses two one-sided tests at level ``alpha`` and therefore the
    corresponding two-sided ``1 - 2 * alpha`` interval.  Keeping these intervals
    separate prevents a 95% descriptive interval from being mislabeled as the
    90% equivalence interval when ``alpha=0.05``.
    """

    positive = np.asarray(positive_cluster_effects, dtype=np.float64)
    null = np.asarray(null_cluster_effects, dtype=np.float64)
    if positive.ndim != 1 or null.ndim != 1 or positive.size < 2 or null.size < 2:
        raise ValueError("Power simulation needs at least two positive and null clusters.")
    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(null)):
        raise ValueError("Power simulation cluster effects must be finite.")
    clusters = int(clusters_per_trial)
    repetitions = int(simulation_repetitions)
    bootstraps = int(bootstrap_iterations)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in (clusters_per_trial, simulation_repetitions, bootstrap_iterations)
    ):
        raise ValueError("Power simulation counts must be positive integers.")
    if clusters < 2 or bootstraps < 2:
        raise ValueError(
            "Power simulation needs at least two clusters per trial and two "
            "bootstrap iterations."
        )
    alpha_value = float(alpha)
    if not 0.0 < alpha_value < 0.5:
        raise ValueError("Power simulation alpha must lie in (0, 0.5) for TOST.")
    if not math.isfinite(float(efficacy_margin)):
        raise ValueError("Power simulation efficacy_margin must be finite.")
    if not math.isfinite(float(equivalence_margin)) or float(equivalence_margin) <= 0.0:
        raise ValueError("Power simulation equivalence_margin must be positive and finite.")

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("Power simulation seed must be a non-negative integer.")
    seed_streams = np.random.SeedSequence(int(seed)).spawn(4)
    trial_rng = np.random.default_rng(seed_streams[0])
    positive_interval_rng = np.random.default_rng(seed_streams[1])
    null_interval_rng = np.random.default_rng(seed_streams[2])
    null_tost_rng = np.random.default_rng(seed_streams[3])
    positive_success = np.zeros(repetitions, dtype=bool)
    null_success = np.zeros(repetitions, dtype=bool)
    descriptive_tail = alpha_value / 2.0
    tost_tail = alpha_value

    def interval(
        sample: np.ndarray,
        *,
        tail_probability: float,
        generator: np.random.Generator,
    ) -> tuple[float, float]:
        draws = generator.integers(
            0,
            sample.size,
            size=(bootstraps, sample.size),
        )
        means = sample[draws].mean(axis=1, dtype=np.float64)
        lower, upper = np.quantile(
            means,
            (tail_probability, 1.0 - tail_probability),
        )
        return float(lower), float(upper)

    positive_effect = float(positive.mean(dtype=np.float64))
    null_effect = float(null.mean(dtype=np.float64))
    positive_confidence_interval = interval(
        positive,
        tail_probability=descriptive_tail,
        generator=positive_interval_rng,
    )
    null_confidence_interval = interval(
        null,
        tail_probability=descriptive_tail,
        generator=null_interval_rng,
    )
    null_tost_confidence_interval = interval(
        null,
        tail_probability=tost_tail,
        generator=null_tost_rng,
    )
    null_lower_test = bool(
        null_tost_confidence_interval[0] > -float(equivalence_margin)
    )
    null_upper_test = bool(
        null_tost_confidence_interval[1] < float(equivalence_margin)
    )

    for repetition in range(repetitions):
        positive_sample = trial_rng.choice(positive, size=clusters, replace=True)
        null_sample = trial_rng.choice(null, size=clusters, replace=True)
        positive_interval = interval(
            positive_sample,
            tail_probability=descriptive_tail,
            generator=trial_rng,
        )
        null_interval = interval(
            null_sample,
            tail_probability=tost_tail,
            generator=trial_rng,
        )
        positive_success[repetition] = (
            positive_interval[0] > float(efficacy_margin)
        )
        null_success[repetition] = bool(
            null_interval[0] > -float(equivalence_margin)
            and null_interval[1] < float(equivalence_margin)
        )

    positive_power = float(positive_success.mean())
    null_power = float(null_success.mean())
    joint_power = float((positive_success & null_success).mean())

    def monte_carlo_standard_error(probability: float) -> float:
        return float(
            math.sqrt(probability * (1.0 - probability) / repetitions)
        )

    positive_report = {
        "effect": positive_effect,
        "confidence_level": float(1.0 - alpha_value),
        "confidence_interval": positive_confidence_interval,
        "efficacy_margin": float(efficacy_margin),
        "observed_success": bool(
            positive_confidence_interval[0] > float(efficacy_margin)
        ),
        "power": positive_power,
        "power_monte_carlo_standard_error": monte_carlo_standard_error(
            positive_power
        ),
    }
    null_report = {
        "effect": null_effect,
        "confidence_level": float(1.0 - alpha_value),
        "confidence_interval": null_confidence_interval,
        "equivalence_margin": float(equivalence_margin),
        "two_one_sided_tests": {
            "alpha_per_test": alpha_value,
            "confidence_level": float(1.0 - 2.0 * alpha_value),
            "confidence_interval": null_tost_confidence_interval,
            "lower_bound_above_negative_margin": null_lower_test,
            "upper_bound_below_positive_margin": null_upper_test,
            "equivalent": bool(null_lower_test and null_upper_test),
        },
        "power": null_power,
        "power_monte_carlo_standard_error": monte_carlo_standard_error(
            null_power
        ),
    }
    joint_report = {
        "power": joint_power,
        "power_monte_carlo_standard_error": monte_carlo_standard_error(
            joint_power
        ),
    }
    return {
        "schema_version": "path_c_cluster_power_simulation_v1",
        "cluster_unit": "identity_group",
        "clusters_per_trial": clusters,
        "simulation_repetitions": repetitions,
        "bootstrap_iterations": bootstraps,
        "alpha": alpha_value,
        "efficacy_margin": float(efficacy_margin),
        "equivalence_margin": float(equivalence_margin),
        "positive": positive_report,
        "null": null_report,
        "joint": joint_report,
        "positive_power": positive_power,
        "null_equivalence_power": null_power,
        "joint_power": joint_power,
        "joint_power_monte_carlo_standard_error": joint_report[
            "power_monte_carlo_standard_error"
        ],
        "seed": int(seed),
    }


def _curves_by_paired_unit(
    curves: Sequence[ReturnProbeBudgetCurve],
) -> dict[tuple[str, str, str, str, str, int], ReturnProbeBudgetCurve]:
    result: dict[tuple[str, str, str, str, str, int], ReturnProbeBudgetCurve] = {}
    for curve in curves:
        key = (
            str(curve.split_group_id),
            str(curve.mechanism),
            str(curve.identity_group),
            str(curve.style_group),
            str(curve.seed_group),
            int(curve.seed),
        )
        if key in result:
            raise ValueError(f"Duplicate primary cluster {key!r}.")
        result[key] = curve
    return result


def _validate_budget_grid(values: Sequence[int]) -> None:
    if isinstance(values, (str, bytes)) or any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in values
    ):
        raise TypeError("Probe budget grid must contain integers.")
    grid = tuple(int(value) for value in values)
    if not grid or grid[0] != 0:
        raise ValueError("Probe budget grid must be non-empty and begin at zero.")
    if any(value < 0 for value in grid) or any(
        right <= left for left, right in zip(grid, grid[1:], strict=False)
    ):
        raise ValueError("Probe budget grid must be strictly increasing and non-negative.")


def _json_comparable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
