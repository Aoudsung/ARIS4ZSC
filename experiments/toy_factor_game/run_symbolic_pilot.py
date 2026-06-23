"""
Symbolic diagnostic sanity suite for the toy factor game.

This runner is intentionally separate from the neural ARIS-Bellman experiments.
It first synthesizes diagnostic-critical cases, then checks whether task-valid
partner observations create Bellman value that predicts downstream return.
"""

import argparse
import copy
import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.metrics import bootstrap_ci, expected_calibration_error, pareto_frontier
from common.utils import save_results, set_seed
from toy_factor_game.env import (
    BOTTLENECK_COL,
    BOTTLENECK_ROW,
    DELIVER_LEFT,
    DELIVER_RIGHT,
    FACTOR_MODES,
    NUM_FACTORS,
    RESOURCE_A,
    RESOURCE_B,
    Action,
    ConventionAssignment,
    ToyFactorGameEnv,
)
from toy_factor_game.options import OPTION_NAMES, OptionID, get_option_action


METHODS = ("gtvoi", "mi", "passive", "random", "oracle", "oracle_greedy")
DEFAULT_METHODS = ("gtvoi", "mi", "passive", "random", "oracle")
ROLLOUT_DEPTH = 3
ROLLOUT_GAMMA = 0.99
DELTA_INFO_THRESHOLD = 0.05
SYMBOLIC_SCHEMA = "symbolic_sanity_v4_phase3"
DEFAULT_DELTA_GAP = 0.5
DEFAULT_DELTA_ACTION = 0.3
DEFAULT_DELTA_OBS = 0.3
DEFAULT_DELTA_RETURN = 0.3
DEFAULT_DELTA_MI = 0.1
DEFAULT_DELTA_VALUE = 0.1
DEFAULT_MIN_CASES = 3
DEFAULT_MAX_DIAGNOSTIC_COST = 1.0
DEFAULT_ORACLE_MARGIN = 0.1
DEFAULT_DEBUG_TOP_K = 20

ConventionKey = tuple[int, ...]


@dataclass(frozen=True)
class DiagnosticScenario:
    name: str
    description: str
    conventions: tuple[ConventionKey, ...]
    ego_pos: tuple[int, int]
    partner_pos: tuple[int, int]
    ego_carrying: bool = False
    partner_carrying: bool = False
    resource_a_available: bool = True
    resource_b_available: bool = True
    deliveries_left: int = 0
    deliveries_right: int = 0
    step_count: int = 0


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    case_type: str
    scenario: DiagnosticScenario | None
    conventions: tuple[ConventionKey, ...]
    oracle_passive_gap: float | None = None
    oracle_value: float | None = None
    passive_value: float | None = None
    oracle_first_actions: tuple[str, ...] = ()
    passive_first_action: str | None = None
    best_response_flip: bool = False
    action_gap: float | None = None
    best_diag_option: str | None = None
    observation_separation: float | None = None
    diagnostic_opportunity_cost: float | None = None
    best_diag_return_gain: float | None = None
    best_diag_delta_info: float | None = None
    best_diag_mi_gain: float | None = None
    max_delta_info: float | None = None
    high_mi_low_value_distractor: bool = False
    mi_option: str | None = None
    gtvoi_option: str | None = None
    mi_gap: float | None = None
    delta_info_gap: float | None = None
    distractor_return_gap: float | None = None
    passed_filters: bool = False
    failure_reason: str | None = None


HANDPICKED_DIAGNOSTIC_SCENARIOS = (
    DiagnosticScenario(
        name="bottleneck_critical",
        description="Agents face each other at the bottleneck; yielding convention changes best action.",
        conventions=((0, 0, 0), (1, 0, 0), (2, 0, 0)),
        ego_pos=(BOTTLENECK_ROW, BOTTLENECK_COL - 1),
        partner_pos=(BOTTLENECK_ROW, BOTTLENECK_COL + 1),
    ),
    DiagnosticScenario(
        name="resource_critical",
        description="Ego is near resource A; ownership convention determines whether A or B is valuable.",
        conventions=((0, 0, 0), (0, 1, 0)),
        ego_pos=(RESOURCE_A[0], RESOURCE_A[1] - 1),
        partner_pos=(RESOURCE_A[0] + 1, RESOURCE_A[1]),
    ),
    DiagnosticScenario(
        name="delivery_critical",
        description="Both agents carry soup; delivery role convention changes the correct target.",
        conventions=((0, 0, 0), (0, 0, 1)),
        ego_pos=(BOTTLENECK_ROW - 1, BOTTLENECK_COL),
        partner_pos=(BOTTLENECK_ROW + 1, BOTTLENECK_COL),
        ego_carrying=True,
        partner_carrying=True,
        resource_a_available=False,
        resource_b_available=False,
    ),
    DiagnosticScenario(
        name="high_mi_low_value_distractor",
        description="Resource observations are easy, but bottleneck yielding is the value-critical factor.",
        conventions=((0, 0, 0), (1, 1, 0)),
        ego_pos=(BOTTLENECK_ROW, BOTTLENECK_COL - 1),
        partner_pos=(BOTTLENECK_ROW, BOTTLENECK_COL + 1),
    ),
)


def all_convention_keys() -> list[ConventionKey]:
    keys = []
    for f0 in range(FACTOR_MODES[0]):
        for f1 in range(FACTOR_MODES[1]):
            for f2 in range(FACTOR_MODES[2]):
                keys.append((f0, f1, f2))
    return keys


def key_to_assignment(key: ConventionKey) -> ConventionAssignment:
    return ConventionAssignment(modes={factor: key[factor] for factor in range(NUM_FACTORS)})


def convention_label(key: ConventionKey) -> str:
    return "-".join(str(value) for value in key)


def entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0.0]
    if len(probs) == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


@dataclass
class BeliefState:
    """Joint posterior over the toy game's three ground-truth factor modes."""

    weights: dict[ConventionKey, float]

    @classmethod
    def uniform(cls, keys: Iterable[ConventionKey] | None = None) -> "BeliefState":
        keys = list(keys) if keys is not None else all_convention_keys()
        prob = 1.0 / len(keys)
        return cls({key: prob for key in keys})

    @classmethod
    def point_mass(cls, key: ConventionKey) -> "BeliefState":
        return cls({candidate: float(candidate == key) for candidate in all_convention_keys()})

    def normalized(self) -> "BeliefState":
        total = sum(self.weights.values())
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Belief normalization failed: posterior mass is zero or non-finite")
        return BeliefState({key: value / total for key, value in self.weights.items()})

    def marginals(self) -> list[np.ndarray]:
        out = [np.zeros(FACTOR_MODES[factor], dtype=np.float64) for factor in range(NUM_FACTORS)]
        for key, prob in self.weights.items():
            for factor_idx, mode in enumerate(key):
                out[factor_idx][mode] += prob
        return out

    def entropy_vector(self) -> np.ndarray:
        return np.array([entropy(marginal) for marginal in self.marginals()], dtype=np.float64)

    def mode_predictions(self) -> list[int]:
        return [int(np.argmax(marginal)) for marginal in self.marginals()]

    def confidence_for_truth(self, truth: ConventionKey) -> list[float]:
        marginals = self.marginals()
        return [float(marginals[factor][truth[factor]]) for factor in range(NUM_FACTORS)]

    def update_from_partner_action(
        self,
        env_before: ToyFactorGameEnv,
        observed_partner_action: int,
        likelihood_error: float,
    ) -> "BeliefState":
        if not 0.0 <= likelihood_error < 1.0:
            raise ValueError("--likelihood_error must be in [0, 1)")

        miss_prob = likelihood_error / max(1, len(Action) - 1)
        updated = {}
        for key, prior in self.weights.items():
            predicted = int(predict_partner_action(env_before, key))
            likelihood = 1.0 - likelihood_error if predicted == observed_partner_action else miss_prob
            updated[key] = prior * likelihood
        return BeliefState(updated).normalized()


def predict_partner_action(env_before: ToyFactorGameEnv, convention: ConventionKey) -> Action:
    candidate = copy.deepcopy(env_before)
    candidate.partner_convention = key_to_assignment(convention)
    return candidate._partner_action()


def env_cache_key(env: ToyFactorGameEnv) -> tuple:
    return (
        tuple(env.ego_pos),
        tuple(env.partner_pos),
        bool(env.ego_carrying),
        bool(env.partner_carrying),
        bool(env.resource_a_available),
        bool(env.resource_b_available),
        int(env.deliveries_left),
        int(env.deliveries_right),
        int(env.step_count),
    )


def belief_cache_key(belief: BeliefState) -> tuple[float, ...]:
    return tuple(
        value
        for key, prob in sorted(belief.weights.items())
        for value in (*key, round(float(prob), 8))
    )


def apply_scenario(env: ToyFactorGameEnv, scenario: DiagnosticScenario | None) -> None:
    if scenario is None:
        return
    env.step_count = scenario.step_count
    env.ego_pos = list(scenario.ego_pos)
    env.partner_pos = list(scenario.partner_pos)
    env.ego_carrying = scenario.ego_carrying
    env.partner_carrying = scenario.partner_carrying
    env.resource_a_available = scenario.resource_a_available
    env.resource_b_available = scenario.resource_b_available
    env.deliveries_left = scenario.deliveries_left
    env.deliveries_right = scenario.deliveries_right
    env.collisions = 0
    env.total_reward = 0.0


def valid_options(env: ToyFactorGameEnv) -> list[OptionID]:
    options = [OptionID.NOOP, OptionID.CROSS_CORRIDOR, OptionID.WAIT_AT_BOTTLENECK]
    if env.ego_carrying:
        options.extend([OptionID.DELIVER_LEFT, OptionID.DELIVER_RIGHT])
        if tuple(env.ego_pos) in (DELIVER_LEFT, DELIVER_RIGHT):
            options.append(OptionID.DROP)
    else:
        if env.resource_a_available:
            options.append(OptionID.GOTO_RESOURCE_A)
        if env.resource_b_available:
            options.append(OptionID.GOTO_RESOURCE_B)
        if tuple(env.ego_pos) in (RESOURCE_A, RESOURCE_B):
            options.append(OptionID.PICKUP)
    return sorted(set(options), key=int)


def option_target(option: OptionID):
    if option == OptionID.GOTO_RESOURCE_A:
        return RESOURCE_A
    if option == OptionID.GOTO_RESOURCE_B:
        return RESOURCE_B
    if option == OptionID.DELIVER_LEFT:
        return DELIVER_LEFT
    if option == OptionID.DELIVER_RIGHT:
        return DELIVER_RIGHT
    if option == OptionID.CROSS_CORRIDOR:
        return (BOTTLENECK_ROW, BOTTLENECK_COL)
    if option == OptionID.WAIT_AT_BOTTLENECK:
        return (BOTTLENECK_ROW, BOTTLENECK_COL - 1)
    return None


def manhattan(a: Iterable[int], b: Iterable[int]) -> int:
    ar, ac = a
    br, bc = b
    return abs(ar - br) + abs(ac - bc)


def greedy_task_option(env: ToyFactorGameEnv, belief: BeliefState) -> OptionID:
    marginals = belief.marginals()

    if env.ego_carrying:
        if tuple(env.ego_pos) in (DELIVER_LEFT, DELIVER_RIGHT):
            return OptionID.DROP
        return OptionID.DELIVER_LEFT if marginals[2][0] >= marginals[2][1] else OptionID.DELIVER_RIGHT

    if tuple(env.ego_pos) in (RESOURCE_A, RESOURCE_B):
        return OptionID.PICKUP

    if env.resource_a_available and env.resource_b_available:
        return OptionID.GOTO_RESOURCE_A if marginals[1][0] >= marginals[1][1] else OptionID.GOTO_RESOURCE_B
    if env.resource_a_available:
        return OptionID.GOTO_RESOURCE_A
    if env.resource_b_available:
        return OptionID.GOTO_RESOURCE_B
    return OptionID.NOOP


def factor_observation_weights(env: ToyFactorGameEnv, option: OptionID) -> np.ndarray:
    target = option_target(option) or tuple(env.ego_pos)
    bottleneck_weight = 1.0 if manhattan(target, (BOTTLENECK_ROW, BOTTLENECK_COL)) <= 2 else 0.2
    resource_weight = 1.0 if min(manhattan(target, RESOURCE_A), manhattan(target, RESOURCE_B)) <= 3 else 0.25
    delivery_weight = 1.0 if min(manhattan(target, DELIVER_LEFT), manhattan(target, DELIVER_RIGHT)) <= 3 else 0.2
    if option in (OptionID.GOTO_RESOURCE_A, OptionID.GOTO_RESOURCE_B, OptionID.PICKUP):
        resource_weight = 1.2
    if option in (OptionID.DELIVER_LEFT, OptionID.DELIVER_RIGHT, OptionID.DROP):
        delivery_weight = 1.2
    if option == OptionID.CROSS_CORRIDOR:
        bottleneck_weight = 1.4
    elif option == OptionID.WAIT_AT_BOTTLENECK:
        bottleneck_weight = 1.15
    return np.array([bottleneck_weight, resource_weight, delivery_weight], dtype=np.float64)


def option_coordination_bonus(env: ToyFactorGameEnv, option: OptionID, belief: BeliefState) -> float:
    marginals = belief.marginals()
    bonus = 0.0
    near_bottleneck = (
        env.ego_pos[0] == BOTTLENECK_ROW
        or env.partner_pos[0] == BOTTLENECK_ROW
        or option in (OptionID.CROSS_CORRIDOR, OptionID.WAIT_AT_BOTTLENECK)
    )
    if near_bottleneck:
        if option == OptionID.CROSS_CORRIDOR:
            bonus += 2.0 * marginals[0][1] + 0.7 * marginals[0][2] - 2.0 * marginals[0][0]
        elif option == OptionID.WAIT_AT_BOTTLENECK:
            bonus += 1.6 * marginals[0][0] + 0.4 * marginals[0][2] - 0.9 * marginals[0][1]

    if not env.ego_carrying and (env.resource_a_available or env.resource_b_available):
        if option == OptionID.GOTO_RESOURCE_A:
            bonus += 1.7 * marginals[1][0] - 1.2 * marginals[1][1]
        elif option == OptionID.GOTO_RESOURCE_B:
            bonus += 1.5 * marginals[1][1] - 0.7 * marginals[1][0]

    if env.ego_carrying:
        if option == OptionID.DELIVER_LEFT:
            bonus += 2.0 * marginals[2][0] - 1.1 * marginals[2][1]
        elif option == OptionID.DELIVER_RIGHT:
            bonus += 2.0 * marginals[2][1] - 1.1 * marginals[2][0]
    return float(bonus)


def progress_potential(env: ToyFactorGameEnv, belief: BeliefState) -> float:
    if env.deliveries_left > 0 and env.deliveries_right > 0:
        return 0.0
    marginals = belief.marginals()
    if env.ego_carrying:
        target = DELIVER_LEFT if marginals[2][0] >= marginals[2][1] else DELIVER_RIGHT
        return 3.0 - 0.25 * manhattan(env.ego_pos, target)
    if tuple(env.ego_pos) in (RESOURCE_A, RESOURCE_B):
        return 2.0
    if env.resource_a_available and env.resource_b_available:
        target = RESOURCE_A if marginals[1][0] >= marginals[1][1] else RESOURCE_B
        return 1.5 - 0.2 * manhattan(env.ego_pos, target)
    if env.resource_a_available:
        return 1.2 - 0.2 * manhattan(env.ego_pos, RESOURCE_A)
    if env.resource_b_available:
        return 1.2 - 0.2 * manhattan(env.ego_pos, RESOURCE_B)
    return 0.0


def future_potential(env: ToyFactorGameEnv, belief: BeliefState) -> float:
    options = valid_options(env)
    coordination = max((option_coordination_bonus(env, option, belief) for option in options), default=0.0)
    return float(progress_potential(env, belief) + coordination)


def expected_entropy_after_option(
    env: ToyFactorGameEnv,
    option: OptionID,
    belief: BeliefState,
    likelihood_error: float,
) -> np.ndarray:
    action = get_option_action(option, env.ego_pos, env.ego_carrying)
    obs_prob: dict[int, float] = {}

    for key, prob in belief.weights.items():
        branch = copy.deepcopy(env)
        branch.partner_convention = key_to_assignment(key)
        _obs, _reward, _done, info = branch.step(action)
        observed = int(info["partner_action"])
        obs_prob[observed] = obs_prob.get(observed, 0.0) + prob

    expected = np.zeros(NUM_FACTORS, dtype=np.float64)
    for observed, prob in obs_prob.items():
        posterior = belief.update_from_partner_action(env, observed, likelihood_error)
        expected += prob * posterior.entropy_vector()
    return expected


def rollout_value(
    env: ToyFactorGameEnv,
    belief: BeliefState,
    depth: int,
    gamma: float,
    true_key: ConventionKey,
    likelihood_error: float,
    use_observation: bool,
    cache: dict[tuple, float] | None = None,
) -> float:
    if depth <= 0:
        return 0.0
    cache_key = None
    if cache is not None:
        cache_key = (
            "rollout",
            env_cache_key(env),
            belief_cache_key(belief),
            int(depth),
            true_key,
            bool(use_observation),
        )
        if cache_key in cache:
            return cache[cache_key]
    options = valid_options(env)
    if not options:
        value = 0.0
    else:
        value = max(
            option_lookahead_value(
                env,
                option,
                belief,
                gamma,
                depth,
                likelihood_error,
                use_observation,
                cache,
            )
            for option in options
        )
    if cache is not None and cache_key is not None:
        cache[cache_key] = value
    return value


def option_lookahead_value(
    env: ToyFactorGameEnv,
    option: OptionID,
    belief: BeliefState,
    gamma: float,
    depth: int,
    likelihood_error: float,
    use_observation: bool,
    cache: dict[tuple, float] | None = None,
) -> float:
    cache_key = None
    if cache is not None:
        cache_key = (
            "option",
            env_cache_key(env),
            belief_cache_key(belief),
            int(option),
            int(depth),
            bool(use_observation),
        )
        if cache_key in cache:
            return cache[cache_key]
    action = get_option_action(option, env.ego_pos, env.ego_carrying)
    total = 0.0
    for key, prob in belief.weights.items():
        if prob <= 0.0:
            continue
        branch = copy.deepcopy(env)
        branch.partner_convention = key_to_assignment(key)
        env_before = copy.deepcopy(branch)
        _obs, reward, done, info = branch.step(action)
        if done:
            total += prob * float(reward)
            continue
        posterior = (
            belief.update_from_partner_action(env_before, int(info["partner_action"]), likelihood_error)
            if use_observation
            else belief
        )
        future = rollout_value(branch, posterior, depth - 1, gamma, key, likelihood_error, use_observation, cache)
        total += prob * (float(reward) + gamma * future)
    value = float(total)
    if cache is not None and cache_key is not None:
        cache[cache_key] = value
    return value


def state_value(
    env: ToyFactorGameEnv,
    belief: BeliefState,
    gamma: float,
    depth: int,
    likelihood_error: float,
    use_observation: bool,
    cache: dict[tuple, float] | None = None,
) -> float:
    options = valid_options(env)
    if not options:
        return 0.0
    return max(
        option_lookahead_value(env, option, belief, gamma, depth, likelihood_error, use_observation, cache)
        for option in options
    )


def score_options(
    env: ToyFactorGameEnv,
    belief: BeliefState,
    method: str,
    alpha: float,
    beta: float,
    likelihood_error: float,
    truth: ConventionKey,
    cache: dict[tuple, float] | None = None,
) -> list[tuple[float, dict[str, float | str], OptionID]]:
    options = valid_options(env)
    task_values = {
        option: option_lookahead_value(
            env,
            option,
            belief,
            ROLLOUT_GAMMA,
            ROLLOUT_DEPTH,
            likelihood_error,
            use_observation=False,
            cache=cache,
        )
        for option in options
    }
    task_best_option = max(options, key=lambda option: (task_values[option], -int(option)))
    task_best_value = task_values[task_best_option]
    before_entropy = belief.entropy_vector()

    scored = []
    for option in options:
        after_entropy = expected_entropy_after_option(env, option, belief, likelihood_error)
        info_gain = np.maximum(before_entropy - after_entropy, 0.0)
        weighted_mi = float(np.dot(factor_observation_weights(env, option), info_gain))
        bellman_value = option_lookahead_value(
            env,
            option,
            belief,
            ROLLOUT_GAMMA,
            ROLLOUT_DEPTH,
            likelihood_error,
            use_observation=True,
            cache=cache,
        )
        delta_info = bellman_value - task_values[option]
        diagnostic_cost = max(0.0, task_best_value - task_values[option])

        if method == "passive":
            score = task_values[option]
            active_gain = 0.0
        elif method == "gtvoi":
            score = bellman_value + alpha * max(delta_info, 0.0) - beta * diagnostic_cost
            active_gain = delta_info
        elif method == "mi":
            score = task_values[option] + alpha * weighted_mi - beta * diagnostic_cost
            active_gain = weighted_mi
        elif method == "oracle_greedy":
            true_belief = BeliefState.point_mass(truth)
            score = option_lookahead_value(
                env,
                option,
                true_belief,
                ROLLOUT_GAMMA,
                ROLLOUT_DEPTH,
                likelihood_error,
                use_observation=False,
                cache=cache,
            )
            active_gain = 0.0
        elif method == "oracle":
            true_belief = BeliefState.point_mass(truth)
            score = option_lookahead_value(
                env,
                option,
                true_belief,
                ROLLOUT_GAMMA,
                ROLLOUT_DEPTH,
                likelihood_error,
                use_observation=True,
                cache=cache,
            )
            active_gain = 0.0
        else:
            raise ValueError(f"Unsupported deterministic method: {method}")

        scored.append(
            (
                float(score),
                {
                    "task_value": float(task_values[option]),
                    "bellman_value": float(bellman_value),
                    "active_gain": float(active_gain),
                    "cost": float(diagnostic_cost),
                    "entropy_gain_total": float(info_gain.sum()),
                    "entropy_gain_value_weighted": weighted_mi,
                    "delta_info": float(delta_info),
                    "mi_gain": weighted_mi,
                    "task_best_option": OPTION_NAMES[task_best_option],
                    "task_best_value": float(task_best_value),
                    "diagnostic_cost": float(diagnostic_cost),
                },
                option,
            )
        )
    return scored


def choose_option(
    env: ToyFactorGameEnv,
    belief: BeliefState,
    method: str,
    rng: np.random.RandomState,
    alpha: float,
    beta: float,
    likelihood_error: float,
    truth: ConventionKey,
    cache: dict[tuple, float] | None = None,
) -> tuple[OptionID, dict[str, float | str]]:
    options = valid_options(env)

    if method == "random":
        option = options[int(rng.randint(0, len(options)))]
        scored = score_options(env, belief, "passive", alpha, beta, likelihood_error, truth, cache)
        diag_by_option = {scored_option: diagnostics for _score, diagnostics, scored_option in scored}
        diagnostics = dict(diag_by_option[option])
        diagnostics["active_gain"] = 0.0
        diagnostics["score"] = 0.0
        return option, diagnostics

    scored = score_options(env, belief, method, alpha, beta, likelihood_error, truth, cache)
    score, diagnostics, option = max(scored, key=lambda item: (item[0], -int(item[2])))
    diagnostics = dict(diagnostics)
    diagnostics["score"] = float(score)
    return option, diagnostics


def env_for_scenario(
    scenario: DiagnosticScenario | None,
    convention: ConventionKey,
    max_steps: int,
    seed: int = 0,
) -> ToyFactorGameEnv:
    env = ToyFactorGameEnv(partner_convention=key_to_assignment(convention), max_steps=max_steps, seed=seed)
    apply_scenario(env, scenario)
    return env


def option_values(
    env: ToyFactorGameEnv,
    belief: BeliefState,
    depth: int,
    likelihood_error: float,
    use_observation: bool,
    cache: dict[tuple, float] | None = None,
) -> dict[OptionID, float]:
    return {
        option: option_lookahead_value(
            env,
            option,
            belief,
            ROLLOUT_GAMMA,
            depth,
            likelihood_error,
            use_observation,
            cache,
        )
        for option in valid_options(env)
    }


def best_option_and_values(
    env: ToyFactorGameEnv,
    belief: BeliefState,
    depth: int,
    likelihood_error: float,
    use_observation: bool,
    cache: dict[tuple, float] | None = None,
) -> tuple[OptionID, dict[OptionID, float]]:
    values = option_values(env, belief, depth, likelihood_error, use_observation, cache)
    option = max(values, key=lambda candidate: (values[candidate], -int(candidate)))
    return option, values


def observation_distribution(
    scenario: DiagnosticScenario,
    option: OptionID,
    convention: ConventionKey,
    max_steps: int,
) -> dict[int, float]:
    env = env_for_scenario(scenario, convention, max_steps)
    action = get_option_action(option, env.ego_pos, env.ego_carrying)
    _obs, _reward, _done, info = env.step(action)
    return {int(info["partner_action"]): 1.0}


def total_variation(left: dict[int, float], right: dict[int, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def observation_separation(
    scenario: DiagnosticScenario,
    option: OptionID,
    conventions: tuple[ConventionKey, ConventionKey],
    max_steps: int,
) -> float:
    left = observation_distribution(scenario, option, conventions[0], max_steps)
    right = observation_distribution(scenario, option, conventions[1], max_steps)
    return float(total_variation(left, right))


def focused_convention_pairs(factor_idx: int) -> list[tuple[ConventionKey, ConventionKey]]:
    pairs = []
    keys = all_convention_keys()
    for left in keys:
        for right in keys:
            if left >= right:
                continue
            if left[factor_idx] == right[factor_idx]:
                continue
            if all(left[idx] == right[idx] for idx in range(NUM_FACTORS) if idx != factor_idx):
                pairs.append((left, right))
    return pairs


def candidate_scenarios() -> list[DiagnosticScenario]:
    scenarios: list[DiagnosticScenario] = []

    bottleneck_pairs = tuple(sorted({key for pair in focused_convention_pairs(0) for key in pair}))
    delivery_pairs = tuple(sorted({key for pair in focused_convention_pairs(2) for key in pair}))
    resource_pairs = tuple(sorted({key for pair in focused_convention_pairs(1) for key in pair}))

    for offset, partner_col in enumerate((BOTTLENECK_COL + 1, BOTTLENECK_COL + 2)):
        scenarios.append(
            DiagnosticScenario(
                name=f"candidate_bottleneck_carrying_{offset}",
                description="Partner must cross the bottleneck while carrying; yielding convention changes collision risk.",
                conventions=bottleneck_pairs,
                ego_pos=(BOTTLENECK_ROW, BOTTLENECK_COL - 1),
                partner_pos=(BOTTLENECK_ROW, partner_col),
                partner_carrying=True,
                resource_a_available=False,
                resource_b_available=False,
            )
        )
    for offset, partner_col in enumerate((BOTTLENECK_COL - 1, BOTTLENECK_COL - 2)):
        scenarios.append(
            DiagnosticScenario(
                name=f"candidate_bottleneck_mirror_carrying_{offset}",
                description="Mirrored bottleneck case where partner must cross right; yielding convention changes collision risk.",
                conventions=bottleneck_pairs,
                ego_pos=(BOTTLENECK_ROW, BOTTLENECK_COL + 1),
                partner_pos=(BOTTLENECK_ROW, partner_col),
                partner_carrying=True,
                resource_a_available=False,
                resource_b_available=False,
            )
        )

    scenarios.extend(
        [
            DiagnosticScenario(
                name="candidate_delivery_left_loaded",
                description="Ego is at left delivery with soup; role convention determines whether to drop or reroute.",
                conventions=delivery_pairs,
                ego_pos=DELIVER_LEFT,
                partner_pos=DELIVER_RIGHT,
                ego_carrying=True,
                partner_carrying=True,
                resource_a_available=False,
                resource_b_available=False,
            ),
            DiagnosticScenario(
                name="candidate_delivery_right_loaded",
                description="Ego is at right delivery with soup; role convention determines whether to drop or reroute.",
                conventions=delivery_pairs,
                ego_pos=DELIVER_RIGHT,
                partner_pos=DELIVER_LEFT,
                ego_carrying=True,
                partner_carrying=True,
                resource_a_available=False,
                resource_b_available=False,
            ),
            DiagnosticScenario(
                name="candidate_resource_a_contested",
                description="Both agents are near resource A; ownership convention changes which resource ego should pursue.",
                conventions=resource_pairs,
                ego_pos=RESOURCE_A,
                partner_pos=(RESOURCE_A[0] + 1, RESOURCE_A[1]),
                resource_a_available=True,
                resource_b_available=True,
            ),
            DiagnosticScenario(
                name="candidate_resource_b_contested",
                description="Ego is near resource B while partner approaches A; ownership convention changes target choice.",
                conventions=resource_pairs,
                ego_pos=RESOURCE_B,
                partner_pos=(RESOURCE_A[0] + 1, RESOURCE_A[1]),
                resource_a_available=True,
                resource_b_available=True,
            ),
            DiagnosticScenario(
                name="candidate_high_mi_low_value_distractor",
                description="Resource signal is observable, but bottleneck yielding is the high-value control factor.",
                conventions=((0, 0, 1), (1, 1, 1)),
                ego_pos=(BOTTLENECK_ROW, BOTTLENECK_COL - 1),
                partner_pos=(BOTTLENECK_ROW, BOTTLENECK_COL + 1),
                partner_carrying=True,
                resource_a_available=True,
                resource_b_available=True,
            ),
        ]
    )
    return scenarios


def scenario_convention_pairs(scenario: DiagnosticScenario) -> list[tuple[ConventionKey, ConventionKey]]:
    if "bottleneck" in scenario.name:
        delivery_mode = 0 if "mirror" in scenario.name else 1
        return [
            pair
            for pair in focused_convention_pairs(0)
            if pair[0][2] == delivery_mode and pair[1][2] == delivery_mode
        ]
    if "resource" in scenario.name:
        return focused_convention_pairs(1)
    if "delivery" in scenario.name:
        return focused_convention_pairs(2)
    if "distractor" in scenario.name:
        if len(scenario.conventions) != 2:
            raise ValueError(f"Distractor scenario {scenario.name} must define exactly two conventions")
        return [(scenario.conventions[0], scenario.conventions[1])]
    raise ValueError(f"No convention-pair generator for scenario {scenario.name}")


def evaluate_candidate_case(
    scenario: DiagnosticScenario,
    conventions: tuple[ConventionKey, ConventionKey],
    case_idx: int,
    max_steps: int,
    depth: int,
    likelihood_error: float,
    delta_gap: float,
    delta_action: float,
    delta_obs: float,
    delta_return: float,
    delta_mi: float,
    delta_value: float,
    max_diagnostic_cost: float,
) -> DiagnosticCase:
    belief = BeliefState.uniform(conventions)
    env = env_for_scenario(scenario, conventions[0], max_steps)
    cache: dict[tuple, float] = {}

    passive_option, passive_values = best_option_and_values(
        env, belief, depth, likelihood_error, use_observation=False, cache=cache
    )
    passive_value = passive_values[passive_option]

    oracle_values = []
    oracle_first_options = []
    oracle_value_tables: list[dict[OptionID, float]] = []
    for convention in conventions:
        oracle_env = env_for_scenario(scenario, convention, max_steps)
        oracle_belief = BeliefState.point_mass(convention)
        oracle_option, values = best_option_and_values(
            oracle_env,
            oracle_belief,
            depth,
            likelihood_error,
            use_observation=True,
            cache=cache,
        )
        oracle_values.append(values[oracle_option])
        oracle_first_options.append(oracle_option)
        oracle_value_tables.append(values)

    oracle_value = float(np.mean(oracle_values))
    oracle_passive_gap = oracle_value - passive_value
    if oracle_passive_gap < delta_gap:
        return DiagnosticCase(
            case_id=f"{scenario.name}_pair{case_idx}",
            case_type="candidate",
            scenario=scenario,
            conventions=conventions,
            oracle_passive_gap=float(oracle_passive_gap),
            oracle_value=float(oracle_value),
            passive_value=float(passive_value),
            oracle_first_actions=tuple(OPTION_NAMES[option] for option in oracle_first_options),
            passive_first_action=OPTION_NAMES[passive_option],
            failure_reason="oracle_passive_gap",
        )

    best_response_flip = len(set(oracle_first_options)) > 1
    if not best_response_flip:
        return DiagnosticCase(
            case_id=f"{scenario.name}_pair{case_idx}",
            case_type="candidate",
            scenario=scenario,
            conventions=conventions,
            oracle_passive_gap=float(oracle_passive_gap),
            oracle_value=float(oracle_value),
            passive_value=float(passive_value),
            oracle_first_actions=tuple(OPTION_NAMES[option] for option in oracle_first_options),
            passive_first_action=OPTION_NAMES[passive_option],
            best_response_flip=False,
            failure_reason="best_response_flip",
        )

    left_best, right_best = oracle_first_options
    left_values, right_values = oracle_value_tables
    wrong_action_regrets = [
        left_values[left_best] - left_values.get(right_best, -1e9),
        right_values[right_best] - right_values.get(left_best, -1e9),
    ]
    action_gap = min(wrong_action_regrets)
    if action_gap < delta_action:
        return DiagnosticCase(
            case_id=f"{scenario.name}_pair{case_idx}",
            case_type="candidate",
            scenario=scenario,
            conventions=conventions,
            oracle_passive_gap=float(oracle_passive_gap),
            oracle_value=float(oracle_value),
            passive_value=float(passive_value),
            oracle_first_actions=tuple(OPTION_NAMES[option] for option in oracle_first_options),
            passive_first_action=OPTION_NAMES[passive_option],
            best_response_flip=True,
            action_gap=float(action_gap),
            failure_reason="action_gap",
        )

    task_values = passive_values
    best_diag: tuple[OptionID, float, float, float, float, float] | None = None
    option_stats = []
    high_delta_low_return: tuple[OptionID, float, float, float, float, float] | None = None
    before_entropy = belief.entropy_vector()
    for option in valid_options(env):
        separation = observation_separation(scenario, option, conventions, max_steps)
        if separation < delta_obs:
            continue
        after_entropy = expected_entropy_after_option(env, option, belief, likelihood_error)
        info_gain = np.maximum(before_entropy - after_entropy, 0.0)
        mi_gain = float(np.dot(factor_observation_weights(env, option), info_gain))
        value_with_observation = option_lookahead_value(
            env,
            option,
            belief,
            ROLLOUT_GAMMA,
            depth,
            likelihood_error,
            use_observation=True,
            cache=cache,
        )
        value_without_observation = task_values[option]
        delta_info = value_with_observation - value_without_observation
        return_gain = value_with_observation - passive_value
        diagnostic_cost = max(0.0, passive_value - task_values[option])
        option_stats.append((option, mi_gain, delta_info, return_gain, diagnostic_cost))
        if delta_info >= DELTA_INFO_THRESHOLD and return_gain < delta_return:
            artifact_candidate = (option, separation, return_gain, delta_info, mi_gain, diagnostic_cost)
            if high_delta_low_return is None or artifact_candidate[3] > high_delta_low_return[3]:
                high_delta_low_return = artifact_candidate
        if diagnostic_cost > max_diagnostic_cost:
            continue
        if return_gain < delta_return:
            continue
        candidate = (option, separation, return_gain, delta_info, mi_gain, diagnostic_cost)
        if best_diag is None or candidate[2] > best_diag[2]:
            best_diag = candidate

    if best_diag is None:
        artifact_reason = high_delta_low_return is not None
        fallback = high_delta_low_return
        return DiagnosticCase(
            case_id=f"{scenario.name}_pair{case_idx}",
            case_type="candidate",
            scenario=scenario,
            conventions=conventions,
            oracle_passive_gap=float(oracle_passive_gap),
            oracle_value=float(oracle_value),
            passive_value=float(passive_value),
            oracle_first_actions=tuple(OPTION_NAMES[option] for option in oracle_first_options),
            passive_first_action=OPTION_NAMES[passive_option],
            best_response_flip=True,
            action_gap=float(action_gap),
            best_diag_option=OPTION_NAMES[fallback[0]] if fallback is not None else None,
            observation_separation=float(fallback[1]) if fallback is not None else None,
            best_diag_return_gain=float(fallback[2]) if fallback is not None else None,
            best_diag_delta_info=float(fallback[3]) if fallback is not None else None,
            best_diag_mi_gain=float(fallback[4]) if fallback is not None else None,
            diagnostic_opportunity_cost=float(fallback[5]) if fallback is not None else None,
            max_delta_info=float(fallback[3]) if fallback is not None else None,
            failure_reason=(
                "HIGH_DELTA_INFO_LOW_RETURN_ARTIFACT"
                if artifact_reason
                else "diagnostic_return_gain"
            ),
        )

    mi_option = max(option_stats, key=lambda item: (item[1], item[2], item[3])) if option_stats else None
    gtvoi_option = max(option_stats, key=lambda item: (item[2], item[3], item[1])) if option_stats else None
    high_mi_low_value = False
    mi_gap = delta_info_gap = distractor_return_gap = None
    if mi_option is not None and gtvoi_option is not None and mi_option[0] != gtvoi_option[0]:
        mi_gap = mi_option[1] - gtvoi_option[1]
        delta_info_gap = gtvoi_option[2] - mi_option[2]
        distractor_return_gap = gtvoi_option[3] - mi_option[3]
        high_mi_low_value = (
            mi_gap >= delta_mi
            and delta_info_gap >= delta_value
            and distractor_return_gap >= delta_return
        )

    case_type = "high_mi_low_value_distractor" if high_mi_low_value else "diagnostic_critical"
    return DiagnosticCase(
        case_id=f"{scenario.name}_pair{case_idx}",
        case_type=case_type,
        scenario=scenario,
        conventions=conventions,
        oracle_passive_gap=float(oracle_passive_gap),
        oracle_value=float(oracle_value),
        passive_value=float(passive_value),
        oracle_first_actions=tuple(OPTION_NAMES[option] for option in oracle_first_options),
        passive_first_action=OPTION_NAMES[passive_option],
        best_response_flip=True,
        action_gap=float(action_gap),
        best_diag_option=OPTION_NAMES[best_diag[0]],
        observation_separation=float(best_diag[1]),
        diagnostic_opportunity_cost=float(best_diag[5]),
        best_diag_return_gain=float(best_diag[2]),
        best_diag_delta_info=float(best_diag[3]),
        best_diag_mi_gain=float(best_diag[4]),
        max_delta_info=float(max((item[2] for item in option_stats), default=0.0)),
        high_mi_low_value_distractor=high_mi_low_value,
        mi_option=OPTION_NAMES[mi_option[0]] if mi_option is not None else None,
        gtvoi_option=OPTION_NAMES[gtvoi_option[0]] if gtvoi_option is not None else None,
        mi_gap=float(mi_gap) if mi_gap is not None else None,
        delta_info_gap=float(delta_info_gap) if delta_info_gap is not None else None,
        distractor_return_gap=float(distractor_return_gap) if distractor_return_gap is not None else None,
        passed_filters=True,
    )


def synthesize_diagnostic_cases(
    max_steps: int,
    depth: int,
    likelihood_error: float,
    delta_gap: float,
    delta_action: float,
    delta_obs: float,
    delta_return: float,
    delta_mi: float,
    delta_value: float,
    max_diagnostic_cost: float = DEFAULT_MAX_DIAGNOSTIC_COST,
) -> tuple[list[DiagnosticCase], list[DiagnosticCase]]:
    accepted: list[DiagnosticCase] = []
    rejected: list[DiagnosticCase] = []
    for scenario in candidate_scenarios():
        for pair_idx, pair in enumerate(scenario_convention_pairs(scenario)):
            case = evaluate_candidate_case(
                scenario=scenario,
                conventions=pair,
                case_idx=pair_idx,
                max_steps=max_steps,
                depth=depth,
                likelihood_error=likelihood_error,
                delta_gap=delta_gap,
                delta_action=delta_action,
                delta_obs=delta_obs,
                delta_return=delta_return,
                delta_mi=delta_mi,
                delta_value=delta_value,
                max_diagnostic_cost=max_diagnostic_cost,
            )
            if case.passed_filters:
                accepted.append(case)
            else:
                rejected.append(case)
    accepted.sort(
        key=lambda case: (
            bool(case.high_mi_low_value_distractor),
            case.oracle_passive_gap or 0.0,
            case.best_diag_return_gain or 0.0,
        ),
        reverse=True,
    )
    return accepted, rejected


def is_aligned(belief: BeliefState, truth: ConventionKey, threshold: float) -> bool:
    predictions = belief.mode_predictions()
    for factor_idx in range(NUM_FACTORS):
        if belief.confidence_for_truth(truth)[factor_idx] >= threshold and predictions[factor_idx] != truth[factor_idx]:
            return False
    return True


def safe_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_episode(
    method: str,
    convention: ConventionKey,
    seed: int,
    max_steps: int,
    alpha: float,
    beta: float,
    likelihood_error: float,
    alignment_weight_threshold: float,
    case: DiagnosticCase,
    shared_value_cache: dict[tuple, float] | None = None,
) -> dict[str, float | int | str | None]:
    scenario = case.scenario
    rng = np.random.RandomState(seed)
    env = ToyFactorGameEnv(partner_convention=key_to_assignment(convention), max_steps=max_steps, seed=seed)
    apply_scenario(env, scenario)
    belief = BeliefState.uniform(case.conventions if scenario is not None else None)

    total_reward = 0.0
    early_reward = 0.0
    diagnostic_cost_total = 0.0
    diagnostic_count = 0
    positive_delta_count = 0
    collisions = 0
    alignment_time = max_steps + 1
    total_option_cost = 0.0
    selected_options: list[str] = []
    active_gains: list[float] = []
    delta_infos: list[float] = []
    mi_gains: list[float] = []
    diagnostic_costs: list[float] = []
    rewards: list[float] = []
    reward_after_first_diagnostic: float | None = None
    first_diagnostic_step: int | None = None
    first_high_delta_step: int | None = None
    reward_after_first_high_delta_info: float | None = None
    value_cache: dict[tuple, float] = shared_value_cache if shared_value_cache is not None else {}

    for step_idx in range(max_steps):
        env_before = copy.deepcopy(env)
        option, diagnostics = choose_option(
            env,
            belief,
            method,
            rng,
            alpha,
            beta,
            likelihood_error,
            convention,
            value_cache,
        )
        action = get_option_action(option, env.ego_pos, env.ego_carrying)
        _obs, reward, done, info = env.step(action)
        posterior = belief.update_from_partner_action(env_before, int(info["partner_action"]), likelihood_error)

        delta_info = float(diagnostics["delta_info"])
        mi_gain = float(diagnostics["mi_gain"])
        task_best_option = str(diagnostics["task_best_option"])
        is_diagnostic = (
            method not in ("random", "oracle", "oracle_greedy")
            and delta_info > DELTA_INFO_THRESHOLD
            and OPTION_NAMES[option] != task_best_option
        )
        diagnostic_cost = float(diagnostics["diagnostic_cost"]) if is_diagnostic else 0.0

        selected_options.append(OPTION_NAMES[option])
        total_reward += reward
        rewards.append(float(reward))
        if step_idx < max(1, max_steps // 5):
            early_reward += reward
        collisions += int(info["collision"])
        total_option_cost += float(diagnostics["diagnostic_cost"])
        active_gains.append(float(diagnostics["active_gain"]))
        delta_infos.append(delta_info)
        mi_gains.append(mi_gain)
        diagnostic_costs.append(diagnostic_cost)

        if delta_info > DELTA_INFO_THRESHOLD:
            positive_delta_count += 1
            if first_high_delta_step is None:
                first_high_delta_step = step_idx
        if is_diagnostic:
            diagnostic_count += 1
            diagnostic_cost_total += diagnostic_cost
            if first_diagnostic_step is None:
                first_diagnostic_step = step_idx

        belief = posterior
        if alignment_time == max_steps + 1 and is_aligned(
            belief, convention, alignment_weight_threshold
        ):
            alignment_time = step_idx + 1

        if done:
            break

    if first_diagnostic_step is not None:
        reward_after_first_diagnostic = float(sum(rewards[first_diagnostic_step:]))
    if first_high_delta_step is not None:
        reward_after_first_high_delta_info = float(sum(rewards[first_high_delta_step:]))

    future_returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + ROLLOUT_GAMMA * running
        future_returns.append(running)
    future_returns = list(reversed(future_returns))

    final_entropies = belief.entropy_vector()
    final_predictions = belief.mode_predictions()
    truth_confidences = belief.confidence_for_truth(convention)
    final_factor_correct = [int(final_predictions[factor] == convention[factor]) for factor in range(NUM_FACTORS)]
    sorted_delta = sorted(delta_infos, reverse=True)
    top_k = sorted_delta[: min(3, len(sorted_delta))]

    row: dict[str, float | int | str | None] = {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "scenario": scenario.name if scenario is not None else "default_start",
        "method": method,
        "convention": convention_label(convention),
        "seed": seed,
        "oracle_passive_gap": case.oracle_passive_gap,
        "oracle_first_actions": ",".join(case.oracle_first_actions),
        "passive_first_action": case.passive_first_action,
        "best_response_flip": int(case.best_response_flip),
        "action_gap": case.action_gap,
        "best_diag_option": case.best_diag_option,
        "observation_separation": case.observation_separation,
        "best_diag_return_gain": case.best_diag_return_gain,
        "high_mi_low_value_distractor": int(case.high_mi_low_value_distractor),
        "first_action": selected_options[0] if selected_options else None,
        "episode_reward": float(total_reward),
        "early_reward": float(early_reward),
        "probe_cost": float(diagnostic_cost_total),
        "probe_count": int(diagnostic_count),
        "diagnostic_cost": float(diagnostic_cost_total),
        "diagnostic_count": int(diagnostic_count),
        "positive_delta_info_count": int(positive_delta_count),
        "high_delta_info_count": int(positive_delta_count),
        "positive_delta_info_rate": float(positive_delta_count / max(1, len(delta_infos))),
        "max_delta_info": float(max(delta_infos) if delta_infos else 0.0),
        "max_mi": float(max(mi_gains) if mi_gains else 0.0),
        "top_delta_info_mean": float(np.mean(top_k)) if top_k else 0.0,
        "mean_delta_info": float(np.mean(delta_infos)) if delta_infos else 0.0,
        "mean_mi_gain": float(np.mean(mi_gains)) if mi_gains else 0.0,
        "future_return_delta_info_corr": safe_corr(delta_infos, future_returns),
        "future_return_mi_corr": safe_corr(mi_gains, future_returns),
        "reward_after_first_diagnostic": reward_after_first_diagnostic,
        "first_high_delta_step": first_high_delta_step,
        "reward_after_first_high_delta_info": reward_after_first_high_delta_info,
        "future_reward_gain_after_high_delta_info": None,
        "oracle_gap_before_high_delta_info": None,
        "oracle_gap_after_high_delta_info": None,
        "total_option_cost": float(total_option_cost),
        "collisions": int(collisions),
        "time_to_alignment": int(alignment_time),
        "final_entropy_total": float(final_entropies.sum()),
        "final_entropy_critical": float(final_entropies[:NUM_FACTORS].sum()),
        "final_factor_accuracy": float(np.mean(final_factor_correct)),
        "final_factor_confidence": float(np.mean(truth_confidences)),
        "mean_active_gain": float(np.mean(active_gains)) if active_gains else 0.0,
        "selected_options": ",".join(selected_options),
        "_rewards": rewards,
    }
    for factor_idx in range(NUM_FACTORS):
        row[f"factor_{factor_idx}_correct"] = int(final_factor_correct[factor_idx])
        row[f"factor_{factor_idx}_confidence"] = float(truth_confidences[factor_idx])
    return row


def _mean_numeric(rows: list[dict[str, float | int | str | None]], key: str, default: float = 0.0) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else default


def summarize_method(rows: list[dict[str, float | int | str | None]]) -> dict[str, float | list[float] | None]:
    rewards = np.array([float(row["episode_reward"]) for row in rows], dtype=np.float64)
    probe_costs = np.array([float(row["probe_cost"]) for row in rows], dtype=np.float64)
    regrets = np.array([float(row["regret_to_oracle"]) for row in rows], dtype=np.float64)
    early_regrets = np.array([float(row["early_regret_to_oracle"]) for row in rows], dtype=np.float64)
    alignments = np.array([float(row["time_to_alignment"]) for row in rows], dtype=np.float64)
    factor_confidences = np.array(
        [
            float(row[f"factor_{factor_idx}_confidence"])
            for row in rows
            for factor_idx in range(NUM_FACTORS)
        ],
        dtype=np.float64,
    )
    factor_correct = np.array(
        [
            float(row[f"factor_{factor_idx}_correct"])
            for row in rows
            for factor_idx in range(NUM_FACTORS)
        ],
        dtype=np.float64,
    )
    episode_correct = np.array([float(row["final_factor_accuracy"]) for row in rows], dtype=np.float64)
    delta_corr = [float(row["future_return_delta_info_corr"]) for row in rows if row.get("future_return_delta_info_corr") is not None]
    mi_corr = [float(row["future_return_mi_corr"]) for row in rows if row.get("future_return_mi_corr") is not None]
    high_delta_counts = np.array([float(row.get("high_delta_info_count", 0.0)) for row in rows], dtype=np.float64)

    reward_ci = bootstrap_ci(rewards)
    regret_ci = bootstrap_ci(regrets)
    return {
        "n": len(rows),
        "episode_reward_mean": float(rewards.mean()),
        "episode_reward_ci": list(reward_ci),
        "probe_cost_mean": float(probe_costs.mean()),
        "diagnostic_cost_mean": _mean_numeric(rows, "diagnostic_cost"),
        "diagnostic_count_mean": _mean_numeric(rows, "diagnostic_count"),
        "high_delta_info_count_mean": _mean_numeric(rows, "high_delta_info_count"),
        "positive_delta_info_rate_mean": _mean_numeric(rows, "positive_delta_info_rate"),
        "max_delta_info_mean": _mean_numeric(rows, "max_delta_info"),
        "max_mi_mean": _mean_numeric(rows, "max_mi"),
        "top_delta_info_mean": _mean_numeric(rows, "top_delta_info_mean"),
        "mean_delta_info": _mean_numeric(rows, "mean_delta_info"),
        "mean_mi_gain": _mean_numeric(rows, "mean_mi_gain"),
        "future_return_delta_info_corr_mean": float(np.mean(delta_corr)) if delta_corr else None,
        "future_return_mi_corr_mean": float(np.mean(mi_corr)) if mi_corr else None,
        "future_reward_gain_after_high_delta_info_mean": _mean_numeric(
            rows, "future_reward_gain_after_high_delta_info", default=float("nan")
        ),
        "oracle_gap_before_high_delta_info_mean": _mean_numeric(
            rows, "oracle_gap_before_high_delta_info", default=float("nan")
        ),
        "oracle_gap_after_high_delta_info_mean": _mean_numeric(
            rows, "oracle_gap_after_high_delta_info", default=float("nan")
        ),
        "high_delta_count_reward_corr": safe_corr(list(high_delta_counts), list(rewards)),
        "reward_after_first_diagnostic_mean": _mean_numeric(rows, "reward_after_first_diagnostic", default=float("nan")),
        "regret_to_oracle_mean": float(regrets.mean()),
        "regret_to_oracle_ci": list(regret_ci),
        "early_regret_to_oracle_mean": float(early_regrets.mean()),
        "time_to_alignment_mean": float(alignments.mean()),
        "final_factor_accuracy_mean": float(episode_correct.mean()),
        "final_ece": expected_calibration_error(factor_confidences, factor_correct),
    }


def add_oracle_regrets(rows: list[dict[str, float | int | str | None]]) -> None:
    oracle_by_case = {
        (row["case_id"], row["seed"], row["convention"]): row
        for row in rows
        if row["method"] == "oracle"
    }
    passive_by_case = {
        (row["case_id"], row["seed"], row["convention"]): row
        for row in rows
        if row["method"] == "passive"
    }
    for row in rows:
        key = (row["case_id"], row["seed"], row["convention"])
        oracle = oracle_by_case[key]
        row["regret_to_oracle"] = float(oracle["episode_reward"]) - float(row["episode_reward"])
        row["early_regret_to_oracle"] = float(oracle["early_reward"]) - float(row["early_reward"])
        first_high_delta_step = row.get("first_high_delta_step")
        if first_high_delta_step is None:
            continue
        step = int(first_high_delta_step)
        row_rewards = [float(value) for value in row.get("_rewards", [])]
        oracle_rewards = [float(value) for value in oracle.get("_rewards", [])]
        passive_rewards = [
            float(value)
            for value in passive_by_case.get(key, {}).get("_rewards", [])
        ]
        if step >= len(row_rewards):
            continue
        row_after = float(sum(row_rewards[step:]))
        oracle_after = float(sum(oracle_rewards[step:])) if step < len(oracle_rewards) else float("nan")
        passive_after = float(sum(passive_rewards[step:])) if step < len(passive_rewards) else float("nan")
        row["reward_after_first_high_delta_info"] = row_after
        if not math.isnan(passive_after):
            row["future_reward_gain_after_high_delta_info"] = row_after - passive_after
        row["oracle_gap_before_high_delta_info"] = row["regret_to_oracle"]
        if not math.isnan(oracle_after):
            row["oracle_gap_after_high_delta_info"] = oracle_after - row_after


def _case_metric(case_rows: list[dict[str, object]], key: str, default: float = 0.0) -> float:
    vals = [float(row[key]) for row in case_rows if row.get(key) is not None]
    return float(np.median(vals)) if vals else default


def _mean_reward_for(
    rows: list[dict[str, float | int | str | None]],
    method: str,
    case_type: str | None = None,
) -> float | None:
    vals = [
        float(row["episode_reward"])
        for row in rows
        if row["method"] == method and (case_type is None or row.get("case_type") == case_type)
    ]
    return float(np.mean(vals)) if vals else None


def tiered_validation(
    summary: dict[str, dict],
    rows: list[dict[str, float | int | str | None]],
    case_rows: list[dict[str, object]],
    methods: list[str],
    min_cases: int,
    delta_gap: float,
    delta_action: float,
    delta_return: float,
    oracle_margin: float,
) -> dict:
    required = [method for method in DEFAULT_METHODS if method in methods]
    missing = [method for method in DEFAULT_METHODS if method not in summary]
    accepted_cases = [row for row in case_rows if row.get("passed_filters")]
    distractor_cases = [
        row for row in accepted_cases if row.get("high_mi_low_value_distractor")
    ]
    median_gap = _case_metric(accepted_cases, "oracle_passive_gap")
    median_action_gap = _case_metric(accepted_cases, "action_gap")
    median_return_gain = _case_metric(accepted_cases, "best_diag_return_gain")
    action_flip_rate = float(np.mean([float(row.get("best_response_flip", 0.0)) for row in accepted_cases])) if accepted_cases else 0.0
    tier0_checks = {
        "n_cases_ge_min": len(accepted_cases) >= min_cases,
        "median_oracle_passive_gap_ge_threshold": median_gap >= delta_gap,
        "action_flip_rate_positive": action_flip_rate > 0.0,
        "median_action_gap_ge_threshold": median_action_gap >= delta_action,
        "median_best_diag_return_gain_ge_threshold": median_return_gain >= delta_return,
    }
    tier0_status = (
        "PASS"
        if all(tier0_checks.values())
        else "NO_DIAGNOSTIC_CRITICAL_CASES_FOUND"
        if not accepted_cases
        else "FAIL_CASE_CONSTRUCTION"
    )
    tier0 = {
        "n_cases": len(accepted_cases),
        "min_cases": min_cases,
        "n_high_mi_low_value_distractor_cases": len(distractor_cases),
        "median_oracle_passive_gap": float(median_gap),
        "oracle_passive_gap_threshold": float(delta_gap),
        "action_flip_rate": float(action_flip_rate),
        "median_action_gap": float(median_action_gap),
        "action_gap_threshold": float(delta_action),
        "median_best_diag_return_gain": float(median_return_gain),
        "diagnosis_return_gain_threshold": float(delta_return),
        "checks": tier0_checks,
        "status": tier0_status,
    }

    tier1_checks = {
        "oracle_ge_passive_plus_margin": (
            "oracle" in summary
            and "passive" in summary
            and summary["oracle"]["episode_reward_mean"]
            >= summary["passive"]["episode_reward_mean"] + oracle_margin
        ),
        "oracle_ge_all_directed": (
            "oracle" in summary
            and all(
                method not in summary
                or method == "oracle"
                or summary["oracle"]["episode_reward_mean"] >= summary[method]["episode_reward_mean"]
                for method in ("gtvoi", "mi", "passive")
            )
        ),
    }
    tier1 = {
        "checks": tier1_checks,
        "oracle_margin": float(oracle_margin),
        "status": "PASS" if tier0["status"] == "PASS" and all(tier1_checks.values()) else "NOT_EVALUATED" if tier0["status"] != "PASS" else "FAIL",
    }

    gtvoi_reward = summary.get("gtvoi", {}).get("episode_reward_mean")
    passive_reward = summary.get("passive", {}).get("episode_reward_mean")
    gtvoi_distractor = _mean_reward_for(rows, "gtvoi", "high_mi_low_value_distractor")
    mi_distractor = _mean_reward_for(rows, "mi", "high_mi_low_value_distractor")
    tier2_checks = {
        "gtvoi_ge_passive": (
            gtvoi_reward is not None
            and passive_reward is not None
            and gtvoi_reward >= passive_reward
        ),
        "distractor_subset_available": len(distractor_cases) > 0,
        "gtvoi_ge_mi_on_distractor_subset": (
            gtvoi_distractor is not None
            and mi_distractor is not None
            and gtvoi_distractor >= mi_distractor
        ),
        "gtvoi_cost_le_mi": (
            "gtvoi" in summary
            and "mi" in summary
            and summary["gtvoi"]["diagnostic_cost_mean"] <= summary["mi"]["diagnostic_cost_mean"]
        ),
    }
    tier2 = {
        "checks": tier2_checks,
        "status": (
            "PASS"
            if tier1["status"] == "PASS"
            and tier2_checks["gtvoi_ge_passive"]
            and (
                tier2_checks["gtvoi_ge_mi_on_distractor_subset"]
                or tier2_checks["gtvoi_cost_le_mi"]
            )
            else "NOT_EVALUATED" if tier1["status"] != "PASS" else "FAIL"
        ),
    }

    gtvoi_delta_corr = summary.get("gtvoi", {}).get("future_return_delta_info_corr_mean")
    gtvoi_mi_corr = summary.get("gtvoi", {}).get("future_return_mi_corr_mean")
    gtvoi_gap_before = summary.get("gtvoi", {}).get("oracle_gap_before_high_delta_info_mean")
    gtvoi_gap_after = summary.get("gtvoi", {}).get("oracle_gap_after_high_delta_info_mean")
    gtvoi_high_delta_count_corr = summary.get("gtvoi", {}).get("high_delta_count_reward_corr")
    tier3_checks = {
        "delta_corr_positive": gtvoi_delta_corr is not None and gtvoi_delta_corr > 0.0,
        "delta_corr_ge_mi_corr": (
            gtvoi_delta_corr is not None
            and gtvoi_mi_corr is not None
            and gtvoi_delta_corr >= gtvoi_mi_corr
        ),
        "oracle_gap_drops_after_high_delta_info": (
            gtvoi_gap_before is not None
            and gtvoi_gap_after is not None
            and not math.isnan(float(gtvoi_gap_before))
            and not math.isnan(float(gtvoi_gap_after))
            and float(gtvoi_gap_after) < float(gtvoi_gap_before)
        ),
        "high_delta_count_predicts_reward": (
            gtvoi_high_delta_count_corr is not None
            and gtvoi_high_delta_count_corr > 0.0
        ),
    }
    tier3 = {
        "checks": tier3_checks,
        "status": "SOFT_PASS" if all(tier3_checks.values()) else "WARN",
    }
    if tier0["status"] != "PASS":
        overall = tier0["status"]
    elif missing:
        overall = "FAIL_MISSING_METHODS"
    elif tier1["status"] != "PASS":
        overall = "FAIL_ORACLE_SANITY"
    elif tier2["status"] != "PASS":
        overall = "FAIL_DIAGNOSTIC_POLICY"
    else:
        overall = "PASS"

    return {
        "required_methods": required,
        "missing_methods": missing,
        "tier0_case_validity": tier0,
        "tier1_oracle_sanity": tier1,
        "tier2_diagnostic_policy": tier2,
        "tier3_soft_correlation": tier3,
        "overall_status": overall,
    }


def write_csv(rows: list[dict[str, float | int | str | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "case_type",
        "scenario",
        "method",
        "convention",
        "seed",
        "oracle_passive_gap",
        "oracle_first_actions",
        "passive_first_action",
        "best_response_flip",
        "action_gap",
        "best_diag_option",
        "observation_separation",
        "best_diag_return_gain",
        "high_mi_low_value_distractor",
        "first_action",
        "episode_reward",
        "early_reward",
        "regret_to_oracle",
        "early_regret_to_oracle",
        "probe_cost",
        "probe_count",
        "diagnostic_cost",
        "diagnostic_count",
        "positive_delta_info_count",
        "high_delta_info_count",
        "positive_delta_info_rate",
        "max_delta_info",
        "max_mi",
        "top_delta_info_mean",
        "mean_delta_info",
        "mean_mi_gain",
        "future_return_delta_info_corr",
        "future_return_mi_corr",
        "first_high_delta_step",
        "reward_after_first_high_delta_info",
        "future_reward_gain_after_high_delta_info",
        "oracle_gap_before_high_delta_info",
        "oracle_gap_after_high_delta_info",
        "reward_after_first_diagnostic",
        "total_option_cost",
        "collisions",
        "time_to_alignment",
        "final_entropy_total",
        "final_entropy_critical",
        "final_factor_accuracy",
        "final_factor_confidence",
        "factor_0_correct",
        "factor_0_confidence",
        "factor_1_correct",
        "factor_1_confidence",
        "factor_2_correct",
        "factor_2_confidence",
        "mean_active_gain",
        "selected_options",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def case_to_row(case: DiagnosticCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "scenario": case.scenario.name if case.scenario is not None else "default_start",
        "description": case.scenario.description if case.scenario is not None else "Original reset state",
        "conventions": ",".join(convention_label(convention) for convention in case.conventions),
        "oracle_passive_gap": case.oracle_passive_gap,
        "oracle_value": case.oracle_value,
        "passive_value": case.passive_value,
        "oracle_first_actions": ",".join(case.oracle_first_actions),
        "passive_first_action": case.passive_first_action,
        "best_response_flip": int(case.best_response_flip),
        "action_gap": case.action_gap,
        "best_diag_option": case.best_diag_option,
        "observation_separation": case.observation_separation,
        "diagnostic_opportunity_cost": case.diagnostic_opportunity_cost,
        "best_diag_return_gain": case.best_diag_return_gain,
        "best_diag_delta_info": case.best_diag_delta_info,
        "best_diag_mi_gain": case.best_diag_mi_gain,
        "max_delta_info": case.max_delta_info,
        "high_mi_low_value_distractor": int(case.high_mi_low_value_distractor),
        "mi_option": case.mi_option,
        "gtvoi_option": case.gtvoi_option,
        "mi_gap": case.mi_gap,
        "delta_info_gap": case.delta_info_gap,
        "distractor_return_gap": case.distractor_return_gap,
        "passed_filters": int(case.passed_filters),
        "failure_reason": case.failure_reason,
    }


def write_cases_csv(case_rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "case_type",
        "scenario",
        "description",
        "conventions",
        "oracle_passive_gap",
        "oracle_value",
        "passive_value",
        "oracle_first_actions",
        "passive_first_action",
        "best_response_flip",
        "action_gap",
        "best_diag_option",
        "observation_separation",
        "diagnostic_opportunity_cost",
        "best_diag_return_gain",
        "best_diag_delta_info",
        "best_diag_mi_gain",
        "max_delta_info",
        "high_mi_low_value_distractor",
        "mi_option",
        "gtvoi_option",
        "mi_gap",
        "delta_info_gap",
        "distractor_return_gap",
        "passed_filters",
        "failure_reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_scenario_debug_csv(
    rows: list[dict[str, float | int | str | None]],
    case_rows: list[dict[str, object]],
    path: Path,
) -> list[dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    case_by_id = {str(row["case_id"]): row for row in case_rows if row.get("passed_filters")}
    oracle_rewards: dict[str, float] = {}
    passive_rewards: dict[str, float] = {}
    for case_id in case_by_id:
        oracle_vals = [
            float(row["episode_reward"])
            for row in rows
            if row["case_id"] == case_id and row["method"] == "oracle"
        ]
        passive_vals = [
            float(row["episode_reward"])
            for row in rows
            if row["case_id"] == case_id and row["method"] == "passive"
        ]
        oracle_rewards[case_id] = float(np.mean(oracle_vals)) if oracle_vals else float("nan")
        passive_rewards[case_id] = float(np.mean(passive_vals)) if passive_vals else float("nan")

    fieldnames = [
        "case_id",
        "case_type",
        "scenario",
        "method",
        "n",
        "conventions",
        "seeds",
        "episode_reward_mean",
        "oracle_reward_mean",
        "passive_reward_mean",
        "oracle_passive_gap_case",
        "oracle_passive_gap_realized",
        "regret_to_oracle_mean",
        "first_actions",
        "oracle_first_actions",
        "passive_first_action",
        "first_action_matches_oracle_rate",
        "max_delta_info_mean",
        "max_delta_info_max",
        "max_mi",
        "mean_delta_info_mean",
        "mean_mi_gain_mean",
        "future_return_delta_info_corr_mean",
        "future_return_mi_corr_mean",
        "future_reward_gain_after_high_delta_info",
        "oracle_gap_after_high_delta_info_mean",
        "high_delta_info_count_mean",
        "diagnostic_cost_mean",
        "diagnostic_count_mean",
        "reward_after_first_diagnostic_mean",
        "best_response_flip",
        "action_gap",
        "best_diag_option",
        "observation_separation",
        "best_diag_return_gain",
        "high_mi_low_value_distractor",
    ]
    grouped: dict[tuple[str, str], list[dict[str, float | int | str | None]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case_id"]), str(row["method"])), []).append(row)

    debug_rows: list[dict[str, object]] = []
    for (case_id, method), group in sorted(grouped.items()):
        case = case_by_id.get(case_id, {})
        oracle_actions = set(str(group[0].get("oracle_first_actions", "")).split(","))
        first_actions = [str(row.get("first_action")) for row in group if row.get("first_action")]
        match_rate = (
            float(np.mean([action in oracle_actions for action in first_actions]))
            if first_actions
            else float("nan")
        )
        realized_gap = oracle_rewards.get(case_id, float("nan")) - passive_rewards.get(case_id, float("nan"))
        debug_rows.append(
            {
                "case_id": case_id,
                "case_type": group[0].get("case_type"),
                "scenario": group[0].get("scenario"),
                "method": method,
                "n": len(group),
                "conventions": ",".join(sorted({str(row["convention"]) for row in group})),
                "seeds": ",".join(sorted({str(row["seed"]) for row in group})),
                "episode_reward_mean": _mean_numeric(group, "episode_reward"),
                "oracle_reward_mean": oracle_rewards.get(case_id),
                "passive_reward_mean": passive_rewards.get(case_id),
                "oracle_passive_gap_case": case.get("oracle_passive_gap"),
                "oracle_passive_gap_realized": realized_gap,
                "regret_to_oracle_mean": _mean_numeric(group, "regret_to_oracle"),
                "first_actions": ",".join(sorted(set(first_actions))),
                "oracle_first_actions": group[0].get("oracle_first_actions"),
                "passive_first_action": group[0].get("passive_first_action"),
                "first_action_matches_oracle_rate": match_rate,
                "max_delta_info_mean": _mean_numeric(group, "max_delta_info"),
                "max_delta_info_max": max(float(row["max_delta_info"]) for row in group),
                "max_mi": max(float(row.get("max_mi", 0.0)) for row in group),
                "mean_delta_info_mean": _mean_numeric(group, "mean_delta_info"),
                "mean_mi_gain_mean": _mean_numeric(group, "mean_mi_gain"),
                "future_return_delta_info_corr_mean": _mean_numeric(
                    group, "future_return_delta_info_corr", default=float("nan")
                ),
                "future_return_mi_corr_mean": _mean_numeric(
                    group, "future_return_mi_corr", default=float("nan")
                ),
                "future_reward_gain_after_high_delta_info": _mean_numeric(
                    group, "future_reward_gain_after_high_delta_info", default=float("nan")
                ),
                "oracle_gap_after_high_delta_info_mean": _mean_numeric(
                    group, "oracle_gap_after_high_delta_info", default=float("nan")
                ),
                "high_delta_info_count_mean": _mean_numeric(group, "high_delta_info_count"),
                "diagnostic_cost_mean": _mean_numeric(group, "diagnostic_cost"),
                "diagnostic_count_mean": _mean_numeric(group, "diagnostic_count"),
                "reward_after_first_diagnostic_mean": _mean_numeric(
                    group, "reward_after_first_diagnostic", default=float("nan")
                ),
                "best_response_flip": case.get("best_response_flip"),
                "action_gap": case.get("action_gap"),
                "best_diag_option": case.get("best_diag_option"),
                "observation_separation": case.get("observation_separation"),
                "best_diag_return_gain": case.get("best_diag_return_gain"),
                "high_mi_low_value_distractor": case.get("high_mi_low_value_distractor"),
            }
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in debug_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return debug_rows


def _finite_or_low(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -float("inf")
    if math.isnan(numeric):
        return -float("inf")
    return numeric


def write_scenario_debug_views(
    debug_rows: list[dict[str, object]],
    output_dir: Path,
    top_k: int,
) -> dict[str, dict[str, str | int]]:
    view_specs = {
        "top_oracle_passive_gap": (
            "scenario_debug_top_oracle_passive_gap.csv",
            "oracle_passive_gap_realized",
            lambda row: _finite_or_low(row.get("oracle_passive_gap_realized")),
        ),
        "top_max_delta_info": (
            "scenario_debug_top_max_delta_info.csv",
            "max_delta_info_max",
            lambda row: _finite_or_low(row.get("max_delta_info_max")),
        ),
        "top_delta_info_but_low_return": (
            "scenario_debug_top_delta_info_but_low_return.csv",
            "max_delta_info_max_minus_future_reward_gain_after_high_delta_info",
            lambda row: _finite_or_low(row.get("max_delta_info_max"))
            - max(0.0, _finite_or_low(row.get("future_reward_gain_after_high_delta_info"))),
        ),
        "top_mi_but_low_return": (
            "scenario_debug_top_mi_but_low_return.csv",
            "max_mi_minus_future_reward_gain_after_high_delta_info",
            lambda row: _finite_or_low(row.get("max_mi"))
            - max(0.0, _finite_or_low(row.get("future_reward_gain_after_high_delta_info"))),
        ),
    }
    fieldnames = list(debug_rows[0].keys()) if debug_rows else []
    metadata: dict[str, dict[str, str | int]] = {}
    for name, (filename, sort_key, key_fn) in view_specs.items():
        path = output_dir / filename
        sorted_rows = sorted(debug_rows, key=key_fn, reverse=True)[:top_k]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted_rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        metadata[name] = {
            "file": filename,
            "sort_key": sort_key,
            "order": "desc",
            "top_k": top_k,
        }
    return metadata


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_method_list(raw: str) -> list[str]:
    methods = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}; expected subset of {METHODS}")
    if "oracle" not in methods:
        methods.append("oracle")
    return methods


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_metadata() -> dict[str, str | None]:
    return {
        "git_commit": git_value(["git", "rev-parse", "--short", "HEAD"]),
        "branch": git_value(["git", "branch", "--show-current"]),
    }


def build_cases(
    scenario_set: str,
    max_conventions: int,
    max_steps: int,
    depth: int,
    likelihood_error: float,
    delta_gap: float,
    delta_action: float,
    delta_obs: float,
    delta_return: float,
    delta_mi: float,
    delta_value: float,
    max_diagnostic_cost: float,
) -> tuple[list[DiagnosticCase], list[DiagnosticCase]]:
    if scenario_set == "default_start":
        conventions = all_convention_keys()
        if max_conventions > 0:
            conventions = conventions[:max_conventions]
        cases = [
            DiagnosticCase(
                case_id=f"default_start_{convention_label(convention)}",
                case_type="default_start",
                scenario=None,
                conventions=(convention,),
                passed_filters=True,
            )
            for convention in conventions
        ]
        return cases, []
    if scenario_set == "handpicked":
        cases: list[DiagnosticCase] = []
        for scenario in HANDPICKED_DIAGNOSTIC_SCENARIOS:
            if len(scenario.conventions) < 2:
                raise ValueError(f"Diagnostic scenario {scenario.name} must include at least two conventions")
            conventions = list(scenario.conventions)
            if max_conventions > 0:
                conventions = conventions[:max_conventions]
            cases.append(
                DiagnosticCase(
                    case_id=scenario.name,
                    case_type="handpicked",
                    scenario=scenario,
                    conventions=tuple(conventions),
                    passed_filters=True,
                )
            )
        return cases, []
    if scenario_set != "diagnostic":
        raise ValueError("--scenario_set must be one of: diagnostic, handpicked, default_start")

    accepted, rejected = synthesize_diagnostic_cases(
        max_steps=max_steps,
        depth=depth,
        likelihood_error=likelihood_error,
        delta_gap=delta_gap,
        delta_action=delta_action,
        delta_obs=delta_obs,
        delta_return=delta_return,
        delta_mi=delta_mi,
        delta_value=delta_value,
        max_diagnostic_cost=max_diagnostic_cost,
    )
    if max_conventions > 0:
        accepted = accepted[:max_conventions]
    return accepted, rejected


def scenario_metadata(scenario_set: str, cases: list[DiagnosticCase]) -> list[dict[str, object]]:
    if scenario_set == "default_start":
        return [{"name": "default_start", "description": "Original reset state over convention assignments"}]
    if scenario_set == "diagnostic":
        return [
            {
                "case_id": case.case_id,
                "case_type": case.case_type,
                "scenario": case.scenario.name if case.scenario is not None else "default_start",
                "conventions": [convention_label(convention) for convention in case.conventions],
                "oracle_passive_gap": case.oracle_passive_gap,
                "best_diag_return_gain": case.best_diag_return_gain,
                "high_mi_low_value_distractor": case.high_mi_low_value_distractor,
            }
            for case in cases
        ]
    return [
        {
            "name": scenario.name,
            "description": scenario.description,
            "conventions": [convention_label(convention) for convention in scenario.conventions],
        }
        for scenario in HANDPICKED_DIAGNOSTIC_SCENARIOS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--episodes_per_convention", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_conventions", type=int, default=0,
                        help="Use the first N convention assignments/case conventions; 0 means all")
    parser.add_argument("--scenario_set", type=str, default="diagnostic",
                        choices=("diagnostic", "handpicked", "default_start"))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--likelihood_error", type=float, default=0.05)
    parser.add_argument("--alignment_weight_threshold", type=float, default=0.5)
    parser.add_argument("--case_horizon", type=int, default=ROLLOUT_DEPTH)
    parser.add_argument("--min_cases", type=int, default=DEFAULT_MIN_CASES)
    parser.add_argument("--delta_gap", type=float, default=DEFAULT_DELTA_GAP)
    parser.add_argument("--delta_action", type=float, default=DEFAULT_DELTA_ACTION)
    parser.add_argument("--delta_obs", type=float, default=DEFAULT_DELTA_OBS)
    parser.add_argument("--delta_return", type=float, default=DEFAULT_DELTA_RETURN)
    parser.add_argument("--delta_mi", type=float, default=DEFAULT_DELTA_MI)
    parser.add_argument("--delta_value", type=float, default=DEFAULT_DELTA_VALUE)
    parser.add_argument("--max_diagnostic_cost", type=float, default=DEFAULT_MAX_DIAGNOSTIC_COST)
    parser.add_argument("--oracle_margin", type=float, default=DEFAULT_ORACLE_MARGIN)
    parser.add_argument("--debug_top_k", type=int, default=DEFAULT_DEBUG_TOP_K)
    parser.add_argument("--output_dir", type=str, default="results/toy_symbolic")
    parser.add_argument("--progress_every", type=int, default=0,
                        help="Print progress every N completed method episodes; 0 disables")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    methods = parse_method_list(args.methods)
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    if args.episodes_per_convention <= 0:
        raise ValueError("--episodes_per_convention must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive")
    if args.max_conventions < 0:
        raise ValueError("--max_conventions must be non-negative")
    if args.case_horizon <= 0:
        raise ValueError("--case_horizon must be positive")
    if args.min_cases < 0:
        raise ValueError("--min_cases must be non-negative")
    if args.max_diagnostic_cost < 0:
        raise ValueError("--max_diagnostic_cost must be non-negative")
    if args.oracle_margin < 0:
        raise ValueError("--oracle_margin must be non-negative")
    if args.debug_top_k <= 0:
        raise ValueError("--debug_top_k must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress_every must be non-negative")
    set_seed(min(seeds) if seeds else 0)

    rows: list[dict[str, float | int | str | None]] = []
    shared_value_cache: dict[tuple, float] = {}
    cases, rejected_cases = build_cases(
        args.scenario_set,
        args.max_conventions,
        args.max_steps,
        args.case_horizon,
        args.likelihood_error,
        args.delta_gap,
        args.delta_action,
        args.delta_obs,
        args.delta_return,
        args.delta_mi,
        args.delta_value,
        args.max_diagnostic_cost,
    )
    case_rows = [case_to_row(case) for case in cases] + [case_to_row(case) for case in rejected_cases]
    total_runs = (
        len(seeds)
        * sum(len(case.conventions) for case in cases)
        * args.episodes_per_convention
        * len(methods)
    )
    completed_runs = 0
    for seed in seeds:
        for case in cases:
            for convention in case.conventions:
                for repeat_idx in range(args.episodes_per_convention):
                    episode_seed = seed * 1000 + repeat_idx
                    for method in methods:
                        rows.append(
                            run_episode(
                                method=method,
                                convention=convention,
                                seed=episode_seed,
                                max_steps=args.max_steps,
                                alpha=args.alpha,
                                beta=args.beta,
                                likelihood_error=args.likelihood_error,
                                alignment_weight_threshold=args.alignment_weight_threshold,
                                case=case,
                                shared_value_cache=shared_value_cache,
                            )
                        )
                        completed_runs += 1
                        if args.progress_every and completed_runs % args.progress_every == 0:
                            print(f"Completed {completed_runs}/{total_runs} method episodes", flush=True)

    if rows:
        add_oracle_regrets(rows)
    summary = {
        method: summarize_method([row for row in rows if row["method"] == method])
        for method in methods
        if any(row["method"] == method for row in rows)
    }

    summary_methods = [method for method in methods if method in summary]
    if summary_methods:
        costs = np.array([summary[method]["probe_cost_mean"] for method in summary_methods], dtype=np.float64)
        rewards = np.array([summary[method]["episode_reward_mean"] for method in summary_methods], dtype=np.float64)
        frontier_idx = pareto_frontier(costs, rewards)
        frontier_methods = [summary_methods[int(idx)] for idx in frontier_idx]
    else:
        frontier_methods = []

    output_dir = Path(args.output_dir)
    validation = tiered_validation(
        summary,
        rows,
        case_rows,
        methods,
        args.min_cases,
        args.delta_gap,
        args.delta_action,
        args.delta_return,
        args.oracle_margin,
    )
    debug_rows = write_scenario_debug_csv(rows, case_rows, output_dir / "scenario_debug.csv")
    debug_views = write_scenario_debug_views(debug_rows, output_dir, args.debug_top_k)
    output = {
        "schema": SYMBOLIC_SCHEMA,
        **git_metadata(),
        "config": vars(args),
        "methods": methods,
        "scenario_set": args.scenario_set,
        "scenarios": scenario_metadata(args.scenario_set, cases),
        "case_synthesis": {
            "thresholds": {
                "delta_gap": args.delta_gap,
                "delta_action": args.delta_action,
                "delta_obs": args.delta_obs,
                "delta_return": args.delta_return,
                "delta_mi": args.delta_mi,
                "delta_value": args.delta_value,
                "max_diagnostic_cost": args.max_diagnostic_cost,
                "oracle_margin": args.oracle_margin,
            },
            "case_horizon": args.case_horizon,
            "n_accepted_cases": len(cases),
            "n_rejected_cases": len(rejected_cases),
            "rejection_counts": {
                reason: sum(1 for case in rejected_cases if case.failure_reason == reason)
                for reason in sorted({case.failure_reason for case in rejected_cases if case.failure_reason})
            },
        },
        "n_cases": len(cases),
        "n_candidate_cases": len(cases) + len(rejected_cases),
        "n_conventions": len({convention for case in cases for convention in case.conventions}),
        "n_rows": len(rows),
        "summary": summary,
        "validation": validation,
        "debug_views": debug_views,
        "reward_probe_cost_pareto_methods": frontier_methods,
        "ground_truth_note": (
            "Factor accuracy, confidence, ECE, and regret use the toy environment's "
            "ConventionAssignment ground truth. No metric compares against another model output."
        ),
    }
    save_results(output, str(output_dir / "summary.json"))
    write_csv(rows, output_dir / "episodes.csv")
    write_cases_csv(case_rows, output_dir / "cases.csv")

    print(f"Saved symbolic toy pilot summary to {output_dir / 'summary.json'}")
    print(f"Saved per-episode rows to {output_dir / 'episodes.csv'}")
    print(f"Saved diagnostic case rows to {output_dir / 'cases.csv'}")
    print(f"Saved per-scenario debug rows to {output_dir / 'scenario_debug.csv'}")
    print(f"Saved per-scenario debug top-k views to {output_dir}")
    print(
        "Tiered validation:",
        validation["overall_status"],
        "| Tier0",
        validation["tier0_case_validity"]["status"],
        "| Tier1",
        validation["tier1_oracle_sanity"]["status"],
        "| Tier2",
        validation["tier2_diagnostic_policy"]["status"],
        "| Tier3",
        validation["tier3_soft_correlation"]["status"],
    )


if __name__ == "__main__":
    main()
