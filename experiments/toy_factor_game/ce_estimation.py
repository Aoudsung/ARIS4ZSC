"""Deterministic CE estimation for toy option-pair support induction."""

from __future__ import annotations

import functools
import numpy as np

from .env import (
    BOTTLENECK_COL,
    BOTTLENECK_ROW,
    DELIVER_LEFT,
    DELIVER_RIGHT,
    RESOURCE_A,
    Action,
    ConventionAssignment,
    ToyFactorGameEnv,
)
from .options import NUM_OPTIONS, OptionID, get_option_action


DEFAULT_CE_THRESHOLD = 1.0


def _all_conventions() -> list[ConventionAssignment]:
    return [
        ConventionAssignment(modes={0: f0, 1: f1, 2: f2})
        for f0 in range(3)
        for f1 in range(2)
        for f2 in range(2)
    ]


def _apply_scenario(env: ToyFactorGameEnv, scenario: dict) -> None:
    env.ego_pos = list(scenario["ego_pos"])
    env.partner_pos = list(scenario["partner_pos"])
    env.ego_carrying = bool(scenario.get("ego_carrying", False))
    env.partner_carrying = bool(scenario.get("partner_carrying", False))
    env.resource_a_available = bool(scenario.get("resource_a_available", True))
    env.resource_b_available = bool(scenario.get("resource_b_available", True))


def _probe_scenarios() -> list[dict]:
    return [
        {
            "name": "bottleneck",
            "ego_pos": (BOTTLENECK_ROW, BOTTLENECK_COL - 1),
            "partner_pos": (BOTTLENECK_ROW, BOTTLENECK_COL + 1),
            "horizon": 3,
        },
        {
            "name": "resource_a",
            "ego_pos": (RESOURCE_A[0], RESOURCE_A[1] - 1),
            "partner_pos": (RESOURCE_A[0], RESOURCE_A[1] + 1),
            "horizon": 3,
        },
        {
            "name": "delivery_roles",
            "ego_pos": (DELIVER_LEFT[0], DELIVER_LEFT[1] + 1),
            "partner_pos": (DELIVER_RIGHT[0], DELIVER_RIGHT[1] - 1),
            "ego_carrying": True,
            "partner_carrying": True,
            "resource_a_available": False,
            "resource_b_available": False,
            "horizon": 6,
        },
    ]


def _rollout_return(
    ego_option: OptionID | None,
    partner_option: OptionID | None,
    convention: ConventionAssignment,
    scenario: dict,
) -> tuple[float, int, float]:
    env = ToyFactorGameEnv(partner_convention=convention, max_steps=int(scenario["horizon"]), seed=0)
    _apply_scenario(env, scenario)
    total = 0.0
    collisions = 0
    for _step in range(int(scenario["horizon"])):
        ego_action = (
            int(Action.NOOP)
            if ego_option is None
            else get_option_action(ego_option, env.ego_pos, env.ego_carrying)
        )
        partner_action = (
            int(Action.NOOP)
            if partner_option is None
            else get_option_action(partner_option, env.partner_pos, env.partner_carrying)
        )
        _, reward, done, info = env.step(ego_action, partner_action_override=partner_action)
        total += float(reward)
        collisions += int(info["collision"])
        if done:
            break
    delivery_coverage = float(env.deliveries_left > 0) + float(env.deliveries_right > 0)
    return total, collisions, delivery_coverage


@functools.lru_cache(maxsize=None)
def estimate_ce_matrix(n_conventions: int | None = None, seed: int = 42) -> np.ndarray:
    """Estimate CE(w_i, w_j) averaged over conventions and probe scenarios.

    The default is the all-convention expectation. Passing ``n_conventions`` is
    an explicit sampled-CE diagnostic and is not used for support induction.
    """
    all_conventions = tuple(_all_conventions())
    if n_conventions is None or n_conventions >= len(all_conventions):
        conventions = all_conventions
    else:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(all_conventions), size=max(1, n_conventions), replace=False)
        conventions = tuple(all_conventions[int(idx)] for idx in sorted(indices))
    scenarios = _probe_scenarios()
    ce = np.zeros((NUM_OPTIONS, NUM_OPTIONS), dtype=np.float64)

    for option_i in OptionID:
        for option_j in OptionID:
            effects = []
            for convention in conventions:
                for scenario in scenarios:
                    joint = _rollout_return(option_i, option_j, convention, scenario)
                    ego_only = _rollout_return(option_i, None, convention, scenario)
                    partner_only = _rollout_return(None, option_j, convention, scenario)
                    baseline = _rollout_return(None, None, convention, scenario)
                    reward_effect = abs((joint[0] - baseline[0]) - (ego_only[0] - baseline[0]) - (partner_only[0] - baseline[0]))
                    collision_effect = 2.0 * abs((joint[1] - baseline[1]) - (ego_only[1] - baseline[1]) - (partner_only[1] - baseline[1]))
                    coverage_effect = abs((joint[2] - baseline[2]) - (ego_only[2] - baseline[2]) - (partner_only[2] - baseline[2]))
                    joint_success_effect = 4.0 * max(0.0, joint[2] - max(ego_only[2], partner_only[2], baseline[2]))
                    effects.append(reward_effect + collision_effect + coverage_effect + joint_success_effect)
            ce[int(option_i), int(option_j)] = float(np.mean(effects))

    ce.flags.writeable = False
    return ce


def ce_metadata(
    n_conventions: int | None = None,
    seed: int = 42,
    threshold: float = DEFAULT_CE_THRESHOLD,
) -> dict:
    all_conventions = tuple(_all_conventions())
    if n_conventions is None or n_conventions >= len(all_conventions):
        count = len(all_conventions)
        mode = "all_conventions"
    else:
        count = max(1, int(n_conventions))
        mode = "sampled_conventions"
    return {
        "mode": mode,
        "n_conventions": count,
        "seed": int(seed),
        "n_scenarios": len(_probe_scenarios()),
        "threshold": float(threshold),
    }


def induce_graph(ce_matrix: np.ndarray, threshold: float = DEFAULT_CE_THRESHOLD) -> list[tuple[OptionID, OptionID, float]]:
    factors = []
    for option_i in OptionID:
        for option_j in OptionID:
            value = float(ce_matrix[int(option_i), int(option_j)])
            if value > threshold:
                factors.append((option_i, option_j, value))
    return factors
