"""
Macro-option library for the toy factor game.
Each option is a task-valid policy that pursues a specific subgoal.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .env import (
    BOTTLENECK_COL,
    BOTTLENECK_ROW,
    DELIVER_LEFT,
    DELIVER_RIGHT,
    RESOURCE_A,
    RESOURCE_B,
    Action,
)


class OptionID(IntEnum):
    GOTO_RESOURCE_A = 0
    GOTO_RESOURCE_B = 1
    CROSS_CORRIDOR = 2
    WAIT_AT_BOTTLENECK = 3
    DELIVER_LEFT = 4
    DELIVER_RIGHT = 5
    PICKUP = 6
    DROP = 7
    NOOP = 8


NUM_OPTIONS = len(OptionID)

OPTION_NAMES = {
    OptionID.GOTO_RESOURCE_A: "go_to_resource_A",
    OptionID.GOTO_RESOURCE_B: "go_to_resource_B",
    OptionID.CROSS_CORRIDOR: "cross_corridor",
    OptionID.WAIT_AT_BOTTLENECK: "wait_at_bottleneck",
    OptionID.DELIVER_LEFT: "deliver_left",
    OptionID.DELIVER_RIGHT: "deliver_right",
    OptionID.PICKUP: "pickup",
    OptionID.DROP: "drop",
    OptionID.NOOP: "noop",
}


@dataclass
class InteractionFactor:
    factor_id: int
    option_i: OptionID
    option_j: OptionID
    ce_value: float
    n_modes: int
    description: str


GROUND_TRUTH_FACTORS = [
    InteractionFactor(
        factor_id=0,
        option_i=OptionID.CROSS_CORRIDOR,
        option_j=OptionID.CROSS_CORRIDOR,
        ce_value=4.5,
        n_modes=3,
        description="bottleneck traversal: who yields",
    ),
    InteractionFactor(
        factor_id=1,
        option_i=OptionID.GOTO_RESOURCE_A,
        option_j=OptionID.GOTO_RESOURCE_A,
        ce_value=3.2,
        n_modes=2,
        description="resource A contention: who owns it",
    ),
    InteractionFactor(
        factor_id=2,
        option_i=OptionID.DELIVER_LEFT,
        option_j=OptionID.DELIVER_RIGHT,
        ce_value=2.8,
        n_modes=2,
        description="delivery role: who goes left vs right",
    ),
]

NON_CRITICAL_FACTORS = [
    InteractionFactor(
        factor_id=3,
        option_i=OptionID.GOTO_RESOURCE_B,
        option_j=OptionID.GOTO_RESOURCE_B,
        ce_value=0.3,
        n_modes=2,
        description="resource B access order (low externality)",
    ),
]

IRRELEVANT_FACTORS = [
    InteractionFactor(
        factor_id=4,
        option_i=OptionID.NOOP,
        option_j=OptionID.NOOP,
        ce_value=0.0,
        n_modes=2,
        description="both idle (zero externality)",
    ),
]

CE_THRESHOLD = 1.0


def compute_support_graph(
    factors: list[InteractionFactor], threshold: float = CE_THRESHOLD
) -> list[InteractionFactor]:
    return [f for f in factors if f.ce_value > threshold]


def get_option_action(option: OptionID, ego_pos: list, carrying: bool) -> int:
    target = _option_target(option)
    if target is None:
        if option == OptionID.PICKUP:
            return int(Action.PICKUP)
        if option == OptionID.DROP:
            return int(Action.DROP)
        return int(Action.NOOP)

    if carrying and tuple(ego_pos) == target and option in (OptionID.DELIVER_LEFT, OptionID.DELIVER_RIGHT):
        return int(Action.DROP)

    dr = np.sign(target[0] - ego_pos[0])
    dc = np.sign(target[1] - ego_pos[1])
    if dr != 0:
        return int(Action.DOWN) if dr > 0 else int(Action.UP)
    if dc != 0:
        return int(Action.RIGHT) if dc > 0 else int(Action.LEFT)
    return int(Action.NOOP)


def get_option_cost(option: OptionID) -> float:
    if option == OptionID.NOOP:
        return 1.5
    if option == OptionID.WAIT_AT_BOTTLENECK:
        return 1.2
    if option == OptionID.CROSS_CORRIDOR:
        return 1.1
    return 1.0


def get_valid_options(env) -> list[OptionID]:
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


def _option_target(option: OptionID):
    targets = {
        OptionID.GOTO_RESOURCE_A: RESOURCE_A,
        OptionID.GOTO_RESOURCE_B: RESOURCE_B,
        OptionID.CROSS_CORRIDOR: (BOTTLENECK_ROW, BOTTLENECK_COL),
        OptionID.WAIT_AT_BOTTLENECK: (BOTTLENECK_ROW, BOTTLENECK_COL - 1),
        OptionID.DELIVER_LEFT: DELIVER_LEFT,
        OptionID.DELIVER_RIGHT: DELIVER_RIGHT,
    }
    return targets.get(option)
