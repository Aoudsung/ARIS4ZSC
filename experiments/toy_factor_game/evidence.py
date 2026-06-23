"""Deterministic factor-local evidence routing for Toy Factor Game."""

from __future__ import annotations

from .env import (
    BOTTLENECK_COL,
    BOTTLENECK_ROW,
    DELIVER_LEFT,
    DELIVER_RIGHT,
    RESOURCE_A,
    RESOURCE_B,
    Action,
)
from .graph_config import GraphConfig, GraphFactorSpec
from .options import NUM_OPTIONS, OptionID


EVIDENCE_DIM = 18


def _manhattan(pos_a, pos_b) -> float:
    return float(abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]))


def _near(pos, landmark, radius: int = 1) -> bool:
    return _manhattan(pos, landmark) <= radius


def _norm_dist(pos, landmark) -> float:
    return min(_manhattan(pos, landmark) / 12.0, 1.0)


def relevance_options_for_factor(factor: GraphFactorSpec) -> set[OptionID]:
    if factor.env_factor_id == 0:
        return {OptionID.CROSS_CORRIDOR, OptionID.WAIT_AT_BOTTLENECK}
    if factor.env_factor_id == 1:
        return {OptionID.GOTO_RESOURCE_A, OptionID.PICKUP}
    if factor.env_factor_id == 2:
        return {OptionID.DELIVER_LEFT, OptionID.DELIVER_RIGHT, OptionID.DROP}
    return {factor.option_i, factor.option_j}


def option_relevance_mask(graph_config: GraphConfig) -> list[list[bool]]:
    mask = [[False for _ in range(NUM_OPTIONS)] for _ in range(graph_config.n_factors)]
    for factor_idx, factor in enumerate(graph_config.factors):
        for option in relevance_options_for_factor(factor):
            mask[factor_idx][int(option)] = True
    return mask


def route_event_to_factors(event: dict, graph_config: GraphConfig) -> list[list[float]]:
    return [_route_event_to_factor(event, factor) for factor in graph_config.factors]


def _route_event_to_factor(event: dict, factor: GraphFactorSpec) -> list[float]:
    vec = [0.0] * EVIDENCE_DIM
    option_id = OptionID(int(event.get("ego_option", int(OptionID.NOOP))))
    ego_action = Action(int(event.get("ego_action", int(Action.NOOP))))
    partner_action = Action(int(event.get("partner_action", int(Action.NOOP))))
    ego_before = tuple(event["ego_pos_before"])
    ego_after = tuple(event["ego_pos_after"])
    partner_before = tuple(event["partner_pos_before"])
    partner_after = tuple(event["partner_pos_after"])

    vec[0] = 1.0
    vec[1] = float(option_id in relevance_options_for_factor(factor))
    vec[2] = float(option_id in (factor.option_i, factor.option_j))

    if factor.env_factor_id == 0:
        bottleneck = (BOTTLENECK_ROW, BOTTLENECK_COL)
        vec[3] = float(event["collision"] and (_near(ego_after, bottleneck, 1) or _near(partner_after, bottleneck, 1)))
        vec[4] = float(_near(ego_after, bottleneck, 1))
        vec[5] = float(_near(partner_after, bottleneck, 1))
        vec[6] = float(ego_action == Action.NOOP and _near(ego_before, bottleneck, 2))
        vec[7] = float(partner_action == Action.NOOP and _near(partner_before, bottleneck, 2))
        vec[16] = (_norm_dist(partner_after, bottleneck) - _norm_dist(ego_after, bottleneck))
        return vec

    if factor.env_factor_id == 1:
        vec[8] = float(
            ego_action == Action.PICKUP
            and _near(ego_after, RESOURCE_A, 0)
            and not event["ego_carrying_before"]
            and event["ego_carrying_after"]
        )
        vec[9] = float(
            partner_action == Action.PICKUP
            and _near(partner_after, RESOURCE_A, 0)
            and not event["partner_carrying_before"]
            and event["partner_carrying_after"]
        )
        vec[10] = 1.0 - _norm_dist(ego_after, RESOURCE_A)
        vec[11] = 1.0 - _norm_dist(partner_after, RESOURCE_A)
        return vec

    if factor.env_factor_id == 2:
        left_delta = event["deliveries_left_after"] - event["deliveries_left_before"]
        right_delta = event["deliveries_right_after"] - event["deliveries_right_before"]
        vec[12] = float(left_delta > 0 and ego_action == Action.DROP and _near(ego_after, DELIVER_LEFT, 0))
        vec[13] = float(left_delta > 0 and partner_action == Action.DROP and _near(partner_after, DELIVER_LEFT, 0))
        vec[14] = float(right_delta > 0 and ego_action == Action.DROP and _near(ego_after, DELIVER_RIGHT, 0))
        vec[15] = float(right_delta > 0 and partner_action == Action.DROP and _near(partner_after, DELIVER_RIGHT, 0))
        vec[17] = float(event["ego_carrying_after"]) - float(event["partner_carrying_after"])
        return vec

    if factor.option_i == OptionID.GOTO_RESOURCE_B or factor.option_j == OptionID.GOTO_RESOURCE_B:
        vec[8] = float(
            ego_action == Action.PICKUP
            and _near(ego_after, RESOURCE_B, 0)
            and not event["ego_carrying_before"]
            and event["ego_carrying_after"]
        )
        vec[9] = float(
            partner_action == Action.PICKUP
            and _near(partner_after, RESOURCE_B, 0)
            and not event["partner_carrying_before"]
            and event["partner_carrying_after"]
        )
        vec[10] = 1.0 - _norm_dist(ego_after, RESOURCE_B)
        vec[11] = 1.0 - _norm_dist(partner_after, RESOURCE_B)
    elif factor.option_i == OptionID.NOOP and factor.option_j == OptionID.NOOP:
        vec[6] = float(ego_action == Action.NOOP)
        vec[7] = float(partner_action == Action.NOOP)
    else:
        vec[3] = float(event["collision"])
    return vec
