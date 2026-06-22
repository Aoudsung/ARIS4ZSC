"""
Toy Interaction Factor Game: 2-agent gridworld with known ground-truth
coordination factors, modes, and CE values.

Layout: 7x7 grid with:
- 2 agents (ego at top-left, partner at bottom-right)
- 1 bottleneck corridor (center row, width 1)
- 2 resource piles (top-right, bottom-left)
- 2 delivery zones (left-center, right-center)

Ground-truth interaction factors:
  f1 = (cross_corridor_ego, cross_corridor_partner) — bottleneck yielding
  f2 = (fetch_resource_A, fetch_resource_A) — resource A contention
  f3 = (deliver_left, deliver_right) — delivery role assignment

Each factor has K=2 or K=3 modes representing different coordination protocols.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class Action(IntEnum):
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    PICKUP = 5
    DROP = 6


DELTAS = {
    Action.NOOP: (0, 0),
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.PICKUP: (0, 0),
    Action.DROP: (0, 0),
}

GRID_H, GRID_W = 7, 7

BOTTLENECK_ROW = 3
BOTTLENECK_COL = 3

RESOURCE_A = (1, 5)
RESOURCE_B = (5, 1)
DELIVER_LEFT = (3, 0)
DELIVER_RIGHT = (3, 6)

EGO_START = (0, 0)
PARTNER_START = (6, 6)


@dataclass
class FactorMode:
    factor_id: int
    mode: int
    description: str


@dataclass
class ConventionAssignment:
    modes: dict[int, int] = field(default_factory=dict)

    def __hash__(self):
        return hash(tuple(sorted(self.modes.items())))


FACTOR_DESCRIPTIONS = {
    0: {
        0: "ego yields at bottleneck, partner goes first",
        1: "partner yields at bottleneck, ego goes first",
        2: "alternate: whoever arrives first goes",
    },
    1: {
        0: "ego owns resource A, partner fetches B",
        1: "partner owns resource A, ego fetches B",
    },
    2: {
        0: "ego delivers left, partner delivers right",
        1: "ego delivers right, partner delivers left",
    },
}

NUM_FACTORS = 3
FACTOR_MODES = {0: 3, 1: 2, 2: 2}


class ToyFactorGameEnv:
    def __init__(
        self,
        partner_convention: Optional[ConventionAssignment] = None,
        max_steps: int = 50,
        seed: Optional[int] = None,
    ):
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.partner_convention = partner_convention or self._random_convention()
        self.reset()

    def _random_convention(self) -> ConventionAssignment:
        return ConventionAssignment(
            modes={f: self.rng.randint(0, FACTOR_MODES[f]) for f in range(NUM_FACTORS)}
        )

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.ego_pos = list(EGO_START)
        self.partner_pos = list(PARTNER_START)
        self.ego_carrying = False
        self.partner_carrying = False
        self.resource_a_available = True
        self.resource_b_available = True
        self.deliveries_left = 0
        self.deliveries_right = 0
        self.collisions = 0
        self.total_reward = 0.0
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = np.zeros(GRID_H * GRID_W * 4 + 6, dtype=np.float32)
        ego_idx = self.ego_pos[0] * GRID_W + self.ego_pos[1]
        obs[ego_idx] = 1.0
        partner_idx = GRID_H * GRID_W + self.partner_pos[0] * GRID_W + self.partner_pos[1]
        obs[partner_idx] = 1.0
        if self.resource_a_available:
            ra_idx = 2 * GRID_H * GRID_W + RESOURCE_A[0] * GRID_W + RESOURCE_A[1]
            obs[ra_idx] = 1.0
        if self.resource_b_available:
            rb_idx = 2 * GRID_H * GRID_W + RESOURCE_B[0] * GRID_W + RESOURCE_B[1]
            obs[rb_idx] = 1.0
        landmark_base = 3 * GRID_H * GRID_W
        obs[landmark_base + BOTTLENECK_ROW * GRID_W + BOTTLENECK_COL] = 1.0
        obs[landmark_base + RESOURCE_A[0] * GRID_W + RESOURCE_A[1]] = 0.5
        obs[landmark_base + RESOURCE_B[0] * GRID_W + RESOURCE_B[1]] = 0.5
        obs[landmark_base + DELIVER_LEFT[0] * GRID_W + DELIVER_LEFT[1]] = 0.75
        obs[landmark_base + DELIVER_RIGHT[0] * GRID_W + DELIVER_RIGHT[1]] = 0.75
        base = 4 * GRID_H * GRID_W
        obs[base] = float(self.ego_carrying)
        obs[base + 1] = float(self.partner_carrying)
        obs[base + 2] = float(self.deliveries_left)
        obs[base + 3] = float(self.deliveries_right)
        obs[base + 4] = float(self.step_count) / self.max_steps
        obs[base + 5] = float(self.collisions)
        return obs

    @property
    def obs_dim(self) -> int:
        return GRID_H * GRID_W * 4 + 6

    @property
    def n_actions(self) -> int:
        return len(Action)

    def _move(self, pos: list, action: Action) -> list:
        dr, dc = DELTAS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
            return [nr, nc]
        return pos

    def _partner_action(self) -> Action:
        conv = self.partner_convention.modes
        target = self._partner_target(conv)
        if target is None:
            return Action.NOOP
        dr = np.sign(target[0] - self.partner_pos[0])
        dc = np.sign(target[1] - self.partner_pos[1])
        if self.partner_pos[0] == BOTTLENECK_ROW and abs(self.partner_pos[1] - BOTTLENECK_COL) <= 1:
            if conv[0] == 0:
                pass
            elif conv[0] == 1:
                if self.ego_pos[0] == BOTTLENECK_ROW and abs(self.ego_pos[1] - BOTTLENECK_COL) <= 1:
                    return Action.NOOP
            elif conv[0] == 2:
                if (
                    self.ego_pos[0] == BOTTLENECK_ROW
                    and abs(self.ego_pos[1] - BOTTLENECK_COL) <= 1
                    and self.step_count % 2 == 0
                ):
                    return Action.NOOP

        if tuple(self.partner_pos) == target:
            if not self.partner_carrying and (
                (target == RESOURCE_A and self.resource_a_available)
                or (target == RESOURCE_B and self.resource_b_available)
            ):
                return Action.PICKUP
            if self.partner_carrying and target in (DELIVER_LEFT, DELIVER_RIGHT):
                return Action.DROP
            return Action.NOOP

        if dr != 0:
            return Action.DOWN if dr > 0 else Action.UP
        if dc != 0:
            return Action.RIGHT if dc > 0 else Action.LEFT
        return Action.NOOP

    def _partner_target(self, conv: dict):
        if not self.partner_carrying:
            if conv[1] == 0 and self.resource_b_available:
                return RESOURCE_B
            elif conv[1] == 1 and self.resource_a_available:
                return RESOURCE_A
            elif self.resource_a_available:
                return RESOURCE_A
            elif self.resource_b_available:
                return RESOURCE_B
            return None
        else:
            if conv[2] == 0:
                return DELIVER_RIGHT
            else:
                return DELIVER_LEFT

    def step(
        self, ego_action: int, partner_action_override: int | None = None
    ) -> tuple[np.ndarray, float, bool, dict]:
        ego_action = Action(ego_action)
        partner_action = (
            Action(partner_action_override)
            if partner_action_override is not None
            else self._partner_action()
        )

        new_ego = self._move(self.ego_pos, ego_action)
        new_partner = self._move(self.partner_pos, partner_action)

        collision = new_ego == new_partner
        if collision:
            self.collisions += 1
            new_ego = self.ego_pos
            new_partner = self.partner_pos

        self.ego_pos = new_ego
        self.partner_pos = new_partner

        reward = -0.1
        if ego_action == Action.PICKUP and not self.ego_carrying:
            if tuple(self.ego_pos) == RESOURCE_A and self.resource_a_available:
                self.ego_carrying = True
                self.resource_a_available = False
            elif tuple(self.ego_pos) == RESOURCE_B and self.resource_b_available:
                self.ego_carrying = True
                self.resource_b_available = False
        if partner_action == Action.PICKUP and not self.partner_carrying:
            if tuple(self.partner_pos) == RESOURCE_A and self.resource_a_available:
                self.partner_carrying = True
                self.resource_a_available = False
            elif tuple(self.partner_pos) == RESOURCE_B and self.resource_b_available:
                self.partner_carrying = True
                self.resource_b_available = False

        if ego_action == Action.DROP and self.ego_carrying:
            if tuple(self.ego_pos) == DELIVER_LEFT:
                self.ego_carrying = False
                self.deliveries_left += 1
                reward += 5.0
            elif tuple(self.ego_pos) == DELIVER_RIGHT:
                self.ego_carrying = False
                self.deliveries_right += 1
                reward += 5.0
        if partner_action == Action.DROP and self.partner_carrying:
            if tuple(self.partner_pos) == DELIVER_LEFT:
                self.partner_carrying = False
                self.deliveries_left += 1
                reward += 3.0
            elif tuple(self.partner_pos) == DELIVER_RIGHT:
                self.partner_carrying = False
                self.deliveries_right += 1
                reward += 3.0

        if collision:
            reward -= 2.0

        self.step_count += 1
        self.total_reward += reward
        done = self.step_count >= self.max_steps
        info = {
            "collision": collision,
            "partner_action": int(partner_action),
            "step": self.step_count,
            "convention": self.partner_convention.modes.copy(),
        }
        return self._obs(), reward, done, info

    def get_ground_truth_factors(self) -> dict:
        return self.partner_convention.modes.copy()

    def get_all_conventions(self) -> list[ConventionAssignment]:
        conventions = []
        for f0 in range(FACTOR_MODES[0]):
            for f1 in range(FACTOR_MODES[1]):
                for f2 in range(FACTOR_MODES[2]):
                    conventions.append(ConventionAssignment(modes={0: f0, 1: f1, 2: f2}))
        return conventions
