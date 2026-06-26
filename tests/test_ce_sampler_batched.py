from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from jaxmarl.environments.overcooked_v2.common import Actions

from experiments.overcooked_v2 import batched_rollout, ce_sampler


def _state() -> SimpleNamespace:
    agents = SimpleNamespace(
        pos=SimpleNamespace(
            x=np.asarray([0, 1]),
            y=np.asarray([0, 1]),
        ),
        dir=np.asarray([0, 0]),
        inventory=np.asarray([0, 0]),
    )
    return SimpleNamespace(
        agents=agents,
        grid=np.zeros((2, 2, 3), dtype=np.int32),
        recipe=np.asarray(0),
    )


class _FakeBatchedEnvPool:
    def __init__(self, raw_env, batch_size: int):
        del raw_env
        self.batch_size = int(batch_size)
        self._states = [_state() for _ in range(self.batch_size)]

    def reset(self, seeds: np.ndarray) -> None:
        del seeds
        self._states = [_state() for _ in range(self.batch_size)]

    def reset_indices(self, indices: np.ndarray, seeds: np.ndarray) -> None:
        del seeds
        for idx in np.asarray(indices, dtype=int):
            self._states[int(idx)] = _state()

    def snapshot(self) -> list[SimpleNamespace]:
        return list(self._states)

    def snapshot_obs(self) -> dict[str, np.ndarray]:
        return {"agent_1": np.zeros((self.batch_size, 1), dtype=np.float32)}

    def step(self, ego_actions: np.ndarray, partner_actions: np.ndarray):
        del ego_actions, partner_actions
        self._states = [_state() for _ in range(self.batch_size)]
        rewards = {"agent_0": np.zeros((self.batch_size,), dtype=np.float32)}
        dones = {"__all__": np.ones((self.batch_size,), dtype=bool)}
        return None, None, rewards, dones, {}

    def get_info_i(self, idx: int, info: dict) -> dict:
        del idx, info
        return {"shaped_reward": {"agent_0": 0.0}}


class _FakePartner:
    def __init__(self, name: str, partner_id: int):
        self.name = name
        self.partner_id = int(partner_id)

    def reset(self, seed: int) -> None:
        del seed

    def act(self, obs, state, rng) -> SimpleNamespace:
        del obs, state, rng
        return SimpleNamespace(
            primitive_action=int(Actions.stay),
            option_id=None,
            option_dist=None,
            option_confidence=0.0,
        )


class _FakeOption:
    id = 0
    kind = "noop"
    max_steps = 1


class _FakeOptionLib:
    options = [_FakeOption()]

    def valid_options(self, state, agent_id: int) -> np.ndarray:
        del state, agent_id
        return np.asarray([True])

    def primitive_action(self, state, agent_id: int, option_id: int) -> int:
        del state, agent_id, option_id
        return int(Actions.stay)

    def option_terminated(
        self,
        opt,
        prev_state,
        state,
        event,
        *,
        agent_id: int,
        elapsed: int,
        runtime,
    ) -> tuple[bool, str]:
        del opt, prev_state, state, event, agent_id, elapsed, runtime
        return True, "max_steps"


def test_batched_replay_uses_unique_episode_ids_per_partner(monkeypatch) -> None:
    monkeypatch.setattr(batched_rollout, "BatchedEnvPool", _FakeBatchedEnvPool)
    episodes = 5
    partners = [_FakePartner("p0", 0), _FakePartner("p1", 1)]
    env = SimpleNamespace(env=object(), layout_name="unit", max_steps=1)

    rows = ce_sampler.collect_option_replay_batched(
        env,
        partners,
        _FakeOptionLib(),
        layout_name="unit",
        episodes=episodes,
        max_options_per_episode=1,
        seed=0,
        batch_size=3,
    )

    episode_ids_by_partner: dict[int, set[int]] = {}
    for row in rows:
        episode_ids_by_partner.setdefault(int(row.partner_id), set()).add(
            int(row.episode_id)
        )

    assert episode_ids_by_partner == {
        0: set(range(episodes)),
        1: set(range(episodes, 2 * episodes)),
    }
