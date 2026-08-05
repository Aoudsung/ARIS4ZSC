"""Static partner distribution and held-out heuristic semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.heuristic_partners import (  # noqa: E402
    ACTION_EAST,
    ACTION_INTERACT,
    ACTION_NOOP,
    HEURISTIC_COURIER,
    HEURISTIC_STATIONARY_HELPER,
    OfficialHeuristicPolicy,
)
from src.path_c.partner_sources import (  # noqa: E402
    build_partner_pool,
    make_static_pool_partner_functions,
)
from src.path_c.task_encoder import official_partner_channel_indexes  # noqa: E402


class Manifest:
    def __init__(self, runs):
        self.runs = tuple(runs)

    def by_role(self, role):
        return tuple(run for run in self.runs if run.role == role)


def _run(index, mechanism, stage, hyperparameter_family="default"):
    return SimpleNamespace(
        run_id=f"run-{index}",
        role="development_support",
        parent_training_run_id=f"parent-{index}",
        generation_mechanism=mechanism,
        checkpoint_stage=stage,
        hyperparameter_family=hyperparameter_family,
        seed=index,
        checkpoint=Path(f"/tmp/run-{index}"),
    )


def _config():
    return SimpleNamespace(
        partner_pool=SimpleNamespace(
            checkpoint_stages=(0.0, 0.5, 1.0),
            heuristic_family_test_only=True,
        )
    )


def test_pool_is_uniform_by_family_then_stage_and_run() -> None:
    runs = []
    index = 0
    for mechanism in ("rnn-sp", "rnn-op"):
        for stage in (0.0, 0.5, 1.0):
            for _ in range(2):
                index += 1
                runs.append(_run(index, mechanism, stage))
    index += 1
    runs.append(_run(index, "rnn-op", 1.0, "wide-a"))
    index += 1
    runs.append(_run(index, "rnn-op", 1.0, "wide-b"))
    members = build_partner_pool(_config(), Manifest(runs), "train")
    assert sum(member.probability for member in members) == pytest.approx(1.0)
    family_mass = {
        family: sum(
            member.probability for member in members if member.family_id == family
        )
        for family in ("sp:default", "op:default", "op:wide-a", "op:wide-b")
    }
    assert family_mass == pytest.approx(
        {
            "sp:default": 0.5,
            "op:default": 1.0 / 6.0,
            "op:wide-a": 1.0 / 6.0,
            "op:wide-b": 1.0 / 6.0,
        }
    )
    mechanism_mass = {
        mechanism: sum(
            member.probability for member in members if member.mechanism == mechanism
        )
        for mechanism in ("sp", "op")
    }
    assert mechanism_mass == pytest.approx({"sp": 0.5, "op": 0.5})
    assert all(member.split == "train" and member.checkpoint is not None for member in members)


def test_test_split_contains_only_checkpoint_free_heuristics() -> None:
    members = build_partner_pool(_config(), Manifest(()), "test")
    assert len(members) == 2
    assert {member.mechanism for member in members} == {"heuristic"}
    assert all(member.checkpoint is None and member.probability == 0.5 for member in members)


def test_manifest_partitions_reserve_comparator_lanes_then_return_them_to_support() -> None:
    class FakePool:
        def initial_carry(self, count):
            return jnp.zeros((count, 1), dtype=jnp.float32)

        def step_with_keys(self, members, observations, carry, starts, keys):
            del observations, starts, keys
            return jnp.zeros_like(members), carry

    functions = make_static_pool_partner_functions(
        external_pool=FakePool(),
        member_probabilities=jnp.full((6,), 1.0 / 6.0),
        member_family_ids=jnp.arange(6),
        member_checkpoint_stages=jnp.ones((6,)),
        member_sampling_groups=jnp.asarray([0, 0, 1, 1, 2, 2]),
    )
    state = functions.initial_state(16, jax.random.PRNGKey(1))
    np.testing.assert_array_equal(
        np.bincount(np.asarray(state.sampling_group), minlength=3), [12, 2, 2]
    )
    assert np.all(
        np.asarray(state.sampling_group)
        == np.asarray([0, 0, 1, 1, 2, 2])[np.asarray(state.member)]
    )
    actions, stepped, context, _ = functions.step(
        jnp.asarray(False),
        state,
        jnp.zeros((16, 5, 5, 39)),
        jnp.ones((16,), dtype=jnp.bool_),
        jax.random.split(jax.random.PRNGKey(2), 16),
    )
    diagnostics = functions.diagnostics(jnp.asarray(False), state, context)
    np.testing.assert_array_equal(
        np.asarray(diagnostics["ppo_mask"]),
        (np.asarray(state.sampling_group) == 0).astype(np.float32),
    )
    reset = functions.observe(
        jnp.asarray(True),
        stepped,
        context,
        jnp.zeros((16, 5, 5, 39)),
        actions,
        jnp.zeros((16,)),
        jnp.ones((16,), dtype=jnp.bool_),
        jnp.zeros((16, 5, 5, 39)),
    )
    np.testing.assert_array_equal(np.asarray(reset.sampling_group), np.zeros(16))


def test_heuristics_use_the_pinned_other_agent_plane_and_are_deterministic() -> None:
    partner_position = official_partner_channel_indexes(39)[0]
    observation = jnp.zeros((5, 5, 39), dtype=jnp.float32)
    courier = OfficialHeuristicPolicy(HEURISTIC_COURIER)
    helper = OfficialHeuristicPolicy(HEURISTIC_STATIONARY_HELPER)
    state = courier.init_hstate(1)

    # Two cells east of the centre -> deterministic east movement.
    east = observation.at[2, 4, partner_position].set(1.0)
    action_a, _ = courier.compute_action(east, False, state, jnp.asarray([0, 1]))
    action_b, _ = courier.compute_action(east, False, state, jnp.asarray([9, 9]))
    assert int(action_a) == int(action_b) == ACTION_EAST

    adjacent = observation.at[2, 3, partner_position].set(1.0)
    action, _ = courier.compute_action(adjacent, False, state, jnp.asarray([0, 1]))
    assert int(action) == ACTION_INTERACT
    invisible_action, _ = courier.compute_action(
        observation, False, state, jnp.asarray([0, 1])
    )
    assert int(invisible_action) == ACTION_NOOP
    helper_action, _ = helper.compute_action(east, False, state, jnp.asarray([0, 1]))
    assert int(helper_action) == ACTION_NOOP
