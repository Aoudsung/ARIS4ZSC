from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _config() -> SimpleNamespace:
    from src.cetr_zsc.config import EnvironmentConfig

    return SimpleNamespace(
        environment=EnvironmentConfig(
            layout="test_time_simple",
            agent_view_size=2,
            indicate_successful_delivery=True,
            negative_rewards=True,
            random_agent_positions=True,
            sample_recipe_on_delivery=True,
            episode_steps=400,
            num_envs=4,
        )
    )


def _pool():
    from src.cetr_zsc.partners import PartnerPool, PartnerPoolMember

    members = []
    parent_members = []
    parent_ids = []
    parent_mechanisms = []
    for parent_index, mechanism in enumerate(("fcp", "op", "sa", "sp")):
        parent_ids.append(f"{mechanism}-parent")
        parent_mechanisms.append(mechanism)
        indexes = []
        for stage_slot, stage in enumerate((0.0, 0.5, 1.0)):
            indexes.append(len(members))
            members.append(
                PartnerPoolMember(
                    run_id=f"{mechanism}-{stage}",
                    parent_training_run_id=f"{mechanism}-parent",
                    mechanism=mechanism,
                    checkpoint_stage=stage,
                    checkpoint=Path(f"checkpoint-{mechanism}-{stage}"),
                    parent_index=parent_index,
                    stage_slot=stage_slot,
                    probability=1.0 / 12.0,
                )
            )
        parent_members.append(tuple(indexes))
        
    return PartnerPool(
        members=tuple(members),
        parent_ids=tuple(parent_ids),
        parent_mechanisms=tuple(parent_mechanisms),
        parent_nominal_weights=(0.25,) * 4,
        parent_members=tuple(parent_members),
    )


class _ToyModel:
    def initial_carry(self, batch_size: int):
        import jax.numpy as jnp

        return jnp.zeros((int(batch_size), 8), dtype=jnp.float32)

    def step(self, params, carry, observation, episode_start):
        import jax.numpy as jnp

        del params, observation, episode_start
        batch_size = int(carry.shape[0])
        return (
            carry,
            jnp.zeros((batch_size, 6), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.float32),
        )


class _ToyOfficialPool:
    member_count = 12

    def initial_carry(self, batch_size: int):
        import jax.numpy as jnp

        return jnp.zeros((int(batch_size), 8), dtype=jnp.float32)

    def step_with_keys(self, member, observations, carry, episode_start, keys):
        import jax.numpy as jnp

        del member, episode_start, keys
        return jnp.zeros((observations.shape[0],), dtype=jnp.int32), carry


class _PrematureEnvironment:
    num_envs = 4
    episode_steps = 3

    def reset_with_keys(self, keys):
        import jax.numpy as jnp

        count = int(keys.shape[0])
        return (
            jnp.zeros((count,), dtype=jnp.int32),
            jnp.zeros((count, 2, 5, 5, 26), dtype=jnp.float32),
        )

    def step_training_fast_with_keys(self, state, joint_actions, keys):
        import jax.numpy as jnp

        del joint_actions, keys
        done = jnp.ones((self.num_envs,), dtype=jnp.bool_)
        return (
            state,
            jnp.zeros((self.num_envs, 2, 5, 5, 26), dtype=jnp.float32),
            jnp.zeros((self.num_envs,), dtype=jnp.float32),
            done,
            {"raw_rewards_by_agent": jnp.zeros((self.num_envs, 2), dtype=jnp.float32)},
        )


def test_vector_environment_collects_one_complete_self_and_external_episode() -> None:
    import jax

    from experiments.overcooked_v2.official_adapter import VectorEnvironment
    from src.cetr_zsc.runner import collect_episodes, make_static_partner_functions

    environment = VectorEnvironment.create(_config())
    partner_functions = make_static_partner_functions(
        pool_metadata=_pool(), official_pool=_ToyOfficialPool()
    )
    _, batch, metrics = collect_episodes(
        environment=environment,
        model=_ToyModel(),
        params=None,
        partner_functions=partner_functions,
        random_key=jax.random.PRNGKey(23),
        update_index=0,
    )

    assert batch.observations.shape[0] == 400
    assert batch.actions.shape[:2] == (400, 4)
    assert batch.sp_other_observations.shape[0] == 400
    assert batch.sp_other_observations.shape[1] == 2
    assert batch.old_values.shape == (400, 4)
    assert batch.sp_other_old_values.shape == (400, 2)
    np.testing.assert_array_equal(np.asarray(batch.lane_stream), [0, 0, 1, 1])
    np.testing.assert_array_equal(np.asarray(batch.lane_parent)[:2], [-1, -1])
    np.testing.assert_array_equal(np.asarray(batch.member_index)[:2], [-1, -1])
    assert float(metrics["final_done_fraction"]) == 1.0
    assert np.all(np.isfinite(np.asarray(batch.episode_return)))
    assert np.all(np.isfinite(np.asarray(batch.old_log_probabilities)))


def test_collection_metrics_expose_premature_done_signal() -> None:
    import jax

    from src.cetr_zsc.runner import collect_episodes, make_static_partner_functions

    partner_functions = make_static_partner_functions(
        pool_metadata=_pool(), official_pool=_ToyOfficialPool()
    )
    _, _, metrics = collect_episodes(
        environment=_PrematureEnvironment(),
        model=_ToyModel(),
        params=None,
        partner_functions=partner_functions,
        random_key=jax.random.PRNGKey(24),
        update_index=0,
    )
    assert float(metrics["premature_done_count"]) > 0.0


def test_external_lane_assignment_covers_fold_role_and_rotates_stage() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.runner import external_lane_assignment

    parent_members = jnp.asarray(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=jnp.int32
    )
    member, parent, fold, role = external_lane_assignment(
        16, 4, jnp.asarray(0, dtype=jnp.int32), parent_members
    )
    del member
    for parent_index in range(4):
        indexes = np.flatnonzero(np.asarray(parent) == parent_index)
        np.testing.assert_array_equal(
            np.stack((np.asarray(fold)[indexes], np.asarray(role)[indexes]), axis=1),
            np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]),
        )

    first, unused_parent, unused_fold, unused_role = external_lane_assignment(
        16, 4, jnp.asarray(0, dtype=jnp.int32), parent_members
    )
    second, _, _, _ = external_lane_assignment(
        16, 4, jnp.asarray(1, dtype=jnp.int32), parent_members
    )
    del unused_parent, unused_fold, unused_role
    np.testing.assert_array_equal(
        np.asarray(first).reshape(4, 4) % 3,
        np.asarray([[0] * 4, [1] * 4, [2] * 4, [0] * 4]),
    )
    np.testing.assert_array_equal(
        np.asarray(second).reshape(4, 4) % 3,
        np.asarray([[1] * 4, [2] * 4, [0] * 4, [1] * 4]),
    )
