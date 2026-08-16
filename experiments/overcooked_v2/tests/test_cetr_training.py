from __future__ import annotations

from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np


def _batch_for_weights():
    import jax.numpy as jnp

    from src.cetr_zsc.types import EpisodeBatch

    lanes = 8
    self_lanes = lanes // 2
    lane_stream = jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
    lane_parent = jnp.asarray([-1, -1, -1, -1, 0, 1, 0, 1], dtype=jnp.int32)
    lane_fold = jnp.asarray([-1, -1, -1, -1, 0, 0, 1, 1], dtype=jnp.int32)
    returns = jnp.asarray([0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0])
    return EpisodeBatch(
        observations=jnp.zeros((1, lanes, 1), dtype=jnp.float32),
        actions=jnp.zeros((1, lanes), dtype=jnp.int32),
        old_log_probabilities=jnp.zeros((1, lanes), dtype=jnp.float32),
        old_values=jnp.zeros((1, lanes), dtype=jnp.float32),
        rewards=jnp.zeros((1, lanes), dtype=jnp.float32),
        dones=jnp.ones((1, lanes), dtype=jnp.bool_),
        episode_starts=jnp.ones((1, lanes), dtype=jnp.bool_),
        sp_other_observations=jnp.zeros((1, self_lanes, 1), dtype=jnp.float32),
        sp_other_actions=jnp.zeros((1, self_lanes), dtype=jnp.int32),
        sp_other_old_log_probabilities=jnp.zeros(
            (1, self_lanes), dtype=jnp.float32
        ),
        sp_other_old_values=jnp.zeros((1, self_lanes), dtype=jnp.float32),
        lane_stream=lane_stream,
        lane_parent=lane_parent,
        lane_fold=lane_fold,
        lane_weight=jnp.ones((lanes,), dtype=jnp.float32),
        episode_return=returns,
        ego_roles=jnp.zeros((lanes,), dtype=jnp.int32),
        member_index=jnp.arange(lanes, dtype=jnp.int32),
    )


def test_compute_tail_weights_matches_two_fold_hand_calculation() -> None:
    from src.cetr_zsc.training import compute_tail_weights

    weights, metrics = compute_tail_weights(
        _batch_for_weights(),
        parent_nominal_weights=(0.5, 0.5),
        parent_count=2,
    )
    np.testing.assert_allclose(
        np.asarray(weights), [1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0]
    )
    np.testing.assert_allclose(float(metrics["tail_objective"]), 20.0)
    assert float(metrics["fold_0_observed_parent_count"]) == 2.0
    assert float(metrics["fold_1_observed_parent_count"]) == 2.0
    assert float(metrics["tail_weight_max"]) == 2.0


def test_ppo_learning_rate_has_registered_warmup_and_cosine_endpoints() -> None:
    from src.cetr_zsc.config import PPOConfig
    from src.cetr_zsc.training import ppo_learning_rate

    config = SimpleNamespace(
        ppo=PPOConfig(
            update_epochs=1,
            learning_rate=1.0,
            gradient_clip_norm=1.0,
            clip_epsilon=0.2,
            value_clip_epsilon=0.2,
            entropy_weight=0.01,
            value_weight=0.5,
            lr_warmup_fraction=0.2,
            anneal_learning_rate=True,
            adam_epsilon=1.0e-5,
        )
    )
    import math

    # The registered schedule (warmup then cosine) reaches zero only at
    # progress 1, i.e. one step past the last real optimizer step; the final
    # real step sees the cosine value at (total-1-warmup)/(total-warmup).
    warmup = int(0.2 * 10)
    expected_final = 0.5 * 1.0 * (1.0 + math.cos(math.pi * (9 - warmup) / (10 - warmup)))
    assert float(ppo_learning_rate(config=config, optimizer_step=0, total_optimizer_steps=10)) == 0.5
    np.testing.assert_allclose(
        float(ppo_learning_rate(config=config, optimizer_step=9, total_optimizer_steps=10)),
        expected_final,
        rtol=1e-6,
    )


def test_minibatch_schedule_pairs_and_covers_each_stream() -> None:
    import jax

    from src.cetr_zsc.training import environment_minibatch_schedule

    schedule = np.asarray(
        environment_minibatch_schedule(
            jax.random.PRNGKey(7),
            environment_count=8,
            minibatches_per_epoch=2,
            update_epochs=3,
        )
    )
    assert schedule.shape == (3, 2, 4)
    for epoch in schedule:
        np.testing.assert_array_equal(np.sort(epoch[:, :2].reshape(-1)), np.arange(4))
        np.testing.assert_array_equal(
            np.sort(epoch[:, 2:].reshape(-1)), np.arange(4, 8)
        )


class _MechanicalEnvironment:
    num_envs = 4
    episode_steps = 4

    def reset_with_keys(self, keys: Any):
        import jax.numpy as jnp

        count = int(keys.shape[0])
        return (
            jnp.zeros((count,), dtype=jnp.int32),
            jnp.zeros((count, 2, 5, 5, 26), dtype=jnp.float32),
        )

    def step_training_fast_with_keys(self, state: Any, joint_actions: Any, keys: Any):
        import jax.numpy as jnp

        del joint_actions, keys
        terminal_state = state + 1
        done = terminal_state >= self.episode_steps
        next_state = jnp.where(done, 0, terminal_state)
        observations = jnp.zeros(
            (self.num_envs, 2, 5, 5, 26), dtype=jnp.float32
        )
        raw_by_agent = jnp.zeros((self.num_envs, 2), dtype=jnp.float32)
        return (
            next_state,
            observations,
            jnp.zeros((self.num_envs,), dtype=jnp.float32),
            done,
            {"raw_rewards_by_agent": raw_by_agent},
        )


class _PartnerState(NamedTuple):
    carry: Any
    member: Any
    parent: Any
    fold: Any
    stage_slot: Any


def _partner_functions():
    import jax.numpy as jnp

    from src.cetr_zsc.runner import PartnerFunctions

    def initial_state(batch_size: int, key: Any) -> _PartnerState:
        del key
        count = int(batch_size)
        return _PartnerState(
            carry=jnp.zeros((count, 8), dtype=jnp.float32),
            member=jnp.zeros((count,), dtype=jnp.int32),
            parent=jnp.zeros((count,), dtype=jnp.int32),
            fold=jnp.tile(jnp.asarray((0, 1), dtype=jnp.int32), count // 2),
            stage_slot=jnp.zeros((count,), dtype=jnp.int32),
        )

    def step(
        parameters: Any,
        state: _PartnerState,
        observations: Any,
        episode_start: Any,
        keys: Any,
    ):
        del parameters, episode_start, keys
        action = jnp.zeros((observations.shape[0],), dtype=jnp.int32)
        return action, state, jnp.asarray(0, dtype=jnp.int32), jnp.zeros_like(action)

    def observe(
        parameters: Any,
        state: _PartnerState,
        context: Any,
        observations: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        next_observations: Any,
    ) -> _PartnerState:
        del parameters, context, observations, actions, rewards, dones, next_observations
        return state

    def run_id(parameters: Any, state: _PartnerState, context: Any):
        del parameters, context
        return state.member

    def diagnostics(parameters: Any, state: _PartnerState, context: Any):
        del parameters, context
        return {"member": state.member, "parent_count": jnp.asarray(1, jnp.int32)}

    return PartnerFunctions(initial_state, step, observe, run_id, diagnostics)


def _model_and_batch():
    import jax
    import jax.numpy as jnp

    from src.cetr_zsc.config import ModelConfig, PPOConfig
    from src.cetr_zsc.model import CetrModel
    from src.cetr_zsc.types import EpisodeBatch

    config = SimpleNamespace(
        model=ModelConfig(task_hidden_dim=128, task_embedding_dim=128),
        ppo=PPOConfig(
            update_epochs=1,
            learning_rate=1.0e-3,
            gradient_clip_norm=1.0,
            clip_epsilon=0.2,
            value_clip_epsilon=0.2,
            entropy_weight=0.01,
            value_weight=0.5,
            lr_warmup_fraction=0.0,
            anneal_learning_rate=False,
            adam_epsilon=1.0e-5,
        ),
    )
    model = CetrModel(config, (5, 5, 26), 6)
    params = model.init_parameters(jax.random.PRNGKey(31))
    time, lanes, self_lanes = 4, 4, 2
    observations = jax.random.normal(
        jax.random.PRNGKey(32), (time, lanes, 5, 5, 26)
    )
    self_observations = observations[:, :self_lanes]
    actions = jax.random.randint(jax.random.PRNGKey(33), (time, lanes), 0, 6)
    self_actions = actions[:, :self_lanes]
    batch = EpisodeBatch(
        observations=observations,
        actions=actions,
        old_log_probabilities=jnp.zeros((time, lanes), dtype=jnp.float32),
        old_values=jnp.zeros((time, lanes), dtype=jnp.float32),
        rewards=jnp.zeros((time, lanes), dtype=jnp.float32),
        dones=jnp.zeros((time, lanes), dtype=jnp.bool_).at[-1].set(True),
        episode_starts=jnp.zeros((time, lanes), dtype=jnp.bool_).at[0].set(True),
        sp_other_observations=self_observations,
        sp_other_actions=self_actions,
        sp_other_old_log_probabilities=jnp.zeros(
            (time, self_lanes), dtype=jnp.float32
        ),
        sp_other_old_values=jnp.zeros((time, self_lanes), dtype=jnp.float32),
        lane_stream=jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        lane_parent=jnp.asarray([-1, -1, 0, 1], dtype=jnp.int32),
        lane_fold=jnp.asarray([-1, -1, 0, 1], dtype=jnp.int32),
        lane_weight=jnp.ones((lanes,), dtype=jnp.float32),
        episode_return=jnp.zeros((lanes,), dtype=jnp.float32),
        ego_roles=jnp.zeros((lanes,), dtype=jnp.int32),
        member_index=jnp.arange(lanes, dtype=jnp.int32),
    )
    return model, params, batch


def test_mechanical_update_changes_parameters_and_dual_moves_up_below_target() -> None:
    import jax
    import jax.numpy as jnp

    from src.cetr_zsc.optimizer import init_adam
    from src.cetr_zsc.risk import update_sp_dual
    from src.cetr_zsc.runner import collect_episodes
    from src.cetr_zsc.training import (
        compute_tail_weights,
        environment_minibatch_schedule,
        make_training_update_kernel,
    )

    model, params, unused_batch = _model_and_batch()
    del unused_batch
    _, batch, collection_metrics = collect_episodes(
        environment=_MechanicalEnvironment(),
        model=model,
        params=params,
        partner_functions=_partner_functions(),
        random_key=jax.random.PRNGKey(34),
    )
    assert float(collection_metrics["final_done_fraction"]) == 1.0
    lane_weight, tail_metrics = compute_tail_weights(
        batch, parent_nominal_weights=(1.0,), parent_count=1
    )
    batch = batch._replace(lane_weight=lane_weight)
    assert float(tail_metrics["tail_objective"]) == 0.0
    kernel = make_training_update_kernel(model=model, total_optimizer_steps=1)
    schedule = environment_minibatch_schedule(
        jax.random.PRNGKey(35),
        environment_count=4,
        minibatches_per_epoch=1,
        update_epochs=1,
    )
    updated_params, _, metrics = kernel(
        params,
        init_adam(params),
        batch,
        schedule,
        jnp.asarray(0.3, dtype=jnp.float32),
    )
    before = jax.tree_util.tree_leaves(params)
    after = jax.tree_util.tree_leaves(updated_params)
    assert any(not np.array_equal(np.asarray(left), np.asarray(right)) for left, right in zip(before, after))
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in metrics.values())
    assert float(metrics["ppo_update_applied"]) == 1.0

    increased = update_sp_dual(0.0, 1.0, 0.0, 0.1)
    assert float(increased) > 0.0
