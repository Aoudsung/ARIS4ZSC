from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_return_to_go_is_reverse_undiscounted_sum() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.losses import return_to_go

    rewards = jnp.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    expected = jnp.asarray([[6.0, 60.0], [5.0, 50.0], [3.0, 30.0]])
    np.testing.assert_allclose(np.asarray(return_to_go(rewards)), np.asarray(expected))


def test_joint_advantage_normalization_uses_one_masked_mean_and_deviation() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.losses import joint_normalize_advantages

    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    mean = (1.0 + 2.0 + 3.0) / 3.0
    deviation = np.sqrt(((1.0 - mean) ** 2 + (2.0 - mean) ** 2 + (3.0 - mean) ** 2) / 3.0)
    expected = (values - mean) / (deviation + 1.0e-8)
    actual = joint_normalize_advantages(jnp.asarray(values), jnp.asarray(mask))
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2.0e-6)


def _test_config() -> SimpleNamespace:
    from src.cetr_zsc.config import ModelConfig, PPOConfig

    return SimpleNamespace(
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


def _make_model_and_batch():
    import jax
    import jax.numpy as jnp

    from src.cetr_zsc.model import CetrModel
    from src.cetr_zsc.types import EpisodeBatch

    time, lanes, sp_lanes = 6, 4, 2
    config = _test_config()
    model = CetrModel(config, (5, 5, 26), 6)
    params = model.init_parameters(jax.random.PRNGKey(31))
    keys = jax.random.split(jax.random.PRNGKey(32), 3)
    observations = jax.random.normal(keys[0], (time, lanes, 5, 5, 26))
    sp_observations = jax.random.normal(keys[1], (time, sp_lanes, 5, 5, 26))
    actions = jax.random.randint(keys[2], (time, lanes), 0, 6)
    sp_actions = jnp.mod(actions[:, :sp_lanes] + 1, 6)
    starts = jnp.zeros((time, lanes), dtype=jnp.bool_).at[0].set(True)
    rewards = jnp.arange(time * lanes, dtype=jnp.float32).reshape(time, lanes) / 10.0
    dones = jnp.zeros((time, lanes), dtype=jnp.bool_).at[-1].set(True)
    episode_return = jnp.sum(rewards, axis=0)
    batch = EpisodeBatch(
        observations=observations,
        actions=actions,
        old_log_probabilities=jnp.zeros((time, lanes), dtype=jnp.float32),
        old_values=jnp.zeros((time, lanes), dtype=jnp.float32),
        rewards=rewards,
        dones=dones,
        episode_starts=starts,
        sp_other_observations=sp_observations,
        sp_other_actions=sp_actions,
        sp_other_old_log_probabilities=jnp.zeros(
            (time, sp_lanes), dtype=jnp.float32
        ),
        sp_other_old_values=jnp.zeros((time, sp_lanes), dtype=jnp.float32),
        lane_stream=jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        lane_parent=jnp.asarray([-1, -1, 0, 1], dtype=jnp.int32),
        lane_fold=jnp.asarray([0, 0, 0, 1], dtype=jnp.int32),
        lane_weight=jnp.asarray([1.0, 1.0, 0.5, 0.25], dtype=jnp.float32),
        episode_return=episode_return,
        ego_roles=jnp.zeros((lanes,), dtype=jnp.int32),
        member_index=jnp.arange(lanes, dtype=jnp.int32),
    )
    return config, model, params, batch


def test_cetr_ppo_loss_is_finite_with_real_model() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.losses import cetr_ppo_loss

    _, model, params, batch = _make_model_and_batch()
    result = cetr_ppo_loss(model, params, batch, jnp.asarray(0.3))
    assert bool(jnp.isfinite(result.total))
    assert all(bool(jnp.isfinite(value)) for value in result.metrics.values())


def test_zero_external_weights_zero_external_actor_term() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.losses import cetr_ppo_loss

    _, model, params, batch = _make_model_and_batch()
    zero_weight_batch = batch._replace(lane_weight=jnp.zeros_like(batch.lane_weight))
    result = cetr_ppo_loss(model, params, zero_weight_batch, jnp.asarray(0.3))
    np.testing.assert_allclose(
        np.asarray(result.metrics["actor_external"]),
        0.0,
        atol=1.0e-6,
    )


def test_zero_dual_multiplier_removes_self_play_actor_from_total() -> None:
    import jax.numpy as jnp

    from src.cetr_zsc.losses import cetr_ppo_loss

    config, model, params, batch = _make_model_and_batch()
    result = cetr_ppo_loss(model, params, batch, jnp.asarray(0.0))
    expected = (
        result.metrics["actor_external"]
        + config.ppo.value_weight * result.metrics["value_loss"]
        - config.ppo.entropy_weight * result.metrics["entropy"]
    )
    np.testing.assert_allclose(np.asarray(result.total), np.asarray(expected), atol=2.0e-6)
