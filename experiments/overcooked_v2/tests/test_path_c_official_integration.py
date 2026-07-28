from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("hydra")
pytest.importorskip("jaxmarl")
pytest.importorskip("orbax.checkpoint")
pytest.importorskip("overcooked_v2_experiments")

from experiments.overcooked_v2.official_adapter import (
    ACTION_ORDER,
    OfficialNetwork,
    VectorEnvironment,
    compose_official_config,
    initialize_official_parameters,
    official_policy,
    official_rollout,
    recorded_rollout,
    restore_official_checkpoint,
)
from experiments.overcooked_v2.deployment import Deployment
from experiments.overcooked_v2.standard_evaluation_app import pairing_batch
from src.path_c.evaluation import Pairing
from src.path_c.experiment import load_config
from src.path_c.method import empty_codebook, uniform_slot_log_belief
from src.path_c.model import build_model, initialize_heads


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/overcooked_v2/configs/path_c_simple.yaml"


def _small_config():
    config = load_config(CONFIG, run_kind="development")
    return replace(
        config,
        environment=replace(config.environment, num_envs=2),
        training=replace(config.training, environment_steps=800),
    )


def test_action_order_matches_jaxmarl_public_enum() -> None:
    from jaxmarl.environments.overcooked_v2.common import Actions

    observed = tuple(
        int(value)
        for value in (
            Actions.right,
            Actions.down,
            Actions.left,
            Actions.up,
            Actions.stay,
            Actions.interact,
        )
    )
    assert ACTION_ORDER == ("right", "down", "left", "up", "stay", "interact")
    assert observed == (0, 1, 2, 3, 4, 5)


def test_recurrent_adapter_matches_official_public_apply_with_nonzero_carry(
    tmp_path: Path,
) -> None:
    config = _small_config()
    official_config = compose_official_config(
        config,
        algorithm="rnn-sp",
        seed=11,
        output_directory=tmp_path,
    )
    environment = VectorEnvironment.create(config)
    network = OfficialNetwork(official_config)
    params = initialize_official_parameters(
        network,
        random_key=jax.random.PRNGKey(1),
        observation_shape=environment.observation_shape,
        batch_size=2,
    )
    state, observations = environment.reset(jax.random.PRNGKey(2))
    del state
    ego = observations[:, 0]
    carry = jax.tree_util.tree_map(
        lambda value: value + jnp.asarray(0.125, dtype=value.dtype),
        network.initial_carry(2),
    )
    starts = jnp.asarray([False, True])

    adapted_carry, features, adapted_logits, adapted_value = network.step(
        params, carry, ego, starts
    )
    expected_carry, distribution, expected_value = network.network().apply(
        params,
        carry,
        (ego[None, ...], starts[None, ...]),
    )
    for adapted, expected in zip(
        jax.tree_util.tree_leaves(adapted_carry),
        jax.tree_util.tree_leaves(expected_carry),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(adapted), np.asarray(expected))
    for feature, expected in zip(
        jax.tree_util.tree_leaves(features),
        jax.tree_util.tree_leaves(expected_carry),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(feature), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(adapted_logits), np.asarray(distribution.logits[0])
    )
    np.testing.assert_array_equal(
        np.asarray(adapted_value), np.asarray(expected_value[0])
    )


def test_vector_environment_preserves_terminal_observation_then_resets() -> None:
    config = _small_config()
    environment = VectorEnvironment.create(config)
    keys = jax.random.split(jax.random.PRNGKey(3), environment.num_envs)
    state, observations = environment.reset_with_keys(keys)
    actions = jnp.full((environment.num_envs, 2), 4, dtype=jnp.int32)
    last_info = None
    last_done = None
    for step in range(config.environment.episode_steps):
        step_keys = jax.vmap(lambda key: jax.random.fold_in(key, step + 1))(keys)
        state, observations, rewards, last_done, last_info = environment.step_with_keys(
            state, actions, step_keys
        )
        assert rewards.shape == (environment.num_envs,)
    assert last_info is not None and last_done is not None
    np.testing.assert_array_equal(np.asarray(last_done), [True, True])
    assert last_info["terminal_observations"].shape == observations.shape
    np.testing.assert_array_equal(np.asarray(state.time), [0, 0])


def test_delivery_records_follow_the_official_interaction_rules() -> None:
    from jaxmarl.environments.overcooked_v2.common import (
        Actions,
        Direction,
        DynamicObject,
        StaticObject,
    )

    config = _small_config()
    environment = VectorEnvironment.create(config)
    state, unused_observations = environment.reset(jax.random.PRNGKey(30))
    del unused_observations
    static = np.asarray(environment.environment.layout.static_objects)
    goal_y, goal_x = np.argwhere(static == int(StaticObject.GOAL))[0]
    candidates = (
        (goal_y + 1, goal_x, int(Direction.UP)),
        (goal_y - 1, goal_x, int(Direction.DOWN)),
        (goal_y, goal_x - 1, int(Direction.RIGHT)),
        (goal_y, goal_x + 1, int(Direction.LEFT)),
    )
    agent_y, agent_x, direction = next(
        (y, x, facing)
        for y, x, facing in candidates
        if 0 <= y < static.shape[0]
        and 0 <= x < static.shape[1]
        and static[y, x] == int(StaticObject.EMPTY)
    )
    partner_y, partner_x = next(
        (int(y), int(x))
        for y, x in np.argwhere(static == int(StaticObject.EMPTY))
        if (int(y), int(x)) != (agent_y, agent_x)
    )
    x_positions = jnp.full_like(state.agents.pos.x, partner_x).at[:, 0].set(
        agent_x
    )
    y_positions = jnp.full_like(state.agents.pos.y, partner_y).at[:, 0].set(
        agent_y
    )
    directions = jnp.full_like(state.agents.dir, int(Direction.UP)).at[:, 0].set(
        direction
    )
    plated = (
        state.recipe
        | int(DynamicObject.PLATE)
        | int(DynamicObject.COOKED)
    )
    wrong_plated = plated ^ int(DynamicObject.ingredient(0))
    inventories = jnp.zeros_like(state.agents.inventory)
    inventories = inventories.at[0, 0].set(plated[0])
    inventories = inventories.at[1, 0].set(wrong_plated[1])
    agents = state.agents.replace(
        pos=state.agents.pos.replace(x=x_positions, y=y_positions),
        dir=directions,
        inventory=inventories,
    )
    state = state.replace(agents=agents)
    actions = jnp.asarray(
        [
            [int(Actions.interact), int(Actions.stay)],
            [int(Actions.interact), int(Actions.stay)],
        ],
        dtype=jnp.int32,
    )
    keys = jax.random.split(jax.random.PRNGKey(31), 2)
    next_state, unused_obs, unused_reward, unused_done, info = (
        environment.step_with_keys(state, actions, keys)
    )
    del unused_obs, unused_reward, unused_done
    np.testing.assert_array_equal(
        np.asarray(info["correct_delivery"]), [1, 0]
    )
    np.testing.assert_array_equal(
        np.asarray(info["wrong_delivery"]), [0, 1]
    )

    stay = jnp.full((2, 2), int(Actions.stay), dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(32), 2)
    unused_state, unused_obs, unused_reward, unused_done, next_info = (
        environment.step_with_keys(next_state, stay, keys)
    )
    del unused_state, unused_obs, unused_reward, unused_done
    np.testing.assert_array_equal(
        np.asarray(next_info["correct_delivery"]), [0, 0]
    )
    np.testing.assert_array_equal(
        np.asarray(next_info["wrong_delivery"]), [0, 0]
    )


def test_official_orbax_checkpoint_loads_without_parameter_remapping(
    tmp_path: Path,
) -> None:
    import orbax.checkpoint as ocp

    config = {"model": {"TYPE": "RNN"}, "marker": "official"}
    params = {"params": {"Dense_0": {"kernel": np.arange(12).reshape(3, 4)}}}
    path = tmp_path / "checkpoint"
    ocp.PyTreeCheckpointer().save(str(path), {"config": config, "params": params})
    restored_config, restored_params = restore_official_checkpoint(path)
    assert restored_config == config
    np.testing.assert_array_equal(
        restored_params["params"]["Dense_0"]["kernel"],
        params["params"]["Dense_0"]["kernel"],
    )


def test_recorded_rollout_matches_official_public_rollout_return(
    tmp_path: Path,
) -> None:
    config = _small_config()
    official_config = compose_official_config(
        config,
        algorithm="rnn-sp",
        seed=12,
        output_directory=tmp_path,
    )
    environment = VectorEnvironment.create(config).environment
    network = OfficialNetwork(official_config)
    params = initialize_official_parameters(
        network,
        random_key=jax.random.PRNGKey(4),
        observation_shape=tuple(environment.observation_space().shape),
        batch_size=1,
    )
    left = official_policy(params, official_config)
    right = official_policy(params, official_config)
    key = jax.random.PRNGKey(5)
    official = official_rollout(
        left_policy=left,
        right_policy=right,
        environment=environment,
        key=key,
    )
    recorded_return, recorded_rows = recorded_rollout(
        policies=(left, right),
        environment=environment,
        key=key,
        observe_step=lambda step, state, actions, reward, info: {
            "step": step,
            "left_action": int(actions["agent_0"]),
            "right_action": int(actions["agent_1"]),
        },
    )
    np.testing.assert_array_equal(
        np.asarray(recorded_return), np.asarray(official.total_reward)
    )
    assert len(recorded_rows) == environment.max_steps



def test_two_episode_method_pairing_runs_through_the_real_environment(
    tmp_path: Path,
) -> None:
    config = _small_config()
    config = replace(
        config,
        environment=replace(config.environment, episode_steps=8),
        evaluation=replace(config.evaluation, episodes_per_pairing=2),
    )
    official_config = compose_official_config(
        config,
        algorithm="rnn-sp",
        seed=19,
        output_directory=tmp_path,
    )
    environment = VectorEnvironment.create(config)
    network = OfficialNetwork(official_config)
    official_params = initialize_official_parameters(
        network,
        random_key=jax.random.PRNGKey(40),
        observation_shape=environment.observation_shape,
        batch_size=2,
    )
    unused_state, observations = environment.reset(jax.random.PRNGKey(41))
    del unused_state
    starts = jnp.ones((2,), dtype=jnp.bool_)
    unused_carry, features, unused_logits, unused_value = network.step(
        official_params,
        network.initial_carry(2),
        observations[:, 0],
        starts,
    )
    del unused_carry, unused_logits, unused_value
    heads = build_model(
        hidden_dim=config.model.hidden_dim,
        slot_count=config.model.slot_count,
        action_count=config.model.action_count,
        response_count=config.model.response_count,
        prior_scale=config.model.prior_scale,
        action_embedding_dim=config.model.action_embedding_dim,
        log_standard_deviation_minimum=(
            config.model.log_standard_deviation_minimum
        ),
        log_standard_deviation_maximum=(
            config.model.log_standard_deviation_maximum
        ),
    )
    head_params = initialize_heads(
        heads,
        random_key=jax.random.PRNGKey(42),
        example_official_features=features,
        example_previous_actions=jnp.full(
            (2,), config.model.action_count, dtype=jnp.int32
        ),
        example_previous_team_rewards=jnp.zeros((2,), dtype=jnp.float32),
        example_episode_start=starts,
        example_slot_log_belief=uniform_slot_log_belief(
            (2,), config.model.slot_count
        ),
        example_observations=observations[:, 0][None, ...],
        hidden_dim=config.model.hidden_dim,
    )
    deployment = Deployment(
        outer_unit_id=0,
        network=network,
        reference_params=official_params,
        online_params=official_params,
        heads=heads,
        head_params=head_params,
        codebook=empty_codebook(
            code_count=config.model.response_count - 1,
            signature_dim=2 * config.model.action_count,
        ),
        log_temperature=jnp.asarray(0.0),
        generic_log_temperature=jnp.asarray(0.0),
    )
    rows, decisions = pairing_batch(
        config=config,
        left=deployment,
        right=deployment,
        pairing=Pairing("posterior_use", "sp", 0, 0),
        population_name="mechanical-test",
        evaluation_seed=43,
    )
    assert len(rows) == 2
    assert all(row.environment_steps == 8 for row in rows)
    assert len(list(decisions)) == 2 * 2 * 8
