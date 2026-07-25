"""Pure JaxMARL vector environment and official-observation safety filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


OFFICIAL_ENVIRONMENT_KWARGS = {
    "layout": "test_time_simple",
    "max_steps": 400,
    "observation_type": "DEFAULT",
    "agent_view_size": 2,
    "negative_rewards": True,
    "random_agent_positions": True,
    "sample_recipe_on_delivery": True,
    "indicate_successful_delivery": True,
    "force_path_planning": False,
    "random_reset": False,
}


def _stack_observations(observations: Mapping[str, Any]) -> Any:
    import jax.numpy as jnp

    return jnp.stack((observations["agent_0"], observations["agent_1"]), axis=1)


def _select_done(done: Any, reset_value: Any, terminal_value: Any) -> Any:
    import jax
    import jax.numpy as jnp

    done = jnp.asarray(done, dtype=jnp.bool_)

    def select(reset_leaf: Any, terminal_leaf: Any) -> Any:
        mask = done.reshape(done.shape + (1,) * (jnp.ndim(reset_leaf) - done.ndim))
        return jnp.where(mask, reset_leaf, terminal_leaf)

    return jax.tree_util.tree_map(select, reset_value, terminal_value)


@dataclass(frozen=True)
class OvercookedV2VectorEnvironment:
    """Vectorize the unmodified pure JaxMARL reset and step functions."""

    environment: Any
    num_envs: int
    observation_shape: tuple[int, ...]
    num_actions: int = 6
    episode_steps: int = 400

    @classmethod
    def create(
        cls, *, num_envs: int, layout: str = "test_time_simple"
    ) -> "OvercookedV2VectorEnvironment":
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if layout not in {"test_time_simple", "test_time_wide"}:
            raise ValueError("The OvercookedV2 Test Time layout is not registered.")
        from jaxmarl.environments.overcooked_v2.overcooked import (
            ObservationType,
            OvercookedV2,
        )

        environment_kwargs = dict(OFFICIAL_ENVIRONMENT_KWARGS)
        environment_kwargs["layout"] = layout
        environment_kwargs["observation_type"] = ObservationType.DEFAULT
        environment = OvercookedV2(**environment_kwargs)
        shape = tuple(int(value) for value in environment.observation_space().shape)
        if environment.num_agents != 2 or environment.max_steps != 400:
            raise ValueError("Path C requires the official two-agent 400-step environment.")
        return cls(environment=environment, num_envs=int(num_envs), observation_shape=shape)

    def reset(self, key: Any) -> tuple[Any, Any]:
        import jax

        keys = jax.random.split(key, self.num_envs)
        return self.reset_with_keys(keys)

    def reset_with_keys(self, keys: Any) -> tuple[Any, Any]:
        """Reset from one explicit random key per environment lane."""

        import jax

        if keys.shape[0] != self.num_envs:
            raise ValueError("reset_with_keys requires one key per environment lane.")
        observations, state = jax.vmap(self.environment.reset)(keys)
        return state, _stack_observations(observations)

    def _event_values(self, state: Any, joint_actions: Any) -> tuple[Any, Any]:
        import jax
        import jax.numpy as jnp
        from jaxmarl.environments.overcooked_v2.common import (
            Actions,
            DynamicObject,
            StaticObject,
        )

        def one(environment_state: Any, actions: Any) -> tuple[Any, Any]:
            forward = jax.vmap(lambda agent: agent.get_fwd_pos())(environment_state.agents)
            cells = environment_state.grid[forward.y, forward.x]
            interact = actions == int(Actions.interact)
            deliveries = (
                interact
                & (cells[:, 0] == int(StaticObject.GOAL))
                & ((environment_state.agents.inventory & int(DynamicObject.COOKED)) != 0)
            )
            indicator = (
                interact
                & (cells[:, 0] == int(StaticObject.BUTTON_RECIPE_INDICATOR))
                & (environment_state.agents.inventory == int(DynamicObject.EMPTY))
                & (cells[:, 1] == int(DynamicObject.EMPTY))
            )
            return jnp.sum(deliveries), jnp.sum(indicator)

        return jax.vmap(one)(state, joint_actions)

    def step(
        self, state: Any, joint_actions: Any, key: Any
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        """Step and reset completed lanes inside the same device graph."""

        import jax

        keys = jax.random.split(key, self.num_envs)
        return self.step_with_keys(state, joint_actions, keys)

    def step_with_keys(
        self, state: Any, joint_actions: Any, keys: Any
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        """Step from one explicit random stream per environment lane."""

        import jax
        import jax.numpy as jnp

        actions = jnp.asarray(joint_actions, dtype=jnp.int32)
        if actions.shape != (self.num_envs, 2):
            raise ValueError("joint_actions must have shape [num_envs, 2].")
        key_array = jnp.asarray(keys)
        if key_array.shape != (self.num_envs, 2):
            raise ValueError("step_with_keys requires one legacy JAX key per environment lane.")
        split_keys = jax.vmap(lambda value: jax.random.split(value, 2))(key_array)
        step_keys = split_keys[:, 0]
        reset_keys = split_keys[:, 1]
        action_mapping = {"agent_0": actions[:, 0], "agent_1": actions[:, 1]}
        delivery_attempts, indicator_activations = self._event_values(state, actions)
        terminal_observation, terminal_state, rewards, dones, raw_info = jax.vmap(
            self.environment.step_env
        )(step_keys, state, action_mapping)
        reset_observation, reset_state = jax.vmap(self.environment.reset)(reset_keys)
        done = jnp.asarray(dones["__all__"], dtype=jnp.bool_)
        observations = _select_done(done, reset_observation, terminal_observation)
        next_state = _select_done(done, reset_state, terminal_state)
        correct = jnp.asarray(terminal_state.new_correct_delivery, dtype=jnp.int32)
        wrong = jnp.maximum(
            jnp.asarray(delivery_attempts, dtype=jnp.int32) - correct,
            0,
        )
        info = {
            "correct_delivery": correct,
            "wrong_delivery": wrong,
            "indicator_cost": jnp.asarray(indicator_activations, dtype=jnp.float32),
            "terminal_observations": _stack_observations(terminal_observation),
            "environment_info": raw_info,
        }
        return (
            next_state,
            _stack_observations(observations),
            jnp.asarray(rewards["agent_0"], dtype=jnp.float32),
            done,
            info,
        )


def infer_default_observation_layout(
    observations: Any, *, indicate_successful_delivery: bool = True
) -> Mapping[str, int]:
    """Infer registered layer offsets from the official public channel count."""

    channels = int(observations.shape[-1])
    extra_delivery_layer = 1 if indicate_successful_delivery else 0
    remainder = channels - 26 - extra_delivery_layer
    if remainder < 0 or remainder % 4:
        raise ValueError("Observation channels do not match the official default encoding.")
    ingredients = remainder // 4
    return {
        "num_ingredients": ingredients,
        "other_agent_position_channel": ingredients + 7,
        "first_non_agent_channel": 2 * (ingredients + 7),
        "goal_channel": 2 * (ingredients + 7) + 1,
    }


def visible_goal_safe_action_mask(
    observations: Any,
    *,
    action_order: Sequence[str],
) -> Any:
    """Block only ``interact`` when the local observation faces a visible goal."""

    import jax.numpy as jnp

    values = jnp.asarray(observations)
    if values.ndim != 4:
        raise ValueError("Safety filtering expects [batch, height, width, channel].")
    order = tuple(str(action) for action in action_order)
    if order != ("right", "down", "left", "up", "stay", "interact"):
        raise ValueError("Safety filtering requires the official six primitive actions.")
    layout = infer_default_observation_layout(values)
    direction_mass = jnp.sum(values[..., 1:5], axis=(1, 2))
    direction = jnp.argmax(direction_mass, axis=-1)
    # JaxMARL v0.1.0 encodes Direction as up, down, right, and left.  This
    # differs from the primitive action order, so keep the observation-layer
    # lookup explicit.
    offsets_y = jnp.asarray((-1, 1, 0, 0), dtype=jnp.int32)
    offsets_x = jnp.asarray((0, 0, 1, -1), dtype=jnp.int32)
    center_y = values.shape[1] // 2
    center_x = values.shape[2] // 2
    target_y = center_y + offsets_y[direction]
    target_x = center_x + offsets_x[direction]
    batch = jnp.arange(values.shape[0])
    facing_goal = values[batch, target_y, target_x, layout["goal_channel"]] > 0
    safe = jnp.ones((values.shape[0], len(order)), dtype=jnp.bool_)
    return safe.at[:, order.index("interact")].set(~facing_goal)


def event_values(info: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return info["correct_delivery"], info["wrong_delivery"], info["indicator_cost"]


def _direction_for_delta(delta_x: int, delta_y: int) -> int:
    import numpy as np
    from jaxmarl.environments.overcooked_v2.common import DIR_TO_VEC

    items = DIR_TO_VEC.items() if hasattr(DIR_TO_VEC, "items") else enumerate(DIR_TO_VEC)
    for direction, vector in items:
        observed_x, observed_y = np.asarray(vector, dtype=np.int32).reshape(2)
        if (int(observed_x), int(observed_y)) == (int(delta_x), int(delta_y)):
            return int(direction)
    raise RuntimeError("The official direction table has no entry for the reference displacement.")


def _reference_state(environment: Any) -> tuple[Any, Mapping[str, int]]:
    """Build one startup-only state with a visible partner and faced goal."""

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxmarl.environments.overcooked_v2.common import DIR_TO_VEC, StaticObject

    unused_observation, state = environment.reset(jax.random.PRNGKey(43101))
    del unused_observation
    static_grid = np.asarray(state.grid[..., 0], dtype=np.int32)
    goals = sorted(tuple(map(int, row)) for row in np.argwhere(static_grid == int(StaticObject.GOAL)))
    empty = static_grid == int(StaticObject.EMPTY)
    view = int(environment.agent_view_size)
    selected: tuple[int, int, int, int, int, int] | None = None
    partner_y = partner_x = -1
    for goal_y, goal_x in goals:
        for delta_x, delta_y in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            ego_x = goal_x - delta_x
            ego_y = goal_y - delta_y
            if not (
                0 <= ego_y < empty.shape[0]
                and 0 <= ego_x < empty.shape[1]
                and bool(empty[ego_y, ego_x])
            ):
                continue
            partner_candidates = sorted(
                (int(y), int(x))
                for y, x in np.argwhere(empty)
                if (int(y), int(x)) != (ego_y, ego_x)
                and abs(int(y) - ego_y) <= view
                and abs(int(x) - ego_x) <= view
            )
            if partner_candidates:
                partner_y, partner_x = partner_candidates[0]
                selected = (goal_y, goal_x, ego_y, ego_x, delta_y, delta_x)
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("The registered layout has no reference goal with visible free cells.")
    goal_y, goal_x, ego_y, ego_x, delta_y, delta_x = selected
    facing_direction = _direction_for_delta(delta_x, delta_y)
    direction_items = DIR_TO_VEC.items() if hasattr(DIR_TO_VEC, "items") else enumerate(DIR_TO_VEC)
    direction_vectors = {
        int(direction): tuple(map(int, np.asarray(vector, dtype=np.int32).reshape(2)))
        for direction, vector in direction_items
    }
    away_direction = next(
        direction
        for direction, (away_x, away_y) in sorted(direction_vectors.items())
        if direction != facing_direction
        and not (
            0 <= ego_y + away_y < static_grid.shape[0]
            and 0 <= ego_x + away_x < static_grid.shape[1]
            and static_grid[ego_y + away_y, ego_x + away_x] == int(StaticObject.GOAL)
        )
    )
    position = state.agents.pos.replace(
        x=jnp.asarray((ego_x, partner_x), dtype=state.agents.pos.x.dtype),
        y=jnp.asarray((ego_y, partner_y), dtype=state.agents.pos.y.dtype),
    )
    directions = jnp.asarray(
        (facing_direction, away_direction), dtype=state.agents.dir.dtype
    )
    inventories = jnp.zeros_like(state.agents.inventory)
    agents = state.agents.replace(
        pos=position,
        dir=directions,
        inventory=inventories,
    )
    reference = state.replace(
        agents=agents,
        time=jnp.asarray(0, dtype=jnp.asarray(state.time).dtype),
        terminal=jnp.asarray(False, dtype=jnp.bool_),
        new_correct_delivery=jnp.asarray(False, dtype=jnp.bool_),
    )
    return reference, {
        "goal_y": goal_y,
        "goal_x": goal_x,
        "ego_y": ego_y,
        "ego_x": ego_x,
        "partner_y": partner_y,
        "partner_x": partner_x,
        "facing_direction": facing_direction,
        "away_direction": away_direction,
    }


def verify_environment_startup_contracts(
    environment: OvercookedV2VectorEnvironment,
    *,
    action_order: Sequence[str],
) -> Mapping[str, Any]:
    """Fail before training if official observations or delivery flags drift."""

    import hashlib

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject, StaticObject

    from .response_dock import registered_response_vocabulary, response_tokens

    raw_environment = environment.environment
    reference, coordinates = _reference_state(raw_environment)
    official_observations = raw_environment.get_obs(reference)
    ego_observation = np.asarray(official_observations["agent_0"])
    batched_observation = jnp.asarray(ego_observation[None, ...])
    layout = infer_default_observation_layout(batched_observation)
    center_y = ego_observation.shape[0] // 2
    center_x = ego_observation.shape[1] // 2

    expected_goal = np.zeros(ego_observation.shape[:2], dtype=np.bool_)
    static_grid = np.asarray(reference.grid[..., 0], dtype=np.int32)
    for local_y in range(expected_goal.shape[0]):
        for local_x in range(expected_goal.shape[1]):
            global_y = coordinates["ego_y"] + local_y - center_y
            global_x = coordinates["ego_x"] + local_x - center_x
            if 0 <= global_y < static_grid.shape[0] and 0 <= global_x < static_grid.shape[1]:
                expected_goal[local_y, local_x] = (
                    static_grid[global_y, global_x] == int(StaticObject.GOAL)
                )
    observed_goal = ego_observation[..., layout["goal_channel"]] > 0
    if not np.array_equal(observed_goal, expected_goal):
        raise RuntimeError("The inferred goal channel differs from the official reference state.")

    expected_partner = np.zeros(ego_observation.shape[:2], dtype=np.bool_)
    partner_local_y = coordinates["partner_y"] - coordinates["ego_y"] + center_y
    partner_local_x = coordinates["partner_x"] - coordinates["ego_x"] + center_x
    expected_partner[partner_local_y, partner_local_x] = True
    observed_partner = ego_observation[..., layout["other_agent_position_channel"]] > 0
    if not np.array_equal(observed_partner, expected_partner):
        raise RuntimeError(
            "The inferred partner-position channel differs from the official reference state."
        )

    expected_direction = np.zeros((*ego_observation.shape[:2], 4), dtype=np.bool_)
    expected_direction[
        center_y,
        center_x,
        coordinates["facing_direction"],
    ] = True
    observed_direction = ego_observation[..., 1:5] > 0
    if not np.array_equal(observed_direction, expected_direction):
        raise RuntimeError("The official direction channels differ from the reference state.")
    safe = np.asarray(
        visible_goal_safe_action_mask(batched_observation, action_order=action_order)
    )
    interact_index = tuple(action_order).index("interact")
    if bool(safe[0, interact_index]):
        raise RuntimeError("The reference goal-facing interaction was not filtered.")

    rotated_directions = reference.agents.dir.at[0].set(coordinates["away_direction"])
    rotated = reference.replace(agents=reference.agents.replace(dir=rotated_directions))
    rotated_observation = raw_environment.get_obs(rotated)["agent_0"]
    rotated_safe = np.asarray(
        visible_goal_safe_action_mask(
            jnp.asarray(rotated_observation[None, ...]), action_order=action_order
        )
    )
    if not bool(rotated_safe[0, interact_index]):
        raise RuntimeError("Rotating away from the goal did not restore interaction safety.")

    visible_token = int(
        np.asarray(
            response_tokens(
                batched_observation,
                batched_observation,
                jnp.asarray((False,)),
                {},
            )
        )[0]
    )
    vocabulary = registered_response_vocabulary()
    expected_visible_token = vocabulary.token_ids["response_visible__latency_le_1"]
    if visible_token != expected_visible_token:
        raise RuntimeError("The partner-position channel produced the wrong response token.")

    correct_inventory = (
        reference.recipe | int(DynamicObject.COOKED) | int(DynamicObject.PLATE)
    )
    delivery_inventory = reference.agents.inventory.at[0].set(correct_inventory)
    delivery_state = reference.replace(
        agents=reference.agents.replace(inventory=delivery_inventory)
    )
    delivery_actions = {
        "agent_0": jnp.asarray(int(Actions.interact), dtype=jnp.uint32),
        "agent_1": jnp.asarray(int(Actions.stay), dtype=jnp.uint32),
    }
    unused_obs, delivered_state, unused_reward, unused_done, unused_info = (
        raw_environment.step_env(
            jax.random.PRNGKey(43102), delivery_state, delivery_actions
        )
    )
    del unused_obs, unused_reward, unused_done, unused_info
    if int(np.asarray(delivered_state.new_correct_delivery)) != 1:
        raise RuntimeError("A known correct delivery did not set the per-step success flag.")
    stay_actions = {
        "agent_0": jnp.asarray(int(Actions.stay), dtype=jnp.uint32),
        "agent_1": jnp.asarray(int(Actions.stay), dtype=jnp.uint32),
    }
    unused_obs, following_state, unused_reward, unused_done, unused_info = (
        raw_environment.step_env(
            jax.random.PRNGKey(43103), delivered_state, stay_actions
        )
    )
    del unused_obs, unused_reward, unused_done, unused_info
    if int(np.asarray(following_state.new_correct_delivery)) != 0:
        raise RuntimeError("The correct-delivery flag is cumulative rather than per-step.")

    terminal_state = delivery_state.replace(
        time=jnp.asarray(
            environment.episode_steps - 1,
            dtype=jnp.asarray(delivery_state.time).dtype,
        )
    )

    def repeat_lane(value: Any) -> Any:
        array = jnp.asarray(value)
        return jnp.broadcast_to(array, (environment.num_envs, *array.shape))

    batched_state = jax.tree_util.tree_map(repeat_lane, terminal_state)
    joint_actions = jnp.broadcast_to(
        jnp.asarray((int(Actions.interact), int(Actions.stay)), dtype=jnp.int32),
        (environment.num_envs, 2),
    )
    terminal_keys = jax.random.split(jax.random.PRNGKey(43104), environment.num_envs)
    delivery_attempts, unused_indicator = environment._event_values(
        batched_state, joint_actions
    )
    del unused_indicator
    unused_state, unused_observations, unused_rewards, done, info = environment.step_with_keys(
        batched_state, joint_actions, terminal_keys
    )
    del unused_state, unused_observations, unused_rewards
    correct = np.asarray(info["correct_delivery"], dtype=np.int32)
    wrong = np.asarray(info["wrong_delivery"], dtype=np.int32)
    if not np.asarray(done, dtype=np.bool_).all():
        raise RuntimeError("The reference terminal step did not finish every vector lane.")
    if not np.array_equal(correct, np.ones_like(correct)):
        raise RuntimeError("Autoreset discarded a terminal correct-delivery flag.")
    attempts = np.asarray(delivery_attempts, dtype=np.int32)
    if np.any(correct > attempts):
        raise RuntimeError("Correct deliveries exceed observable delivery attempts.")
    if not np.array_equal(wrong, np.zeros_like(wrong)) or np.any(wrong < 0):
        raise RuntimeError("Terminal delivery counters violate their non-negative contract.")

    return {
        "observation_layout": {
            "passed": True,
            "goal_channel": int(layout["goal_channel"]),
            "other_agent_position_channel": int(layout["other_agent_position_channel"]),
            "facing_direction": int(coordinates["facing_direction"]),
            "reference_observation_sha256": hashlib.sha256(
                np.ascontiguousarray(ego_observation).tobytes()
            ).hexdigest(),
            "visible_response_token": visible_token,
        },
        "delivery_counters": {
            "passed": True,
            "per_step_success_flag": True,
            "terminal_autoreset_preserves_success": True,
            "wrong_delivery_floor": 0,
            "checked_vector_lanes": int(environment.num_envs),
        },
    }
