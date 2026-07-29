"""Thin integration with the locked official OvercookedV2 software.

Public model, trainer, checkpoint, environment, policy, and rollout interfaces
are used directly.  Two missing data surfaces are adapted locally: the training
loop retains terminal observations and delivery events around JaxMARL
``step_env``, and the evaluation loop retains per-step method diagnostics.  The
integration tests compare their observable transitions and return with the
public environment and evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from src.path_c.experiment import RunConfig


ACTION_ORDER = ("right", "down", "left", "up", "stay", "interact")


def _official_package_root() -> Path:
    spec = importlib.util.find_spec("overcooked_v2_experiments")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("overcooked_v2_experiments is not installed.")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _official_symbol(module: str, name: str) -> Any:
    """Import one public symbol while containing the official relative import."""

    package_root = _official_package_root()
    ppo_directory = str(package_root / "ppo")
    inserted = ppo_directory not in sys.path
    if inserted:
        sys.path.insert(0, ppo_directory)
    try:
        loaded = importlib.import_module(module)
    finally:
        if inserted:
            sys.path.remove(ppo_directory)
    return getattr(loaded, name)


def compose_official_config(
    config: RunConfig,
    *,
    algorithm: str,
    seed: int,
    output_directory: str | Path,
) -> Mapping[str, Any]:
    """Compose the official recurrent Independent PPO configuration."""

    if algorithm not in {"rnn-sp", "rnn-op"}:
        raise ValueError("algorithm must be rnn-sp or rnn-op.")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = [
        f"+experiment={algorithm}",
        "+env=test_time_simple",
        f"env.ENV_KWARGS.layout={config.environment.layout}",
        f"env.ENV_KWARGS.agent_view_size={config.environment.agent_view_size}",
        (
            "env.ENV_KWARGS.indicate_successful_delivery="
            f"{str(config.environment.indicate_successful_delivery).lower()}"
        ),
        f"SEED={int(seed)}",
        "NUM_SEEDS=1",
        f"NUM_CHECKPOINTS={len(config.upstream.checkpoint_progress)}",
        "VISUALIZE=false",
        "TUNE=false",
        "wandb.WANDB_MODE=disabled",
        f"model.TOTAL_TIMESTEPS={int(config.upstream.total_timesteps)}",
        (
            "model.REW_SHAPING_HORIZON="
            f"{int(config.upstream.reward_shaping_horizon)}"
        ),
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(_official_package_root() / "ppo" / "config"),
    ):
        composed = compose(config_name="base", overrides=overrides)
    resolved = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(resolved, Mapping):
        raise ValueError("Official Hydra configuration did not resolve to a mapping.")
    result = dict(resolved)
    result["RUN_BASE_DIR"] = str(Path(output_directory).resolve())
    _validate_official_config(result, config=config, algorithm=algorithm, seed=seed)
    return result


def _validate_official_config(
    resolved: Mapping[str, Any],
    *,
    config: RunConfig,
    algorithm: str,
    seed: int,
) -> None:
    model = resolved.get("model")
    environment = resolved.get("env")
    if not isinstance(model, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("Official Hydra config lacks model or environment sections.")
    expected = {
        "TYPE": "RNN",
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "TOTAL_TIMESTEPS": config.upstream.total_timesteps,
        "REW_SHAPING_HORIZON": config.upstream.reward_shaping_horizon,
        "NUM_STEPS": 256,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 64,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
    }
    for name, value in expected.items():
        if model.get(name) != value:
            raise ValueError(f"Official training config changed {name}: {model.get(name)!r}.")
    variant = (
        {"NUM_ENVS": 256, "ENT_COEF": 0.01}
        if algorithm == "rnn-sp"
        else {"NUM_ENVS": 64, "ENT_COEF": 0.02}
    )
    for name, value in variant.items():
        if model.get(name) != value:
            raise ValueError(f"Official {algorithm} config changed {name}.")
    kwargs = environment.get("ENV_KWARGS")
    if not isinstance(kwargs, Mapping):
        raise ValueError("Official environment kwargs are missing.")
    for name, value in {
        "layout": config.environment.layout,
        "agent_view_size": config.environment.agent_view_size,
        "negative_rewards": True,
        "random_agent_positions": True,
        "sample_recipe_on_delivery": True,
        "indicate_successful_delivery": config.environment.indicate_successful_delivery,
    }.items():
        if kwargs.get(name) != value:
            raise ValueError(f"Official environment config changed {name}.")
    if algorithm == "rnn-op" and list(kwargs.get("op_ingredient_permutations", ())) != [0, 1]:
        raise ValueError("Official Other-Play symmetry is not enabled.")
    if int(resolved.get("SEED", -1)) != int(seed) or int(resolved.get("NUM_SEEDS", -1)) != 1:
        raise ValueError("Official seed resolution differs from the requested run.")


def official_checkpoint_layout(config: Mapping[str, Any]) -> str:
    environment = config.get("env")
    if not isinstance(environment, Mapping):
        raise ValueError("Official checkpoint config lacks its environment section.")
    kwargs = environment.get("ENV_KWARGS")
    if not isinstance(kwargs, Mapping) or not kwargs.get("layout"):
        raise ValueError("Official checkpoint config lacks its layout.")
    return str(kwargs["layout"])


@dataclass(frozen=True, slots=True)
class OfficialNetwork:
    config: Mapping[str, Any]

    @property
    def layout(self) -> str:
        return official_checkpoint_layout(self.config)

    def network(self) -> Any:
        get_actor_critic = _official_symbol(
            "overcooked_v2_experiments.ppo.models.model",
            "get_actor_critic",
        )
        return get_actor_critic(self.config)

    def initial_carry(self, batch_size: int) -> Any:
        initialize_carry = _official_symbol(
            "overcooked_v2_experiments.ppo.models.model",
            "initialize_carry",
        )
        return initialize_carry(self.config, int(batch_size))

    def step(
        self,
        params: Mapping[str, Any],
        carry: Any,
        observations: Any,
        episode_start: Any,
    ) -> tuple[Any, Any, Any, Any]:
        import jax.numpy as jnp

        obs = jnp.asarray(observations)
        starts = jnp.asarray(episode_start, dtype=jnp.bool_)
        next_carry, distribution, value = self.network().apply(
            params,
            carry,
            (obs[None, ...], starts[None, ...]),
        )
        return next_carry, next_carry, distribution.logits[0], value[0]

    def sequence(
        self,
        params: Mapping[str, Any],
        carry: Any,
        observations: Any,
        episode_start: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Expose every recurrent feature through repeated public one-step calls."""

        import jax

        def one(current: Any, values: tuple[Any, Any]) -> tuple[Any, tuple[Any, ...]]:
            observation, start = values
            next_carry, feature, logits, value = self.step(
                params, current, observation, start
            )
            return next_carry, (feature, logits, value)

        final_carry, (features, logits, values) = jax.lax.scan(
            one, carry, (observations, episode_start)
        )
        return final_carry, features, logits, values


def initialize_official_parameters(
    network: OfficialNetwork,
    *,
    random_key: Any,
    observation_shape: Sequence[int],
    batch_size: int,
) -> Mapping[str, Any]:
    import jax.numpy as jnp

    observations = jnp.zeros(
        (1, int(batch_size), *tuple(observation_shape)), dtype=jnp.float32
    )
    episode_start = jnp.ones((1, int(batch_size)), dtype=jnp.bool_)
    return network.network().init(
        random_key,
        network.initial_carry(int(batch_size)),
        (observations, episode_start),
    )


def restore_official_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import orbax.checkpoint as ocp

    restored = ocp.PyTreeCheckpointer().restore(
        str(Path(checkpoint_path).resolve())
    )
    return restored["config"], restored["params"]


def store_official_checkpoint(
    *,
    config: Mapping[str, Any],
    params: Mapping[str, Any],
    run_number: int,
    update_step: int,
    final: bool,
) -> Path:
    store_checkpoint = _official_symbol(
        "overcooked_v2_experiments.ppo.utils.store", "store_checkpoint"
    )

    store_checkpoint(
        config,
        params,
        int(run_number),
        int(update_step),
        final=bool(final),
    )
    name = "ckpt_final" if final else f"ckpt_{int(update_step)}"
    return (
        Path(config["RUN_BASE_DIR"]).resolve()
        / f"run_{int(run_number)}"
        / name
    )


def train_upstream(
    official_config: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_progress: Sequence[float],
) -> Mapping[str, Any]:
    """Run the official trainer and save its fixed-progress parameter trees."""

    import jax
    import jax.numpy as jnp
    make_train = _official_symbol(
        "overcooked_v2_experiments.ppo.ippo", "make_train"
    )
    mini_batch_pmap = _official_symbol(
        "overcooked_v2_experiments.utils.utils", "mini_batch_pmap"
    )

    train = make_train(official_config)
    mapped_train = mini_batch_pmap(jax.jit(train), 1)
    output = mapped_train(
        jax.random.split(jax.random.PRNGKey(int(seed)), 1)
    )
    jax.block_until_ready(output["metrics"]["env_step"])

    metrics = jax.tree_util.tree_map(lambda value: value[0], output["metrics"])
    checkpoint_states = output["runner_state"][1]
    leaves = jax.tree_util.tree_leaves(checkpoint_states)
    checkpoint_count = int(leaves[0].shape[1])
    effective_environment_steps = int(output["metrics"]["env_step"][0, -1])
    steps_per_update = int(official_config["model"]["NUM_ENVS"]) * int(
        official_config["model"]["NUM_STEPS"]
    )
    if effective_environment_steps % steps_per_update:
        raise RuntimeError("Official environment-step count is not a whole update.")
    total_updates = effective_environment_steps // steps_per_update
    completed_episodes = int(
        round(float(jnp.sum(metrics["returned_episode"])) * steps_per_update)
    )
    if checkpoint_count != len(tuple(checkpoint_progress)):
        raise RuntimeError("Official trainer returned the wrong checkpoint count.")
    scheduled_updates = tuple(
        int(float(progress) * total_updates)
        for progress in checkpoint_progress
    )
    paths = []
    for index, update_step in enumerate(scheduled_updates):
        parameters = jax.tree_util.tree_map(
            lambda value, current=index: value[0, current],
            checkpoint_states,
        )
        paths.append(
            store_official_checkpoint(
                config=official_config,
                params=parameters,
                run_number=0,
                update_step=update_step,
                final=index == checkpoint_count - 1,
            )
        )
    return {
        "checkpoint_paths": tuple(paths),
        "metrics": metrics,
        "effective_environment_steps": effective_environment_steps,
        "completed_episodes": completed_episodes,
        "update_count": total_updates,
    }


def _stack_observations(observations: Mapping[str, Any]) -> Any:
    import jax.numpy as jnp

    return jnp.stack(
        (observations["agent_0"], observations["agent_1"]), axis=1
    )


def _select_done(done: Any, reset_value: Any, terminal_value: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def select(reset_leaf: Any, terminal_leaf: Any) -> Any:
        mask = jnp.asarray(done, dtype=jnp.bool_).reshape(
            jnp.asarray(done).shape
            + (1,) * (jnp.ndim(reset_leaf) - jnp.asarray(done).ndim)
        )
        return jnp.where(mask, reset_leaf, terminal_leaf)

    return jax.tree_util.tree_map(select, reset_value, terminal_value)


@dataclass(frozen=True, slots=True)
class VectorEnvironment:
    environment: Any
    num_envs: int
    episode_steps: int

    @classmethod
    def create(cls, config: RunConfig) -> "VectorEnvironment":
        import jaxmarl
        from jaxmarl.environments.overcooked_v2.overcooked import ObservationType

        environment = jaxmarl.make(
            "overcooked_v2",
            layout=config.environment.layout,
            max_steps=config.environment.episode_steps,
            observation_type=ObservationType.DEFAULT,
            agent_view_size=config.environment.agent_view_size,
            negative_rewards=True,
            random_agent_positions=True,
            sample_recipe_on_delivery=True,
            indicate_successful_delivery=(
                config.environment.indicate_successful_delivery
            ),
        )
        return cls(
            environment=environment,
            num_envs=config.environment.num_envs,
            episode_steps=config.environment.episode_steps,
        )

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.environment.observation_space().shape)

    def reset(self, key: Any) -> tuple[Any, Any]:
        import jax

        keys = jax.random.split(key, self.num_envs)
        return self.reset_with_keys(keys)

    def reset_with_keys(self, keys: Any) -> tuple[Any, Any]:
        import jax

        observations, state = jax.vmap(self.environment.reset)(keys)
        return state, _stack_observations(observations)

    def _events(self, state: Any, actions: Any) -> tuple[Any, Any, Any]:
        import jax
        import jax.numpy as jnp
        from jaxmarl.environments.overcooked_v2.common import (
            Actions,
            DynamicObject,
            StaticObject,
        )

        def one(
            current: Any, current_actions: Any
        ) -> tuple[Any, Any, Any]:
            forward = jax.vmap(lambda agent: agent.get_fwd_pos())(
                current.agents
            )
            cells = current.grid[forward.y, forward.x]
            interact = current_actions == int(Actions.interact)
            delivery_attempt = (
                interact
                & (cells[:, 0] == int(StaticObject.GOAL))
                & ((current.agents.inventory & int(DynamicObject.COOKED)) != 0)
            )
            plated_recipe = (
                current.recipe
                | int(DynamicObject.PLATE)
                | int(DynamicObject.COOKED)
            )
            correct = delivery_attempt & (
                current.agents.inventory == plated_recipe
            )
            wrong = delivery_attempt & ~correct
            indicator = (
                interact
                & (
                    cells[:, 0]
                    == int(StaticObject.BUTTON_RECIPE_INDICATOR)
                )
                & (current.agents.inventory == int(DynamicObject.EMPTY))
                & (cells[:, 1] == int(DynamicObject.EMPTY))
            )
            return (
                jnp.sum(correct),
                jnp.sum(wrong),
                jnp.sum(indicator),
            )

        return jax.vmap(one)(state, actions)

    def step(
        self, state: Any, joint_actions: Any, key: Any
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        import jax

        keys = jax.random.split(key, self.num_envs)
        return self.step_with_keys(state, joint_actions, keys)

    def step_with_keys(
        self, state: Any, joint_actions: Any, keys: Any
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        import jax
        import jax.numpy as jnp

        split = jax.vmap(lambda item: jax.random.split(item, 2))(
            keys
        )
        action_mapping = {
            "agent_0": joint_actions[:, 0],
            "agent_1": joint_actions[:, 1],
        }
        correct, wrong, indicator = self._events(state, joint_actions)
        terminal_observations, terminal_state, rewards, dones, environment_info = (
            jax.vmap(self.environment.step_env)(
                split[:, 0], state, action_mapping
            )
        )
        reset_observations, reset_state = jax.vmap(self.environment.reset)(
            split[:, 1]
        )
        done = dones["__all__"]
        observations = _select_done(
            done, reset_observations, terminal_observations
        )
        next_state = _select_done(done, reset_state, terminal_state)
        return (
            next_state,
            _stack_observations(observations),
            jnp.asarray(rewards["agent_0"], dtype=jnp.float32),
            jnp.asarray(done, dtype=jnp.bool_),
            {
                "terminal_observations": _stack_observations(
                    terminal_observations
                ),
                "correct_delivery": jnp.asarray(correct, dtype=jnp.int32),
                "wrong_delivery": jnp.asarray(wrong, dtype=jnp.int32),
                "indicator_activation": indicator,
                "environment": environment_info,
            },
        )


@dataclass(frozen=True, slots=True)
class FrozenPartnerPool:
    network: OfficialNetwork
    stacked_params: Any
    member_count: int

    @classmethod
    def from_checkpoints(
        cls, checkpoint_paths: Sequence[str | Path]
    ) -> "FrozenPartnerPool":
        import jax
        import jax.numpy as jnp

        restored = [
            restore_official_checkpoint(path) for path in checkpoint_paths
        ]
        if not restored:
            raise ValueError("The frozen partner pool cannot be empty.")
        configs = [item[0] for item in restored]
        params = [item[1] for item in restored]
        layouts = [official_checkpoint_layout(config) for config in configs]
        if len(set(layouts)) != 1:
            raise ValueError("Frozen partner checkpoints mix different layouts.")
        model_signatures = [
            (
                config.get("model", {}).get("TYPE"),
                config.get("model", {}).get("GRU_HIDDEN_DIM"),
                config.get("model", {}).get("FC_DIM_SIZE"),
            )
            for config in configs
        ]
        if len(set(model_signatures)) != 1:
            raise ValueError("Frozen partner checkpoints use different network shapes.")
        stacked = jax.tree_util.tree_map(
            lambda *values: jnp.stack(values), *params
        )
        return cls(
            network=OfficialNetwork(configs[0]),
            stacked_params=stacked,
            member_count=len(params),
        )

    def initial_carry(self, batch_size: int) -> Any:
        return self.network.initial_carry(batch_size)

    def step(
        self,
        member_indexes: Any,
        observations: Any,
        carry: Any,
        episode_start: Any,
        key: Any,
    ) -> tuple[Any, Any]:
        import jax

        keys = jax.random.split(key, observations.shape[0])
        return self.step_with_keys(
            member_indexes, observations, carry, episode_start, keys
        )

    def step_with_keys(
        self,
        member_indexes: Any,
        observations: Any,
        carry: Any,
        episode_start: Any,
        keys: Any,
    ) -> tuple[Any, Any]:
        """Advance each frozen partner with an explicitly paired action key."""

        import jax

        def one(
            member: Any,
            observation: Any,
            member_carry: Any,
            start: Any,
            action_key: Any,
        ) -> tuple[Any, Any]:
            params = jax.tree_util.tree_map(
                lambda values: values[member], self.stacked_params
            )
            next_carry, unused_feature, logits, unused_value = self.network.step(
                params,
                jax.tree_util.tree_map(
                    lambda value: value[None, ...], member_carry
                ),
                observation[None, ...],
                start[None, ...],
            )
            del unused_feature, unused_value
            action = jax.random.categorical(action_key, logits[0])
            return action, jax.tree_util.tree_map(
                lambda value: value[0], next_carry
            )

        return jax.vmap(one)(
            member_indexes, observations, carry, episode_start, keys
        )


def official_policy(params: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
    PPOPolicy = _official_symbol(
        "overcooked_v2_experiments.ppo.policy", "PPOPolicy"
    )
    return PPOPolicy(params, config, stochastic=True)


def official_rollout(
    *,
    left_policy: Any,
    right_policy: Any,
    environment: Any,
    key: Any,
) -> Any:
    PolicyPairing = _official_symbol(
        "overcooked_v2_experiments.eval.policy", "PolicyPairing"
    )
    get_rollout = _official_symbol(
        "overcooked_v2_experiments.eval.rollout", "get_rollout"
    )

    return get_rollout(
        PolicyPairing(left_policy, right_policy), environment, key
    )


def recorded_rollout(
    *,
    policies: Sequence[Any],
    environment: Any,
    key: Any,
    observe_step: Callable[[int, Any, Any, Any, Any], Mapping[str, Any]],
) -> tuple[Any, tuple[Mapping[str, Any], ...]]:
    """Mirror the official public policy loop while retaining every step row."""

    import jax

    if len(policies) != environment.num_agents:
        raise ValueError("One policy is required for each environment agent.")
    key, reset_key = jax.random.split(key)
    observations, state = environment.reset(reset_key)
    done = {
        f"agent_{index}": False for index in range(environment.num_agents)
    }
    done["__all__"] = False
    carries = {
        f"agent_{index}": policies[index].init_hstate(1)
        for index in range(environment.num_agents)
    }
    total_reward = 0.0
    rows = []
    for step, step_key in enumerate(
        jax.random.split(key, environment.max_steps)
    ):
        action_key, environment_key = jax.random.split(step_key)
        action_keys = jax.random.split(action_key, environment.num_agents)
        actions = {}
        next_carries = {}
        for index, policy in enumerate(policies):
            agent = f"agent_{index}"
            actions[agent], next_carries[agent] = policy.compute_action(
                observations[agent],
                done[agent],
                carries[agent],
                action_keys[index],
            )
        next_observations, next_state, reward, next_done, info = environment.step(
            environment_key, state, actions
        )
        rows.append(
            dict(
                observe_step(step, state, actions, reward, info)
            )
        )
        total_reward = total_reward + reward["agent_0"]
        observations, state, done, carries = (
            next_observations,
            next_state,
            next_done,
            next_carries,
        )
    return total_reward, tuple(rows)


__all__ = [
    "ACTION_ORDER",
    "OfficialNetwork",
    "FrozenPartnerPool",
    "VectorEnvironment",
    "compose_official_config",
    "initialize_official_parameters",
    "official_checkpoint_layout",
    "official_policy",
    "official_rollout",
    "recorded_rollout",
    "restore_official_checkpoint",
    "store_official_checkpoint",
    "train_upstream",
]
