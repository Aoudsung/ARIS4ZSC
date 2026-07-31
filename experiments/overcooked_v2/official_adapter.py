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
import ast
import copy
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlparse

from src.path_c.experiment import (
    OFFICIAL_CORRECT_DELIVERY_REWARD,
    OFFICIAL_SOURCE_COMMIT,
    RunConfig,
)
from src.path_c.experiment import (
    OFFICIAL_NUM_MINIBATCHES,
    OFFICIAL_OP_NUM_ENVS,
    OFFICIAL_OP_TOTAL_TIMESTEPS,
    OFFICIAL_ROLLOUT_LENGTH,
    OFFICIAL_SP_NUM_ENVS,
    OFFICIAL_SP_TOTAL_TIMESTEPS,
    OFFICIAL_TRAINING_ROOT_SEED,
    OFFICIAL_TRAINING_RUN_COUNT,
    OFFICIAL_UPDATE_EPOCHS,
)


ACTION_ORDER = ("right", "down", "left", "up", "stay", "interact")


def _distribution_source(distribution_name: str) -> Mapping[str, Any]:
    """Read PEP-610 provenance for one installed Official distribution."""

    distribution = importlib.metadata.distribution(distribution_name)
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise RuntimeError(
            f"{distribution_name} has no direct_url.json; its fixed Git source "
            "cannot be established."
        )
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{distribution_name} direct_url.json is malformed.")
    vcs = payload.get("vcs_info")
    source_kind = "locked-git-install"
    if isinstance(vcs, Mapping):
        commit = str(vcs.get("commit_id", ""))
        requested = str(vcs.get("requested_revision", ""))
        if commit != OFFICIAL_SOURCE_COMMIT or requested != OFFICIAL_SOURCE_COMMIT:
            raise RuntimeError(
                f"{distribution_name} source differs from Official commit "
                f"{OFFICIAL_SOURCE_COMMIT}: commit={commit!r}, requested={requested!r}."
            )
    else:
        # An editable install from a local checkout is acceptable only when the
        # checkout itself is an exact, clean copy of the registered commit.
        # This path supports mechanical tests without weakening formal source
        # provenance: both HEAD and worktree cleanliness are verified here.
        parsed = urlparse(str(payload.get("url", "")))
        if parsed.scheme != "file":
            raise RuntimeError(
                f"{distribution_name} was not installed from locked Git or an "
                "auditable local Git checkout."
            )
        source = Path(unquote(parsed.path)).resolve()
        repository = next(
            (candidate for candidate in (source, *source.parents) if (candidate / ".git").exists()),
            None,
        )
        if repository is None:
            raise RuntimeError(f"{distribution_name} local source is not inside Git.")
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != OFFICIAL_SOURCE_COMMIT or dirty:
            raise RuntimeError(
                f"{distribution_name} editable checkout is not the clean Official "
                f"commit {OFFICIAL_SOURCE_COMMIT}: commit={commit!r}, dirty={bool(dirty)}."
            )
        requested = OFFICIAL_SOURCE_COMMIT
        source_kind = "clean-local-git-checkout"
        # Some editable installers do not activate their generated finder on
        # every supported Python version.  After provenance is verified, make
        # the exact checkout importable without copying or modifying it.
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "url": str(payload.get("url", "")),
        "commit_id": commit,
        "requested_revision": requested,
        "source_kind": source_kind,
    }


def validate_official_runtime() -> Mapping[str, Any]:
    """Require both Official packages to originate from the same fixed commit."""

    experiment_source = _distribution_source("overcooked-v2-experiments")
    jaxmarl_source = _distribution_source("jaxmarl")
    overcooked_ai = importlib.metadata.distribution("overcooked-ai")
    if overcooked_ai.version != "1.1.0":
        raise RuntimeError(
            "The fixed Official PPO entrypoint requires overcooked-ai==1.1.0; "
            f"observed {overcooked_ai.version}."
        )
    spec = importlib.util.find_spec("jaxmarl")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("The fixed JaxMARL source is not importable.")
    settings_path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "environments"
        / "overcooked_v2"
        / "settings.py"
    )
    delivery_reward = None
    tree = ast.parse(settings_path.read_text(encoding="utf-8"), filename=str(settings_path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "DELIVERY_REWARD" for target in node.targets)
        ):
            delivery_reward = float(ast.literal_eval(node.value))
            break
    if delivery_reward != OFFICIAL_CORRECT_DELIVERY_REWARD:
        raise RuntimeError("Official correct-delivery reward scale changed.")
    experiment_spec = importlib.util.find_spec("overcooked_v2_experiments")
    if experiment_spec is None or experiment_spec.submodule_search_locations is None:
        raise ModuleNotFoundError("The fixed Official Experiment source is not importable.")
    config_root = (
        Path(next(iter(experiment_spec.submodule_search_locations))) / "ppo" / "config"
    )
    if not config_root.is_dir():
        raise RuntimeError(
            "The fixed Official wheel omits its Hydra configuration tree. Install "
            "JaxMARL and experiments editably from one clean checkout of commit "
            f"{OFFICIAL_SOURCE_COMMIT}."
        )
    return {
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "correct_delivery_reward": delivery_reward,
        "experiments": experiment_source,
        "jaxmarl": jaxmarl_source,
        "overcooked_ai_version": overcooked_ai.version,
    }


def _official_package_root() -> Path:
    spec = importlib.util.find_spec("overcooked_v2_experiments")
    if spec is None:
        _distribution_source("overcooked-v2-experiments")
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
    seed_index: int,
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
        f"SEED={OFFICIAL_TRAINING_ROOT_SEED}",
        f"NUM_SEEDS={OFFICIAL_TRAINING_RUN_COUNT}",
        f"NUM_CHECKPOINTS={len(config.upstream.checkpoint_progress)}",
        "VISUALIZE=false",
        "TUNE=false",
        "wandb.WANDB_MODE=disabled",
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
    if config.run_kind == "mechanical":
        # Mechanical E2E runs exercise the unmodified Official trainer with a
        # deliberately tiny budget.  This branch is never reachable from a
        # formal config and does not alter the registered scientific recipe.
        if (
            int(config.upstream.total_timesteps)
            != int(config.training.environment_steps)
        ):
            raise ValueError(
                "Mechanical upstream and DELTA trajectory budgets must match; "
                "use the dedicated mechanical E2E config."
            )
        model = dict(result["model"])
        model.update(
            {
                "TOTAL_TIMESTEPS": int(config.upstream.total_timesteps),
                "REW_SHAPING_HORIZON": int(
                    config.upstream.reward_shaping_horizon
                ),
                "NUM_ENVS": int(config.environment.num_envs),
                "NUM_STEPS": int(config.training.rollout_length),
                "UPDATE_EPOCHS": int(config.ppo.update_epochs),
                "NUM_MINIBATCHES": int(
                    config.training.minibatches_per_epoch
                ),
            }
        )
        result["model"] = model
    result["RUN_BASE_DIR"] = str(Path(output_directory).resolve())
    _validate_official_config(
        result, config=config, algorithm=algorithm, seed_index=seed_index
    )
    return result


OFFICIAL_BASELINE_EXPERIMENTS: Mapping[str, str] = {
    "sp": "rnn-sp",
    "state-augmented": "rnn-sa",
    "op": "rnn-op",
    "fcp": "rnn-fcp",
}


def compose_official_baseline_config(
    *,
    layout: str,
    method: str,
    fcp_population: str | Path | None = None,
) -> Mapping[str, Any]:
    """Resolve an unmodified Official baseline Hydra recipe.

    This function validates, but does not normalize, algorithm-specific
    hyperparameters.  In particular OP remains 50M/64 environments and FCP
    retains its own learning rate, entropy coefficient, and GAE lambda.
    """

    if method not in OFFICIAL_BASELINE_EXPERIMENTS:
        raise ValueError(f"Unknown Official baseline method: {method}")
    if layout not in {"test_time_simple", "test_time_wide"}:
        raise ValueError(f"Unknown Official layout: {layout}")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = [
        f"+experiment={OFFICIAL_BASELINE_EXPERIMENTS[method]}",
        f"+env={layout}",
        f"SEED={OFFICIAL_TRAINING_ROOT_SEED}",
        "NUM_CHECKPOINTS=1",
        "VISUALIZE=false",
        "TUNE=false",
        "wandb.WANDB_MODE=disabled",
    ]
    if method == "fcp":
        if fcp_population is None:
            raise ValueError("Official FCP requires an explicit population directory.")
        overrides.append(f"+FCP={Path(fcp_population).resolve()}")
    elif fcp_population is not None:
        raise ValueError("fcp_population is only valid for Official FCP.")
    if method != "fcp":
        overrides.append(f"NUM_SEEDS={OFFICIAL_TRAINING_RUN_COUNT}")
    with initialize_config_dir(
        version_base=None,
        config_dir=str(_official_package_root() / "ppo" / "config"),
    ):
        composed = compose(config_name="base", overrides=overrides)
    payload = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Official baseline config did not resolve to a mapping.")
    _validate_official_baseline_config(payload, layout=layout, method=method)
    return dict(payload)


def compose_ippo_large_config(
    *, layout: str, hidden_dimension: int
) -> Mapping[str, Any]:
    """Capacity-control recipe using the fixed Official RNN-SP trainer.

    Only the two width fields are changed.  Environment, optimizer, trajectory
    budget, action-path recurrent depth, and evaluator remain the registered
    Official SP recipe.
    """

    base = compose_official_baseline_config(layout=layout, method="sp")
    dimension = int(hidden_dimension)
    if dimension <= 0:
        raise ValueError("IPPO-Large hidden_dimension must be positive.")
    payload = copy.deepcopy(base)
    payload["model"]["FC_DIM_SIZE"] = dimension
    payload["model"]["GRU_HIDDEN_DIM"] = dimension
    return payload


def _validate_official_baseline_config(
    payload: Mapping[str, Any], *, layout: str, method: str
) -> None:
    model = payload.get("model")
    environment = payload.get("env")
    if not isinstance(model, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("Official baseline config lacks model/environment sections.")
    kwargs = environment.get("ENV_KWARGS")
    if not isinstance(kwargs, Mapping):
        raise ValueError("Official baseline environment kwargs are missing.")
    environment_expected = {
        "layout": layout,
        "agent_view_size": 2,
        "negative_rewards": True,
        "random_agent_positions": True,
        "sample_recipe_on_delivery": True,
        "indicate_successful_delivery": True,
    }
    for name, expected in environment_expected.items():
        if kwargs.get(name) != expected:
            raise ValueError(
                f"Official {method} environment field {name} changed: "
                f"{kwargs.get(name)!r}."
            )
    shared = {
        "TYPE": "RNN",
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "NUM_STEPS": 256,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "GAMMA": 0.99,
    }
    for name, expected in shared.items():
        if model.get(name) != expected:
            raise ValueError(f"Official {method} model field {name} changed.")
    expected_by_method: Mapping[str, Mapping[str, Any]] = {
        "sp": {
            "TOTAL_TIMESTEPS": 30_000_000,
            "NUM_ENVS": 256,
            "LR": 0.00025,
            "ENT_COEF": 0.01,
            "GAE_LAMBDA": 0.95,
        },
        "state-augmented": {
            "TOTAL_TIMESTEPS": 30_000_000,
            "NUM_ENVS": 256,
            "LR": 0.00025,
            "ENT_COEF": 0.01,
            "GAE_LAMBDA": 0.95,
        },
        "op": {
            "TOTAL_TIMESTEPS": 50_000_000,
            "NUM_ENVS": 64,
            "LR": 0.00025,
            "ENT_COEF": 0.02,
            "GAE_LAMBDA": 0.95,
        },
        "fcp": {
            "TOTAL_TIMESTEPS": 30_000_000,
            "NUM_ENVS": 256,
            "LR": 0.0007,
            "ENT_COEF": 0.04,
            "GAE_LAMBDA": 0.9,
        },
    }
    for name, expected in expected_by_method[method].items():
        if model.get(name) != expected:
            raise ValueError(
                f"Official {method} field {name} changed: {model.get(name)!r}."
            )
    if method == "state-augmented" and int(payload.get("NUM_ITERATIONS", -1)) != 10:
        raise ValueError("Official State-Augmented must use ten iterations.")
    if method == "op" and list(kwargs.get("op_ingredient_permutations", ())) != [0, 1]:
        raise ValueError("Official Other-Play ingredient symmetry changed.")
    if method == "fcp" and int(payload.get("NUM_SEEDS", -1)) != 1:
        raise ValueError(
            "Official FCP retains NUM_SEEDS=1; its ten runs are population-indexed."
        )
    if method != "fcp" and int(payload.get("NUM_SEEDS", -1)) != 10:
        raise ValueError(f"Official {method} must train ten runs.")


def _validate_official_config(
    resolved: Mapping[str, Any],
    *,
    config: RunConfig,
    algorithm: str,
    seed_index: int,
) -> None:
    model = resolved.get("model")
    environment = resolved.get("env")
    if not isinstance(model, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("Official Hydra config lacks model or environment sections.")
    mechanical = config.run_kind == "mechanical"
    total_timesteps = (
        int(config.upstream.total_timesteps)
        if mechanical
        else (
            OFFICIAL_SP_TOTAL_TIMESTEPS
            if algorithm == "rnn-sp"
            else OFFICIAL_OP_TOTAL_TIMESTEPS
        )
    )
    reward_horizon = (
        int(config.upstream.reward_shaping_horizon)
        if mechanical
        else total_timesteps // 2
    )
    expected = {
        "TYPE": "RNN",
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "TOTAL_TIMESTEPS": total_timesteps,
        "REW_SHAPING_HORIZON": reward_horizon,
        "NUM_STEPS": (
            config.training.rollout_length
            if mechanical
            else OFFICIAL_ROLLOUT_LENGTH
        ),
        "UPDATE_EPOCHS": (
            config.ppo.update_epochs if mechanical else OFFICIAL_UPDATE_EPOCHS
        ),
        "NUM_MINIBATCHES": (
            config.training.minibatches_per_epoch
            if mechanical
            else OFFICIAL_NUM_MINIBATCHES
        ),
        "LR": 0.00025,
        "LR_WARMUP": 0.05,
        "ANNEAL_LR": True,
        "MAX_GRAD_NORM": 0.25,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
    }
    for name, value in expected.items():
        if model.get(name) != value:
            raise ValueError(f"Official training config changed {name}: {model.get(name)!r}.")
    variant = {
        "NUM_ENVS": (
            config.environment.num_envs
            if mechanical
            else (
                OFFICIAL_SP_NUM_ENVS
                if algorithm == "rnn-sp"
                else OFFICIAL_OP_NUM_ENVS
            )
        ),
        "ENT_COEF": 0.01 if algorithm == "rnn-sp" else 0.02,
    }
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
    if int(resolved.get("SEED", -1)) != OFFICIAL_TRAINING_ROOT_SEED:
        raise ValueError("Official training root seed must be 42.")
    if int(resolved.get("NUM_SEEDS", -1)) != OFFICIAL_TRAINING_RUN_COUNT:
        raise ValueError("Official training population must contain ten keys.")
    if not 0 <= int(seed_index) < OFFICIAL_TRAINING_RUN_COUNT:
        raise ValueError("Official seed_index must lie in 0..9.")
    steps_per_update = int(model["NUM_ENVS"]) * int(model["NUM_STEPS"])
    if mechanical and (
        total_timesteps < steps_per_update
        or total_timesteps % steps_per_update
    ):
        raise ValueError(
            "Mechanical upstream budget must contain whole vectorized updates."
        )


def official_checkpoint_layout(config: Mapping[str, Any]) -> str:
    environment = config.get("env")
    if not isinstance(environment, Mapping):
        raise ValueError("Official checkpoint config lacks its environment section.")
    kwargs = environment.get("ENV_KWARGS")
    if not isinstance(kwargs, Mapping) or not kwargs.get("layout"):
        raise ValueError("Official checkpoint config lacks its layout.")
    return str(kwargs["layout"])


def validate_official_partner_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: RunConfig,
    algorithm: str,
    seed_index: int,
) -> None:
    """Bind one DELTA support checkpoint to its claimed Official recipe."""

    checkpoint_config, unused_params = restore_official_checkpoint(checkpoint_path)
    del unused_params
    _validate_official_config(
        checkpoint_config,
        config=config,
        algorithm=algorithm,
        seed_index=int(seed_index),
    )


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

    # Keep the persisted resolved configuration JSON-serializable while
    # restoring the pathlib contract used by the fixed Official implementation.
    official_store_config = dict(config)
    official_store_config["RUN_BASE_DIR"] = Path(
        str(config["RUN_BASE_DIR"])
    ).resolve()
    store_checkpoint(
        official_store_config,
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
    run_key: Any,
    seed_index: int,
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
    import wandb

    train = make_train(official_config)
    mapped_train = mini_batch_pmap(jax.jit(train), 1)
    model_name = str(official_config["model"]["TYPE"])
    layout_name = str(official_config["env"]["ENV_KWARGS"]["layout"])
    agent_view_size = official_config["env"]["ENV_KWARGS"].get(
        "agent_view_size"
    )
    run_name = (
        f"ippo_{model_name}_ov2_{layout_name}_avs-{agent_view_size}"
    )
    # The fixed Official ``single_run_with_viz`` entrypoint always surrounds
    # ``single_run`` with this context.  ``ippo.make_train`` contains an
    # unconditional ``jax.debug.callback(wandb.log, ...)``; calling it without
    # the Official context fails even when WANDB_MODE is disabled.
    with wandb.init(
        entity=official_config["wandb"]["ENTITY"],
        project=official_config["wandb"]["PROJECT"],
        tags=["IPPO", model_name, "OvercookedV2"],
        config=dict(official_config),
        mode=official_config["wandb"]["WANDB_MODE"],
        name=run_name,
    ):
        output = mapped_train(jnp.asarray(run_key, dtype=jnp.uint32)[None, :])
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
                run_number=int(seed_index),
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
        raw_by_agent = jnp.stack(
            (rewards["agent_0"], rewards["agent_1"]), axis=-1
        ).astype(jnp.float32)
        shaped = environment_info.get("shaped_reward")
        if not isinstance(shaped, Mapping):
            raise RuntimeError(
                "Locked OvercookedV2 environment did not return shaped_reward."
            )
        official_shaped_by_agent = jnp.stack(
            (shaped["agent_0"], shaped["agent_1"]), axis=-1
        ).astype(jnp.float32)
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
                "raw_rewards_by_agent": raw_by_agent,
                "official_shaped_rewards_by_agent": official_shaped_by_agent,
                "environment": environment_info,
            },
        )


@dataclass(frozen=True, slots=True)
class FrozenPartnerPool:
    network: OfficialNetwork
    stacked_params: Any
    member_count: int
    parent_members: Any
    parent_count: int
    checkpoints_per_parent: int

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: Sequence[str | Path],
        *,
        parent_training_run_ids: Sequence[str] | None = None,
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
        if parent_training_run_ids is None:
            parents = tuple(f"member-{index}" for index in range(len(params)))
        else:
            parents = tuple(str(value) for value in parent_training_run_ids)
            if len(parents) != len(params):
                raise ValueError("Frozen partner checkpoints and parent IDs do not align.")
        parent_order = tuple(dict.fromkeys(parents))
        member_groups = tuple(
            tuple(index for index, parent in enumerate(parents) if parent == current)
            for current in parent_order
        )
        group_sizes = {len(group) for group in member_groups}
        if len(group_sizes) != 1:
            raise ValueError(
                "Every frozen parent run must contribute the same checkpoint count."
            )
        checkpoints_per_parent = next(iter(group_sizes))
        return cls(
            network=OfficialNetwork(configs[0]),
            stacked_params=stacked,
            member_count=len(params),
            parent_members=jnp.asarray(member_groups, dtype=jnp.int32),
            parent_count=len(member_groups),
            checkpoints_per_parent=checkpoints_per_parent,
        )

    def sample_members(self, key: Any, batch_size: int) -> Any:
        """Sample parent uniformly, then checkpoint uniformly within parent."""

        import jax

        parent_key, checkpoint_key = jax.random.split(key)
        parents = jax.random.randint(
            parent_key, (int(batch_size),), 0, self.parent_count
        )
        checkpoints = jax.random.randint(
            checkpoint_key,
            (int(batch_size),),
            0,
            self.checkpoints_per_parent,
        )
        return self.parent_members[parents, checkpoints]

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


def official_pairing_rollouts(
    *,
    left_policy: Any,
    right_policy: Any,
    environment: Any,
    root_key: Any,
    episodes: int,
) -> tuple[Any, Any]:
    """Vectorize the fixed Official ``get_rollout`` without changing its loop."""

    import jax

    PolicyPairing = _official_symbol(
        "overcooked_v2_experiments.eval.policy", "PolicyPairing"
    )
    get_rollout = _official_symbol(
        "overcooked_v2_experiments.eval.rollout", "get_rollout"
    )
    episode_keys = jax.random.split(root_key, int(episodes))
    pairing = PolicyPairing(left_policy, right_policy)
    rollouts = jax.vmap(lambda key: get_rollout(pairing, environment, key))(
        episode_keys
    )
    return rollouts, episode_keys


def official_delivery_counts(
    *, environment: Any, rollouts: Any, episode_keys: Any
) -> tuple[Any, Any]:
    """Read delivery events post hoc from exact Official rollout states/actions.

    This function never calls a policy and cannot change a trajectory.  The
    initial state is reconstructed from the reset half of each exact episode
    key; later pre-transition states are the preceding Official ``state_seq``.
    """

    import jax
    import jax.numpy as jnp
    from jaxmarl.environments.overcooked_v2.common import (
        Actions,
        DynamicObject,
        StaticObject,
    )

    reset_keys = jax.vmap(lambda key: jax.random.split(key, 2)[1])(episode_keys)
    unused_observations, initial_states = jax.vmap(environment.reset)(reset_keys)
    del unused_observations
    pre_states = jax.tree_util.tree_map(
        lambda initial, sequence: jnp.concatenate(
            (initial[:, None, ...], sequence[:, :-1, ...]), axis=1
        ),
        initial_states,
        rollouts.state_seq,
    )
    actions = jnp.stack(
        (rollouts.actions_seq["agent_0"], rollouts.actions_seq["agent_1"]),
        axis=-1,
    )

    def one(state: Any, current_actions: Any) -> tuple[Any, Any]:
        forward = jax.vmap(lambda agent: agent.get_fwd_pos())(state.agents)
        cells = state.grid[forward.y, forward.x]
        interact = current_actions == int(Actions.interact)
        delivery = (
            interact
            & (cells[:, 0] == int(StaticObject.GOAL))
            & ((state.agents.inventory & int(DynamicObject.COOKED)) != 0)
        )
        plated_recipe = (
            state.recipe | int(DynamicObject.PLATE) | int(DynamicObject.COOKED)
        )
        correct = delivery & (state.agents.inventory == plated_recipe)
        wrong = delivery & ~correct
        return jnp.sum(correct, dtype=jnp.int32), jnp.sum(wrong, dtype=jnp.int32)

    correct, wrong = jax.vmap(jax.vmap(one))(pre_states, actions)
    return jnp.sum(correct, axis=1), jnp.sum(wrong, axis=1)


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
    "compose_official_baseline_config",
    "compose_ippo_large_config",
    "initialize_official_parameters",
    "official_checkpoint_layout",
    "official_policy",
    "official_pairing_rollouts",
    "official_delivery_counts",
    "official_rollout",
    "recorded_rollout",
    "restore_official_checkpoint",
    "store_official_checkpoint",
    "train_upstream",
    "validate_official_partner_checkpoint",
    "validate_official_runtime",
]
