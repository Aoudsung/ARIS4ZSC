"""远端官方 OvercookedV2 训练器与 R015 证据接口的适配层。

本文件不复制或修改官方源码。它从可编辑安装定位
``overcooked_v2_experiments`` 和 JaxMARL，使用官方 Hydra 配置、训练核、
网络定义、Orbax 保存函数和环境；本仓只补上明确的启动、产物与证据边界。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.overcooked_v2.path_c_flax_policy import OfficialFlaxPolicy
from experiments.overcooked_v2.path_c_official_artifact import (
    OFFICIAL_ACTION_RULE_ID,
    OFFICIAL_ARTIFACT_SCHEMA_VERSION,
    OFFICIAL_EFFECTIVE_STEP_CONTRACTS,
    OFFICIAL_NOMINAL_TOTAL_TIMESTEPS,
    OFFICIAL_OP_FAMILY_ID,
    OFFICIAL_REFERENCE_ACCEPTANCE_RULE_ID,
    OFFICIAL_REFERENCE_ID,
    OFFICIAL_RUN_DESCRIPTOR_SCHEMA_VERSION,
    OFFICIAL_SP_FAMILY_ID,
    build_official_artifact_manifest,
    checkpoint_artifact_sha256,
    flax_weights_sha256,
    load_flax_parameter_tree,
    load_official_launch_config,
    locate_editable_package,
    validate_official_artifact_manifest,
)
from experiments.overcooked_v2.path_c_official_evidence import (
    OfficialR015RolloutBackend,
)
from experiments.overcooked_v2.path_c_pool_admission import R015PartnerSupportSpec
from experiments.overcooked_v2.path_c_seed import derive_ocv2_execution_seed


OFFICIAL_ADAPTER_SCHEMA_VERSION = "path_c_official_experiments_adapter_v1"
OFFICIAL_CHECKPOINT_FORMAT = "orbax_pytree_directory_v1"
OFFICIAL_PARAMETER_TREE_PATH = ("params",)
OFFICIAL_ACTION_ORDER = ("right", "down", "left", "up", "stay", "interact")
OFFICIAL_HYDRA_ENTRYPOINT = "overcooked_v2_experiments/ppo/main.py"
OFFICIAL_NATIVE_SAVE_MECHANISM = (
    "overcooked_v2_experiments.ppo.utils.store.store_checkpoint"
)
OFFICIAL_NETWORK_DEFINITION = (
    "overcooked_v2_experiments.ppo.models.rnn.ActorCriticRNN"
)
OFFICIAL_RECURRENT_STATE_INITIALIZER = (
    "overcooked_v2_experiments.ppo.models.model.initialize_carry"
)
OFFICIAL_APPLY_INTERFACE = (
    "network.apply(params, recurrent_state, (observation[time,batch], "
    "episode_start[time,batch])) -> (next_state, categorical_distribution, value)"
)
OFFICIAL_REFERENCE_ACCEPTANCE_SCHEMA_VERSION = (
    "path_c_official_reference_acceptance_v1"
)
OFFICIAL_REFERENCE_ACCEPTANCE_REPORT_SCHEMA_VERSION = (
    "path_c_official_reference_acceptance_report_v1"
)

_EXPERIMENTS_TRAINING_FILES = (
    "ppo/__init__.py",
    "ppo/config/base.yaml",
    "ppo/config/env/test_time_simple.yaml",
    "ppo/config/wandb/default.yaml",
    "ppo/ippo.py",
    "ppo/main.py",
    "ppo/models/__init__.py",
    "ppo/models/abstract.py",
    "ppo/models/cnn.py",
    "ppo/models/common.py",
    "ppo/models/model.py",
    "ppo/models/rnn.py",
    "ppo/policy.py",
    "ppo/utils/store.py",
    "utils/utils.py",
)
_EXPERIMENTS_EVALUATION_FILES = (
    "eval/__init__.py",
    "eval/evaluate.py",
    "eval/policy.py",
    "eval/rollout.py",
    "eval/utils.py",
)
_JAXMARL_FILES = (
    "__init__.py",
    "environments/__init__.py",
    "registration.py",
    "environments/multi_agent_env.py",
    "environments/overcooked_v2/__init__.py",
    "environments/overcooked_v2/common.py",
    "environments/overcooked_v2/layouts.py",
    "environments/overcooked_v2/overcooked.py",
    "environments/overcooked_v2/settings.py",
    "environments/overcooked_v2/utils.py",
    "viz/__init__.py",
    "viz/overcooked_v2_visualizer.py",
    "wrappers/__init__.py",
    "wrappers/baselines.py",
)
_REPOSITORY_ADAPTER_FILES = (
    "path_c_flax_policy.py",
    "path_c_official_artifact.py",
    "path_c_official_evidence.py",
    "path_c_pool_admission.py",
    "path_c_seed.py",
    "scripts/run_path_c_official_reference_acceptance.py",
    "scripts/run_path_c_official_training.py",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON mapping in {path}.")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one completed JSON record without exposing a partial canonical file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_registered_path(launch_path: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = launch_path.parent / path
    return path.resolve()


def _completed_episode_count_from_metrics(
    returned_episode_by_update: Sequence[float],
    *,
    num_steps: int,
    num_envs: int,
) -> int:
    """Recover completed environment episodes from the official averaged metric."""

    values = np.asarray(returned_episode_by_update, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Official returned_episode metrics must be one finite sequence.")
    if num_steps <= 0 or num_envs <= 0:
        raise ValueError("Official environment and rollout counts must be positive.")
    episode_count = float(values.sum()) * float(num_steps * num_envs)
    rounded = int(round(episode_count))
    if not np.isclose(episode_count, float(rounded), rtol=0.0, atol=1e-5):
        raise RuntimeError("Official returned_episode metrics do not encode an integer count.")
    return rounded


def _official_roots() -> tuple[Path, Path]:
    experiments = Path(
        locate_editable_package("overcooked_v2_experiments")["package_source_root"]
    ).resolve()
    jaxmarl = Path(locate_editable_package("jaxmarl")["package_source_root"]).resolve()
    return experiments, jaxmarl


def _prepare_official_imports() -> tuple[Path, Path]:
    """Expose the package's script-relative ``models`` import without source edits."""

    experiments, jaxmarl = _official_roots()
    ppo_dir = experiments / "ppo"
    normalized = str(ppo_dir)
    if normalized not in sys.path:
        sys.path.insert(0, normalized)
    return experiments, jaxmarl


def official_source_dependency_records(
    experiment_variant: str,
) -> list[dict[str, str]]:
    """Return the checked training, saving, evaluation and environment source set."""

    if experiment_variant not in {"rnn-sp", "rnn-op"}:
        raise ValueError("Official source closure supports only rnn-sp and rnn-op.")
    experiments, jaxmarl = _official_roots()
    variant_file = (
        "ppo/config/experiment/rnn-sp.yaml"
        if experiment_variant == "rnn-sp"
        else "ppo/config/experiment/rnn-op.yaml"
    )
    model_file = (
        "ppo/config/model/rnn.yaml"
        if experiment_variant == "rnn-sp"
        else "ppo/config/model/rnn-op.yaml"
    )
    repository_overcooked_dir = Path(__file__).resolve().parents[1]
    paths = [
        *(experiments / relative for relative in _EXPERIMENTS_TRAINING_FILES),
        experiments / variant_file,
        experiments / model_file,
        *(experiments / relative for relative in _EXPERIMENTS_EVALUATION_FILES),
        *(jaxmarl / relative for relative in _JAXMARL_FILES),
        *(
            repository_overcooked_dir / relative
            for relative in _REPOSITORY_ADAPTER_FILES
        ),
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Official source dependency closure is incomplete: " + ", ".join(missing)
        )
    return [
        {"path": str(path), "sha256": _file_sha256(path)}
        for path in sorted(set(paths), key=lambda item: str(item))
    ]


def compose_official_training_config(
    launch_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the real Hydra configuration used by the external package."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    variant = str(launch_config["experiment_variant"])
    if variant not in {"rnn-sp", "rnn-op"}:
        raise ValueError("Unsupported official experiment variant.")
    experiments, _ = _prepare_official_imports()
    overrides = [
        f"+experiment={variant}",
        "+env=test_time_simple",
        f"SEED={int(launch_config['seed'])}",
        "NUM_SEEDS=1",
        "NUM_CHECKPOINTS=3",
        "VISUALIZE=false",
        "TUNE=false",
        "wandb.WANDB_MODE=disabled",
        "model.TOTAL_TIMESTEPS=3e7",
        "model.REW_SHAPING_HORIZON=1.5e7",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(experiments / "ppo" / "config"),
    ):
        composed = compose(config_name="base", overrides=overrides)
    resolved = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(resolved, Mapping):
        raise ValueError("Official Hydra composition did not produce a mapping.")
    config = dict(resolved)
    _validate_resolved_config(config, launch_config)
    return config


def _validate_resolved_config(
    config: Mapping[str, Any], launch_config: Mapping[str, Any]
) -> None:
    model = config.get("model")
    environment = config.get("env")
    if not isinstance(model, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("Official Hydra configuration lacks model or environment.")
    expected_common = {
        "TYPE": "RNN",
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "ACTIVATION": "relu",
        "TOTAL_TIMESTEPS": 30_000_000.0,
        "REW_SHAPING_HORIZON": 15_000_000.0,
        "LR": 0.00025,
        "NUM_STEPS": 256,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 64,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ANNEAL_LR": True,
        "LR_WARMUP": 0.05,
    }
    for field, expected in expected_common.items():
        if model.get(field) != expected:
            raise ValueError(f"Official resolved model changed {field}.")
    variant = str(launch_config["experiment_variant"])
    expected_variant = (
        {"NUM_ENVS": 256, "ENT_COEF": 0.01}
        if variant == "rnn-sp"
        else {"NUM_ENVS": 64, "ENT_COEF": 0.02}
    )
    for field, expected in expected_variant.items():
        if model.get(field) != expected:
            raise ValueError(f"Official resolved {variant} model changed {field}.")
    kwargs = environment.get("ENV_KWARGS")
    if not isinstance(kwargs, Mapping):
        raise ValueError("Official resolved environment kwargs are missing.")
    for field, expected in {
        "layout": "test_time_simple",
        "agent_view_size": 2,
        "negative_rewards": True,
        "random_agent_positions": True,
        "sample_recipe_on_delivery": True,
        "indicate_successful_delivery": True,
    }.items():
        if kwargs.get(field) != expected:
            raise ValueError(f"Official resolved environment changed {field}.")
    if variant == "rnn-op" and list(kwargs.get("op_ingredient_permutations", ())) != [
        0,
        1,
    ]:
        raise ValueError("Official rnn-op did not enable the registered symmetry.")
    if int(config.get("SEED", -1)) != int(launch_config["seed"]):
        raise ValueError("Official resolved seed differs from launch registration.")
    if int(config.get("NUM_SEEDS", -1)) != 1:
        raise ValueError("Each registered official run must contain one seed.")
    wandb = config.get("wandb")
    if not isinstance(wandb, Mapping) or wandb.get("WANDB_MODE") != "disabled":
        raise ValueError("Official registered runs must keep Weights & Biases disabled.")


def reference_configuration_comparison(
    resolved_config: Mapping[str, Any],
    *,
    accepted_reference_source: str | Path | None = None,
) -> dict[str, Any]:
    """Confirm that the registered reference is the production package network."""

    model = resolved_config["model"]
    environment = resolved_config["env"]["ENV_KWARGS"]
    experiments, _ = _official_roots()
    candidate_network = experiments / "ppo" / "models" / "rnn.py"
    reference_path = (
        candidate_network
        if accepted_reference_source is None
        else Path(accepted_reference_source).resolve()
    )
    reference_source = reference_path.read_text(encoding="utf-8")
    candidate_source = candidate_network.read_text(encoding="utf-8")
    fields = {
        "layout": environment.get("layout") == "test_time_simple",
        "total_timesteps": model.get("TOTAL_TIMESTEPS") == 30_000_000.0,
        "num_envs": model.get("NUM_ENVS") == 256,
        "num_steps": model.get("NUM_STEPS") == 256,
        "update_epochs": model.get("UPDATE_EPOCHS") == 4,
        "num_minibatches": model.get("NUM_MINIBATCHES") == 64,
        "reward_shaping_horizon": model.get("REW_SHAPING_HORIZON")
        == 15_000_000.0,
        "action_order": OFFICIAL_ACTION_ORDER
        == ("right", "down", "left", "up", "stay", "interact"),
    }
    reference_class = reference_source.split("class ActorCriticRNN", 1)[1].split(
        "\nclass ", 1
    )[0]
    candidate_class = candidate_source.split("class ActorCriticRNN", 1)[1].split(
        "\nclass ", 1
    )[0]
    reference_initializer = "kernel_init=orthogonal(jnp.sqrt(2))"
    initializer_matches = (
        reference_class.count(reference_initializer) >= 2
        and candidate_class.count(reference_initializer) >= 2
    )
    source_identity_matches = (
        reference_path == candidate_network
        and _file_sha256(reference_path) == _file_sha256(candidate_network)
    )
    fields["actor_critic_hidden_initializer"] = initializer_matches
    fields["registered_reference_is_candidate_network"] = source_identity_matches
    return {
        "schema_version": "path_c_official_reference_comparison_v2",
        "reference_id": OFFICIAL_REFERENCE_ID,
        "accepted_reference_source": str(reference_path),
        "accepted_reference_source_sha256": _file_sha256(reference_path),
        "candidate_network_source": str(candidate_network),
        "candidate_network_source_sha256": _file_sha256(candidate_network),
        "checks": fields,
        "material_differences": (
            []
            if initializer_matches and source_identity_matches
            else [
                {
                    "field": "registered_reference_identity",
                    "accepted_reference": (
                        "experiments-package ActorCriticRNN with orthogonal gain sqrt(2)"
                    ),
                    "candidate": "same" if source_identity_matches else "different source",
                }
            ]
        ),
        "matches_accepted_reference": all(fields.values()),
    }


def validate_reference_acceptance_contract(
    launch_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the frozen seed-999 mechanical acceptance rule."""

    contract = launch_config.get("reference_acceptance")
    expected = {
        "schema_version": OFFICIAL_REFERENCE_ACCEPTANCE_SCHEMA_VERSION,
        "rule_id": OFFICIAL_REFERENCE_ACCEPTANCE_RULE_ID,
        "reference_id": OFFICIAL_REFERENCE_ID,
        "reference_source": (
            "overcooked_v2_experiments/ppo/models/rnn.py"
        ),
        "reference_source_sha256": (
            "b97c92133ed073e780b643e8e4b0c8fc27f4963ab99c2d8bcf0f099a5a78a1d0"
        ),
        "seed": 999,
        "metric": "returned_episode_returns",
        "metric_meaning": "official_wrapper_raw_episode_return",
        "expected_update_count": 457,
        "quarter_partition": "contiguous_integer_update_quarters_v1",
        "minimum_final_quarter_mean_raw_return": 100.0,
        "minimum_final_to_peak_quarter_ratio": 0.9,
        "decision_rule": "both_thresholds_required",
    }
    if not isinstance(contract, Mapping) or dict(contract) != expected:
        raise ValueError("Official reference acceptance contract changed.")
    if launch_config.get("experiment_variant") != "rnn-sp" or int(
        launch_config.get("seed", -1)
    ) != 999:
        raise ValueError("Official reference acceptance requires rnn-sp seed 999.")
    if dict(launch_config["effective_environment_steps_contract"]) != (
        OFFICIAL_EFFECTIVE_STEP_CONTRACTS["rnn-sp"]
    ):
        raise ValueError("Official reference acceptance changed its effective steps.")
    return contract


def evaluate_official_reference_acceptance(
    returned_episode_returns: Sequence[float],
    *,
    launch_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen final-quarter performance and no-collapse conditions."""

    contract = validate_reference_acceptance_contract(launch_config)
    values = np.asarray(returned_episode_returns, dtype=np.float64)
    expected_count = int(contract["expected_update_count"])
    if values.ndim != 1 or values.shape[0] != expected_count:
        raise ValueError("Reference acceptance requires one value for every update.")
    if not np.isfinite(values).all():
        raise ValueError("Reference acceptance values must all be finite.")
    boundaries = [index * expected_count // 4 for index in range(5)]
    quarter_means = [
        float(values[boundaries[index] : boundaries[index + 1]].mean())
        for index in range(4)
    ]
    final_mean = quarter_means[-1]
    peak_mean = max(quarter_means)
    final_to_peak_ratio = final_mean / peak_mean if peak_mean > 0.0 else float("-inf")
    performance_pass = final_mean >= float(
        contract["minimum_final_quarter_mean_raw_return"]
    )
    no_collapse_pass = final_to_peak_ratio >= float(
        contract["minimum_final_to_peak_quarter_ratio"]
    )
    return {
        "schema_version": OFFICIAL_REFERENCE_ACCEPTANCE_REPORT_SCHEMA_VERSION,
        "scientific_readout_allowed": False,
        "reference_id": contract["reference_id"],
        "rule_id": contract["rule_id"],
        "seed": int(contract["seed"]),
        "effective_environment_steps": OFFICIAL_EFFECTIVE_STEP_CONTRACTS[
            "rnn-sp"
        ]["effective_environment_steps"],
        "metric": contract["metric"],
        "quarter_boundaries_zero_based": boundaries,
        "quarter_means": quarter_means,
        "final_quarter_mean_raw_return": final_mean,
        "peak_quarter_mean_raw_return": peak_mean,
        "final_to_peak_quarter_ratio": final_to_peak_ratio,
        "performance_threshold_pass": performance_pass,
        "no_collapse_threshold_pass": no_collapse_pass,
        "acceptance_pass": performance_pass and no_collapse_pass,
    }


def run_official_reference_acceptance(
    launch_config_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Run the registered seed-999 Type-A curve and write its mechanical verdict."""

    import copy

    import jax
    import wandb

    _prepare_official_imports()
    from overcooked_v2_experiments.ppo.ippo import make_train
    from overcooked_v2_experiments.utils.utils import mini_batch_pmap

    launch = load_official_launch_config(launch_config_path)
    contract = validate_reference_acceptance_contract(launch)
    config = copy.deepcopy(compose_official_training_config(launch))
    comparison = reference_configuration_comparison(config)
    if not comparison["matches_accepted_reference"]:
        raise RuntimeError("Registered production network no longer matches its reference.")
    if comparison["candidate_network_source_sha256"] != contract[
        "reference_source_sha256"
    ]:
        raise RuntimeError("Registered reference network source hash changed.")
    config["NUM_CHECKPOINTS"] = 0
    train = make_train(config)
    mapped = mini_batch_pmap(jax.jit(train), 1)
    started = time.monotonic()
    with wandb.init(
        project="r015-official-reference-acceptance",
        mode="disabled",
        reinit=True,
    ):
        output = mapped(jax.random.split(jax.random.PRNGKey(999), 1))
        jax.block_until_ready(output["metrics"]["env_step"])

    metric_values = np.asarray(
        output["metrics"]["returned_episode_returns"], dtype=np.float64
    )
    if metric_values.ndim >= 2 and metric_values.shape[0] == 1:
        metric_values = metric_values[0]
    if metric_values.ndim > 1:
        metric_values = metric_values.mean(
            axis=tuple(range(1, metric_values.ndim))
        )
    report = evaluate_official_reference_acceptance(
        metric_values,
        launch_config=launch,
    )
    completed_environment_steps = int(
        np.asarray(output["metrics"]["env_step"]).max()
    )
    expected_environment_steps = OFFICIAL_EFFECTIVE_STEP_CONTRACTS["rnn-sp"][
        "effective_environment_steps"
    ]
    if completed_environment_steps != expected_environment_steps:
        raise RuntimeError("Reference run did not complete its effective-step contract.")
    report.update(
        {
            "launch_config_path": str(Path(launch_config_path).resolve()),
            "launch_config_sha256": _file_sha256(
                Path(launch_config_path).resolve()
            ),
            "reference_acceptance_contract": dict(contract),
            "nominal_total_timesteps": OFFICIAL_NOMINAL_TOTAL_TIMESTEPS,
            "completed_environment_steps": completed_environment_steps,
            "returned_episode_returns_by_update": metric_values.tolist(),
            "reference_comparison": comparison,
            "source_dependencies": official_source_dependency_records("rnn-sp"),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    target = Path(report_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def validate_official_reference_acceptance_report(
    launch_config_path: str | Path,
    report_path: str | Path,
) -> Mapping[str, Any]:
    """Recompute the seed-999 verdict before any production launch is allowed."""

    launch = load_official_launch_config(launch_config_path)
    contract = validate_reference_acceptance_contract(launch)
    report = _load_json(Path(report_path).resolve())
    launch_path = Path(launch_config_path).resolve()
    if report.get("launch_config_path") != str(launch_path) or report.get(
        "launch_config_sha256"
    ) != _file_sha256(launch_path) or report.get(
        "reference_acceptance_contract"
    ) != dict(contract):
        raise ValueError("Reference acceptance report changed its launch contract.")
    raw_values = report.get("returned_episode_returns_by_update")
    if not isinstance(raw_values, Sequence) or isinstance(
        raw_values, (str, bytes, bytearray)
    ):
        raise ValueError("Reference acceptance report lacks its per-update evidence.")
    recomputed = evaluate_official_reference_acceptance(
        raw_values,
        launch_config=launch,
    )
    for field, expected in recomputed.items():
        if report.get(field) != expected:
            raise ValueError(f"Reference acceptance report changed {field}.")
    if int(report.get("completed_environment_steps", -1)) != (
        OFFICIAL_EFFECTIVE_STEP_CONTRACTS["rnn-sp"]["effective_environment_steps"]
    ):
        raise ValueError("Reference acceptance report changed its effective steps.")
    comparison = report.get("reference_comparison")
    if not isinstance(comparison, Mapping) or comparison.get(
        "reference_id"
    ) != OFFICIAL_REFERENCE_ID or comparison.get(
        "candidate_network_source_sha256"
    ) != contract["reference_source_sha256"]:
        raise ValueError("Reference acceptance report changed its network identity.")
    if report.get("source_dependencies") != official_source_dependency_records(
        "rnn-sp"
    ):
        raise ValueError("Reference acceptance report changed its source closure.")
    if report.get("acceptance_pass") is not True:
        raise ValueError("Reference acceptance failed; production must not start.")
    return report


def verify_other_play_symmetry(
    resolved_config: Mapping[str, Any], *, reset_count: int = 64
) -> dict[str, Any]:
    """Execute fixed resets and confirm independent per-agent ingredient renaming."""

    if isinstance(reset_count, bool) or not isinstance(reset_count, int) or reset_count < 2:
        raise ValueError("Other-Play symmetry verification requires at least two resets.")
    import jax
    import jax.numpy as jnp
    from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

    kwargs = dict(resolved_config["env"]["ENV_KWARGS"])
    if list(kwargs.get("op_ingredient_permutations", ())) != [0, 1]:
        raise ValueError("Resolved configuration did not request the registered symmetry.")
    environment = OvercookedV2(**kwargs)
    keys = jax.random.split(jax.random.PRNGKey(202), reset_count)
    _, states = jax.vmap(environment.reset)(keys)
    permutations = np.asarray(states.ingredient_permutations)
    if permutations.shape != (reset_count, 2, 3):
        raise RuntimeError("Other-Play reset produced an unexpected permutation shape.")
    expected = {(0, 1, 2), (1, 0, 2)}
    observed_by_agent = [
        {tuple(int(value) for value in row) for row in permutations[:, agent_id, :]}
        for agent_id in range(2)
    ]
    if any(observed != expected for observed in observed_by_agent):
        raise RuntimeError("Other-Play did not exercise both registered symmetries.")
    independently_sampled = bool(
        jnp.any(
            jnp.asarray(permutations[:, 0, :])
            != jnp.asarray(permutations[:, 1, :])
        )
    )
    if not independently_sampled:
        raise RuntimeError("Other-Play used one shared permutation for both agents.")
    return {
        "schema_version": "path_c_official_other_play_symmetry_check_v1",
        "reset_count": reset_count,
        "permuted_ingredient_indices": [0, 1],
        "ingredient_2_fixed": True,
        "both_symmetries_seen_per_agent": True,
        "independent_agent_permutations_seen": True,
    }


@dataclass(frozen=True)
class OvercookedV2ExperimentsNetworkAdapter:
    """官方循环策略的初始化与逐步 ``apply`` 桥接。"""

    config: Mapping[str, Any]

    def _network(self) -> Any:
        _prepare_official_imports()
        from overcooked_v2_experiments.ppo.models.model import get_actor_critic

        return get_actor_critic(self.config)

    def initial_state(self, batch_size: int) -> Any:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Official recurrent batch size must be positive.")
        _prepare_official_imports()
        from overcooked_v2_experiments.ppo.models.model import initialize_carry

        return initialize_carry(self.config, batch_size)

    def apply_actor(
        self,
        params: Mapping[str, Any],
        recurrent_state: Any,
        observation: Any,
        episode_start: Any,
    ) -> tuple[Any, Any]:
        import jax.numpy as jnp

        observation_array = jnp.asarray(observation)
        start_array = jnp.asarray(episode_start, dtype=jnp.bool_)
        single = observation_array.ndim == 3
        if single:
            observation_array = observation_array[jnp.newaxis, ...]
            start_array = start_array.reshape((1,))
        if observation_array.ndim != 4 or start_array.ndim != 1 or (
            observation_array.shape[0] != start_array.shape[0]
        ):
            raise ValueError("Official actor expects one batch of local observations.")
        ac_input = (
            observation_array[jnp.newaxis, ...],
            start_array[jnp.newaxis, ...],
        )
        next_state, distribution, _ = self._network().apply(
            params,
            recurrent_state,
            ac_input,
        )
        logits = distribution.logits[0]
        if logits.shape[-1] != 6:
            raise ValueError("Official actor must expose six logits.")
        if single:
            logits = logits[0]
        return next_state, logits


def initialize_official_parameters(
    config: Mapping[str, Any], *, random_key: Any
) -> Mapping[str, Any]:
    """Initialize the official parameter tree for a Type-A round trip."""

    import jax.numpy as jnp
    import jaxmarl

    adapter = OvercookedV2ExperimentsNetworkAdapter(config)
    environment = jaxmarl.make(
        config["env"]["ENV_NAME"], **dict(config["env"]["ENV_KWARGS"])
    )
    observation = jnp.zeros((1, 1, *environment.observation_space().shape))
    episode_start = jnp.zeros((1, 1), dtype=jnp.bool_)
    params = adapter._network().init(
        random_key,
        adapter.initial_state(1),
        (observation, episode_start),
    )
    return params


def save_official_checkpoint(
    checkpoint_root: str | Path,
    *,
    config: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Path:
    """Use the official Orbax save function and return ``run_0/ckpt_final``."""

    _prepare_official_imports()
    from overcooked_v2_experiments.ppo.utils.store import store_checkpoint

    root = Path(checkpoint_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stored_config = dict(config)
    stored_config["RUN_BASE_DIR"] = root
    store_checkpoint(stored_config, params, 0, 0, final=True)
    target = root / "run_0" / "ckpt_final"
    if not target.is_dir():
        raise RuntimeError("Official checkpoint save did not create ckpt_final.")
    return target


def save_official_scheduled_checkpoint(
    checkpoint_root: str | Path,
    *,
    config: Mapping[str, Any],
    params: Mapping[str, Any],
    update_step: int,
) -> Path:
    """Persist one checkpoint selected only by the official update schedule."""

    _prepare_official_imports()
    from overcooked_v2_experiments.ppo.utils.store import store_checkpoint

    step = int(update_step)
    if step < 0:
        raise ValueError("A scheduled checkpoint update must be non-negative.")
    root = Path(checkpoint_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stored_config = dict(config)
    stored_config["RUN_BASE_DIR"] = root
    store_checkpoint(stored_config, params, 0, step, final=False)
    target = root / "run_0" / f"ckpt_{step}"
    if not target.is_dir():
        raise RuntimeError("Official checkpoint save did not create its scheduled path.")
    return target


def restore_official_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Restore the native ``config`` and ``params`` objects from Orbax."""

    try:
        import orbax.checkpoint as ocp
    except ImportError as error:  # pragma: no cover - remote dependency
        raise RuntimeError("Official checkpoint restore requires Orbax.") from error
    restored = ocp.PyTreeCheckpointer().restore(str(Path(checkpoint_path).resolve()))
    if not isinstance(restored, Mapping) or set(restored) != {"config", "params"}:
        raise ValueError("Official Orbax checkpoint has an unexpected root schema.")
    config = restored["config"]
    params = restored["params"]
    if not isinstance(config, Mapping) or not isinstance(params, Mapping):
        raise ValueError("Official Orbax checkpoint lacks config or params mappings.")
    return config, params


def run_official_production_training(
    launch_config_path: str | Path,
) -> dict[str, Any]:
    """Execute one registered official run and bind its final native checkpoint."""

    import copy

    import jax
    import wandb

    _prepare_official_imports()
    from overcooked_v2_experiments.ppo.ippo import make_train
    from overcooked_v2_experiments.utils.utils import mini_batch_pmap

    launch_path = Path(launch_config_path).resolve()
    launch = load_official_launch_config(launch_path)
    if int(launch["seed"]) == 999:
        raise ValueError("The reference seed cannot enter official partner production.")

    acceptance = launch["type_a_acceptance"]
    acceptance_config = _resolve_registered_path(
        launch_path,
        acceptance["reference_acceptance_config"],
    )
    acceptance_report = _resolve_registered_path(
        launch_path,
        acceptance["reference_acceptance_report"],
    )
    verified_acceptance = validate_official_reference_acceptance_report(
        acceptance_config,
        acceptance_report,
    )

    config = copy.deepcopy(compose_official_training_config(launch))
    comparison = reference_configuration_comparison(config)
    if not comparison["matches_accepted_reference"]:
        raise RuntimeError("Official production network differs from its accepted reference.")
    if int(config.get("NUM_CHECKPOINTS", -1)) != 3:
        raise RuntimeError("Official production changed its three-checkpoint training loop.")

    output_dir = _resolve_registered_path(launch_path, launch["output_dir"])
    expected_checkpoint = _resolve_registered_path(
        launch_path,
        launch["checkpoint"]["expected_path"],
    )
    if expected_checkpoint != output_dir / "run_0" / "ckpt_final":
        raise ValueError("Official launch output and final checkpoint paths disagree.")
    if output_dir.exists():
        raise FileExistsError(
            "Official production output already exists and will not be overwritten."
        )

    dependencies = official_source_dependency_records(
        str(launch["experiment_variant"])
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train = make_train(config)
    mapped = mini_batch_pmap(jax.jit(train), 1)
    with wandb.init(
        project="r015-official-partner-production",
        mode="disabled",
        reinit=True,
    ):
        output = mapped(
            jax.random.split(
                jax.random.PRNGKey(int(launch["seed"])),
                1,
            )
        )
        jax.block_until_ready(output["metrics"]["env_step"])

    metric_arrays = jax.tree_util.tree_map(
        lambda value: np.asarray(value)[0],
        output["metrics"],
    )
    expected_updates = int(
        launch["effective_environment_steps_contract"]["num_updates"]
    )
    for value in jax.tree_util.tree_leaves(metric_arrays):
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != expected_updates:
            raise RuntimeError("Official training metrics changed their update axis.")
        if not np.isfinite(array).all():
            raise RuntimeError("Official training produced a non-finite metric.")
    completed_environment_steps = int(np.asarray(metric_arrays["env_step"]).max())
    expected_environment_steps = int(
        launch["effective_environment_steps_contract"][
            "effective_environment_steps"
        ]
    )
    if completed_environment_steps != expected_environment_steps:
        raise RuntimeError("Official training did not complete its effective-step contract.")

    model_config = config["model"]
    completed_episode_count = _completed_episode_count_from_metrics(
        np.asarray(metric_arrays["returned_episode"], dtype=np.float64),
        num_steps=int(model_config["NUM_STEPS"]),
        num_envs=int(model_config["NUM_ENVS"]),
    )
    checkpoint_state = output["runner_state"][1]
    checkpoint_leaves = jax.tree_util.tree_leaves(checkpoint_state)
    if not checkpoint_leaves or any(
        np.asarray(value).ndim < 2 or np.asarray(value).shape[:2] != (1, 3)
        for value in checkpoint_leaves
    ):
        raise RuntimeError(
            "Official training did not return the registered three checkpoint states."
        )
    scheduled_updates = tuple(
        int(value)
        for value in np.linspace(0, expected_updates, num=3, dtype=np.int64)
    )
    history_params = tuple(
        jax.tree_util.tree_map(
            lambda value, checkpoint_index=index: value[0, checkpoint_index],
            checkpoint_state,
        )
        for index in range(3)
    )
    final_params = history_params[-1]
    scheduled_paths = tuple(
        save_official_scheduled_checkpoint(
            output_dir,
            config=config,
            params=params_at_checkpoint,
            update_step=update_step,
        )
        for params_at_checkpoint, update_step in zip(
            history_params, scheduled_updates, strict=True
        )
    )
    checkpoint_path = save_official_checkpoint(
        output_dir,
        config=config,
        params=final_params,
    )
    if checkpoint_path != expected_checkpoint:
        raise RuntimeError("Official native save created an unregistered checkpoint path.")

    def jsonable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): jsonable(child) for key, child in value.items()}
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()

    elapsed_seconds = time.monotonic() - started
    metrics_record = {
        "schema_version": "path_c_official_training_metrics_v1",
        "scientific_readout_allowed": False,
        "experiment_variant": launch["experiment_variant"],
        "seed": int(launch["seed"]),
        "effective_environment_steps": completed_environment_steps,
        "completed_episode_count": completed_episode_count,
        "metric_record_count": expected_updates,
        "metrics_by_update": jsonable(metric_arrays),
    }
    metrics_path = output_dir / "training_metrics.json"
    _write_json_atomic(metrics_path, metrics_record)

    descriptor = {
        "schema_version": OFFICIAL_RUN_DESCRIPTOR_SCHEMA_VERSION,
        "run_status": "completed",
        "scientific_readout_allowed": False,
        "experiment_variant": launch["experiment_variant"],
        "layout": launch["layout"],
        "seed": int(launch["seed"]),
        "nominal_total_timesteps": OFFICIAL_NOMINAL_TOTAL_TIMESTEPS,
        "effective_environment_steps": completed_environment_steps,
        "completed_episode_count": completed_episode_count,
        "metric_record_count": expected_updates,
        "metrics_path": str(metrics_path),
        "environment_kwargs": dict(launch["environment_kwargs"]),
        "architecture": dict(launch["model"]),
        "checkpoint": {
            "path": str(checkpoint_path),
            "format": OFFICIAL_CHECKPOINT_FORMAT,
            "parameter_tree_path": list(OFFICIAL_PARAMETER_TREE_PATH),
        },
        "reference_acceptance_report": str(acceptance_report),
        "reference_acceptance_report_sha256": _file_sha256(acceptance_report),
        "reference_acceptance_rule_id": OFFICIAL_REFERENCE_ACCEPTANCE_RULE_ID,
        "reference_acceptance_pass": verified_acceptance["acceptance_pass"],
        "training_source_dependencies": dependencies,
        "elapsed_seconds": elapsed_seconds,
    }
    descriptor_path = output_dir / "run_descriptor.json"
    _write_json_atomic(descriptor_path, descriptor)
    manifest = build_official_artifact_manifest(
        launch_path,
        descriptor_path,
        params=final_params,
    )
    if manifest.get("schema_version") != OFFICIAL_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Official artifact builder returned the wrong schema.")
    history = []
    environment_steps_per_update = int(model_config["NUM_ENVS"]) * int(
        model_config["NUM_STEPS"]
    )
    for index, (update_step, history_path, params_at_checkpoint) in enumerate(
        zip(scheduled_updates, scheduled_paths, history_params, strict=True)
    ):
        history.append(
            {
                "checkpoint_index": index,
                "update_step": update_step,
                "effective_environment_steps": (
                    update_step * environment_steps_per_update
                ),
                "path": str(history_path),
                "format": OFFICIAL_CHECKPOINT_FORMAT,
                "parameter_tree_path": list(OFFICIAL_PARAMETER_TREE_PATH),
                "selection_rule": "mechanical_official_update_schedule_v1",
                "checkpoint_sha256": checkpoint_artifact_sha256(history_path),
                "model_weights_sha256": flax_weights_sha256(
                    params_at_checkpoint
                ),
            }
        )
    if history[-1]["model_weights_sha256"] != manifest["checkpoint"][
        "model_weights_sha256"
    ]:
        raise RuntimeError(
            "The last scheduled checkpoint differs from the admitted final policy."
        )
    manifest = {**manifest, "checkpoint_history": history}
    manifest_path = output_dir / "official_artifact_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    provenance = validate_official_artifact_manifest(
        launch_path,
        manifest_path,
        expected_checkpoint_path=checkpoint_path,
        params=final_params,
    )
    return {
        "schema_version": "path_c_official_training_completion_v1",
        "scientific_readout_allowed": False,
        "output_dir": str(output_dir),
        "run_descriptor_path": str(descriptor_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "completed_environment_steps": completed_environment_steps,
        "completed_episode_count": completed_episode_count,
        "metric_record_count": expected_updates,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "model_weights_sha256": provenance["model_weights_sha256"],
        "training_run_id": provenance["training_run_id"],
        "checkpoint_history": history,
    }


@dataclass(frozen=True)
class _LoadedPolicy:
    policy: OfficialFlaxPolicy
    adapter: OvercookedV2ExperimentsNetworkAdapter
    config: Mapping[str, Any]


@dataclass
class OCV2R015OfficialRolloutBackendV1:
    """在官方 JaxMARL 环境中批量产生 R015 逐回合机械证据。"""

    policies: Mapping[str, _LoadedPolicy]

    def evaluate_pairing(
        self,
        policy_0_candidate_id: str,
        policy_1_candidate_id: str,
        *,
        canonical_episode_seeds: Sequence[int],
    ) -> Sequence[Mapping[str, Any]]:
        if policy_0_candidate_id not in self.policies or (
            policy_1_candidate_id not in self.policies
        ):
            raise ValueError("Official rollout requested an unknown candidate.")
        seeds = tuple(int(seed) for seed in canonical_episode_seeds)
        if not seeds:
            raise ValueError("Official rollout requires at least one episode seed.")
        return self._evaluate_batch(
            self.policies[policy_0_candidate_id],
            self.policies[policy_1_candidate_id],
            seeds,
        )

    @staticmethod
    def _evaluate_batch(
        left: _LoadedPolicy,
        right: _LoadedPolicy,
        seeds: Sequence[int],
    ) -> list[dict[str, Any]]:
        rows, _ = OCV2R015OfficialRolloutBackendV1._evaluate_batch_with_trace(
            left,
            right,
            seeds,
        )
        return rows

    @staticmethod
    def _evaluate_batch_with_trace(
        left: _LoadedPolicy,
        right: _LoadedPolicy,
        seeds: Sequence[int],
    ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        import jax
        import jax.numpy as jnp
        from jaxmarl.environments.overcooked_v2.common import (
            Actions,
            DynamicObject,
            StaticObject,
        )
        from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

        env_kwargs = dict(left.config["env"]["ENV_KWARGS"])
        env_kwargs.pop("op_ingredient_permutations", None)
        env_kwargs["max_steps"] = 400
        environment = OvercookedV2(**env_kwargs)
        if environment.num_agents != 2 or environment.max_steps != 400:
            raise ValueError("R015 official rollout requires two agents and 400 steps.")

        def event_counts(state: Any, actions: Mapping[str, Any]) -> tuple[Any, Any]:
            action_array = jnp.stack((actions["agent_0"], actions["agent_1"]))
            forward = jax.vmap(lambda agent: agent.get_fwd_pos())(state.agents)
            cells = state.grid[forward.y, forward.x]
            interact = action_array == int(Actions.interact)
            delivery = (
                interact
                & (cells[:, 0] == int(StaticObject.GOAL))
                & ((state.agents.inventory & int(DynamicObject.COOKED)) != 0)
            )
            indicator = (
                interact
                & (cells[:, 0] == int(StaticObject.BUTTON_RECIPE_INDICATOR))
                & (state.agents.inventory == int(DynamicObject.EMPTY))
                & (cells[:, 1] == int(DynamicObject.EMPTY))
            )
            return jnp.sum(delivery), jnp.sum(indicator)

        def one_episode(root_key: Any) -> tuple[Any, ...]:
            rollout_key, reset_key = jax.random.split(root_key, 2)
            observation, state = environment.reset(reset_key)
            done = {"agent_0": False, "agent_1": False, "__all__": False}
            carry = (
                observation,
                state,
                done,
                left.adapter.initial_state(1),
                right.adapter.initial_state(1),
                jnp.array(0.0),
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32),
                jnp.zeros((6,), dtype=jnp.int32),
                jnp.zeros((6,), dtype=jnp.int32),
            )

            def step(step_carry: tuple[Any, ...], step_key: Any):
                (
                    obs,
                    env_state,
                    previous_done,
                    left_state,
                    right_state,
                    total,
                    correct,
                    wrong,
                    indicator,
                    left_counts,
                    right_counts,
                ) = step_carry
                sample_key, environment_key = jax.random.split(step_key, 2)
                left_key, right_key = jax.random.split(sample_key, 2)
                next_left_state, left_logits = left.adapter.apply_actor(
                    left.policy.params,
                    left_state,
                    obs["agent_0"],
                    previous_done["agent_0"],
                )
                next_right_state, right_logits = right.adapter.apply_actor(
                    right.policy.params,
                    right_state,
                    obs["agent_1"],
                    previous_done["agent_1"],
                )
                actions = {
                    "agent_0": jax.random.categorical(left_key, left_logits),
                    "agent_1": jax.random.categorical(right_key, right_logits),
                }
                deliveries, activations = event_counts(env_state, actions)
                next_obs, next_env_state, reward, next_done, _ = environment.step(
                    environment_key,
                    env_state,
                    actions,
                )
                new_correct = next_env_state.new_correct_delivery.astype(jnp.int32)
                new_wrong = deliveries.astype(jnp.int32) - new_correct
                return (
                    next_obs,
                    next_env_state,
                    next_done,
                    next_left_state,
                    next_right_state,
                    total + reward["agent_0"],
                    correct + new_correct,
                    wrong + new_wrong,
                    indicator + activations.astype(jnp.int32),
                    left_counts
                    + jax.nn.one_hot(actions["agent_0"], 6, dtype=jnp.int32),
                    right_counts
                    + jax.nn.one_hot(actions["agent_1"], 6, dtype=jnp.int32),
                ), (
                    actions["agent_0"],
                    actions["agent_1"],
                    reward["agent_0"],
                    next_env_state.new_correct_delivery,
                )

            final, trace = jax.lax.scan(
                step,
                carry,
                jax.random.split(rollout_key, 400),
            )
            return (
                final[5],
                final[6],
                final[7],
                final[8],
                final[9],
                final[10],
                *trace,
            )

        execution_seeds = np.asarray(
            [derive_ocv2_execution_seed(seed) for seed in seeds], dtype=np.uint32
        )
        keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(execution_seeds))
        outputs = jax.jit(jax.vmap(one_episode))(keys)
        host = [np.asarray(value) for value in outputs]
        rows: list[dict[str, Any]] = []
        for index in range(len(seeds)):
            rows.append(
                {
                    "raw_episode_return": float(host[0][index]),
                    "correct_delivery_count": int(host[1][index]),
                    "wrong_delivery_count": int(host[2][index]),
                    "indicator_activation_count": int(host[3][index]),
                    "ambiguous_reward_step_count": 0,
                    "policy_0_action_counts": [
                        int(value) for value in host[4][index].tolist()
                    ],
                    "policy_1_action_counts": [
                        int(value) for value in host[5][index].tolist()
                    ],
                }
            )
        return rows, {
            "policy_0_actions": host[6],
            "policy_1_actions": host[7],
            "raw_step_rewards": host[8],
            "successful_delivery_indicators": host[9],
        }


def _load_candidate_policy(candidate: Any) -> _LoadedPolicy:
    provenance = validate_official_artifact_manifest(
        candidate.training_config_path,
        candidate.training_manifest_path,
        expected_checkpoint_path=candidate.checkpoint_path,
    )
    manifest = _load_json(candidate.training_manifest_path)
    checkpoint = manifest["checkpoint"]
    params = load_flax_parameter_tree(
        candidate.checkpoint_path,
        checkpoint_format=str(checkpoint["format"]),
        parameter_tree_path=tuple(checkpoint["parameter_tree_path"]),
    )
    launch = load_official_launch_config(candidate.training_config_path)
    config = compose_official_training_config(launch)
    adapter = OvercookedV2ExperimentsNetworkAdapter(config)
    policy = OfficialFlaxPolicy(
        params=params,
        network=adapter,
        expected_model_weights_sha256=provenance["model_weights_sha256"],
        action_rule=OFFICIAL_ACTION_RULE_ID,
    )
    return _LoadedPolicy(policy=policy, adapter=adapter, config=config)


def build_official_r015_rollout_backend(
    config: Mapping[str, Any],
    support_spec: R015PartnerSupportSpec,
) -> OfficialR015RolloutBackend:
    """Build the admission backend only from four verified official artifacts."""

    del config
    policies = {
        candidate.candidate_id: _load_candidate_policy(candidate)
        for candidate in support_spec.candidates
    }
    backend = OCV2R015OfficialRolloutBackendV1(policies=policies)
    if not isinstance(backend, OfficialR015RolloutBackend):
        raise TypeError("Official R015 rollout backend does not satisfy its protocol.")
    return backend


def run_type_a_round_trip(
    launch_config_path: str | Path,
    output_dir: str | Path,
    *,
    accepted_reference_source: str | Path | None = None,
    seed: int = 915,
) -> dict[str, Any]:
    """Run initialization, Orbax round trip and one 400-step wiring episode."""

    import jax
    import jaxlib
    import jax.numpy as jnp
    from overcooked_v2_experiments.ppo.policy import PPOPolicy

    launch = load_official_launch_config(launch_config_path)
    config = compose_official_training_config(launch)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    params = initialize_official_parameters(config, random_key=jax.random.PRNGKey(seed))
    initial_hash = flax_weights_sha256(params)
    checkpoint = save_official_checkpoint(output / "checkpoint_round_trip", config=config, params=params)
    restored_config, restored_params = restore_official_checkpoint(checkpoint)
    restored_hash = flax_weights_sha256(restored_params)
    if restored_hash != initial_hash:
        raise RuntimeError("Official Orbax round trip changed the Flax parameter tree.")

    adapter = OvercookedV2ExperimentsNetworkAdapter(restored_config)
    wrapped = OfficialFlaxPolicy(
        params=restored_params,
        network=adapter,
        expected_model_weights_sha256=restored_hash,
    )
    official = PPOPolicy(restored_params, restored_config, stochastic=True)
    observation = jnp.zeros((5, 5, 39), dtype=jnp.float32)
    episode_start = jnp.array(True)
    key = jax.random.PRNGKey(seed + 1)
    wrapped_state = wrapped.initial_state(1)
    official_state = official.init_hstate(1)
    wrapped_step = wrapped.act(observation, episode_start, wrapped_state, key)
    official_action, official_next_state = official.compute_action(
        observation,
        episode_start,
        official_state,
        key,
    )
    _, official_distribution, _ = official.network.apply(
        restored_params,
        official_state,
        (observation[jnp.newaxis, jnp.newaxis, ...], episode_start.reshape((1, 1))),
    )
    if not np.array_equal(np.asarray(wrapped_step.action), np.asarray(official_action)):
        raise RuntimeError("Official policy and repository wrapper sampled different actions.")
    if not np.array_equal(
        np.asarray(wrapped_step.next_recurrent_state), np.asarray(official_next_state)
    ):
        raise RuntimeError("Official policy and wrapper produced different recurrent states.")
    if not np.array_equal(
        np.asarray(wrapped_step.logits), np.asarray(official_distribution.logits[0, 0])
    ):
        raise RuntimeError("Official policy and wrapper produced different logits.")

    loaded = _LoadedPolicy(policy=wrapped, adapter=adapter, config=restored_config)
    rollout_backend = OCV2R015OfficialRolloutBackendV1(
        policies={"left": loaded, "right": loaded}
    )
    episode_seed = seed + 2
    episodes, repository_trace = rollout_backend._evaluate_batch_with_trace(
        loaded,
        loaded,
        (episode_seed,),
    )
    episode = episodes[0]

    import jaxmarl
    from jaxmarl.environments.overcooked_v2.common import (
        Actions,
        DynamicObject,
        StaticObject,
    )
    from overcooked_v2_experiments.eval.policy import PolicyPairing
    from overcooked_v2_experiments.eval.rollout import get_rollout

    evaluation_kwargs = dict(restored_config["env"]["ENV_KWARGS"])
    evaluation_kwargs.pop("op_ingredient_permutations", None)
    evaluation_kwargs["max_steps"] = 400
    evaluation_environment = jaxmarl.make(
        restored_config["env"]["ENV_NAME"], **evaluation_kwargs
    )
    official_root_key = jax.random.PRNGKey(
        derive_ocv2_execution_seed(episode_seed)
    )
    official_rollout = get_rollout(
        PolicyPairing(official, official),
        evaluation_environment,
        official_root_key,
    )
    official_actions_0 = np.asarray(official_rollout.actions_seq["agent_0"])
    official_actions_1 = np.asarray(official_rollout.actions_seq["agent_1"])
    if not np.array_equal(official_actions_0, repository_trace["policy_0_actions"][0]) or (
        not np.array_equal(
            official_actions_1,
            repository_trace["policy_1_actions"][0],
        )
    ):
        raise RuntimeError("Official evaluator and repository rollout used different actions.")
    if not np.isclose(
        float(np.asarray(official_rollout.total_reward)),
        episode["raw_episode_return"],
    ):
        raise RuntimeError("Official evaluator and repository rollout returned different totals.")

    _, official_reset_key = jax.random.split(official_root_key, 2)
    _, initial_environment_state = evaluation_environment.reset(official_reset_key)
    previous_states = jax.tree_util.tree_map(
        lambda initial, sequence: jnp.concatenate(
            (jnp.asarray(initial)[jnp.newaxis, ...], sequence[:-1]), axis=0
        ),
        initial_environment_state,
        official_rollout.state_seq,
    )

    def official_events(state: Any, action_0: Any, action_1: Any):
        action_array = jnp.stack((action_0, action_1))
        forward = jax.vmap(lambda agent: agent.get_fwd_pos())(state.agents)
        cells = state.grid[forward.y, forward.x]
        interact = action_array == int(Actions.interact)
        deliveries = (
            interact
            & (cells[:, 0] == int(StaticObject.GOAL))
            & ((state.agents.inventory & int(DynamicObject.COOKED)) != 0)
        )
        activations = (
            interact
            & (cells[:, 0] == int(StaticObject.BUTTON_RECIPE_INDICATOR))
            & (state.agents.inventory == int(DynamicObject.EMPTY))
            & (cells[:, 1] == int(DynamicObject.EMPTY))
        )
        return jnp.sum(deliveries), jnp.sum(activations)

    official_deliveries, official_activations = jax.vmap(official_events)(
        previous_states,
        official_rollout.actions_seq["agent_0"],
        official_rollout.actions_seq["agent_1"],
    )
    official_correct = int(
        np.asarray(official_rollout.state_seq.new_correct_delivery).sum()
    )
    official_wrong = int(np.asarray(official_deliveries).sum()) - official_correct
    official_indicator = int(np.asarray(official_activations).sum())
    if (
        official_correct != episode["correct_delivery_count"]
        or official_wrong != episode["wrong_delivery_count"]
        or official_indicator != episode["indicator_activation_count"]
    ):
        raise RuntimeError("Official evaluator metric recomputation differs from evidence.")
    comparison = reference_configuration_comparison(
        config,
        accepted_reference_source=accepted_reference_source,
    )
    report = {
        "schema_version": "path_c_official_type_a_round_trip_v1",
        "scientific_readout_allowed": False,
        "launch_config_path": str(Path(launch_config_path).resolve()),
        "experiment_variant": launch["experiment_variant"],
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "checkpoint_path": str(checkpoint),
        "checkpoint_format": OFFICIAL_CHECKPOINT_FORMAT,
        "parameter_tree_path": list(OFFICIAL_PARAMETER_TREE_PATH),
        "model_weights_sha256_before_save": initial_hash,
        "model_weights_sha256_after_restore": restored_hash,
        "official_wrapper_action_equal": True,
        "official_wrapper_recurrent_state_equal": True,
        "official_wrapper_logits_equal": True,
        "official_full_episode_actions_equal": True,
        "official_full_episode_metrics_equal": True,
        "action_order": list(OFFICIAL_ACTION_ORDER),
        "episode_steps": 400,
        "episode_metrics": episode,
        "success_indicator_count": episode["correct_delivery_count"],
        "source_dependencies": official_source_dependency_records(
            str(launch["experiment_variant"])
        ),
        "reference_comparison": comparison,
        "elapsed_seconds": time.monotonic() - start,
    }
    report_path = output / "type_a_round_trip.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def measure_type_a_training_throughput(
    launch_config_path: str | Path,
    *,
    smoke_updates: int = 2,
) -> dict[str, Any]:
    """Measure compile-inclusive and compiled update cost without a formal run."""

    if (
        isinstance(smoke_updates, bool)
        or not isinstance(smoke_updates, int)
        or smoke_updates <= 0
    ):
        raise ValueError("Type-A throughput measurement requires positive updates.")
    import copy

    import jax
    import wandb

    _prepare_official_imports()
    from overcooked_v2_experiments.ppo.ippo import make_train
    from overcooked_v2_experiments.utils.utils import mini_batch_pmap

    launch = load_official_launch_config(launch_config_path)
    config = copy.deepcopy(compose_official_training_config(launch))
    config["NUM_CHECKPOINTS"] = 0
    train = make_train(config, update_step_num_overwrite=smoke_updates)
    mapped = mini_batch_pmap(jax.jit(train), 1)

    def execute(seed_offset: int) -> tuple[float, int]:
        keys = jax.random.split(
            jax.random.PRNGKey(int(launch["seed"]) + seed_offset), 1
        )
        started = time.monotonic()
        output = mapped(keys)
        jax.block_until_ready(output["metrics"]["env_step"])
        completed_environment_steps = int(
            np.asarray(output["metrics"]["env_step"]).max()
        )
        return time.monotonic() - started, completed_environment_steps

    with wandb.init(
        project="r015-type-a-wiring",
        mode="disabled",
        reinit=True,
    ):
        compile_inclusive_seconds, first_completed_steps = execute(10_000)
        compiled_seconds, second_completed_steps = execute(20_000)
    model = config["model"]
    smoke_environment_steps = (
        int(model["NUM_ENVS"]) * int(model["NUM_STEPS"]) * smoke_updates
    )
    if {first_completed_steps, second_completed_steps} != {smoke_environment_steps}:
        raise RuntimeError("Official training smoke did not report its executed steps.")
    effective_full_run_environment_steps = (
        int(model["TOTAL_TIMESTEPS"])
        // int(model["NUM_ENVS"])
        // int(model["NUM_STEPS"])
        * int(model["NUM_ENVS"])
        * int(model["NUM_STEPS"])
    )
    compiled_steps_per_gpu_hour = (
        smoke_environment_steps / compiled_seconds * 3600.0
    )
    # 成本继续按名义 3000 万步保守外推；有效步数身份另由上面的整数日程字段绑定。
    full_run_compiled_seconds = OFFICIAL_NOMINAL_TOTAL_TIMESTEPS / (
        smoke_environment_steps / compiled_seconds
    )
    conservative_full_run_seconds = compile_inclusive_seconds + full_run_compiled_seconds
    return {
        "schema_version": "path_c_official_type_a_throughput_v1",
        "scientific_readout_allowed": False,
        "experiment_variant": launch["experiment_variant"],
        "smoke_updates": smoke_updates,
        "smoke_environment_steps": smoke_environment_steps,
        "completed_environment_steps_per_execution": smoke_environment_steps,
        "measurement_execution_count": 2,
        "total_type_a_environment_steps": 2 * smoke_environment_steps,
        "compile_inclusive_seconds": compile_inclusive_seconds,
        "compiled_seconds": compiled_seconds,
        "compiled_environment_steps_per_gpu_hour": compiled_steps_per_gpu_hour,
        "conservative_full_run_gpu_hours": conservative_full_run_seconds / 3600.0,
        "registered_total_timesteps": int(model["TOTAL_TIMESTEPS"]),
        "effective_full_run_environment_steps": effective_full_run_environment_steps,
        "effective_step_contract_match": (
            effective_full_run_environment_steps
            == OFFICIAL_EFFECTIVE_STEP_CONTRACTS[str(launch["experiment_variant"])][
                "effective_environment_steps"
            ]
        ),
    }


def summarize_type_a_production_cost(
    sp_measurement: Mapping[str, Any],
    op_measurement: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the registered three-SP plus two-Other-Play cost arithmetic."""

    if sp_measurement.get("experiment_variant") != "rnn-sp" or (
        op_measurement.get("experiment_variant") != "rnn-op"
    ):
        raise ValueError("P2 cost summary requires one SP and one Other-Play measure.")
    predicted = 3.0 * float(sp_measurement["conservative_full_run_gpu_hours"]) + (
        2.0 * float(op_measurement["conservative_full_run_gpu_hours"])
    )
    return {
        "schema_version": "path_c_official_p2_cost_projection_v1",
        "scientific_readout_allowed": False,
        "nominal_total_timesteps": 150_000_000,
        "effective_environment_steps_contract": 149_848_064,
        "predicted_total_gpu_hours": predicted,
        "reference_acceptance_gpu_hours_ceiling": 0.6,
        "acceptance_plus_production_gpu_hours": predicted + 0.6,
        "acceptance_plus_production_within_hard_stop": predicted + 0.6 <= 6.0,
        "start_ceiling_gpu_hours": 5.4,
        "within_start_ceiling": predicted <= 5.4,
        "hard_stop_gpu_hours": 6.0,
    }


def family_id_for_variant(variant: str) -> str:
    if variant == "rnn-sp":
        return OFFICIAL_SP_FAMILY_ID
    if variant == "rnn-op":
        return OFFICIAL_OP_FAMILY_ID
    raise ValueError("Unsupported official experiment variant.")
