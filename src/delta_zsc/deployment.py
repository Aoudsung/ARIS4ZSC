"""Self-contained deployment bundle for unified DELTA-ZSC."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .config import (
    ArchitectureConfig,
    CHECKPOINT_SCHEMA_VERSION,
    CONFIG_VERSION,
    DEPLOYMENT_SCHEMA_VERSION,
    DataConfig,
    EnvironmentConfig,
    EvaluationConfig,
    LatentOptimizerConfig,
    METHOD_VERSION,
    MethodConfig,
    PPOConfig,
    UnifiedConfig,
    validate_config,
)
from .model import (
    METHOD_VARIANTS,
    UnifiedAgent,
    build_models,
    initial_agent_state,
)
from .runner import deployment_action, update_agent_after_transition


@dataclass(frozen=True, slots=True)
class Deployment:
    config: UnifiedConfig
    variant: str
    observation_shape: tuple[int, ...]
    action_count: int
    base_model: Any
    latent_model: Any
    agent: UnifiedAgent
    base_params: Any
    latent_params: Any


def config_from_mapping(value: Mapping[str, Any]) -> UnifiedConfig:
    if not isinstance(value, Mapping):
        raise ValueError("Deployment config must be a mapping.")
    config = UnifiedConfig(
        version=int(value["version"]),
        environment=EnvironmentConfig(**value["environment"]),
        method=MethodConfig(**value["method"]),
        architecture=ArchitectureConfig(**value["architecture"]),
        ppo=PPOConfig(**value["ppo"]),
        latent_optimizer=LatentOptimizerConfig(**value["latent_optimizer"]),
        data=DataConfig(**value["data"]),
        evaluation=EvaluationConfig(**value["evaluation"]),
    )
    validate_config(config)
    return config


def parameter_fingerprint(tree: Any) -> str:
    import hashlib
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def export_deployment(
    directory: str | Path,
    *,
    config: UnifiedConfig,
    variant: str,
    observation_shape: tuple[int, ...],
    action_count: int,
    base_params: Any,
    latent_params: Any,
    source_training_run: str | Path,
) -> Path:
    import orbax.checkpoint as ocp

    normalized = str(variant).lower()
    if normalized not in METHOD_VARIANTS:
        raise ValueError("Deployment variant is not registered.")
    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Deployment directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    parameter_tree = {
        "base_params": base_params,
        "latent_params": latent_params,
    }
    ocp.PyTreeCheckpointer().save(str(root / "params"), parameter_tree, force=True)
    payload = {
        "version": DEPLOYMENT_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "method": METHOD_VERSION,
        "variant": normalized,
        "config": config.to_mapping(),
        "config_fingerprint": config.fingerprint,
        "observation_shape": [int(value) for value in observation_shape],
        "action_count": int(action_count),
        "base_parameter_fingerprint": parameter_fingerprint(base_params),
        "latent_parameter_fingerprint": parameter_fingerprint(latent_params),
        "parameter_fingerprint": parameter_fingerprint(parameter_tree),
        "source_training_run": str(Path(source_training_run).resolve()),
    }
    (root / "deployment_bundle.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def load_deployment(
    directory: str | Path,
    *,
    expected_config: UnifiedConfig | None = None,
    expected_variant: str | None = None,
) -> Deployment:
    import orbax.checkpoint as ocp

    root = Path(directory).resolve()
    payload_path = root / "deployment_bundle.json"
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    required = {
        "version",
        "checkpoint_schema_version",
        "config_version",
        "method",
        "variant",
        "config",
        "config_fingerprint",
        "observation_shape",
        "action_count",
        "base_parameter_fingerprint",
        "latent_parameter_fingerprint",
        "parameter_fingerprint",
        "source_training_run",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("Deployment bundle schema differs.")
    if (
        int(payload["version"]) != DEPLOYMENT_SCHEMA_VERSION
        or int(payload["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION
        or int(payload["config_version"]) != CONFIG_VERSION
        or payload["method"] != METHOD_VERSION
    ):
        raise ValueError("Deployment identity is stale or incompatible.")
    config = config_from_mapping(payload["config"])
    if config.fingerprint != payload["config_fingerprint"]:
        raise ValueError("Deployment config fingerprint differs.")
    if expected_config is not None and expected_config != config:
        raise ValueError("Deployment config differs from the requested config.")
    variant = str(payload["variant"])
    if variant not in METHOD_VARIANTS:
        raise ValueError("Deployment variant is not registered.")
    if expected_variant is not None and variant != str(expected_variant).lower():
        raise ValueError("Deployment variant differs from the requested variant.")
    params = ocp.PyTreeCheckpointer().restore(str(root / "params"))
    if set(params) != {"base_params", "latent_params"}:
        raise ValueError("Deployment parameter tree differs.")
    if parameter_fingerprint(params) != payload["parameter_fingerprint"]:
        raise ValueError("Deployment parameters changed after export.")
    if (
        parameter_fingerprint(params["base_params"])
        != payload["base_parameter_fingerprint"]
        or parameter_fingerprint(params["latent_params"])
        != payload["latent_parameter_fingerprint"]
    ):
        raise ValueError("Deployment base/latent parameter identity differs.")
    observation_shape = tuple(int(value) for value in payload["observation_shape"])
    action_count = int(payload["action_count"])
    base_model, latent_model = build_models(
        observation_shape=observation_shape,
        action_count=action_count,
        task_hidden_dim=config.architecture.task_hidden_dim,
        instant_partner_dim=config.architecture.instant_partner_dim,
        latent_hidden_dim=config.architecture.latent_hidden_dim,
        response_hidden_dim=config.architecture.response_hidden_dim,
        action_embedding_dim=config.architecture.action_embedding_dim,
        component_count=config.method.latent_components,
    )
    agent = UnifiedAgent(
        base_model=base_model,
        latent_model=latent_model,
        action_count=action_count,
        component_count=config.method.latent_components,
        adaptation_kl_budget=config.method.adaptation_kl_budget,
        gamma=config.ppo.gamma,
    )
    return Deployment(
        config=config,
        variant=variant,
        observation_shape=observation_shape,
        action_count=action_count,
        base_model=base_model,
        latent_model=latent_model,
        agent=agent,
        base_params=params["base_params"],
        latent_params=params["latent_params"],
    )


def reset_state(deployment: Deployment, *, batch_size: int) -> Any:
    return initial_agent_state(
        batch_size=int(batch_size),
        observation_shape=deployment.observation_shape,
        task_hidden_dim=deployment.config.architecture.task_hidden_dim,
        component_count=deployment.config.method.latent_components,
    )


def act(
    deployment: Deployment,
    *,
    state: Any,
    observation: Any,
    keys: Any,
) -> tuple[Any, Any, Any, Any]:
    return deployment_action(
        agent=deployment.agent,
        base_params=deployment.base_params,
        latent_params=deployment.latent_params,
        state=state,
        observation=observation,
        keys=keys,
        variant=deployment.variant,
    )


def observe_transition(
    deployment: Deployment,
    *,
    stepped_state: Any,
    action: Any,
    done: Any,
    next_observation: Any,
) -> Any:
    return update_agent_after_transition(
        state=stepped_state,
        action=action,
        done=done,
        next_observation=next_observation,
        task_hidden_dim=deployment.config.architecture.task_hidden_dim,
        component_count=deployment.config.method.latent_components,
    )


__all__ = [
    "Deployment",
    "act",
    "config_from_mapping",
    "export_deployment",
    "load_deployment",
    "observe_transition",
    "parameter_fingerprint",
    "reset_state",
]
