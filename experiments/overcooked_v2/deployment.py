"""Export and execute the primary DELTA-ZSC V6 E2E policy.

The primary artifact always contains the final end-to-end parameter tree.  It
has no deployment tier, owner-policy fallback, residual branch, or hard gate.
The optional safety wrapper is a separate artifact and command.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.path_c.experiment import CONFIG_VERSION, METHOD_VERSION, RunConfig
from src.path_c.model import build_model, initial_policy_state
from src.path_c.runner import observe_policy_after_transition
from src.path_c.storage import (
    orbax_manager,
    pytree_fingerprint,
    read_run_identity,
    restore_latest_checkpoint,
    write_json,
)
from src.path_c.training import categorical_log_probability


DEPLOYABLE_PARAM_NAMES = (
    "task_encoder",
    "capability_encoder",
    "protocol_encoder",
    "protocol_component_embeddings",
    "universal_actor",
)
DEPLOYMENT_BUNDLE_VERSION = 3
PRIMARY_ARTIFACT_NAME = "DELTA-ZSC-E2E"


@dataclass(frozen=True, slots=True)
class Deployment:
    ego_run_id: str
    config: RunConfig
    model: Any
    params: Any
    artifact_name: str = PRIMARY_ARTIFACT_NAME
    safety: Any | None = None


def _build_model(
    config: RunConfig, observation_shape: tuple[int, ...], action_count: int
) -> Any:
    return build_model(
        observation_shape=observation_shape,
        action_count=action_count,
        task_hidden_dim=config.model.task_hidden_dim,
        capability_hidden_dim=config.model.capability_hidden_dim,
        protocol_hidden_dim=config.model.protocol_hidden_dim,
        capability_dim=config.model.capability_dim,
        protocol_components=config.model.protocol_components,
        component_embedding_dim=config.model.component_embedding_dim,
        actor_hidden_dim=config.model.actor_hidden_dim,
        critic_hidden_dim=config.model.critic_hidden_dim,
        response_hidden_dim=config.model.response_hidden_dim,
        modulation_rank=config.model.modulation_rank,
        action_embedding_dim=config.model.action_embedding_dim,
    )


def deployable_parameters(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Apply the exact V6 deployment whitelist."""

    missing = [name for name in DEPLOYABLE_PARAM_NAMES if name not in params]
    if missing:
        raise ValueError(f"Model parameters lack deployable subtrees: {missing}.")
    return {name: params[name] for name in DEPLOYABLE_PARAM_NAMES}


def _checkpoint_values(state: Any) -> Mapping[str, Any]:
    if isinstance(state, Mapping):
        return state
    if hasattr(state, "_asdict"):
        return state._asdict()
    raise TypeError("Training checkpoint is not a named or mapping V6 state.")


def _validate_training_identity(identity: Mapping[str, Any], config: RunConfig) -> None:
    if identity.get("method") != METHOD_VERSION or identity.get("stage") != "train":
        raise ValueError("Run directory is not a DELTA-ZSC V6 training run.")
    if int(identity.get("config", {}).get("version", -1)) != CONFIG_VERSION:
        raise ValueError("V5 checkpoints and identities cannot be loaded by V6.")
    if identity.get("config") != config.to_mapping():
        raise ValueError("Training model config differs from the requested config.")


def load_training_model(run_directory: str | Path, config: RunConfig) -> Deployment:
    """Load the current final V6 model for export or optional offline audit."""

    root = Path(run_directory).resolve()
    identity = read_run_identity(root)
    _validate_training_identity(identity, config)
    manager = orbax_manager(root / "checkpoints", create=False)
    restored = restore_latest_checkpoint(manager)
    if restored is None:
        raise FileNotFoundError(f"No checkpoint in {root / 'checkpoints'}")
    unused_step, state = restored
    del unused_step
    values = _checkpoint_values(state)
    required = {
        "params",
        "target_params",
        "belief_optimizer_state",
        "generator_params",
        "anchor_replay",
        "update_count",
    }
    if not required.issubset(values):
        raise ValueError("Checkpoint lacks V6 E2E state fields; V5 restore is forbidden.")
    observation_shape = tuple(int(value) for value in identity["observation_shape"])
    action_count = int(identity.get("action_count", 6))
    return Deployment(
        ego_run_id=str(identity.get("ego_run_id", root.name)),
        config=config,
        model=_build_model(config, observation_shape, action_count),
        params=values["params"],
    )


def export_deployment_bundle(
    directory: str | Path,
    *,
    source_training_run: str | Path,
    deployment: Deployment,
) -> Path:
    """Write the final V6 actor artifact; no fallback selection is permitted."""

    import orbax.checkpoint as ocp

    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Deployment bundle directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    identity = read_run_identity(source_training_run)
    _validate_training_identity(identity, deployment.config)
    pruned = deployable_parameters(deployment.params)
    params_path = root / "params"
    ocp.PyTreeCheckpointer().save(str(params_path), pruned, force=True)
    payload = {
        "version": DEPLOYMENT_BUNDLE_VERSION,
        "method": METHOD_VERSION,
        "artifact_name": PRIMARY_ARTIFACT_NAME,
        "ego_run_id": deployment.ego_run_id,
        "config": deployment.config.to_mapping(),
        "config_fingerprint": deployment.config.fingerprint,
        "observation_shape": list(identity["observation_shape"]),
        "action_count": int(identity.get("action_count", 6)),
        "deployable_param_names": list(DEPLOYABLE_PARAM_NAMES),
        "params_fingerprint": pytree_fingerprint(pruned),
        "source_training_run": str(Path(source_training_run).resolve()),
        "safety_wrapper": False,
    }
    write_json(root / "deployment_bundle.json", payload)
    return root


def load_deployment(bundle_directory: str | Path, config: RunConfig) -> Deployment:
    """Load only an exact primary V6 deployment bundle."""

    import orbax.checkpoint as ocp

    root = Path(bundle_directory).resolve()
    payload_path = root / "deployment_bundle.json"
    if not payload_path.is_file():
        raise FileNotFoundError(f"V6 deployment bundle is missing: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    required = {
        "version",
        "method",
        "artifact_name",
        "ego_run_id",
        "config",
        "config_fingerprint",
        "observation_shape",
        "action_count",
        "deployable_param_names",
        "params_fingerprint",
        "source_training_run",
        "safety_wrapper",
    }
    if set(payload) != required:
        raise ValueError("Deployment bundle is not the exact V6 primary schema.")
    if (
        int(payload["version"]) != DEPLOYMENT_BUNDLE_VERSION
        or payload["method"] != METHOD_VERSION
        or payload["artifact_name"] != PRIMARY_ARTIFACT_NAME
        or bool(payload["safety_wrapper"])
    ):
        raise ValueError("V5, tier, fallback, and safety artifacts are not primary V6.")
    if (
        payload["config"] != config.to_mapping()
        or payload["config_fingerprint"] != config.fingerprint
    ):
        raise ValueError("Deployment config differs from evaluation config.")
    if tuple(payload["deployable_param_names"]) != DEPLOYABLE_PARAM_NAMES:
        raise ValueError("Deployment parameter whitelist differs from V6.")
    params = ocp.PyTreeCheckpointer().restore(str(root / "params"))
    if set(params) != set(DEPLOYABLE_PARAM_NAMES):
        raise ValueError("Deployment contains training-only or missing parameter subtrees.")
    if pytree_fingerprint(params) != payload["params_fingerprint"]:
        raise ValueError("Deployment parameter fingerprint differs from metadata.")
    shape = tuple(int(value) for value in payload["observation_shape"])
    action_count = int(payload["action_count"])
    return Deployment(
        ego_run_id=str(payload["ego_run_id"]),
        config=config,
        model=_build_model(config, shape, action_count),
        params=params,
    )


def reset_deployment_state(
    deployment: Deployment,
    *,
    batch_size: int,
    observation_shape: tuple[int, ...],
) -> Any:
    return initial_policy_state(
        batch_size=batch_size,
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=deployment.config.model.task_hidden_dim,
        capability_hidden_dim=deployment.config.model.capability_hidden_dim,
        protocol_hidden_dim=deployment.config.model.protocol_hidden_dim,
        capability_dim=deployment.config.model.capability_dim,
        component_embedding_dim=deployment.config.model.component_embedding_dim,
        protocol_components=deployment.config.model.protocol_components,
    )


def deployment_action(
    *,
    deployment: Deployment,
    state: Any,
    observation: Any,
    keys: Any,
) -> tuple[Any, Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    batch_size = int(jnp.asarray(observation).shape[0])
    stepped, output = deployment.model.apply(
        {"params": deployment.params},
        state,
        observation,
        jnp.zeros((batch_size,), dtype=jnp.bool_),
        method=deployment.model.step,
    )
    key_array = jnp.asarray(keys)
    action = (
        jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(
            key_array, output.policy_logits
        )
        if key_array.ndim == 2
        else jax.random.categorical(key_array, output.policy_logits)
    )
    log_probability = categorical_log_probability(output.policy_logits, action)
    return stepped, action, output, log_probability


def update_after_transition(
    *,
    deployment: Deployment,
    stepped_state: Any,
    action: Any,
    reward: Any,
    done: Any,
    next_observation: Any,
) -> Any:
    return observe_policy_after_transition(
        stepped_state=stepped_state,
        action=action,
        reward=reward,
        done=done,
        next_observation=next_observation,
        model_config=deployment.config.model,
    )


__all__ = [
    "DEPLOYABLE_PARAM_NAMES",
    "DEPLOYMENT_BUNDLE_VERSION",
    "PRIMARY_ARTIFACT_NAME",
    "Deployment",
    "deployable_parameters",
    "deployment_action",
    "export_deployment_bundle",
    "load_deployment",
    "load_training_model",
    "reset_deployment_state",
    "update_after_transition",
]
