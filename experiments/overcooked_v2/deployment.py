"""Self-contained deployment bundle for unified DELTA-ZSC."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

from src.delta_zsc.config import (
    CHECKPOINT_SCHEMA_VERSION,
    METHOD_VERSION,
    run_config_from_mapping,
)
from src.delta_zsc.losses import categorical_log_probability
from src.delta_zsc.model import DeltaModel, observe_after_transition
from src.delta_zsc.storage import write_json


DEPLOYMENT_BUNDLE_VERSION = 2


@dataclass(frozen=True, slots=True)
class Deployment:
    ego_run_id: str
    config: Any
    model: DeltaModel
    base_params: Any
    latent_params: Any


def export_deployment_bundle(
    directory: str | Path,
    *,
    ego_run_id: str,
    config: Any,
    observation_shape: tuple[int, ...],
    base_params: Any,
    latent_params: Any,
    source_training_run: str | Path,
) -> Path:
    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Deployment directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    parameters = {"base_params": base_params, "latent_params": latent_params}
    params_path = root / "params.pkl"
    with params_path.open("wb") as handle:
        pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(
        root / "deployment_bundle.json",
        {
            "version": DEPLOYMENT_BUNDLE_VERSION,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "method": METHOD_VERSION,
            "ego_run_id": str(ego_run_id),
            "method_variant": config.method_variant,
            "config": config.to_mapping(),
            "observation_shape": [int(value) for value in observation_shape],
            "action_count": int(6),
            "params": params_path.name,
            "source_training_run": str(Path(source_training_run).resolve()),
        },
    )
    return root


def load_deployment(directory: str | Path) -> Deployment:
    root = Path(directory).resolve()
    payload_path = root / "deployment_bundle.json"
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    required = {
        "version",
        "checkpoint_schema_version",
        "method",
        "ego_run_id",
        "method_variant",
        "config",
        "observation_shape",
        "action_count",
        "params",
        "source_training_run",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Deployment bundle schema differs.")
    if (
        int(payload["version"]) != DEPLOYMENT_BUNDLE_VERSION
        or int(payload["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION
        or payload["method"] != METHOD_VERSION
    ):
        raise ValueError("Deployment method/schema identity differs.")
    config = run_config_from_mapping(payload["config"])
    if config.method_variant != payload["method_variant"]:
        raise ValueError("Deployment method variant differs.")
    params_path = root / str(payload["params"])
    with params_path.open("rb") as handle:
        parameters = pickle.load(handle)
    if set(parameters) != {"base_params", "latent_params"}:
        raise ValueError("Deployment parameter tree differs.")
    shape = tuple(int(value) for value in payload["observation_shape"])
    action_count = int(payload["action_count"])
    return Deployment(
        ego_run_id=str(payload["ego_run_id"]),
        config=config,
        model=DeltaModel(config, shape, action_count),
        base_params=parameters["base_params"],
        latent_params=parameters["latent_params"],
    )


def reset_deployment_state(deployment: Deployment, *, batch_size: int) -> Any:
    return deployment.model.initial_state(int(batch_size))


def deployment_action(
    *,
    deployment: Deployment,
    state: Any,
    observation: Any,
    keys: Any,
    force_base: bool = False,
) -> tuple[Any, Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    stepped, output = deployment.model.step(
        deployment.base_params,
        deployment.latent_params,
        state,
        observation,
    )
    logits = output.base_policy_logits if bool(force_base) else output.policy_logits
    key_array = jnp.asarray(keys)
    if key_array.ndim == 1:
        action = jax.random.categorical(key_array, logits)
    else:
        action = jax.vmap(lambda key, current: jax.random.categorical(key, current))(
            key_array, logits
        )
    log_probability = categorical_log_probability(logits, action)
    return stepped, action, output, log_probability


def observe_deployment_after_transition(
    state: Any, *, action: Any, done: Any
) -> Any:
    return observe_after_transition(state, action=action, done=done)


__all__ = [
    "Deployment",
    "deployment_action",
    "export_deployment_bundle",
    "load_deployment",
    "observe_deployment_after_transition",
    "reset_deployment_state",
]
