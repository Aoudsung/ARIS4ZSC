"""Strict CETR-ZSC deployment bundle and single execution path."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Mapping

from src.cetr_zsc.config import (
    CHECKPOINT_SCHEMA_VERSION,
    METHOD_VERSION,
    OFFICIAL_ACTION_COUNT,
    run_config_from_mapping,
)
from src.cetr_zsc.model import CetrModel
from src.cetr_zsc.storage import read_json, write_json
from src.cetr_zsc.types import PolicyState


DEPLOYMENT_BUNDLE_VERSION = 6
_REFERENCE_FIELDS = {
    "artifact_type",
    "version",
    "layout",
    "seed_index",
    "tau_sp",
    "episodes_per_pairing",
    "evaluation_root_seed",
    "source_checkpoint",
}
_BUNDLE_FIELDS = {
    "version",
    "checkpoint_schema_version",
    "method",
    "ego_run_id",
    "config",
    "observation_shape",
    "action_count",
    "params",
    "source_training_run",
    "reference_sp_artifact",
    "training_parent_manifest",
}


@dataclass(frozen=True, slots=True)
class Deployment:
    ego_run_id: str
    config: Any
    model: CetrModel
    params: Any
    reference_sp_artifact: Mapping[str, Any]


def _reference_artifact(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    payload = read_json(value) if isinstance(value, (str, Path)) else value
    if not isinstance(payload, Mapping) or set(payload) != _REFERENCE_FIELDS:
        raise ValueError("Reference-SP artifact schema differs.")
    if payload.get("artifact_type") != "cetr_reference_sp" or int(payload["version"]) != 1:
        raise ValueError("Reference-SP artifact identity differs.")
    if not isinstance(payload.get("layout"), str):
        raise ValueError("Reference-SP artifact layout is missing.")
    if not isinstance(payload.get("source_checkpoint"), str):
        raise ValueError("Reference-SP artifact source checkpoint is missing.")
    if not isinstance(payload.get("tau_sp"), (int, float)):
        raise ValueError("Reference-SP artifact tau is not numeric.")
    return dict(payload)


def _training_parent_manifest(source_training_run: Path) -> Mapping[str, Any]:
    for name in ("partner_manifest.json", "training_parent_manifest.json"):
        candidate = source_training_run / name
        if candidate.is_file():
            payload = read_json(candidate)
            if not isinstance(payload, Mapping):
                raise ValueError("Training parent manifest is not a JSON object.")
            return {"path": str(candidate), "manifest": dict(payload)}
    return {"path": str(source_training_run / "partner_manifest.json")}


def export_deployment_bundle(
    directory: str | Path,
    *,
    ego_run_id: str,
    config: Any,
    observation_shape: tuple[int, ...],
    params: Any,
    source_training_run: str | Path,
    reference_sp_artifact: Mapping[str, Any] | str | Path,
) -> Path:
    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Deployment directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    reference = _reference_artifact(reference_sp_artifact)
    source = Path(source_training_run).resolve()
    params_path = root / "params.pkl"
    with params_path.open("wb") as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(
        root / "deployment_bundle.json",
        {
            "version": DEPLOYMENT_BUNDLE_VERSION,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "method": METHOD_VERSION,
            "ego_run_id": str(ego_run_id),
            "config": config.to_mapping(),
            "observation_shape": [int(value) for value in observation_shape],
            "action_count": int(OFFICIAL_ACTION_COUNT),
            "params": params_path.name,
            "source_training_run": str(source),
            "reference_sp_artifact": reference,
            "training_parent_manifest": _training_parent_manifest(source),
        },
    )
    return root


def load_deployment(directory: str | Path) -> Deployment:
    root = Path(directory).resolve()
    payload_path = root / "deployment_bundle.json"
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = read_json(payload_path)
    if not isinstance(payload, Mapping) or set(payload) != _BUNDLE_FIELDS:
        raise ValueError("Deployment bundle schema differs.")
    if (
        int(payload["version"]) != DEPLOYMENT_BUNDLE_VERSION
        or int(payload["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION
        or payload["method"] != METHOD_VERSION
    ):
        raise ValueError("Deployment method/schema identity differs.")
    config = run_config_from_mapping(payload["config"])
    if str(payload["ego_run_id"]) == "" or str(payload["source_training_run"]) == "":
        raise ValueError("Deployment run binding is empty.")
    reference = _reference_artifact(payload["reference_sp_artifact"])
    if str(reference["layout"]) != str(config.environment.layout):
        raise ValueError("Deployment reference-SP layout differs from the config.")
    shape_value = payload["observation_shape"]
    if (
        not isinstance(shape_value, list)
        or not shape_value
        or any(int(value) <= 0 for value in shape_value)
    ):
        raise ValueError("Deployment observation shape differs.")
    shape = tuple(int(value) for value in shape_value)
    if int(payload["action_count"]) != OFFICIAL_ACTION_COUNT:
        raise ValueError("Deployment action count differs.")
    parent_manifest = payload["training_parent_manifest"]
    if (
        not isinstance(parent_manifest, Mapping)
        or set(parent_manifest) not in ({"path"}, {"path", "manifest"})
        or not str(parent_manifest["path"])
    ):
        raise ValueError("Deployment parent-manifest binding differs.")
    params_name = str(payload["params"])
    params_path = root / params_name
    if params_name != Path(params_name).name or not params_path.is_file():
        raise ValueError("Deployment parameter binding differs.")
    with params_path.open("rb") as handle:
        params = pickle.load(handle)
    model = CetrModel(config, shape, OFFICIAL_ACTION_COUNT)
    return Deployment(
        ego_run_id=str(payload["ego_run_id"]),
        config=config,
        model=model,
        params=params,
        reference_sp_artifact=reference,
    )


def reset_deployment_state(deployment: Deployment, batch_size: int) -> PolicyState:
    import jax
    import jax.numpy as jnp

    size = int(batch_size)
    carry = jax.tree_util.tree_map(
        lambda value: jnp.zeros_like(value), deployment.model.initial_carry(size)
    )
    return PolicyState(
        carry=carry,
        episode_start=jnp.ones((size,), dtype=jnp.bool_),
    )


def deployment_action(
    deployment: Deployment,
    state: PolicyState,
    observation: Any,
    keys: Any,
) -> tuple[PolicyState, Any, Any]:
    import jax
    import jax.numpy as jnp

    observations = jnp.asarray(observation, dtype=jnp.float32)
    if observations.ndim == len(deployment.model.observation_shape):
        observations = observations[None, ...]
    next_carry, logits, unused_value = deployment.model.step(
        deployment.params,
        state.carry,
        observations,
        state.episode_start,
    )
    del unused_value
    logits = jnp.asarray(logits)
    key_array = jnp.asarray(keys)
    if key_array.ndim == 1:
        key_array = jax.random.split(key_array, int(observations.shape[0]))
    actions = jax.vmap(
        lambda key, current: jax.random.categorical(key, current)
    )(key_array, logits)
    log_probability = jax.nn.log_softmax(logits, axis=-1)[
        jnp.arange(actions.shape[0]), actions
    ]
    next_state = PolicyState(
        carry=next_carry,
        episode_start=jnp.zeros_like(state.episode_start, dtype=jnp.bool_),
    )
    return next_state, actions, log_probability


__all__ = [
    "DEPLOYMENT_BUNDLE_VERSION",
    "Deployment",
    "deployment_action",
    "export_deployment_bundle",
    "load_deployment",
    "reset_deployment_state",
]
