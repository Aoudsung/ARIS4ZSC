"""Self-contained deployment bundle for unified DELTA-ZSC."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

from src.delta_zsc.config import (
    CHECKPOINT_SCHEMA_VERSION,
    METHOD_VERSION,
    OFFICIAL_ACTION_COUNT,
    run_config_from_mapping,
)
from src.delta_zsc.losses import categorical_log_probability
from src.delta_zsc.model import DeltaModel, observe_after_transition
from src.delta_zsc.storage import write_json
from src.delta_zsc.semantic_initializer import SEMANTIC_INITIALIZER_SCHEMA_VERSION


DEPLOYMENT_BUNDLE_VERSION = 5
EXECUTION_MODES = ("reference_only", "residual", "passive", "active")


@dataclass(frozen=True, slots=True)
class Deployment:
    ego_run_id: str
    config: Any
    model: DeltaModel
    base_params: Any
    latent_params: Any
    execution_mode: str
    semantic_initializer: Mapping[str, Any]


def export_deployment_bundle(
    directory: str | Path,
    *,
    ego_run_id: str,
    config: Any,
    observation_shape: tuple[int, ...],
    base_params: Any,
    latent_params: Any,
    source_training_run: str | Path,
    semantic_initializer: Mapping[str, Any],
    execution_mode: str = "active",
) -> Path:
    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Deployment directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"execution_mode must be one of {EXECUTION_MODES}.")
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
            "execution_mode": str(execution_mode),
            "config": config.to_mapping(),
            "observation_shape": [int(value) for value in observation_shape],
            "action_count": int(OFFICIAL_ACTION_COUNT),
            "params": params_path.name,
            "source_training_run": str(Path(source_training_run).resolve()),
            "semantic_initializer": dict(semantic_initializer),
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
        "execution_mode",
        "config",
        "observation_shape",
        "action_count",
        "params",
        "source_training_run",
        "semantic_initializer",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Deployment bundle schema differs.")
    if (
        int(payload["version"]) != DEPLOYMENT_BUNDLE_VERSION
        or int(payload["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION
        or payload["method"] != METHOD_VERSION
    ):
        raise ValueError("Deployment method/schema identity differs.")
    initializer = payload["semantic_initializer"]
    if (
        not isinstance(initializer, dict)
        or initializer.get("artifact_type")
        != "delta_semantic_component_initializer"
        or int(initializer.get("version", -1)) != SEMANTIC_INITIALIZER_SCHEMA_VERSION
        or initializer.get("uses_partner_labels") is not False
    ):
        raise ValueError("Deployment semantic-initializer identity differs.")
    config = run_config_from_mapping(payload["config"])
    if config.method_variant != payload["method_variant"]:
        raise ValueError("Deployment method variant differs.")
    if payload["execution_mode"] not in EXECUTION_MODES:
        raise ValueError("Deployment execution mode differs.")
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
        execution_mode=str(payload["execution_mode"]),
        semantic_initializer=dict(initializer),
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
    execution_mode: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    mode = (
        "residual"
        if bool(force_base)
        else str(deployment.execution_mode if execution_mode is None else execution_mode)
    )
    if mode not in EXECUTION_MODES:
        raise ValueError(f"execution_mode must be one of {EXECUTION_MODES}.")
    # Passive is computed below from a non-adapting model step.  Letting an
    # active DELTA model adapt here and merely replacing its logits would still
    # mutate ``probe_continuation_pending``, contaminating the passive rollout.
    execute_adaptation = mode == "active"
    stepped, output = deployment.model.step(
        deployment.base_params,
        deployment.latent_params,
        state,
        observation,
        execute_adaptation=execute_adaptation,
    )
    if mode == "reference_only":
        logits = output.reference_policy_logits
    elif mode == "residual":
        logits = output.base_policy_logits
    elif mode == "passive":
        from src.delta_zsc.mirror_policy import project_policy_logits, robust_mirror_policy_logits
        from src.delta_zsc.mirror_policy import MIRROR_UNCERTAINTY_PENALTY

        mirror, _, _ = robust_mirror_policy_logits(
            output.base_policy_logits,
            output.expected_decision_values,
            output.expected_decision_variances ** 0.5,
            kl_budget=deployment.config.method.adaptation_kl_budget,
            uncertainty_penalty=MIRROR_UNCERTAINTY_PENALTY,
        )
        logits, _, _ = project_policy_logits(
            output.reference_policy_logits,
            mirror,
            kl_budget=deployment.config.method.adaptation_kl_budget,
        )
    else:
        logits = output.policy_logits
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
    "EXECUTION_MODES",
    "deployment_action",
    "export_deployment_bundle",
    "load_deployment",
    "observe_deployment_after_transition",
    "reset_deployment_state",
]
