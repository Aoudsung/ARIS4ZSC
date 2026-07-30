"""Export, load, and execute frozen DELTA-ZSC deployment artifacts.

Training checkpoints contain training-only teacher encoders and the continuous
partner generator in ``TrainState``.  Confirmatory evaluation never loads those
objects.  Calibration exports a pruned artifact containing only the legal-history
model subtrees required by ``model.step`` plus the frozen conformal gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.path_c.belief_set_encoder import mixture_moments
from src.path_c.calibration import (
    calibration_from_mapping,
    calibration_to_mapping,
    hard_adaptation_gate,
    latent_support_score,
    predicted_policy_gain,
)
from src.path_c.experiment import METHOD_VERSION, RunConfig
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
    "belief_encoder",
    "belief_set_encoder",
    "universal_actor",
    "universal_critic",
    "response_decoder",
)
DEPLOYMENT_BUNDLE_VERSION = 1


@dataclass(frozen=True, slots=True)
class Deployment:
    ego_run_id: str
    config: RunConfig
    model: Any
    params: Any
    calibration: Any


def _build_model(config: RunConfig, observation_shape: tuple[int, ...], action_count: int) -> Any:
    return build_model(
        observation_shape=observation_shape,
        action_count=action_count,
        **{
            name: getattr(config.model, name)
            for name in (
                "task_hidden_dim",
                "belief_hidden_dim",
                "latent_dim",
                "mixture_components",
                "belief_embedding_dim",
                "actor_hidden_dim",
                "critic_hidden_dim",
                "response_hidden_dim",
                "modulation_rank",
                "action_embedding_dim",
                "log_variance_minimum",
                "log_variance_maximum",
                "response_log_std_minimum",
                "response_log_std_maximum",
            )
        },
    )


def deployable_parameters(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an exact whitelist; training-only subtrees cannot leak by accident."""

    missing = [name for name in DEPLOYABLE_PARAM_NAMES if name not in params]
    if missing:
        raise ValueError(f"Model parameters lack deployable subtrees: {missing}.")
    return {name: params[name] for name in DEPLOYABLE_PARAM_NAMES}


def _fingerprint_words(fingerprint: str) -> Any:
    import hashlib
    import numpy as np

    return np.frombuffer(
        hashlib.sha256(str(fingerprint).encode("utf-8")).digest()[:8],
        dtype=">u4",
    ).astype(np.uint32)


def load_training_model(run_directory: str | Path, config: RunConfig) -> Deployment:
    """Load the full training model for calibration/anchor collection only."""

    root = Path(run_directory).resolve()
    identity = read_run_identity(root)
    if identity.get("method") != METHOD_VERSION or identity.get("stage") != "train":
        raise ValueError("Run directory is not a DELTA-ZSC v5 training run.")
    if identity.get("config") != config.to_mapping():
        raise ValueError("Training model config differs from calibration config.")
    manager = orbax_manager(root / "checkpoints", create=False)
    restored = restore_latest_checkpoint(manager)
    if restored is None:
        raise FileNotFoundError(f"No checkpoint in {root / 'checkpoints'}")
    unused_step, state = restored
    del unused_step
    values = state if isinstance(state, Mapping) else state._asdict()
    params = values["params"]
    observation_shape = tuple(int(value) for value in identity["observation_shape"])
    action_count = int(identity.get("action_count", 6))
    return Deployment(
        ego_run_id=str(identity.get("ego_run_id", root.name)),
        config=config,
        model=_build_model(config, observation_shape, action_count),
        params=params,
        calibration=values["calibration"],
    )


def export_deployment_bundle(
    directory: str | Path,
    *,
    source_training_run: str | Path,
    deployment: Deployment,
    calibration: Any,
) -> Path:
    """Write a self-contained deployment artifact without teacher/generator state."""

    import orbax.checkpoint as ocp

    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Deployment bundle directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    pruned = deployable_parameters(deployment.params)
    params_path = root / "params"
    ocp.PyTreeCheckpointer().save(str(params_path), pruned, force=True)
    identity = read_run_identity(source_training_run)
    payload = {
        "version": DEPLOYMENT_BUNDLE_VERSION,
        "method": METHOD_VERSION,
        "ego_run_id": deployment.ego_run_id,
        "config": deployment.config.to_mapping(),
        "config_fingerprint": deployment.config.fingerprint,
        "observation_shape": list(identity["observation_shape"]),
        "action_count": int(identity.get("action_count", 6)),
        "deployable_param_names": list(DEPLOYABLE_PARAM_NAMES),
        "params_fingerprint": pytree_fingerprint(pruned),
        "source_training_run": str(Path(source_training_run).resolve()),
        "calibration": calibration_to_mapping(calibration),
    }
    write_json(root / "deployment_bundle.json", payload)
    return root


def load_deployment(bundle_directory: str | Path, config: RunConfig) -> Deployment:
    """Load only the pruned confirmatory deployment artifact."""

    import orbax.checkpoint as ocp
    import numpy as np

    root = Path(bundle_directory).resolve()
    payload_path = root / "deployment_bundle.json"
    if not payload_path.is_file():
        raise FileNotFoundError(
            f"Pruned deployment bundle is missing: {payload_path}. Run calibration first."
        )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    required = {
        "version",
        "method",
        "ego_run_id",
        "config",
        "config_fingerprint",
        "observation_shape",
        "action_count",
        "deployable_param_names",
        "params_fingerprint",
        "source_training_run",
        "calibration",
    }
    if set(payload) != required:
        raise ValueError(
            "Deployment bundle fields differ: "
            f"missing={sorted(required-set(payload))}, "
            f"unknown={sorted(set(payload)-required)}."
        )
    if int(payload["version"]) != DEPLOYMENT_BUNDLE_VERSION:
        raise ValueError("Unknown deployment bundle version.")
    if payload["method"] != METHOD_VERSION:
        raise ValueError("Deployment bundle belongs to another method.")
    if payload["config"] != config.to_mapping() or payload["config_fingerprint"] != config.fingerprint:
        raise ValueError("Deployment bundle config differs from evaluation config.")
    if tuple(payload["deployable_param_names"]) != DEPLOYABLE_PARAM_NAMES:
        raise ValueError("Deployment parameter whitelist differs from the active method.")
    params = ocp.PyTreeCheckpointer().restore(str(root / "params"))
    if set(params) != set(DEPLOYABLE_PARAM_NAMES):
        raise ValueError("Deployment artifact contains missing or training-only parameter subtrees.")
    if pytree_fingerprint(params) != payload["params_fingerprint"]:
        raise ValueError("Deployment parameter fingerprint differs from bundle metadata.")
    calibration = calibration_from_mapping(payload["calibration"])
    expected = _fingerprint_words(payload["params_fingerprint"])
    observed = np.asarray(calibration.model_fingerprint, dtype=np.uint32)
    if observed.shape != (2,) or not np.array_equal(observed, expected):
        raise ValueError("Calibration artifact belongs to a different deployment parameter tree.")
    observation_shape = tuple(int(value) for value in payload["observation_shape"])
    action_count = int(payload["action_count"])
    return Deployment(
        ego_run_id=str(payload["ego_run_id"]),
        config=config,
        model=_build_model(config, observation_shape, action_count),
        params=params,
        calibration=calibration,
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
        belief_hidden_dim=deployment.config.model.belief_hidden_dim,
        latent_dim=deployment.config.model.latent_dim,
        mixture_components=deployment.config.model.mixture_components,
    )


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

    stepped, provisional = deployment.model.apply(
        {"params": deployment.params},
        state,
        observation,
        jnp.ones(state.previous_action.shape, dtype=jnp.float32),
        method=deployment.model.step,
    )
    conditional_logits = provisional.base_logits + provisional.residual_logits
    gain = predicted_policy_gain(
        provisional.action_values,
        provisional.base_logits,
        conditional_logits,
    )
    posterior_mean, unused_variance = mixture_moments(
        provisional.mixture_logits,
        provisional.mixture_means,
        provisional.mixture_log_variances,
    )
    del unused_variance
    support = latent_support_score(posterior_mean, deployment.calibration)
    gate = (
        hard_adaptation_gate(gain, support, deployment.calibration)
        if deployment.config.calibration.enable_hard_gate_at_evaluation
        else jnp.ones_like(gain, dtype=jnp.float32)
    )
    if force_base:
        gate = jnp.zeros_like(gate)
    execution_logits = provisional.base_logits + gate[..., None] * provisional.residual_logits
    key_array = jnp.asarray(keys)
    action = (
        jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(
            key_array, execution_logits
        )
        if key_array.ndim == 2
        else jax.random.categorical(key_array, execution_logits)
    )
    log_probability = categorical_log_probability(execution_logits, action)
    output = provisional._replace(
        gate=gate,
        support_score=support,
        execution_logits=execution_logits,
    )
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
    "Deployment",
    "deployable_parameters",
    "deployment_action",
    "export_deployment_bundle",
    "load_deployment",
    "load_training_model",
    "reset_deployment_state",
    "update_after_transition",
]
