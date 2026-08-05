from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("orbax.checkpoint")

from experiments.overcooked_v2.deployment import (
    DEPLOYABLE_PARAM_NAMES,
    Deployment,
    PRIMARY_ARTIFACT_NAME,
    deployment_action,
    deployable_parameters,
    export_deployment_bundle,
    load_deployment,
    reset_deployment_state,
)
from src.path_c.experiment import METHOD_VERSION, load_config
from src.path_c.model import build_model, initialize_model_parameters
from src.path_c.storage import (
    ensure_run_identity,
    pytree_fingerprint,
    read_run_identity,
    sha256_path,
)


def test_sha256_path_binds_names_and_contents(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.bin").write_bytes(b"alpha")
    (root / "b.bin").write_bytes(b"beta")
    first = sha256_path(root)
    assert first == sha256_path(root)
    assert sha256_path(root / "a.bin") != sha256_path(root / "b.bin")
    (root / "b.bin").write_bytes(b"changed")
    assert sha256_path(root) != first


def test_pytree_fingerprint_is_order_invariant_and_value_sensitive() -> None:
    left = {"b": np.asarray([2, 3]), "a": {"x": np.asarray(1.0)}}
    reordered = {"a": {"x": np.asarray(1.0)}, "b": np.asarray([2, 3])}
    changed = {"a": {"x": np.asarray(1.0)}, "b": np.asarray([2, 4])}
    assert pytree_fingerprint(left) == pytree_fingerprint(reordered)
    assert pytree_fingerprint(left) != pytree_fingerprint(changed)
    assert pytree_fingerprint([np.asarray(1)]) != pytree_fingerprint((np.asarray(1),))


def test_run_identity_is_immutable(tmp_path: Path) -> None:
    identity = {"stage": "train", "seed": 7, "nested": {"a": 1}}
    path = ensure_run_identity(tmp_path, identity)
    assert path == ensure_run_identity(tmp_path, identity)
    assert read_run_identity(tmp_path) == identity
    with pytest.raises(RuntimeError, match="different fields"):
        ensure_run_identity(tmp_path, {**identity, "seed": 8})


def test_deployment_parameter_whitelist_excludes_training_only_subtrees() -> None:
    assert PRIMARY_ARTIFACT_NAME == "DEPI"
    params = {
        name: {"weight": np.asarray([index], dtype=np.float32)}
        for index, name in enumerate(DEPLOYABLE_PARAM_NAMES)
    }
    params["training_only_bootstrap"] = {"secret": np.asarray([1.0])}
    params["optimizer_state"] = {"secret": np.asarray([2.0])}
    pruned = deployable_parameters(params)
    assert tuple(pruned) == DEPLOYABLE_PARAM_NAMES
    assert "training_only_bootstrap" not in pruned
    assert "optimizer_state" not in pruned

    incomplete = dict(params)
    incomplete.pop("universal_actor")
    with pytest.raises(ValueError, match="lack deployable subtrees"):
        deployable_parameters(incomplete)


def test_actual_deployment_export_load_and_forward_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_config(
        root / "experiments/overcooked_v2/configs/depi_simple_mechanical_e2e.yaml",
        run_kind="mechanical",
    )
    config = replace(
        config,
        model=replace(
            config.model,
            task_hidden_dim=8,
            capability_hidden_dim=8,
            capability_dim=4,
            component_embedding_dim=4,
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            response_hidden_dim=8,
            modulation_rank=2,
            action_embedding_dim=4,
        ),
    )
    observation_shape = (5, 5, 39)
    model = build_model(
        observation_shape=observation_shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        capability_hidden_dim=config.model.capability_hidden_dim,
        capability_dim=config.model.capability_dim,
        protocol_components=config.model.protocol_components,
        component_embedding_dim=config.model.component_embedding_dim,
        actor_hidden_dim=config.model.actor_hidden_dim,
        critic_hidden_dim=config.model.critic_hidden_dim,
        response_hidden_dim=config.model.response_hidden_dim,
        modulation_rank=config.model.modulation_rank,
        action_embedding_dim=config.model.action_embedding_dim,
        method_variant=config.method_variant,
    )
    state = reset_deployment_state(
        Deployment("ego", config, model, {}),
        batch_size=2,
        observation_shape=observation_shape,
    )
    observations = jax.random.normal(jax.random.PRNGKey(1), (2,) + observation_shape)
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(2),
        example_state=state,
        example_observation=observations,
    )
    training_run = tmp_path / "training"
    ensure_run_identity(
        training_run,
        {
            "stage": "train",
            "method": METHOD_VERSION,
            "method_variant": config.method_variant,
            "config": config.to_mapping(),
            "observation_shape": list(observation_shape),
            "action_count": 6,
            "ego_run_id": "ego",
        },
    )
    bundle = export_deployment_bundle(
        tmp_path / "deployment",
        source_training_run=training_run,
        deployment=Deployment("ego", config, model, params),
    )
    loaded = load_deployment(bundle, config)
    assert pytree_fingerprint(loaded.params) == pytree_fingerprint(
        deployable_parameters(params)
    )
    stepped, actions, output, log_probability = deployment_action(
        deployment=loaded,
        state=state,
        observation=observations,
        keys=jax.random.split(jax.random.PRNGKey(3), 2),
    )
    assert stepped.protocol_carry.shape == (2, config.model.protocol_components)
    assert actions.shape == log_probability.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(output.policy_logits)))
