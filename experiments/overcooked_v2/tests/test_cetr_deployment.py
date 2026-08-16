from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _reference(config, checkpoint: Path) -> dict[str, object]:
    return {
        "artifact_type": "cetr_reference_sp",
        "version": 2,
        "layout": config.environment.layout,
        "seed_index": 0,
        "tau_sp": 1.0,
        "episodes_per_pairing": config.evaluation.episodes_per_pairing,
        "evaluation_root_seed": config.evaluation.evaluation_seed,
        "source_checkpoint": str(checkpoint.resolve()),
    }


def test_actor_only_bundle_separates_provenance_and_has_strict_schema(tmp_path: Path) -> None:
    import jax

    from experiments.overcooked_v2.deployment import (
        DEPLOYMENT_BUNDLE_VERSION,
        export_deployment_bundle,
        load_deployment,
    )
    from src.cetr_zsc.config import CHECKPOINT_SCHEMA_VERSION, load_config
    from src.cetr_zsc.model import CetrModel
    from src.cetr_zsc.storage import read_json, write_json

    config = load_config(
        "experiments/overcooked_v2/configs/cetr_simple_mechanical.yaml",
        run_kind="mechanical",
    )
    model = CetrModel(config, (5, 5, 26), 6)
    params = model.init_parameters(jax.random.PRNGKey(0))
    source = tmp_path / "training"
    source.mkdir()
    write_json(source / "partner_manifest.json", {"version": 2, "layout": config.environment.layout, "runs": []})
    checkpoint = tmp_path / "initializer"
    checkpoint.write_bytes(b"checkpoint")
    bundle = tmp_path / "deployment"
    export_deployment_bundle(
        bundle,
        ego_run_id="ego-0",
        config=config,
        observation_shape=(5, 5, 26),
        params=params,
        source_training_run=source,
        reference_sp_artifact=_reference(config, checkpoint),
    )

    loaded = load_deployment(bundle)
    assert set(loaded.params) == {
        "task_conv",
        "task_dense",
        "task_norm",
        "task_gru",
        "actor_trunk",
        "actor",
    }
    assert "value" not in loaded.params
    assert "value_trunk" not in loaded.params
    assert loaded.provenance_path == (bundle / "provenance.json").resolve()

    descriptor = read_json(bundle / "deployment_bundle.json")
    assert descriptor["version"] == DEPLOYMENT_BUNDLE_VERSION == 7
    assert descriptor["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION == 8
    assert set(descriptor) == {
        "version",
        "checkpoint_schema_version",
        "method",
        "ego_run_id",
        "config",
        "observation_shape",
        "action_count",
        "params",
        "source_training_run",
        "provenance",
    }
    assert "reference_sp_artifact" not in descriptor
    assert "training_parent_manifest" not in descriptor
    provenance = read_json(bundle / "provenance.json")
    assert set(provenance) == {"reference_sp_artifact", "training_parent_manifest"}

    descriptor["unexpected"] = True
    (bundle / "deployment_bundle.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema"):
        load_deployment(bundle)

    descriptor.pop("unexpected")
    descriptor["version"] = 6
    (bundle / "deployment_bundle.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="method/schema"):
        load_deployment(bundle)

    descriptor["version"] = 7
    descriptor["checkpoint_schema_version"] = 6
    (bundle / "deployment_bundle.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="method/schema"):
        load_deployment(bundle)


def test_deployment_action_uses_actor_parameters_and_returns_finite_logits(tmp_path: Path) -> None:
    import jax
    import jax.numpy as jnp

    from experiments.overcooked_v2.deployment import (
        deployment_action,
        export_deployment_bundle,
        load_deployment,
        reset_deployment_state,
    )
    from src.cetr_zsc.config import load_config
    from src.cetr_zsc.model import CetrModel

    config = load_config(
        "experiments/overcooked_v2/configs/cetr_simple_mechanical.yaml",
        run_kind="mechanical",
    )
    model = CetrModel(config, (5, 5, 26), 6)
    params = model.init_parameters(jax.random.PRNGKey(1))
    checkpoint = tmp_path / "initializer"
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "training"
    source.mkdir()
    bundle = tmp_path / "deployment"
    export_deployment_bundle(
        bundle,
        ego_run_id="ego-1",
        config=config,
        observation_shape=(5, 5, 26),
        params=params,
        source_training_run=source,
        reference_sp_artifact=_reference(config, checkpoint),
    )
    deployment = load_deployment(bundle)
    state = reset_deployment_state(deployment, 1)
    next_state, actions, log_probability = deployment_action(
        deployment,
        state,
        jnp.zeros((1, 5, 5, 26), dtype=jnp.float32),
        jax.random.PRNGKey(2),
    )
    del next_state
    assert actions.shape == (1,)
    assert log_probability.shape == (1,)
    assert np.all(np.isfinite(np.asarray(log_probability)))
