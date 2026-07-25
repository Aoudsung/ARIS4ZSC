from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.path_c.contracts.config import ModelConfig, PathCModelConfig
from src.path_c.contracts.records import CheckpointMetadata
from src.path_c.model.adaptation_model import (
    build_model,
    initialize_from_official,
)
from src.path_c.model.backbone import copy_explicit_parameter_leaves
from src.path_c.model.checkpoint import load_checkpoint, save_checkpoint


def _jax_modules():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    return jax, pytest.importorskip("jax.numpy")


def _metadata(stage: str = "prefit") -> CheckpointMetadata:
    shared = stage == "prefit"
    return CheckpointMetadata(
        schema_version="path_c_model_checkpoint_metadata_v1",
        stage=stage,
        condition_id=None if shared else "decision_focused",
        controller=None if shared else "registered_response_sequential_branch_v1",
        shared_across_conditions=shared,
        run_kind="development",
        scientific_readout_allowed=False,
        environment_steps=128,
        completed_episodes=4,
        response_vocabulary_sha256="1" * 64,
        backbone_origin_sha256="2" * 64,
        config_sha256="3" * 64,
        extra={"response_vocabulary": ["terminal", "visible"]},
    )


def test_explicit_parameter_mapping_requires_named_paths_and_exact_shapes() -> None:
    initial = {"backbone": {"dense": {"kernel": np.zeros((2, 3))}}}
    official = {"Dense_0": {"kernel": np.ones((2, 3))}}
    copied = copy_explicit_parameter_leaves(
        initial,
        official,
        {("backbone", "dense", "kernel"): ("Dense_0", "kernel")},
    )
    np.testing.assert_array_equal(copied["backbone"]["dense"]["kernel"], 1.0)
    with pytest.raises(ValueError, match="shape"):
        copy_explicit_parameter_leaves(
            initial,
            {"Dense_0": {"kernel": np.ones((3, 2))}},
            {("backbone", "dense", "kernel"): ("Dense_0", "kernel")},
        )
    with pytest.raises(ValueError, match="explicit"):
        copy_explicit_parameter_leaves(initial, official, {})


def test_checkpoint_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    pytest.importorskip("flax")
    params = {"backbone": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)}}
    save_checkpoint(tmp_path, params=params, metadata=_metadata())
    restored, metadata, manifest = load_checkpoint(tmp_path)
    np.testing.assert_array_equal(restored["backbone"]["kernel"], params["backbone"]["kernel"])
    assert metadata == _metadata()
    assert manifest["metadata"]["condition_id"] is None
    weights = tmp_path / "weights.msgpack"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="do not match"):
        load_checkpoint(tmp_path)


def test_checkpoint_metadata_distinguishes_shared_prefit_and_condition_adaptation() -> None:
    assert _metadata("prefit").shared_across_conditions is True
    adapted = _metadata("adaptation")
    assert adapted.condition_id == "decision_focused"
    with pytest.raises(ValueError, match="condition and controller"):
        replace(adapted, condition_id=None)


def test_only_critic_loss_reaches_backbone_parameters() -> None:
    jax, jnp = _jax_modules()
    model = build_model(
        num_prototypes=2,
        action_count=3,
        response_count=4,
        official_dimensions={
            "encoder_dim": 4,
            "gru_hidden_dim": 4,
            "actor_critic_hidden_dim": 4,
            "activation": "relu",
        },
        model_config=ModelConfig(
            encoder_dim=4,
            gru_hidden_dim=4,
            response_hidden_dim=4,
            value_hidden_dim=4,
            transition_hidden_dim=4,
            next_feature_summary_dim=4,
        ),
    )
    observations = jnp.ones((2, 2, 5, 5, 3), dtype=jnp.float32)
    starts = jnp.zeros((2, 2), dtype=jnp.bool_)
    carry = model.initial_carry(2)
    params = model.init(jax.random.PRNGKey(7), carry, observations, starts)["params"]

    def output(candidate_params):
        return model.apply({"params": candidate_params}, carry, observations, starts)[1]

    losses = {
        "critic": lambda values: jnp.sum(jnp.square(values["shared_value"])),
        "actor": lambda values: jnp.sum(values["actor_logits"]),
        "prototype_value": lambda values: jnp.sum(values["prototype_values"]),
        "response": lambda values: jnp.sum(values["response_logits"]),
        "transition": lambda values: (
            jnp.sum(values["transition_response_logits"])
            + jnp.sum(values["reward_estimates"])
            + jnp.sum(values["next_feature_summaries"])
        ),
    }

    def norm(tree) -> float:
        leaves = jax.tree_util.tree_leaves(tree)
        return float(jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves)))

    backbone_norms = {}
    for name, loss in losses.items():
        gradients = jax.grad(lambda candidate: loss(output(candidate)))(params)
        backbone_norms[name] = norm(gradients["backbone"])
    assert backbone_norms["critic"] > 0.0
    assert backbone_norms["actor"] == 0.0
    assert backbone_norms["prototype_value"] == 0.0
    assert backbone_norms["response"] == 0.0
    assert backbone_norms["transition"] == 0.0


def test_remote_official_parity_matches_carry_logits_and_critic() -> None:
    """Run after official artifacts and packages are present on the remote host."""

    jax, jnp = _jax_modules()
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("jaxmarl")
    from experiments.overcooked_v2.model_dock.env_dock import (
        OvercookedV2VectorEnvironment,
    )
    from experiments.overcooked_v2.model_dock.official_dock import (
        EXPLICIT_OFFICIAL_LEAF_MAPPING,
        OfficialBackboneDock,
        verify_initialized_actor_critic_parity,
    )
    from experiments.overcooked_v2.model_dock.response_dock import (
        registered_response_vocabulary,
    )

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "path_c_model_development_decision_focused.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PathCModelConfig.from_mapping(payload, base_dir=config_path.parent)
    manifest_candidates = (
        config.backbone_init.checkpoint_path.parent / "official_artifact_manifest.json",
        config.backbone_init.checkpoint_path.parent.parent
        / "official_artifact_manifest.json",
    )
    if not config.backbone_init.checkpoint_path.exists() or not any(
        path.is_file() for path in manifest_candidates
    ):
        pytest.skip("Official seed-100 artifact is not present on this host.")
    launch_path = (
        config_path.parent / "path_c_official_sp_simple_seed100.yaml"
    ).resolve()
    dock = OfficialBackboneDock.from_launch_config(launch_path)
    environment = OvercookedV2VectorEnvironment.create(num_envs=2)
    vocabulary = registered_response_vocabulary()
    model = build_model(
        num_prototypes=4,
        action_count=len(dock.action_order()),
        response_count=vocabulary.size,
        official_dimensions=dock.network_dimensions(),
        model_config=config.model,
    )
    observation_shape = environment.observation_shape
    observations = jnp.linspace(
        0.0,
        1.0,
        num=2 * 2 * int(np.prod(observation_shape)),
        dtype=jnp.float32,
    ).reshape((2, 2, *observation_shape))
    starts = jnp.asarray([[False, True], [False, False]], dtype=jnp.bool_)
    from experiments.overcooked_v2.path_c_official_artifact import (
        flax_weights_sha256,
        load_flax_parameter_tree,
    )
    from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
        OFFICIAL_CHECKPOINT_FORMAT,
        OFFICIAL_PARAMETER_TREE_PATH,
    )

    # This parity fixture is a historical seed-100 checkpoint whose original
    # training-source receipt predates the family-pool implementation.  Verify
    # its checkpoint weights directly, while production pool loading continues
    # to require the current complete source-closure receipt.
    official_params = load_flax_parameter_tree(
        config.backbone_init.checkpoint_path,
        checkpoint_format=OFFICIAL_CHECKPOINT_FORMAT,
        parameter_tree_path=OFFICIAL_PARAMETER_TREE_PATH,
    )
    assert (
        flax_weights_sha256(official_params)
        == config.backbone_init.flax_weights_sha256
    )
    params = initialize_from_official(
        model,
        random_key=jax.random.PRNGKey(17),
        official_params=official_params,
        explicit_leaf_mapping=EXPLICIT_OFFICIAL_LEAF_MAPPING,
        example_observations=observations,
        example_episode_start=starts,
    )
    report = verify_initialized_actor_critic_parity(
        dock=dock,
        model=model,
        initialized_params=params,
        official_params=official_params,
        observation_shape=observation_shape,
    )
    assert report["passed"] is True
    assert report["comparison"] == "exact"
    assert report["maximum_absolute_error"] == {
        "carry": 0.0,
        "actor_logits": 0.0,
        "critic": 0.0,
    }

    swapped = copy.deepcopy(params)
    swapped["shared_heads"]["actor_hidden"], swapped["shared_heads"]["critic_hidden"] = (
        swapped["shared_heads"]["critic_hidden"],
        swapped["shared_heads"]["actor_hidden"],
    )
    with pytest.raises(RuntimeError, match="differs"):
        verify_initialized_actor_critic_parity(
            dock=dock,
            model=model,
            initialized_params=swapped,
            official_params=official_params,
            observation_shape=observation_shape,
        )
