"""Exact official network, checkpoint, and frozen-partner adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_CONVOLUTION_LEAVES = tuple(
    ("backbone", "CNN_0", f"Conv_{index}", leaf)
    for index in range(6)
    for leaf in ("kernel", "bias")
)
_DENSE_ENCODER_LEAVES = tuple(
    ("backbone", "CNN_0", "Dense_0", leaf) for leaf in ("kernel", "bias")
)
_NORMALIZATION_LEAVES = tuple(
    ("backbone", "LayerNorm_0", leaf) for leaf in ("scale", "bias")
)
_GRU_LEAVES = tuple(
    ("backbone", "ScannedRNN_0", "GRUCell_1", gate, leaf)
    for gate, leaves in (
        ("ir", ("kernel", "bias")),
        ("hr", ("kernel",)),
        ("iz", ("kernel", "bias")),
        ("hz", ("kernel",)),
        ("in", ("kernel", "bias")),
        ("hn", ("kernel", "bias")),
    )
    for leaf in leaves
)
_SHARED_HEAD_LEAVES = tuple(
    ("shared_heads", target, leaf)
    for target in ("actor_hidden", "actor_logits", "critic_hidden", "critic_value")
    for leaf in ("kernel", "bias")
)
EXPLICIT_TARGET_LEAVES = (
    *_CONVOLUTION_LEAVES,
    *_DENSE_ENCODER_LEAVES,
    *_NORMALIZATION_LEAVES,
    *_GRU_LEAVES,
    *_SHARED_HEAD_LEAVES,
)
_SHARED_SOURCE_COMPONENTS = {
    "actor_hidden": "Dense_0",
    "actor_logits": "Dense_1",
    "critic_hidden": "Dense_2",
    "critic_value": "Dense_3",
}


def _source_leaf(target: tuple[str, ...]) -> tuple[str, ...]:
    if target[0] == "backbone":
        return ("params", *target[1:])
    return (
        "params",
        _SHARED_SOURCE_COMPONENTS[target[1]],
        target[2],
    )


EXPLICIT_OFFICIAL_LEAF_MAPPING = {
    target: _source_leaf(target) for target in EXPLICIT_TARGET_LEAVES
}


def _manifest_path(checkpoint_path: Path) -> Path:
    candidates = (
        checkpoint_path.parent / "official_artifact_manifest.json",
        checkpoint_path.parent.parent / "official_artifact_manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("The official artifact manifest is missing beside the checkpoint.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_launch_tools(launch_config_path: str | Path) -> tuple[Any, Any, Any]:
    import yaml

    path = Path(launch_config_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("The official launch config must be a mapping.")
    if payload.get("layout") == "test_time_wide":
        from experiments.overcooked_v2.official.overcooked_v2_experiments_wide_adapter import (
            compose_official_wide_training_config,
            load_official_wide_launch_config,
            validate_official_wide_artifact_manifest,
        )

        return (
            load_official_wide_launch_config,
            compose_official_wide_training_config,
            validate_official_wide_artifact_manifest,
        )
    from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
        compose_official_training_config,
    )
    from experiments.overcooked_v2.path_c_official_artifact import (
        load_official_launch_config,
        validate_official_artifact_manifest,
    )

    return (
        load_official_launch_config,
        compose_official_training_config,
        validate_official_artifact_manifest,
    )


def _load_verified_params(
    checkpoint_ref: Any, *, launch_config_path: str | Path | None = None
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from experiments.overcooked_v2.path_c_official_artifact import (
        flax_weights_sha256,
        load_flax_parameter_tree,
    )

    checkpoint_path = Path(checkpoint_ref.checkpoint_path).resolve()
    manifest_path = _manifest_path(checkpoint_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("The official artifact manifest must be a mapping.")
    if manifest.get("training_run_id") != checkpoint_ref.training_run_id:
        raise ValueError("Official training run identifier does not match the model config.")
    final_checkpoint = manifest.get("checkpoint")
    if not isinstance(final_checkpoint, Mapping):
        raise ValueError("The official artifact manifest has no checkpoint contract.")
    checkpoint = final_checkpoint
    history_match = False
    if Path(str(final_checkpoint.get("path", ""))).resolve() != checkpoint_path:
        history = manifest.get("checkpoint_history")
        if not isinstance(history, Sequence) or isinstance(
            history, (str, bytes, bytearray)
        ):
            raise ValueError(
                "The official artifact manifest has no matching checkpoint history."
            )
        matches = [
            item
            for item in history
            if isinstance(item, Mapping)
            and Path(str(item.get("path", ""))).resolve() == checkpoint_path
        ]
        if len(matches) != 1:
            raise ValueError(
                "The official checkpoint history does not uniquely identify this path."
            )
        checkpoint = matches[0]
        history_match = True
    params = load_flax_parameter_tree(
        checkpoint_path,
        checkpoint_format=str(checkpoint["format"]),
        parameter_tree_path=tuple(checkpoint["parameter_tree_path"]),
    )
    if history_match:
        from experiments.overcooked_v2.path_c_official_artifact import (
            checkpoint_artifact_sha256,
        )

        if checkpoint.get("checkpoint_sha256") != checkpoint_artifact_sha256(
            checkpoint_path
        ):
            raise ValueError("Official scheduled checkpoint content hash changed.")
    actual = flax_weights_sha256(params)
    if actual != checkpoint_ref.flax_weights_sha256 or checkpoint.get("model_weights_sha256") != actual:
        raise ValueError("Official Flax weight hash does not match the model config and manifest.")
    if Path(str(checkpoint.get("path", ""))).resolve() != checkpoint_path:
        raise ValueError("Official artifact manifest identifies another checkpoint path.")
    if launch_config_path is not None:
        launch_path = Path(launch_config_path).resolve()
        if (
            manifest.get("training_config_path") != str(launch_path)
            or manifest.get("training_config_sha256") != _file_sha256(launch_path)
        ):
            raise ValueError("Official checkpoint history changed its launch configuration.")
        if not history_match:
            unused_loader, unused_composer, validate_official_artifact_manifest = (
                _official_launch_tools(launch_path)
            )
            del unused_loader, unused_composer
            validated = validate_official_artifact_manifest(
                launch_path,
                manifest_path,
                expected_checkpoint_path=checkpoint_path,
                params=params,
            )
            if validated.get("training_run_id") != checkpoint_ref.training_run_id:
                raise ValueError(
                    "Validated official provenance changed the training run identifier."
                )
    return params, manifest


@dataclass(frozen=True)
class OfficialBackboneDock:
    """Read network facts from one resolved official experiment configuration."""

    resolved_config: Mapping[str, Any]
    launch_config_path: Path

    @classmethod
    def from_launch_config(cls, launch_config_path: str | Path) -> "OfficialBackboneDock":
        resolved_path = Path(launch_config_path).resolve()
        load_official_launch_config, compose_official_training_config, unused_validator = (
            _official_launch_tools(resolved_path)
        )
        del unused_validator
        launch = load_official_launch_config(resolved_path)
        return cls(compose_official_training_config(launch), resolved_path)

    def official_parameter_tree(self, checkpoint_ref: Any) -> Mapping[str, Any]:
        return _load_verified_params(
            checkpoint_ref, launch_config_path=self.launch_config_path
        )[0]

    def network_dimensions(self) -> Mapping[str, Any]:
        model = self.resolved_config["model"]
        return {
            "encoder_dim": int(model["GRU_HIDDEN_DIM"]),
            "gru_hidden_dim": int(model["GRU_HIDDEN_DIM"]),
            "actor_critic_hidden_dim": int(model["FC_DIM_SIZE"]),
            "activation": str(model["ACTIVATION"]),
        }

    def action_order(self) -> Sequence[str]:
        from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
            OFFICIAL_ACTION_ORDER,
        )

        return tuple(OFFICIAL_ACTION_ORDER)

    def initial_recurrent_state(self, batch_size: int) -> Any:
        from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
            OvercookedV2ExperimentsNetworkAdapter,
        )

        return OvercookedV2ExperimentsNetworkAdapter(self.resolved_config).initial_state(batch_size)

    def apply_reference_actor_critic(
        self,
        params: Mapping[str, Any],
        carry: Any,
        observations: Any,
        episode_start: Any,
    ) -> tuple[Any, Any, Any]:
        """Expose the unchanged official carry, logits, and critic for parity checks."""

        from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
            OvercookedV2ExperimentsNetworkAdapter,
        )

        adapter = OvercookedV2ExperimentsNetworkAdapter(self.resolved_config)
        next_carry, distribution, value = adapter._network().apply(
            params,
            carry,
            (observations, episode_start),
        )
        return next_carry, distribution.logits, value

    def frozen_policy(
        self, params: Mapping[str, Any], *, policy_id: str
    ) -> "FrozenOfficialPartner":
        """Bind verified parameters to the unchanged official actor network."""

        from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
            OvercookedV2ExperimentsNetworkAdapter,
        )

        return FrozenOfficialPartner(
            params=params,
            network_adapter=OvercookedV2ExperimentsNetworkAdapter(self.resolved_config),
            prototype_id=policy_id,
        )

    def explicit_parameter_mapping(
        self, target_params: Mapping[str, Any]
    ) -> Mapping[tuple[str, ...], tuple[str, ...]]:
        from src.path_c.model.backbone import get_leaf

        for target_path in EXPLICIT_OFFICIAL_LEAF_MAPPING:
            get_leaf(target_params, target_path)
        return dict(EXPLICIT_OFFICIAL_LEAF_MAPPING)


@dataclass(frozen=True)
class FrozenOfficialPartner:
    params: Mapping[str, Any]
    network_adapter: Any
    prototype_id: str

    def initial_state(self, batch_size: int) -> Any:
        return self.network_adapter.initial_state(batch_size)

    def act(
        self, observation: Any, carry: Any, key: Any, episode_start: Any = False
    ) -> tuple[Any, Any]:
        import jax

        next_carry, logits = self.network_adapter.apply_actor(
            self.params, carry, observation, episode_start
        )
        return jax.random.categorical(key, logits, axis=-1), next_carry


def load_frozen_partner_pool(
    members: Sequence[Any], *, launch_config_paths: Sequence[str | Path]
) -> tuple[FrozenOfficialPartner, ...]:
    """Load immutable policies and verify every weight and run identity."""

    from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
        OvercookedV2ExperimentsNetworkAdapter,
    )

    references = tuple(members)
    launches = tuple(Path(path).resolve() for path in launch_config_paths)
    if not references or len(references) != len(launches):
        raise ValueError(
            "The frozen partner pool requires one launch config per reference."
        )
    policies = []
    for index, (reference, launch_path) in enumerate(zip(references, launches, strict=True)):
        params, unused_manifest = _load_verified_params(
            reference, launch_config_path=launch_path
        )
        del unused_manifest
        load_official_launch_config, compose_official_training_config, unused_validator = (
            _official_launch_tools(launch_path)
        )
        del unused_validator
        resolved = compose_official_training_config(load_official_launch_config(launch_path))
        policies.append(
            FrozenOfficialPartner(
                params=params,
                network_adapter=OvercookedV2ExperimentsNetworkAdapter(resolved),
                prototype_id=f"prototype_{index}_{reference.training_run_id}",
            )
        )
    return tuple(policies)


def make_batched_partner_step(partners: Sequence[FrozenOfficialPartner]) -> Any:
    """Dispatch one selected immutable network per vector-environment lane."""

    import jax
    import jax.numpy as jnp

    policies = tuple(partners)
    if not policies:
        raise ValueError("Batched partner dispatch requires at least one policy.")
    adapter = policies[0].network_adapter
    stacked_params = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values, axis=0),
        *(policy.params for policy in policies),
    )

    def batched(
        prototype_indices: Any,
        observations: Any,
        carry: Any,
        episode_start: Any,
        key: Any,
    ) -> tuple[Any, Any]:
        indices = jnp.asarray(prototype_indices, dtype=jnp.int32)
        if indices.ndim != 1 or indices.shape[0] != observations.shape[0]:
            raise ValueError("Partner member indexes must align with the vector batch.")
        selected_params = jax.tree_util.tree_map(
            lambda value: value[indices], stacked_params
        )

        def apply_one(
            selected: Any,
            carry_at_lane: Any,
            observation_at_lane: Any,
            start_at_lane: Any,
        ) -> tuple[Any, Any]:
            batched_carry = jax.tree_util.tree_map(
                lambda value: value[None, ...], carry_at_lane
            )
            next_carry, logits = adapter.apply_actor(
                selected,
                batched_carry,
                observation_at_lane,
                start_at_lane,
            )
            return (
                jax.tree_util.tree_map(lambda value: value[0], next_carry),
                logits,
            )

        next_state, logits = jax.vmap(apply_one)(
            selected_params, carry, observations, episode_start
        )
        lane_keys = jax.random.split(key, observations.shape[0])
        actions = jax.vmap(
            lambda lane_key, lane_logits: jax.random.categorical(
                lane_key, lane_logits, axis=-1
            )
        )(lane_keys, logits)
        return actions, next_state

    return batched


def validate_checkpoint_reference(
    checkpoint_ref: Any, *, launch_config_path: str | Path | None = None
) -> Mapping[str, Any]:
    params, manifest = _load_verified_params(
        checkpoint_ref, launch_config_path=launch_config_path
    )
    del params
    return {
        "checkpoint_path": str(Path(checkpoint_ref.checkpoint_path).resolve()),
        "flax_weights_sha256": checkpoint_ref.flax_weights_sha256,
        "training_run_id": checkpoint_ref.training_run_id,
        "training_seed": int(manifest.get("seed", -1)),
        "layout": str(manifest.get("layout", "")),
        "family_id": str(
            manifest.get("partner_family", {}).get("family_id", "")
            if isinstance(manifest.get("partner_family"), Mapping)
            else ""
        ),
        "manifest_path": str(_manifest_path(Path(checkpoint_ref.checkpoint_path).resolve())),
        "manifest_schema_version": str(manifest.get("schema_version")),
        "verified": True,
    }


def verify_initialized_actor_critic_parity(
    *,
    dock: OfficialBackboneDock,
    model: Any,
    initialized_params: Mapping[str, Any],
    official_params: Mapping[str, Any],
    observation_shape: Sequence[int],
) -> Mapping[str, Any]:
    """Require exact official carry, actor, and critic outputs before training."""

    import hashlib

    import jax
    import jax.numpy as jnp
    import numpy as np

    shape = tuple(int(value) for value in observation_shape)
    observations = jnp.linspace(
        0.0,
        1.0,
        num=2 * 2 * int(np.prod(shape)),
        dtype=jnp.float32,
    ).reshape((2, 2, *shape))
    episode_start = jnp.asarray(
        ((False, True), (False, False)), dtype=jnp.bool_
    )
    carry = jax.tree_util.tree_map(
        jnp.ones_like, dock.initial_recurrent_state(batch_size=2)
    )
    expected_carry, expected_logits, expected_critic = dock.apply_reference_actor_critic(
        official_params,
        carry,
        observations,
        episode_start,
    )
    actual_carry, actual = model.apply(
        {"params": initialized_params},
        carry,
        observations,
        episode_start,
    )

    def compare(name: str, actual_value: Any, expected_value: Any) -> tuple[list[list[int]], float]:
        actual_leaves = [np.asarray(value) for value in jax.tree_util.tree_leaves(actual_value)]
        expected_leaves = [np.asarray(value) for value in jax.tree_util.tree_leaves(expected_value)]
        if len(actual_leaves) != len(expected_leaves):
            raise RuntimeError(f"Official parity changed the {name} tree structure.")
        shapes: list[list[int]] = []
        maximum_error = 0.0
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
            shapes.append([int(value) for value in actual_leaf.shape])
            if actual_leaf.shape != expected_leaf.shape:
                raise RuntimeError(f"Official parity changed a {name} leaf shape.")
            if actual_leaf.size:
                maximum_error = max(
                    maximum_error,
                    float(
                        np.max(
                            np.abs(
                                actual_leaf.astype(np.float64)
                                - expected_leaf.astype(np.float64)
                            )
                        )
                    ),
                )
            if not np.array_equal(actual_leaf, expected_leaf):
                raise RuntimeError(f"Initialized model {name} differs from the official network.")
        return shapes, maximum_error

    carry_shapes, carry_error = compare("carry", actual_carry, expected_carry)
    logits_shapes, logits_error = compare(
        "actor logits", actual["actor_logits"], expected_logits
    )
    critic_shapes, critic_error = compare(
        "critic", actual["shared_value"], expected_critic
    )
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(observations)).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(episode_start)).tobytes())
    for leaf in jax.tree_util.tree_leaves(carry):
        digest.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
    return {
        "passed": True,
        "comparison": "exact",
        "input_sha256": digest.hexdigest(),
        "carry_shapes": carry_shapes,
        "actor_logits_shapes": logits_shapes,
        "critic_shapes": critic_shapes,
        "maximum_absolute_error": {
            "carry": carry_error,
            "actor_logits": logits_error,
            "critic": critic_error,
        },
    }
