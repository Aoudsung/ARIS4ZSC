from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.overcooked_v2.deployment import (
    DEPLOYABLE_PARAM_NAMES,
    PRIMARY_ARTIFACT_NAME,
    deployable_parameters,
)
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
    assert PRIMARY_ARTIFACT_NAME == "DELTA-ZSC-E2E"
    params = {
        name: {"weight": np.asarray([index], dtype=np.float32)}
        for index, name in enumerate(DEPLOYABLE_PARAM_NAMES)
    }
    params["partner_generator"] = {"secret": np.asarray([1.0])}
    params["optimizer_state"] = {"secret": np.asarray([2.0])}
    pruned = deployable_parameters(params)
    assert tuple(pruned) == DEPLOYABLE_PARAM_NAMES
    assert "partner_generator" not in pruned
    assert "optimizer_state" not in pruned

    incomplete = dict(params)
    incomplete.pop("universal_actor")
    with pytest.raises(ValueError, match="lack deployable subtrees"):
        deployable_parameters(incomplete)
