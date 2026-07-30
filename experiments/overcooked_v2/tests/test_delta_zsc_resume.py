from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("orbax.checkpoint")

from src.path_c.storage import (  # noqa: E402
    orbax_manager,
    restore_latest_checkpoint,
    save_checkpoint,
)


def _next_update(state):
    draw_key, next_key = jax.random.split(state["random_key"])
    noise = jax.random.normal(draw_key, state["params"].shape)
    return {
        "params": state["params"] + noise + state["generator_params"],
        "optimizer_state": state["optimizer_state"] + 1,
        "random_key": next_key,
        "runner_state": state["runner_state"] + 3,
        "generator_params": state["generator_params"] * 0.9,
    }


def test_checkpoint_resume_preserves_rng_runner_generator_and_next_update(
    tmp_path: Path,
) -> None:
    state = {
        "params": jnp.asarray([1.0, 2.0]),
        "optimizer_state": jnp.asarray(4, dtype=jnp.int32),
        "random_key": jax.random.PRNGKey(17),
        "runner_state": jnp.asarray([5, 6], dtype=jnp.int32),
        "generator_params": jnp.asarray([0.25, -0.5]),
    }
    manager = orbax_manager(tmp_path / "checkpoints")
    try:
        save_checkpoint(manager, step=9, state=state)
        restored = restore_latest_checkpoint(manager, item=state)
        assert restored is not None and restored[0] == 9
        direct = _next_update(state)
        resumed = _next_update(restored[1])
        for left, right in zip(
            jax.tree_util.tree_leaves(direct),
            jax.tree_util.tree_leaves(resumed),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    finally:
        manager.close()
