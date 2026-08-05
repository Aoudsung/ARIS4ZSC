from __future__ import annotations


def _state():
    import jax.numpy as jnp

    from src.delta_zsc.types import (
        AgentState,
        BehaviorPosterior,
        UnifiedTrainState,
    )

    return UnifiedTrainState(
        base_params={"base": jnp.arange(3.0)},
        latent_params={"latent": jnp.arange(4.0)},
        base_optimizer_state={"moment": jnp.ones((3,))},
        latent_optimizer_state={"moment": jnp.ones((4,))},
        agent_state=AgentState(
            task_carry=jnp.zeros((2, 5)),
            behavior=BehaviorPosterior(
                alpha=jnp.ones((2, 4)), beta=2.0 * jnp.ones((2, 4))
            ),
            belief=jnp.full((2, 3), 1.0 / 3.0),
            previous_observation=jnp.zeros((2, 5, 5, 39)),
            previous_action=jnp.asarray([1, 2], dtype=jnp.int32),
            episode_start=jnp.asarray([False, True]),
        ),
        random_key=jnp.asarray([123, 456], dtype=jnp.uint32),
        environment_steps=jnp.asarray(1024, dtype=jnp.int64),
        update_count=jnp.asarray(7, dtype=jnp.int32),
        resource_ledger={"ego_policy_steps": 1024},
    )


def test_unified_state_roundtrips_with_named_types(tmp_path) -> None:
    import jax
    import numpy as np
    import orbax.checkpoint as ocp

    expected = _state()
    path = tmp_path / "state"
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(path), expected, force=True)
    restored = checkpointer.restore(str(path), item=expected)
    assert type(restored) is type(expected)
    assert type(restored.agent_state) is type(expected.agent_state)
    assert type(restored.agent_state.behavior) is type(expected.agent_state.behavior)
    for left, right in zip(
        jax.tree_util.tree_leaves(expected),
        jax.tree_util.tree_leaves(restored),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
