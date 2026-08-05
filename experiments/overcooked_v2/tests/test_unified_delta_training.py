from __future__ import annotations


def test_active_package_contains_no_retired_patch_chain_tokens() -> None:
    from pathlib import Path

    root = Path("src/delta_zsc")
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in root.glob("*.py")
    )
    retired = {
        "pair_comparator",
        "separation_margin",
        "context_dropout",
        "capability_consistency_weight",
        "posterior_decision_weight",
        "decision_policy_weight",
        "gradient_routing",
    }
    assert not any(token in text for token in retired)


def test_base_and_latent_optimizers_are_distinct() -> None:
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.training import make_optimizer

    base = {"x": jnp.zeros((2,))}
    latent = {"y": jnp.zeros((3,))}
    base_optimizer, base_state = make_optimizer(
        base,
        learning_rate=1e-3,
        gradient_clip_norm=1.0,
        adam_epsilon=1e-5,
    )
    latent_optimizer, latent_state = make_optimizer(
        latent,
        learning_rate=2e-3,
        gradient_clip_norm=0.5,
        adam_epsilon=1e-5,
    )
    assert base_optimizer is not latent_optimizer
    base_shapes = [tuple(leaf.shape) for leaf in jax.tree_util.tree_leaves(base_state)]
    latent_shapes = [tuple(leaf.shape) for leaf in jax.tree_util.tree_leaves(latent_state)]
    assert (2,) in base_shapes
    assert (3,) in latent_shapes
    assert base_shapes != latent_shapes


def test_decision_anchor_schema_has_no_stale_posterior_or_comparator() -> None:
    from src.delta_zsc.types import DecisionAnchorBatch

    fields = set(DecisionAnchorBatch._fields)
    assert fields == {
        "time_indexes",
        "lane_indexes",
        "centered_returns",
        "standard_errors",
        "action_mask",
        "fit_replica_returns",
        "evaluation_returns",
    }
