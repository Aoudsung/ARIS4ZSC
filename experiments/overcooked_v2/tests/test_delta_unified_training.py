from __future__ import annotations

from dataclasses import replace

import numpy as np


def _setup(variant: str = "delta_active"):
    import jax

    from src.delta_zsc.config import load_config
    from src.delta_zsc.model import DeltaModel

    config = load_config(
        "experiments/overcooked_v2/configs/delta_unified_simple_mechanical.yaml",
        run_kind="mechanical",
    )
    config = replace(
        config,
        method_variant=variant,
        model=replace(
            config.model,
            task_hidden_dim=16,
            task_embedding_dim=16,
            instant_partner_dim=8,
            latent_hidden_dim=16,
            latent_embedding_dim=8,
            action_embedding_dim=4,
            voi_quadrature_samples=4,
        ),
        ppo=replace(config.ppo, update_epochs=1),
    )
    model = DeltaModel(config, (5, 5, 39), 6)
    base, latent = model.init_parameters(jax.random.PRNGKey(0))
    return config, model, base, latent


def _batch(model, base, latent, *, time_count: int = 4, lanes: int = 4):
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.losses import categorical_log_probability
    from src.delta_zsc.types import RolloutBatch

    observations = jax.random.normal(
        jax.random.PRNGKey(2), (time_count + 1, lanes, 5, 5, 39)
    ) * 0.01
    previous_actions = jnp.zeros((time_count + 1, lanes), dtype=jnp.int32)
    starts = jnp.zeros((time_count + 1, lanes), dtype=jnp.bool_).at[0].set(True)
    _, output = model.sequence(
        base,
        latent,
        model.initial_state(lanes),
        observations,
        previous_actions,
        starts,
    )
    actions = jnp.argmax(output.base_policy_logits[:-1], axis=-1)
    return RolloutBatch(
        observations=observations,
        response_next_observations=observations[1:],
        previous_actions=previous_actions,
        episode_starts=starts,
        actions=actions,
        rewards=jnp.full((time_count, lanes), 0.1),
        shaped_rewards=jnp.zeros((time_count, lanes)),
        dones=jnp.zeros((time_count, lanes), dtype=jnp.bool_),
        old_log_probabilities=categorical_log_probability(
            output.base_policy_logits[:-1], actions
        ),
        old_values=output.value[:-1],
        ppo_mask=jnp.ones((time_count, lanes)),
        initial_policy_state=model.initial_state(lanes),
    )


def _anchors():
    import jax.numpy as jnp

    from src.delta_zsc.types import AnchorBatch

    fit = jnp.asarray(
        [[2.0, 1.0, 0.0, -1.0, -2.0, -3.0], [-1, 0, 1, 2, 3, 4]],
        dtype=jnp.float32,
    )
    replicas = jnp.stack((fit - 0.1, fit + 0.1), axis=-1)
    return AnchorBatch(
        time_indexes=jnp.asarray([1, 2], dtype=jnp.int32),
        lane_indexes=jnp.asarray([0, 1], dtype=jnp.int32),
        fit_returns_by_action=fit,
        evaluation_returns_by_action=fit,
        measurement_covariances=jnp.broadcast_to(
            jnp.eye(5, dtype=jnp.float32)[None] * 0.01, (2, 5, 5)
        ),
        action_mask=jnp.ones_like(fit, dtype=jnp.bool_),
        fit_replica_returns_by_action=replicas,
        evaluation_replica_returns_by_action=replicas,
    )


def test_base_and_latent_updates_are_separate_finite_transactions() -> None:
    import jax

    from src.delta_zsc.optimizer import init_adam
    from src.delta_zsc.storage import pytree_fingerprint
    from src.delta_zsc.training import environment_minibatch_schedule, training_update

    _, model, base, latent = _setup()
    batch = _batch(model, base, latent)
    schedule = environment_minibatch_schedule(
        jax.random.PRNGKey(4),
        environment_count=4,
        minibatches_per_epoch=1,
        update_epochs=1,
    )
    updated_base, updated_latent, _, _, metrics = training_update(
        model=model,
        base_params=base,
        latent_params=latent,
        base_optimizer_state=init_adam(base),
        latent_optimizer_state=init_adam(latent),
        batch=batch,
        anchors=_anchors(),
        schedule=schedule,
        total_optimizer_steps=10,
    )
    assert pytree_fingerprint(updated_base) != pytree_fingerprint(base)
    assert pytree_fingerprint(updated_latent) != pytree_fingerprint(latent)
    assert float(metrics["ppo"]["base_update_applied"]) == 1.0
    assert float(metrics["latent"]["latent_update_applied"]) == 1.0
    assert np.isfinite(float(metrics["latent"]["latent_composite_nll"]))
    assert float(metrics["latent"]["latent_decision_observations"]) == 2.0



def test_outer_transaction_commits_latent_before_ppo(monkeypatch) -> None:
    """CRN labels must be scored against their collection-time base tree."""

    import jax.numpy as jnp
    import src.delta_zsc.training as training

    events: list[str] = []

    def fake_latent(**kwargs):
        assert kwargs["base_params"] == "base-collection"
        assert kwargs["latent_params"] == "latent-0"
        events.append("latent")
        return "latent-1", "latent-state-1", {"latent_update_applied": 1.0}

    def fake_base(**kwargs):
        assert events == ["latent"]
        assert kwargs["base_params"] == "base-collection"
        assert kwargs["latent_params"] == "latent-1"
        events.append("ppo")
        return "base-1", "base-state-1", {"base_update_applied": 1.0}

    monkeypatch.setattr(training, "update_latent_model", fake_latent)
    monkeypatch.setattr(training, "update_base_policy", fake_base)
    monkeypatch.setattr(training, "slice_rollout_lanes", lambda batch, indexes: batch)

    result = training.training_update(
        model=object(),
        base_params="base-collection",
        latent_params="latent-0",
        base_optimizer_state="base-state-0",
        latent_optimizer_state="latent-state-0",
        batch=object(),
        anchors=object(),
        schedule=jnp.zeros((1, 1, 1), dtype=jnp.int32),
        total_optimizer_steps=1,
    )
    assert events == ["latent", "ppo"]
    assert result[:4] == (
        "base-1",
        "latent-1",
        "base-state-1",
        "latent-state-1",
    )

def test_response_only_variant_never_uses_decision_anchor_channel() -> None:
    from src.delta_zsc.losses import latent_composite_loss

    _, model, base, latent = _setup("response_only")
    result = latent_composite_loss(
        model, latent, base, _batch(model, base, latent), _anchors()
    )
    assert float(result.metrics["latent_decision_observations"]) == 0.0


def test_anchor_schema_contains_no_stale_posterior_or_comparator() -> None:
    from src.delta_zsc.types import AnchorBatch

    assert set(AnchorBatch._fields) == {
        "time_indexes",
        "lane_indexes",
        "fit_returns_by_action",
        "evaluation_returns_by_action",
        "measurement_covariances",
        "action_mask",
        "fit_replica_returns_by_action",
        "evaluation_replica_returns_by_action",
    }
