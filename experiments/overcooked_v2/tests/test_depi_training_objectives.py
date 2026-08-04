"""Actor coupling, owner initialization, and post-update KL regressions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

from src.path_c.base_distillation import (  # noqa: E402
    OwnerBehaviorBatch,
    distill_owner_actor,
    owner_behavior_loss,
)
from src.path_c.model import (  # noqa: E402
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.storage import pytree_fingerprint  # noqa: E402
import src.path_c.training as training  # noqa: E402
from src.path_c.training import decision_policy_loss  # noqa: E402
from src.path_c.types import LossBundle, TrainingCoreState  # noqa: E402


def test_empirical_decision_target_has_a_live_actor_gradient() -> None:
    logits = jnp.zeros((2, 6), dtype=jnp.float32)
    returns = jnp.asarray(
        [[8.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 7.0, 0.0, 0.0, 0.0, 0.0]]
    )

    def objective(candidate):
        return decision_policy_loss(
            policy_logits=candidate,
            advantage_targets=returns,
            action_mask=jnp.ones_like(returns, dtype=jnp.bool_),
            temperature=1.0,
            confidence_weight=jnp.asarray([1.0, 0.5]),
        )

    before, gradient = jax.value_and_grad(objective)(logits)
    after = objective(logits - 0.5 * gradient)
    assert float(jnp.linalg.norm(gradient)) > 0.0
    assert float(after) < float(before)


def _small_model_and_params():
    kwargs = dict(
        observation_shape=(5, 5, 39),
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        capability_dim=4,
        protocol_components=4,
        component_embedding_dim=4,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        response_hidden_dim=8,
        modulation_rank=2,
        action_embedding_dim=4,
        method_variant="b2",
    )
    model = build_model(**kwargs)

    def state_factory(count):
        return initial_policy_state(
            batch_size=count,
            observation_shape=(5, 5, 39),
            action_count=6,
            task_hidden_dim=8,
            capability_hidden_dim=8,
            capability_dim=4,
            component_embedding_dim=4,
            protocol_components=4,
        )

    state = state_factory(2)
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(123),
        example_state=state,
        example_observation=jnp.zeros((2, 5, 5, 39)),
    )
    return model, state_factory, params


def test_owner_distillation_updates_task_and_actor_and_reduces_kl() -> None:
    model, state_factory, params = _small_model_and_params()
    observations = jax.random.normal(jax.random.PRNGKey(124), (3, 2, 5, 5, 39))
    owner_logits = jnp.zeros((2, 2, 6)).at[..., 0].set(5.0)
    batch = OwnerBehaviorBatch(
        observations=observations,
        previous_actions=jnp.zeros((3, 2), dtype=jnp.int32),
        episode_starts=jnp.zeros((3, 2), dtype=jnp.bool_).at[0].set(True),
        owner_logits=owner_logits,
        valid_mask=jnp.ones((2, 2)),
        source_members=jnp.zeros((2, 2), dtype=jnp.int32),
    )
    optimizer = optax.sgd(0.02)
    before = owner_behavior_loss(
        model=model, params=params, initial_state=state_factory(2), batch=batch
    )[0]
    next_params, _, _ = distill_owner_actor(
        model=model,
        params=params,
        batch=batch,
        initial_state_factory=state_factory,
        optimizer=optimizer,
        optimizer_state=optimizer.init(params),
        schedule=jnp.asarray([[0, 1]], dtype=jnp.int32),
    )
    after = owner_behavior_loss(
        model=model, params=next_params, initial_state=state_factory(2), batch=batch
    )[0]
    assert float(after) < float(before)
    assert pytree_fingerprint(next_params["task_encoder"]) != pytree_fingerprint(
        params["task_encoder"]
    )
    assert pytree_fingerprint(next_params["universal_actor"]) != pytree_fingerprint(
        params["universal_actor"]
    )
    for name in (
        "capability_encoder",
        "protocol_component_embeddings",
        "universal_critic",
        "response_decoder",
    ):
        assert pytree_fingerprint(next_params[name]) == pytree_fingerprint(params[name])


def test_combined_policy_kl_is_computed_from_candidate_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_loss(*, params, **unused):
        total = jnp.square(params["w"] - 1.0)
        return LossBundle(total=total, metrics={"total_loss": total})

    def fake_post_update_kl(*, params, **unused):
        return params["w"]

    monkeypatch.setattr(training, "compute_loss", fake_loss)
    monkeypatch.setattr(training, "post_update_combined_policy_kl", fake_post_update_kl)
    params = {"w": jnp.asarray(0.0)}
    optimizer = optax.sgd(0.1)
    core = TrainingCoreState(
        params=params,
        target_params=params,
        ppo_optimizer_state=optimizer.init(params),
    )
    updated, metrics = training.apply_training_core_update(
        model=None,
        core=core,
        optimizer=optimizer,
        batch=None,
        config=SimpleNamespace(
            method_variant="b1",
            loss_v2=SimpleNamespace(combined_policy_kl_threshold=0.04),
        ),
    )
    assert float(updated.params["w"]) == pytest.approx(0.2)
    assert float(metrics["combined_policy_kl"]) == pytest.approx(0.2)
    assert float(metrics["kl_early_stop"]) == 1.0
    assert float(metrics["nonfinite_failure"]) == 0.0
