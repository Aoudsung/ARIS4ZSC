"""Unit tests for the three new DEPI core components.

Specification entries covered (docs/METHOD_SPEC.md):
- §2.1/§2.3 synthetic-HMM Bayes filtering recursion: the exact reference
  filter ``posterior ∝ (T^T pi_{t-1}) ⊙ likelihood`` with the parameter-free
  sticky transition (0.9 self / 0.1/3 jump) and uniform-prior episode reset.
- §1.1/§1.2 structural input-layer isolation of the task pathway and the
  prior-context replacement under context dropout.
- §3.3 the combined four-loss objective is computable in one pass with finite
  gradients over every active parameter subtree.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from src.path_c.model import (  # noqa: E402
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.protocol_encoder import (  # noqa: E402
    PROTOCOL_COMPONENTS,
    STICKY_JUMP_PROBABILITY,
    STICKY_SELF_PROBABILITY,
    exact_bayes_filter_step,
    sticky_predictive,
    sticky_predictive_log_probabilities,
    uniform_prior,
)
from src.path_c.training import compute_loss  # noqa: E402
from src.path_c.types import RolloutBatch, SeparationTerms  # noqa: E402


OBSERVATION_SHAPE = (5, 5, 39)
ACTION_COUNT = 6


def _small_model():
    model = build_model(
        observation_shape=OBSERVATION_SHAPE,
        action_count=ACTION_COUNT,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        protocol_hidden_dim=8,
        capability_dim=4,
        protocol_components=4,
        component_embedding_dim=4,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        response_hidden_dim=8,
        modulation_rank=2,
        action_embedding_dim=4,
    )
    state = initial_policy_state(
        batch_size=2,
        observation_shape=OBSERVATION_SHAPE,
        action_count=ACTION_COUNT,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        protocol_hidden_dim=8,
        capability_dim=4,
        component_embedding_dim=4,
        protocol_components=4,
    )
    return model, state


def test_synthetic_hmm_bayes_filter_recursion_matches_spec() -> None:
    # §2.1 frozen sticky transition: 0.9 self-stay, 0.1/3 jump to each other.
    assert float(STICKY_SELF_PROBABILITY) == pytest.approx(0.9)
    assert float(STICKY_JUMP_PROBABILITY) == pytest.approx(0.1 / 3.0)
    transition = np.full((4, 4), 0.1 / 3.0, dtype=np.float64)
    np.fill_diagonal(transition, 0.9)

    predictive = np.asarray(sticky_predictive(jnp.asarray([0.7, 0.1, 0.1, 0.1])))
    np.testing.assert_allclose(predictive, transition.T @ np.asarray(
        [0.7, 0.1, 0.1, 0.1]), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(sticky_predictive_log_probabilities(
            jnp.asarray([0.7, 0.1, 0.1, 0.1])
        )),
        np.log(predictive),
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(uniform_prior()), 0.25, atol=1e-7)

    # §2.3 five-step filtering recursion against an independent manual Bayes
    # computation: posterior_t ∝ (T^T pi_{t-1}) ⊙ likelihood_t.
    likelihoods = np.exp(
        np.asarray(
            [
                [2.0, 0.0, -1.0, 0.5],
                [1.5, 0.5, 0.0, -2.0],
                [-1.0, 3.0, 0.0, 0.0],
                [0.0, 2.5, 1.0, -0.5],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
    )
    manual = np.full(4, 0.25, dtype=np.float64)
    posterior = uniform_prior()
    for likelihood in likelihoods:
        manual = transition.T @ manual
        manual = manual * likelihood
        manual = manual / manual.sum()
        posterior = exact_bayes_filter_step(
            posterior, jnp.asarray(np.log(likelihood), dtype=jnp.float32)
        )
        np.testing.assert_allclose(np.asarray(posterior), manual, atol=1e-5)
        assert float(jnp.sum(posterior)) == pytest.approx(1.0, abs=1e-6)
    # Strong evidence must concentrate the posterior on the supported regime.
    concentrated = exact_bayes_filter_step(
        uniform_prior(), jnp.asarray([8.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    )
    assert float(concentrated[0]) > 0.99
    # Uniform likelihoods leave the predictive (sticky-smoothed) prior intact.
    neutral = exact_bayes_filter_step(
        jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.zeros(4, dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        np.asarray(neutral),
        np.asarray(sticky_predictive(jnp.asarray([1.0, 0.0, 0.0, 0.0]))),
        atol=1e-6,
    )
    assert PROTOCOL_COMPONENTS == 4


def test_task_pathway_input_isolation_and_prior_context_dropout() -> None:
    model, state = _small_model()
    observations = jax.random.uniform(
        jax.random.PRNGKey(3), (3, 2) + OBSERVATION_SHAPE
    )
    starts = jnp.asarray([[True, True], [False, False], [False, False]])

    # §1.1: the task pathway only reads the current observation, so different
    # previous-action histories leave task_features untouched while the
    # capability/protocol pathways (which read a^ego_{t-1}) must respond.
    _, output_a = model.apply(
        {"params": _params(model, state, observations[0])},
        state, observations, jnp.zeros((3, 2), dtype=jnp.int32), starts,
        jnp.zeros((3, 2), dtype=jnp.bool_), method=model.sequence,
    )
    _, output_b = model.apply(
        {"params": _params(model, state, observations[0])},
        state, observations, jnp.full((3, 2), 5, dtype=jnp.int32), starts,
        jnp.zeros((3, 2), dtype=jnp.bool_), method=model.sequence,
    )
    np.testing.assert_allclose(
        output_a.task_features, output_b.task_features, atol=1e-6
    )
    assert not np.allclose(
        np.asarray(output_a.context_summary),
        np.asarray(output_b.context_summary),
        atol=1e-6,
    )

    # §1.2: context dropout replaces (u, c_t) with the prior context and never
    # touches the task pathway.
    params = _params(model, state, observations[0])
    _, dropped = model.apply(
        {"params": params}, state, observations[0],
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.ones((2,), dtype=jnp.bool_), method=model.step,
    )
    _, kept = model.apply(
        {"params": params}, state, observations[0],
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.bool_), method=model.step,
    )
    np.testing.assert_allclose(
        dropped.task_features, kept.task_features, atol=1e-6
    )
    np.testing.assert_allclose(dropped.capability, 0.0, atol=1e-7)
    np.testing.assert_allclose(
        dropped.context_summary[:, :4], 0.0, atol=1e-7
    )
    components = np.asarray(
        params["protocol_component_embeddings"]["embedding"]
    )
    np.testing.assert_allclose(
        np.asarray(dropped.context_summary[:, 4:]),
        components.mean(axis=0)[None],
        atol=1e-6,
    )
    # The posterior itself is unaffected by dropout (§1.2 drops only the
    # behavior context, not the filter state).
    np.testing.assert_allclose(
        dropped.protocol_probabilities, kept.protocol_probabilities, atol=1e-6
    )


def _params(model: Any, state: Any, example_observation: Any) -> Any:
    return initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(0),
        example_state=state,
        example_observation=example_observation,
    )


def test_combined_four_loss_is_computable_with_finite_gradients() -> None:
    model, state = _small_model()
    time_count, batch_size = 3, 2
    key = jax.random.PRNGKey(11)
    observations = jax.random.uniform(key, (time_count + 1, batch_size) + OBSERVATION_SHAPE)
    batch = RolloutBatch(
        observations=observations,
        response_next_observations=observations[1:],
        previous_actions=jnp.zeros((time_count + 1, batch_size), dtype=jnp.int32),
        episode_starts=jnp.zeros((time_count + 1, batch_size), dtype=jnp.bool_),
        action_keys=jnp.zeros((time_count + 1, batch_size, 2), dtype=jnp.uint32),
        context_dropout_masks=jnp.zeros((time_count + 1, batch_size), dtype=jnp.bool_),
        actions=jnp.asarray([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32),
        rewards=jnp.asarray([[0.0, 1.0], [2.0, -1.0], [3.0, 0.5]]),
        official_shaped_rewards=jnp.zeros((time_count, batch_size)),
        official_shaping_factors=jnp.ones((time_count, batch_size)),
        decision_regret_shaping=jnp.zeros((time_count, batch_size)),
        shaped_rewards=jnp.asarray([[0.0, 1.0], [2.0, -1.0], [3.0, 0.5]]),
        dones=jnp.zeros((time_count, batch_size), dtype=jnp.bool_),
        old_log_probabilities=jnp.full((time_count, batch_size), -np.log(6.0)),
        old_values=jnp.zeros((time_count + 1, batch_size)),
        behavior_probabilities=jnp.full((time_count, batch_size), 1.0 / 6.0),
        ppo_mask=jnp.ones((time_count, batch_size)),
        partner_codes=jnp.zeros((time_count, batch_size, 8)),
        partner_sources=jnp.zeros((time_count, batch_size), dtype=jnp.int32),
        partner_run_ids=jnp.zeros((time_count, batch_size), dtype=jnp.int32),
        initial_policy_state=state,
        initial_target_policy_state=state,
    )
    config = SimpleNamespace(
        ppo=SimpleNamespace(
            gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
            value_clip_epsilon=0.2, value_weight=0.5, entropy_weight=0.01,
            normalize_advantages=True,
        ),
        loss_v2=SimpleNamespace(
            signature_weight=1.0,
            response_weight=1.0,
            separation_weight=0.1,
            combined_policy_kl_threshold=0.04,
            rank_hinge_margin=0.1,
            rank_hinge_advantage_gap=2.0,
            separation_margin_scale=0.25,
        ),
    )
    params = _params(model, state, observations[0])

    # §3.3: one combined objective over all four losses (anchor signature and
    # separation contribute exactly zero while their paths are disabled).
    def total(candidate):
        return compute_loss(
            model=model, params=candidate, batch=batch, config=config,
            anchors=None, separation_terms=None,
        ).total

    loss_value = total(params)
    assert bool(jnp.isfinite(loss_value))
    bundle = compute_loss(
        model=model, params=params, batch=batch, config=config,
        anchors=None, separation_terms=None,
    )
    for name in (
        "ppo_total_loss", "response_total_loss", "signature_loss",
        "separation_loss", "combined_policy_kl",
    ):
        assert bool(jnp.isfinite(bundle.metrics[name])), name
    assert float(bundle.metrics["signature_loss"]) == 0.0
    assert float(bundle.metrics["separation_loss"]) == 0.0
    assert float(bundle.metrics["response_total_loss"]) > 0.0

    gradients = jax.grad(total)(params)
    leaves = jax.tree_util.tree_leaves(gradients)
    assert leaves and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    for subtree_name in (
        "task_encoder",
        "capability_encoder",
        "protocol_encoder",
        "protocol_component_embeddings",
        "universal_actor",
        "universal_critic",
        "response_decoder",
    ):
        subtree_leaves = jax.tree_util.tree_leaves(gradients[subtree_name])
        assert subtree_leaves, subtree_name
        assert all(
            bool(jnp.all(jnp.isfinite(leaf))) for leaf in subtree_leaves
        ), subtree_name
    total_norm = float(
        jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
    )
    assert total_norm > 0.0


def _tiny_batch(state: Any, observations: Any) -> RolloutBatch:
    time_count, batch_size = 3, 2
    return RolloutBatch(
        observations=observations,
        response_next_observations=observations[1:],
        previous_actions=jnp.zeros((time_count + 1, batch_size), dtype=jnp.int32),
        episode_starts=jnp.zeros((time_count + 1, batch_size), dtype=jnp.bool_),
        action_keys=jnp.zeros((time_count + 1, batch_size, 2), dtype=jnp.uint32),
        context_dropout_masks=jnp.zeros((time_count + 1, batch_size), dtype=jnp.bool_),
        actions=jnp.asarray([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32),
        rewards=jnp.asarray([[0.0, 1.0], [2.0, -1.0], [3.0, 0.5]]),
        official_shaped_rewards=jnp.zeros((time_count, batch_size)),
        official_shaping_factors=jnp.ones((time_count, batch_size)),
        decision_regret_shaping=jnp.zeros((time_count, batch_size)),
        shaped_rewards=jnp.asarray([[0.0, 1.0], [2.0, -1.0], [3.0, 0.5]]),
        dones=jnp.zeros((time_count, batch_size), dtype=jnp.bool_),
        old_log_probabilities=jnp.full((time_count, batch_size), -np.log(6.0)),
        old_values=jnp.zeros((time_count + 1, batch_size)),
        behavior_probabilities=jnp.full((time_count, batch_size), 1.0 / 6.0),
        ppo_mask=jnp.ones((time_count, batch_size)),
        partner_codes=jnp.zeros((time_count, batch_size, 8)),
        partner_sources=jnp.zeros((time_count, batch_size), dtype=jnp.int32),
        partner_run_ids=jnp.zeros((time_count, batch_size), dtype=jnp.int32),
        initial_policy_state=state,
        initial_target_policy_state=state,
    )


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        ppo=SimpleNamespace(
            gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
            value_clip_epsilon=0.2, value_weight=0.5, entropy_weight=0.01,
            normalize_advantages=True,
        ),
        loss_v2=SimpleNamespace(
            signature_weight=1.0,
            response_weight=1.0,
            separation_weight=0.1,
            combined_policy_kl_threshold=0.04,
            rank_hinge_margin=0.1,
            rank_hinge_advantage_gap=2.0,
            separation_margin_scale=0.25,
        ),
    )


def test_separation_terms_carry_a_live_param_dependent_loss() -> None:
    """§3.2/§3.4: L_separation is re-evaluated inside compute_loss on the
    current params, so its gradient is non-zero and reaches the protocol
    pathway that produces c_t (the eager-scalar regression is pinned out).

    Ownership note: c_t = Σ_k π_k m_k depends on the protocol encoder
    posterior and the component embeddings; the capability vector enters the
    actor summary but not c_t itself, so its separation gradient is
    structurally zero here.
    """

    model, state = _small_model()
    key = jax.random.PRNGKey(23)
    observations = jax.random.uniform(key, (4, 2) + OBSERVATION_SHAPE)
    batch = _tiny_batch(state, observations)
    config = _tiny_config()
    params = _params(model, state, observations[0])

    # Minimal synthetic matched pair payload: two equivalent pairs built from
    # the initial policy state probed with two different observations, so the
    # two protocol embeddings differ.
    probe_a = jax.random.uniform(jax.random.PRNGKey(29), (2,) + OBSERVATION_SHAPE)
    probe_b = jax.random.uniform(jax.random.PRNGKey(31), (2,) + OBSERVATION_SHAPE)
    terms = SeparationTerms(
        ego_state_a=state,
        ego_state_b=state,
        probe_observations=(probe_a, probe_b),
        equivalent_mask=jnp.asarray([True, True]),
        weights=jnp.ones((2,)),
        margin=jnp.asarray(0.5),
    )

    def separation_only(candidate: Any) -> Any:
        return compute_loss(
            model=model, params=candidate, batch=batch, config=config,
            anchors=None, separation_terms=terms,
        ).metrics["separation_loss"]

    value_before = float(separation_only(params))
    assert value_before > 0.0
    assert bool(jnp.isfinite(jnp.asarray(value_before)))

    # Param dependence: perturbing params changes the separation component
    # (a jit-external constant scalar could never do this).
    perturbed = jax.tree_util.tree_map(
        lambda leaf: leaf + 0.05 * jax.random.normal(
            jax.random.PRNGKey(37), leaf.shape
        ),
        params,
    )
    value_after = float(separation_only(perturbed))
    assert value_after != pytest.approx(value_before, abs=1e-9)

    # Non-zero gradient, finite everywhere, and owned by the §3.4 protocol
    # subtrees that produce c_t.
    gradients = jax.grad(separation_only)(params)
    leaves = jax.tree_util.tree_leaves(gradients)
    assert leaves and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    total_norm = float(
        jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
    )
    assert total_norm > 0.0
    for subtree_name in (
        "protocol_encoder",
        "protocol_component_embeddings",
    ):
        subtree_norm = float(
            jnp.sqrt(
                sum(
                    jnp.sum(jnp.square(leaf))
                    for leaf in jax.tree_util.tree_leaves(gradients[subtree_name])
                )
            )
        )
        assert subtree_norm > 0.0, subtree_name

    # The combined objective applies the frozen λ_S = 0.1 on the live term.
    bundle = compute_loss(
        model=model, params=params, batch=batch, config=config,
        anchors=None, separation_terms=terms,
    )
    assert float(bundle.metrics["separation_loss"]) == pytest.approx(value_before)
    assert bool(jnp.isfinite(bundle.total))


def test_separation_terms_none_keeps_the_component_at_zero() -> None:
    model, state = _small_model()
    key = jax.random.PRNGKey(41)
    observations = jax.random.uniform(key, (4, 2) + OBSERVATION_SHAPE)
    batch = _tiny_batch(state, observations)
    params = _params(model, state, observations[0])
    bundle = compute_loss(
        model=model, params=params, batch=batch, config=_tiny_config(),
        anchors=None, separation_terms=None,
    )
    assert float(bundle.metrics["separation_loss"]) == 0.0
