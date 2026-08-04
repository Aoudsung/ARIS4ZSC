"""Exact-filter, response-likelihood, isolation, and model-shape tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

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
    exact_bayes_filter_step,
    sticky_predictive,
    sticky_transition_matrix,
)
from src.path_c.response_decoder import (  # noqa: E402
    add_component_axis,
    response_decoder_class,
)
from src.path_c.response_targets import (  # noqa: E402
    PartnerResponseTargets,
    ResponsePrediction,
    component_joint_log_probability,
    extract_partner_response_targets,
    mixture_response_loss,
    official_partner_observation_planes,
    pairwise_component_response_divergence,
)
from src.path_c.task_encoder import official_partner_channel_indexes  # noqa: E402
from src.path_c.training import (  # noqa: E402
    capability_auxiliary_objective,
    compute_loss,
)
from src.path_c.types import RolloutBatch  # noqa: E402


OBSERVATION_SHAPE = (5, 5, 39)


def test_response_event_likelihood_has_no_task_feature_input() -> None:
    parameters = inspect.signature(response_decoder_class().__call__).parameters
    assert "task_features" not in parameters


def _model(
    component_count: int = 4, *, batch_size: int = 2, variant: str = "b2"
):
    model = build_model(
        observation_shape=OBSERVATION_SHAPE,
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        capability_dim=4,
        protocol_components=component_count,
        component_embedding_dim=4,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        response_hidden_dim=8,
        modulation_rank=2,
        action_embedding_dim=4,
        method_variant=variant,
    )
    state = initial_policy_state(
        batch_size=batch_size,
        observation_shape=OBSERVATION_SHAPE,
        action_count=6,
        task_hidden_dim=8,
        capability_hidden_dim=8,
        capability_dim=4,
        component_embedding_dim=4,
        protocol_components=component_count,
    )
    observation = jnp.zeros((batch_size,) + OBSERVATION_SHAPE, dtype=jnp.float32)
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(component_count),
        example_state=state,
        example_observation=observation,
    )
    return model, state, params


@pytest.mark.parametrize("component_count", (2, 4, 8))
def test_sticky_transition_is_row_stochastic_and_exact(component_count: int) -> None:
    matrix = np.asarray(sticky_transition_matrix(component_count))
    np.testing.assert_allclose(matrix.sum(axis=-1), 1.0, atol=1e-7)
    np.testing.assert_allclose(
        np.full(component_count, 1.0 / component_count) @ matrix,
        np.full(component_count, 1.0 / component_count),
        atol=1e-7,
    )
    one_hot = np.eye(component_count)[1]
    np.testing.assert_allclose(np.asarray(sticky_predictive(one_hot)), matrix[1])
    assert float(np.asarray(sticky_predictive(one_hot)).sum()) == pytest.approx(1.0)


def test_exact_filter_matches_a_manual_multistep_recursion() -> None:
    transition = np.asarray(sticky_transition_matrix(4), dtype=np.float64)
    posterior = jnp.full((4,), 0.25)
    manual = np.full((4,), 0.25)
    emissions = np.asarray([[2.0, 0.0, -1.0, 0.5], [-1.0, 3.0, 0.0, 0.0]])
    for emission in emissions:
        manual = manual @ transition
        manual *= np.exp(emission)
        manual /= manual.sum()
        posterior = exact_bayes_filter_step(posterior, jnp.asarray(emission))
        np.testing.assert_allclose(np.asarray(posterior), manual, atol=1e-6)


@pytest.mark.parametrize("prefix", ((3,), (2, 3), (6, 2, 3)))
def test_component_axis_is_inserted_explicitly_for_every_registered_prefix(prefix) -> None:
    value = jnp.arange(np.prod(prefix) * 5, dtype=jnp.float32).reshape(prefix + (5,))
    expanded = add_component_axis(value, 4)
    assert expanded.shape == prefix + (4, 5)
    for component in range(4):
        np.testing.assert_array_equal(np.asarray(expanded[..., component, :]), value)


@pytest.mark.parametrize("component_count", (2, 4, 8))
def test_model_initialization_and_response_axes_work_for_k_sensitivity(
    component_count: int,
) -> None:
    model, state, params = _model(component_count)
    observation = jnp.zeros((2,) + OBSERVATION_SHAPE)
    next_state, output, response = model.apply(
        {"params": params}, state, observation, method=model.initialize_all
    )
    assert next_state.protocol_carry.shape == (2, component_count)
    assert output.protocol_probabilities.shape == (2, component_count)
    assert response.visibility_logit.shape == (2, component_count)
    assert response.relative_position_logits.shape == (2, component_count, 26)
    assert response.inventory_logits.shape[:3] == (2, component_count, 5)


@pytest.mark.parametrize("component_count", (2, 4, 8))
def test_one_hot_component_intervention_uses_shared_actor_and_critic_heads(
    component_count: int,
) -> None:
    model, state, params = _model(component_count)
    observation = jax.random.normal(
        jax.random.PRNGKey(700 + component_count), (2,) + OBSERVATION_SHAPE
    )
    _, output = model.apply(
        {"params": params},
        state,
        observation,
        method=model.component_intervention_step,
    )
    assert output.protocol_probabilities.shape == (2, component_count)
    assert output.policy_logits.shape == (2, component_count, 6)
    assert output.raw_q1.shape == output.raw_q2.shape == (2, component_count, 6)
    assert output.action_signatures.shape == (2, component_count, 6)
    np.testing.assert_allclose(
        np.asarray(output.action_signatures).mean(axis=-1), 0.0, atol=1.0e-6
    )
    _, response = model.apply(
        {"params": params},
        state,
        observation,
        jnp.zeros((2,), dtype=jnp.int32),
        method=model.response_intervention_step,
    )
    assert response.visibility_logit.shape == (2, component_count)


def test_joint_mixture_marginalizes_the_shared_component_once() -> None:
    prediction = ResponsePrediction(
        posterior_log_probabilities=jnp.log(jnp.asarray([[0.5, 0.5]])),
        visibility_logit=jnp.asarray([[6.0, -6.0]]),
        relative_position_logits=jnp.zeros((1, 2, 26)),
        direction_logits=jnp.zeros((1, 2, 4)),
        inventory_logits=jnp.zeros((1, 2, 1, 4)),
        interaction_change_logit=jnp.asarray([[-6.0, 6.0]]),
    )
    targets = PartnerResponseTargets(
        visibility=jnp.asarray([1.0]),
        relative_position=jnp.asarray([0]),
        direction=jnp.asarray([0]),
        inventory=jnp.asarray([[0]]),
        interaction_change=jnp.asarray([1.0]),
        visible_mask=jnp.asarray([1.0]),
        event_mask=jnp.asarray([1.0]),
    )
    component_logp = component_joint_log_probability(prediction, targets)
    manual = -jax.scipy.special.logsumexp(
        prediction.posterior_log_probabilities + component_logp, axis=-1
    )[0]
    loss = mixture_response_loss(prediction, targets)
    assert float(loss.total) == pytest.approx(float(manual), abs=1e-6)
    # Opposite components explain visibility and event; a separately
    # marginalized head objective would be spuriously small.
    separate = float(loss.visibility + loss.interaction_change)
    assert float(loss.total) > separate + 3.0


def test_pairwise_response_divergence_detects_distinct_components() -> None:
    common = dict(
        posterior_log_probabilities=jnp.log(jnp.asarray([[0.5, 0.5]])),
        relative_position_logits=jnp.zeros((1, 2, 26)),
        direction_logits=jnp.zeros((1, 2, 4)),
        inventory_logits=jnp.zeros((1, 2, 1, 4)),
        interaction_change_logit=jnp.zeros((1, 2)),
    )
    same = ResponsePrediction(
        visibility_logit=jnp.zeros((1, 2)),
        **common,
    )
    distinct = ResponsePrediction(
        visibility_logit=jnp.asarray([[5.0, -5.0]]),
        **common,
    )
    assert float(pairwise_component_response_divergence(same)) == pytest.approx(
        0.0, abs=1.0e-7
    )
    assert float(pairwise_component_response_divergence(distinct)) > 0.9


def test_task_recurrence_is_invariant_to_current_partner_planes() -> None:
    model, state, params = _model()
    base = jnp.zeros((3, 2) + OBSERVATION_SHAPE)
    partner_channels = list(official_partner_channel_indexes(39))
    changed = base.at[1:, ..., partner_channels].set(1.0)
    actions = jnp.zeros((3, 2), dtype=jnp.int32)
    starts = jnp.asarray([[True, True], [False, False], [False, False]])
    drops = jnp.zeros((3, 2), dtype=jnp.bool_)
    _, original = model.apply(
        {"params": params}, state, base, actions, starts, drops, method=model.sequence
    )
    _, altered = model.apply(
        {"params": params}, state, changed, actions, starts, drops, method=model.sequence
    )
    np.testing.assert_array_equal(original.task_features, altered.task_features)


def test_b0_keeps_current_partner_geometry_memoryless_while_r0_uses_full_history() -> None:
    partner_channels = list(official_partner_channel_indexes(39))
    base = jnp.zeros((1,) + OBSERVATION_SHAPE).at[0, 2, 2, 0].set(1.0)
    changed = base.at[0, 2, 3, partner_channels[0]].set(1.0)

    b0, b0_state, b0_params = _model(batch_size=1, variant="b0")
    _, b0_base = b0.apply(
        {"params": b0_params}, b0_state, base, False, method=b0.step
    )
    _, b0_changed = b0.apply(
        {"params": b0_params}, b0_state, changed, False, method=b0.step
    )
    np.testing.assert_array_equal(b0_base.task_features, b0_changed.task_features)
    assert not np.allclose(b0_base.instant_partner, b0_changed.instant_partner)
    assert not np.allclose(b0_base.policy_logits, b0_changed.policy_logits)

    r0, r0_state, r0_params = _model(batch_size=1, variant="r0")
    _, r0_base = r0.apply(
        {"params": r0_params}, r0_state, base, False, method=r0.step
    )
    _, r0_changed = r0.apply(
        {"params": r0_params}, r0_state, changed, False, method=r0.step
    )
    assert not np.allclose(r0_base.task_features, r0_changed.task_features)
    # R0 has the same parameter capacity but its explicit instantaneous branch
    # is inert; current partner state enters through the full-history task GRU.
    assert np.allclose(r0_base.context_summary, r0_changed.context_summary)


def test_visibility_transition_is_not_double_counted_as_interaction_event() -> None:
    planes = official_partner_observation_planes(39)
    previous = jnp.zeros(OBSERVATION_SHAPE)
    visible = previous.at[2, 3, planes.visibility_channel].set(1.0)
    visible = visible.at[2, 3, planes.inventory_channels[0]].set(1.0)
    appeared = extract_partner_response_targets(previous, visible, planes=planes)
    assert float(appeared.visibility) == 1.0
    assert float(appeared.interaction_change) == 0.0
    assert float(appeared.event_mask) == 0.0

    changed_inventory = visible.at[2, 3, planes.inventory_channels[0]].set(0.0)
    changed_inventory = changed_inventory.at[
        2, 3, planes.inventory_channels[1]
    ].set(1.0)
    inventory_event = extract_partner_response_targets(
        visible, changed_inventory, planes=planes
    )
    assert float(inventory_event.interaction_change) == 1.0
    assert float(inventory_event.event_mask) == 1.0


def test_capability_publishes_only_at_step_16_while_filter_updates_each_step() -> None:
    model, state, params = _model(batch_size=1)
    observations = jax.random.normal(
        jax.random.PRNGKey(99), (17, 1) + OBSERVATION_SHAPE
    )
    actions = jnp.zeros((17, 1), dtype=jnp.int32)
    starts = jnp.zeros((17, 1), dtype=jnp.bool_).at[0].set(True)
    _, output = model.apply(
        {"params": params},
        state,
        observations,
        actions,
        starts,
        jnp.zeros_like(starts),
        method=model.sequence,
    )
    np.testing.assert_allclose(np.asarray(output.capability[:15]), 0.0, atol=0.0)
    assert not np.allclose(np.asarray(output.capability[15]), 0.0)
    prior = jnp.full((4,), 0.25)
    first = exact_bayes_filter_step(prior, jnp.asarray([2.0, 0.0, 0.0, 0.0]))
    second = exact_bayes_filter_step(first, jnp.asarray([0.0, 2.0, 0.0, 0.0]))
    assert not np.allclose(np.asarray(first), np.asarray(prior))
    assert not np.allclose(np.asarray(second), np.asarray(first))


def test_capability_is_stable_between_windows_and_resets_at_episode_boundary() -> None:
    model, state, params = _model(batch_size=1)
    observations = jax.random.normal(
        jax.random.PRNGKey(100), (20, 1) + OBSERVATION_SHAPE
    )
    actions = jnp.zeros((20, 1), dtype=jnp.int32)
    starts = jnp.zeros((20, 1), dtype=jnp.bool_).at[0].set(True).at[17].set(True)
    _, output = model.apply(
        {"params": params},
        state,
        observations,
        actions,
        starts,
        jnp.zeros_like(starts),
        method=model.sequence,
    )
    published = np.asarray(output.capability[:, 0])
    np.testing.assert_allclose(published[:15], 0.0, atol=0.0)
    np.testing.assert_allclose(published[15], published[16], atol=0.0)
    np.testing.assert_allclose(published[17:], 0.0, atol=0.0)


def test_capability_auxiliary_makes_the_zero_representation_nonoptimal() -> None:
    time_count, lane_count = 16, 2
    observations = jnp.zeros(
        (time_count, lane_count) + OBSERVATION_SHAPE, dtype=jnp.float32
    )
    starts = jnp.zeros((time_count, lane_count), dtype=jnp.bool_).at[0].set(True)
    mask = jnp.ones((time_count, lane_count), dtype=jnp.float32)
    values = jnp.zeros((time_count, lane_count, 4), dtype=jnp.float32)

    def prediction(candidate):
        return capability_auxiliary_objective(
            capability_sequence=candidate,
            observations=observations,
            initial_previous_observation=jnp.zeros(
                (lane_count,) + OBSERVATION_SHAPE, dtype=jnp.float32
            ),
            initial_steps=jnp.zeros((lane_count,), dtype=jnp.int32),
            episode_starts=starts,
            transition_mask=mask,
            variance_floor=0.05,
        )[0]

    zero_loss, gradient = jax.value_and_grad(prediction)(values)
    improved = values.at[-1, :, 0].set(-1.0)
    _, variance_loss, metrics = capability_auxiliary_objective(
        capability_sequence=values,
        observations=observations,
        initial_previous_observation=jnp.zeros(
            (lane_count,) + OBSERVATION_SHAPE, dtype=jnp.float32
        ),
        initial_steps=jnp.zeros((lane_count,), dtype=jnp.int32),
        episode_starts=starts,
        transition_mask=mask,
        variance_floor=0.05,
    )
    assert float(zero_loss) == pytest.approx(1.0)
    assert float(prediction(improved)) == pytest.approx(0.0)
    assert float(jnp.linalg.norm(gradient)) > 0.0
    assert float(variance_loss) > 0.0
    assert float(metrics["capability_publication_count"]) == 2.0
    assert float(metrics["capability_published_partner_count"]) == 2.0


def test_capability_variance_floor_is_computed_between_partner_runs() -> None:
    time_count, lane_count = 16, 2
    observations = jnp.zeros(
        (time_count, lane_count) + OBSERVATION_SHAPE, dtype=jnp.float32
    )
    starts = jnp.zeros((time_count, lane_count), dtype=jnp.bool_).at[0].set(True)
    values = jnp.zeros((time_count, lane_count, 4), dtype=jnp.float32)
    values = values.at[-1, 0].set(-0.1).at[-1, 1].set(0.1)

    def floor(ids):
        return capability_auxiliary_objective(
            capability_sequence=values,
            observations=observations,
            initial_previous_observation=jnp.zeros(
                (lane_count,) + OBSERVATION_SHAPE, dtype=jnp.float32
            ),
            initial_steps=jnp.zeros((lane_count,), dtype=jnp.int32),
            episode_starts=starts,
            transition_mask=jnp.ones((time_count, lane_count)),
            partner_run_ids=jnp.broadcast_to(
                jnp.asarray(ids, dtype=jnp.int32), (time_count, lane_count)
            ),
            variance_floor=0.05,
        )

    # Production static-pool IDs use the 10_000 + member convention.  The
    # variance grouping must be invariant to that source-specific offset.
    _, same_partner_floor, same_metrics = floor([10_007, 10_007])
    _, distinct_partner_floor, distinct_metrics = floor([10_007, 10_008])
    assert float(same_partner_floor) > 0.0
    assert float(distinct_partner_floor) == pytest.approx(0.0, abs=1.0e-7)
    assert float(same_metrics["capability_published_partner_count"]) == 1.0
    assert float(distinct_metrics["capability_published_partner_count"]) == 2.0


def test_registered_mechanism_ablations_change_only_the_intended_live_wires() -> None:
    observation = jax.random.normal(jax.random.PRNGKey(808), (1,) + OBSERVATION_SHAPE)

    deterministic, state, params = _model(batch_size=1, variant="deterministic_context")
    _, deterministic_output = deterministic.apply(
        {"params": params}, state, observation, False, method=deterministic.step
    )
    np.testing.assert_allclose(
        np.asarray(deterministic_output.protocol_probabilities), 0.25, atol=0.0
    )

    no_capability, state, params = _model(batch_size=1, variant="no_capability")
    _, no_capability_output = no_capability.apply(
        {"params": params}, state, observation, False, method=no_capability.step
    )
    np.testing.assert_allclose(
        np.asarray(no_capability_output.capability), 0.0, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(no_capability_output.context_summary[..., 32:36]),
        0.0,
        atol=0.0,
    )


def _tiny_batch(state, observations) -> RolloutBatch:
    time_count, batch_size = 2, 2
    transition_shape = (time_count, batch_size)
    return RolloutBatch(
        observations=observations,
        response_next_observations=observations[1:],
        previous_actions=jnp.zeros((time_count + 1, batch_size), dtype=jnp.int32),
        episode_starts=jnp.zeros((time_count + 1, batch_size), dtype=jnp.bool_),
        action_keys=jnp.zeros((time_count + 1, batch_size, 2), dtype=jnp.uint32),
        context_dropout_masks=jnp.zeros(
            (time_count + 1, batch_size), dtype=jnp.bool_
        ),
        actions=jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        rewards=jnp.ones(transition_shape),
        official_shaped_rewards=jnp.zeros(transition_shape),
        official_shaping_factors=jnp.ones(transition_shape),
        shaped_rewards=jnp.ones(transition_shape),
        dones=jnp.zeros(transition_shape, dtype=jnp.bool_),
        old_log_probabilities=jnp.full(transition_shape, -np.log(6.0)),
        old_values=jnp.zeros((time_count + 1, batch_size)),
        behavior_probabilities=jnp.full(transition_shape + (6,), 1.0 / 6.0),
        ppo_mask=jnp.ones(transition_shape),
        partner_sources=jnp.zeros(transition_shape, dtype=jnp.int32),
        partner_members=jnp.zeros(transition_shape, dtype=jnp.int32),
        partner_family_ids=jnp.zeros(transition_shape, dtype=jnp.int32),
        partner_checkpoint_stages=jnp.zeros(transition_shape),
        partner_run_ids=jnp.zeros(transition_shape, dtype=jnp.int32),
        initial_policy_state=state,
        initial_target_policy_state=state,
    )


def test_combined_b1_objective_has_finite_live_gradients() -> None:
    model, state, params = _model()
    observations = jax.random.uniform(
        jax.random.PRNGKey(101), (3, 2) + OBSERVATION_SHAPE
    )
    batch = _tiny_batch(state, observations)
    config = SimpleNamespace(
        method_variant="b1",
        ppo=SimpleNamespace(
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_clip_epsilon=0.2,
            value_weight=0.5,
            entropy_weight=0.01,
            normalize_advantages=True,
        ),
        loss_v2=SimpleNamespace(
            signature_weight=1.0,
            response_weight=1.0,
            separation_weight=0.1,
            rank_hinge_margin=0.1,
            rank_hinge_advantage_gap=2.0,
            decision_policy_weight=0.25,
            decision_policy_temperature=1.0,
            capability_consistency_weight=0.01,
        ),
    )

    def objective(candidate):
        return compute_loss(
            model=model,
            params=candidate,
            batch=batch,
            config=config,
            anchors=None,
            separation_terms=None,
        ).total

    value, gradient = jax.value_and_grad(objective)(params)
    assert bool(jnp.isfinite(value))
    leaves = jax.tree_util.tree_leaves(gradient)
    assert leaves and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))) > 0.0
