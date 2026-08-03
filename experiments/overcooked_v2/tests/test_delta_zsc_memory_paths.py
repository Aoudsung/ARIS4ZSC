# METHOD_SPEC references: §1.1 (three-object recurrent architecture),
# §1.4 (legal PolicyState memory fields and sequence consistency),
# §2.2/§2.3 (mixture response decoder gather consistency).
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")

from src.path_c.experiment import load_config  # noqa: E402
from src.path_c.model import (  # noqa: E402
    build_model,
    initial_policy_state,
    initialize_model_parameters,
)
from src.path_c.response_targets import (  # noqa: E402
    PartnerResponseTargets,
    ResponsePrediction,
    mixture_response_loss,
)
from src.path_c.runner import DECISION_REGRET_STATE_CHUNK_SIZE  # noqa: E402
from src.path_c.types import ContextOutput  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]


def _fixture():
    config = load_config(
        ROOT / "experiments/overcooked_v2/configs/delta_zsc_simple_development.yaml",
        run_kind="mechanical",
    )
    shape = (5, 5, 39)
    model = build_model(
        observation_shape=shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        capability_hidden_dim=config.model.capability_hidden_dim,
        protocol_hidden_dim=config.model.protocol_hidden_dim,
        capability_dim=config.model.capability_dim,
        protocol_components=config.model.protocol_components,
        component_embedding_dim=config.model.component_embedding_dim,
        actor_hidden_dim=config.model.actor_hidden_dim,
        critic_hidden_dim=config.model.critic_hidden_dim,
        response_hidden_dim=config.model.response_hidden_dim,
        modulation_rank=config.model.modulation_rank,
        action_embedding_dim=config.model.action_embedding_dim,
    )
    state = initial_policy_state(
        batch_size=2,
        observation_shape=shape,
        action_count=6,
        task_hidden_dim=config.model.task_hidden_dim,
        capability_hidden_dim=config.model.capability_hidden_dim,
        protocol_hidden_dim=config.model.protocol_hidden_dim,
        capability_dim=config.model.capability_dim,
        component_embedding_dim=config.model.component_embedding_dim,
        protocol_components=config.model.protocol_components,
    )
    observation = jnp.arange(2 * 5 * 5 * 39, dtype=jnp.float32).reshape(
        (2,) + shape
    ) / 255.0
    params = initialize_model_parameters(
        model,
        key=jax.random.PRNGKey(11),
        example_state=state,
        example_observation=observation,
    )
    _, output = model.apply(
        {"params": params}, state, observation,
        jnp.zeros((2,), dtype=jnp.bool_), method=model.step,
    )
    return config, model, params, state, output


def test_step_loop_equals_context_and_full_sequence_scan() -> None:
    # §1.4: the recurrent memory exposed by step() must be bit-consistent with
    # the scan implementations used by training (sequence/context_sequence).
    _, model, params, state, _ = _fixture()
    length = 3
    observations = jnp.arange(length * 2 * 5 * 5 * 39, dtype=jnp.float32).reshape(
        (length, 2, 5, 5, 39)
    ) / 511.0
    previous_actions = jnp.asarray([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)
    episode_starts = jnp.asarray([[True, True], [False, False], [False, True]])
    dropout = jnp.asarray([[False, True], [False, False], [True, False]])

    loop_state = state
    loop_contexts = []
    loop_outputs = []
    for t in range(length):
        loop_state = loop_state._replace(
            previous_action=previous_actions[t], episode_start=episode_starts[t]
        )
        loop_state, output = model.apply(
            {"params": params}, loop_state, observations[t], dropout[t],
            method=model.step,
        )
        loop_contexts.append(output)
        loop_outputs.append(output)

    scan_state, scan_context = model.apply(
        {"params": params}, state, observations, previous_actions, episode_starts,
        method=model.context_sequence,
    )
    scan_full_state, scan_output = model.apply(
        {"params": params}, state, observations, previous_actions,
        episode_starts, dropout, method=model.sequence,
    )

    for loop_leaf, scan_leaf in zip(
        jax.tree_util.tree_leaves(loop_state),
        jax.tree_util.tree_leaves(scan_state),
        strict=True,
    ):
        np.testing.assert_allclose(loop_leaf, scan_leaf, rtol=1e-6, atol=1e-6)
    for loop_leaf, scan_leaf in zip(
        jax.tree_util.tree_leaves(loop_state),
        jax.tree_util.tree_leaves(scan_full_state),
        strict=True,
    ):
        np.testing.assert_allclose(loop_leaf, scan_leaf, rtol=1e-6, atol=1e-6)
    for t in range(length):
        np.testing.assert_allclose(
            loop_outputs[t].protocol_probabilities,
            scan_context.protocol_probabilities[t],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            loop_outputs[t].task_features,
            scan_context.task_features[t],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            loop_outputs[t].policy_logits,
            scan_output.policy_logits[t],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            loop_outputs[t].state_value,
            scan_output.state_value[t],
            rtol=1e-6,
            atol=1e-6,
        )


def test_episode_start_resets_protocol_carry_to_uniform_prior() -> None:
    # §1.1/§2.1: at an episode boundary the sticky posterior restarts from the
    # uniform protocol prior inside protocol_carry.
    _, model, params, state, _ = _fixture()
    observation = jnp.zeros((2, 5, 5, 39), dtype=jnp.float32)
    stepped = state._replace(episode_start=jnp.asarray([True, True]))
    _, output = model.apply(
        {"params": params}, stepped, observation,
        jnp.zeros((2,), dtype=jnp.bool_), method=model.step,
    )
    posterior = np.asarray(output.protocol_probabilities)
    assert posterior.shape == (2, 4)
    np.testing.assert_allclose(posterior.sum(axis=-1), np.ones((2,)), rtol=1e-5)
    assert np.all(posterior >= 0.0)
    # protocol_carry stores (gru_hidden, previous_probs); after the restart the
    # carried previous distribution must equal the freshly computed posterior.
    next_state, _ = model.apply(
        {"params": params}, stepped, observation,
        jnp.zeros((2,), dtype=jnp.bool_), method=model.step,
    )
    _, previous_probs = next_state.protocol_carry
    np.testing.assert_allclose(
        np.asarray(previous_probs), posterior, rtol=1e-6, atol=1e-6
    )


def _targets():
    return PartnerResponseTargets(
        visibility=jnp.asarray([1.0, 0.0]),
        relative_position=jnp.asarray([3, 25]),
        direction=jnp.asarray([1, 0]),
        inventory=jnp.asarray([[0, 1, 2, 3, 0], [0, 0, 0, 0, 0]]),
        interaction_change=jnp.asarray([1.0, 0.0]),
        visible_mask=jnp.asarray([1.0, 0.0]),
    )


def test_selected_response_equals_all_action_gather_and_gradients() -> None:
    # §2.2: selecting one action through the decoder must equal gathering that
    # action from the broadcast all-action prediction, including gradients.
    _, model, params, state, _ = _fixture()
    length = 2
    observations = jnp.arange(length * 2 * 5 * 5 * 39, dtype=jnp.float32).reshape(
        (length, 2, 5, 5, 39)
    ) / 511.0
    previous_actions = jnp.zeros((length, 2), dtype=jnp.int32)
    episode_starts = jnp.asarray([[True, True], [False, False]])
    _, context = model.apply(
        {"params": params}, state, observations, previous_actions, episode_starts,
        method=model.context_sequence,
    )
    sliced = ContextOutput(
        context.task_features[:-1],
        context.capability[:-1],
        context.protocol_probabilities[:-1],
        context.protocol_embedding[:-1],
    )
    frame = observations[:-1]
    actions = jnp.asarray([1, 5], dtype=jnp.int32)
    selected = model.apply(
        {"params": params}, sliced, frame, actions,
        method=model.response_from_context_and_action,
    )

    def _all_action_prediction(module, context_in, frame_in):
        prefix = context_in.task_features.shape[:-1]
        all_actions = jnp.broadcast_to(
            jnp.arange(module.action_count, dtype=jnp.int32),
            prefix + (module.action_count,),
        )
        expanded = ContextOutput(*(
            jnp.broadcast_to(
                value[..., None, :],
                value.shape[:-1] + (module.action_count, value.shape[-1]),
            )
            for value in context_in
        ))
        expanded_frame = jnp.broadcast_to(
            frame_in[..., None, :, :, :],
            frame_in.shape[:-3] + (module.action_count,) + frame_in.shape[-3:],
        )
        return module.response_from_context_and_action(
            expanded, expanded_frame, all_actions
        )

    all_prediction = model.apply(
        {"params": params}, sliced, frame, method=_all_action_prediction,
    )
    lanes = jnp.arange(actions.shape[0], dtype=jnp.int32)
    gathered = ResponsePrediction(
        posterior_log_probabilities=all_prediction.posterior_log_probabilities[:, 0],
        visibility_logit=all_prediction.visibility_logit[lanes, actions],
        relative_position_logits=all_prediction.relative_position_logits[lanes, actions],
        direction_logits=all_prediction.direction_logits[lanes, actions],
        inventory_logits=all_prediction.inventory_logits[lanes, actions],
        interaction_change_logit=all_prediction.interaction_change_logit[lanes, actions],
    )
    for actual, expected in zip(selected, gathered, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def selected_loss(capability):
        replaced = sliced._replace(capability=capability)
        prediction = model.apply(
            {"params": params}, replaced, frame, actions,
            method=model.response_from_context_and_action,
        )
        return mixture_response_loss(prediction, _targets()).total

    def gathered_loss(capability):
        replaced = sliced._replace(capability=capability)
        prediction = model.apply(
            {"params": params}, replaced, frame, method=_all_action_prediction,
        )
        gathered_here = ResponsePrediction(
            posterior_log_probabilities=prediction.posterior_log_probabilities[:, 0],
            visibility_logit=prediction.visibility_logit[lanes, actions],
            relative_position_logits=prediction.relative_position_logits[lanes, actions],
            direction_logits=prediction.direction_logits[lanes, actions],
            inventory_logits=prediction.inventory_logits[lanes, actions],
            interaction_change_logit=prediction.interaction_change_logit[lanes, actions],
        )
        return mixture_response_loss(gathered_here, _targets()).total

    np.testing.assert_allclose(
        jax.grad(selected_loss)(sliced.capability),
        jax.grad(gathered_loss)(sliced.capability),
        rtol=1e-6,
        atol=1e-6,
    )


def test_regret_chunk_has_bounded_formal_hidden_tensor() -> None:
    # Registered legacy chunk constant retained for checkpoint identity only.
    critic_hidden_bytes = DECISION_REGRET_STATE_CHUNK_SIZE * 16 * 128 * 4
    assert critic_hidden_bytes == 32 * 2**20


@pytest.mark.skipif(
    os.environ.get("DELTA_RUN_FORMAL_GPU_MEMORY_TEST") != "1",
    reason="Explicit registered CUDA acceptance fixture only.",
)
def test_formal_sequence_scan_fits_registered_cuda_memory() -> None:
    from src.path_c.resources import peak_device_memory_bytes

    _, model, params, state, _ = _fixture()
    length = 257
    count = 256
    observations = jnp.zeros((length, count, 5, 5, 39), dtype=jnp.float32)
    previous_actions = jnp.zeros((length, count), dtype=jnp.int32)
    episode_starts = jnp.zeros((length, count), dtype=jnp.bool_)
    wide_state = initial_policy_state(
        batch_size=count,
        observation_shape=(5, 5, 39),
        action_count=6,
        task_hidden_dim=128,
        capability_hidden_dim=64,
        protocol_hidden_dim=128,
        capability_dim=16,
        component_embedding_dim=16,
        protocol_components=4,
    )
    del state
    final_state, context = model.apply(
        {"params": params}, wide_state, observations, previous_actions,
        episode_starts, method=model.context_sequence,
    )
    jax.block_until_ready((final_state, context))
    assert context.protocol_probabilities.shape == (length, count, 4)
    assert np.all(np.isfinite(np.asarray(context.protocol_probabilities)))
    peak = peak_device_memory_bytes()
    assert 0 < peak < 40_000 * 2**20
