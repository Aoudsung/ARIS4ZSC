from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _model():
    from src.cetr_zsc.config import ModelConfig
    from src.cetr_zsc.model import CetrModel

    config = SimpleNamespace(
        model=ModelConfig(task_hidden_dim=128, task_embedding_dim=128)
    )
    return CetrModel(config, (5, 5, 26), 6)


def _array(jnp, shape, offset):
    size = int(np.prod(shape))
    return (jnp.arange(size, dtype=jnp.float32) + float(offset)).reshape(shape)


def _official_tree(params):
    import jax.numpy as jnp

    def affine(source, offset):
        return {
            "kernel": _array(jnp, source["kernel"].shape, offset),
            "bias": _array(jnp, source["bias"].shape, offset + 1.0),
        }

    cnn = {}
    for index, source in enumerate(params["task_conv"]):
        cnn[f"Conv_{index}"] = affine(source, 10.0 * (index + 1))
    cnn["Dense_0"] = affine(params["task_dense"], 80.0)

    gru = {}
    for name, source_name, offset in (
        ("ir", "input_reset", 100.0),
        ("iz", "input_update", 110.0),
        ("in", "input_candidate", 120.0),
    ):
        gru[name] = affine(params["task_gru"][source_name], offset)
    for name, source_name, offset in (
        ("hr", "hidden_reset", 130.0),
        ("hz", "hidden_update", 140.0),
    ):
        gru[name] = {
            "kernel": _array(
                jnp, params["task_gru"][source_name]["kernel"].shape, offset
            )
        }
    gru["hn"] = affine(params["task_gru"]["hidden_candidate"], 150.0)

    return {
        "CNN_0": cnn,
        "LayerNorm_0": {
            "scale": _array(jnp, params["task_norm"]["scale"].shape, 160.0),
            "bias": _array(jnp, params["task_norm"]["bias"].shape, 161.0),
        },
        "ScannedRNN_0": {"GRUCell_1": gru},
        "Dense_0": affine(params["actor_trunk"], 170.0),
        "Dense_1": affine(params["actor"], 180.0),
    }


def test_parameter_tree_keys_and_shapes():
    import jax

    from src.cetr_zsc.model import OFFICIAL_CONV_STACK

    params = _model().init_parameters(jax.random.PRNGKey(0))
    assert set(params) == {
        "task_conv",
        "task_dense",
        "task_norm",
        "task_gru",
        "actor_trunk",
        "actor",
        "value_trunk",
        "value",
    }
    assert OFFICIAL_CONV_STACK == (
        ((1, 1), 128),
        ((1, 1), 128),
        ((1, 1), 8),
        ((3, 3), 16),
        ((3, 3), 32),
        ((3, 3), 32),
    )
    expected_kernels = (
        (1, 1, 26, 128),
        (1, 1, 128, 128),
        (1, 1, 128, 8),
        (3, 3, 8, 16),
        (3, 3, 16, 32),
        (3, 3, 32, 32),
    )
    for source, kernel_shape in zip(params["task_conv"], expected_kernels):
        assert source["kernel"].shape == kernel_shape
        assert source["bias"].shape == (kernel_shape[-1],)

    assert params["task_dense"]["kernel"].shape == (5 * 5 * 32, 128)
    assert params["task_dense"]["bias"].shape == (128,)
    assert params["task_norm"]["scale"].shape == (128,)
    assert params["task_norm"]["bias"].shape == (128,)
    for name in (
        "input_reset",
        "input_update",
        "input_candidate",
        "hidden_reset",
        "hidden_update",
        "hidden_candidate",
    ):
        assert params["task_gru"][name]["kernel"].shape == (128, 128)
        assert params["task_gru"][name]["bias"].shape == (128,)
    assert params["actor_trunk"]["kernel"].shape == (128, 128)
    assert params["actor_trunk"]["bias"].shape == (128,)
    assert params["actor"]["kernel"].shape == (128, 6)
    assert params["actor"]["bias"].shape == (6,)
    assert params["value_trunk"]["kernel"].shape == (128, 128)
    assert params["value_trunk"]["bias"].shape == (128,)
    assert params["value"]["kernel"].shape == (128, 1)
    assert params["value"]["bias"].shape == (1,)


def test_initialization_is_deterministic():
    import jax

    model = _model()
    first = model.init_parameters(jax.random.PRNGKey(1))
    second = model.init_parameters(jax.random.PRNGKey(1))
    for left, right in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_step_shapes_and_episode_reset():
    import jax
    import jax.numpy as jnp

    model = _model()
    params = model.init_parameters(jax.random.PRNGKey(2))
    observations = jax.random.normal(jax.random.PRNGKey(3), (3, 5, 5, 26))
    nonzero = jnp.ones((3, 128), dtype=jnp.float32)
    starts = jnp.ones((3,), dtype=jnp.bool_)
    next_carry, logits, values = model.step(
        params, nonzero, observations, starts
    )
    zero_carry, zero_logits, zero_values = model.step(
        params, jnp.zeros_like(nonzero), observations, starts
    )
    assert next_carry.shape == (3, 128)
    assert logits.shape == (3, 6)
    assert values.shape == (3,)
    np.testing.assert_allclose(np.asarray(next_carry), np.asarray(zero_carry))
    np.testing.assert_allclose(np.asarray(logits), np.asarray(zero_logits))
    np.testing.assert_allclose(np.asarray(values), np.asarray(zero_values))


def test_sequence_matches_stepwise_evaluation():
    import jax
    import jax.numpy as jnp

    model = _model()
    params = model.init_parameters(jax.random.PRNGKey(4))
    time_count, batch_size = 5, 3
    observations = jax.random.normal(
        jax.random.PRNGKey(5), (time_count, batch_size, 5, 5, 26)
    )
    starts = jnp.asarray(
        [
            [True, True, True],
            [False, False, False],
            [False, True, False],
            [False, False, False],
            [True, False, True],
        ]
    )
    initial = jax.random.normal(jax.random.PRNGKey(6), (batch_size, 128))
    sequence_logits, sequence_values = model.sequence(
        params, initial, observations, starts
    )

    carry = initial
    step_logits = []
    step_values = []
    for index in range(time_count):
        carry, logits, values = model.step(
            params, carry, observations[index], starts[index]
        )
        step_logits.append(logits)
        step_values.append(values)
    step_logits = jnp.stack(step_logits)
    step_values = jnp.stack(step_values)

    assert sequence_logits.shape == (time_count, batch_size, 6)
    assert sequence_values.shape == (time_count, batch_size)
    np.testing.assert_allclose(
        np.asarray(sequence_logits), np.asarray(step_logits), rtol=1.0e-5, atol=1.0e-5
    )
    np.testing.assert_allclose(
        np.asarray(sequence_values), np.asarray(step_values), rtol=1.0e-5, atol=1.0e-5
    )


def test_official_transplant_and_shape_check():
    import jax
    import jax.numpy as jnp

    from src.cetr_zsc.official_init import transplant_official_params

    model = _model()
    params = model.init_parameters(jax.random.PRNGKey(7))
    official = _official_tree(params)
    moved = transplant_official_params(params, official)

    for index in range(6):
        for field in ("kernel", "bias"):
            np.testing.assert_array_equal(
                np.asarray(moved["task_conv"][index][field]),
                np.asarray(official["CNN_0"][f"Conv_{index}"][field]),
            )
    for field in ("kernel", "bias"):
        np.testing.assert_array_equal(
            np.asarray(moved["task_dense"][field]),
            np.asarray(official["CNN_0"]["Dense_0"][field]),
        )
    for field in ("scale", "bias"):
        np.testing.assert_array_equal(
            np.asarray(moved["task_norm"][field]),
            np.asarray(official["LayerNorm_0"][field]),
        )
    for source_name, official_name in (
        ("input_reset", "ir"),
        ("input_update", "iz"),
        ("input_candidate", "in"),
        ("hidden_reset", "hr"),
        ("hidden_update", "hz"),
        ("hidden_candidate", "hn"),
    ):
        official_source = official["ScannedRNN_0"]["GRUCell_1"][official_name]
        np.testing.assert_array_equal(
            np.asarray(moved["task_gru"][source_name]["kernel"]),
            np.asarray(official_source["kernel"]),
        )
        if official_name not in {"hr", "hz"}:
            np.testing.assert_array_equal(
                np.asarray(moved["task_gru"][source_name]["bias"]),
                np.asarray(official_source["bias"]),
            )
    np.testing.assert_array_equal(
        np.asarray(moved["task_gru"]["hidden_reset"]["bias"]), 0.0
    )
    np.testing.assert_array_equal(
        np.asarray(moved["task_gru"]["hidden_update"]["bias"]), 0.0
    )
    for source_name, official_name in (("actor_trunk", "Dense_0"), ("actor", "Dense_1")):
        for field in ("kernel", "bias"):
            np.testing.assert_array_equal(
                np.asarray(moved[source_name][field]),
                np.asarray(official[official_name][field]),
            )
    for name in ("value_trunk", "value"):
        for left, right in zip(
            jax.tree_util.tree_leaves(moved[name]),
            jax.tree_util.tree_leaves(params[name]),
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    wrong = dict(official)
    wrong["Dense_1"] = dict(official["Dense_1"])
    wrong["Dense_1"]["kernel"] = jnp.zeros((128, 5), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Dense_1/kernel"):
        transplant_official_params(params, wrong)


def test_initializer_seed_index_reads_unresolved_path_parts():
    from src.cetr_zsc.official_init import initializer_seed_index

    assert initializer_seed_index(".../run-3/ckpt_final") == 3
    assert initializer_seed_index(".../ckpt_final") is None
