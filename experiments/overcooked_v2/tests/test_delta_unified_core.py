from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np


def _small_config(variant: str = "delta_active"):
    from src.delta_zsc.config import load_config

    config = load_config(
        Path("experiments/overcooked_v2/configs/delta_unified_simple_mechanical.yaml"),
        run_kind="mechanical",
    )
    return replace(
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
        ),
    )


def test_all_registered_configs_load_and_method_has_three_fields() -> None:
    from src.delta_zsc.config import load_config

    files = sorted(
        Path("experiments/overcooked_v2/configs").glob("delta_unified_*.yaml")
    )
    assert len(files) == 7
    for path in files:
        kind = path.stem.rsplit("_", 1)[-1]
        if kind == "collector":
            kind = "development"
        config = load_config(path, run_kind=kind)
        assert set(config.method.__dataclass_fields__) == {
            "latent_components",
            "continuation_horizon",
            "adaptation_kl_budget",
        }


def test_pairwise_contrasts_are_offset_invariant() -> None:
    """What replaced the orthonormal contrast basis.

    The basis existed to give the five-dimensional Gaussian decision likelihood
    an offset-invariant coordinate system.  The target is now the action
    differences themselves, which carry that invariance directly.
    """

    import jax.numpy as jnp

    from src.delta_zsc.contrast import pairwise_contrasts_from_replicas

    replicas = jnp.asarray([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    mask = jnp.ones((1, 3), dtype=bool)
    base = pairwise_contrasts_from_replicas(replicas, mask)
    shifted = pairwise_contrasts_from_replicas(replicas + 100.0, mask)
    np.testing.assert_allclose(
        np.asarray(base.mean), np.asarray(shifted.mean), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(base.mean)[0, 1, 0], 1.0, atol=1e-6
    )


def test_episode_static_filter_persists_and_resets_only_at_episode_start() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.belief_filter import episode_static_prior, filter_update

    belief = jnp.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=jnp.float32)
    prior = episode_static_prior(
        belief, jnp.asarray([False, True]), component_count=2
    )
    np.testing.assert_allclose(np.asarray(prior[0]), np.asarray(belief[0]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(prior[1]), [0.5, 0.5], atol=1e-6)
    neutral = jnp.zeros_like(prior)
    np.testing.assert_allclose(
        np.asarray(filter_update(prior, neutral)), np.asarray(prior), atol=1e-6
    )
    evidence = jnp.asarray([[0.0, -10.0], [-10.0, 0.0]], dtype=jnp.float32)
    corrected = filter_update(prior, evidence)
    assert float(corrected[0, 0]) > 0.99
    assert float(corrected[1, 1]) > 0.99


def test_interface_alignment_and_interact_exclusion_contract() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.observation import extract_interface_target

    previous = jnp.zeros((5, 5, 39), dtype=jnp.float32)
    # Asymmetric static counter pattern uniquely selects successful right move.
    previous = previous.at[0, 1, 20].set(1.0)
    previous = previous.at[3, 3, 20].set(1.0)
    current = jnp.zeros_like(previous)
    current = current.at[:, :4, 20:29].set(previous[:, 1:, 20:29])
    current = current.at[0, 0, 29].set(1.0)
    available, changed, event, _, _ = extract_interface_target(
        previous, current, jnp.asarray(0), jnp.asarray(False)
    )
    assert bool(available)
    assert bool(changed)
    assert int(event) == 0  # counter, appeared, plate

    stationary = jnp.zeros((5, 5, 39), dtype=jnp.float32)
    stationary = stationary.at[1, 2, 20].set(1.0)  # ego-facing counter
    stationary = stationary.at[4, 4, 20].set(1.0)  # unaffected target facility
    stationary = stationary.at[2, 2, 1].set(1.0)   # ego faces up
    after_interact = stationary.at[1, 2, 29].set(1.0)
    available, changed, _, _, _ = extract_interface_target(
        stationary, after_interact, jnp.asarray(5), jnp.asarray(False)
    )
    assert bool(available)
    assert not bool(changed)


def test_mirror_policy_satisfies_kl_constraint() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.mirror_policy import mirror_policy_logits

    base = jnp.zeros((4, 6), dtype=jnp.float32)
    values = jnp.eye(6, dtype=jnp.float32)[:4] * 10.0
    _, kl, temperature = mirror_policy_logits(base, values, kl_budget=0.04)
    assert bool(jnp.all(kl <= 0.04001))
    assert bool(jnp.all(jnp.isfinite(temperature)))


def test_model_executes_all_variants_with_one_shared_interface() -> None:
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.model import DeltaModel

    for variant in (
        "history_rnn",
        "base",
        "response_only",
        "delta_passive",
        "delta_active",
    ):
        config = _small_config(variant)
        model = DeltaModel(config, (5, 5, 39), 6)
        base, latent = model.init_parameters(jax.random.PRNGKey(0))
        state = model.initial_state(3)
        observation = jnp.zeros((3, 5, 5, 39), dtype=jnp.float32)
        next_state, output = jax.jit(model.step)(base, latent, state, observation)
        assert next_state.belief.shape == (3, 4)
        assert output.policy_logits.shape == (3, 6)
        assert output.component_decision_means.shape == (3, 4, 6)
        assert output.component_successor_decision_means.shape == (3, 6, 4, 6)
        assert output.probe_response_prediction.interface_event_logits.shape == (
            3, 6, 4, 31
        )
        assert output.active_voi.shape == (3, 6)
        assert bool(jnp.all(jnp.isfinite(output.policy_logits)))
        if variant in {"delta_passive", "delta_active"}:
            assert float(jnp.max(output.adaptation_kl)) <= 0.04001
        else:
            np.testing.assert_allclose(
                np.asarray(output.policy_logits),
                np.asarray(output.base_policy_logits),
                atol=1e-6,
            )


def test_decision_parameters_cannot_change_online_belief() -> None:
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.model import DeltaModel, observe_after_transition

    config = _small_config("delta_passive")
    model = DeltaModel(config, (5, 5, 39), 6)
    base, latent = model.init_parameters(jax.random.PRNGKey(1))
    state = model.initial_state(2)
    observation = jnp.zeros((2, 5, 5, 39), dtype=jnp.float32)
    state, _ = model.step(base, latent, state, observation)
    state = observe_after_transition(
        state, action=jnp.asarray([0, 1]), done=jnp.asarray([False, False])
    )
    # Neither decision-side head may touch the posterior: the belief is formed
    # from the response likelihood alone, and a critic or successor model that
    # could move it would be reading its own training signal back into the
    # filter.
    changed = dict(latent)
    for name in ("belief_value", "successor_feature"):
        changed[name] = jax.tree_util.tree_map(
            lambda value: value + 10.0, latent[name]
        )
    _, original = model.step(base, latent, state, observation)
    _, modified = model.step(base, changed, state, observation)
    np.testing.assert_allclose(
        np.asarray(original.belief), np.asarray(modified.belief), atol=1e-6
    )


def test_response_and_component_value_residuals_are_centered() -> None:
    """Component axes carry contrast only; the shared level lives elsewhere."""

    import jax
    import jax.numpy as jnp

    from src.delta_zsc.model import DeltaModel, component_action_values
    from src.delta_zsc.response_model import response_predict

    config = _small_config("delta_active")
    model = DeltaModel(config, (5, 5, 39), 6)
    _, latent = model.init_parameters(jax.random.PRNGKey(81))
    frame = jnp.zeros((2, 5, 5, 39), dtype=jnp.float32)
    behavior = jnp.zeros((2, 12), dtype=jnp.float32)
    actions = jnp.asarray([0, 1], dtype=jnp.int32)
    response = response_predict(
        latent["response"], latent["component_embeddings"], frame, behavior, actions
    )
    event_residual = response.interface_event_logits - jnp.mean(
        response.interface_event_logits, axis=-2, keepdims=True
    )
    np.testing.assert_allclose(
        np.asarray(jnp.sum(event_residual, axis=-2)), 0.0, atol=2.0e-6
    )

    policy = jnp.full((2, 6), 1.0 / 6.0)
    means, spread = component_action_values(
        latent,
        jnp.zeros((2, 16), dtype=jnp.float32),
        jnp.zeros((2, 8), dtype=jnp.float32),
        behavior,
        policy,
        4,
    )
    assert means.shape == (2, 4, 6)
    assert spread.shape == (2, 4, 6)
    # Each component's action values are centred under the acting policy, so
    # the component axis cannot smuggle in a state-level offset.
    np.testing.assert_allclose(
        np.asarray(jnp.sum(means * policy[:, None, :], axis=-1)), 0.0, atol=1.0e-5
    )


def test_shared_occurrence_heads_cannot_change_online_belief() -> None:
    import copy
    import jax
    import jax.numpy as jnp

    from src.delta_zsc.model import DeltaModel, observe_after_transition

    config = _small_config("delta_passive")
    model = DeltaModel(config, (5, 5, 39), 6)
    base, latent = model.init_parameters(jax.random.PRNGKey(82))
    state = model.initial_state(1)
    first = jnp.zeros((1, 5, 5, 39), dtype=jnp.float32)
    state, _ = model.step(base, latent, state, first)
    state = observe_after_transition(
        state, action=jnp.asarray([4]), done=jnp.asarray([False])
    )
    second = first.at[0, 2, 2, 13].set(1.0)
    changed = copy.deepcopy(latent)
    for name in (
        "visibility",
        "inventory_change",
        "availability",
        "interface_change",
        "recipe_change",
    ):
        changed["response"][name]["bias"] = (
            changed["response"][name]["bias"] + 50.0
        )
    _, original = model.step(base, latent, state, second)
    _, modified = model.step(base, changed, state, second)
    np.testing.assert_allclose(
        np.asarray(original.belief), np.asarray(modified.belief), atol=1.0e-6
    )


def test_spectral_simplex_initializer_is_centered_and_round_trips(tmp_path: Path) -> None:
    import numpy as np

    from src.delta_zsc.semantic_initializer import (
        fit_spectral_simplex_initializer,
        load_semantic_initializer,
        save_semantic_initializer,
    )

    residuals = np.zeros((8, 31), dtype=np.float32)
    residuals[:4, 12] = 1.0
    residuals[:4, 13] = -1.0
    residuals[4:, 12] = -1.0
    residuals[4:, 13] = 1.0
    initializer = fit_spectral_simplex_initializer(
        residuals,
        component_count=4,
        source={"fixture": True, "uses_partner_labels": False},
    )
    assert initializer.event_component_bias.shape == (4, 31)
    np.testing.assert_allclose(
        np.mean(initializer.event_component_bias, axis=0), 0.0, atol=1.0e-6
    )
    npz, metadata = save_semantic_initializer(tmp_path, initializer)
    assert npz.is_file() and metadata.is_file()
    restored = load_semantic_initializer(
        tmp_path, component_count=4, event_count=31
    )
    np.testing.assert_allclose(
        restored.event_component_bias, initializer.event_component_bias, atol=0.0
    )
    assert restored.source["uses_partner_labels"] is False


def test_delayed_probe_target_uses_second_action_and_masks_terminal_windows() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.observation import extract_probe_response_target

    intermediate = jnp.zeros((5, 5, 39), dtype=jnp.float32)
    intermediate = intermediate.at[1, 2, 20].set(1.0)  # front counter
    intermediate = intermediate.at[4, 4, 20].set(1.0)  # retained facility
    intermediate = intermediate.at[2, 2, 1].set(1.0)  # ego faces up
    delayed = intermediate.at[1, 2, 29].set(1.0)
    interact = extract_probe_response_target(
        intermediate,
        delayed,
        jnp.asarray(5),
        jnp.asarray(False),
    )
    stay = extract_probe_response_target(
        intermediate,
        delayed,
        jnp.asarray(4),
        jnp.asarray(False),
    )
    assert float(interact.valid_mask) == 1.0
    assert not bool(interact.interface_changed)
    assert bool(stay.interface_changed)
    invalid = extract_probe_response_target(
        intermediate,
        delayed,
        jnp.asarray(4),
        jnp.asarray(True),
    )
    assert float(invalid.valid_mask) == 0.0
    assert float(invalid.interface_available) == 0.0
    assert float(invalid.interface_changed) == 0.0


def test_manifest_mechanisms_normalise_to_the_registered_names() -> None:
    """The names the manifest carries must reach the names comparisons use.

    Posterior diagnostics keyed its per-mechanism distributions on the raw
    manifest string ("rnn-sp") and then looked up the registered name ("sp").
    The lookup never matched, so both SP/OP total variations were ``None`` in
    every artifact ever produced -- silently, because ``None`` is a legitimate
    value when a panel holds a single mechanism.
    """

    from src.delta_zsc.manifest import normalized_mechanism

    assert normalized_mechanism("rnn-sp") == "sp"
    assert normalized_mechanism("rnn-op") == "op"
    assert normalized_mechanism("rnn-sa") == "sa"
    assert normalized_mechanism("rnn-fcp") == "fcp"


def test_posterior_diagnostics_keys_distributions_by_canonical_name() -> None:
    """Guard the exact line that made the total variations unreachable."""

    from pathlib import Path

    source = Path("experiments/overcooked_v2/calibration_app.py").read_text()
    head, separator, tail = source.partition("sp_op_tv = None")
    assert separator, "the SP/OP total variation block moved"
    assert 'normalized_mechanism(row["partner_mechanism"])' in head
    assert '"sp" in by_mechanism and "op" in by_mechanism' in tail


def test_task_channel_blocks_derive_from_the_ingredient_count() -> None:
    """Block offsets must follow the layout, not a three-ingredient constant.

    These were absolute constants (20/29/34/39), correct only for a
    three-ingredient layout.  Two of the six registered layouts carry four
    ingredients and 43 channels, and one of them is a ``wide`` layout the
    formal protocol requires -- so the registered wide experiment could not
    have run.
    """

    from src.delta_zsc.observation import task_channel_blocks

    three = task_channel_blocks(39)
    assert three == {
        "static": (20, 29),
        "dynamic": (29, 34),
        "recipe": (34, 39),
    }, "three-ingredient offsets must reproduce the historical constants exactly"

    four = task_channel_blocks(43)
    assert four["static"] == (22, 32)
    assert four["dynamic"] == (32, 38)
    # The environment's own _get_obs_shape undercounts, so the local frame stops
    # one channel into the recipe block.  The bound is clipped to what exists
    # rather than returning an index that would silently read past the end.
    assert four["recipe"] == (38, 43)

    for channels in (39, 43):
        blocks = task_channel_blocks(channels)
        assert blocks["static"][1] == blocks["dynamic"][0]
        assert blocks["dynamic"][1] == blocks["recipe"][0]
        assert blocks["recipe"][1] <= channels


def test_response_extraction_accepts_a_four_ingredient_frame() -> None:
    """The extractor pinned itself to 5x5x39 and rejected wide layouts."""

    import jax.numpy as jnp

    from src.delta_zsc.observation import decode_local_task_state

    for channels, dynamic_width in ((39, 5), (43, 6)):
        frame = jnp.zeros((2, 5, 5, channels), dtype=jnp.float32)
        state = decode_local_task_state(frame)
        assert state.dynamic.shape[-1] == dynamic_width
        assert state.static.shape[-1] > 0 and state.recipe.shape[-1] > 0
