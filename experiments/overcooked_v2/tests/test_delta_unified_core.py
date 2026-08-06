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
    assert len(files) == 6
    for path in files:
        kind = path.stem.rsplit("_", 1)[-1]
        config = load_config(path, run_kind=kind)
        assert set(config.method.__dataclass_fields__) == {
            "latent_components",
            "continuation_horizon",
            "adaptation_kl_budget",
        }


def test_action_contrast_basis_is_orthonormal_and_offset_invariant() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.decision_model import action_contrast_matrix

    basis = np.asarray(action_contrast_matrix(6))
    np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-6)
    np.testing.assert_allclose(np.ones(6) @ basis, 0.0, atol=1e-6)
    values = jnp.asarray([[1, 2, 3, 4, 5, 6]], dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(values @ basis), np.asarray((values + 100.0) @ basis), atol=1e-5
    )


def test_filter_uses_physical_transition_and_exact_response_correction() -> None:
    import jax.numpy as jnp

    from src.delta_zsc.belief_filter import filter_update

    belief = jnp.asarray([[0.9, 0.1]], dtype=jnp.float32)
    transition_logits = jnp.log(
        jnp.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=jnp.float32)
    )
    neutral = jnp.zeros_like(belief)
    predicted = belief @ jnp.exp(transition_logits)
    np.testing.assert_allclose(
        np.asarray(filter_update(belief, transition_logits, neutral)),
        np.asarray(predicted),
        atol=1e-6,
    )
    evidence = jnp.asarray([[0.0, -10.0]], dtype=jnp.float32)
    assert float(filter_update(belief, transition_logits, evidence)[0, 0]) > 0.99


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
    changed = dict(latent)
    changed["decision"] = jax.tree_util.tree_map(
        lambda value: value + 10.0, latent["decision"]
    )
    _, original = model.step(base, latent, state, observation)
    _, modified = model.step(base, changed, state, observation)
    np.testing.assert_allclose(
        np.asarray(original.belief), np.asarray(modified.belief), atol=1e-6
    )
