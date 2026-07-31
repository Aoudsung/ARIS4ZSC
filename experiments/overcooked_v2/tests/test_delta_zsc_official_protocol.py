from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

from experiments.overcooked_v2.official_adapter import (  # noqa: E402
    _official_symbol,
    _validate_official_baseline_config,
    compose_official_config,
    compose_official_baseline_config,
    compose_ippo_large_config,
    store_official_checkpoint,
    train_upstream,
    validate_official_runtime,
)
from experiments.overcooked_v2.official_baseline_app import _official_command  # noqa: E402
from experiments.overcooked_v2.mechanical_e2e_app import (  # noqa: E402
    mechanical_fixture_key,
)
from src.path_c.experiment import (  # noqa: E402
    OFFICIAL_CORRECT_DELIVERY_REWARD,
    OFFICIAL_OP_TOTAL_TIMESTEPS,
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SP_TOTAL_TIMESTEPS,
    load_config,
    official_training_key,
)
from src.path_c.runner import official_ego_roles  # noqa: E402
from src.path_c.resources import delta_anchor_attempted_steps  # noqa: E402
from src.path_c.training import (  # noqa: E402
    official_learning_rate_schedule,
    official_reward_shaping_factor,
)
from experiments.overcooked_v2.upstream_app import _metric_with_row_axis  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "experiments" / "overcooked_v2" / "configs"


def test_official_runtime_is_exact_fixed_commit() -> None:
    runtime = validate_official_runtime()
    assert runtime["source_commit"] == OFFICIAL_SOURCE_COMMIT
    assert runtime["experiments"]["commit_id"] == OFFICIAL_SOURCE_COMMIT
    assert runtime["jaxmarl"]["commit_id"] == OFFICIAL_SOURCE_COMMIT
    assert runtime["correct_delivery_reward"] == OFFICIAL_CORRECT_DELIVERY_REWARD == 20.0
    assert runtime["overcooked_ai_version"] == "1.1.0"


def test_fixed_official_ppo_entrypoint_imports() -> None:
    assert callable(
        _official_symbol(
            "overcooked_v2_experiments.ppo.main", "single_run_with_viz"
        )
    )


@pytest.mark.parametrize("layout", ("test_time_simple", "test_time_wide"))
def test_official_baseline_recipes_preserve_algorithm_budgets(layout: str) -> None:
    sp = compose_official_baseline_config(layout=layout, method="sp")
    fcp = compose_official_baseline_config(
        layout=layout, method="fcp", fcp_population=Path("/tmp/fcp-population")
    )
    state_augmented = compose_official_baseline_config(
        layout=layout, method="state-augmented"
    )
    op = compose_official_baseline_config(layout=layout, method="op")
    assert sp["model"]["TOTAL_TIMESTEPS"] == OFFICIAL_SP_TOTAL_TIMESTEPS
    assert sp["model"]["NUM_ENVS"] == 256
    assert op["model"]["TOTAL_TIMESTEPS"] == OFFICIAL_OP_TOTAL_TIMESTEPS
    assert op["model"]["NUM_ENVS"] == 64
    assert list(op["env"]["ENV_KWARGS"]["op_ingredient_permutations"]) == [0, 1]
    assert sp["NUM_SEEDS"] == state_augmented["NUM_SEEDS"] == 10
    assert fcp["NUM_SEEDS"] == 1
    assert state_augmented["NUM_ITERATIONS"] == 10
    assert fcp["model"]["LR"] == pytest.approx(0.0007)
    assert fcp["model"]["ENT_COEF"] == pytest.approx(0.04)
    assert fcp["model"]["GAE_LAMBDA"] == pytest.approx(0.9)
    _validate_official_baseline_config(sp, layout=layout, method="sp")
    _validate_official_baseline_config(op, layout=layout, method="op")
    _validate_official_baseline_config(
        state_augmented, layout=layout, method="state-augmented"
    )
    _validate_official_baseline_config(fcp, layout=layout, method="fcp")


def test_formal_budget_is_exact_whole_official_updates() -> None:
    config = load_config(CONFIGS / "delta_zsc_simple_formal.yaml", run_kind="formal")
    steps_per_update = config.environment.num_envs * config.training.rollout_length
    assert steps_per_update == 65_536
    assert config.training.environment_steps // steps_per_update == 457
    assert config.training.environment_steps == 29_949_952
    assert config.evaluation.one_sided_alpha == 0.05
    trigger_count = 457 // config.anchors.interval_updates
    matched_pairs = min(
        config.anchors.states_per_interval,
        config.partner_generator.codes_per_update,
    )
    # Each matched-code pair is two intervened worlds, not one.  The resource
    # ledger records what the implementation actually executes.
    assert delta_anchor_attempted_steps(
        trigger_count=trigger_count,
        ordinary_worlds=config.anchors.states_per_interval,
        matched_worlds=2 * matched_pairs,
        action_count=6,
        fit_replicas=config.anchors.fit_replicas,
        evaluation_replicas=config.anchors.evaluation_replicas,
        continuation_horizon=config.anchors.continuation_horizon,
    ) == 1_680_998_400


@pytest.mark.parametrize(
    ("algorithm", "timesteps", "num_envs"),
    (
        ("rnn-sp", OFFICIAL_SP_TOTAL_TIMESTEPS, 256),
        ("rnn-op", OFFICIAL_OP_TOTAL_TIMESTEPS, 64),
    ),
)
def test_formal_delta_upstream_recipe_remains_unmodified(
    algorithm: str,
    timesteps: int,
    num_envs: int,
    tmp_path: Path,
) -> None:
    config = load_config(
        CONFIGS / "delta_zsc_simple_formal.yaml",
        run_kind="formal",
    )
    observed = compose_official_config(
        config,
        algorithm=algorithm,
        seed_index=0,
        output_directory=tmp_path,
    )
    assert observed["model"]["TOTAL_TIMESTEPS"] == timesteps
    assert observed["model"]["NUM_ENVS"] == num_envs
    assert observed["model"]["NUM_STEPS"] == 256
    assert observed["model"]["NUM_MINIBATCHES"] == 64
    assert observed["model"]["UPDATE_EPOCHS"] == 4


@pytest.mark.parametrize(
    ("algorithm", "entropy"),
    (("rnn-sp", 0.01), ("rnn-op", 0.02)),
)
def test_mechanical_upstream_scales_only_execution_budget(
    algorithm: str,
    entropy: float,
    tmp_path: Path,
) -> None:
    config = load_config(
        CONFIGS / "delta_zsc_simple_mechanical_e2e.yaml",
        run_kind="mechanical",
    )
    observed = compose_official_config(
        config,
        algorithm=algorithm,
        seed_index=0,
        output_directory=tmp_path,
    )
    model = observed["model"]
    assert model["TYPE"] == "RNN"
    assert model["FC_DIM_SIZE"] == 128
    assert model["GRU_HIDDEN_DIM"] == 128
    assert model["TOTAL_TIMESTEPS"] == 1_024
    assert model["REW_SHAPING_HORIZON"] == 512
    assert model["NUM_ENVS"] == 4
    assert model["NUM_STEPS"] == 16
    assert model["NUM_MINIBATCHES"] == 1
    assert model["UPDATE_EPOCHS"] == 4
    assert model["ENT_COEF"] == entropy
    assert observed["RUN_BASE_DIR"] == str(tmp_path.resolve())


def test_mechanical_config_exercises_every_delta_auxiliary_stage() -> None:
    config = load_config(
        CONFIGS / "delta_zsc_simple_mechanical_e2e.yaml",
        run_kind="mechanical",
    )
    updates = (
        config.training.environment_steps
        // config.environment.num_envs
        // config.training.rollout_length
    )
    assert updates == 16
    assert updates // config.anchors.interval_updates == 8
    assert updates // config.partner_generator.update_interval == 8
    assert updates // config.partner_generator.snapshot_interval == 4
    assert config.anchors.fit_replicas > 0
    assert config.anchors.evaluation_replicas > 0
    assert config.calibration.minimum_run_count == 2
    assert config.calibration.enable_hard_gate_at_evaluation
    assert config.evaluation.report_br_prox
    assert config.evaluation.episodes_per_pairing == 2


def test_training_keys_are_exact_split_of_prngkey_42() -> None:
    expected = np.asarray(jax.random.split(jax.random.PRNGKey(42), 10), dtype=np.uint32)
    observed = np.asarray([official_training_key(index) for index in range(10)], dtype=np.uint32)
    np.testing.assert_array_equal(observed, expected)


def test_mechanical_fixture_keys_are_deterministic_fresh_and_disjoint() -> None:
    labels = ("calibration_a", "calibration_b", "confirmatory_a", "confirmatory_b")
    first = [mechanical_fixture_key(label) for label in labels]
    second = [mechanical_fixture_key(label) for label in labels]
    assert first == second
    keys = [key for unused_seed, key in first]
    assert len(set(keys)) == len(keys)
    formal = {tuple(official_training_key(index)) for index in range(10)}
    assert not formal.intersection(keys)


def test_official_scalar_metrics_receive_a_lossless_row_axis() -> None:
    scalar = np.asarray(_metric_with_row_axis(np.asarray(3.5)))
    vector = np.asarray([1.0, 2.0])
    observed_vector = _metric_with_row_axis(vector)
    assert scalar.shape == (1,)
    assert scalar[0] == pytest.approx(3.5)
    assert observed_vector is vector


def test_official_learning_rate_schedule_matches_registered_optax_composition() -> None:
    updates = 457
    minibatches = 64
    epochs = 4
    warmup_updates = int(0.05 * updates)
    steps_per_update = minibatches * epochs
    observed = official_learning_rate_schedule(
        learning_rate=0.00025,
        warmup_fraction=0.05,
        update_count=updates,
        minibatches_per_epoch=minibatches,
        update_epochs=epochs,
    )
    expected = optax.join_schedules(
        schedules=(
            optax.linear_schedule(
                init_value=0.0,
                end_value=0.00025,
                transition_steps=warmup_updates * steps_per_update,
            ),
            optax.cosine_decay_schedule(
                init_value=0.00025,
                decay_steps=(updates - warmup_updates) * steps_per_update,
            ),
        ),
        boundaries=(warmup_updates * steps_per_update,),
    )
    points = np.asarray(
        [
            0,
            warmup_updates * steps_per_update - 1,
            warmup_updates * steps_per_update,
            updates * steps_per_update - 1,
        ],
        dtype=np.int32,
    )
    np.testing.assert_allclose(
        np.asarray([observed(int(point)) for point in points]),
        np.asarray([expected(int(point)) for point in points]),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_official_shaping_schedule_and_role_mask() -> None:
    values = np.asarray(
        official_reward_shaping_factor(
            jnp.asarray([0, 7_500_000, 15_000_000, 20_000_000]),
            horizon=15_000_000,
        )
    )
    np.testing.assert_allclose(values, [1.0, 0.5, 0.0, 0.0])
    roles = np.asarray(official_ego_roles(256))
    np.testing.assert_array_equal(roles[:128], np.zeros(128, dtype=np.int32))
    np.testing.assert_array_equal(roles[128:], np.ones(128, dtype=np.int32))


def test_ippo_large_changes_only_the_registered_capacity_fields() -> None:
    base = compose_official_baseline_config(layout="test_time_simple", method="sp")
    large = compose_ippo_large_config(
        layout="test_time_simple", hidden_dimension=257
    )

    def flatten(value, prefix=()):
        if isinstance(value, dict):
            result = {}
            for name, child in value.items():
                result.update(flatten(child, (*prefix, str(name))))
            return result
        return {prefix: value}

    before = flatten(base)
    after = flatten(large)
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {
        ("model", "FC_DIM_SIZE"),
        ("model", "GRU_HIDDEN_DIM"),
    }
    assert large["model"]["FC_DIM_SIZE"] == 257
    assert large["model"]["GRU_HIDDEN_DIM"] == 257


def test_fixed_official_commands_preserve_fcp_population_indexing() -> None:
    fcp = _official_command(
        method="fcp",
        layout="test_time_simple",
        output=Path("/tmp/formal-fcp"),
        fcp_population=Path("/tmp/fcp-populations"),
    )
    sp = _official_command(
        method="sp",
        layout="test_time_simple",
        output=Path("/tmp/formal-sp"),
        fcp_population=None,
    )
    assert "NUM_SEEDS=10" not in fcp
    assert "+FCP=/tmp/fcp-populations" in fcp
    assert "NUM_SEEDS=10" in sp


def test_direct_upstream_trainer_preserves_official_wandb_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.overcooked_v2.official_adapter as adapter
    import wandb

    active = {"value": False}

    @contextmanager
    def fake_init(**kwargs):
        assert kwargs["mode"] == "disabled"
        active["value"] = True
        try:
            yield object()
        finally:
            active["value"] = False

    def fake_make_train(config):
        del config
        return lambda unused_key: unused_key

    output = {
        "metrics": {
            "env_step": jnp.asarray([[65_536]], dtype=jnp.int32),
            "returned_episode": jnp.asarray([[0.0]], dtype=jnp.float32),
        },
        "runner_state": (
            None,
            {"params": jnp.zeros((1, 3, 1), dtype=jnp.float32)},
        ),
    }

    def fake_pmap(train, batch_size):
        del train
        assert batch_size == 1

        def mapped(keys):
            assert keys.shape == (1, 2)
            assert active["value"]
            return output

        return mapped

    def fake_symbol(module, name):
        if module.endswith(".ippo") and name == "make_train":
            return fake_make_train
        if module.endswith(".utils.utils") and name == "mini_batch_pmap":
            return fake_pmap
        raise AssertionError((module, name))

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(adapter, "_official_symbol", fake_symbol)
    monkeypatch.setattr(
        adapter,
        "store_official_checkpoint",
        lambda **kwargs: Path(f"/tmp/ckpt-{kwargs['update_step']}"),
    )
    result = train_upstream(
        {
            "model": {
                "TYPE": "RNN",
                "NUM_ENVS": 256,
                "NUM_STEPS": 256,
            },
            "env": {
                "ENV_KWARGS": {
                    "layout": "test_time_simple",
                    "agent_view_size": 2,
                }
            },
            "wandb": {
                "ENTITY": "disabled",
                "PROJECT": "disabled",
                "WANDB_MODE": "disabled",
            },
        },
        run_key=(1, 2),
        seed_index=0,
        checkpoint_progress=(0.0, 0.5, 1.0),
    )
    assert result["effective_environment_steps"] == 65_536
    assert len(result["checkpoint_paths"]) == 3


def test_official_checkpoint_store_uses_path_but_serializes_string_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import experiments.overcooked_v2.official_adapter as adapter
    import orbax.checkpoint as ocp
    from flax.training import orbax_utils

    observed = {}

    def fake_get_directory(run_base, run_number, update_step, *, final):
        observed["run_base"] = run_base
        observed["run_number"] = run_number
        observed["update_step"] = update_step
        observed["final"] = final
        return run_base / f"run_{run_number}" / (
            "ckpt_final" if final else f"ckpt_{update_step}"
        )

    class FakeCheckpointer:
        def save(self, path, target, *, save_args):
            observed["path"] = path
            observed["target"] = target
            observed["save_args"] = save_args

    monkeypatch.setattr(
        adapter,
        "_official_symbol",
        lambda module, name: (
            fake_get_directory
            if (
                module == "overcooked_v2_experiments.ppo.utils.store"
                and name == "_get_checkpoint_dir"
            )
            else pytest.fail(f"Unexpected Official symbol: {(module, name)}")
        ),
    )
    monkeypatch.setattr(ocp, "PyTreeCheckpointer", FakeCheckpointer)
    monkeypatch.setattr(
        orbax_utils,
        "save_args_from_target",
        lambda target: ("save-args", target),
    )
    config = {"RUN_BASE_DIR": str(tmp_path / "upstream")}
    params = {"weight": jnp.asarray([1.0])}
    path = store_official_checkpoint(
        config=config,
        params=params,
        run_number=3,
        update_step=17,
        final=True,
    )
    assert isinstance(observed["run_base"], Path)
    assert observed["run_base"] == (tmp_path / "upstream").resolve()
    assert isinstance(config["RUN_BASE_DIR"], str)
    assert isinstance(observed["target"]["config"]["RUN_BASE_DIR"], str)
    assert observed["target"]["params"] is params
    assert observed["run_number"] == 3
    assert observed["update_step"] == 17
    assert observed["final"] is True
    assert observed["save_args"] == ("save-args", observed["target"])
    assert path == (tmp_path / "upstream" / "run_3" / "ckpt_final").resolve()
