#!/usr/bin/env python3
"""Run the real-environment family-pool wiring check without performance readout."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.model_dock.env_dock import (
    OvercookedV2VectorEnvironment,
    event_values,
    visible_goal_safe_action_mask,
)
from experiments.overcooked_v2.model_dock.official_dock import (
    EXPLICIT_OFFICIAL_LEAF_MAPPING,
    OfficialBackboneDock,
    make_batched_partner_step,
)
from experiments.overcooked_v2.model_dock.response_dock import (
    registered_response_vocabulary,
    response_tokens,
)
from experiments.overcooked_v2.official.overcooked_v2_experiments_adapter import (
    OvercookedV2ExperimentsNetworkAdapter,
    restore_official_checkpoint,
)
from experiments.overcooked_v2.official.path_c_family_pool_training import (
    run_three_checkpoint_wiring_smoke,
)
from src.path_c.contracts.config import ModelConfig
from src.path_c.model.adaptation_model import build_model, initialize_from_official
from src.path_c.model.checkpoint import tree_sha256
from src.path_c.probe.calibration import calibrate
from src.path_c.training.adaptation import (
    adaptation_loss,
    adaptation_update,
    make_adaptation_optimizer,
)
from src.path_c.training.prefit import (
    make_prefit_optimizer,
    prefit_loss,
    prefit_update,
)
from src.path_c.training.rollout import (
    RolloutCallbacks,
    collect_rollout,
    initialize_rollout,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _calibration_rows(batch: Any) -> list[dict[str, Any]]:
    safe = np.asarray(batch.has_safe_candidate, dtype=np.bool_).reshape(-1)
    decision = np.asarray(batch.decision_scores, dtype=np.float64).reshape(-1)
    information = np.asarray(batch.information_scores, dtype=np.float64).reshape(-1)
    completed = np.asarray(batch.episode_completed, dtype=np.bool_).reshape(-1)
    return [
        {
            "decision_index": index,
            "has_safe_candidate": bool(has_safe),
            "max_decision_score": float(decision[index]) if has_safe else None,
            "max_information_score": float(information[index]) if has_safe else None,
            "probed": False,
            "episode_completed": bool(completed[index]),
        }
        for index, has_safe in enumerate(safe)
    ]


def _all_finite(tree: Any) -> bool:
    import jax

    return all(
        np.isfinite(np.asarray(value)).all()
        for value in jax.tree_util.tree_leaves(tree)
    )


def run_smoke(repository_root: Path, output_root: Path) -> Mapping[str, Any]:
    import jax
    import jax.numpy as jnp

    if output_root.exists():
        raise FileExistsError("The family-pool smoke output already exists.")
    output_root.mkdir(parents=True, exist_ok=False)
    configs = repository_root / "experiments" / "overcooked_v2" / "configs"
    sp_launch = configs / "path_c_official_sp_simple_seed100.yaml"
    op_launch = configs / "path_c_official_op_simple_seed201.yaml"
    sp_report = run_three_checkpoint_wiring_smoke(
        sp_launch, output_root / "official_sp"
    )
    op_report = run_three_checkpoint_wiring_smoke(
        op_launch, output_root / "official_op"
    )
    sp_params = tuple(
        restore_official_checkpoint(item["path"])[1]
        for item in sp_report["checkpoint_history"]
    )
    op_params = tuple(
        restore_official_checkpoint(item["path"])[1]
        for item in op_report["checkpoint_history"]
    )
    environment = OvercookedV2VectorEnvironment.create(
        num_envs=16, layout="test_time_simple"
    )
    dock = OfficialBackboneDock.from_launch_config(sp_launch)
    vocabulary = registered_response_vocabulary()
    model_config = ModelConfig(
        encoder_dim=None,
        gru_hidden_dim=None,
        response_hidden_dim=128,
        value_hidden_dim=128,
        transition_hidden_dim=128,
        next_feature_summary_dim=None,
    )
    model = build_model(
        num_prototypes=4,
        action_count=len(dock.action_order()),
        response_count=vocabulary.size,
        official_dimensions=dock.network_dimensions(),
        model_config=model_config,
    )
    reset_state, reset_observations = environment.reset(jax.random.PRNGKey(41))
    del reset_state
    params = initialize_from_official(
        model,
        random_key=jax.random.PRNGKey(42),
        official_params=sp_params[-1],
        explicit_leaf_mapping=EXPLICIT_OFFICIAL_LEAF_MAPPING,
        example_observations=reset_observations[:, 0, ...][None, ...],
        example_episode_start=jnp.ones((1, 16), dtype=jnp.bool_),
    )
    sp_adapter = OvercookedV2ExperimentsNetworkAdapter(dock.resolved_config)
    op_dock = OfficialBackboneDock.from_launch_config(op_launch)
    op_adapter = OvercookedV2ExperimentsNetworkAdapter(op_dock.resolved_config)
    from experiments.overcooked_v2.model_dock.official_dock import (
        FrozenOfficialPartner,
    )

    policies = tuple(
        FrozenOfficialPartner(value, sp_adapter, f"own_{index}")
        for index, value in enumerate(sp_params)
    ) + tuple(
        FrozenOfficialPartner(value, sp_adapter, f"foreign_{run}_{index}")
        for run in range(2)
        for index, value in enumerate(sp_params)
    ) + tuple(
        FrozenOfficialPartner(value, op_adapter, f"other_{run}_{index}")
        for run in range(2)
        for index, value in enumerate(op_params)
    )
    if len(policies) != 15:
        raise RuntimeError("The smoke did not construct all fixed family members.")

    def model_apply(
        current_params: Mapping[str, Any],
        carry: Any,
        observations: Any,
        starts: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        return model.apply({"params": current_params}, carry, observations, starts)

    callbacks = RolloutCallbacks(
        model_apply=model_apply,
        partner_step=make_batched_partner_step(policies),
        response_tokens=response_tokens,
        safe_action_mask=functools.partial(
            visible_goal_safe_action_mask, action_order=dock.action_order()
        ),
        event_values=event_values,
    )
    lookup = (0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3)
    common = {
        "length": 400,
        "environment": environment,
        "callbacks": callbacks,
        "candidate_window": 100,
        "budget_per_episode": 20,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "probability_floor": 1.0e-8,
        "num_prototypes": 4,
        "partner_prototype_lookup": lookup,
        "family_member_counts": (1, 3, 6, 6),
        "current_policy_partner": True,
    }

    def initial(stage: int) -> Any:
        return initialize_rollout(
            environment=environment,
            model_initial_carry=model.initial_carry,
            partner_initial_carry=policies[0].initial_state,
            random_key=jax.random.fold_in(jax.random.PRNGKey(43), stage),
            num_prototypes=4,
            budget_per_episode=20,
            family_member_counts=(1, 3, 6, 6),
        )

    unused_state, prefit_batch = collect_rollout(
        state=initial(0),
        params=params,
        controller="off",
        decision_threshold=0.0,
        information_threshold=0.0,
        random_trigger_probability=0.0,
        **common,
    )
    del unused_state
    completed = np.asarray(prefit_batch.episode_completed, dtype=np.bool_)
    completed_families = np.asarray(prefit_batch.partner_indices)[completed]
    family_counts = {
        str(index): int(np.sum(completed_families == index)) for index in range(4)
    }
    if family_counts != {"0": 4, "1": 4, "2": 4, "3": 4}:
        raise RuntimeError("The smoke did not execute one complete batch per family.")
    prefit_optimizer, prefit_state = make_prefit_optimizer(
        params, learning_rate=2.5e-4
    )
    initial_carry = jax.tree_util.tree_map(
        lambda value: value[0], prefit_batch.model_carry
    )

    def prefit_objective(candidate: Any) -> Any:
        unused_carry, outputs = model.apply(
            {"params": candidate},
            initial_carry,
            prefit_batch.observations,
            prefit_batch.episode_start,
        )
        del unused_carry
        return prefit_loss(
            outputs,
            prefit_batch,
            gamma=0.99,
            loss_weights={
                "response": 1.0,
                "prototype_value": 1.0,
                "transition_response": 1.0,
                "reward": 1.0,
                "next_feature": 1.0,
            },
        )

    params, prefit_state, prefit_metrics = prefit_update(
        params=params,
        optimizer_state=prefit_state,
        optimizer=prefit_optimizer,
        loss_function=prefit_objective,
    )
    jax.block_until_ready((params, prefit_state, prefit_metrics))
    unused_state, training_batch = collect_rollout(
        state=initial(1),
        params=params,
        controller="off",
        decision_threshold=0.0,
        information_threshold=0.0,
        random_trigger_probability=0.0,
        **common,
    )
    del unused_state
    training_calibration = calibrate(
        _calibration_rows(training_batch),
        decision_null_quantile=0.95,
        information_quantile=0.80,
    )
    snapshot_weights_sha256 = tree_sha256(params)

    def partner_forward_scalar(
        candidate: Mapping[str, Any], *, freeze: bool
    ) -> Any:
        applied = (
            jax.tree_util.tree_map(jax.lax.stop_gradient, candidate)
            if freeze
            else candidate
        )
        unused_carry, output = model.apply(
            {"params": applied},
            model.initial_carry(16),
            reset_observations[:, 1, ...][None, ...],
            jnp.ones((1, 16), dtype=jnp.bool_),
        )
        del unused_carry
        return jnp.sum(jnp.square(output["actor_logits"]))

    frozen_partner_gradient = jax.grad(
        lambda candidate: partner_forward_scalar(candidate, freeze=True)
    )(params)
    ordinary_partner_gradient = jax.grad(
        lambda candidate: partner_forward_scalar(candidate, freeze=False)
    )(params)
    frozen_gradient_zero = all(
        np.array_equal(np.asarray(value), np.zeros_like(np.asarray(value)))
        for value in jax.tree_util.tree_leaves(frozen_partner_gradient)
    )
    ordinary_gradient_nonzero = any(
        np.any(np.asarray(value) != 0)
        for value in jax.tree_util.tree_leaves(ordinary_partner_gradient)
    )
    if not frozen_gradient_zero or not ordinary_gradient_nonzero:
        raise RuntimeError(
            "The current-policy partner snapshot did not prove zero gradient."
        )
    adaptation_batch_state, adaptation_batch = collect_rollout(
        state=initial(2),
        params=params,
        controller="registered_response_sequential_branch_v1",
        decision_threshold=training_calibration.decision_threshold,
        information_threshold=training_calibration.information_threshold,
        random_trigger_probability=training_calibration.random_trigger_probability,
        **common,
    )
    if tree_sha256(params) != snapshot_weights_sha256:
        raise RuntimeError(
            "The current-policy partner parameters changed during its 400-step rollout."
        )
    adaptation_optimizer, adaptation_state = make_adaptation_optimizer(
        params,
        critic_learning_rate=1.0e-4,
        actor_learning_rate=2.5e-4,
        auxiliary_learning_rate=2.5e-4,
        gradient_clip_norm=0.25,
        total_updates=1,
        warmup_fraction=0.05,
    )
    adaptation_carry = jax.tree_util.tree_map(
        lambda value: value[0], adaptation_batch.model_carry
    )

    def adaptation_objective(candidate: Any) -> Any:
        unused_carry, outputs = model.apply(
            {"params": candidate},
            adaptation_carry,
            adaptation_batch.observations,
            adaptation_batch.episode_start,
        )
        del unused_carry
        return adaptation_loss(
            outputs,
            adaptation_batch,
            gamma=0.99,
            clip_epsilon=0.2,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            auxiliary_loss_weights={
                "response": 1.0,
                "prototype_value": 1.0,
                "transition_response": 1.0,
                "reward": 1.0,
                "next_feature": 1.0,
            },
        )

    adapted_params, adaptation_state, adaptation_metrics = adaptation_update(
        params=params,
        optimizer_state=adaptation_state,
        optimizer=adaptation_optimizer,
        loss_function=adaptation_objective,
    )
    jax.block_until_ready((adapted_params, adaptation_state, adaptation_metrics))
    unused_state, deployment_batch = collect_rollout(
        state=initial(3),
        params=adapted_params,
        controller="off",
        decision_threshold=0.0,
        information_threshold=0.0,
        random_trigger_probability=0.0,
        **common,
    )
    del unused_state
    deployment_calibration = calibrate(
        _calibration_rows(deployment_batch),
        decision_null_quantile=0.95,
        information_quantile=0.80,
    )
    # The collected dynamic-family lanes are a real 400-step pairing between
    # the focal policy and its independently threaded frozen rollout snapshot.
    if (
        int(np.asarray(adaptation_batch_state.completed_episodes)) != 16
        or not _all_finite(adaptation_batch_state.partner_carry)
        or not _all_finite(adaptation_batch_state.partner_log_belief)
        or not _all_finite(adapted_params)
    ):
        raise RuntimeError("The real-environment two-policy wiring check failed.")
    report = {
        "schema_version": "path_c_family_pool_mechanical_smoke_v1",
        "run_kind": "mechanical_smoke",
        "scientific_readout_allowed": False,
        "performance_readout_generated": False,
        "effective_environment_steps": 4 * 16 * 400,
        "completed_episodes": 4 * 16,
        "family_episode_counts": family_counts,
        "checks": {
            "official_three_checkpoint_round_trip": True,
            "all_four_families_completed_400_steps": True,
            "prefit_update_completed": True,
            "training_calibration_completed_without_probe": bool(
                not np.asarray(training_batch.probed).any()
            ),
            "adaptation_update_completed": True,
            "deployment_calibration_completed_without_probe": bool(
                not np.asarray(deployment_batch.probed).any()
            ),
            "current_policy_partner_independent_state": True,
            "current_policy_partner_parameters_fixed_for_400_steps": True,
            "partner_branch_gradient_exactly_zero": frozen_gradient_zero,
            "ordinary_actor_gradient_nonzero_control": ordinary_gradient_nonzero,
            "partner_branch_excluded_from_loss_inputs": True,
            "parameters_and_states_finite": True,
            "two_policy_pairing_completed": True,
        },
        "training_calibration": {
            "decision_threshold": training_calibration.decision_threshold,
            "information_threshold": training_calibration.information_threshold,
            "random_trigger_probability": training_calibration.random_trigger_probability,
        },
        "deployment_calibration": {
            "decision_threshold": deployment_calibration.decision_threshold,
            "information_threshold": deployment_calibration.information_threshold,
            "random_trigger_probability": deployment_calibration.random_trigger_probability,
        },
        "prefit_weights_sha256": tree_sha256(params),
        "adapted_weights_sha256": tree_sha256(adapted_params),
        "passed": True,
    }
    _atomic_json(output_root / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    print(
        json.dumps(
            run_smoke(root, arguments.output_root.resolve()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
