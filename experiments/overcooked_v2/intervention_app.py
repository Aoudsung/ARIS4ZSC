"""Crossed causal decision-value intervention for DELTA belief use."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from src.delta_zsc.anchors import collect_anchor_batch
from src.delta_zsc.base_policy import base_policy_sequence
from src.delta_zsc.bayes_voi import myopic_value_of_information_details
from src.delta_zsc.belief_value import belief_value_predict
from src.delta_zsc.config import METHOD_VERSION, OFFICIAL_ACTION_COUNT, load_config
from src.delta_zsc.manifest import load_partner_manifest
from src.delta_zsc.mirror_policy import (
    MIRROR_UNCERTAINTY_PENALTY,
    categorical_kl_from_logits,
    mirror_policy_logits,
    project_policy_logits,
    robust_mirror_policy_logits,
)
from src.delta_zsc.model import (
    EXECUTION_MODES,
    successor_action_values,
)
from src.delta_zsc.observation import extract_response_target
from src.delta_zsc.partners import (
    make_fixed_partner_functions,
    make_static_partner_functions,
)
from src.delta_zsc.response_model import probe_response_predict
from src.delta_zsc.resources import ResourceLedger
from src.delta_zsc.runner import collect_rollout, initialize_runner
from src.delta_zsc.storage import ensure_run_identity, read_json, write_json

from .deployment import load_deployment


POSTERIOR_INTERVENTIONS = ("observed", "uniform", "episode_shuffle")
POSTERIOR_INTERVENTION_DEFINITIONS = {
    "observed": "legal response posterior on the frozen trajectory",
    "uniform": "1/K at every frozen trajectory step",
    "episode_shuffle": (
        "posterior time points permuted within each realized episode and lane"
    ),
}
DECISION_CHAIN_AUDIT_VERSION = 1


def _crossed_bootstrap(matrix: np.ndarray, *, replicates: int = 9_999) -> np.ndarray:
    """Resample ego and partner nodes independently."""

    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError(
            "Belief intervention needs at least two ego and two partner runs."
        )
    rng = np.random.default_rng(0)
    draws = np.empty((int(replicates),), dtype=np.float64)
    for index in range(int(replicates)):
        ego = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        partner = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draws[index] = float(np.mean(matrix[np.ix_(ego, partner)]))
    return draws


def run_belief_value_intervention(args: argparse.Namespace) -> None:
    """Estimate correct-belief minus shuffled-belief value in the same world.

    Belief vectors are exchanged across independent ego-history lanes, but the
    task state, current partner, all-action continuation vector, role, and CRN
    keys remain fixed.  Formal inference resamples both ego and partner runs.
    """

    started = time.perf_counter()
    import jax
    import jax.numpy as jnp

    from .official_adapter import FrozenPartnerPool, VectorEnvironment
    from .training_app import _anchor_functions

    config = load_config(args.config, run_kind=args.run_kind)
    deployment_paths = tuple(Path(value).resolve() for value in args.deployment)
    deployments = tuple(load_deployment(path) for path in deployment_paths)
    for deployment in deployments:
        if deployment.config.environment.layout != config.environment.layout:
            raise ValueError("Belief-intervention deployment layout differs.")
        if deployment.config.method_variant not in {"delta_passive", "delta_active"}:
            raise ValueError("Belief intervention requires a DELTA deployment.")
    if config.run_kind == "formal" and len(deployments) < config.evaluation.minimum_ego_runs:
        raise ValueError("Formal H3 requires the registered number of ego runs.")

    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_file_check),
    )
    runs = (
        manifest.by_role("confirmatory")
        if config.run_kind == "formal"
        else (manifest.by_role("development_coverage") or manifest.by_role("confirmatory"))
    )
    if len(runs) < 2:
        raise ValueError("Belief intervention needs at least two held-out partner runs.")

    effects = np.empty((len(deployments), len(runs)), dtype=np.float64)
    node_rows = []
    total_steps = 0
    for ego_index, deployment in enumerate(deployments):
        for partner_index, run in enumerate(runs):
            pool = FrozenPartnerPool.from_checkpoints(
                (run.checkpoint,),
                parent_training_run_ids=(run.parent_training_run_id,),
            )
            partner_functions = make_static_partner_functions(
                pool=pool,
                probabilities=jnp.asarray([1.0]),
                run_ids=jnp.asarray([0]),
            )
            environment = VectorEnvironment.create(config)
            root = jax.random.fold_in(
                jax.random.fold_in(jax.random.PRNGKey(30_000), ego_index),
                partner_index,
            )
            runner = initialize_runner(
                environment=environment,
                model=deployment.model,
                partner_functions=partner_functions,
                random_key=root,
            )
            _, batch, records = collect_rollout(
                state=runner,
                length=config.training.rollout_length,
                environment=environment,
                model=deployment.model,
                base_params=deployment.base_params,
                latent_params=deployment.latent_params,
                partner_functions=partner_functions,
                partner_parameters=None,
                official_shaping_factor=0.0,
                record_anchors=True,
                use_deployment_policy=True,
            )
            functions = _anchor_functions(
                model=deployment.model,
                partner_functions=partner_functions,
                partner_parameters=None,
                environment=environment,
            )
            anchors = collect_anchor_batch(
                key=jax.random.fold_in(root, 31_000),
                records=records,
                functions=functions,
                base_params=deployment.base_params,
                latent_params=deployment.latent_params,
                states_per_trigger=config.anchors.states_per_trigger,
                action_count=OFFICIAL_ACTION_COUNT,
                fit_replicas=config.anchors.fit_replicas,
                evaluation_replicas=config.anchors.evaluation_replicas,
                horizon=config.method.continuation_horizon,
                gamma=config.ppo.gamma,
                collect_successor=False,
            )
            _, output = deployment.model.sequence(
                deployment.base_params,
                deployment.latent_params,
                batch.initial_policy_state,
                batch.observations,
                batch.previous_actions,
                batch.episode_starts,
            )
            time, lane = anchors.time_indexes, anchors.lane_indexes
            belief = output.belief[time, lane]
            if belief.shape[0] < 2:
                raise ValueError("Belief intervention needs at least two anchor histories.")
            shuffled = jnp.roll(belief, shift=1, axis=0)
            means = output.component_decision_means[time, lane]
            correct_q = jnp.sum(belief[..., :, None] * means, axis=-2)
            shuffled_q = jnp.sum(shuffled[..., :, None] * means, axis=-2)
            base_logits = output.base_policy_logits[time, lane]
            correct_logits, _, _ = mirror_policy_logits(
                base_logits,
                correct_q,
                kl_budget=config.method.adaptation_kl_budget,
            )
            shuffled_logits, _, _ = mirror_policy_logits(
                base_logits,
                shuffled_q,
                kl_budget=config.method.adaptation_kl_budget,
            )
            correct_policy = jax.nn.softmax(correct_logits, axis=-1)
            shuffled_policy = jax.nn.softmax(shuffled_logits, axis=-1)
            returns = anchors.evaluation_returns_by_action
            current_effects = np.asarray(
                jnp.sum((correct_policy - shuffled_policy) * returns, axis=-1),
                dtype=np.float64,
            )
            value = float(np.mean(current_effects))
            effects[ego_index, partner_index] = value
            node_rows.append(
                {
                    "ego_run_index": ego_index,
                    "ego_run_id": deployment.ego_run_id,
                    "partner_run_index": partner_index,
                    "partner_run_id": run.run_id,
                    "partner_mechanism": run.generation_mechanism,
                    "anchor_count": int(current_effects.size),
                    "mean_effect": value,
                }
            )
            total_steps += (
                config.environment.num_envs * config.training.rollout_length
                + config.anchors.states_per_trigger
                * 6
                * (config.anchors.fit_replicas + config.anchors.evaluation_replicas)
                * config.method.continuation_horizon
            )

    boot = _crossed_bootstrap(effects)
    output_dir = Path(args.output).resolve()
    deployment_sources = [{"path": str(path)} for path in deployment_paths]
    ensure_run_identity(
        output_dir,
        {
            "stage": "belief-value-intervention",
            "method": METHOD_VERSION,
            "layout": config.environment.layout,
            "deployments": deployment_sources,
            "partner_manifest": {"path": str(manifest_path)},
        },
    )
    elapsed = time.perf_counter() - started
    ledger = ResourceLedger(
        intervention_steps=total_steps,
        measurement_wall_clock_hours=elapsed / 3600.0,
        measurement_gpu_hours=(
            elapsed / 3600.0
            if any(device.platform == "gpu" for device in jax.devices())
            else 0.0
        ),
    )
    write_json(
        output_dir / "belief_value_intervention.json",
        {
            "version": 2,
            "artifact_type": "delta_belief_value_intervention",
            "method": METHOD_VERSION,
            "execution_mode": "active",
            "layout": config.environment.layout,
            "estimand": (
                "same-source-world expected return of correct legal-history "
                "belief policy minus shuffled-belief policy"
            ),
            "estimate": float(np.mean(effects)),
            "interval_95": [float(v) for v in np.quantile(boot, (0.025, 0.975))],
            "one_sided_lcb": float(np.quantile(boot, 0.05)),
            "ego_run_count": int(effects.shape[0]),
            "partner_run_count": int(effects.shape[1]),
            "node_effects": node_rows,
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output_dir / "resource_ledger.json", ledger.to_mapping())


def _categorical_tv(left_logits: Any, right_logits: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    return 0.5 * jnp.sum(
        jnp.abs(jnn.softmax(left_logits, axis=-1) - jnn.softmax(right_logits, axis=-1)),
        axis=-1,
    )


def _active_probe_schedule(episode_starts: Any) -> Any:
    import jax
    import jax.numpy as jnp

    starts = jnp.asarray(episode_starts, dtype=jnp.bool_)

    def one(pending: Any, start: Any) -> tuple[Any, Any]:
        current = jnp.where(start, jnp.zeros_like(pending), pending)
        eligible = ~current
        return eligible, eligible

    _, eligible = jax.lax.scan(
        one, jnp.zeros(starts.shape[1:], dtype=jnp.bool_), starts
    )
    return eligible


def _decision_chain(
    *,
    model: Any,
    base_params: Any,
    latent_params: Any,
    initial_task_carry: Any,
    observations: Any,
    episode_starts: Any,
    behavior: Any,
    belief: Any,
) -> Mapping[str, Any]:
    """Recompute the complete policy chain under one supplied posterior path."""

    import jax
    import jax.nn as jnn
    import jax.numpy as jnp

    posterior = jnp.asarray(belief, dtype=jnp.float32)
    (
        unused_final_carry,
        unused_task_carries,
        task,
        instant,
        reference_logits,
        residual_logits,
        residual_composed_logits,
        unused_value,
    ) = base_policy_sequence(
        base_params,
        initial_task_carry,
        observations,
        episode_starts,
        posterior,
    )
    del unused_final_carry, unused_task_carries, unused_value
    budget = float(model.config.method.adaptation_kl_budget)
    base_logits, base_kl, base_alpha = project_policy_logits(
        reference_logits,
        residual_composed_logits,
        kl_budget=budget,
    )
    critic = belief_value_predict(
        latent_params["belief_value"],
        task,
        instant,
        behavior,
        posterior,
        jnn.softmax(base_logits, axis=-1),
        latent_params["component_embeddings"],
    )
    q_values = critic.advantage_mean
    dispersion = critic.advantage_dispersion()

    lead = tuple(jnp.asarray(observations).shape[:-3])
    action_count = int(model.action_count)
    component_count = int(model.config.method.latent_components)
    probe_actions = jnp.broadcast_to(
        jnp.arange(action_count, dtype=jnp.int32), lead + (action_count,)
    )
    frames = jnp.broadcast_to(
        jnp.asarray(observations, dtype=jnp.float32)[..., None, :, :, :],
        lead + (action_count,) + tuple(model.observation_shape),
    )
    probe_behavior = jnp.broadcast_to(
        jnp.asarray(behavior, dtype=jnp.float32)[..., None, :],
        lead + (action_count, int(behavior.shape[-1])),
    )
    probe_prediction = probe_response_predict(
        latent_params["probe_response"],
        latent_params["component_embeddings"],
        frames,
        probe_behavior,
        probe_actions,
    )
    successor_values = successor_action_values(
        latent_params,
        task,
        instant,
        behavior,
        posterior,
        jnn.softmax(base_logits, axis=-1),
        probe_prediction,
        component_count,
        action_count,
    )
    voi = myopic_value_of_information_details(
        posterior, probe_prediction, successor_values
    )

    passive_candidate, passive_mirror_kl, passive_temperature = (
        robust_mirror_policy_logits(
            base_logits,
            q_values,
            dispersion,
            kl_budget=budget,
            uncertainty_penalty=MIRROR_UNCERTAINTY_PENALTY,
        )
    )
    passive_final, passive_final_kl, passive_alpha = project_policy_logits(
        reference_logits,
        passive_candidate,
        kl_budget=budget,
    )
    active_values = q_values + (float(model.config.ppo.gamma) ** 2) * voi.value
    active_mirror, active_mirror_kl, active_temperature = robust_mirror_policy_logits(
        base_logits,
        active_values,
        dispersion,
        kl_budget=budget,
        uncertainty_penalty=MIRROR_UNCERTAINTY_PENALTY,
    )
    probe_eligible = _active_probe_schedule(episode_starts)
    active_candidate = jnp.where(
        probe_eligible[..., None], active_mirror, base_logits
    )
    active_final, active_final_kl, active_alpha = project_policy_logits(
        reference_logits,
        active_candidate,
        kl_budget=budget,
    )

    candidates = jnp.stack(
        (
            reference_logits,
            residual_composed_logits,
            passive_candidate,
            active_candidate,
        ),
        axis=0,
    )
    finals = jnp.stack(
        (reference_logits, base_logits, passive_final, active_final), axis=0
    )
    reference = jnp.broadcast_to(reference_logits, candidates.shape)
    base = jnp.broadcast_to(base_logits, candidates.shape)
    projection_alpha = jnp.stack(
        (
            jnp.ones_like(base_kl),
            base_alpha,
            passive_alpha,
            active_alpha,
        ),
        axis=0,
    )
    mirror_kl = jnp.stack(
        (
            jnp.zeros_like(base_kl),
            base_kl,
            passive_mirror_kl,
            active_mirror_kl,
        ),
        axis=0,
    )
    temperatures = jnp.stack(
        (
            jnp.full_like(base_kl, jnp.inf),
            jnp.full_like(base_kl, jnp.inf),
            passive_temperature,
            jnp.where(
                probe_eligible,
                active_temperature,
                jnp.full_like(active_temperature, jnp.inf),
            ),
        ),
        axis=0,
    )
    final_kl = jnp.stack(
        (jnp.zeros_like(base_kl), base_kl, passive_final_kl, active_final_kl),
        axis=0,
    )
    entropy = -jnp.sum(
        posterior * jnp.log(jnp.maximum(posterior, 1.0e-30)), axis=-1
    )
    return {
        "belief": posterior,
        "posterior_entropy": entropy,
        "reference_logits": reference_logits,
        "residual_logits": residual_logits,
        "residual_composed_logits": residual_composed_logits,
        "base_policy_logits": base_logits,
        "critic_advantages": q_values,
        "ensemble_dispersion": dispersion,
        "voi": voi.value,
        "information_gain": voi.expected_information_gain,
        "probe_eligible": probe_eligible,
        "candidate_logits": candidates,
        "final_policy_logits": finals,
        "candidate_reference_kl": categorical_kl_from_logits(candidates, reference),
        "final_reference_kl": categorical_kl_from_logits(finals, reference),
        "candidate_reference_tv": _categorical_tv(candidates, reference),
        "final_reference_tv": _categorical_tv(finals, reference),
        "candidate_base_tv": _categorical_tv(candidates, base),
        "final_base_tv": _categorical_tv(finals, base),
        "mirror_base_kl": mirror_kl,
        "reported_final_kl": final_kl,
        "projection_alpha": projection_alpha,
        "adaptation_temperature": temperatures,
        "greedy_action": jnp.argmax(finals, axis=-1),
        "candidate_greedy_action": jnp.argmax(candidates, axis=-1),
    }


def _latent_history(
    *,
    model: Any,
    base_params: Any,
    latent_params: Any,
    batch: Any,
) -> Any:
    _, output = model.sequence(
        base_params,
        latent_params,
        batch.initial_policy_state,
        batch.observations[:-1],
        batch.previous_actions[:-1],
        batch.episode_starts[:-1],
        compute_latent=True,
        compute_decision=False,
        execute_adaptation=False,
    )
    return output


def _episode_shuffled_belief(
    posterior: np.ndarray, episode_starts: np.ndarray, *, seed: int
) -> np.ndarray:
    shuffled = np.empty_like(posterior)
    rng = np.random.default_rng(int(seed))
    time_count, lane_count = posterior.shape[:2]
    for lane in range(lane_count):
        boundaries = list(np.flatnonzero(episode_starts[:, lane]))
        if not boundaries or boundaries[0] != 0:
            boundaries.insert(0, 0)
        boundaries.append(time_count)
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            order = rng.permutation(np.arange(left, right))
            shuffled[left:right, lane] = posterior[order, lane]
    return shuffled


def _sample_actions(keys: Any, logits: Any) -> Any:
    import jax
    import jax.numpy as jnp

    key_array = jnp.asarray(keys).reshape((-1, 2))
    flat_logits = jnp.asarray(logits).reshape((-1, logits.shape[-1]))
    actions = jax.vmap(lambda key, row: jax.random.categorical(key, row))(
        key_array, flat_logits
    )
    return actions.reshape(logits.shape[:-1])


def _host_tree(value: Any) -> Any:
    import jax

    return jax.tree_util.tree_map(lambda item: np.asarray(jax.device_get(item)), value)


def _response_trace(
    *, target: Any, prediction: Any, behavior: Any, negative_log_likelihood: Any
) -> Mapping[str, Any]:
    return {
        "observed_response__visibility": target.direct.visibility,
        "observed_response__relative_position": target.direct.relative_position,
        "observed_response__direction": target.direct.direction,
        "observed_response__inventory": target.direct.inventory,
        "observed_response__inventory_change": target.direct.inventory_change,
        "observed_response__visible_mask": target.direct.visible_mask,
        "observed_response__event_mask": target.direct.event_mask,
        "observed_response__movement": target.direct.movement,
        "observed_response__carrying": target.direct.carrying,
        "observed_response__interface_available": target.interface_available,
        "observed_response__interface_changed": target.interface_changed,
        "observed_response__interface_event": target.interface_event,
        "observed_response__recipe_mask": target.recipe_mask,
        "observed_response__recipe_changed": target.recipe_changed,
        "response_prediction__visibility_logit": prediction.direct.visibility_logit,
        "response_prediction__relative_position_logits": (
            prediction.direct.relative_position_logits
        ),
        "response_prediction__direction_logits": prediction.direct.direction_logits,
        "response_prediction__inventory_logits": prediction.direct.inventory_logits,
        "response_prediction__inventory_change_logit": (
            prediction.direct.inventory_change_logit
        ),
        "response_prediction__interface_availability_logit": (
            prediction.interface_availability_logit
        ),
        "response_prediction__interface_change_logit": (
            prediction.interface_change_logit
        ),
        "response_prediction__interface_event_logits": (
            prediction.interface_event_logits
        ),
        "response_prediction__recipe_change_logit": prediction.recipe_change_logit,
        "behavior_features": behavior,
        "response_negative_log_likelihood": negative_log_likelihood,
    }


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _pairwise_disagreement(actions: Mapping[str, np.ndarray]) -> Mapping[str, float]:
    return {
        f"{left}__vs__{right}": float(np.mean(actions[left] != actions[right]))
        for left, right in combinations(actions, 2)
    }


def _chain_summary(chain: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    base_greedy = np.asarray(chain["greedy_action"])[1]
    eligible = np.asarray(chain["probe_eligible"], dtype=bool)
    projection: dict[str, Any] = {}
    for mode_index, mode in enumerate(EXECUTION_MODES):
        mask = eligible if mode == "active" else np.ones_like(eligible, dtype=bool)
        candidate_greedy = np.asarray(chain["candidate_greedy_action"])[mode_index]
        final_greedy = np.asarray(chain["greedy_action"])[mode_index]
        candidate_changed = (candidate_greedy != base_greedy) & mask
        reverted = candidate_changed & (final_greedy == base_greedy)
        changed_count = int(np.sum(candidate_changed))
        projection[mode] = {
            "mean_candidate_reference_kl": float(
                np.mean(np.asarray(chain["candidate_reference_kl"])[mode_index][mask])
            ),
            "mean_final_reference_kl": float(
                np.mean(np.asarray(chain["final_reference_kl"])[mode_index][mask])
            ),
            "mean_candidate_reference_tv": float(
                np.mean(np.asarray(chain["candidate_reference_tv"])[mode_index][mask])
            ),
            "mean_final_reference_tv": float(
                np.mean(np.asarray(chain["final_reference_tv"])[mode_index][mask])
            ),
            "mean_candidate_base_tv": float(
                np.mean(np.asarray(chain["candidate_base_tv"])[mode_index][mask])
            ),
            "mean_final_base_tv": float(
                np.mean(np.asarray(chain["final_base_tv"])[mode_index][mask])
            ),
            "candidate_base_greedy_disagreement": float(
                np.mean((candidate_greedy[mask] != base_greedy[mask]))
            ),
            "final_base_greedy_disagreement": float(
                np.mean((final_greedy[mask] != base_greedy[mask]))
            ),
            "candidate_changed_count": changed_count,
            "projection_reverted_count": int(np.sum(reverted)),
            "projection_reverted_given_candidate_change": (
                float(np.sum(reverted) / changed_count) if changed_count else 0.0
            ),
            "mean_projection_alpha": float(
                np.mean(np.asarray(chain["projection_alpha"])[mode_index][mask])
            ),
        }
    voi = np.asarray(chain["voi"])
    return {
        "step_count": int(np.asarray(chain["belief"]).shape[0] * np.asarray(chain["belief"]).shape[1]),
        "mean_posterior_entropy": float(np.mean(chain["posterior_entropy"])),
        "mean_absolute_residual_logit": float(np.mean(np.abs(chain["residual_logits"]))),
        "mean_absolute_critic_advantage": float(
            np.mean(np.abs(chain["critic_advantages"]))
        ),
        "mean_ensemble_dispersion": float(np.mean(chain["ensemble_dispersion"])),
        "mean_voi": float(np.mean(voi[eligible])),
        "mean_voi_action_spread": float(
            np.mean((np.max(voi, axis=-1) - np.min(voi, axis=-1))[eligible])
        ),
        "projection": projection,
    }


def _kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float:
    concordant = discordant = ties_left = ties_right = 0
    for first, second in combinations(range(left.shape[0]), 2):
        left_sign = np.sign(left[first] - left[second])
        right_sign = np.sign(right[first] - right[second])
        if left_sign == 0 and right_sign == 0:
            continue
        if left_sign == 0:
            ties_left += 1
        elif right_sign == 0:
            ties_right += 1
        elif left_sign == right_sign:
            concordant += 1
        else:
            discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    return float((concordant - discordant) / denominator) if denominator else 0.0


def _ranking_summary(predicted: np.ndarray, target: np.ndarray) -> Mapping[str, float]:
    selected = np.argmax(predicted, axis=-1)
    oracle = np.argmax(target, axis=-1)
    rows = np.arange(selected.shape[0])
    return {
        "mean_kendall_tau_b": float(
            np.mean([_kendall_tau_b(left, right) for left, right in zip(predicted, target, strict=True)])
        ),
        "top_action_agreement": float(np.mean(selected == oracle)),
        "mean_action_regret": float(
            np.mean(target[rows, oracle] - target[rows, selected])
        ),
    }


def _anchor_analysis(
    *,
    anchors: Any,
    chains: Mapping[str, Mapping[str, Any]],
    partner_runs: Any,
    lane_members: np.ndarray,
    lane_roles: np.ndarray,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    host_anchors = _host_tree(anchors)
    time_index = np.asarray(host_anchors.time_indexes, dtype=np.int64)
    lane_index = np.asarray(host_anchors.lane_indexes, dtype=np.int64)
    evaluation = np.asarray(host_anchors.evaluation_returns_by_action, dtype=np.float64)
    fit = np.asarray(host_anchors.fit_returns_by_action, dtype=np.float64)
    oracle_repeatability = _ranking_summary(fit, evaluation)

    posterior_metrics: dict[str, Any] = {}
    policy_metrics: dict[str, Any] = {}
    rows: list[Mapping[str, Any]] = []
    gathered: dict[str, Any] = {}
    for posterior_name, raw_chain in chains.items():
        chain = _host_tree(raw_chain)
        q_values = np.asarray(chain["critic_advantages"])[time_index, lane_index]
        final_logits = np.asarray(chain["final_policy_logits"])[
            :, time_index, lane_index
        ]
        probabilities = np.exp(
            final_logits - np.max(final_logits, axis=-1, keepdims=True)
        )
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        posterior_metrics[posterior_name] = _ranking_summary(q_values, evaluation)
        policy_metrics[posterior_name] = {}
        for mode_index, mode in enumerate(EXECUTION_MODES):
            expected = np.sum(probabilities[mode_index] * evaluation, axis=-1)
            greedy = np.argmax(final_logits[mode_index], axis=-1)
            oracle = np.argmax(evaluation, axis=-1)
            anchor_rows = np.arange(evaluation.shape[0])
            policy_metrics[posterior_name][mode] = {
                "mean_same_world_expected_return": float(np.mean(expected)),
                "greedy_top_action_agreement": float(np.mean(greedy == oracle)),
                "mean_greedy_action_regret": float(
                    np.mean(evaluation[anchor_rows, oracle] - evaluation[anchor_rows, greedy])
                ),
            }
        gathered[posterior_name] = {
            "chain": chain,
            "q": q_values,
            "probability": probabilities,
        }

    observed = gathered["observed"]
    posterior_effects = {}
    for control in ("uniform", "episode_shuffle"):
        current = gathered[control]
        posterior_effects[f"observed__vs__{control}"] = {
            "mean_absolute_residual_logit_change": float(
                np.mean(
                    np.abs(
                        observed["chain"]["residual_logits"][time_index, lane_index]
                        - current["chain"]["residual_logits"][time_index, lane_index]
                    )
                )
            ),
            "mean_absolute_critic_change": float(
                np.mean(np.abs(observed["q"] - current["q"]))
            ),
            "critic_greedy_disagreement": float(
                np.mean(np.argmax(observed["q"], axis=-1) != np.argmax(current["q"], axis=-1))
            ),
            "critic_top_action_agreement_increment": float(
                posterior_metrics["observed"]["top_action_agreement"]
                - posterior_metrics[control]["top_action_agreement"]
            ),
            "critic_kendall_increment": float(
                posterior_metrics["observed"]["mean_kendall_tau_b"]
                - posterior_metrics[control]["mean_kendall_tau_b"]
            ),
        }

    for index in range(evaluation.shape[0]):
        member = int(lane_members[lane_index[index]])
        row: dict[str, Any] = {
            "anchor_index": index,
            "time_index": int(time_index[index]),
            "lane_index": int(lane_index[index]),
            "partner_run_index": member,
            "partner_run_id": partner_runs[member].run_id,
            "ego_role": int(lane_roles[lane_index[index]]),
            "fit_crn_returns_by_action": fit[index].tolist(),
            "evaluation_crn_returns_by_action": evaluation[index].tolist(),
        }
        for posterior_name in POSTERIOR_INTERVENTIONS:
            q_values = gathered[posterior_name]["q"][index]
            probability = gathered[posterior_name]["probability"][:, index]
            row[posterior_name] = {
                "critic_advantages": q_values.tolist(),
                "critic_greedy_action": int(np.argmax(q_values)),
                "mode_probabilities": {
                    mode: probability[mode_index].tolist()
                    for mode_index, mode in enumerate(EXECUTION_MODES)
                },
                "mode_same_world_expected_return": {
                    mode: float(np.dot(probability[mode_index], evaluation[index]))
                    for mode_index, mode in enumerate(EXECUTION_MODES)
                },
            }
        rows.append(row)

    return (
        {
            "anchor_count": int(evaluation.shape[0]),
            "fit_evaluation_oracle_repeatability": oracle_repeatability,
            "posterior_ranking": posterior_metrics,
            "policy_same_world": policy_metrics,
            "posterior_effects": posterior_effects,
        },
        rows,
    )


def _save_chain_archive(
    path: Path,
    *,
    chain: Mapping[str, Any],
    response_trace: Mapping[str, Any],
    predictive_belief: Any,
    actions: Any,
    action_keys: Any,
    environment_keys: Any,
    lane_members: np.ndarray,
    lane_roles: np.ndarray,
) -> None:
    host = _host_tree(chain)
    predictive = np.asarray(_host_tree(predictive_belief))
    posterior = np.asarray(host["belief"])
    filter_kl = np.sum(
        posterior
        * (
            np.log(np.maximum(posterior, 1.0e-30))
            - np.log(np.maximum(predictive, 1.0e-30))
        ),
        axis=-1,
    )
    payload = {
        **host,
        **_host_tree(response_trace),
        "predictive_belief": predictive,
        "posterior_increment": posterior - predictive,
        "posterior_increment_kl": filter_kl,
        "executed_action": np.asarray(_host_tree(actions)),
        "action_keys": np.asarray(_host_tree(action_keys), dtype=np.uint32),
        "environment_keys": np.asarray(
            _host_tree(environment_keys), dtype=np.uint32
        ),
        "lane_partner_run_index": np.asarray(lane_members, dtype=np.int32),
        "lane_ego_role": np.asarray(lane_roles, dtype=np.int32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _posterior_effect_summary(
    observed: Mapping[str, np.ndarray], control: Mapping[str, np.ndarray]
) -> Mapping[str, float]:
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probability = np.exp(shifted)
        return probability / np.sum(probability, axis=-1, keepdims=True)

    observed_base = softmax(observed["base_policy_logits"])
    control_base = softmax(control["base_policy_logits"])
    return {
        "mean_absolute_residual_logit_change": float(
            np.mean(np.abs(observed["residual_logits"] - control["residual_logits"]))
        ),
        "mean_base_policy_tv": float(
            np.mean(0.5 * np.sum(np.abs(observed_base - control_base), axis=-1))
        ),
        "mean_absolute_critic_change": float(
            np.mean(
                np.abs(observed["critic_advantages"] - control["critic_advantages"])
            )
        ),
        "critic_greedy_disagreement": float(
            np.mean(
                np.argmax(observed["critic_advantages"], axis=-1)
                != np.argmax(control["critic_advantages"], axis=-1)
            )
        ),
    }


def run_decision_chain_audit(args: argparse.Namespace) -> None:
    """Run the matched-key four-mode and posterior-intervention audit."""

    started = time.perf_counter()
    import jax
    import jax.numpy as jnp

    from .official_adapter import FrozenPartnerPool, VectorEnvironment
    from .training_app import _anchor_functions

    config = load_config(args.config, run_kind=args.run_kind)
    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_file_check),
    )
    partner_role = str(args.partner_role)
    partner_runs = manifest.by_role(partner_role)
    if not partner_runs:
        raise ValueError("Decision-chain audit partner role is empty.")

    deployment_paths = tuple(Path(value).resolve() for value in args.deployment)
    deployments = tuple(load_deployment(path) for path in deployment_paths)
    for deployment in deployments:
        if deployment.config.environment.layout != config.environment.layout:
            raise ValueError("Decision-chain deployment layout differs.")
        if deployment.config.method_variant != "delta_active":
            raise ValueError("Decision-chain audit requires delta_active checkpoints.")

    lane_count = 2 * len(partner_runs)
    audit_config = replace(
        config,
        environment=replace(config.environment, num_envs=lane_count),
        training=replace(
            config.training, rollout_length=int(args.trajectory_steps)
        ),
    )
    environment = VectorEnvironment.create(audit_config)
    pool = FrozenPartnerPool.from_checkpoints(
        tuple(run.checkpoint for run in partner_runs),
        parent_training_run_ids=tuple(
            run.parent_training_run_id for run in partner_runs
        ),
    )
    lane_members = np.repeat(np.arange(len(partner_runs), dtype=np.int32), 2)
    lane_roles = np.tile(np.asarray((0, 1), dtype=np.int32), len(partner_runs))
    partner_functions = make_fixed_partner_functions(
        pool=pool,
        member_indexes=jnp.asarray(lane_members),
        run_ids=jnp.arange(len(partner_runs), dtype=jnp.int32),
    )

    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        {
            "stage": "decision-chain-audit",
            "method": METHOD_VERSION,
            "layout": config.environment.layout,
            "deployments": [{"path": str(path)} for path in deployment_paths],
            "partner_manifest": {"path": str(manifest_path)},
            "partner_role": partner_role,
            "execution_modes": list(EXECUTION_MODES),
            "posterior_interventions": list(POSTERIOR_INTERVENTIONS),
            "root_seed": int(args.seed),
            "trajectory_steps": int(args.trajectory_steps),
            "anchor_states": int(args.anchor_states),
        },
    )

    model = deployments[0].model

    @jax.jit
    def history_kernel(base_params: Any, latent_params: Any, batch: Any):
        output_history = _latent_history(
            model=model,
            base_params=base_params,
            latent_params=latent_params,
            batch=batch,
        )
        return (
            output_history.predictive_belief,
            output_history.belief,
            output_history.behavior_features,
            output_history.response_negative_log_likelihood,
            output_history.response_prediction,
        )

    @jax.jit
    def chain_kernel(
        base_params: Any,
        latent_params: Any,
        initial_task_carry: Any,
        observations: Any,
        episode_starts: Any,
        behavior: Any,
        belief: Any,
    ):
        return _decision_chain(
            model=model,
            base_params=base_params,
            latent_params=latent_params,
            initial_task_carry=initial_task_carry,
            observations=observations,
            episode_starts=episode_starts,
            behavior=behavior,
            belief=belief,
        )

    seed_summaries = []
    total_steps = 0
    for ego_index, deployment in enumerate(deployments):
        seed_dir = output / f"seed-{ego_index}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        root = jax.random.fold_in(jax.random.PRNGKey(int(args.seed)), ego_index)
        mode_actions: dict[str, np.ndarray] = {}
        mode_environment_keys: dict[str, np.ndarray] = {}
        mode_action_keys: dict[str, np.ndarray] = {}
        mode_summaries: dict[str, Any] = {}
        active_batch = active_records = active_history = active_observed_chain = None

        for mode_index, mode in enumerate(EXECUTION_MODES):
            runner = initialize_runner(
                environment=environment,
                model=deployment.model,
                partner_functions=partner_functions,
                random_key=root,
            )
            _, batch, records = collect_rollout(
                state=runner,
                length=int(args.trajectory_steps),
                environment=environment,
                model=deployment.model,
                base_params=deployment.base_params,
                latent_params=deployment.latent_params,
                partner_functions=partner_functions,
                partner_parameters=None,
                official_shaping_factor=0.0,
                record_anchors=True,
                use_deployment_policy=True,
                execution_mode=mode,
            )
            (
                predictive,
                posterior,
                behavior,
                response_nll,
                response_prediction,
            ) = history_kernel(deployment.base_params, deployment.latent_params, batch)
            response_target = extract_response_target(
                records["ego_policy_state"].previous_observation,
                batch.observations[:-1],
                batch.previous_actions[:-1],
                batch.episode_starts[:-1],
            )
            chain = chain_kernel(
                deployment.base_params,
                deployment.latent_params,
                batch.initial_policy_state.task_carry,
                batch.observations[:-1],
                batch.episode_starts[:-1],
                behavior,
                posterior,
            )
            replay_actions = _sample_actions(
                records["ego_action_keys"], chain["final_policy_logits"][mode_index]
            )
            np.testing.assert_array_equal(
                np.asarray(jax.device_get(replay_actions)),
                np.asarray(jax.device_get(batch.actions)),
            )
            selected_logits = np.asarray(
                jax.device_get(chain["final_policy_logits"][mode_index])
            )
            np.testing.assert_allclose(
                selected_logits,
                np.asarray(jax.device_get(records["deployment_policy_logits"])),
                atol=1.0e-3,
                rtol=1.0e-3,
            )
            host_chain = _host_tree(chain)
            host_chain["predictive_belief"] = np.asarray(jax.device_get(predictive))
            host_chain["posterior_increment"] = (
                host_chain["belief"] - host_chain["predictive_belief"]
            )
            host_chain["posterior_increment_kl"] = np.sum(
                host_chain["belief"]
                * (
                    np.log(np.maximum(host_chain["belief"], 1.0e-30))
                    - np.log(
                        np.maximum(host_chain["predictive_belief"], 1.0e-30)
                    )
                ),
                axis=-1,
            )
            mode_summary = dict(_chain_summary(host_chain))
            mode_summary["mean_posterior_increment_kl"] = float(
                np.mean(host_chain["posterior_increment_kl"])
            )
            mode_summary["mean_response_negative_log_likelihood"] = float(
                np.mean(np.asarray(jax.device_get(response_nll)))
            )
            mode_summaries[mode] = mode_summary
            _save_chain_archive(
                seed_dir / f"trajectory_{mode}.npz",
                chain=chain,
                response_trace=_response_trace(
                    target=response_target,
                    prediction=response_prediction,
                    behavior=behavior,
                    negative_log_likelihood=response_nll,
                ),
                predictive_belief=predictive,
                actions=batch.actions,
                action_keys=records["ego_action_keys"],
                environment_keys=records["environment_keys"],
                lane_members=lane_members,
                lane_roles=lane_roles,
            )
            mode_actions[mode] = np.asarray(jax.device_get(batch.actions))
            mode_environment_keys[mode] = np.asarray(
                jax.device_get(records["environment_keys"]), dtype=np.uint32
            )
            mode_action_keys[mode] = np.asarray(
                jax.device_get(records["ego_action_keys"]), dtype=np.uint32
            )
            if mode == "active":
                active_batch = batch
                active_records = records
                active_history = (predictive, posterior, behavior)
                active_observed_chain = chain

        first_mode = EXECUTION_MODES[0]
        if any(
            not np.array_equal(mode_environment_keys[first_mode], mode_environment_keys[mode])
            or not np.array_equal(mode_action_keys[first_mode], mode_action_keys[mode])
            for mode in EXECUTION_MODES[1:]
        ):
            raise RuntimeError("Four-mode audit did not preserve matched random keys.")

        predictive, posterior, behavior = active_history
        posterior_host = np.asarray(jax.device_get(posterior))
        starts_host = np.asarray(
            jax.device_get(active_batch.episode_starts[:-1]), dtype=bool
        )
        posterior_paths = {
            "observed": posterior,
            "uniform": jnp.full_like(
                posterior, 1.0 / deployment.config.method.latent_components
            ),
            "episode_shuffle": jnp.asarray(
                _episode_shuffled_belief(
                    posterior_host,
                    starts_host,
                    seed=int(args.seed) + 10_000 + ego_index,
                )
            ),
        }
        intervention_chains: dict[str, Any] = {
            "observed": active_observed_chain
        }
        for name in ("uniform", "episode_shuffle"):
            intervention_chains[name] = chain_kernel(
                deployment.base_params,
                deployment.latent_params,
                active_batch.initial_policy_state.task_carry,
                active_batch.observations[:-1],
                active_batch.episode_starts[:-1],
                behavior,
                posterior_paths[name],
            )

        intervention_payload: dict[str, np.ndarray] = {
            "action_keys": mode_action_keys["active"],
            "environment_keys": mode_environment_keys["active"],
            "lane_partner_run_index": lane_members,
            "lane_ego_role": lane_roles,
        }
        matched_actions: dict[str, dict[str, np.ndarray]] = {}
        intervention_summaries: dict[str, Any] = {}
        host_intervention_chains: dict[str, Any] = {}
        for name in POSTERIOR_INTERVENTIONS:
            host_chain = _host_tree(intervention_chains[name])
            host_intervention_chains[name] = host_chain
            for field, value in host_chain.items():
                intervention_payload[f"{name}__{field}"] = value
            current_actions = {}
            for mode_index, mode in enumerate(EXECUTION_MODES):
                sampled = np.asarray(
                    jax.device_get(
                        _sample_actions(
                            active_records["ego_action_keys"],
                            intervention_chains[name]["final_policy_logits"][mode_index],
                        )
                    )
                )
                current_actions[mode] = sampled
                intervention_payload[f"{name}__matched_action__{mode}"] = sampled
            matched_actions[name] = current_actions
            current_summary = dict(_chain_summary(host_chain))
            current_summary["matched_key_action_disagreement"] = _pairwise_disagreement(
                current_actions
            )
            greedy = {
                mode: host_chain["greedy_action"][mode_index]
                for mode_index, mode in enumerate(EXECUTION_MODES)
            }
            current_summary["greedy_disagreement"] = _pairwise_disagreement(greedy)
            intervention_summaries[name] = current_summary
        np.savez_compressed(
            seed_dir / "posterior_interventions.npz", **intervention_payload
        )

        posterior_effects = {
            f"observed__vs__{control}": _posterior_effect_summary(
                host_intervention_chains["observed"],
                host_intervention_chains[control],
            )
            for control in ("uniform", "episode_shuffle")
        }
        functions = _anchor_functions(
            model=deployment.model,
            partner_functions=partner_functions,
            partner_parameters=None,
            environment=environment,
        )
        anchors = collect_anchor_batch(
            key=jax.random.fold_in(root, 31_000),
            records=active_records,
            functions=functions,
            base_params=deployment.base_params,
            latent_params=deployment.latent_params,
            states_per_trigger=int(args.anchor_states),
            action_count=OFFICIAL_ACTION_COUNT,
            fit_replicas=config.anchors.fit_replicas,
            evaluation_replicas=config.anchors.evaluation_replicas,
            horizon=config.method.continuation_horizon,
            gamma=config.ppo.gamma,
            collect_successor=False,
        )
        anchor_summary, anchor_rows = _anchor_analysis(
            anchors=anchors,
            chains=intervention_chains,
            partner_runs=partner_runs,
            lane_members=lane_members,
            lane_roles=lane_roles,
        )
        _write_jsonl(seed_dir / "anchor_rows.jsonl", anchor_rows)
        write_json(seed_dir / "anchor_summary.json", anchor_summary)

        seed_summary = {
            "version": DECISION_CHAIN_AUDIT_VERSION,
            "artifact_type": "delta_decision_chain_seed_audit",
            "ego_run_index": ego_index,
            "ego_run_id": deployment.ego_run_id,
            "partner_count": len(partner_runs),
            "lane_count": lane_count,
            "trajectory_steps_per_lane": int(args.trajectory_steps),
            "matched_environment_and_action_keys": True,
            "trajectory_mode_summaries": mode_summaries,
            "divergent_trajectory_executed_action_disagreement": _pairwise_disagreement(
                mode_actions
            ),
            "posterior_interventions": intervention_summaries,
            "posterior_effects": posterior_effects,
            "anchors": anchor_summary,
        }
        write_json(seed_dir / "seed_summary.json", seed_summary)
        seed_summaries.append(seed_summary)
        total_steps += lane_count * int(args.trajectory_steps) * len(EXECUTION_MODES)
        total_steps += (
            int(args.anchor_states)
            * OFFICIAL_ACTION_COUNT
            * (config.anchors.fit_replicas + config.anchors.evaluation_replicas)
            * config.method.continuation_horizon
        )

    elapsed = time.perf_counter() - started
    ledger = ResourceLedger(
        intervention_steps=total_steps,
        measurement_wall_clock_hours=elapsed / 3600.0,
        measurement_gpu_hours=(
            elapsed / 3600.0
            if any(device.platform == "gpu" for device in jax.devices())
            else 0.0
        ),
    )
    write_json(
        output / "decision_chain_audit.json",
        {
            "version": DECISION_CHAIN_AUDIT_VERSION,
            "artifact_type": "delta_decision_chain_audit",
            "method": METHOD_VERSION,
            "layout": config.environment.layout,
            "execution_modes": list(EXECUTION_MODES),
            "posterior_interventions": list(POSTERIOR_INTERVENTIONS),
            "partner_role": partner_role,
            "partner_run_ids": [run.run_id for run in partner_runs],
            "trajectory_key_schedule": (
                "fold_in_root_by_ego_then_vector_lanes_partner_role_pairs"
            ),
            "seed_summaries": seed_summaries,
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _evaluation_node_matrix(rows: list[Mapping[str, Any]]) -> np.ndarray:
    egos = sorted({int(row["ego_run_index"]) for row in rows})
    partners = sorted({str(row["partner_run_id"]) for row in rows})
    matrix = np.empty((len(egos), len(partners)), dtype=np.float64)
    for ego_offset, ego in enumerate(egos):
        for partner_offset, partner in enumerate(partners):
            matrix[ego_offset, partner_offset] = np.mean(
                [
                    float(row["raw_return"])
                    for row in rows
                    if int(row["ego_run_index"]) == ego
                    and str(row["partner_run_id"]) == partner
                ]
            )
    return matrix


def _crossed_mode_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> Mapping[str, Any]:
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(replicates),), dtype=np.float64)
    for index in range(int(replicates)):
        ego = rng.integers(0, left.shape[0], size=left.shape[0])
        partner = rng.integers(0, left.shape[1], size=left.shape[1])
        draws[index] = np.mean(
            left[np.ix_(ego, partner)] - right[np.ix_(ego, partner)]
        )
    return {
        "estimate": float(np.mean(left - right)),
        "interval_95": [float(value) for value in np.quantile(draws, (0.025, 0.975))],
        "ego_seed_means": [
            float(value) for value in np.mean(left - right, axis=1)
        ],
    }


def _mean_seed_metric(
    seed_summaries: list[Mapping[str, Any]], *path: str
) -> float:
    values = []
    for summary in seed_summaries:
        current: Any = summary
        for name in path:
            current = current[name]
        values.append(float(current))
    return float(np.mean(values))


def _format_interval(value: Mapping[str, Any]) -> str:
    lower, upper = value["interval_95"]
    return f'{value["estimate"]:.6f} [{lower:.6f}, {upper:.6f}]'


def summarize_decision_chain_audit(args: argparse.Namespace) -> None:
    """Join matched-key episodic returns to chain and CRN diagnostics."""

    audit_path = Path(args.audit).resolve()
    audit = read_json(audit_path)
    if audit.get("artifact_type") != "delta_decision_chain_audit":
        raise ValueError("Decision-chain audit identity differs.")
    seed_summaries = list(audit["seed_summaries"])

    evaluation_rows: dict[str, list[Mapping[str, Any]]] = {}
    evaluation_summaries: dict[str, Mapping[str, Any]] = {}
    matrices: dict[str, np.ndarray] = {}
    reference_keys = reference_environment = reference_identity = None
    for value in args.evaluation:
        mode, raw_path = value.split("=", 1)
        if mode not in EXECUTION_MODES or mode in evaluation_rows:
            raise ValueError("Decision-chain evaluation mode is invalid or duplicated.")
        directory = Path(raw_path).resolve()
        summary = read_json(directory / "evaluation_summary.json")
        if (
            summary.get("artifact_type") != "delta_raw_evaluation"
            or summary.get("evaluation_mode") != "common_partner"
            or summary.get("execution_mode") != mode
            or summary.get("partner_role") != "development_support"
        ):
            raise ValueError("Decision-chain episodic evaluation identity differs.")
        rows = _read_jsonl(Path(summary["raw"]["path"]).resolve())
        keys = {
            (
                int(row["ego_run_index"]),
                str(row["partner_run_id"]),
                int(row["ego_role"]),
                int(row["episode_index"]),
            )
            for row in rows
        }
        environment_keys = {
            (
                int(row["ego_run_index"]),
                str(row["partner_run_id"]),
                int(row["ego_role"]),
                int(row["episode_index"]),
            ): tuple(int(word) for word in row["environment_key"])
            for row in rows
        }
        identity = (
            str(Path(summary["policy_manifest"]["path"]).resolve()),
            str(Path(summary["partner_manifest"]["path"]).resolve()),
            str(summary["layout"]),
            int(summary["root_seed"]),
            str(summary["key_schedule"]),
            str(summary["observation_protocol"]),
        )
        if reference_keys is None:
            reference_keys = keys
            reference_environment = environment_keys
            reference_identity = identity
        elif (
            keys != reference_keys
            or environment_keys != reference_environment
            or identity != reference_identity
        ):
            raise ValueError("Four episodic modes are not paired on identical keys.")
        evaluation_rows[mode] = rows
        evaluation_summaries[mode] = summary
        matrices[mode] = _evaluation_node_matrix(rows)
    if set(evaluation_rows) != set(EXECUTION_MODES):
        raise ValueError("Decision-chain report requires all four execution modes.")

    contrast_pairs = (
        ("residual", "reference_only"),
        ("passive", "residual"),
        ("active", "passive"),
        ("active", "residual"),
        ("active", "reference_only"),
    )
    episodic_contrasts = {
        f"{left}__minus__{right}": _crossed_mode_difference(
            matrices[left],
            matrices[right],
            replicates=int(args.bootstrap_replicates),
            seed=int(args.seed) + index,
        )
        for index, (left, right) in enumerate(contrast_pairs)
    }
    episodic_means = {
        mode: float(np.mean(matrix)) for mode, matrix in matrices.items()
    }

    chain_metrics = (
        "mean_posterior_entropy",
        "mean_posterior_increment_kl",
        "mean_response_negative_log_likelihood",
        "mean_absolute_residual_logit",
        "mean_absolute_critic_advantage",
        "mean_ensemble_dispersion",
        "mean_voi",
        "mean_voi_action_spread",
    )
    trajectory_diagnostics = {
        mode: {
            metric: _mean_seed_metric(
                seed_summaries, "trajectory_mode_summaries", mode, metric
            )
            for metric in chain_metrics
        }
        for mode in EXECUTION_MODES
    }
    action_pairs = tuple(
        f"{left}__vs__{right}" for left, right in combinations(EXECUTION_MODES, 2)
    )
    divergent_action_disagreement = {
        pair: _mean_seed_metric(
            seed_summaries,
            "divergent_trajectory_executed_action_disagreement",
            pair,
        )
        for pair in action_pairs
    }
    matched_action_disagreement = {
        posterior: {
            pair: _mean_seed_metric(
                seed_summaries,
                "posterior_interventions",
                posterior,
                "matched_key_action_disagreement",
                pair,
            )
            for pair in action_pairs
        }
        for posterior in POSTERIOR_INTERVENTIONS
    }
    greedy_action_disagreement = {
        posterior: {
            pair: _mean_seed_metric(
                seed_summaries,
                "posterior_interventions",
                posterior,
                "greedy_disagreement",
                pair,
            )
            for pair in action_pairs
        }
        for posterior in POSTERIOR_INTERVENTIONS
    }

    posterior_ranking = {
        posterior: {
            metric: _mean_seed_metric(
                seed_summaries, "anchors", "posterior_ranking", posterior, metric
            )
            for metric in (
                "mean_kendall_tau_b",
                "top_action_agreement",
                "mean_action_regret",
            )
        }
        for posterior in POSTERIOR_INTERVENTIONS
    }
    posterior_effects = {
        control: {
            metric: _mean_seed_metric(
                seed_summaries,
                "anchors",
                "posterior_effects",
                f"observed__vs__{control}",
                metric,
            )
            for metric in (
                "mean_absolute_residual_logit_change",
                "mean_absolute_critic_change",
                "critic_greedy_disagreement",
                "critic_top_action_agreement_increment",
                "critic_kendall_increment",
            )
        }
        for control in ("uniform", "episode_shuffle")
    }
    projection = {
        posterior: {
            metric: _mean_seed_metric(
                seed_summaries,
                "posterior_interventions",
                posterior,
                "projection",
                "active",
                metric,
            )
            for metric in (
                "mean_candidate_reference_kl",
                "mean_final_reference_kl",
                "mean_candidate_reference_tv",
                "mean_final_reference_tv",
                "mean_candidate_base_tv",
                "mean_final_base_tv",
                "candidate_base_greedy_disagreement",
                "final_base_greedy_disagreement",
                "projection_reverted_given_candidate_change",
                "mean_projection_alpha",
            )
        }
        for posterior in POSTERIOR_INTERVENTIONS
    }
    same_world = {
        posterior: {
            mode: _mean_seed_metric(
                seed_summaries,
                "anchors",
                "policy_same_world",
                posterior,
                mode,
                "mean_same_world_expected_return",
            )
            for mode in EXECUTION_MODES
        }
        for posterior in POSTERIOR_INTERVENTIONS
    }
    same_world_contrasts = {
        posterior: {
            f"{left}__minus__{right}": same_world[posterior][left]
            - same_world[posterior][right]
            for left, right in contrast_pairs
        }
        for posterior in POSTERIOR_INTERVENTIONS
    }
    oracle_repeatability = {
        metric: _mean_seed_metric(
            seed_summaries,
            "anchors",
            "fit_evaluation_oracle_repeatability",
            metric,
        )
        for metric in (
            "mean_kendall_tau_b",
            "top_action_agreement",
            "mean_action_regret",
        )
    }

    summary_payload = {
        "version": DECISION_CHAIN_AUDIT_VERSION,
        "artifact_type": "delta_decision_chain_report",
        "method": METHOD_VERSION,
        "layout": audit["layout"],
        "exploratory_panel": "development_support",
        "posterior_intervention_definitions": POSTERIOR_INTERVENTION_DEFINITIONS,
        "matched_episodic_environment_keys": True,
        "episodic_means": episodic_means,
        "episodic_contrasts": episodic_contrasts,
        "trajectory_chain_diagnostics": trajectory_diagnostics,
        "divergent_trajectory_executed_action_disagreement": (
            divergent_action_disagreement
        ),
        "matched_key_executed_action_disagreement": matched_action_disagreement,
        "greedy_action_disagreement": greedy_action_disagreement,
        "held_out_crn_oracle_repeatability": oracle_repeatability,
        "posterior_ranking": posterior_ranking,
        "posterior_effects": posterior_effects,
        "active_projection": projection,
        "same_world_crn_expected_return": same_world,
        "same_world_crn_contrasts": same_world_contrasts,
        "audit_source": {"path": str(audit_path)},
        "evaluation_sources": {
            mode: {"path": str(Path(path.split("=", 1)[1]).resolve())}
            for mode, path in zip(
                (item.split("=", 1)[0] for item in args.evaluation),
                args.evaluation,
                strict=True,
            )
        },
    }
    output_path = Path(args.output).resolve()
    write_json(output_path.with_suffix(".json"), summary_payload)

    lines = [
        "# DELTA decision-chain audit",
        "",
        "## Scope",
        "",
        (
            "Five frozen `delta_active` checkpoints were evaluated against the same "
            "`development_support` partners. The four execution modes use identical "
            "partner, role, episode-index, action-key and environment-key schedules. "
            "The partner panel is exploratory and is not confirmatory evidence."
        ),
        "",
        (
            "`episode_shuffle` permutes posterior time points within each realized "
            "episode and lane. Observations, legal behavior statistics, anchor worlds "
            "and random keys remain fixed; policy outputs and matched-key actions are "
            "recomputed."
        ),
        "",
        "## Matched-key episodic returns",
        "",
        "| execution mode | mean episodic return |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{mode}` | {episodic_means[mode]:.6f} |" for mode in EXECUTION_MODES
    )
    lines.extend(
        [
            "",
            "| paired contrast | mean [crossed 95% interval] |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{name}` | {_format_interval(value)} |"
        for name, value in episodic_contrasts.items()
    )
    lines.extend(
        [
            "",
            "## Per-step chain diagnostics",
            "",
            "The table reports five-checkpoint means; the trajectory archives retain every step of the observed-response, posterior, residual, critic, mirror, projection and executed-action chain.",
            "",
            "| mode trajectory | posterior entropy | posterior increment KL | response NLL | abs residual logit | abs critic advantage | ensemble dispersion | VOI | VOI action spread |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, row in trajectory_diagnostics.items():
        lines.append(
            f'| `{mode}` | {row["mean_posterior_entropy"]:.6f} | '
            f'{row["mean_posterior_increment_kl"]:.6f} | '
            f'{row["mean_response_negative_log_likelihood"]:.6f} | '
            f'{row["mean_absolute_residual_logit"]:.6f} | '
            f'{row["mean_absolute_critic_advantage"]:.6f} | '
            f'{row["mean_ensemble_dispersion"]:.6f} | '
            f'{row["mean_voi"]:.6f} | '
            f'{row["mean_voi_action_spread"]:.6f} |'
        )
    lines.extend(
        [
            "",
            "## Action disagreement",
            "",
            "Divergent-trajectory values compare realized rollouts under identical random-key schedules; their visited states may diverge after actions differ.",
            "",
            "| mode pair | divergent-trajectory executed-action disagreement |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{pair}` | {value:.6f} |"
        for pair, value in divergent_action_disagreement.items()
    )
    lines.extend(
        [
            "",
            "The following comparisons hold the active trajectory fixed and replay every posterior/mode with the same action keys.",
            "",
            "| posterior | mode pair | matched-key executed-action disagreement | greedy disagreement |",
            "|---|---|---:|---:|",
        ]
    )
    for posterior in POSTERIOR_INTERVENTIONS:
        for pair in action_pairs:
            lines.append(
                f"| `{posterior}` | `{pair}` | "
                f"{matched_action_disagreement[posterior][pair]:.6f} | "
                f"{greedy_action_disagreement[posterior][pair]:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Held-out CRN action ranking",
            "",
            "| posterior | Kendall tau-b | top-action agreement | mean regret |",
            "|---|---:|---:|---:|",
        ]
    )
    for posterior in POSTERIOR_INTERVENTIONS:
        row = posterior_ranking[posterior]
        lines.append(
            f'| `{posterior}` | {row["mean_kendall_tau_b"]:.6f} | '
            f'{row["top_action_agreement"]:.6f} | {row["mean_action_regret"]:.6f} |'
        )
    lines.extend(
        [
            "",
            (
                "Fit/evaluation CRN oracle repeatability: "
                f'Kendall tau-b {oracle_repeatability["mean_kendall_tau_b"]:.6f}, '
                f'top-action agreement {oracle_repeatability["top_action_agreement"]:.6f}, '
                f'mean regret {oracle_repeatability["mean_action_regret"]:.6f}.'
            ),
            "",
            "## Posterior intervention effects at matched anchor states",
            "",
            "| observed posterior vs control | residual-logit change | critic change | critic greedy disagreement | top-agreement increment | Kendall increment |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for control, row in posterior_effects.items():
        lines.append(
            f'| `{control}` | {row["mean_absolute_residual_logit_change"]:.6f} | '
            f'{row["mean_absolute_critic_change"]:.6f} | '
            f'{row["critic_greedy_disagreement"]:.6f} | '
            f'{row["critic_top_action_agreement_increment"]:.6f} | '
            f'{row["critic_kendall_increment"]:.6f} |'
        )
    lines.extend(
        [
            "",
            "## Active mirror and final projection",
            "",
            "| posterior | candidate KL(ref) | final KL(ref) | candidate TV(ref) | final TV(ref) | candidate TV(base) | final TV(base) | candidate/base greedy diff | final/base greedy diff | reverted among candidate changes | projection alpha |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for posterior, row in projection.items():
        lines.append(
            f'| `{posterior}` | {row["mean_candidate_reference_kl"]:.6f} | '
            f'{row["mean_final_reference_kl"]:.6f} | '
            f'{row["mean_candidate_reference_tv"]:.6f} | '
            f'{row["mean_final_reference_tv"]:.6f} | '
            f'{row["mean_candidate_base_tv"]:.6f} | '
            f'{row["mean_final_base_tv"]:.6f} | '
            f'{row["candidate_base_greedy_disagreement"]:.6f} | '
            f'{row["final_base_greedy_disagreement"]:.6f} | '
            f'{row["projection_reverted_given_candidate_change"]:.6f} | '
            f'{row["mean_projection_alpha"]:.6f} |'
        )
    lines.extend(
        [
            "",
            "## Same-world CRN expected continuation return",
            "",
            "These are policy expectations against held-out all-action CRN continuation vectors, not episodic evaluation returns.",
            "",
            "| posterior | reference_only | residual | passive | active | active-passive | active-residual |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for posterior in POSTERIOR_INTERVENTIONS:
        values = same_world[posterior]
        contrasts = same_world_contrasts[posterior]
        lines.append(
            f'| `{posterior}` | {values["reference_only"]:.6f} | '
            f'{values["residual"]:.6f} | {values["passive"]:.6f} | '
            f'{values["active"]:.6f} | '
            f'{contrasts["active__minus__passive"]:.6f} | '
            f'{contrasts["active__minus__residual"]:.6f} |'
        )
    lines.extend(
        [
            "",
            "## Fault-localization rules",
            "",
            "- Compare observed-posterior ranking and residual/critic changes against uniform and episode-shuffled controls first.",
            "- If that link is present, compare critic ranking, agreement and regret against held-out CRN vectors.",
            "- If critic ranking is retained, compare the mirror candidate with the final reference-relative projection.",
            "- If final actions change, compare same-world CRN and matched episodic return contrasts.",
            "- A positive local CRN contrast with a null episodic contrast isolates the local-estimand/long-horizon mismatch.",
            "",
            "No additional pass threshold or weighted diagnostic score is applied.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "run_belief_value_intervention",
    "run_decision_chain_audit",
    "summarize_decision_chain_audit",
]
