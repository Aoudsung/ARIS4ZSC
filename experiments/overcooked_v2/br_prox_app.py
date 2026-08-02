"""Empirical held-out local BR-Prox audit for frozen DELTA-ZSC deployments.

The audit never treats the learned critic as ground truth.  At legal-history
states sampled from confirmatory trajectories, it clones the simulator and
frozen partner recurrent state, forces every ego action, and evaluates the
fit-selected action on an independent future-replica split.  The resulting
quantity is a local one-action-deviation approximation to best-response
proximity for the deployed continuation controller; it is deliberately not
claimed to be the unrestricted full-policy best response.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from experiments.overcooked_v2.deployment import (
    deployment_action,
    reset_deployment_state,
    update_after_transition,
)
from experiments.overcooked_v2.official_adapter import FrozenPartnerPool, VectorEnvironment
from src.path_c.anchor_sampling import gather_time_lanes, stratified_anchor_indexes
from src.path_c.calibration import predicted_policy_gain
from src.path_c.counterfactual_anchor import (
    AnchorFunctions,
    AnchorWorld,
    collect_counterfactual_anchors,
)
from src.path_c.evaluation import br_prox


class _AuditState(NamedTuple):
    environment_state: Any
    observations: Any
    ego_state: Any
    partner_carry: Any
    partner_episode_start: Any


def canonical_br_prox_anchor_world(
    *,
    environment_state: Any,
    observations: Any,
    ego_state: Any,
    partner_state: Any,
    partner_episode_start: Any,
) -> AnchorWorld:
    """Build an anchor world after observations were reordered ego-first.

    ``empirical_local_br_prox_pairing`` canonicalizes both physical roles to
    ``[ego, partner]`` before sampling anchor states.  The counterfactual
    collector therefore must see ego role zero for every selected lane; the
    physical role is restored exactly once by ``environment_step_ego_first``.
    """

    import jax.numpy as jnp

    anchor_count = int(jnp.asarray(observations).shape[0])
    return AnchorWorld(
        environment_state=environment_state,
        observations=observations,
        ego_state=ego_state,
        partner_state=partner_state,
        partner_episode_start=partner_episode_start,
        ego_roles=jnp.zeros((anchor_count,), dtype=jnp.int32),
        done=jnp.zeros((anchor_count,), dtype=jnp.bool_),
        raw_return=jnp.zeros((anchor_count,), dtype=jnp.float32),
    )


def _pairing_seed(
    *,
    root_seed: int,
    ego_run_id: str,
    partner_run_id: str,
    role: int,
) -> int:
    payload = (
        f"{int(root_seed)}\0{ego_run_id}\0{partner_run_id}\0"
        f"role-{int(role)}\0local-br-prox"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _reorder_observations(observations: Any, role: int) -> Any:
    import jax.numpy as jnp

    values = jnp.asarray(observations)
    return values if int(role) == 0 else values[:, ::-1]


def _tree_reset(mask: Any, fresh: Any, current: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def choose(left: Any, right: Any) -> Any:
        values = jnp.asarray(mask, dtype=jnp.bool_)
        expanded = values.reshape(values.shape + (1,) * (jnp.ndim(left) - values.ndim))
        return jnp.where(expanded, left, right)

    return jax.tree_util.tree_map(choose, fresh, current)


def empirical_local_br_prox_pairing(
    *,
    config: Any,
    deployment: Any,
    partner_checkpoint: Path,
    partner_run_id: str,
    partner_mechanism: str,
    partner_numeric_id: int,
    role: int,
    evaluation_seed: int,
) -> list[Mapping[str, Any]]:
    """Return independent-split local BR-Prox rows for one held-out pairing."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    if int(role) not in {0, 1}:
        raise ValueError("role must be zero or one.")
    episode_count = int(config.evaluation.episodes_per_pairing)
    runtime = replace(
        config,
        environment=replace(config.environment, num_envs=episode_count),
    )
    environment = VectorEnvironment.create(runtime)
    pool = FrozenPartnerPool.from_checkpoints((partner_checkpoint,))
    root_seed = _pairing_seed(
        root_seed=evaluation_seed,
        ego_run_id=deployment.ego_run_id,
        partner_run_id=partner_run_id,
        role=role,
    )
    root = jax.random.PRNGKey(root_seed)
    root_keys = jax.random.split(root, episode_count)
    reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(root_keys)
    environment_state, physical_observations = environment.reset_with_keys(reset_keys)
    observations = _reorder_observations(physical_observations, role)
    initial = _AuditState(
        environment_state=environment_state,
        observations=observations,
        ego_state=reset_deployment_state(
            deployment,
            batch_size=episode_count,
            observation_shape=environment.observation_shape,
        ),
        partner_carry=pool.initial_carry(episode_count),
        partner_episode_start=jnp.ones((episode_count,), dtype=jnp.bool_),
    )

    def environment_step_ego_first(
        state: Any, joint_ego_partner: Any, keys: Any
    ) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        physical_joint = (
            joint_ego_partner
            if role == 0
            else joint_ego_partner[:, ::-1]
        )
        next_state, next_physical, rewards, dones, info = environment.step_with_keys(
            state, physical_joint, keys
        )
        reordered = _reorder_observations(next_physical, role)
        terminal = _reorder_observations(info["terminal_observations"], role)
        return next_state, reordered, rewards, dones, {
            **info,
            "terminal_observations": terminal,
        }

    def rollout_step(current: _AuditState, step: Any) -> tuple[_AuditState, Mapping[str, Any]]:
        ego_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 1 + 4 * step)
        )(root_keys)
        partner_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 2 + 4 * step)
        )(root_keys)
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 3 + 4 * step)
        )(root_keys)
        stepped, ego_action, output, unused_log_probability = deployment_action(
            deployment=deployment,
            state=current.ego_state,
            observation=current.observations[:, 0],
            keys=ego_keys,
            force_base=not bool(config.evaluation.enable_adaptation),
        )
        del unused_log_probability
        members = jnp.zeros((episode_count,), dtype=jnp.int32)
        partner_action, partner_carry = pool.step_with_keys(
            members,
            current.observations[:, 1],
            current.partner_carry,
            current.partner_episode_start,
            partner_keys,
        )
        (
            next_environment,
            next_observations,
            rewards,
            dones,
            info,
        ) = environment_step_ego_first(
            current.environment_state,
            jnp.stack((ego_action, partner_action), axis=-1),
            environment_keys,
        )
        terminal = info["terminal_observations"][:, 0]
        mask = dones.reshape(
            dones.shape + (1,) * (next_observations[:, 0].ndim - dones.ndim)
        )
        response_next = jnp.where(mask, terminal, next_observations[:, 0])
        next_ego = update_after_transition(
            deployment=deployment,
            stepped_state=stepped,
            action=ego_action,
            reward=rewards,
            done=dones,
            next_observation=response_next,
        )
        partner_carry = _tree_reset(
            dones, pool.initial_carry(episode_count), partner_carry
        )
        gain = predicted_policy_gain(
            output.action_values,
            output.base_logits,
            output.base_logits + output.residual_logits,
        )
        next_state = _AuditState(
            environment_state=next_environment,
            observations=next_observations,
            ego_state=next_ego,
            partner_carry=partner_carry,
            partner_episode_start=dones,
        )
        return next_state, {
            "environment_state": current.environment_state,
            "observations": current.observations,
            "ego_state": current.ego_state,
            "partner_carry": current.partner_carry,
            "partner_episode_start": current.partner_episode_start,
            "selected_action": ego_action,
            "predicted_gain": gain,
            "support_score": output.support_score,
            "gate": output.gate,
        }

    unused_final, records = jax.lax.scan(
        rollout_step,
        initial,
        jnp.arange(config.environment.episode_steps, dtype=jnp.int32),
    )
    del unused_final
    indexes = stratified_anchor_indexes(
        jax.random.fold_in(root, 90_001),
        time_count=config.environment.episode_steps,
        environment_count=episode_count,
        requested=config.evaluation.br_prox_anchors_per_pairing,
        time_bins=min(8, config.environment.episode_steps),
        regret_bins=4,
        valid_mask=jnp.ones(
            (config.environment.episode_steps, episode_count), dtype=jnp.bool_
        ),
        regret_values=records["predicted_gain"],
        source_values=jnp.full(
            (config.environment.episode_steps, episode_count),
            int(partner_numeric_id),
            dtype=jnp.int32,
        ),
    )
    selected_environment = gather_time_lanes(records["environment_state"], indexes)
    selected_observations = gather_time_lanes(records["observations"], indexes)
    selected_ego = gather_time_lanes(records["ego_state"], indexes)
    selected_partner = gather_time_lanes(records["partner_carry"], indexes)
    selected_partner_start = gather_time_lanes(
        records["partner_episode_start"], indexes
    )
    selected_action = gather_time_lanes(records["selected_action"], indexes)
    selected_gain = gather_time_lanes(records["predicted_gain"], indexes)
    selected_support = gather_time_lanes(records["support_score"], indexes)
    selected_gate = gather_time_lanes(records["gate"], indexes)
    anchor_count = int(indexes.shape[0])
    world = canonical_br_prox_anchor_world(
        environment_state=selected_environment,
        observations=selected_observations,
        ego_state=selected_ego,
        partner_state=selected_partner,
        partner_episode_start=selected_partner_start,
    )

    def ego_policy_step(
        state: Any, observation: Any, unused_gate: Any, keys: Any
    ) -> tuple[Any, Any]:
        del unused_gate
        stepped, unused_action, output, unused_log_probability = deployment_action(
            deployment=deployment,
            state=state,
            observation=observation,
            keys=keys,
            force_base=not bool(config.evaluation.enable_adaptation),
        )
        del unused_action, unused_log_probability
        return stepped, jax.nn.softmax(output.execution_logits, axis=-1)

    def ego_observe(
        stepped: Any,
        unused_observation: Any,
        action: Any,
        reward: Any,
        done: Any,
        next_observation: Any,
    ) -> Any:
        del unused_observation
        return update_after_transition(
            deployment=deployment,
            stepped_state=stepped,
            action=action,
            reward=reward,
            done=done,
            next_observation=next_observation,
        )

    def partner_policy_step(
        state: Any, observation: Any, episode_start: Any, keys: Any
    ) -> tuple[Any, Any, None]:
        action, next_state = pool.step_with_keys(
            jnp.zeros((observation.shape[0],), dtype=jnp.int32),
            observation,
            state,
            episode_start,
            keys,
        )
        return action, next_state, None

    def partner_observe(
        state: Any,
        unused_context: Any,
        unused_observation: Any,
        unused_action: Any,
        unused_reward: Any,
        done: Any,
        unused_next_observation: Any,
    ) -> Any:
        del (
            unused_context,
            unused_observation,
            unused_action,
            unused_reward,
            unused_next_observation,
        )
        return _tree_reset(done, pool.initial_carry(done.shape[0]), state)

    anchor_roots = jax.random.split(
        jax.random.fold_in(root, 90_002), anchor_count
    )
    anchors = collect_counterfactual_anchors(
        anchor_ids=(
            jnp.asarray(int(partner_numeric_id) * 10 + int(role), dtype=jnp.int32)
            * 100_000
            + jnp.arange(anchor_count, dtype=jnp.int32)
        ),
        root_keys=anchor_roots,
        world=world,
        rollout_flat_indexes=indexes,
        policy_states=selected_ego,
        observations=selected_observations[:, 0],
        partner_codes=jnp.full(
            (anchor_count, config.partner_generator.code_dim),
            jnp.nan,
            dtype=jnp.float32,
        ),
        partner_sources=jnp.full(
            (anchor_count,), 2, dtype=jnp.int32
        ),
        partner_run_ids=jnp.full(
            (anchor_count,), int(partner_numeric_id), dtype=jnp.int32
        ),
        functions=AnchorFunctions(
            ego_policy_step=ego_policy_step,
            ego_observe=ego_observe,
            partner_policy_step=partner_policy_step,
            partner_observe=partner_observe,
            environment_step=environment_step_ego_first,
            # BR-Prox uses full 400-step empirical continuations and therefore
            # has no truncated target-policy bootstrap component.
            ego_endpoint_value=lambda state, observation, gate: jnp.zeros(
                (observation.shape[0],), dtype=jnp.float32
            ),
        ),
        action_count=6,
        fit_replicas=config.evaluation.br_prox_fit_replicas,
        evaluation_replicas=config.evaluation.br_prox_evaluation_replicas,
        continuation_horizon=config.evaluation.br_prox_continuation_horizon,
    )
    fit = np.asarray(anchors.fit_returns_by_action, dtype=np.float64)
    evaluation = np.asarray(anchors.evaluation_returns_by_action, dtype=np.float64)
    actual = np.asarray(selected_action, dtype=np.int64)
    oracle = np.argmax(fit, axis=-1)
    rows: list[Mapping[str, Any]] = []
    for anchor_index in range(anchor_count):
        oracle_return = float(evaluation[anchor_index, oracle[anchor_index]])
        actual_return = float(evaluation[anchor_index, actual[anchor_index]])
        worst_return = float(np.min(evaluation[anchor_index]))
        raw_regret = oracle_return - actual_return
        range_normalized = raw_regret / max(oracle_return - worst_return, 1.0)
        flat_index = int(np.asarray(indexes[anchor_index]))
        time_index = flat_index // episode_count
        episode_index = flat_index % episode_count
        rows.append(
            {
                "ego_run_id": deployment.ego_run_id,
                "partner_run_id": partner_run_id,
                "partner_mechanism": partner_mechanism,
                "ego_role": int(role),
                "anchor_index": int(anchor_index),
                "rollout_flat_index": flat_index,
                "episode_index": episode_index,
                "time_index": time_index,
                "selected_action": int(actual[anchor_index]),
                "fit_oracle_action": int(oracle[anchor_index]),
                "fit_action_returns": fit[anchor_index].tolist(),
                "evaluation_action_returns": evaluation[anchor_index].tolist(),
                "selected_evaluation_return": actual_return,
                "oracle_evaluation_return": oracle_return,
                "worst_evaluation_return": worst_return,
                "raw_local_br_regret": raw_regret,
                "br_prox": br_prox(actual_return, oracle_return),
                "range_normalized_br_regret": range_normalized,
                "oracle_action_agreement": bool(
                    actual[anchor_index] == oracle[anchor_index]
                ),
                "predicted_gain": float(np.asarray(selected_gain[anchor_index])),
                "support_score": float(np.asarray(selected_support[anchor_index])),
                "gate": float(np.asarray(selected_gate[anchor_index])),
                "fit_replicas": int(config.evaluation.br_prox_fit_replicas),
                "evaluation_replicas": int(
                    config.evaluation.br_prox_evaluation_replicas
                ),
                "continuation_horizon": int(
                    config.evaluation.br_prox_continuation_horizon
                ),
                "scope": "one_action_deviation_with_frozen_deployed_continuation",
            }
        )
    return rows


__all__ = ["empirical_local_br_prox_pairing"]
