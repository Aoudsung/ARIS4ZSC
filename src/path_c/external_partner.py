"""Adapters exposing frozen official policies through DELTA-ZSC PartnerFunctions."""

from __future__ import annotations

from typing import Any, NamedTuple

from .counterfactual_anchor import tree_select
from .runner import PartnerFunctions


class ExternalPartnerState(NamedTuple):
    carry: Any
    member: Any


class ExternalPartnerContext(NamedTuple):
    member: Any
    reset_keys: Any


def make_external_partner_functions(
    *,
    pool: Any,
    member_indexes: Any,
    latent_dim: int,
    code_dim: int,
    run_ids: Any | None = None,
) -> PartnerFunctions:
    """Create a fixed-member vectorized external partner runtime.

    ``member_indexes`` and ``run_ids`` may be scalars or arrays broadcastable to
    the rollout batch.  No partner metadata enters the ego model; run IDs are
    emitted only for run-blocked supervision and statistics.
    """

    import jax
    import jax.numpy as jnp

    members_template = jnp.asarray(member_indexes, dtype=jnp.int32)
    run_template = (
        members_template
        if run_ids is None
        else jnp.asarray(run_ids, dtype=jnp.int32)
    )


    def initial_state(batch_size: int, key: Any) -> ExternalPartnerState:
        del key
        members = jnp.broadcast_to(members_template, (int(batch_size),))
        return ExternalPartnerState(
            carry=pool.initial_carry(int(batch_size)),
            member=members,
        )

    def step(
        parameters: Any,
        state: ExternalPartnerState,
        observations: Any,
        episode_start: Any,
        keys: Any,
    ) -> tuple[Any, ExternalPartnerState, ExternalPartnerContext, Any]:
        del parameters
        action, next_carry = pool.step_with_keys(
            state.member,
            observations,
            state.carry,
            episode_start,
            keys,
        )
        reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 911))(keys)
        return (
            action,
            ExternalPartnerState(next_carry, state.member),
            ExternalPartnerContext(state.member, reset_keys),
            jnp.zeros_like(action, dtype=jnp.float32),
        )

    def observe(
        parameters: Any,
        state: ExternalPartnerState,
        context: ExternalPartnerContext,
        observations: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        next_observations: Any,
    ) -> ExternalPartnerState:
        del parameters, context, observations, actions, rewards, next_observations
        done = jnp.asarray(dones, dtype=jnp.bool_)
        fresh = ExternalPartnerState(
            carry=pool.initial_carry(int(done.shape[0])),
            member=state.member,
        )
        return tree_select(done, fresh, state)

    def pre_teacher_latent(parameters: Any, state: Any, task_features: Any) -> Any:
        del parameters, state
        return jnp.full(
            task_features.shape[:-1] + (int(latent_dim),),
            jnp.nan,
            dtype=jnp.float32,
        )

    def teacher_latent(parameters: Any, state: Any, context: Any, task_features: Any) -> Any:
        del parameters, state, context
        return jnp.full(
            task_features.shape[:-1] + (int(latent_dim),),
            jnp.nan,
            dtype=jnp.float32,
        )

    def run_id(parameters: Any, state: ExternalPartnerState, context: Any) -> Any:
        del parameters, context
        values = jnp.broadcast_to(run_template, state.member.shape)
        return values.astype(jnp.int32)

    def diagnostics(parameters: Any, state: ExternalPartnerState, context: Any) -> Any:
        del parameters, context
        count = state.member.shape[0]
        return {
            "source": jnp.full((count,), 2, dtype=jnp.int32),
            "code": jnp.full(
                (count, int(code_dim)), jnp.nan, dtype=jnp.float32
            ),
            "generator_logits": jnp.zeros((count, 6), dtype=jnp.float32),
            "generator_value": jnp.zeros((count,), dtype=jnp.float32),
        }

    return PartnerFunctions(
        initial_state=initial_state,
        step=step,
        observe=observe,
        pre_teacher_latent=pre_teacher_latent,
        teacher_latent=teacher_latent,
        run_id=run_id,
        diagnostics=diagnostics,
    )


__all__ = [
    "ExternalPartnerContext",
    "ExternalPartnerState",
    "make_external_partner_functions",
]
