"""Mixed continuous-generator, snapshot, and frozen-external partner support."""

from __future__ import annotations

from typing import Any, NamedTuple

from .counterfactual_anchor import tree_select
from .partner_generator import initial_generator_carry, sample_partner_codes
from .runner import PartnerFunctions


class MixedPartnerParameters(NamedTuple):
    generator_params: Any
    snapshot_params: Any


class MixedPartnerState(NamedTuple):
    source: Any
    generator_carry: Any
    snapshot_carry: Any
    external_carry: Any
    code: Any
    snapshot_member: Any
    external_member: Any


class MixedPartnerContext(NamedTuple):
    source: Any
    code: Any
    generator_output: Any
    snapshot_output: Any
    reset_keys: Any



def _select_tree_by_index(stacked: Any, indexes: Any) -> Any:
    import jax

    return jax.tree_util.tree_map(lambda values: values[indexes], stacked)


def make_mixed_partner_functions(
    *,
    generator: Any,
    generator_hidden_dim: int,
    generator_code_dim: int,
    snapshot_count: int,
    external_pool: Any | None,
    current_probability: float,
    snapshot_probability: float,
    frozen_external_probability: float,
) -> PartnerFunctions:
    """Build one immutable partner runtime used by training rollouts."""

    import jax
    import jax.numpy as jnp

    probabilities = jnp.asarray(
        (current_probability, snapshot_probability, frozen_external_probability),
        dtype=jnp.float32,
    )
    if abs(float(jnp.sum(probabilities)) - 1.0) > 1.0e-6:
        raise ValueError("Mixed partner probabilities must sum to one.")
    if snapshot_probability > 0.0 and snapshot_count <= 0:
        raise ValueError("Snapshot probability is positive but no snapshots are loaded.")
    if frozen_external_probability > 0.0 and external_pool is None:
        raise ValueError("External probability is positive but no external pool is loaded.")

    external_count = 0 if external_pool is None else int(external_pool.member_count)

    def sample_metadata(batch_size: int, key: Any) -> tuple[Any, Any, Any, Any]:
        source_key, code_key, snapshot_key, external_key = jax.random.split(key, 4)
        source = jax.random.categorical(
            source_key,
            jnp.log(jnp.maximum(probabilities, 1.0e-12)),
            shape=(int(batch_size),),
        )
        code = sample_partner_codes(
            code_key, batch_size=batch_size, code_dim=generator_code_dim
        )
        snapshot_member = (
            jax.random.randint(snapshot_key, (batch_size,), 0, snapshot_count)
            if snapshot_count > 0
            else jnp.zeros((batch_size,), dtype=jnp.int32)
        )
        external_member = (
            external_pool.sample_members(external_key, batch_size)
            if external_count > 0
            else jnp.zeros((batch_size,), dtype=jnp.int32)
        )
        return source, code, snapshot_member, external_member

    def initial_state(batch_size: int, key: Any) -> MixedPartnerState:
        source, code, snapshot_member, external_member = sample_metadata(
            int(batch_size), key
        )
        external_carry = (
            external_pool.initial_carry(batch_size)
            if external_pool is not None
            else jnp.zeros((batch_size, 1), dtype=jnp.float32)
        )
        return MixedPartnerState(
            source=source,
            generator_carry=initial_generator_carry(
                batch_size, generator_hidden_dim
            ),
            snapshot_carry=initial_generator_carry(
                batch_size, generator_hidden_dim
            ),
            external_carry=external_carry,
            code=code,
            snapshot_member=snapshot_member,
            external_member=external_member,
        )

    def snapshot_step(
        stacked_params: Any,
        member_indexes: Any,
        carry: Any,
        observations: Any,
        code: Any,
        starts: Any,
        keys: Any,
    ) -> tuple[Any, Any]:
        if snapshot_count <= 0:
            output_carry, output = generator.apply(
                {"params": stacked_params},
                carry,
                observations,
                code,
                starts,
                keys,
                method=generator.step,
            )
            return output_carry, output

        def one(
            member: Any,
            member_carry: Any,
            observation: Any,
            member_code: Any,
            start: Any,
            key: Any,
        ) -> tuple[Any, Any]:
            params = jax.tree_util.tree_map(lambda values: values[member], stacked_params)
            next_carry, output = generator.apply(
                {"params": params},
                jax.tree_util.tree_map(lambda value: value[None, ...], member_carry),
                observation[None, ...],
                member_code[None, ...],
                start[None, ...],
                key[None, ...],
                method=generator.step,
            )
            return (
                jax.tree_util.tree_map(lambda value: value[0], next_carry),
                jax.tree_util.tree_map(lambda value: value[0], output),
            )

        return jax.vmap(one)(
            member_indexes, carry, observations, code, starts, keys
        )

    def step(
        parameters: MixedPartnerParameters,
        state: MixedPartnerState,
        observations: Any,
        episode_start: Any,
        keys: Any,
    ) -> tuple[Any, MixedPartnerState, MixedPartnerContext, Any]:
        action_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(keys)
        reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 1))(keys)
        next_generator_carry, generator_output = generator.apply(
            {"params": parameters.generator_params},
            state.generator_carry,
            observations,
            state.code,
            episode_start,
            action_keys,
            method=generator.step,
        )
        next_snapshot_carry, snapshot_output = snapshot_step(
            parameters.snapshot_params,
            state.snapshot_member,
            state.snapshot_carry,
            observations,
            state.code,
            episode_start,
            action_keys,
        )
        if external_pool is None:
            external_action = jnp.zeros_like(generator_output.action)
            next_external_carry = state.external_carry
        else:
            external_action, next_external_carry = external_pool.step_with_keys(
                state.external_member,
                observations,
                state.external_carry,
                episode_start,
                action_keys,
            )
        action = jnp.where(
            state.source == 0,
            generator_output.action,
            jnp.where(
                state.source == 1, snapshot_output.action, external_action
            ),
        )
        log_probability = jnp.where(
            state.source == 0,
            generator_output.log_probability,
            jnp.where(
                state.source == 1,
                snapshot_output.log_probability,
                jnp.zeros_like(generator_output.log_probability),
            ),
        )
        next_state = state._replace(
            generator_carry=next_generator_carry,
            snapshot_carry=next_snapshot_carry,
            external_carry=next_external_carry,
        )
        return action, next_state, MixedPartnerContext(
            source=state.source,
            code=state.code,
            generator_output=generator_output,
            snapshot_output=snapshot_output,
            reset_keys=reset_keys,
        ), log_probability

    def observe(
        parameters: MixedPartnerParameters,
        state: MixedPartnerState,
        context: MixedPartnerContext,
        observations: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        next_observations: Any,
    ) -> MixedPartnerState:
        del parameters, observations, actions, rewards, next_observations
        done = jnp.asarray(dones, dtype=jnp.bool_)
        count = int(done.shape[0])

        def one_reset(key: Any) -> tuple[Any, Any, Any, Any]:
            return sample_metadata(1, key)

        reset = jax.vmap(one_reset)(context.reset_keys)
        reset_source = reset[0][:, 0]
        reset_code = reset[1][:, 0]
        reset_snapshot = reset[2][:, 0]
        reset_external = reset[3][:, 0]
        fresh = MixedPartnerState(
            source=reset_source,
            generator_carry=initial_generator_carry(count, generator_hidden_dim),
            snapshot_carry=initial_generator_carry(count, generator_hidden_dim),
            external_carry=(
                external_pool.initial_carry(count)
                if external_pool is not None
                else jnp.zeros((count, 1), dtype=jnp.float32)
            ),
            code=reset_code,
            snapshot_member=reset_snapshot,
            external_member=reset_external,
        )
        return tree_select(done, fresh, state)

    def run_id(
        parameters: MixedPartnerParameters,
        state: MixedPartnerState,
        context: MixedPartnerContext,
    ) -> Any:
        del parameters, context
        return jnp.where(
            state.source == 0,
            -1,
            jnp.where(
                state.source == 1,
                1_000 + state.snapshot_member,
                10_000 + state.external_member,
            ),
        ).astype(jnp.int32)

    def diagnostics(
        parameters: MixedPartnerParameters,
        state: MixedPartnerState,
        context: MixedPartnerContext,
    ) -> Any:
        del parameters, state
        return {
            "source": context.source,
            "code": context.code,
            "generator_logits": context.generator_output.logits,
            "generator_value": context.generator_output.value,
        }

    return PartnerFunctions(
        initial_state=initial_state,
        step=step,
        observe=observe,
        run_id=run_id,
        diagnostics=diagnostics,
    )


__all__ = [
    "MixedPartnerContext",
    "MixedPartnerParameters",
    "MixedPartnerState",
    "make_mixed_partner_functions",
]
