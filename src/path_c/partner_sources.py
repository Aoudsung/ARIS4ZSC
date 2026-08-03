"""Continuous current/EMA/external V6 partner mixture."""

from __future__ import annotations

from typing import Any, NamedTuple

from .counterfactual_anchor import tree_select
from .partner_generator import initial_generator_carry, sample_partner_codes
from .runner import PartnerFunctions


class MixedPartnerParameters(NamedTuple):
    generator_params: Any
    generator_target_params: Any
    source_probabilities: Any


class MixedPartnerState(NamedTuple):
    source: Any
    current_carry: Any
    target_carry: Any
    external_carry: Any
    code: Any
    external_member: Any


class MixedPartnerContext(NamedTuple):
    source: Any
    code: Any
    current_output: Any
    target_output: Any
    reset_keys: Any


class StaticPoolPartnerState(NamedTuple):
    carry: Any
    member: Any


class StaticPoolPartnerContext(NamedTuple):
    member: Any
    reset_keys: Any


def make_static_pool_partner_functions(*, external_pool: Any) -> PartnerFunctions:
    """Default static wide partner pool (METHOD_SPEC §7.3).

    Every episode reset resamples a pool member uniformly; between resets the
    member is fixed.  Pool composition (SP/OP x10 seeds x3 checkpoint stages,
    OP width variants, heuristic family held out for testing) is decided where
    the pool itself is built; the learned partner generator stays disabled on
    this path.
    """

    import jax
    import jax.numpy as jnp

    def initial_state(batch_size: int, key: Any) -> StaticPoolPartnerState:
        member = external_pool.sample_members(key, int(batch_size))
        return StaticPoolPartnerState(
            carry=external_pool.initial_carry(int(batch_size)), member=member
        )

    def step(parameters: Any, state: StaticPoolPartnerState, observations: Any, episode_start: Any, keys: Any):
        del parameters
        action_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(keys)
        reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 911))(keys)
        action, next_carry = external_pool.step_with_keys(
            state.member, observations, state.carry, episode_start, action_keys
        )
        next_state = state._replace(carry=next_carry)
        context = StaticPoolPartnerContext(state.member, reset_keys)
        return action, next_state, context, jnp.zeros_like(action, dtype=jnp.float32)

    def observe(parameters: Any, state: StaticPoolPartnerState, context: StaticPoolPartnerContext, observations: Any, actions: Any, rewards: Any, dones: Any, next_observations: Any):
        del parameters, observations, actions, rewards, next_observations
        done = jnp.asarray(dones, dtype=jnp.bool_)
        count = int(done.shape[0])
        fresh_members = external_pool.sample_members_from_keys(
            context.reset_keys
        ) if hasattr(external_pool, "sample_members_from_keys") else jax.vmap(
            lambda key: external_pool.sample_members(key, 1)[0]
        )(context.reset_keys)
        fresh = StaticPoolPartnerState(
            carry=external_pool.initial_carry(count), member=fresh_members
        )
        return tree_select(done, fresh, state)

    def run_id(parameters: Any, state: StaticPoolPartnerState, context: StaticPoolPartnerContext):
        del parameters, context
        return (10_000 + state.member).astype(jnp.int32)

    def diagnostics(parameters: Any, state: StaticPoolPartnerState, context: StaticPoolPartnerContext):
        del parameters, state
        count = context.member.shape[0]
        return {
            "source": jnp.full((count,), 2, dtype=jnp.int32),
            "member": context.member,
            # Static pools have no continuous generator code; the member index
            # fills the rollout partner_code slot (run_id carries identity).
            "code": jnp.asarray(context.member, dtype=jnp.float32)[:, None],
            "generator_logits": jnp.zeros((count, 6), dtype=jnp.float32),
            "generator_value": jnp.zeros((count,), dtype=jnp.float32),
        }

    return PartnerFunctions(initial_state, step, observe, run_id, diagnostics)


def soft_generator_mixture(
    *,
    progress: Any,
    generator_cvar_ema: Any,
    reference_cvar_ema: Any,
    maximum_probability: float = 0.75,
    ramp_fraction: float = 0.30,
    competence_temperature: float = 20.0,
) -> Any:
    import jax
    import jax.numpy as jnp

    rho = float(maximum_probability) * jnp.clip(
        jnp.asarray(progress, dtype=jnp.float32) / float(ramp_fraction), 0.0, 1.0
    )
    competence = jax.nn.sigmoid(
        (jnp.asarray(generator_cvar_ema) - jnp.asarray(reference_cvar_ema))
        / float(competence_temperature)
    )
    generator_probability = rho * competence
    return jnp.asarray(
        (0.5 * generator_probability, 0.5 * generator_probability, 1.0 - generator_probability),
        dtype=jnp.float32,
    )


def make_mixed_partner_functions(
    *,
    generator: Any,
    generator_hidden_dim: int,
    generator_code_dim: int,
    external_pool: Any,
) -> PartnerFunctions:
    import jax
    import jax.numpy as jnp

    if external_pool is None:
        raise ValueError("V6 always retains capable external partner support.")

    def sample_metadata(batch_size: int, key: Any, probabilities: Any):
        source_key, code_key, external_key = jax.random.split(key, 3)
        probs = jnp.asarray(probabilities, dtype=jnp.float32)
        source = jax.random.categorical(
            source_key, jnp.log(jnp.maximum(probs, 1.0e-12)), shape=(int(batch_size),)
        )
        code = sample_partner_codes(code_key, batch_size=batch_size, code_dim=generator_code_dim)
        external_member = external_pool.sample_members(external_key, batch_size)
        return source, code, external_member

    def initial_state(batch_size: int, key: Any) -> MixedPartnerState:
        # The first rollout begins with external support; later episode resets
        # use the dynamic probabilities from MixedPartnerParameters.
        source, code, member = sample_metadata(
            int(batch_size), key, jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float32)
        )
        return MixedPartnerState(
            source,
            initial_generator_carry(batch_size, generator_hidden_dim),
            initial_generator_carry(batch_size, generator_hidden_dim),
            external_pool.initial_carry(batch_size),
            code,
            member,
        )

    def step(parameters: MixedPartnerParameters, state: MixedPartnerState, observations: Any, episode_start: Any, keys: Any):
        action_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(keys)
        reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 1))(keys)
        next_current, current_output = generator.apply(
            {"params": parameters.generator_params}, state.current_carry,
            observations, state.code, episode_start, action_keys, method=generator.step,
        )
        next_target, target_output = generator.apply(
            {"params": parameters.generator_target_params}, state.target_carry,
            observations, state.code, episode_start, action_keys, method=generator.step,
        )
        external_action, next_external = external_pool.step_with_keys(
            state.external_member, observations, state.external_carry, episode_start, action_keys
        )
        action = jnp.where(
            state.source == 0, current_output.action,
            jnp.where(state.source == 1, target_output.action, external_action),
        )
        log_probability = jnp.where(
            state.source == 0, current_output.log_probability,
            jnp.where(state.source == 1, target_output.log_probability, jnp.zeros_like(current_output.log_probability)),
        )
        next_state = state._replace(
            current_carry=next_current, target_carry=next_target, external_carry=next_external
        )
        return action, next_state, MixedPartnerContext(
            state.source, state.code, current_output, target_output, reset_keys
        ), log_probability

    def observe(parameters: MixedPartnerParameters, state: MixedPartnerState, context: MixedPartnerContext, observations: Any, actions: Any, rewards: Any, dones: Any, next_observations: Any):
        del observations, actions, rewards, next_observations
        done = jnp.asarray(dones, dtype=jnp.bool_)
        count = int(done.shape[0])

        def reset(key: Any):
            source, code, member = sample_metadata(1, key, parameters.source_probabilities)
            return source[0], code[0], member[0]

        source, code, member = jax.vmap(reset)(context.reset_keys)
        fresh = MixedPartnerState(
            source,
            initial_generator_carry(count, generator_hidden_dim),
            initial_generator_carry(count, generator_hidden_dim),
            external_pool.initial_carry(count),
            code,
            member,
        )
        return tree_select(done, fresh, state)

    def run_id(parameters: MixedPartnerParameters, state: MixedPartnerState, context: MixedPartnerContext):
        del parameters, context
        return jnp.where(
            state.source == 0, -1,
            jnp.where(state.source == 1, -2, 10_000 + state.external_member),
        ).astype(jnp.int32)

    def diagnostics(parameters: MixedPartnerParameters, state: MixedPartnerState, context: MixedPartnerContext):
        del parameters, state
        selected_logits = jnp.where(
            (context.source == 0)[..., None], context.current_output.logits,
            context.target_output.logits,
        )
        selected_value = jnp.where(
            context.source == 0, context.current_output.value, context.target_output.value
        )
        return {
            "source": context.source,
            "code": context.code,
            "generator_logits": selected_logits,
            "generator_value": selected_value,
        }

    return PartnerFunctions(initial_state, step, observe, run_id, diagnostics)


__all__ = [
    "MixedPartnerContext",
    "MixedPartnerParameters",
    "MixedPartnerState",
    "StaticPoolPartnerContext",
    "StaticPoolPartnerState",
    "make_mixed_partner_functions",
    "make_static_pool_partner_functions",
    "soft_generator_mixture",
]
