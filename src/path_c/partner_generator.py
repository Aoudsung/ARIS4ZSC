"""Single continuous code-conditioned partner generator."""

from __future__ import annotations

from typing import Any, Mapping

from .types import PartnerGeneratorOutput, PartnerGeneratorState

_GENERATOR_CLASS: Any | None = None


def partner_generator_class() -> Any:
    global _GENERATOR_CLASS
    if _GENERATOR_CLASS is not None:
        return _GENERATOR_CLASS

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class ContinuousPartnerGenerator(nn.Module):
        observation_shape: tuple[int, ...]
        action_count: int
        code_dim: int
        hidden_dim: int
        modulation_rank: int

        @nn.compact
        def step(
            self,
            carry: Any,
            observation: Any,
            code: Any,
            episode_start: Any,
            key: Any,
        ) -> tuple[Any, PartnerGeneratorOutput]:
            obs = jnp.asarray(observation, dtype=jnp.float32)
            start = jnp.asarray(episode_start, dtype=jnp.bool_)
            flat = obs.reshape(start.shape + (-1,))
            context = jnp.asarray(code, dtype=jnp.float32)
            if context.shape[:-1] != start.shape:
                context = jnp.broadcast_to(
                    context, start.shape + (context.shape[-1],)
                )
            base = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="observation_hidden",
                )(flat)
            )
            task_basis = nn.Dense(
                self.modulation_rank,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="task_basis",
            )(base)
            code_gain = nn.tanh(
                nn.Dense(
                    self.modulation_rank,
                    kernel_init=orthogonal(0.5),
                    bias_init=zeros,
                    name="code_gain",
                )(context)
            )
            modulation = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="modulation_projection",
            )(task_basis * code_gain)
            recurrent_input = nn.LayerNorm(name="generator_layer_norm")(
                base + modulation
            )
            carry = jnp.where(start[..., None], jnp.zeros_like(carry), carry)
            next_carry, hidden = nn.GRUCell(
                features=self.hidden_dim,
                name="generator_gru",
            )(carry, recurrent_input)
            logits = nn.Dense(
                self.action_count,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="generator_logits",
            )(hidden)
            value = nn.Dense(
                1,
                kernel_init=orthogonal(1.0),
                bias_init=zeros,
                name="generator_value",
            )(hidden)[..., 0]
            key_array = jnp.asarray(key)
            if key_array.shape[:-1] == logits.shape[:-1] and key_array.shape[-1] == 2:
                flat_keys = key_array.reshape((-1, 2))
                flat_logits = logits.reshape((-1, logits.shape[-1]))
                action = jax.vmap(
                    lambda lane_key, lane_logits: jax.random.categorical(
                        lane_key, lane_logits
                    )
                )(flat_keys, flat_logits).reshape(logits.shape[:-1])
            else:
                action = jax.random.categorical(key_array, logits, axis=-1)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            selected_log_prob = jnp.take_along_axis(
                log_probs, action[..., None], axis=-1
            )[..., 0]
            probabilities = jnp.exp(log_probs)
            entropy = -jnp.sum(probabilities * log_probs, axis=-1)
            return next_carry, PartnerGeneratorOutput(
                logits=logits,
                value=value,
                action=action,
                log_probability=selected_log_prob,
                entropy=entropy,
            )

        def sequence(
            self,
            initial_carry: Any,
            observations: Any,
            code: Any,
            episode_starts: Any,
            keys: Any,
        ) -> tuple[Any, PartnerGeneratorOutput]:
            import jax

            code_array = jnp.asarray(code, dtype=jnp.float32)
            observation_array = jnp.asarray(observations)
            if (
                code_array.ndim == 3
                and code_array.shape[:2] == observation_array.shape[:2]
            ):
                # Exact replay surface: one code is recorded at every step.
                # Complete-episode validation requires it to be constant, but
                # accepting the time axis prevents the old ``codes[0]`` bug
                # from ever being reintroduced at this interface.
                sequence_codes = code_array
            else:
                sequence_codes = jnp.broadcast_to(
                    code_array,
                    (observation_array.shape[0],) + code_array.shape,
                )

            def one(carry: Any, values: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
                observation, current_code, start, key = values
                return self.step(carry, observation, current_code, start, key)

            return jax.lax.scan(
                one,
                initial_carry,
                (observations, sequence_codes, episode_starts, keys),
            )

        def __call__(
            self,
            initial_carry: Any,
            observations: Any,
            code: Any,
            episode_starts: Any,
            keys: Any,
        ) -> tuple[Any, PartnerGeneratorOutput]:
            return self.sequence(
                initial_carry, observations, code, episode_starts, keys
            )

    _GENERATOR_CLASS = ContinuousPartnerGenerator
    return ContinuousPartnerGenerator


def build_partner_generator(
    *,
    observation_shape: tuple[int, ...],
    action_count: int,
    code_dim: int,
    hidden_dim: int,
    modulation_rank: int,
) -> Any:
    return partner_generator_class()(
        observation_shape=tuple(observation_shape),
        action_count=int(action_count),
        code_dim=int(code_dim),
        hidden_dim=int(hidden_dim),
        modulation_rank=int(modulation_rank),
    )


def initial_generator_carry(batch_size: int, hidden_dim: int) -> Any:
    import jax.numpy as jnp

    return jnp.zeros((int(batch_size), int(hidden_dim)), dtype=jnp.float32)


def initialize_generator_parameters(
    generator: Any,
    *,
    key: Any,
    observation_shape: tuple[int, ...],
    code_dim: int,
    batch_size: int,
    hidden_dim: int,
) -> Mapping[str, Any]:
    import jax
    import jax.numpy as jnp

    parameter_key, action_key = jax.random.split(key)
    variables = generator.init(
        parameter_key,
        initial_generator_carry(batch_size, hidden_dim),
        jnp.zeros((batch_size,) + tuple(observation_shape), dtype=jnp.float32),
        jnp.zeros((batch_size, code_dim), dtype=jnp.float32),
        jnp.ones((batch_size,), dtype=jnp.bool_),
        jax.random.split(action_key, batch_size),
        method=generator.step,
    )
    return variables["params"]


def sample_partner_codes(
    key: Any,
    *,
    batch_size: int,
    code_dim: int,
    alpha: Any | None = None,
    anchors: Any | None = None,
) -> Any:
    """Sample codes inside the fitted Dirichlet simplex support (§7.1).

    Barycentric weights are drawn from Dirichlet(alpha) over the four
    tetrahedral anchors and mapped to codes via ``code = V^T w``; every
    sample is therefore a convex combination of the anchors and lies within
    the support region (alpha is fitted by method of moments from real
    collected codes upstream, defaulting to the uniform distribution).
    """

    import jax
    import jax.numpy as jnp

    from .generator_training import codes_from_barycentric

    if int(code_dim) != 3:
        raise ValueError(
            "METHOD_SPEC §7.1 fixes the generator code space to the "
            "3-dimensional tetrahedral simplex; code_dim must be 3."
        )
    concentration = (
        jnp.asarray(alpha, dtype=jnp.float32)
        if alpha is not None
        else jnp.ones((4,), dtype=jnp.float32)
    )
    weights = jax.random.dirichlet(key, concentration, shape=(int(batch_size),))
    return codes_from_barycentric(weights, anchors)


__all__ = [
    "build_partner_generator",
    "initial_generator_carry",
    "initialize_generator_parameters",
    "partner_generator_class",
    "sample_partner_codes",
]
