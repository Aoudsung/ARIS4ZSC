"""Action-conditioned protocol response decoder (METHOD_SPEC §2.2).

The decoder implements the per-component likelihood
``p(y_{t+1} | z_t=k, a_t^ego, sg[frame_t])`` of the mixture model

    p(y | z, H, a) = sum_k pi_{t,k} p(y | z=k, a, sg[frame_t]).

* Kinematic components 1-4 (visibility, relative position, direction,
  inventory) read the stop-gradient frame feature ``sg[frame_t]`` -- predicting
  kinematics requires a coordinate frame -- plus ``(m_k, u, a^ego)``.
* Component 5 (observable interaction event) reads only ``(m_k, u, a^ego)``.
  It cannot bypass the isolated task trunk or let task-state consequences
  dominate the response-regime likelihood.

All five heads emit a trailing ``K`` component axis so that
``response_targets.mixture_response_loss`` can form the proper mixture NLL
(METHOD_SPEC §2.3).
"""

from __future__ import annotations

from typing import Any

_DECODER: Any | None = None


def add_component_axis(value: Any, component_count: int) -> Any:
    """Insert and explicitly broadcast a protocol-component axis.

    ``value`` has shape ``[..., D]`` and the result has shape
    ``[..., K, D]``.  The singleton insertion is required because JAX cannot
    infer that a newly requested axis belongs immediately before the feature
    dimension.
    """

    import jax.numpy as jnp

    array = jnp.asarray(value)
    if array.ndim < 1:
        raise ValueError("A component-conditioned feature needs a feature axis.")
    return jnp.broadcast_to(
        array[..., None, :],
        array.shape[:-1] + (int(component_count), array.shape[-1]),
    )


def response_decoder_class() -> Any:
    global _DECODER
    if _DECODER is not None:
        return _DECODER

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros
    from .response_targets import (
        PARTNER_DIRECTION_CLASSES,
        PARTNER_INVENTORY_FACTOR_CLASSES,
        PARTNER_POSITION_CLASSES,
    )

    class ResponseDecoder(nn.Module):
        action_count: int
        hidden_dim: int
        action_embedding_dim: int
        inventory_factor_count: int

        @nn.compact
        def __call__(
            self,
            frame_features: Any,
            component_embeddings: Any,
            capability: Any,
            actions: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            """Predict per-component response logits.

            Args:
                frame_features: ``sg[frame_t]``; the caller applies
                    stop-gradient.  Any spatial axes are flattened.
                component_embeddings: matrix ``(K, D)`` of protocol component
                    embeddings ``m_k``.
                capability: partner capability embedding ``u``.
                actions: ego actions ``a_t^ego`` (integer codes).

            Returns:
                Five logit tensors each carrying a trailing component axis K.
            """

            frame = jnp.asarray(frame_features, dtype=jnp.float32)
            components = jnp.asarray(component_embeddings, dtype=jnp.float32)
            u = jnp.asarray(capability, dtype=jnp.float32)
            action = jnp.asarray(actions, dtype=jnp.int32)
            if components.ndim != 2:
                raise ValueError("Component embeddings must be a (K, D) matrix.")
            component_count, component_dim = components.shape
            lead = action.shape
            frame = frame.reshape(lead + (-1,))
            if u.shape[:-1] != lead:
                raise ValueError("Response decoder batch axes differ.")

            action_embedding = nn.Embed(
                num_embeddings=self.action_count,
                features=self.action_embedding_dim,
                embedding_init=nn.initializers.normal(0.02),
                name="action_embedding",
            )(action)
            # Broadcast the component embedding to (..., K, D).
            broadcast_components = jnp.broadcast_to(
                components.reshape((1,) * len(lead) + components.shape),
                lead + (component_count, component_dim),
            )
            broadcast_u = add_component_axis(u, component_count)
            broadcast_action = add_component_axis(action_embedding, component_count)
            broadcast_frame = add_component_axis(frame, component_count)

            # Kinematic trunk: [sg(frame), m_k, u, a] (components 1-4).
            kinematic_input = jnp.concatenate(
                (broadcast_frame, broadcast_components, broadcast_u, broadcast_action),
                axis=-1,
            )
            kinematic_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="kinematic_hidden_0",
                )(kinematic_input)
            )
            kinematic_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="kinematic_hidden_1",
                )(kinematic_hidden)
            )
            visibility_logit = nn.Dense(
                1,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_visibility_logit",
            )(kinematic_hidden)[..., 0]
            relative_position_logits = nn.Dense(
                PARTNER_POSITION_CLASSES,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_relative_position_logits",
            )(kinematic_hidden)
            direction_logits = nn.Dense(
                PARTNER_DIRECTION_CLASSES,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_direction_logits",
            )(kinematic_hidden)
            inventory_logits = nn.Dense(
                self.inventory_factor_count * PARTNER_INVENTORY_FACTOR_CLASSES,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_inventory_logits",
            )(kinematic_hidden).reshape(
                lead + (component_count, self.inventory_factor_count,
                        PARTNER_INVENTORY_FACTOR_CLASSES)
            )

            # Event head: response-regime, capability, and ego action only.
            event_input = jnp.concatenate(
                (
                    broadcast_components,
                    broadcast_u,
                    broadcast_action,
                ),
                axis=-1,
            )
            event_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="event_hidden_0",
                )(event_input)
            )
            event_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="event_hidden_1",
                )(event_hidden)
            )
            interaction_change_logit = nn.Dense(
                1,
                kernel_init=orthogonal(0.01),
                bias_init=zeros,
                name="partner_interaction_change_logit",
            )(event_hidden)[..., 0]
            return (
                visibility_logit,
                relative_position_logits,
                direction_logits,
                inventory_logits,
                interaction_change_logit,
            )

    _DECODER = ResponseDecoder
    return ResponseDecoder


def bernoulli_logit_loss(target: Any, logit: Any) -> Any:
    import jax.numpy as jnp

    label = jnp.asarray(target, dtype=jnp.float32)
    return jnp.maximum(logit, 0.0) - logit * label + jnp.log1p(
        jnp.exp(-jnp.abs(logit))
    )


__all__ = [
    "add_component_axis",
    "bernoulli_logit_loss",
    "response_decoder_class",
]
