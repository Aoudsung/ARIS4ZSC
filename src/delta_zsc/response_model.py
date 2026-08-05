"""Nested action-conditioned response emission.

A trained base response model is shared by every coordination mode; each mode
adds a residual.  Removing the residual is therefore an in-distribution nested
control, unlike replacing learned component embeddings by an unseen zero
vector.  The stopped spatial encoder retains coordinates and can represent all
25 relative positions in the Official 5x5 view.
"""

from __future__ import annotations

from typing import Any

from .types import ResponseLogits


_RESPONSE_CLASS: Any | None = None


def response_emission_class() -> Any:
    global _RESPONSE_CLASS
    if _RESPONSE_CLASS is not None:
        return _RESPONSE_CLASS

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    from src.path_c.response_targets import (
        PARTNER_DIRECTION_CLASSES,
        PARTNER_INVENTORY_FACTOR_CLASSES,
        PARTNER_POSITION_CLASSES,
    )

    class ResponseEmission(nn.Module):
        action_count: int
        component_count: int
        component_dim: int
        hidden_dim: int
        action_embedding_dim: int
        inventory_factor_count: int

        @nn.compact
        def __call__(
            self,
            previous_frame: Any,
            behavior_features: Any,
            previous_action: Any,
            *,
            include_component_residual: bool = True,
        ) -> ResponseLogits:
            frame = jnp.asarray(previous_frame, dtype=jnp.float32)
            behavior = jnp.asarray(behavior_features, dtype=jnp.float32)
            action = jnp.asarray(previous_action, dtype=jnp.int32)
            lead = action.shape
            if frame.shape[:-3] != lead or behavior.shape[:-1] != lead:
                raise ValueError("Response emission batch axes differ.")
            if frame.shape[-3:-1] != (5, 5):
                raise ValueError("Unified DELTA response model expects the Official 5x5 view.")

            # Coordinate-aware stopped physical encoder.  Flattening the final
            # spatial map preserves the location of one-hot partner planes.
            spatial = frame
            for index, features in enumerate((16, 32)):
                spatial = nn.relu(
                    nn.Conv(
                        features=features,
                        kernel_size=(3, 3),
                        padding="SAME",
                        kernel_init=orthogonal(jnp.sqrt(2.0)),
                        bias_init=zeros,
                        name=f"physical_conv_{index}",
                    )(spatial)
                )
            spatial = spatial.reshape(lead + (-1,))
            spatial = nn.LayerNorm(name="physical_layer_norm")(
                nn.relu(
                    nn.Dense(
                        self.hidden_dim,
                        kernel_init=orthogonal(jnp.sqrt(2.0)),
                        bias_init=zeros,
                        name="physical_dense",
                    )(spatial)
                )
            )
            action_embedding = nn.Embed(
                num_embeddings=self.action_count,
                features=self.action_embedding_dim,
                embedding_init=nn.initializers.normal(0.02),
                name="action_embedding",
            )(action)
            common = jnp.concatenate((spatial, behavior, action_embedding), axis=-1)
            base_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="base_hidden_0",
                )(common)
            )
            base_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="base_hidden_1",
                )(base_hidden)
            )

            def base_head(name: str, size: int) -> Any:
                return nn.Dense(
                    size,
                    kernel_init=orthogonal(0.01),
                    bias_init=zeros,
                    name=f"base_{name}",
                )(base_hidden)

            base_visibility = base_head("visibility", 1)[..., 0]
            base_position = base_head("position", PARTNER_POSITION_CLASSES)
            base_direction = base_head("direction", PARTNER_DIRECTION_CLASSES)
            base_inventory = base_head(
                "inventory",
                self.inventory_factor_count * PARTNER_INVENTORY_FACTOR_CLASSES,
            ).reshape(
                lead
                + (
                    self.inventory_factor_count,
                    PARTNER_INVENTORY_FACTOR_CLASSES,
                )
            )
            base_event = base_head("inventory_change", 1)[..., 0]

            components = self.param(
                "component_embeddings",
                nn.initializers.normal(0.02),
                (self.component_count, self.component_dim),
            )
            components = components / jnp.maximum(
                jnp.linalg.norm(components, axis=-1, keepdims=True), 1.0e-6
            )
            common_k = jnp.broadcast_to(
                common[..., None, :], lead + (self.component_count, common.shape[-1])
            )
            components_k = jnp.broadcast_to(
                components.reshape((1,) * len(lead) + components.shape),
                lead + components.shape,
            )
            residual_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="residual_hidden_0",
                )(jnp.concatenate((common_k, components_k), axis=-1))
            )
            residual_hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="residual_hidden_1",
                )(residual_hidden)
            )

            def residual_head(name: str, size: int) -> Any:
                residual = nn.Dense(
                    size,
                    kernel_init=zeros,
                    bias_init=zeros,
                    name=f"residual_{name}",
                )(residual_hidden)
                return residual if include_component_residual else jnp.zeros_like(residual)

            visibility = base_visibility[..., None] + residual_head("visibility", 1)[..., 0]
            position = base_position[..., None, :] + residual_head(
                "position", PARTNER_POSITION_CLASSES
            )
            direction = base_direction[..., None, :] + residual_head(
                "direction", PARTNER_DIRECTION_CLASSES
            )
            inventory = base_inventory[..., None, :, :] + residual_head(
                "inventory",
                self.inventory_factor_count * PARTNER_INVENTORY_FACTOR_CLASSES,
            ).reshape(
                lead
                + (
                    self.component_count,
                    self.inventory_factor_count,
                    PARTNER_INVENTORY_FACTOR_CLASSES,
                )
            )
            event = base_event[..., None] + residual_head("inventory_change", 1)[..., 0]
            return ResponseLogits(
                visibility=visibility,
                relative_position=position,
                direction=direction,
                inventory=inventory,
                inventory_change=event,
            )

    _RESPONSE_CLASS = ResponseEmission
    return ResponseEmission


def _bernoulli_log_probability(logits: Any, target: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    prediction = jnp.asarray(logits, dtype=jnp.float32)
    label = jnp.asarray(target, dtype=jnp.float32)
    return label * jnn.log_sigmoid(prediction) + (1.0 - label) * jnn.log_sigmoid(
        -prediction
    )


def _categorical_log_probability(logits: Any, target: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    prediction = jnp.asarray(logits, dtype=jnp.float32)
    label = jnp.asarray(target, dtype=jnp.int32)
    index = jnp.broadcast_to(label[..., None, None], prediction.shape[:-1] + (1,))
    return jnp.take_along_axis(jnn.log_softmax(prediction, axis=-1), index, axis=-1)[
        ..., 0
    ]


def _factorized_categorical_log_probability(logits: Any, target: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    prediction = jnp.asarray(logits, dtype=jnp.float32)
    label = jnp.asarray(target, dtype=jnp.int32)
    index = jnp.broadcast_to(
        label[..., None, :, None], prediction.shape[:-1] + (1,)
    )
    return jnp.take_along_axis(jnn.log_softmax(prediction, axis=-1), index, axis=-1)[
        ..., 0
    ]


def response_log_probability(logits: ResponseLogits, targets: Any) -> Any:
    """Return log p(y_t | z_t=k,H_{t-1},a_{t-1}) for every component."""

    import jax.numpy as jnp

    visible = jnp.asarray(targets.visible_mask, dtype=jnp.float32)[..., None]
    event_valid = jnp.asarray(targets.event_mask, dtype=jnp.float32)[..., None]
    visibility = _bernoulli_log_probability(
        logits.visibility,
        jnp.asarray(targets.visibility, dtype=jnp.float32)[..., None],
    )
    position = _categorical_log_probability(
        logits.relative_position, targets.relative_position
    )
    direction = _categorical_log_probability(logits.direction, targets.direction)
    inventory = jnp.sum(
        _factorized_categorical_log_probability(logits.inventory, targets.inventory),
        axis=-1,
    )
    event = _bernoulli_log_probability(
        logits.inventory_change,
        jnp.asarray(targets.interaction_change, dtype=jnp.float32)[..., None],
    )
    return visibility + visible * (position + direction + inventory) + event_valid * event


__all__ = [
    "response_emission_class",
    "response_log_probability",
]
