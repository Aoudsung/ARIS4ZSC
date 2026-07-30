"""Training-only privileged context encoders."""

from __future__ import annotations

from typing import Any

_CODE_TEACHER: Any | None = None


def code_teacher_class() -> Any:
    global _CODE_TEACHER
    if _CODE_TEACHER is not None:
        return _CODE_TEACHER

    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import orthogonal, zeros

    class CodeConditionedTeacher(nn.Module):
        latent_dim: int
        hidden_dim: int

        @nn.compact
        def __call__(self, partner_code: Any, task_features: Any) -> Any:
            code = jnp.asarray(partner_code, dtype=jnp.float32)
            task = jnp.asarray(task_features, dtype=jnp.float32)
            if code.shape[:-1] != task.shape[:-1]:
                code = jnp.broadcast_to(code, task.shape[:-1] + (code.shape[-1],))
            joined = jnp.concatenate((code, task), axis=-1)
            hidden = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(jnp.sqrt(2.0)),
                    bias_init=zeros,
                    name="teacher_hidden",
                )(joined)
            )
            return nn.Dense(
                self.latent_dim,
                kernel_init=orthogonal(0.05),
                bias_init=zeros,
                name="teacher_latent",
            )(hidden)

    _CODE_TEACHER = CodeConditionedTeacher
    return CodeConditionedTeacher


def degenerate_gaussian_mixture(
    latent: Any,
    *,
    mixture_components: int,
    log_variance: float = -12.0,
) -> tuple[Any, Any, Any]:
    """Represent one privileged context through the shared belief interface."""

    import jax.numpy as jnp

    if mixture_components <= 0:
        raise ValueError("mixture_components must be positive.")
    z = jnp.asarray(latent, dtype=jnp.float32)
    means = jnp.broadcast_to(
        z[..., None, :], z.shape[:-1] + (int(mixture_components), z.shape[-1])
    )
    log_variances = jnp.full_like(means, float(log_variance))
    logits = jnp.full(
        z.shape[:-1] + (int(mixture_components),), -1.0e9, dtype=jnp.float32
    )
    logits = logits.at[..., 0].set(0.0)
    return logits, means, log_variances


__all__ = ["code_teacher_class", "degenerate_gaussian_mixture"]
