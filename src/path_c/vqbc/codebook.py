"""Value-vector quantization codebook for the fifteen non-terminal responses."""

from __future__ import annotations

from typing import Any, NamedTuple

from .types import VQBCCodebookState


class CodebookUpdate(NamedTuple):
    state: VQBCCodebookState
    assignments: Any
    quantization_error: Any
    replaced_codes: Any


def empty_codebook(*, code_count: int, signature_dim: int) -> VQBCCodebookState:
    import jax.numpy as jnp

    if code_count <= 1 or signature_dim <= 0:
        raise ValueError("Codebook dimensions must be positive.")
    return VQBCCodebookState(
        embeddings=jnp.zeros((code_count, signature_dim), dtype=jnp.float32),
        exponential_counts=jnp.zeros((code_count,), dtype=jnp.float32),
        exponential_sums=jnp.zeros(
            (code_count, signature_dim), dtype=jnp.float32
        ),
        unused_rollouts=jnp.zeros((code_count,), dtype=jnp.int32),
        initialized=jnp.asarray(False),
        replacement_count=jnp.asarray(0, dtype=jnp.int32),
        last_replaced_codes=jnp.zeros((code_count,), dtype=jnp.bool_),
    )


def farthest_point_initialization(
    signatures: Any, key: Any, code_count: int, valid_mask: Any | None = None
) -> Any:
    """Choose one keyed center and then the farthest signature repeatedly."""

    import jax
    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32)
    if values.ndim != 2 or values.shape[0] < code_count:
        raise ValueError("Codebook initialization needs at least one row per code.")
    valid = (
        jnp.ones((values.shape[0],), dtype=jnp.bool_)
        if valid_mask is None
        else jnp.asarray(valid_mask, dtype=jnp.bool_)
    )
    if valid.shape != (values.shape[0],):
        raise ValueError("Codebook valid mask has the wrong shape.")
    probabilities = valid.astype(jnp.float32)
    probabilities = probabilities / jnp.maximum(jnp.sum(probabilities), 1.0)
    first = jax.random.choice(key, values.shape[0], shape=(), p=probabilities)
    centers = jnp.zeros((code_count, values.shape[-1]), dtype=values.dtype)
    centers = centers.at[0].set(values[first])
    minimum_distance = jnp.where(
        valid,
        jnp.sum(jnp.square(values - centers[0]), axis=-1),
        -jnp.inf,
    )

    def add_center(index: int, carry: tuple[Any, Any]) -> tuple[Any, Any]:
        current_centers, distances = carry
        selected = jnp.argmax(distances)
        center = values[selected]
        current_centers = current_centers.at[index].set(center)
        candidate_distance = jnp.where(
            valid,
            jnp.sum(jnp.square(values - center), axis=-1),
            -jnp.inf,
        )
        return current_centers, jnp.where(
            valid, jnp.minimum(distances, candidate_distance), -jnp.inf
        )

    return jax.lax.fori_loop(
        1, code_count, add_center, (centers, minimum_distance)
    )[0]


def nearest_codes(signatures: Any, embeddings: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32)
    codes = jnp.asarray(embeddings, dtype=jnp.float32)
    distances = jnp.sum(
        jnp.square(values[..., None, :] - codes), axis=-1
    )
    assignments = jnp.argmin(distances, axis=-1)
    error = jnp.take_along_axis(distances, assignments[..., None], axis=-1)[..., 0]
    return assignments, error


def update_codebook(
    state: VQBCCodebookState,
    *,
    signatures: Any,
    key: Any,
    valid_mask: Any | None = None,
    decay: float = 0.99,
    replacement_after_rollouts: int = 10,
    epsilon: float = 1.0e-5,
) -> CodebookUpdate:
    """Apply initialization, exponential moving averages, and stale-code repair."""

    import jax
    import jax.numpy as jnp

    values = jnp.asarray(signatures, dtype=jnp.float32).reshape(
        (-1, state.embeddings.shape[-1])
    )
    valid = (
        jnp.ones((values.shape[0],), dtype=jnp.float32)
        if valid_mask is None
        else jnp.asarray(valid_mask, dtype=jnp.float32).reshape((-1,))
    )
    if valid.shape != (values.shape[0],):
        raise ValueError("Codebook valid mask must align with signatures.")
    code_count = int(state.embeddings.shape[0])
    initialized_embeddings = jax.lax.cond(
        state.initialized,
        lambda unused: state.embeddings,
        lambda unused: farthest_point_initialization(
            values, key, code_count, valid > 0.0
        ),
        operand=None,
    )
    assignments, error = nearest_codes(values, initialized_embeddings)
    one_hot = jax.nn.one_hot(assignments, code_count, dtype=jnp.float32)
    weighted_one_hot = one_hot * valid[:, None]
    batch_counts = jnp.sum(weighted_one_hot, axis=0)
    batch_sums = jnp.einsum("nc,nd->cd", weighted_one_hot, values)
    old_counts = jnp.where(
        state.initialized, state.exponential_counts, jnp.zeros_like(batch_counts)
    )
    old_sums = jnp.where(
        state.initialized,
        state.exponential_sums,
        jnp.zeros_like(batch_sums),
    )
    exponential_counts = float(decay) * old_counts + (1.0 - float(decay)) * batch_counts
    exponential_sums = float(decay) * old_sums + (1.0 - float(decay)) * batch_sums
    candidate_embeddings = exponential_sums / jnp.maximum(
        exponential_counts[:, None], epsilon
    )
    candidate_embeddings = jnp.where(
        (exponential_counts > epsilon)[:, None],
        candidate_embeddings,
        initialized_embeddings,
    )
    unused = jnp.where(
        batch_counts > 0.0,
        jnp.zeros_like(state.unused_rollouts),
        state.unused_rollouts + 1,
    )
    stale = unused >= int(replacement_after_rollouts)
    descending_error_indexes = jnp.argsort(
        jnp.where(valid > 0.0, error, -jnp.inf)
    )[::-1]
    replacement_rows = values[
        descending_error_indexes[
            jnp.arange(code_count, dtype=jnp.int32) % values.shape[0]
        ]
    ]
    embeddings = jnp.where(stale[:, None], replacement_rows, candidate_embeddings)
    exponential_counts = jnp.where(stale, jnp.ones_like(exponential_counts), exponential_counts)
    exponential_sums = jnp.where(
        stale[:, None], replacement_rows, exponential_sums
    )
    unused = jnp.where(stale, jnp.zeros_like(unused), unused)
    next_state = VQBCCodebookState(
        embeddings=embeddings,
        exponential_counts=exponential_counts,
        exponential_sums=exponential_sums,
        unused_rollouts=unused,
        initialized=jnp.asarray(True),
        replacement_count=state.replacement_count + jnp.sum(stale.astype(jnp.int32)),
        last_replaced_codes=stale,
    )
    return CodebookUpdate(
        state=next_state,
        assignments=assignments.reshape(jnp.asarray(signatures).shape[:-1]),
        quantization_error=error.reshape(jnp.asarray(signatures).shape[:-1]),
        replaced_codes=stale,
    )


def target_response_signatures(*, target_centered_advantages: Any) -> Any:
    """Return an assignment-independent next-control response signature.

    The response alphabet must not depend on the latent E-step that it later
    helps evaluate. We therefore average the two target estimators and then
    average over the permutation-symmetric slot bank. Partner-dependent public
    transitions can still change this value signature through the next history,
    but no current responsibility or identity label enters the code target.
    """

    import jax.numpy as jnp

    advantages = jnp.asarray(target_centered_advantages)
    if advantages.shape[-3] != 2:
        raise ValueError("Target response signatures require two Q estimators.")
    mean_advantage = 0.5 * (
        advantages[..., 0, :, :] + advantages[..., 1, :, :]
    )
    return jnp.mean(mean_advantage, axis=-2)


__all__ = [
    "CodebookUpdate",
    "empty_codebook",
    "farthest_point_initialization",
    "nearest_codes",
    "target_response_signatures",
    "update_codebook",
]
