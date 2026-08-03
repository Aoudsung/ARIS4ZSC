"""Policy-drift and age weighted V6 counterfactual replay."""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple


class AnchorReplayState(NamedTuple):
    batch: Any
    item_count: Any
    capacity: int
    use_counts: Any


def append_replay_batch(
    state: AnchorReplayState | None, *, batch: Any, capacity: int
) -> AnchorReplayState:
    import jax
    import jax.numpy as jnp

    incoming = int(jnp.asarray(batch.anchor_ids).shape[0])
    if incoming <= 0 or int(capacity) <= 0:
        raise ValueError("Replay batch and capacity must be positive.")
    old_count = 0 if state is None else int(jnp.asarray(state.item_count))
    if state is None:
        valid_old = None
        valid_usage = jnp.zeros((0,), dtype=jnp.int32)
    else:
        valid_old = jax.tree_util.tree_map(
            lambda value: jnp.asarray(value)[:old_count], state.batch
        )
        valid_usage = jnp.asarray(state.use_counts)[:old_count]
    combined_valid = (
        batch
        if valid_old is None
        else jax.tree_util.tree_map(
            lambda old, new: jnp.concatenate((old, new), axis=0), valid_old, batch
        )
    )
    usage_valid = jnp.concatenate(
        (valid_usage, jnp.zeros((incoming,), dtype=jnp.int32)), axis=0
    )
    valid_count = old_count + incoming
    if valid_count > int(capacity):
        combined_valid = jax.tree_util.tree_map(
            lambda value: jnp.asarray(value)[-int(capacity):], combined_valid
        )
        usage_valid = usage_valid[-int(capacity):]
        valid_count = int(capacity)

    padding = int(capacity) - valid_count

    def fixed(value: Any) -> Any:
        array = jnp.asarray(value)
        if not padding:
            return array
        return jnp.concatenate(
            (array, jnp.zeros((padding,) + array.shape[1:], dtype=array.dtype)),
            axis=0,
        )

    combined = jax.tree_util.tree_map(fixed, combined_valid)
    use_counts = jnp.concatenate(
        (usage_valid, jnp.zeros((padding,), dtype=jnp.int32)), axis=0
    )
    return AnchorReplayState(
        batch=combined,
        item_count=jnp.asarray(valid_count, dtype=jnp.int32),
        capacity=int(capacity),
        use_counts=use_counts,
    )


def sample_replay_batch(
    state: AnchorReplayState,
    *,
    key: Any,
    batch_size: int,
) -> tuple[AnchorReplayState, Any]:
    """Sample 32 ordinary rows and 16 complete matched pairs by source/age strata."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    count = int(jnp.asarray(state.item_count))
    size = int(batch_size)
    if count < size or size <= 0 or size % 4:
        raise ValueError("Replay minibatches require a positive 4-divisible size.")
    pair_rows = size // 2
    ordinary_rows = size - pair_rows
    pair_count = pair_rows // 2
    sources = np.asarray(state.batch.partner_sources[:count], dtype=np.int64)
    updates = np.asarray(state.batch.collection_update[:count], dtype=np.int64)
    pair_ids = np.asarray(state.batch.matched_pair_ids[:count], dtype=np.int64)
    priorities = np.asarray(
        jax.random.uniform(key, (count,), dtype=jnp.float32), dtype=np.float64
    )

    def stratified(candidates: list[int], requested: int) -> list[int]:
        strata: dict[tuple[int, int], list[int]] = {}
        for index in candidates:
            strata.setdefault((int(sources[index]), int(updates[index])), []).append(index)
        quota = max(1, int(np.ceil(requested / max(len(strata), 1))))
        result: list[int] = []
        used: set[int] = set()
        for label in sorted(strata):
            ranked = sorted(strata[label], key=lambda item: priorities[item], reverse=True)
            for index in ranked[:quota]:
                result.append(index)
                used.add(index)
        if len(result) < requested:
            ranked = sorted(candidates, key=lambda item: priorities[item], reverse=True)
            result.extend(index for index in ranked if index not in used)
        return result[:requested]

    ordinary_candidates = np.flatnonzero(pair_ids < 0).tolist()
    pair_starts = [
        index
        for index in range(count - 1)
        if pair_ids[index] >= 0
        and pair_ids[index] == pair_ids[index + 1]
        and updates[index] == updates[index + 1]
        and (index == 0 or pair_ids[index - 1] != pair_ids[index] or updates[index - 1] != updates[index])
    ]
    if len(ordinary_candidates) < ordinary_rows or len(pair_starts) < pair_count:
        raise ValueError("Replay lacks complete ordinary/matched strata for one minibatch.")
    ordinary = stratified(ordinary_candidates, ordinary_rows)
    starts = stratified(pair_starts, pair_count)
    selected = ordinary + [row for start in starts for row in (start, start + 1)]
    indexes = jnp.asarray(selected, dtype=jnp.int32)
    sampled = jax.tree_util.tree_map(lambda value: value[indexes], state.batch)
    usage = state.use_counts.at[indexes].add(1)
    return state._replace(use_counts=usage), sampled


def policy_drift_age_weights(
    *,
    current_policy_logits: Any,
    collection_policy_logits: Any,
    current_update: Any,
    collection_update: Any,
    kl_decay: float = 0.05,
    age_decay_updates: float = 64.0,
    minimum_weight: float = 1.0e-3,
) -> Any:
    import jax
    import jax.numpy as jnp

    current_log = jax.nn.log_softmax(current_policy_logits, axis=-1)
    collection_log = jax.nn.log_softmax(collection_policy_logits, axis=-1)
    current_probability = jnp.exp(current_log)
    divergence = jnp.sum(
        current_probability * (current_log - collection_log), axis=-1
    )
    age = jnp.maximum(
        jnp.asarray(current_update, dtype=jnp.float32)
        - jnp.asarray(collection_update, dtype=jnp.float32),
        0.0,
    )
    weight = jnp.exp(-divergence / float(kl_decay)) * jnp.exp(
        -age / float(age_decay_updates)
    )
    weight = jnp.clip(weight, float(minimum_weight), 1.0)
    return weight / jnp.maximum(jnp.mean(weight), 1.0e-8)


def replay_fingerprint(state: AnchorReplayState | None) -> str:
    if state is None:
        return hashlib.sha256(b"delta-zsc-v6/empty-anchor-replay").hexdigest()
    import jax
    import numpy as np

    digest = hashlib.sha256()
    count = int(np.asarray(jax.device_get(state.item_count)))
    valid = jax.tree_util.tree_map(lambda value: value[:count], state.batch)
    for leaf in jax.tree_util.tree_leaves((valid, state.use_counts[:count])):
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


__all__ = [
    "AnchorReplayState",
    "append_replay_batch",
    "policy_drift_age_weights",
    "replay_fingerprint",
    "sample_replay_batch",
]
