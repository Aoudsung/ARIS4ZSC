"""Canonical Path C seed identities and deterministic simulator mapping."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from numbers import Integral


UINT64_MAX = (1 << 64) - 1
OCV2_EXECUTION_SEED_VERSION = "path_c_ocv2_execution_seed_v1"


def canonical_uint64_seed(value: int, *, name: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    seed = int(value)
    if not 0 <= seed <= UINT64_MAX:
        raise ValueError(f"{name} must lie in the unsigned 64-bit range.")
    return seed


def derive_ocv2_execution_seed(canonical_seed: int) -> int:
    """Map one canonical 64-bit identity to a versioned JAX 32-bit seed."""

    seed = canonical_uint64_seed(canonical_seed, name="canonical seed")
    digest = hashlib.sha256(
        f"{OCV2_EXECUTION_SEED_VERSION}:{seed}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def validate_unique_execution_seed_mapping(
    canonical_seeds: Iterable[int],
    *,
    name: str,
) -> dict[int, int]:
    """Reject identity or execution-key collisions in one frozen finite schedule."""

    mapping: dict[int, int] = {}
    owner_by_execution_seed: dict[int, int] = {}
    for raw_seed in canonical_seeds:
        canonical_seed = canonical_uint64_seed(raw_seed, name=name)
        if canonical_seed in mapping:
            raise ValueError(f"{name} repeats canonical seed {canonical_seed}.")
        execution_seed = derive_ocv2_execution_seed(canonical_seed)
        previous = owner_by_execution_seed.get(execution_seed)
        if previous is not None:
            raise ValueError(
                f"{name} has an OCV2 execution-seed collision between canonical "
                f"seeds {previous} and {canonical_seed}."
            )
        mapping[canonical_seed] = execution_seed
        owner_by_execution_seed[execution_seed] = canonical_seed
    return mapping
