"""Graph variant definitions for toy factor-game experiments."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import random

from .ce_estimation import DEFAULT_CE_THRESHOLD, estimate_ce_matrix
from .env import NUM_FACTORS
from .options import GROUND_TRUTH_FACTORS, OptionID


GRAPH_VARIANTS = (
    "full_support",
    "overcomplete",
    "overcomplete_minus_noncritical",
    "minus_critical",
    "random_same_size",
    "complete_option_graph",
    "shuffled_routes",
    "shuffled_relevance",
)


@dataclass(frozen=True)
class GraphFactorSpec:
    name: str
    option_i: OptionID
    option_j: OptionID
    n_modes: int
    env_factor_id: int | None
    ce_value: float
    sparsity_weight: float = 1.0

    @property
    def is_ground_truth(self) -> bool:
        return self.env_factor_id is not None


@dataclass(frozen=True)
class GraphConfig:
    name: str
    factors: tuple[GraphFactorSpec, ...]
    pairwise_pairs: tuple[tuple[int, int], ...]
    route_permutation: tuple[int, ...] | None = None
    relevance_permutation: tuple[int, ...] | None = None

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def factor_modes(self) -> list[int]:
        return [factor.n_modes for factor in self.factors]

    @property
    def sparsity_weights(self) -> list[float]:
        return [factor.sparsity_weight for factor in self.factors]

    @property
    def ground_truth_mask(self) -> list[bool]:
        return [factor.env_factor_id is not None for factor in self.factors]

    def labels_from_convention(self, convention: dict[int, int]) -> list[int]:
        labels = []
        for factor in self.factors:
            if factor.env_factor_id is None:
                labels.append(0)
            else:
                labels.append(int(convention[factor.env_factor_id]))
        return labels

    def value_weights(self) -> list[float]:
        if not self.factors:
            return []
        max_ce = max(abs(factor.ce_value) for factor in self.factors) or 1.0
        return [max(0.1, abs(factor.ce_value) / max_ce) for factor in self.factors]


def _chain_pairs(n_factors: int) -> tuple[tuple[int, int], ...]:
    if n_factors <= 1:
        return ()
    return tuple((i, i + 1) for i in range(n_factors - 1))


def _real_factor_for_option_pair(
    option_i: OptionID,
    option_j: OptionID,
    ce_value: float,
) -> GraphFactorSpec | None:
    for factor in GROUND_TRUTH_FACTORS:
        if factor.option_i == option_i and factor.option_j == option_j:
            return GraphFactorSpec(
                name=factor.description,
                option_i=option_i,
                option_j=option_j,
                n_modes=factor.n_modes,
                env_factor_id=factor.factor_id,
                ce_value=ce_value,
            )
    return None


@functools.lru_cache(maxsize=1)
def _all_pair_factor_specs() -> tuple[GraphFactorSpec, ...]:
    ce_matrix = estimate_ce_matrix()
    specs = []
    for option_i in OptionID:
        for option_j in OptionID:
            ce_value = float(ce_matrix[int(option_i), int(option_j)])
            real = _real_factor_for_option_pair(option_i, option_j, ce_value)
            if real is not None:
                specs.append(real)
            else:
                specs.append(
                    GraphFactorSpec(
                        name=f"ce:{option_i.name}:{option_j.name}",
                        option_i=option_i,
                        option_j=option_j,
                        n_modes=2,
                        env_factor_id=None,
                        ce_value=ce_value,
                        sparsity_weight=1.5,
                    )
                )
    return tuple(specs)


@functools.lru_cache(maxsize=4)
def _full_support_factors(threshold: float = DEFAULT_CE_THRESHOLD) -> tuple[GraphFactorSpec, ...]:
    factors = tuple(factor for factor in _all_pair_factor_specs() if factor.ce_value > threshold)
    if not factors:
        raise RuntimeError(f"CE induction with threshold={threshold} produced an empty support graph")
    return factors


def _overcomplete_factors() -> tuple[GraphFactorSpec, ...]:
    full = _full_support_factors()
    full_pairs = {(factor.option_i, factor.option_j) for factor in full}
    low_ce = [
        factor for factor in _all_pair_factor_specs()
        if (factor.option_i, factor.option_j) not in full_pairs and factor.ce_value > 0.0
    ]
    low_ce.sort(key=lambda factor: (-factor.ce_value, factor.option_i.name, factor.option_j.name))
    extra_count = min(len(low_ce), max(1, len(full)))
    return full + tuple(low_ce[:extra_count])


def _overcomplete_minus_noncritical_factors() -> tuple[GraphFactorSpec, ...]:
    full = _full_support_factors()
    full_pairs = {(factor.option_i, factor.option_j) for factor in full}
    extras = [
        factor for factor in _overcomplete_factors()
        if (factor.option_i, factor.option_j) not in full_pairs
    ]
    if not extras:
        return full
    extras.sort(key=lambda factor: (factor.ce_value, factor.option_i.name, factor.option_j.name))
    remove_count = max(1, len(extras) // 2)
    removed_pairs = {(factor.option_i, factor.option_j) for factor in extras[:remove_count]}
    return tuple(
        factor for factor in _overcomplete_factors()
        if (factor.option_i, factor.option_j) not in removed_pairs
    )


def _minus_critical_factors() -> tuple[GraphFactorSpec, ...]:
    full = _full_support_factors()
    critical = max(full, key=lambda factor: factor.ce_value)
    return tuple(
        factor for factor in full
        if (factor.option_i, factor.option_j) != (critical.option_i, critical.option_j)
    )


def _random_same_size_factors() -> tuple[GraphFactorSpec, ...]:
    full = _full_support_factors()
    full_pairs = {(factor.option_i, factor.option_j) for factor in full}
    candidates = [
        factor for factor in _all_pair_factor_specs()
        if (factor.option_i, factor.option_j) not in full_pairs
    ]
    if len(candidates) < len(full):
        candidates = list(_all_pair_factor_specs())
    rng = random.Random(17)
    selected = rng.sample(candidates, k=len(full))
    return tuple(
        GraphFactorSpec(
            name=f"random:{factor.option_i.name}:{factor.option_j.name}",
            option_i=factor.option_i,
            option_j=factor.option_j,
            n_modes=2,
            env_factor_id=None,
            ce_value=factor.ce_value,
            sparsity_weight=1.5,
        )
        for factor in selected
    )


def _complete_option_graph_factors() -> tuple[GraphFactorSpec, ...]:
    return tuple(
        GraphFactorSpec(
            name=factor.name if factor.env_factor_id is not None else f"complete:{factor.option_i.name}:{factor.option_j.name}",
            option_i=factor.option_i,
            option_j=factor.option_j,
            n_modes=factor.n_modes,
            env_factor_id=factor.env_factor_id,
            ce_value=factor.ce_value,
            sparsity_weight=factor.sparsity_weight if factor.env_factor_id is None else 1.0,
        )
        for factor in _all_pair_factor_specs()
    )


def _rotated_permutation(n_items: int) -> tuple[int, ...] | None:
    if n_items <= 1:
        return None
    return tuple((idx + 1) % n_items for idx in range(n_items))


def get_graph_config(name: str) -> GraphConfig:
    if name not in GRAPH_VARIANTS:
        raise ValueError(f"Unknown graph variant {name!r}; expected one of {GRAPH_VARIANTS}")

    route_permutation = None
    relevance_permutation = None

    if name == "full_support":
        factors = _full_support_factors()
    elif name == "overcomplete":
        factors = _overcomplete_factors()
    elif name == "overcomplete_minus_noncritical":
        factors = _overcomplete_minus_noncritical_factors()
    elif name == "minus_critical":
        factors = _minus_critical_factors()
    elif name == "random_same_size":
        factors = _random_same_size_factors()
    elif name == "complete_option_graph":
        factors = _complete_option_graph_factors()
    elif name == "shuffled_routes":
        factors = _full_support_factors()
        route_permutation = _rotated_permutation(len(factors))
    else:
        factors = _full_support_factors()
        relevance_permutation = _rotated_permutation(len(factors))

    return GraphConfig(
        name=name,
        factors=factors,
        pairwise_pairs=_chain_pairs(len(factors)),
        route_permutation=route_permutation,
        relevance_permutation=relevance_permutation,
    )


def stable_convention_seed(base_seed: int, convention: dict[int, int], trial: int = 0) -> int:
    value = int(base_seed) * 100_000 + int(trial)
    for factor_id in range(NUM_FACTORS):
        value += (factor_id + 1) * 1_000 * int(convention[factor_id])
    return abs(value) % (2**31 - 1)
