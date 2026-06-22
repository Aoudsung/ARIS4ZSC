"""Graph variant definitions for toy factor-game experiments."""

from dataclasses import dataclass
import functools

from .env import NUM_FACTORS
from .ce_estimation import DEFAULT_CE_THRESHOLD, estimate_ce_matrix
from .options import GROUND_TRUTH_FACTORS, IRRELEVANT_FACTORS, NON_CRITICAL_FACTORS, OptionID


GRAPH_VARIANTS = (
    "full_graph",
    "plus_irrelevant",
    "minus_noncritical",
    "minus_critical",
    "random_graph",
    "complete_graph",
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


def _from_interaction_factor(factor, env_factor_id: int | None = None) -> GraphFactorSpec:
    return GraphFactorSpec(
        name=factor.description,
        option_i=factor.option_i,
        option_j=factor.option_j,
        n_modes=factor.n_modes,
        env_factor_id=env_factor_id if env_factor_id is not None else factor.factor_id,
        ce_value=factor.ce_value,
    )


def _chain_pairs(n_factors: int) -> tuple[tuple[int, int], ...]:
    if n_factors <= 1:
        return ()
    return tuple((i, i + 1) for i in range(n_factors - 1))


def _real_factor_for_option_pair(option_i: OptionID, option_j: OptionID) -> GraphFactorSpec | None:
    for factor in GROUND_TRUTH_FACTORS:
        if factor.option_i == option_i and factor.option_j == option_j:
            return _from_interaction_factor(factor, env_factor_id=factor.factor_id)
    return None


@functools.lru_cache(maxsize=1)
def _candidate_factor_specs() -> tuple[GraphFactorSpec, ...]:
    ce_matrix = estimate_ce_matrix()
    specs = []
    for factor in (GROUND_TRUTH_FACTORS + NON_CRITICAL_FACTORS + IRRELEVANT_FACTORS):
        env_factor_id = factor.factor_id if factor in GROUND_TRUTH_FACTORS else None
        specs.append(
            GraphFactorSpec(
                name=factor.description,
                option_i=factor.option_i,
                option_j=factor.option_j,
                n_modes=factor.n_modes,
                env_factor_id=env_factor_id,
                ce_value=float(ce_matrix[int(factor.option_i), int(factor.option_j)]),
                sparsity_weight=1.0 if env_factor_id is not None else 1.5,
            )
        )
    return tuple(specs)


@functools.lru_cache(maxsize=4)
def _induced_full_factors(threshold: float = DEFAULT_CE_THRESHOLD) -> tuple[GraphFactorSpec, ...]:
    factors = tuple(
        factor for factor in _candidate_factor_specs()
        if factor.ce_value > threshold
    )
    true_pairs = {(factor.option_i, factor.option_j) for factor in factors if factor.env_factor_id is not None}
    required_pairs = {(factor.option_i, factor.option_j) for factor in GROUND_TRUTH_FACTORS}
    if not required_pairs.issubset(true_pairs):
        missing = sorted((a.name, b.name) for a, b in (required_pairs - true_pairs))
        raise RuntimeError(
            f"CE induction with threshold={threshold} missed required toy support pairs: {missing}"
        )
    return factors


def get_graph_config(name: str) -> GraphConfig:
    if name not in GRAPH_VARIANTS:
        raise ValueError(f"Unknown graph variant {name!r}; expected one of {GRAPH_VARIANTS}")

    induced_full = _induced_full_factors()

    if name == "full_graph":
        factors = induced_full
    elif name == "plus_irrelevant":
        induced_pairs = {(factor.option_i, factor.option_j) for factor in induced_full}
        synthetic = tuple(
            GraphFactorSpec(
                name=factor.description,
                option_i=factor.option_i,
                option_j=factor.option_j,
                n_modes=factor.n_modes,
                env_factor_id=None,
                ce_value=next(
                    spec.ce_value for spec in _candidate_factor_specs()
                    if spec.option_i == factor.option_i and spec.option_j == factor.option_j
                ),
                sparsity_weight=1.5,
            )
            for factor in (NON_CRITICAL_FACTORS + IRRELEVANT_FACTORS)
            if (factor.option_i, factor.option_j) not in induced_pairs
        )
        factors = induced_full + synthetic
    elif name == "minus_noncritical":
        real_factors = [factor for factor in induced_full if factor.env_factor_id is not None]
        lowest_ce_id = min(real_factors, key=lambda factor: factor.ce_value).env_factor_id
        factors = tuple(
            factor for factor in real_factors if factor.env_factor_id != lowest_ce_id
        )
    elif name == "minus_critical":
        real_factors = [factor for factor in induced_full if factor.env_factor_id is not None]
        highest_ce_id = max(real_factors, key=lambda factor: factor.ce_value).env_factor_id
        factors = tuple(
            factor for factor in real_factors if factor.env_factor_id != highest_ce_id
        )
    elif name == "random_graph":
        factors = (
            GraphFactorSpec("random cross/resource", OptionID.CROSS_CORRIDOR, OptionID.GOTO_RESOURCE_B, 2, None, 1.0),
            GraphFactorSpec("random wait/deliver", OptionID.WAIT_AT_BOTTLENECK, OptionID.DELIVER_RIGHT, 2, None, 1.0),
            GraphFactorSpec("random noop/pickup", OptionID.NOOP, OptionID.PICKUP, 2, None, 1.0),
        )
    else:
        complete_factors = []
        ce_matrix = estimate_ce_matrix()
        for option_i in OptionID:
            for option_j in OptionID:
                real = _real_factor_for_option_pair(option_i, option_j)
                if real is not None:
                    complete_factors.append(
                        GraphFactorSpec(
                            name=real.name,
                            option_i=option_i,
                            option_j=option_j,
                            n_modes=real.n_modes,
                            env_factor_id=real.env_factor_id,
                            ce_value=float(ce_matrix[int(option_i), int(option_j)]),
                        )
                    )
                else:
                    complete_factors.append(
                        GraphFactorSpec(
                            name=f"complete:{option_i.name}:{option_j.name}",
                            option_i=option_i,
                            option_j=option_j,
                            n_modes=2,
                            env_factor_id=None,
                            ce_value=float(ce_matrix[int(option_i), int(option_j)]),
                            sparsity_weight=2.0,
                        )
                    )
        factors = tuple(complete_factors)

    return GraphConfig(name=name, factors=factors, pairwise_pairs=_chain_pairs(len(factors)))


def stable_convention_seed(base_seed: int, convention: dict[int, int], trial: int = 0) -> int:
    value = int(base_seed) * 100_000 + int(trial)
    for factor_id in range(NUM_FACTORS):
        value += (factor_id + 1) * 1_000 * int(convention[factor_id])
    return abs(value) % (2**31 - 1)
