"""Benchmark interfaces used by SRVF-MAPPO collectors."""

from raob.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkStep,
    InterventionSnapshot,
    ResettableBenchmarkAdapter,
    discounted_returns,
)

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkStep",
    "InterventionSnapshot",
    "ResettableBenchmarkAdapter",
    "discounted_returns",
]

try:  # pragma: no cover - only present after server classic Overcooked setup
    from raob.benchmarks.overcooked_classic import (  # noqa: F401
        ClassicOvercookedBenchmarkAdapter,
    )

    __all__.extend(["ClassicOvercookedBenchmarkAdapter"])
except Exception:
    pass

try:  # pragma: no cover - only present after server GOAT setup
    from raob.benchmarks.goat_classic import (  # noqa: F401
        GOATClassicPartnerPolicy,
        load_goat_partner_specs,
    )

    __all__.extend(["GOATClassicPartnerPolicy", "load_goat_partner_specs"])
except Exception:
    pass
