"""SRVF-MAPPO implementation for zero-shot coordination experiments."""

__all__ = [
    "IRFTable",
    "MAPPOActorCritic",
    "NeuralSRVFHeads",
    "RolloutBatch",
    "SRVFBelief",
    "SourceBatch",
    "UnifiedLoss",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module 'raob' has no attribute {name!r}")
    from raob import srvf_mappo

    return getattr(srvf_mappo, name)
