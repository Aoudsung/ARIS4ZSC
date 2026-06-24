from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class OCV2ObsEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2:
            raise ValueError(f"obs must have shape [B, D_obs]; got {tuple(obs.shape)}.")
        return self.net(obs.float())


def infer_obs_dim(env: Any, obs: dict[str, np.ndarray] | None = None) -> int:
    if obs is not None and "agent_0" in obs:
        return _flat_dim(obs["agent_0"])

    adapter_obs = getattr(env, "obs", None)
    if isinstance(adapter_obs, dict) and "agent_0" in adapter_obs:
        return _flat_dim(adapter_obs["agent_0"])

    space = _agent_observation_space(env)
    shape = getattr(space, "shape", None)
    if shape is not None:
        return int(np.prod(tuple(int(dim) for dim in shape)))

    raise ValueError(
        "Cannot infer OvercookedV2 observation dimension. Pass reset obs or use an "
        "environment exposing observation_space['agent_0'].shape."
    )


def _agent_observation_space(env: Any) -> Any:
    candidates = [env, getattr(env, "env", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        space = getattr(candidate, "observation_space", None)
        if isinstance(space, dict) and "agent_0" in space:
            return space["agent_0"]
        if callable(space):
            try:
                value = space("agent_0")
            except TypeError:
                value = space()
            if isinstance(value, dict) and "agent_0" in value:
                return value["agent_0"]
            if value is not None:
                return value
    return None


def _flat_dim(value: Any) -> int:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("Observation for agent_0 is empty.")
    return int(np.prod(array.shape))
