from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

ObsEncoderType = Literal["mlp", "cnn", "auto"]
ObsShape = int | tuple[int, ...]


class OCV2ObsEncoder(nn.Module):
    def __init__(
        self,
        input_shape: ObsShape,
        hidden_dim: int,
        encoder_type: ObsEncoderType = "auto",
    ):
        super().__init__()
        self.input_shape = _normalize_shape(input_shape)
        self.hidden_dim = int(hidden_dim)
        self.encoder_type = _resolve_encoder_type(self.input_shape, encoder_type)

        if self.encoder_type == "mlp":
            self.flat_dim = int(np.prod(self.input_shape))
            self.net = nn.Sequential(
                nn.Linear(self.flat_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
        elif self.encoder_type == "cnn":
            channels = _infer_channels(self.input_shape)
            self.net = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, self.hidden_dim),
                nn.ReLU(),
            )
        else:  # pragma: no cover - guarded by _resolve_encoder_type
            raise ValueError(f"Unsupported encoder_type={self.encoder_type!r}.")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "mlp":
            if obs.dim() < 2:
                raise ValueError(
                    f"MLP obs must include batch dimension; got {tuple(obs.shape)}."
                )
            return self.net(obs.float().reshape(obs.shape[0], -1))

        if obs.dim() != 4:
            raise ValueError(f"CNN obs must have rank 4; got {tuple(obs.shape)}.")
        return self.net(_to_nchw(obs.float(), self.input_shape))


def infer_obs_dim(env: Any, obs: dict[str, np.ndarray] | None = None) -> ObsShape:
    return infer_obs_shape(env, obs)


def infer_obs_shape(env: Any, obs: dict[str, np.ndarray] | None = None) -> ObsShape:
    if obs is not None and "agent_0" in obs:
        return _array_shape(obs["agent_0"])

    adapter_obs = getattr(env, "obs", None)
    if isinstance(adapter_obs, dict) and "agent_0" in adapter_obs:
        return _array_shape(adapter_obs["agent_0"])

    space = _agent_observation_space(env)
    shape = getattr(space, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)

    raise ValueError(
        "Cannot infer OvercookedV2 observation shape. Pass reset obs or use an "
        "environment exposing observation_space['agent_0'].shape."
    )


def _resolve_encoder_type(
    input_shape: tuple[int, ...],
    encoder_type: ObsEncoderType,
) -> Literal["mlp", "cnn"]:
    if encoder_type not in {"mlp", "cnn", "auto"}:
        raise ValueError("encoder_type must be 'mlp', 'cnn', or 'auto'.")
    if encoder_type == "auto":
        return "cnn" if len(input_shape) >= 3 else "mlp"
    if encoder_type == "cnn" and len(input_shape) < 3:
        raise ValueError(
            f"CNN encoder requires image-like observation shape; got {input_shape}."
        )
    return encoder_type


def _normalize_shape(input_shape: ObsShape) -> tuple[int, ...]:
    if isinstance(input_shape, int):
        return (int(input_shape),)
    shape = tuple(int(dim) for dim in input_shape)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"Invalid observation shape {input_shape!r}.")
    return shape


def _infer_channels(shape: tuple[int, ...]) -> int:
    if len(shape) != 3:
        raise ValueError(f"CNN encoder expects rank-3 per-agent obs shape; got {shape}.")
    if shape[-1] <= 32:
        return int(shape[-1])
    if shape[0] <= 32:
        return int(shape[0])
    raise ValueError(
        f"Cannot infer channel axis for observation shape {shape}; expected HWC or CHW."
    )


def _to_nchw(obs: torch.Tensor, input_shape: tuple[int, ...]) -> torch.Tensor:
    if input_shape[-1] <= 32:
        return obs.permute(0, 3, 1, 2).contiguous()
    return obs.contiguous()


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


def _array_shape(value: Any) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("Observation for agent_0 is empty.")
    return tuple(int(dim) for dim in array.shape)
