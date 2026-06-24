from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def pad_marginals(
    marginals: list[torch.Tensor],
    factor_modes: list[int],
) -> torch.Tensor:
    if not marginals:
        return torch.empty(0, 0, 0)
    batch_size = marginals[0].shape[0]
    max_modes = max(factor_modes) if factor_modes else 0
    padded = marginals[0].new_zeros(batch_size, len(marginals), max_modes)
    for factor_idx, marginal in enumerate(marginals):
        padded[:, factor_idx, : marginal.shape[-1]] = marginal
    return padded


def uniform_marginals(
    factor_modes: list[int],
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    return [
        torch.ones(batch_size, n_modes, device=device) / float(n_modes)
        for n_modes in factor_modes
    ]


class FactorLocalQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_options: int | None = None,
        factor_modes: list[int] | None = None,
        relevance_mask: list[list[bool]] | torch.Tensor | None = None,
        hidden_dim: int = 128,
        max_options: int | None = None,
        max_factors: int | None = None,
        max_modes: int | None = None,
        option_feature_dim: int = 0,
        factor_feature_dim: int = 0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_options = int(n_options if n_options is not None else max_options or 1)
        self.factor_modes = list(factor_modes or [])
        self.n_factors = int(
            max_factors if max_factors is not None else len(self.factor_modes)
        )
        self.max_options = int(max_options or self.n_options)
        self.max_modes = int(
            max_modes
            or (max(self.factor_modes) if self.factor_modes else 1)
        )
        self.hidden_dim = hidden_dim
        self.option_feature_dim = int(option_feature_dim)
        self.factor_feature_dim = int(factor_feature_dim)

        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_options),
        )
        self.obs_proj = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.factor_embedding = nn.Embedding(max(self.n_factors, 1), hidden_dim)
        self.option_embedding = nn.Embedding(max(self.max_options, 1), hidden_dim)
        self.option_feature_proj = (
            nn.Linear(self.option_feature_dim, hidden_dim)
            if self.option_feature_dim > 0
            else None
        )
        self.factor_feature_proj = (
            nn.Linear(self.factor_feature_dim, hidden_dim)
            if self.factor_feature_dim > 0
            else None
        )
        self.residual_weight = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_modes),
        )

        mask = _as_relevance_tensor(relevance_mask, self.n_factors, self.max_options)
        self.register_buffer("relevance_mask", mask)

    def q_base_values(
        self,
        obs: torch.Tensor,
        option_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_base = self.base(obs)
        if option_mask is not None:
            q_base = q_base[:, : option_mask.shape[1]]
            return q_base.masked_fill(~option_mask.bool(), -1e9)
        return q_base[:, : self.n_options]

    def q_uniform_values(self, obs: torch.Tensor) -> torch.Tensor:
        marginals = uniform_marginals(self.factor_modes, obs.shape[0], obs.device)
        return self.forward(obs, marginals)

    def forward(
        self,
        obs_feat: torch.Tensor,
        belief: list[torch.Tensor] | torch.Tensor,
        option_mask: torch.Tensor | None = None,
        factor_mask: torch.Tensor | None = None,
        mode_mask: torch.Tensor | None = None,
        relevance_mask: torch.Tensor | None = None,
        option_features: torch.Tensor | None = None,
        factor_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(belief, list):
            return self._forward_toy(obs_feat, belief)
        if option_mask is None or factor_mask is None or mode_mask is None:
            raise ValueError(
                "Padded FactorLocalQNetwork.forward requires option_mask, "
                "factor_mask, and mode_mask."
            )
        return self._forward_padded(
            obs_feat,
            belief,
            option_mask,
            factor_mask,
            mode_mask,
            relevance_mask,
            option_features,
            factor_features,
        )

    def _forward_toy(
        self,
        obs: torch.Tensor,
        marginals: list[torch.Tensor],
    ) -> torch.Tensor:
        q_base = self.q_base_values(obs)
        if self.n_factors == 0:
            return q_base
        belief = pad_marginals(marginals, self.factor_modes).to(obs.device)
        batch_size = obs.shape[0]
        factor_mask = torch.ones(
            batch_size,
            self.n_factors,
            dtype=torch.bool,
            device=obs.device,
        )
        mode_mask = self._toy_mode_mask(batch_size, obs.device)
        relevance_mask = self.relevance_mask[: self.n_factors, : self.n_options]
        return self._forward_padded(
            obs,
            belief,
            option_mask=None,
            factor_mask=factor_mask,
            mode_mask=mode_mask,
            relevance_mask=relevance_mask,
            option_features=None,
            factor_features=None,
        )

    def _forward_padded(
        self,
        obs_feat: torch.Tensor,
        belief: torch.Tensor,
        option_mask: torch.Tensor | None,
        factor_mask: torch.Tensor,
        mode_mask: torch.Tensor,
        relevance_mask: torch.Tensor | None,
        option_features: torch.Tensor | None,
        factor_features: torch.Tensor | None,
    ) -> torch.Tensor:
        _validate_rank(obs_feat, 2, "obs_feat")
        _validate_rank(belief, 3, "belief")
        batch_size, num_factors, max_modes = belief.shape
        num_options = (
            int(option_mask.shape[1])
            if option_mask is not None
            else self.n_options
        )
        if num_factors > self.n_factors:
            raise ValueError(
                f"Received {num_factors} factors, but model supports {self.n_factors}."
            )
        if num_options > self.max_options:
            raise ValueError(
                f"Received {num_options} options, but model supports {self.max_options}."
            )
        if max_modes > self.max_modes:
            raise ValueError(
                f"Received K_max={max_modes}, but model supports {self.max_modes}."
            )
        if mode_mask.shape != belief.shape:
            raise ValueError("mode_mask must have the same shape as belief.")
        if factor_mask.shape != (batch_size, num_factors):
            raise ValueError("factor_mask must have shape [B, F].")

        q_base = self.base(obs_feat)[:, :num_options]
        if num_factors == 0:
            return _mask_options(q_base, option_mask)

        centered_belief = belief - masked_uniform(mode_mask).to(dtype=belief.dtype)
        centered_belief = centered_belief * factor_mask[:, :, None].to(belief.dtype)

        obs_context = self.obs_proj(obs_feat)
        factor_context = self._factor_context(
            batch_size,
            num_factors,
            obs_feat.device,
            factor_features,
        )
        option_context = self._option_context(
            batch_size,
            num_options,
            obs_feat.device,
            option_features,
        )

        obs_term = obs_context[:, None, None, :].expand(
            batch_size,
            num_factors,
            num_options,
            -1,
        )
        factor_term = factor_context[:, :, None, :].expand(
            batch_size,
            num_factors,
            num_options,
            -1,
        )
        option_term = option_context[:, None, :, :].expand(
            batch_size,
            num_factors,
            num_options,
            -1,
        )
        weights = self.residual_weight(
            torch.cat([obs_term, factor_term, option_term], dim=-1)
        )[..., :max_modes]
        adv = (weights * centered_belief[:, :, None, :]).sum(dim=-1)
        rel = self._effective_relevance_mask(
            relevance_mask,
            batch_size,
            num_factors,
            num_options,
            obs_feat.device,
        )
        adv = adv * rel.to(dtype=adv.dtype)
        adv = adv * factor_mask[:, :, None].to(dtype=adv.dtype)
        q_values = q_base + adv.sum(dim=1)
        return _mask_options(q_values, option_mask)

    def _toy_mode_mask(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        mode_mask = torch.zeros(
            batch_size,
            self.n_factors,
            self.max_modes,
            dtype=torch.bool,
            device=device,
        )
        for factor_idx, n_modes in enumerate(self.factor_modes):
            mode_mask[:, factor_idx, :n_modes] = True
        return mode_mask

    def _factor_context(
        self,
        batch_size: int,
        num_factors: int,
        device: torch.device,
        factor_features: torch.Tensor | None,
    ) -> torch.Tensor:
        factor_ids = torch.arange(num_factors, device=device)
        context = self.factor_embedding(factor_ids)
        context = context.unsqueeze(0).expand(batch_size, -1, -1)
        if factor_features is not None and factor_features.shape[-1] > 0:
            if self.factor_feature_proj is None:
                raise ValueError("Construct with factor_feature_dim to use factor_features.")
            context = context + self.factor_feature_proj(factor_features.to(device))
        return context

    def _option_context(
        self,
        batch_size: int,
        num_options: int,
        device: torch.device,
        option_features: torch.Tensor | None,
    ) -> torch.Tensor:
        option_ids = torch.arange(num_options, device=device)
        context = self.option_embedding(option_ids)
        context = context.unsqueeze(0).expand(batch_size, -1, -1)
        if option_features is not None and option_features.shape[-1] > 0:
            if self.option_feature_proj is None:
                raise ValueError("Construct with option_feature_dim to use option_features.")
            context = context + self.option_feature_proj(option_features.to(device))
        return context

    def _effective_relevance_mask(
        self,
        relevance_mask: torch.Tensor | None,
        batch_size: int,
        num_factors: int,
        num_options: int,
        device: torch.device,
    ) -> torch.Tensor:
        if relevance_mask is None:
            rel = self.relevance_mask[:num_factors, :num_options]
        else:
            rel = relevance_mask.to(device)
        if rel.dim() == 2:
            rel = rel.unsqueeze(0).expand(batch_size, -1, -1)
        expected = (batch_size, num_factors, num_options)
        if rel.shape != expected:
            raise ValueError(f"relevance_mask must have shape {expected}.")
        return rel.bool()


def masked_uniform(mode_mask: torch.Tensor) -> torch.Tensor:
    valid = mode_mask.bool()
    denom = valid.sum(dim=-1, keepdim=True).clamp(min=1)
    return valid.to(dtype=torch.float32) / denom.to(dtype=torch.float32)


def _as_relevance_tensor(
    relevance_mask: Sequence[Sequence[bool]] | torch.Tensor | None,
    n_factors: int,
    n_options: int,
) -> torch.Tensor:
    if relevance_mask is None:
        return torch.zeros(n_factors, n_options, dtype=torch.bool)
    mask = torch.as_tensor(relevance_mask, dtype=torch.bool)
    if mask.numel() == 0:
        return torch.zeros(n_factors, n_options, dtype=torch.bool)
    if mask.dim() != 2:
        raise ValueError("relevance_mask must be two-dimensional.")
    padded = torch.zeros(n_factors, n_options, dtype=torch.bool)
    rows = min(n_factors, mask.shape[0])
    cols = min(n_options, mask.shape[1])
    padded[:rows, :cols] = mask[:rows, :cols]
    return padded


def _mask_options(
    q_values: torch.Tensor,
    option_mask: torch.Tensor | None,
) -> torch.Tensor:
    if option_mask is None:
        return q_values
    return q_values.masked_fill(~option_mask.bool(), -1e9)


def _validate_rank(tensor: torch.Tensor, rank: int, name: str) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}; got shape {tuple(tensor.shape)}.")
