"""ARIS-Bellman agents for the toy factor game.

The main method uses factor-local beliefs and a factor-local Bellman Q
decomposition. Auxiliary response, calibration, sparsity, transition, and
selector losses are intentionally absent from training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evidence import EVIDENCE_DIM, option_relevance_mask
from .factor_belief import FactorBeliefModel
from .options import NUM_OPTIONS


METHODS = ("aris_bellman", "flat_latent", "global_gru", "oracle_belief", "random_policy")


def belief_to_features(marginals: list[torch.Tensor]) -> torch.Tensor:
    if not marginals:
        raise ValueError("belief_to_features requires at least one factor marginal")
    flat = torch.cat(marginals, dim=-1)
    entropies = []
    for marginal in marginals:
        entropies.append(
            -(marginal * (marginal + 1e-8).log()).sum(dim=-1, keepdim=True)
        )
    ent_vec = torch.cat(entropies, dim=-1)
    unresolved = (ent_vec > 0.3).float().sum(dim=-1, keepdim=True)
    return torch.cat([flat, ent_vec, unresolved], dim=-1)


def compute_belief_dim(factor_modes: list[int]) -> int:
    return sum(factor_modes) + len(factor_modes) + 1


def pad_marginals(marginals: list[torch.Tensor], factor_modes: list[int]) -> torch.Tensor:
    if not marginals:
        batch = 0
        return torch.empty(batch, 0, 0)
    batch_size = marginals[0].shape[0]
    max_modes = max(factor_modes)
    padded = marginals[0].new_zeros(batch_size, len(marginals), max_modes)
    for factor_idx, marginal in enumerate(marginals):
        padded[:, factor_idx, :marginal.shape[-1]] = marginal
    return padded


def uniform_marginals(
    factor_modes: list[int], batch_size: int, device: torch.device
) -> list[torch.Tensor]:
    return [
        torch.ones(batch_size, n_modes, device=device) / float(n_modes)
        for n_modes in factor_modes
    ]


def labels_to_marginals(
    factor_labels: torch.Tensor,
    factor_modes: list[int],
) -> list[torch.Tensor]:
    marginals = []
    for factor_idx, n_modes in enumerate(factor_modes):
        labels = factor_labels[:, factor_idx].clamp(min=0, max=n_modes - 1)
        marginals.append(F.one_hot(labels, num_classes=n_modes).float())
    return marginals


class FactorLocalQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_options: int,
        factor_modes: list[int],
        relevance_mask: list[list[bool]],
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_options = n_options
        self.factor_modes = factor_modes
        self.n_factors = len(factor_modes)
        self.max_modes = max(factor_modes) if factor_modes else 1

        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_options),
        )
        self.obs_proj = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.factor_embedding = nn.Embedding(max(self.n_factors, 1), hidden_dim)
        self.option_embedding = nn.Embedding(n_options, hidden_dim)
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim * 3 + self.max_modes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        mask = torch.tensor(relevance_mask, dtype=torch.float32)
        if mask.numel() == 0:
            mask = torch.zeros(self.n_factors, n_options, dtype=torch.float32)
        self.register_buffer("relevance_mask", mask)

    def forward(self, obs: torch.Tensor, marginals: list[torch.Tensor]) -> torch.Tensor:
        q_base = self.base(obs)
        if self.n_factors == 0:
            return q_base

        batch_size = obs.shape[0]
        obs_context = self.obs_proj(obs)
        belief = pad_marginals(marginals, self.factor_modes)
        factor_ids = torch.arange(self.n_factors, device=obs.device)
        option_ids = torch.arange(self.n_options, device=obs.device)
        factor_emb = self.factor_embedding(factor_ids)
        option_emb = self.option_embedding(option_ids)

        obs_term = obs_context[:, None, None, :].expand(batch_size, self.n_factors, self.n_options, -1)
        factor_term = factor_emb[None, :, None, :].expand(batch_size, self.n_factors, self.n_options, -1)
        option_term = option_emb[None, None, :, :].expand(batch_size, self.n_factors, self.n_options, -1)
        belief_term = belief[:, :, None, :].expand(batch_size, self.n_factors, self.n_options, -1)
        adv_in = torch.cat([obs_term, factor_term, option_term, belief_term], dim=-1)
        adv = self.advantage(adv_in).squeeze(-1)
        masked_adv = adv * self.relevance_mask[None, :, :]
        return q_base + masked_adv.sum(dim=1)


class FlatLatentQNetwork(nn.Module):
    def __init__(self, obs_dim: int, belief_dim: int, n_options: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_options),
        )

    def forward(self, obs: torch.Tensor, marginals: list[torch.Tensor]) -> torch.Tensor:
        return self.net(torch.cat([obs, belief_to_features(marginals)], dim=-1))


class GlobalGRUQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, n_options: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=obs_dim + n_actions + n_actions,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.q = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_options),
        )

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.encoder.hidden_size, device=device)

    def step_history(
        self,
        obs_t: torch.Tensor,
        ego_action_t: torch.Tensor,
        partner_action_t: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if ego_action_t.dim() == 1:
            ego_action_t = ego_action_t.unsqueeze(0)
        if partner_action_t.dim() == 1:
            partner_action_t = partner_action_t.unsqueeze(0)
        if hidden is None:
            hidden = self.initial_hidden(obs_t.shape[0], obs_t.device)
        x = torch.cat([obs_t, ego_action_t, partner_action_t], dim=-1).unsqueeze(1)
        _, next_hidden = self.encoder(x, hidden)
        return next_hidden

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        return self.q(torch.cat([obs, hidden], dim=-1))


class ActiveFactorAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_options: int,
        graph_config,
        hidden_dim: int = 128,
        method: str = "aris_bellman",
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_options = n_options
        self.factor_modes = graph_config.factor_modes
        self.method = method

        self.belief_model = FactorBeliefModel(
            evidence_dim=EVIDENCE_DIM,
            n_factors=graph_config.n_factors,
            factor_modes=graph_config.factor_modes,
            hidden_dim=hidden_dim,
        )

        belief_dim = compute_belief_dim(graph_config.factor_modes)
        self.factor_q = FactorLocalQNetwork(
            obs_dim=obs_dim,
            n_options=n_options,
            factor_modes=graph_config.factor_modes,
            relevance_mask=option_relevance_mask(graph_config),
            hidden_dim=hidden_dim,
        )
        self.flat_q = FlatLatentQNetwork(obs_dim, belief_dim, n_options, hidden_dim)
        self.global_q = GlobalGRUQNetwork(obs_dim, n_actions, n_options, hidden_dim)

    def get_belief(
        self,
        evidence_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return self.belief_model(evidence_seq, lengths=lengths)

    def q_values(
        self,
        obs: torch.Tensor,
        marginals: list[torch.Tensor] | None = None,
        global_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.method == "random_policy":
            return torch.zeros(obs.shape[0], self.n_options, device=obs.device)
        if self.method == "global_gru":
            if global_hidden is None:
                global_hidden = self.global_q.initial_hidden(obs.shape[0], obs.device)
            return self.global_q(obs, global_hidden)
        if marginals is None:
            marginals = uniform_marginals(self.factor_modes, obs.shape[0], obs.device)
        if self.method == "flat_latent" or self.method == "oracle_belief":
            return self.flat_q(obs, marginals)
        return self.factor_q(obs, marginals)
