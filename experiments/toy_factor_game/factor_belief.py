"""
Latent factor belief model: q_φ(Z_F | h_t, H_l).
Factor graph posterior with factor-factor dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class FactorPotential(nn.Module):
    def __init__(self, hidden_dim: int, n_modes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class PairwisePotential(nn.Module):
    def __init__(self, hidden_dim: int, n_modes_i: int, n_modes_j: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modes_i * n_modes_j),
        )
        self.n_modes_i = n_modes_i
        self.n_modes_j = n_modes_j

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.net(h)
        return logits.view(-1, self.n_modes_i, self.n_modes_j)


class FactorBeliefModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_factors: int,
        factor_modes: list[int],
        hidden_dim: int = 128,
        pairwise_pairs: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.factor_modes = factor_modes
        self.hidden_dim = hidden_dim

        self.history_encoder = nn.GRU(
            input_size=obs_dim + n_actions + n_actions,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.unary_potentials = nn.ModuleList([
            FactorPotential(hidden_dim, fm) for fm in factor_modes
        ])

        self.pairwise_pairs = pairwise_pairs or []
        self.pairwise_potentials = nn.ModuleList([
            PairwisePotential(hidden_dim, factor_modes[i], factor_modes[j])
            for i, j in self.pairwise_pairs
        ])
        self.n_bp_iters = 3

    def encode_history(
        self,
        obs_seq: torch.Tensor,
        ego_act_seq: torch.Tensor,
        partner_act_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
        initial_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([obs_seq, ego_act_seq, partner_act_seq], dim=-1)
        h0 = initial_hidden
        if h0 is not None and h0.dim() == 2:
            h0 = h0.unsqueeze(0)
        if lengths is not None:
            packed = pack_padded_sequence(
                x,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h = self.history_encoder(packed, h0)
        else:
            _, h = self.history_encoder(x, h0)
        return h.squeeze(0)

    def step_history(
        self,
        obs_t: torch.Tensor,
        ego_act_t: torch.Tensor,
        partner_act_t: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if ego_act_t.dim() == 1:
            ego_act_t = ego_act_t.unsqueeze(0)
        if partner_act_t.dim() == 1:
            partner_act_t = partner_act_t.unsqueeze(0)
        x = torch.cat([obs_t, ego_act_t, partner_act_t], dim=-1).unsqueeze(1)
        h0 = hidden
        if h0 is not None and h0.dim() == 2:
            h0 = h0.unsqueeze(0)
        _, h = self.history_encoder(x, h0)
        return h

    def _marginals_from_h(self, h: torch.Tensor) -> list[torch.Tensor]:
        if h.dim() == 3:
            h = h.squeeze(0)
        marginals = []
        unary_logits = [pot(h) for pot in self.unary_potentials]

        if not self.pairwise_pairs:
            for logits in unary_logits:
                marginals.append(F.softmax(logits, dim=-1))
            return marginals

        msgs_to_j = [
            torch.zeros_like(unary_logits[j])
            for i, j in self.pairwise_pairs
        ]
        msgs_to_i = [
            torch.zeros_like(unary_logits[i])
            for i, j in self.pairwise_pairs
        ]

        for _iteration in range(self.n_bp_iters):
            new_msgs_to_j = []
            new_msgs_to_i = []
            for pair_idx, (i, j) in enumerate(self.pairwise_pairs):
                pairwise = self.pairwise_potentials[pair_idx](h)

                log_belief_i = unary_logits[i].clone()
                log_belief_j = unary_logits[j].clone()
                for other_idx, (oi, oj) in enumerate(self.pairwise_pairs):
                    if other_idx == pair_idx:
                        continue
                    if oj == i:
                        log_belief_i = log_belief_i + msgs_to_j[other_idx]
                    elif oi == i:
                        log_belief_i = log_belief_i + msgs_to_i[other_idx]
                    if oj == j:
                        log_belief_j = log_belief_j + msgs_to_j[other_idx]
                    elif oi == j:
                        log_belief_j = log_belief_j + msgs_to_i[other_idx]

                msg_i_to_j = torch.logsumexp(
                    log_belief_i.unsqueeze(-1) + pairwise, dim=-2
                )
                msg_j_to_i = torch.logsumexp(
                    log_belief_j.unsqueeze(-2) + pairwise, dim=-1
                )
                new_msgs_to_j.append(msg_i_to_j - msg_i_to_j.logsumexp(dim=-1, keepdim=True))
                new_msgs_to_i.append(msg_j_to_i - msg_j_to_i.logsumexp(dim=-1, keepdim=True))
            msgs_to_j = new_msgs_to_j
            msgs_to_i = new_msgs_to_i

        for node_idx, logits in enumerate(unary_logits):
            log_belief = logits.clone()
            for pair_idx, (i, j) in enumerate(self.pairwise_pairs):
                if j == node_idx:
                    log_belief = log_belief + msgs_to_j[pair_idx]
                elif i == node_idx:
                    log_belief = log_belief + msgs_to_i[pair_idx]
            marginals.append(F.softmax(log_belief, dim=-1))
        return marginals

    def forward(
        self,
        obs_seq: torch.Tensor,
        ego_act_seq: torch.Tensor,
        partner_act_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        h = self.encode_history(obs_seq, ego_act_seq, partner_act_seq, lengths=lengths)
        return self._marginals_from_h(h)

    def get_entropy(self, marginals: list[torch.Tensor]) -> torch.Tensor:
        entropies = []
        for m in marginals:
            ent = -(m * (m + 1e-8).log()).sum(dim=-1)
            entropies.append(ent)
        return torch.stack(entropies, dim=-1)

    def predict_factor_modes(self, marginals: list[torch.Tensor]) -> list[torch.Tensor]:
        return [m.argmax(dim=-1) for m in marginals]
