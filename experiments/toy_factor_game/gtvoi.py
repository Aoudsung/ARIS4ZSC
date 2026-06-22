"""
G-TVOI and MI option scoring for the toy factor game.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, belief_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, belief_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, belief_features], dim=-1)
        return self.net(x).squeeze(-1)


class OptionQNetwork(nn.Module):
    def __init__(self, obs_dim: int, belief_dim: int, n_options: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + belief_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_options),
        )

    def forward(self, obs: torch.Tensor, belief_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, belief_features], dim=-1)
        return self.net(x)


def belief_to_features(marginals: list[torch.Tensor]) -> torch.Tensor:
    flat = torch.cat(marginals, dim=-1)
    entropies = []
    for m in marginals:
        ent = -(m * (m + 1e-8).log()).sum(dim=-1, keepdim=True)
        entropies.append(ent)
    ent_vec = torch.cat(entropies, dim=-1)
    n_unresolved = (ent_vec > 0.3).float().sum(dim=-1, keepdim=True)
    return torch.cat([flat, ent_vec, n_unresolved], dim=-1)


def compute_belief_dim(factor_modes: list[int]) -> int:
    return sum(factor_modes) + len(factor_modes) + 1


def compute_gtvoi(
    value_net: ValueNetwork,
    obs: torch.Tensor,
    current_belief_features: torch.Tensor,
    simulated_belief_features: torch.Tensor,
) -> torch.Tensor:
    v_current = value_net(obs, current_belief_features)
    v_after = value_net(obs, simulated_belief_features)
    return v_after - v_current


def compute_mi(
    marginals_before: list[torch.Tensor],
    marginals_after: list[torch.Tensor],
) -> torch.Tensor:
    mi = torch.zeros(marginals_before[0].shape[0], device=marginals_before[0].device)
    for mb, ma in zip(marginals_before, marginals_after):
        ent_before = -(mb * (mb + 1e-8).log()).sum(dim=-1)
        ent_after = -(ma * (ma + 1e-8).log()).sum(dim=-1)
        mi += (ent_before - ent_after).clamp(min=0)
    return mi


class OptionSelector:
    def __init__(
        self,
        q_net: OptionQNetwork,
        value_net: ValueNetwork,
        alpha: float = 1.0,
        beta: float = 0.5,
        mode: str = "gtvoi",
    ):
        self.q_net = q_net
        self.value_net = value_net
        self.alpha = alpha
        self.beta = beta
        self.mode = mode

    def scores(
        self,
        obs: torch.Tensor,
        belief_features: torch.Tensor,
        simulated_beliefs: list[torch.Tensor] | None = None,
        option_costs: torch.Tensor | None = None,
        marginals_before: list[torch.Tensor] | None = None,
        simulated_marginals: list[list[torch.Tensor]] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_values = self.q_net(obs, belief_features)

        if self.mode == "passive":
            if option_costs is not None:
                scores = q_values - self.beta * option_costs
            else:
                scores = q_values
        elif self.mode == "random":
            scores = torch.zeros_like(q_values)
        elif self.mode == "gtvoi" and simulated_beliefs is not None and option_costs is not None:
            n_options = q_values.shape[-1]
            gtvoi_scores = torch.zeros_like(q_values)
            for o in range(n_options):
                gtvoi_scores[:, o] = compute_gtvoi(
                    self.value_net, obs, belief_features,
                    simulated_beliefs[o],
                ).detach()
            scores = q_values + self.alpha * gtvoi_scores - self.beta * option_costs
        elif self.mode == "mi" and marginals_before is not None and simulated_marginals is not None:
            n_options = q_values.shape[-1]
            mi_scores = torch.zeros_like(q_values)
            for o in range(n_options):
                mi_scores[:, o] = compute_mi(marginals_before, simulated_marginals[o]).detach()
            if option_costs is not None:
                scores = q_values + self.alpha * mi_scores - self.beta * option_costs
            else:
                scores = q_values + self.alpha * mi_scores
        else:
            scores = q_values

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.bool(), -1e9)
        return scores

    def select(
        self,
        obs: torch.Tensor,
        belief_features: torch.Tensor,
        simulated_beliefs: list[torch.Tensor] | None = None,
        option_costs: torch.Tensor | None = None,
        marginals_before: list[torch.Tensor] | None = None,
        simulated_marginals: list[list[torch.Tensor]] | None = None,
        valid_mask: torch.Tensor | None = None,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.scores(
            obs,
            belief_features,
            simulated_beliefs=simulated_beliefs,
            option_costs=option_costs,
            marginals_before=marginals_before,
            simulated_marginals=simulated_marginals,
            valid_mask=valid_mask,
        )
        if self.mode == "random":
            dist = torch.distributions.Categorical(logits=scores)
            selected = dist.sample()
            return selected, dist.log_prob(selected), dist.entropy()

        if deterministic:
            selected = scores.argmax(dim=-1)
            log_probs = F.log_softmax(scores / max(temperature, 1e-6), dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            return selected, log_probs.gather(1, selected.unsqueeze(1)).squeeze(1), entropy

        dist = torch.distributions.Categorical(logits=scores / max(temperature, 1e-6))
        selected = dist.sample()
        return selected, dist.log_prob(selected), dist.entropy()
