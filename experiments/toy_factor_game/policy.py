"""
Belief-conditioned Bayes-adaptive controller for the toy factor game.
Unified training with L_response + L_value + L_control + KL + sparsity + calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .factor_belief import FactorBeliefModel
from .gtvoi import OptionQNetwork, ValueNetwork, belief_to_features, compute_belief_dim


def brier_calibration_loss(
    marginals: list[torch.Tensor],
    factor_labels: torch.Tensor,
    factor_modes: list[int],
    gt_mask: list[bool] | None = None,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=factor_labels.device)
    count = 0
    for i, m in enumerate(marginals):
        if gt_mask is not None and not gt_mask[i]:
            continue
        labels_i = factor_labels[:, i]
        one_hot = F.one_hot(labels_i, num_classes=factor_modes[i]).float()
        loss = loss + F.mse_loss(m, one_hot)
        count += 1
    if count == 0:
        return loss
    return loss / count


def factor_activity_loss(
    marginals: list[torch.Tensor],
    factor_modes: list[int],
    sparsity_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    entropies = []
    for m in marginals:
        entropies.append(-(m * (m + 1e-8).log()).sum(dim=-1))
    entropy_tensor = torch.stack(entropies, dim=-1)
    max_entropies = torch.tensor(
        [float(torch.log(torch.tensor(n_modes, dtype=torch.float32))) for n_modes in factor_modes],
        device=entropy_tensor.device,
    )
    activity = 1.0 - entropy_tensor / max_entropies.clamp_min(1e-8)
    if sparsity_weights is not None:
        weights = sparsity_weights.to(device=entropy_tensor.device, dtype=activity.dtype).view(1, -1)
        activity = activity * weights
    return activity.mean()


class ResponsePredictor(nn.Module):
    def __init__(self, hidden_dim: int, n_partner_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_partner_actions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class BeliefTransitionModel(nn.Module):
    def __init__(
        self,
        belief_dim: int,
        n_options: int,
        factor_modes: list[int],
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_options = n_options
        self.factor_modes = factor_modes
        self.net = nn.Sequential(
            nn.Linear(belief_dim + n_options, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sum(factor_modes)),
        )

    def forward_marginals(
        self, belief_features: torch.Tensor, option_onehot: torch.Tensor
    ) -> list[torch.Tensor]:
        logits = self.net(torch.cat([belief_features, option_onehot], dim=-1))
        marginals = []
        offset = 0
        for n_modes in self.factor_modes:
            factor_logits = logits[:, offset:offset + n_modes]
            marginals.append(F.softmax(factor_logits, dim=-1))
            offset += n_modes
        return marginals

    def forward(self, belief_features: torch.Tensor, option_onehot: torch.Tensor) -> torch.Tensor:
        return belief_to_features(self.forward_marginals(belief_features, option_onehot))

    def forward_all_options(
        self, belief_features: torch.Tensor
    ) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
        batch_size = belief_features.shape[0]
        option_onehot = F.one_hot(
            torch.arange(self.n_options, device=belief_features.device),
            num_classes=self.n_options,
        ).float()
        option_onehot = option_onehot.unsqueeze(0).expand(batch_size, -1, -1)
        belief_expanded = belief_features.unsqueeze(1).expand(-1, self.n_options, -1)
        logits = self.net(
            torch.cat([belief_expanded, option_onehot], dim=-1).reshape(
                batch_size * self.n_options, -1
            )
        ).view(batch_size, self.n_options, -1)

        per_option_marginals = []
        per_option_features = []
        for option_id in range(self.n_options):
            marginals = []
            offset = 0
            for n_modes in self.factor_modes:
                factor_logits = logits[:, option_id, offset:offset + n_modes]
                marginals.append(F.softmax(factor_logits, dim=-1))
                offset += n_modes
            per_option_marginals.append(marginals)
            per_option_features.append(belief_to_features(marginals))
        return per_option_marginals, per_option_features


class ActiveFactorAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_options: int,
        n_factors: int,
        factor_modes: list[int],
        hidden_dim: int = 128,
        pairwise_pairs: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_options = n_options
        self.factor_modes = factor_modes

        belief_dim = compute_belief_dim(factor_modes)

        self.belief_model = FactorBeliefModel(
            obs_dim=obs_dim,
            n_actions=n_actions,
            n_factors=n_factors,
            factor_modes=factor_modes,
            hidden_dim=hidden_dim,
            pairwise_pairs=pairwise_pairs,
        )

        self.q_net = OptionQNetwork(obs_dim, belief_dim, n_options, hidden_dim)
        self.value_net = ValueNetwork(obs_dim, belief_dim, hidden_dim)
        self.belief_transition = BeliefTransitionModel(
            belief_dim=belief_dim,
            n_options=n_options,
            factor_modes=factor_modes,
            hidden_dim=max(64, hidden_dim // 2),
        )
        self.response_predictor = ResponsePredictor(hidden_dim, n_actions)

    def get_belief(
        self, obs_seq: torch.Tensor, ego_act_seq: torch.Tensor, partner_act_seq: torch.Tensor
    ) -> list[torch.Tensor]:
        return self.belief_model(obs_seq, ego_act_seq, partner_act_seq)

    def get_option_logits(
        self, obs: torch.Tensor, belief_features: torch.Tensor
    ) -> torch.Tensor:
        return self.q_net(obs, belief_features)

    def compute_factor_losses(
        self,
        obs_seq: torch.Tensor,
        ego_act_seq: torch.Tensor,
        partner_act_seq: torch.Tensor,
        partner_actions_next: torch.Tensor,
        factor_labels: torch.Tensor,
        lengths: torch.Tensor | None = None,
        obs_at_eval: torch.Tensor | None = None,
        value_target: torch.Tensor | None = None,
        value_coef: float = 0.0,
        beta_kl: float = 0.01,
        gamma_sparsity: float = 0.1,
        mu_cal: float = 0.5,
        sparsity_weights: torch.Tensor | None = None,
        gt_mask: list[bool] | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self.belief_model.encode_history(
            obs_seq, ego_act_seq, partner_act_seq, lengths=lengths
        )
        marginals = self.belief_model._marginals_from_h(h)

        response_logits = self.response_predictor(h)
        l_response = F.cross_entropy(response_logits, partner_actions_next)

        l_kl = torch.tensor(0.0, device=obs_seq.device)
        for i, m in enumerate(marginals):
            n_modes = self.factor_modes[i]
            uniform = torch.ones_like(m) / n_modes
            kl = (m * ((m + 1e-8).log() - uniform.log())).sum(dim=-1).mean()
            l_kl += kl

        l_sparsity = factor_activity_loss(marginals, self.factor_modes, sparsity_weights)

        if factor_labels is not None:
            l_calibration = brier_calibration_loss(
                marginals, factor_labels, self.factor_modes, gt_mask=gt_mask
            )
        else:
            l_calibration = torch.tensor(0.0, device=obs_seq.device)

        belief_features = None
        if (
            value_coef > 0.0
            and obs_at_eval is not None
            and value_target is not None
        ) or return_intermediates:
            belief_features = belief_to_features(marginals)

        l_value = torch.tensor(0.0, device=obs_seq.device)
        if value_coef > 0.0 and obs_at_eval is not None and value_target is not None:
            v_pred = self.value_net(obs_at_eval, belief_features)
            l_value = F.mse_loss(v_pred, value_target)

        total = (
            l_response
            + beta_kl * l_kl
            + gamma_sparsity * l_sparsity
            + mu_cal * l_calibration
            + value_coef * l_value
        )

        result = {
            "total": total,
            "response": l_response,
            "kl": l_kl,
            "sparsity": l_sparsity,
            "calibration": l_calibration,
            "belief_value": l_value,
        }
        if return_intermediates:
            result.update({
                "marginals": marginals,
                "h": h,
                "belief_features": belief_features,
            })
        return result

    def compute_losses(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self.compute_factor_losses(*args, **kwargs)
