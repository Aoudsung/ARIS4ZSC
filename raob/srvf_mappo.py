"""
V0-anchored posterior-predictive SRVF-MAPPO for classic Overcooked.

Classic Overcooked execution contract:
- Two-agent cooperative classic Overcooked-AI style benchmark.
- Primitive action count A defaults to 6: stay, interact, north, south, east, west
  under the usual environment wrapper convention. The implementation does not
  assume action names; it assumes only A=6 unless overridden.
- Public affordance chart g is the canonical SRVF public state and, with the
  default response summary, Dz == g_dim and Delta z = g_next - g.
- No partner_id, target return, or target action label is accepted by V0,
  SRVFBelief, the alpha calibration rule, or the online actor.
- No state_index_for_g lookup is used anywhere. NeuralSRVFHeads replaces the
  finite-reservoir lookup with a continuous scorer.

# ── Theorem-to-code correspondence table ──

| Theorem object        | Equation ref | PyTorch variable / method | Shape | Notes |
|-----------------------|--------------|---------------------------|-------|-------|
| Public affordance chart | Eq.SCORE / Eq.LOGIT | `g` | `[B, Dz]` | Classic Overcooked public chart; default `Dz == g_dim`. |
| Ego observation | Eq.LOGIT | `obs_ego` | `[B, obs_dim]` | Execution-time local ego observation. |
| Global state | Eq.L_VALUE | `global_state` | `[B, global_state_dim]` | CTDE critic input; not used by actor. |
| Phase feature | Eq.LOGIT | `phase` | `[B, phase_dim]` | User-supplied phase encoding; no option loss is introduced. |
| Partner-blind value prediction | V0 anchor | `V0EnsembleProtocol.predict(state_g).mean` | `[B]` | Thin interface only; must be partner-blind. |
| Partner-blind value uncertainty | V0 anchor | `V0EnsembleProtocol.predict(state_g).variance` | `[B]` | Thin interface only; diagnostic support. |
| Population residual advantage | Eq.SCORE | `NeuralSRVFHeads.forward(g).a0` | `[B, A]` | `A0(g,a) = E_p[Ā0_p(g,a)]`. |
| Value-factor loading | Eq.SCORE | `NeuralSRVFHeads.forward(g).u_c` | `[B, A, K]` | `U(g,a)` in posterior-predictive residual. |
| Population response | Eq.PREC_UPDATE / Eq.ETA_UPDATE | `NeuralSRVFHeads.forward(g).r0` | `[B, A, Dz]` | `z0(g,a)` in response model. |
| Response-factor loading | Eq.PREC_UPDATE / Eq.ETA_UPDATE | `NeuralSRVFHeads.forward(g).r_c` | `[B, A, Dz, K]` | `Rc(g,a)` in response model. |
| Source beta support | Eq.ALPHA_EB | `SRVFBelief.source_beta` | `[P_src, K]` | Used only for source-calibrated OOD support distance. |
| Posterior precision | Eq.PREC_UPDATE | `SRVFBelief.Lambda` | `[B, K, K]` | Batched belief precision; initialized from prior precision. |
| Posterior natural mean | Eq.ETA_UPDATE | `SRVFBelief.eta` | `[B, K]` | Batched natural parameter. |
| Posterior mean | Eq.MEAN | `SRVFBelief.mean()` | `[B, K]` | Solved with `torch.linalg.solve`, never inverse. |
| Posterior covariance | Eq.MEAN | `SRVFBelief.covariance()` | `[B, K, K]` | Solved with `torch.linalg.solve(I)`, with jitter. |
| Posterior covariance diagonal | Eq.LOGIT | `SRVFBelief.diag_covariance()` / `rollout_batch.belief_diag_cov` | `[B, K]` | Actor/critic belief uncertainty input. |
| Model-validity probability | Eq.SCORE / Eq.ALPHA_EB | `SRVFBelief.alpha` / `rollout_batch.belief_alpha` | `[B]` | Empirical-Bayes source-calibrated trust coefficient. |
| Response residual | Eq.ETA_UPDATE | `delta_z - r0_selected` | `[B, Dz]` | Uses only `(g,a,Delta z)` online. |
| Response MSE reliability | Eq.ALPHA_EB | `response_mse` | `[B]` | No target label or return. |
| Beta support distance | Eq.ALPHA_EB | `beta_support_dist` | `[B]` | Distance to source beta support. |
| Posterior contraction | Eq.ALPHA_EB | `posterior_contraction` | `[B]` | `1 - tr(Sigma_t)/tr(Sigma_0)`. |
| Posterior-predictive SRVF score | Eq.SCORE | `SRVFBelief.score(...)` / `SRVFBelief.score_from_posterior(...)` | `[B, A]` | `S(g,a,h)=A0+alpha*U@mu`; alpha=0 gives population fallback. |
| Continuation residual logits | Eq.LOGIT | `MAPPOActorCritic.continuation_head(actor_input)` | `[B, A]` | `C_theta(s,b,a)` learned by MAPPO. |
| Actor logits | Eq.LOGIT | `logits = base_logits + srvf_score / tau` | `[B, A]` | Exact formula; no extra bonus terms. |
| Legal action mask | Eq.LOGIT | `legal_mask` | `[B, A]` | Invalid actions receive a large negative logit. |
| Policy distribution | Eq.L_POLICY | `torch.distributions.Categorical(logits=logits)` | batch `[B]` over `A` | PPO uses belief-MDP policy. |
| PPO ratio | Eq.L_POLICY | `ratio = exp(new_log_prob - old_log_probs)` | `[B]` | Clipped surrogate. |
| Belief-MDP advantage | Eq.L_POLICY | `rollout_batch.advantages` | `[B]` | Detached in policy loss. |
| GAE target | Eq.L_VALUE | `rollout_batch.gae_targets` | `[B]` | Critic regression target. |
| Critic value | Eq.L_VALUE | `MAPPOActorCritic.value(...)` | `[B]` | Centralized value estimate. |
| Source IRF state batch | Eq.L_RESPONSE / Eq.L_RESIDUAL | `source_batch.g` | `[B, Dz]` | Static pre-built IRF table slice. |
| Source beta batch | Eq.L_RESPONSE / Eq.L_RESIDUAL | `source_batch.beta` | `[B, K]` | Fitted source-partner factor. |
| Response target | Eq.L_RESPONSE | `source_batch.delta_z_target` | `[B, A, Dz]` | Static source IRF response target. |
| Value residual target | Eq.L_RESIDUAL | `source_batch.a_bar_target` | `[B, A]` | Static source IRF centered V0-residual target. |
| Response likelihood prediction | Eq.L_RESPONSE | `delta_z_pred = r0 + einsum('badk,bk->bad', r_c,beta)` | `[B, A, Dz]` | Gaussian response channel. |
| Value residual prediction | Eq.L_RESIDUAL | `a_pred = a0 + einsum('bak,bk->ba', u_c,beta)` | `[B, A]` | Gaussian residual channel; chosen instead of rank channel. |
| Unified loss | Eq.ELBO | `UnifiedLoss.compute(...)` | scalar | Negative ELBO estimator with MAPPO and source likelihood terms. |
| Gradient isolation audit | Eq.GRAD_ISO | `gradient_audit()` | `dict[str, bool]` | Verifies allowed and blocked paths. |

Equation labels used in code comments:
- Eq.SCORE: `S(g,a,h)=A0(g,a)+alpha_t*U(g,a)^T mu_t`.
- Eq.LOGIT: `ell(s,b,a)=C_theta(s,b,a)+S(g,a,h)/tau`.
- Eq.PREC_UPDATE: `Lambda += Rc^T Sigma_z^{-1} Rc`.
- Eq.ETA_UPDATE: `eta += Rc^T Sigma_z^{-1}(Delta z - R0)`.
- Eq.MEAN: `mu = solve(Lambda+jitter*I, eta)`.
- Eq.ALPHA_EB: empirical-Bayes source-calibrated alpha from response-only reliability.
- Eq.L_POLICY: PPO clipped surrogate on belief-MDP advantages.
- Eq.L_VALUE: critic MSE against GAE targets.
- Eq.L_RESPONSE: Gaussian response reconstruction likelihood.
- Eq.L_RESIDUAL: Gaussian V0-residual likelihood. We choose L_A, not L_rank,
  because the classic Overcooked IRF table stores scalar centered residual labels;
  adding L_rank would double-count the same observation channel.
- Eq.ELBO: negative control-as-inference/source-likelihood ELBO estimator.
- Eq.GRAD_ISO: explicit gradient isolation contract.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ── Imports complete ──


# ── MODULE 1: V0Ensemble interface (thin wrapper, no reimplementation) ──

@dataclass(frozen=True)
class V0Prediction:
    """Mean and uncertainty returned by the existing partner-blind V0 ensemble."""

    mean: torch.Tensor
    variance: torch.Tensor


class V0EnsembleProtocol(Protocol):
    """Thin protocol matching `value.py`: no partner_id, no beta input."""

    def predict(self, state_g: torch.Tensor, *, device: str | torch.device | None = None) -> V0Prediction:
        """Return partner-blind V0 prediction for `state_g` with shape `[B, Dz]`."""


@dataclass(frozen=True)
class V0TrainingBatch:
    state_g: torch.Tensor
    returns: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_g.ndim != 2:
            raise ValueError("state_g must have shape [N,Dz]")
        if self.returns.ndim != 1:
            raise ValueError("returns must have shape [N]")
        if self.state_g.shape[0] != self.returns.shape[0]:
            raise ValueError("state_g and returns disagree on N")
        if self.state_g.shape[0] == 0:
            raise ValueError("V0TrainingBatch must be nonempty")


class V0MLP(nn.Module):
    """Partner-blind V0 approximator over public chart states."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(input_dim)
        for hidden in hidden_dims:
            layers.append(nn.Linear(current, int(hidden)))
            layers.append(nn.ReLU())
            current = int(hidden)
        layers.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_g: torch.Tensor) -> torch.Tensor:
        return self.net(state_g).squeeze(-1)


@dataclass
class V0Ensemble:
    """Partner-blind V0 ensemble with input/target normalization."""

    models: tuple[V0MLP, ...]
    state_mean: torch.Tensor
    state_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        return int(self.state_mean.shape[0])

    def to(self, device: str | torch.device) -> "V0Ensemble":
        target = torch.device(device)
        for model in self.models:
            model.to(target)
        self.state_mean = self.state_mean.to(target)
        self.state_std = self.state_std.to(target)
        self.target_mean = self.target_mean.to(target)
        self.target_std = self.target_std.to(target)
        return self

    def eval(self) -> "V0Ensemble":
        for model in self.models:
            model.eval()
        return self

    def predict(
        self,
        state_g: torch.Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> V0Prediction:
        if state_g.ndim == 1:
            state_g = state_g.unsqueeze(0)
        if state_g.ndim != 2 or state_g.shape[1] != self.input_dim:
            raise ValueError("state_g must have shape [N,input_dim]")
        target_device = self.state_mean.device if device is None else torch.device(device)
        self.to(target_device).eval()
        states = state_g.to(device=target_device, dtype=torch.float32)
        normalized = (states - self.state_mean) / self.state_std
        with torch.no_grad():
            predictions = torch.stack(
                [
                    model(normalized) * self.target_std + self.target_mean
                    for model in self.models
                ],
                dim=0,
            )
        mean = predictions.mean(dim=0)
        variance = (
            predictions.var(dim=0, unbiased=False)
            if predictions.shape[0] > 1
            else torch.zeros_like(mean)
        )
        return V0Prediction(mean=mean.detach().cpu(), variance=variance.detach().cpu())

    def state_dict_payload(self) -> dict[str, Any]:
        hidden_dims = tuple(int(dim) for dim in self.metadata.get("hidden_dims", (256, 256)))
        return {
            "kind": "v0_ensemble",
            "input_dim": self.input_dim,
            "hidden_dims": hidden_dims,
            "state_dicts": [model.cpu().state_dict() for model in self.models],
            "state_mean": self.state_mean.detach().cpu(),
            "state_std": self.state_std.detach().cpu(),
            "target_mean": self.target_mean.detach().cpu(),
            "target_std": self.target_std.detach().cpu(),
            "metadata": dict(self.metadata),
        }


# ── shared utilities ──

@dataclass(frozen=True)
class SRVFHeadOutput:
    a0: torch.Tensor
    u_c: torch.Tensor
    r0: torch.Tensor
    r_c: torch.Tensor


@dataclass(frozen=True)
class ActorCriticOutput:
    dist: Categorical
    logits: torch.Tensor
    value: torch.Tensor
    base_logits: torch.Tensor
    srvf_score: torch.Tensor


@dataclass(frozen=True)
class RolloutBatch:
    obs_ego: torch.Tensor
    global_state: torch.Tensor
    g: torch.Tensor
    phase: torch.Tensor
    belief_mean: torch.Tensor
    belief_diag_cov: torch.Tensor
    belief_alpha: torch.Tensor
    legal_mask: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    gae_targets: torch.Tensor
    monitor: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceBatch:
    g: torch.Tensor
    beta: torch.Tensor
    delta_z_target: torch.Tensor
    a_bar_target: torch.Tensor
    action_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class IRFTable:
    """Classic source interventional table used only to build `SourceBatch`.

    Layout is state-major:
    - `state_g`: `[S, Dz]`
    - `delta_z`: `[S, A, P, Dz]`
    - `a_raw`: `[S, A, P]`, centered V0-residual target
    - `valid_mask`: `[S, A, P]`
    """

    state_g: torch.Tensor
    delta_z: torch.Tensor
    a_raw: torch.Tensor
    valid_mask: torch.Tensor
    phase_ids: tuple[str, ...] = field(default_factory=tuple)
    partner_ids: tuple[str, ...] = field(default_factory=tuple)
    action_ids: tuple[int, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_g.ndim != 2:
            raise ValueError("state_g must have shape [S,Dz]")
        if self.delta_z.ndim != 4:
            raise ValueError("delta_z must have shape [S,A,P,Dz]")
        if self.a_raw.shape != self.delta_z.shape[:3]:
            raise ValueError("a_raw must have shape [S,A,P]")
        if self.valid_mask.shape != self.delta_z.shape[:3]:
            raise ValueError("valid_mask must have shape [S,A,P]")
        if self.state_g.shape[0] != self.delta_z.shape[0]:
            raise ValueError("state_g and delta_z disagree on S")
        if self.state_g.shape[1] != self.delta_z.shape[3]:
            raise ValueError("default classic contract requires Dz == g_dim")
        if self.valid_mask.dtype != torch.bool:
            object.__setattr__(self, "valid_mask", self.valid_mask.to(dtype=torch.bool))
        if not self.phase_ids:
            object.__setattr__(
                self,
                "phase_ids",
                tuple("unknown" for _ in range(self.num_states)),
            )
        if len(self.phase_ids) != self.num_states:
            raise ValueError("phase_ids length must match S")
        if not self.partner_ids:
            object.__setattr__(
                self,
                "partner_ids",
                tuple(f"partner_{idx}" for idx in range(self.num_partners)),
            )
        if len(self.partner_ids) != self.num_partners:
            raise ValueError("partner_ids length must match P")
        if not self.action_ids:
            object.__setattr__(self, "action_ids", tuple(range(self.num_actions)))
        if len(self.action_ids) != self.num_actions:
            raise ValueError("action_ids length must match A")

    @property
    def num_states(self) -> int:
        return int(self.delta_z.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.delta_z.shape[1])

    @property
    def num_partners(self) -> int:
        return int(self.delta_z.shape[2])

    @property
    def dz_dim(self) -> int:
        return int(self.delta_z.shape[3])

    @classmethod
    def from_tensors(
        cls,
        *,
        state_g: torch.Tensor,
        delta_z: torch.Tensor,
        a_raw: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        phase_ids: Optional[Sequence[str]] = None,
        partner_ids: Optional[Sequence[str]] = None,
        action_ids: Optional[Sequence[int]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "IRFTable":
        if delta_z.ndim != 4:
            raise ValueError("delta_z must have shape [S,A,P,Dz]")
        mask = (
            torch.ones(delta_z.shape[:3], dtype=torch.bool, device=delta_z.device)
            if valid_mask is None
            else valid_mask
        )
        return cls(
            state_g=state_g,
            delta_z=delta_z,
            a_raw=a_raw,
            valid_mask=mask,
            phase_ids=tuple(str(x) for x in phase_ids) if phase_ids is not None else (),
            partner_ids=tuple(str(x) for x in partner_ids) if partner_ids is not None else (),
            action_ids=tuple(int(x) for x in action_ids) if action_ids is not None else (),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class SourceFactorInit:
    beta_source: torch.Tensor
    prior_mean: torch.Tensor
    prior_covariance: torch.Tensor
    partner_ids: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def centered_action_residual(q_raw: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Center `[S,A,P]` values across valid actions for each state/partner."""
    if q_raw.ndim != 3 or valid_mask.shape != q_raw.shape:
        raise ValueError("q_raw and valid_mask must have shape [S,A,P]")
    mask = valid_mask.to(dtype=torch.bool)
    weights = mask.to(dtype=q_raw.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (q_raw * weights).sum(dim=1, keepdim=True) / denom
    return torch.where(mask, q_raw - mean, torch.zeros_like(q_raw))


def _mean_over_partners(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim == 3:
        weights = mask.to(dtype=values.dtype)
        total = (values * weights).sum(dim=2)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return total / denom
    if values.ndim == 4:
        weights = mask.to(dtype=values.dtype).unsqueeze(-1)
        total = (values * weights).sum(dim=2)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return total / denom
    raise ValueError("values must have shape [S,A,P] or [S,A,P,D]")


def _pad_factor_columns(beta: torch.Tensor, factor_dim: int) -> torch.Tensor:
    if beta.shape[1] == factor_dim:
        return beta
    pad = torch.zeros(
        beta.shape[0],
        factor_dim - beta.shape[1],
        dtype=beta.dtype,
        device=beta.device,
    )
    return torch.cat([beta, pad], dim=1)


def initialize_source_beta(
    table: IRFTable,
    *,
    factor_dim: int = 4,
    response_weight: float = 1.0,
    value_weight: float = 1.0,
    covariance_jitter: float = 1e-3,
) -> SourceFactorInit:
    """Initialize source-partner SRVF factors from source IRF descriptors.

    This is a data initializer for the neural SRVF-MAPPO path, not an old
    finite-reservoir checkpoint. It produces only source beta support and a
    Gaussian prior for `SRVFBelief`.
    """
    if factor_dim <= 0:
        raise ValueError("factor_dim must be positive")
    if table.num_partners < 2:
        raise ValueError("source beta initialization requires at least two partners")
    if response_weight < 0.0 or value_weight < 0.0:
        raise ValueError("response_weight and value_weight must be nonnegative")
    if response_weight == 0.0 and value_weight == 0.0:
        raise ValueError("at least one descriptor weight must be positive")
    if covariance_jitter <= 0.0:
        raise ValueError("covariance_jitter must be positive")

    delta_z = table.delta_z.to(dtype=torch.float32)
    a_raw = table.a_raw.to(dtype=torch.float32)
    valid_mask = table.valid_mask.to(dtype=torch.bool)
    a0 = _mean_over_partners(a_raw, valid_mask)
    r0 = _mean_over_partners(delta_z, valid_mask)
    value_residual = torch.where(valid_mask, a_raw - a0.unsqueeze(2), torch.zeros_like(a_raw))
    response_residual = torch.where(
        valid_mask.unsqueeze(-1),
        delta_z - r0.unsqueeze(2),
        torch.zeros_like(delta_z),
    )
    value_descriptor = value_residual.permute(2, 0, 1).reshape(table.num_partners, -1)
    response_descriptor = response_residual.permute(2, 0, 1, 3).reshape(table.num_partners, -1)
    descriptors = torch.cat(
        [
            float(value_weight) * value_descriptor,
            float(response_weight) * response_descriptor,
        ],
        dim=1,
    )
    centered = descriptors - descriptors.mean(dim=0, keepdim=True)
    if bool((centered.square().sum() <= 1e-12).item()):
        beta = torch.zeros(table.num_partners, factor_dim, dtype=torch.float32)
        singular_values = torch.empty(0, dtype=torch.float32)
        rank = 0
    else:
        u, singular_values, _vh = torch.linalg.svd(centered, full_matrices=False)
        threshold = singular_values.max().clamp_min(1e-12) * 1e-6
        rank = int((singular_values > threshold).sum().item())
        used = min(factor_dim, u.shape[1], singular_values.shape[0])
        beta = u[:, :used] * singular_values[:used].unsqueeze(0)
        beta = _pad_factor_columns(beta, factor_dim).to(dtype=torch.float32)
        beta = beta - beta.mean(dim=0, keepdim=True)
        beta = beta / beta.std(dim=0, unbiased=False).clamp_min(1e-6)

    prior_mean = beta.mean(dim=0)
    if beta.shape[0] <= 1:
        prior_covariance = torch.eye(factor_dim, dtype=torch.float32)
    elif factor_dim == 1:
        prior_covariance = beta[:, 0].var(unbiased=False).reshape(1, 1)
    else:
        prior_covariance = torch.cov(beta.transpose(0, 1))
    prior_covariance = prior_covariance.to(dtype=torch.float32) + float(covariance_jitter) * torch.eye(
        factor_dim,
        dtype=torch.float32,
    )
    diagnostics = {
        "initializer": "source_irf_descriptor_svd",
        "shared_factor_rank": rank,
        "singular_values": [float(x) for x in singular_values.detach().cpu()],
        "descriptor_feature_count": int(descriptors.shape[1]),
        "value_descriptor_rms": float(value_descriptor.square().mean().sqrt().item()),
        "response_descriptor_rms": float(response_descriptor.square().mean().sqrt().item()),
        "valid_coverage": float(valid_mask.to(dtype=torch.float32).mean().item()),
        "legacy_checkpoint_created": False,
        "state_index_lookup_used": False,
    }
    return SourceFactorInit(
        beta_source=beta,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
        partner_ids=tuple(table.partner_ids),
        diagnostics=diagnostics,
    )


def source_table_to_batch(
    table: IRFTable,
    beta_source: torch.Tensor,
    *,
    state_indices: Optional[Sequence[int]] = None,
    partner_indices: Optional[Sequence[int]] = None,
    device: str | torch.device = "cpu",
) -> SourceBatch:
    """Flatten `(state, source_partner)` rows into one `SourceBatch`."""
    target_device = torch.device(device)
    if beta_source.ndim != 2 or beta_source.shape[0] != table.num_partners:
        raise ValueError("beta_source must have shape [P,K]")
    states = (
        torch.arange(table.num_states, dtype=torch.long)
        if state_indices is None
        else torch.as_tensor(tuple(int(x) for x in state_indices), dtype=torch.long)
    )
    partners = (
        torch.arange(table.num_partners, dtype=torch.long)
        if partner_indices is None
        else torch.as_tensor(tuple(int(x) for x in partner_indices), dtype=torch.long)
    )
    if states.numel() == 0 or partners.numel() == 0:
        raise ValueError("state_indices and partner_indices must be nonempty")
    if bool(((states < 0) | (states >= table.num_states)).any().item()):
        bad = int(states[((states < 0) | (states >= table.num_states)).nonzero(as_tuple=False)[0]].item())
        raise IndexError(f"state index out of range: {bad}")
    if bool(((partners < 0) | (partners >= table.num_partners)).any().item()):
        bad = int(partners[((partners < 0) | (partners >= table.num_partners)).nonzero(as_tuple=False)[0]].item())
        raise IndexError(f"partner index out of range: {bad}")

    # Preserve the original state-major row order:
    # [(s0,p0), (s0,p1), ..., (s1,p0), ...].
    row_states = states.repeat_interleave(partners.numel())
    row_partners = partners.repeat(states.numel())
    state_index = row_states.to(device=table.state_g.device)
    delta_state_index = row_states.to(device=table.delta_z.device)
    delta_partner_index = row_partners.to(device=table.delta_z.device)
    return SourceBatch(
        g=table.state_g.index_select(0, state_index).to(device=target_device, dtype=torch.float32),
        beta=beta_source.index_select(0, row_partners.to(device=beta_source.device)).to(device=target_device, dtype=torch.float32),
        delta_z_target=table.delta_z[delta_state_index, :, delta_partner_index, :].to(device=target_device, dtype=torch.float32),
        a_bar_target=table.a_raw[
            row_states.to(device=table.a_raw.device),
            :,
            row_partners.to(device=table.a_raw.device),
        ].to(device=target_device, dtype=torch.float32),
        action_mask=table.valid_mask[
            row_states.to(device=table.valid_mask.device),
            :,
            row_partners.to(device=table.valid_mask.device),
        ].to(device=target_device, dtype=torch.bool),
    )


def iter_source_batches(
    table: IRFTable,
    beta_source: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[SourceBatch]:
    """Return deterministic mini-batches over flattened `(state, partner)` rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total = table.num_states * table.num_partners
    order = torch.arange(total)
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed))
        order = order.index_select(0, torch.randperm(total, generator=generator))
    batches: list[SourceBatch] = []
    target_device = torch.device(device)
    for start in range(0, total, int(batch_size)):
        flat = order[start : start + int(batch_size)]
        states = torch.div(flat, table.num_partners, rounding_mode="floor")
        partners = flat.remainder(table.num_partners)
        # Build rows one-to-one instead of cartesian product.
        rows_g = table.state_g.index_select(0, states.to(device=table.state_g.device))
        rows_beta = beta_source.index_select(0, partners.to(device=beta_source.device))
        delta = table.delta_z[
            states.to(device=table.delta_z.device),
            :,
            partners.to(device=table.delta_z.device),
            :,
        ]
        a_raw = table.a_raw[
            states.to(device=table.a_raw.device),
            :,
            partners.to(device=table.a_raw.device),
        ]
        mask = table.valid_mask[
            states.to(device=table.valid_mask.device),
            :,
            partners.to(device=table.valid_mask.device),
        ]
        batches.append(
            SourceBatch(
                g=rows_g.to(device=target_device, dtype=torch.float32),
                beta=rows_beta.to(device=target_device, dtype=torch.float32),
                delta_z_target=delta.to(device=target_device, dtype=torch.float32),
                a_bar_target=a_raw.to(device=target_device, dtype=torch.float32),
                action_mask=mask.to(device=target_device, dtype=torch.bool),
            )
        )
    return batches


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    layers: list[nn.Module] = []
    current = int(input_dim)
    for _ in range(depth):
        layers.append(nn.Linear(current, int(hidden_dim)))
        layers.append(nn.Tanh())
        current = int(hidden_dim)
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise AssertionError(f"{name} contains NaN or Inf")


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    diff = (pred - target).square()
    if mask is None:
        return diff.mean()
    bool_mask = mask.to(dtype=torch.bool, device=pred.device)
    while bool_mask.ndim < diff.ndim:
        bool_mask = bool_mask.unsqueeze(-1)
    selected = diff.masked_select(bool_mask.expand_as(diff))
    if selected.numel() == 0:
        return diff.sum() * 0.0
    return selected.mean()


def _has_any_grad(params: tuple[nn.Parameter, ...]) -> bool:
    return any(param.grad is not None and bool(torch.isfinite(param.grad).all().item()) and float(param.grad.abs().sum().item()) > 0.0 for param in params)


def _grad_norm_from_autograd(loss: torch.Tensor, params: tuple[nn.Parameter, ...]) -> float:
    if not params:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().abs().sum().item())
    return total


# ── MODULE 2: NeuralSRVFHeads ──

class NeuralSRVFHeads(nn.Module):
    """Continuous SRVF heads for classic Overcooked public chart states.

    One forward pass produces all `a0`, `u_c`, `r0`, and `r_c`. There are exactly
    two trunks: one trunk for value-residual objects and one trunk for response
    objects. There is no exact reservoir lookup.
    """

    def __init__(
        self,
        g_dim: int,
        num_actions: int = 6,
        factor_dim: int = 4,
        hidden_dim: int = 128,
        trunk_depth: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if g_dim <= 0:
            raise ValueError("g_dim must be positive")
        if num_actions <= 1:
            raise ValueError("num_actions must be at least 2")
        if factor_dim <= 0:
            raise ValueError("factor_dim must be positive")
        self.g_dim = int(g_dim)
        self.num_actions = int(num_actions)
        self.factor_dim = int(factor_dim)
        self.device = torch.device(device)
        pair_dim = self.g_dim + self.num_actions
        self.value_trunk = _mlp(pair_dim, hidden_dim, hidden_dim, trunk_depth)
        self.value_head = nn.Linear(hidden_dim, 1 + self.factor_dim)
        self.response_trunk = _mlp(pair_dim, hidden_dim, hidden_dim, trunk_depth)
        self.response_head = nn.Linear(hidden_dim, self.g_dim + self.g_dim * self.factor_dim)
        self.register_buffer("action_eye", torch.eye(self.num_actions, dtype=torch.float32))
        self.to(self.device)

    def forward(self, g: torch.Tensor, a: Optional[torch.Tensor] = None) -> SRVFHeadOutput:
        """Return all action heads for `g`; optional `a` is accepted for API parity.

        Eq.SCORE: `a0` and `u_c` define `A0(g,a)` and `U(g,a)`.
        Eq.PREC_UPDATE / Eq.ETA_UPDATE: `r0` and `r_c` define the response model.
        """
        del a
        if g.ndim != 2 or g.shape[1] != self.g_dim:
            raise ValueError(f"g must have shape [B, {self.g_dim}]")
        g = g.to(device=self.device, dtype=torch.float32)
        batch = g.shape[0]
        action_one_hot = self.action_eye.to(device=g.device, dtype=g.dtype).unsqueeze(0).expand(batch, -1, -1)
        g_expanded = g.unsqueeze(1).expand(-1, self.num_actions, -1)
        pair = torch.cat([g_expanded, action_one_hot], dim=-1).reshape(batch * self.num_actions, -1)

        value_features = self.value_trunk(pair)
        value_raw = self.value_head(value_features).reshape(batch, self.num_actions, 1 + self.factor_dim)
        a0 = value_raw[..., 0]
        u_c = value_raw[..., 1:]

        response_features = self.response_trunk(pair)
        response_raw = self.response_head(response_features).reshape(
            batch,
            self.num_actions,
            self.g_dim + self.g_dim * self.factor_dim,
        )
        r0 = response_raw[..., : self.g_dim]
        r_c = response_raw[..., self.g_dim :].reshape(batch, self.num_actions, self.g_dim, self.factor_dim)
        return SRVFHeadOutput(a0=a0, u_c=u_c, r0=r0, r_c=r_c)


def self_test_NeuralSRVFHeads() -> None:
    B, A, K, Dz = 2, 6, 4, 8
    device = torch.device("cpu")
    heads = NeuralSRVFHeads(g_dim=Dz, num_actions=A, factor_dim=K, device=device)
    g = torch.randn(B, Dz, device=device)
    out = heads(g)
    assert out.a0.shape == (B, A)
    assert out.u_c.shape == (B, A, K)
    assert out.r0.shape == (B, A, Dz)
    assert out.r_c.shape == (B, A, Dz, K)
    _assert_finite("a0", out.a0)
    _assert_finite("u_c", out.u_c)
    _assert_finite("r0", out.r0)
    _assert_finite("r_c", out.r_c)


# ── MODULE 3: SRVFBelief ──

class SRVFBelief(nn.Module):
    """Batched Gaussian belief over beta plus empirical-Bayes SRVF validity alpha."""

    def __init__(
        self,
        factor_dim: int,
        dz_dim: int,
        *,
        prior_mean: Optional[torch.Tensor] = None,
        prior_covariance: Optional[torch.Tensor] = None,
        response_noise_var: float = 1.0,
        source_beta: Optional[torch.Tensor] = None,
        feature_mean: Optional[torch.Tensor] = None,
        feature_std: Optional[torch.Tensor] = None,
        alpha_low: float = 0.0,
        alpha_high: float = 1.0,
        reliability_tau: float = 0.0,
        ood_distance_threshold: float = float("inf"),
        ood_alpha_cap: float = 0.25,
        jitter: float = 1e-5,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if factor_dim <= 0:
            raise ValueError("factor_dim must be positive")
        if dz_dim <= 0:
            raise ValueError("dz_dim must be positive")
        if response_noise_var <= 0.0:
            raise ValueError("response_noise_var must be positive")
        if not 0.0 <= alpha_low <= 1.0 or not 0.0 <= alpha_high <= 1.0:
            raise ValueError("alpha_low/high must be in [0,1]")
        if alpha_low > alpha_high:
            raise ValueError("alpha_low cannot exceed alpha_high")
        self.factor_dim = int(factor_dim)
        self.dz_dim = int(dz_dim)
        self.response_noise_var = float(response_noise_var)
        self.alpha_low = float(alpha_low)
        self.alpha_high = float(alpha_high)
        self.reliability_tau = float(reliability_tau)
        self.ood_distance_threshold = float(ood_distance_threshold)
        self.ood_alpha_cap = float(ood_alpha_cap)
        self.jitter = float(jitter)
        self.device = torch.device(device)

        pm = torch.zeros(self.factor_dim, dtype=torch.float32) if prior_mean is None else prior_mean.to(dtype=torch.float32)
        pc = torch.eye(self.factor_dim, dtype=torch.float32) if prior_covariance is None else prior_covariance.to(dtype=torch.float32)
        if pm.shape != (self.factor_dim,):
            raise ValueError("prior_mean must have shape [K]")
        if pc.shape != (self.factor_dim, self.factor_dim):
            raise ValueError("prior_covariance must have shape [K,K]")
        if source_beta is None:
            sb = torch.empty(0, self.factor_dim, dtype=torch.float32)
        else:
            sb = source_beta.to(dtype=torch.float32)
            if sb.ndim != 2 or sb.shape[1] != self.factor_dim:
                raise ValueError("source_beta must have shape [P_src,K]")
        fm = torch.zeros(3, dtype=torch.float32) if feature_mean is None else feature_mean.to(dtype=torch.float32)
        fs = torch.ones(3, dtype=torch.float32) if feature_std is None else feature_std.to(dtype=torch.float32)
        if fm.shape != (3,) or fs.shape != (3,):
            raise ValueError("feature_mean/std must have shape [3]")

        prior_precision = torch.linalg.pinv(pc)
        prior_eta = prior_precision @ pm
        self.register_buffer("prior_mean", pm)
        self.register_buffer("prior_covariance", pc)
        self.register_buffer("prior_precision", prior_precision)
        self.register_buffer("prior_eta", prior_eta)
        self.register_buffer("source_beta", sb)
        self.register_buffer("feature_mean", fm)
        self.register_buffer("feature_std", fs.clamp_min(1e-6))
        self.register_buffer("Lambda", prior_precision.unsqueeze(0).clone())
        self.register_buffer("eta", prior_eta.unsqueeze(0).clone())
        self.register_buffer("alpha", torch.ones(1, dtype=torch.float32))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))
        self.to(self.device)

    def reset(self, batch_size: int = 1) -> "SRVFBelief":
        """Reset batched belief to the prior state."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.Lambda = self.prior_precision.unsqueeze(0).expand(batch_size, -1, -1).clone().to(self.device)
        self.eta = self.prior_eta.unsqueeze(0).expand(batch_size, -1).clone().to(self.device)
        self.alpha = torch.ones(batch_size, dtype=torch.float32, device=self.device)
        self.count = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return self

    def set_state(
        self,
        Lambda: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        count: Optional[torch.Tensor] = None,
    ) -> "SRVFBelief":
        """Set belief tensors from a rollout snapshot."""
        if Lambda.ndim != 3 or Lambda.shape[-2:] != (self.factor_dim, self.factor_dim):
            raise ValueError("Lambda must have shape [B,K,K]")
        if eta.shape != (Lambda.shape[0], self.factor_dim):
            raise ValueError("eta must have shape [B,K]")
        if alpha.shape != (Lambda.shape[0],):
            raise ValueError("alpha must have shape [B]")
        self.Lambda = Lambda.to(device=self.device, dtype=torch.float32)
        self.eta = eta.to(device=self.device, dtype=torch.float32)
        self.alpha = alpha.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        if count is None:
            self.count = torch.zeros(Lambda.shape[0], dtype=torch.long, device=self.device)
        else:
            self.count = count.to(device=self.device, dtype=torch.long)
        return self

    def _jittered_precision(self) -> torch.Tensor:
        eye = torch.eye(self.factor_dim, dtype=self.Lambda.dtype, device=self.Lambda.device)
        return self.Lambda + self.jitter * eye.unsqueeze(0)

    def mean(self) -> torch.Tensor:
        """Eq.MEAN: solve `(Lambda+jitter I) mu = eta`, never explicit inverse."""
        return torch.linalg.solve(self._jittered_precision(), self.eta.unsqueeze(-1)).squeeze(-1)

    def covariance(self) -> torch.Tensor:
        """Return posterior covariance by solving against identity."""
        batch = self.Lambda.shape[0]
        eye = torch.eye(self.factor_dim, dtype=self.Lambda.dtype, device=self.Lambda.device)
        rhs = eye.unsqueeze(0).expand(batch, -1, -1)
        return torch.linalg.solve(self._jittered_precision(), rhs)

    def diag_covariance(self) -> torch.Tensor:
        return torch.diagonal(self.covariance(), dim1=-2, dim2=-1)

    def _gather_action_response(self, heads: SRVFHeadOutput, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = actions.shape[0]
        action_idx = actions.to(device=heads.r0.device, dtype=torch.long).view(batch, 1, 1)
        r0 = heads.r0.gather(1, action_idx.expand(-1, 1, self.dz_dim)).squeeze(1)
        rc_idx = action_idx.unsqueeze(-1).expand(-1, 1, self.dz_dim, self.factor_dim)
        r_c = heads.r_c.gather(1, rc_idx).squeeze(1)
        return r0, r_c

    def _beta_support_distance(self, mean: torch.Tensor) -> torch.Tensor:
        if self.source_beta.numel() == 0:
            return torch.zeros(mean.shape[0], dtype=mean.dtype, device=mean.device)
        distances = torch.linalg.vector_norm(mean.unsqueeze(1) - self.source_beta.to(mean.device).unsqueeze(0), dim=-1)
        return distances.min(dim=1).values

    def _posterior_contraction(self, covariance: torch.Tensor) -> torch.Tensor:
        prior_trace = torch.diagonal(self.prior_covariance.to(covariance.device), dim1=-2, dim2=-1).sum().clamp_min(1e-12)
        posterior_trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
        return (1.0 - posterior_trace / prior_trace).clamp(-1.0, 1.0)

    def calibrated_alpha(
        self,
        response_mse: torch.Tensor,
        beta_support_dist: torch.Tensor,
        posterior_contraction: torch.Tensor,
    ) -> torch.Tensor:
        """Eq.ALPHA_EB: empirical-Bayes source-calibrated alpha.

        This is explicitly not a full Bayesian update for M because the OOD
        likelihood p(Delta z | M=0) is not identified from source data alone.
        It uses only response-derived reliability features and source-calibrated
        thresholds; no target label, return, action value, or partner id enters.
        """
        features = torch.stack([response_mse, beta_support_dist, posterior_contraction], dim=1)
        z = (features - self.feature_mean.to(features.device)) / self.feature_std.to(features.device).clamp_min(1e-6)
        reliability = -z[:, 0] - z[:, 1] + z[:, 2]
        high = torch.full_like(reliability, self.alpha_high)
        low = torch.full_like(reliability, self.alpha_low)
        alpha = torch.where(reliability >= self.reliability_tau, high, low)
        cap = torch.full_like(alpha, self.ood_alpha_cap)
        alpha = torch.where(beta_support_dist >= self.ood_distance_threshold, torch.minimum(alpha, cap), alpha)
        return alpha.clamp(0.0, 1.0)

    def update(self, g: torch.Tensor, a: torch.Tensor, delta_z: torch.Tensor, heads: NeuralSRVFHeads) -> "SRVFBelief":
        """Update posterior using only `(g,a,Delta z)`.

        Eq.PREC_UPDATE: `Lambda += Rc^T Sigma_z^{-1} Rc`.
        Eq.ETA_UPDATE: `eta += Rc^T Sigma_z^{-1}(Delta z - R0)`.
        Complexity per batch element is O(K^2 * Dz), without materializing any
        `[S,A,Dz,K]` table.
        """
        if g.ndim != 2 or g.shape[1] != self.dz_dim:
            raise ValueError(f"g must have shape [B, {self.dz_dim}]")
        if delta_z.shape != g.shape:
            raise ValueError("delta_z must have the same shape as g")
        if a.shape != (g.shape[0],):
            raise ValueError("a must have shape [B]")
        if self.Lambda.shape[0] != g.shape[0]:
            self.reset(batch_size=g.shape[0])
        head_out = heads(g.to(self.device))
        r0_selected, rc_selected = self._gather_action_response(head_out, a.to(self.device))
        residual = delta_z.to(device=self.device, dtype=torch.float32) - r0_selected
        inv_noise = 1.0 / self.response_noise_var
        precision_inc = torch.einsum("bdk,bdl->bkl", rc_selected, rc_selected) * inv_noise
        eta_inc = torch.einsum("bdk,bd->bk", rc_selected, residual) * inv_noise
        self.Lambda = self.Lambda + precision_inc
        self.eta = self.eta + eta_inc
        self.count = self.count + 1

        mu = self.mean()
        covariance = self.covariance()
        response_mse = residual.square().mean(dim=-1)
        beta_dist = self._beta_support_distance(mu)
        contraction = self._posterior_contraction(covariance)
        self.alpha = self.calibrated_alpha(response_mse, beta_dist, contraction)
        return self

    @staticmethod
    def score_from_posterior(
        g: torch.Tensor,
        heads: NeuralSRVFHeads,
        mean: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Eq.SCORE from explicit posterior tensors."""
        head_out = heads(g)
        if mean.shape != (g.shape[0], heads.factor_dim):
            raise ValueError("mean must have shape [B,K]")
        if alpha.shape != (g.shape[0],):
            raise ValueError("alpha must have shape [B]")
        raw = torch.einsum("bak,bk->ba", head_out.u_c, mean.to(head_out.u_c.device))
        return head_out.a0 + alpha.to(head_out.a0.device).view(-1, 1) * raw

    def score(self, g: torch.Tensor, heads: NeuralSRVFHeads) -> torch.Tensor:
        """Eq.SCORE: `S(g,a,h)=A0(g,a)+alpha*U(g,a)^T mu`."""
        return self.score_from_posterior(g.to(self.device), heads, self.mean(), self.alpha)


def self_test_SRVFBelief() -> None:
    B, A, K, Dz = 2, 6, 4, 8
    device = torch.device("cpu")
    heads = NeuralSRVFHeads(g_dim=Dz, num_actions=A, factor_dim=K, device=device)
    belief = SRVFBelief(factor_dim=K, dz_dim=Dz, source_beta=torch.randn(5, K), device=device).reset(B)
    g = torch.randn(B, Dz, device=device)
    actions = torch.tensor([0, 5], dtype=torch.long, device=device)
    delta_z = torch.randn(B, Dz, device=device)
    belief.update(g, actions, delta_z, heads)
    score = belief.score(g, heads)
    assert score.shape == (B, A)
    _assert_finite("belief_score", score)

    head_out = heads(g)
    target_mu = torch.randn(B, K, device=device)
    Lambda = torch.eye(K, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    eta = torch.einsum("bkl,bl->bk", Lambda, target_mu)
    belief.set_state(Lambda, eta, torch.zeros(B, device=device))
    fallback_score = belief.score(g, heads)
    assert torch.allclose(fallback_score, head_out.a0, atol=1e-5), "alpha=0 must give population A0"
    belief.set_state(Lambda, eta, torch.ones(B, device=device))
    raw_score = belief.score(g, heads)
    expected_raw = head_out.a0 + torch.einsum("bak,bk->ba", head_out.u_c, target_mu)
    assert torch.allclose(raw_score, expected_raw, atol=1e-4), "alpha=1 must give raw SRVF"


# ── MODULE 4: MAPPOActorCritic ──

class MAPPOActorCritic(nn.Module):
    """MAPPO actor-critic with exact Eq.LOGIT SRVF injection."""

    def __init__(
        self,
        obs_dim: int,
        global_state_dim: int,
        g_dim: int,
        num_actions: int = 6,
        factor_dim: int = 4,
        phase_dim: int = 0,
        hidden_dim: int = 128,
        tau: float = 1.0,
        detach_srvf_in_actor: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or global_state_dim <= 0 or g_dim <= 0:
            raise ValueError("obs_dim, global_state_dim, and g_dim must be positive")
        if num_actions <= 1:
            raise ValueError("num_actions must be at least 2")
        if factor_dim <= 0:
            raise ValueError("factor_dim must be positive")
        if phase_dim < 0:
            raise ValueError("phase_dim cannot be negative")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.obs_dim = int(obs_dim)
        self.global_state_dim = int(global_state_dim)
        self.g_dim = int(g_dim)
        self.num_actions = int(num_actions)
        self.factor_dim = int(factor_dim)
        self.phase_dim = int(phase_dim)
        self.tau = float(tau)
        self.detach_srvf_in_actor = bool(detach_srvf_in_actor)
        self.device = torch.device(device)
        actor_input_dim = self.obs_dim + self.g_dim + 2 * self.factor_dim + 1 + self.phase_dim
        critic_input_dim = self.global_state_dim + self.g_dim + 2 * self.factor_dim + 1 + self.phase_dim
        self.continuation_head = _mlp(actor_input_dim, hidden_dim, self.num_actions, depth=2)
        self.value_head = _mlp(critic_input_dim, hidden_dim, 1, depth=2)
        self.to(self.device)

    def _actor_input(
        self,
        obs_ego: torch.Tensor,
        g: torch.Tensor,
        belief_mean: torch.Tensor,
        belief_diag_cov: torch.Tensor,
        belief_alpha: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        if phase.ndim != 2 or phase.shape[1] != self.phase_dim:
            raise ValueError(f"phase must have shape [B, {self.phase_dim}]")
        return torch.cat(
            [
                obs_ego.to(device=self.device, dtype=torch.float32),
                g.to(device=self.device, dtype=torch.float32),
                belief_mean.to(device=self.device, dtype=torch.float32),
                belief_diag_cov.to(device=self.device, dtype=torch.float32),
                belief_alpha.to(device=self.device, dtype=torch.float32).unsqueeze(-1),
                phase.to(device=self.device, dtype=torch.float32),
            ],
            dim=-1,
        )

    def _critic_input(
        self,
        global_state: torch.Tensor,
        g: torch.Tensor,
        belief_mean: torch.Tensor,
        belief_diag_cov: torch.Tensor,
        belief_alpha: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                global_state.to(device=self.device, dtype=torch.float32),
                g.to(device=self.device, dtype=torch.float32),
                belief_mean.to(device=self.device, dtype=torch.float32),
                belief_diag_cov.to(device=self.device, dtype=torch.float32),
                belief_alpha.to(device=self.device, dtype=torch.float32).unsqueeze(-1),
                phase.to(device=self.device, dtype=torch.float32),
            ],
            dim=-1,
        )

    def apply_legal_mask(self, logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        if legal_mask.shape != logits.shape:
            raise ValueError("legal_mask must have shape [B,A]")
        mask = legal_mask.to(device=logits.device, dtype=torch.bool)
        if bool((~mask.any(dim=1)).any().item()):
            raise ValueError("every row in legal_mask must contain at least one legal action")
        return logits.masked_fill(~mask, -1.0e9)

    def forward(
        self,
        obs_ego: torch.Tensor,
        global_state: torch.Tensor,
        g: torch.Tensor,
        belief_mean: torch.Tensor,
        belief_diag_cov: torch.Tensor,
        belief_alpha: torch.Tensor,
        phase: torch.Tensor,
        legal_mask: torch.Tensor,
        srvf_heads: NeuralSRVFHeads,
    ) -> ActorCriticOutput:
        """Eq.LOGIT: `logits = C_theta(s,b,a) + S(g,a,h)/tau` exactly."""
        batch = g.shape[0]
        if obs_ego.shape != (batch, self.obs_dim):
            raise ValueError(f"obs_ego must have shape [B, {self.obs_dim}]")
        if global_state.shape != (batch, self.global_state_dim):
            raise ValueError(f"global_state must have shape [B, {self.global_state_dim}]")
        if g.shape != (batch, self.g_dim):
            raise ValueError(f"g must have shape [B, {self.g_dim}]")
        if belief_mean.shape != (batch, self.factor_dim):
            raise ValueError("belief_mean must have shape [B,K]")
        if belief_diag_cov.shape != (batch, self.factor_dim):
            raise ValueError("belief_diag_cov must have shape [B,K]")
        if belief_alpha.shape != (batch,):
            raise ValueError("belief_alpha must have shape [B]")

        actor_in = self._actor_input(obs_ego, g, belief_mean, belief_diag_cov, belief_alpha, phase)
        critic_in = self._critic_input(global_state, g, belief_mean, belief_diag_cov, belief_alpha, phase)
        base_logits = self.continuation_head(actor_in)
        srvf_score = SRVFBelief.score_from_posterior(g.to(self.device), srvf_heads, belief_mean.to(self.device), belief_alpha.to(self.device))
        if self.detach_srvf_in_actor:
            # Eq.GRAD_ISO: policy gradients update C_theta and V_omega, not SRVF heads psi.
            srvf_score = srvf_score.detach()
        logits = base_logits + srvf_score / self.tau
        logits = self.apply_legal_mask(logits, legal_mask.to(self.device))
        value = self.value_head(critic_in).squeeze(-1)
        return ActorCriticOutput(
            dist=Categorical(logits=logits),
            logits=logits,
            value=value,
            base_logits=base_logits,
            srvf_score=srvf_score,
        )


def self_test_MAPPOActorCritic() -> None:
    B, A, K, Dz = 2, 6, 4, 8
    obs_dim, global_dim, phase_dim = 10, 12, 3
    device = torch.device("cpu")
    heads = NeuralSRVFHeads(Dz, A, K, device=device)
    model = MAPPOActorCritic(obs_dim, global_dim, Dz, A, K, phase_dim, device=device)
    obs = torch.randn(B, obs_dim, device=device)
    global_state = torch.randn(B, global_dim, device=device)
    g = torch.randn(B, Dz, device=device)
    mu = torch.randn(B, K, device=device)
    diag = torch.rand(B, K, device=device) + 0.1
    alpha = torch.rand(B, device=device)
    phase = torch.randn(B, phase_dim, device=device)
    legal = torch.ones(B, A, dtype=torch.bool, device=device)
    out = model(obs, global_state, g, mu, diag, alpha, phase, legal, heads)
    assert out.logits.shape == (B, A)
    assert out.value.shape == (B,)
    assert out.dist.log_prob(torch.tensor([0, 1], device=device)).shape == (B,)
    _assert_finite("actor_logits", out.logits)
    _assert_finite("critic_value", out.value)


# ── MODULE 5: UnifiedLoss ──

class UnifiedLoss(nn.Module):
    """Negative ELBO estimator with separated rollout and source-table terms.

    The MAPPO term consumes `rollout_batch`. The SRVF likelihood term consumes
    `source_batch` from a static IRF table slice. The two can be optimized with
    separate `.backward()` calls through `compute_mappo_loss` and
    `compute_srvf_loss`, or combined after verifying gradient isolation.
    """

    def __init__(
        self,
        actor_critic: MAPPOActorCritic,
        srvf_heads: NeuralSRVFHeads,
        *,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        lambda_delta: float = 1.0,
        lambda_a: float = 1.0,
        entropy_coef: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive")
        if value_coef < 0.0 or lambda_delta < 0.0 or lambda_a < 0.0 or entropy_coef < 0.0:
            raise ValueError("loss coefficients must be nonnegative")
        self.actor_critic = actor_critic
        self.srvf_heads = srvf_heads
        self.clip_epsilon = float(clip_epsilon)
        self.value_coef = float(value_coef)
        self.lambda_delta = float(lambda_delta)
        self.lambda_a = float(lambda_a)
        self.entropy_coef = float(entropy_coef)
        self.device = torch.device(device)
        self.to(self.device)

    def compute_mappo_loss(self, rollout_batch: RolloutBatch) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Eq.L_POLICY and Eq.L_VALUE from MAPPO rollout data only."""
        output = self.actor_critic(
            rollout_batch.obs_ego,
            rollout_batch.global_state,
            rollout_batch.g,
            rollout_batch.belief_mean,
            rollout_batch.belief_diag_cov,
            rollout_batch.belief_alpha,
            rollout_batch.phase,
            rollout_batch.legal_mask,
            self.srvf_heads,
        )
        actions = rollout_batch.actions.to(device=self.device, dtype=torch.long)
        new_log_probs = output.dist.log_prob(actions)
        old_log_probs = rollout_batch.old_log_probs.to(device=self.device, dtype=torch.float32)
        advantages = rollout_batch.advantages.to(device=self.device, dtype=torch.float32).detach()
        ratios = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        ppo_objective = torch.minimum(unclipped, clipped).mean()
        entropy = output.dist.entropy().mean()
        loss_policy = -ppo_objective - self.entropy_coef * entropy
        value_targets = rollout_batch.gae_targets.to(device=self.device, dtype=torch.float32)
        loss_value = F.mse_loss(output.value, value_targets)
        loss = loss_policy + self.value_coef * loss_value
        return loss, {
            "L_policy": loss_policy,
            "L_value": loss_value,
            "entropy": entropy.detach(),
            "ratio_mean": ratios.detach().mean(),
        }

    def compute_srvf_loss(self, source_batch: SourceBatch) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Eq.L_RESPONSE and Eq.L_RESIDUAL from source IRF table data only.

        We choose L_A, not L_rank. Both model the same centered V0-residual
        observation channel; using both would double-count that channel.
        """
        g = source_batch.g.to(device=self.device, dtype=torch.float32)
        beta = source_batch.beta.to(device=self.device, dtype=torch.float32)
        if beta.shape != (g.shape[0], self.srvf_heads.factor_dim):
            raise ValueError("source_batch.beta must have shape [B,K]")
        head_out = self.srvf_heads(g)
        delta_z_pred = head_out.r0 + torch.einsum("badk,bk->bad", head_out.r_c, beta)
        a_pred = head_out.a0 + torch.einsum("bak,bk->ba", head_out.u_c, beta)
        delta_target = source_batch.delta_z_target.to(device=self.device, dtype=torch.float32)
        a_target = source_batch.a_bar_target.to(device=self.device, dtype=torch.float32)
        if delta_target.shape != delta_z_pred.shape:
            raise ValueError("delta_z_target must have shape [B,A,Dz]")
        if a_target.shape != a_pred.shape:
            raise ValueError("a_bar_target must have shape [B,A]")
        mask = None if source_batch.action_mask is None else source_batch.action_mask.to(device=self.device)
        loss_delta = _masked_mse(delta_z_pred, delta_target, mask)
        loss_a = _masked_mse(a_pred, a_target, mask)
        loss = self.lambda_delta * loss_delta + self.lambda_a * loss_a
        return loss, {"L_delta": loss_delta, "L_A": loss_a}

    def compute(self, rollout_batch: RolloutBatch, source_batch: SourceBatch) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Eq.ELBO: combined scalar after computing separated terms."""
        mappo_loss, mappo_logs = self.compute_mappo_loss(rollout_batch)
        srvf_loss, srvf_logs = self.compute_srvf_loss(source_batch)
        total = mappo_loss + srvf_loss
        logs: dict[str, torch.Tensor] = {**dict(mappo_logs), **dict(srvf_logs)}
        logs["L_total"] = total.detach()
        return total, logs


def _dummy_batches(
    actor_critic: MAPPOActorCritic,
    srvf_heads: NeuralSRVFHeads,
    *,
    B: int = 2,
) -> tuple[RolloutBatch, SourceBatch]:
    A = actor_critic.num_actions
    K = actor_critic.factor_dim
    Dz = actor_critic.g_dim
    device = actor_critic.device
    obs = torch.randn(B, actor_critic.obs_dim, device=device)
    global_state = torch.randn(B, actor_critic.global_state_dim, device=device)
    g = torch.randn(B, Dz, device=device)
    mu = torch.randn(B, K, device=device)
    diag = torch.rand(B, K, device=device) + 0.1
    alpha = torch.rand(B, device=device)
    phase = torch.randn(B, actor_critic.phase_dim, device=device)
    legal = torch.ones(B, A, dtype=torch.bool, device=device)
    actions = torch.randint(0, A, (B,), device=device)
    with torch.no_grad():
        out = actor_critic(obs, global_state, g, mu, diag, alpha, phase, legal, srvf_heads)
        old_log_probs = out.dist.log_prob(actions).detach() + 0.01 * torch.randn(B, device=device)
    rollout = RolloutBatch(
        obs_ego=obs,
        global_state=global_state,
        g=g,
        phase=phase,
        belief_mean=mu,
        belief_diag_cov=diag,
        belief_alpha=alpha,
        legal_mask=legal,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=torch.randn(B, device=device),
        gae_targets=torch.randn(B, device=device),
    )
    source = SourceBatch(
        g=torch.randn(B, Dz, device=device),
        beta=torch.randn(B, K, device=device),
        delta_z_target=torch.randn(B, A, Dz, device=device),
        a_bar_target=torch.randn(B, A, device=device),
        action_mask=torch.ones(B, A, dtype=torch.bool, device=device),
    )
    return rollout, source


def self_test_UnifiedLoss() -> None:
    B, A, K, Dz = 2, 6, 4, 8
    obs_dim, global_dim, phase_dim = 10, 12, 3
    device = torch.device("cpu")
    heads = NeuralSRVFHeads(Dz, A, K, device=device)
    actor_critic = MAPPOActorCritic(obs_dim, global_dim, Dz, A, K, phase_dim, device=device)
    unified = UnifiedLoss(actor_critic, heads, device=device)
    rollout, source = _dummy_batches(actor_critic, heads, B=B)
    total, logs = unified.compute(rollout, source)
    assert total.ndim == 0
    _assert_finite("L_total", total)
    for value in logs.values():
        _assert_finite("loss_log", value if isinstance(value, torch.Tensor) else torch.as_tensor(value))

    actor_params = tuple(actor_critic.parameters())
    srvf_params = tuple(heads.parameters())
    actor_critic.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    srvf_loss, srvf_logs = unified.compute_srvf_loss(source)
    srvf_logs["L_delta"].backward(retain_graph=True)
    assert _has_any_grad(srvf_params), "L_delta must reach psi / SRVF heads"
    assert not _has_any_grad(actor_params), "L_delta must not reach theta / actor-critic"
    actor_critic.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    total.backward()
    assert _has_any_grad(actor_params), "combined loss must reach theta through MAPPO terms"
    assert _has_any_grad(srvf_params), "combined loss must reach psi through source terms"


# ── gradient_audit() ──

def gradient_audit() -> Mapping[str, bool | float]:
    """Eq.GRAD_ISO audit using `torch.autograd.grad(retain_graph=True)`.

    Expected paths in this implementation:
    - L_policy reaches actor continuation parameters.
    - L_policy does not reach SRVF heads because Eq.LOGIT uses a detached SRVF score.
    - L_delta reaches SRVF heads.
    - L_delta does not reach actor-critic parameters.
    - L_value reaches critic parameters.
    """
    B, A, K, Dz = 2, 6, 4, 8
    obs_dim, global_dim, phase_dim = 10, 12, 3
    device = torch.device("cpu")
    heads = NeuralSRVFHeads(Dz, A, K, device=device)
    actor_critic = MAPPOActorCritic(obs_dim, global_dim, Dz, A, K, phase_dim, detach_srvf_in_actor=True, device=device)
    unified = UnifiedLoss(actor_critic, heads, device=device)
    rollout, source = _dummy_batches(actor_critic, heads, B=B)
    _, mappo_logs = unified.compute_mappo_loss(rollout)
    _, srvf_logs = unified.compute_srvf_loss(source)

    actor_params = tuple(actor_critic.continuation_head.parameters())
    critic_params = tuple(actor_critic.value_head.parameters())
    srvf_params = tuple(heads.parameters())

    policy_to_actor = _grad_norm_from_autograd(mappo_logs["L_policy"], actor_params)
    policy_to_srvf = _grad_norm_from_autograd(mappo_logs["L_policy"], srvf_params)
    value_to_critic = _grad_norm_from_autograd(mappo_logs["L_value"], critic_params)
    delta_to_srvf = _grad_norm_from_autograd(srvf_logs["L_delta"], srvf_params)
    delta_to_actor = _grad_norm_from_autograd(srvf_logs["L_delta"], actor_params)
    a_to_srvf = _grad_norm_from_autograd(srvf_logs["L_A"], srvf_params)
    a_to_actor = _grad_norm_from_autograd(srvf_logs["L_A"], actor_params)

    return {
        "policy_reaches_actor": policy_to_actor > 0.0,
        "policy_blocked_from_srvf": policy_to_srvf == 0.0,
        "value_reaches_critic": value_to_critic > 0.0,
        "delta_reaches_srvf": delta_to_srvf > 0.0,
        "delta_blocked_from_actor": delta_to_actor == 0.0,
        "A_reaches_srvf": a_to_srvf > 0.0,
        "A_blocked_from_actor": a_to_actor == 0.0,
        "policy_to_actor_grad_norm": policy_to_actor,
        "policy_to_srvf_grad_norm": policy_to_srvf,
        "value_to_critic_grad_norm": value_to_critic,
        "delta_to_srvf_grad_norm": delta_to_srvf,
        "delta_to_actor_grad_norm": delta_to_actor,
        "A_to_srvf_grad_norm": a_to_srvf,
        "A_to_actor_grad_norm": a_to_actor,
    }


def _flatten_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(dtype=torch.float32).reshape(-1)


def _step_state_g(step: Any) -> torch.Tensor:
    state_g = getattr(step, "state_g", None)
    if state_g is None:
        affordance = getattr(step, "affordance", None)
        state_g = getattr(affordance, "g", affordance)
    if state_g is None:
        raise ValueError("benchmark step must provide state_g or affordance.g")
    return _flatten_tensor(state_g)


def _adapter_public_g(adapter: Any) -> torch.Tensor:
    if hasattr(adapter, "public_chart_tensor"):
        return _flatten_tensor(adapter.public_chart_tensor())
    affordance = adapter.extract_affordance()
    return _flatten_tensor(getattr(affordance, "g", affordance))


def _adapter_ego_obs(adapter: Any, expected_dim: int) -> torch.Tensor:
    if hasattr(adapter, "ego_observation_tensor"):
        obs = _flatten_tensor(adapter.ego_observation_tensor(agent_index=0))
    else:
        obs = _adapter_public_g(adapter)
    if obs.shape[0] != expected_dim:
        raise ValueError(f"ego observation has dim {obs.shape[0]}, expected {expected_dim}")
    return obs


def _adapter_global_state(adapter: Any, expected_dim: int) -> torch.Tensor:
    if hasattr(adapter, "global_state_tensor"):
        state = _flatten_tensor(adapter.global_state_tensor())
    else:
        state = _adapter_public_g(adapter)
    if state.shape[0] != expected_dim:
        raise ValueError(f"global state has dim {state.shape[0]}, expected {expected_dim}")
    return state


def _adapter_legal_mask(adapter: Any, num_actions: int) -> torch.Tensor:
    mask = adapter.legal_action_mask() if hasattr(adapter, "legal_action_mask") else torch.ones(num_actions, dtype=torch.bool)
    mask = mask.detach().to(dtype=torch.bool).reshape(-1)
    if mask.shape != (num_actions,):
        raise ValueError(f"legal mask must have shape [{num_actions}]")
    if not bool(mask.any().item()):
        raise ValueError("legal mask must contain at least one legal action")
    return mask


def _resolve_device(requested: str | torch.device) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _discounted_cumsum(values: torch.Tensor, discount: float) -> torch.Tensor:
    """Vectorized discounted cumulative sum over a one-dimensional tensor."""
    values = values.to(dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        return values.clone()
    if float(discount) == 0.0:
        return values.clone()
    device = values.device
    steps = torch.arange(values.numel(), dtype=values.dtype, device=device)
    discounts = torch.pow(torch.full_like(steps, float(discount)), steps)
    if bool((discounts == 0.0).any().item()):
        returns = torch.zeros_like(values)
        running = torch.zeros((), dtype=values.dtype, device=device)
        for idx in range(values.numel() - 1, -1, -1):
            running = values[idx] + float(discount) * running
            returns[idx] = running
        return returns
    reversed_values = torch.flip(values, dims=(0,))
    reversed_returns = torch.cumsum(reversed_values / discounts, dim=0) * discounts
    return torch.flip(reversed_returns, dims=(0,))


def _discounted_episode_returns(rewards: Sequence[float] | torch.Tensor, gamma: float) -> torch.Tensor:
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32)
    return _discounted_cumsum(reward_tensor, gamma)


def _distribution(prefix: str, values: torch.Tensor) -> dict[str, float]:
    values = values.to(dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_nonzero_fraction": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
        f"{prefix}_nonzero_fraction": float((values.abs() > 1e-8).to(dtype=torch.float32).mean().item()),
    }


def _action_distribution(actions: torch.Tensor, num_actions: int) -> Mapping[str, Any]:
    actions = actions.to(dtype=torch.long).reshape(-1)
    if actions.numel() == 0:
        return {
            "action_histogram": [0 for _ in range(int(num_actions))],
            "action_fraction": [0.0 for _ in range(int(num_actions))],
            "action_entropy": 0.0,
        }
    counts = torch.bincount(actions.clamp(0, int(num_actions) - 1), minlength=int(num_actions))
    counts = counts[: int(num_actions)].to(dtype=torch.float32)
    fractions = counts / counts.sum().clamp_min(1.0)
    positive = fractions > 0.0
    entropy = -(fractions[positive] * fractions[positive].log()).sum()
    return {
        "action_histogram": [int(x) for x in counts.to(dtype=torch.long).tolist()],
        "action_fraction": [float(x) for x in fractions.tolist()],
        "action_entropy": float(entropy.item()),
    }


def _reward_monitor(
    *,
    episode_returns: Sequence[float],
    discounted_returns: Sequence[float],
    episode_lengths: Sequence[int],
    done_flags: Sequence[bool],
    step_rewards: Sequence[float],
) -> Mapping[str, Any]:
    returns = torch.tensor(list(episode_returns), dtype=torch.float32)
    discounted = torch.tensor(list(discounted_returns), dtype=torch.float32)
    lengths = torch.tensor(list(episode_lengths), dtype=torch.float32)
    rewards = torch.tensor(list(step_rewards), dtype=torch.float32)
    done_tensor = torch.tensor([float(flag) for flag in done_flags], dtype=torch.float32)
    payload: dict[str, Any] = {
        "episode_count": int(returns.numel()),
        "step_count": int(rewards.numel()),
        "done_fraction": float(done_tensor.mean().item()) if done_tensor.numel() else 0.0,
        "mean_episode_length": float(lengths.mean().item()) if lengths.numel() else 0.0,
    }
    payload.update(_distribution("episode_return", returns))
    payload.update(_distribution("discounted_episode_return", discounted))
    payload.update(_distribution("step_reward", rewards))
    return payload


def _alpha_monitor(alphas: Sequence[float]) -> Mapping[str, Any]:
    alpha = torch.tensor(list(alphas), dtype=torch.float32)
    payload = _distribution("alpha", alpha)
    payload["fallback_rate"] = (
        float((alpha <= 0.25).to(dtype=torch.float32).mean().item())
        if alpha.numel()
        else 0.0
    )
    return payload


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _merge_distribution_from_summaries(
    prefix: str,
    summaries: Sequence[Mapping[str, Any]],
    count_key: str,
) -> dict[str, float]:
    total_count = sum(int(summary.get(count_key, 0)) for summary in summaries)
    if total_count <= 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_nonzero_fraction": 0.0,
        }
    mean = 0.0
    nonzero = 0.0
    min_value = float("inf")
    max_value = float("-inf")
    second_moment = 0.0
    for summary in summaries:
        count = int(summary.get(count_key, 0))
        if count <= 0:
            continue
        weight = float(count)
        item_mean = float(summary.get(f"{prefix}_mean", 0.0))
        item_std = float(summary.get(f"{prefix}_std", 0.0))
        mean += weight * item_mean
        second_moment += weight * (item_std * item_std + item_mean * item_mean)
        nonzero += weight * float(summary.get(f"{prefix}_nonzero_fraction", 0.0))
        min_value = min(min_value, float(summary.get(f"{prefix}_min", 0.0)))
        max_value = max(max_value, float(summary.get(f"{prefix}_max", 0.0)))
    mean /= float(total_count)
    variance = max(second_moment / float(total_count) - mean * mean, 0.0)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": float(variance ** 0.5),
        f"{prefix}_min": 0.0 if min_value == float("inf") else min_value,
        f"{prefix}_max": 0.0 if max_value == float("-inf") else max_value,
        f"{prefix}_nonzero_fraction": nonzero / float(total_count),
    }


def _merge_rollout_monitors(monitors: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not monitors:
        return {}
    worker_count = len(monitors)
    episode_count = sum(int(item.get("episode_count", 0)) for item in monitors)
    step_count = sum(int(item.get("step_count", 0)) for item in monitors)
    action_histogram: list[int] = []
    for monitor in monitors:
        current = [int(x) for x in monitor.get("action_histogram", [])]
        if len(action_histogram) < len(current):
            action_histogram.extend([0] * (len(current) - len(action_histogram)))
        for idx, value in enumerate(current):
            action_histogram[idx] += value
    if action_histogram:
        action_counts = torch.tensor(action_histogram, dtype=torch.float32)
        action_fraction = action_counts / action_counts.sum().clamp_min(1.0)
        positive = action_fraction > 0.0
        action_entropy = float((-(action_fraction[positive] * action_fraction[positive].log()).sum()).item())
    else:
        action_fraction = torch.empty(0, dtype=torch.float32)
        action_entropy = 0.0

    by_partner: dict[str, list[Mapping[str, Any]]] = {}
    for monitor in monitors:
        for partner_id, payload in dict(monitor.get("by_partner", {})).items():
            by_partner.setdefault(str(partner_id), []).append(payload)
    merged_by_partner = {
        partner_id: _merge_rollout_monitors(payloads)
        for partner_id, payloads in by_partner.items()
    }

    payload: dict[str, Any] = {
        "worker_count": worker_count,
        "episode_count": episode_count,
        "step_count": step_count,
        "done_fraction": (
            sum(float(item.get("done_fraction", 0.0)) * int(item.get("episode_count", 0)) for item in monitors)
            / max(episode_count, 1)
        ),
        "mean_episode_length": (
            sum(float(item.get("mean_episode_length", 0.0)) * int(item.get("episode_count", 0)) for item in monitors)
            / max(episode_count, 1)
        ),
        "fallback_rate": (
            sum(float(item.get("fallback_rate", 0.0)) * int(item.get("step_count", 0)) for item in monitors)
            / max(step_count, 1)
        ),
        "legal_action_fraction": (
            sum(float(item.get("legal_action_fraction", 0.0)) * int(item.get("step_count", 0)) for item in monitors)
            / max(step_count, 1)
        ),
        "action_histogram": action_histogram,
        "action_fraction": [float(x) for x in action_fraction.tolist()],
        "action_entropy": action_entropy,
        "by_partner": merged_by_partner,
        "by_worker": [dict(item) for item in monitors],
        "monitor_source": "parallel_source_training_rollout" if worker_count > 1 else monitors[0].get("monitor_source", "source_training_rollout"),
        "target_data_used": any(bool(item.get("target_data_used", False)) for item in monitors),
    }
    payload.update(_merge_distribution_from_summaries("episode_return", monitors, "episode_count"))
    payload.update(_merge_distribution_from_summaries("discounted_episode_return", monitors, "episode_count"))
    payload.update(_merge_distribution_from_summaries("step_reward", monitors, "step_count"))
    payload.update(_merge_distribution_from_summaries("alpha", monitors, "step_count"))
    return payload


def _concat_rollout_batches(
    batches: Sequence[RolloutBatch],
    *,
    device: str | torch.device,
) -> RolloutBatch:
    if not batches:
        raise ValueError("batches must be nonempty")
    target_device = torch.device(device)
    return RolloutBatch(
        obs_ego=torch.cat([batch.obs_ego for batch in batches], dim=0).to(target_device),
        global_state=torch.cat([batch.global_state for batch in batches], dim=0).to(target_device),
        g=torch.cat([batch.g for batch in batches], dim=0).to(target_device),
        phase=torch.cat([batch.phase for batch in batches], dim=0).to(target_device),
        belief_mean=torch.cat([batch.belief_mean for batch in batches], dim=0).to(target_device),
        belief_diag_cov=torch.cat([batch.belief_diag_cov for batch in batches], dim=0).to(target_device),
        belief_alpha=torch.cat([batch.belief_alpha for batch in batches], dim=0).to(target_device),
        legal_mask=torch.cat([batch.legal_mask for batch in batches], dim=0).to(target_device),
        actions=torch.cat([batch.actions for batch in batches], dim=0).to(target_device),
        old_log_probs=torch.cat([batch.old_log_probs for batch in batches], dim=0).to(target_device),
        advantages=torch.cat([batch.advantages for batch in batches], dim=0).to(target_device),
        gae_targets=torch.cat([batch.gae_targets for batch in batches], dim=0).to(target_device),
        monitor=_merge_rollout_monitors([batch.monitor for batch in batches]),
    )


def collect_v0_training_batch(
    adapter: Any,
    partners: Sequence[tuple[str, Any]],
    *,
    ego_policy: Any,
    episodes_per_partner: int = 8,
    horizon: int = 400,
    gamma: float = 0.99,
    seed: int = 0,
) -> V0TrainingBatch:
    """Collect source-population raw-reward returns for partner-blind V0."""
    if not partners:
        raise ValueError("partners must be nonempty")
    if episodes_per_partner <= 0:
        raise ValueError("episodes_per_partner must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not getattr(ego_policy, "partner_blind", False):
        raise ValueError("ego_policy must declare partner_blind=True")

    all_states: list[torch.Tensor] = []
    all_returns: list[torch.Tensor] = []
    all_rewards: list[torch.Tensor] = []
    episode_lengths: list[int] = []
    for partner_idx, (_partner_id, partner) in enumerate(partners):
        for episode_idx in range(int(episodes_per_partner)):
            rollout_seed = int(seed + partner_idx * 100_000 + episode_idx)
            observation = adapter.reset(rollout_seed)
            partner_state = partner.reset(rollout_seed)
            ego_state = ego_policy.reset(rollout_seed + 10_000)
            rng = torch.Generator().manual_seed(rollout_seed + 20_000)
            states: list[torch.Tensor] = []
            rewards: list[float] = []
            for _step_idx in range(int(horizon)):
                g = _adapter_public_g(adapter)
                ego_action, ego_state = ego_policy.act(observation, g, ego_state, rng)
                partner_action, partner_state = partner.act(observation, partner_state, rng)
                transition = adapter.step(int(ego_action), int(partner_action))
                states.append(g)
                rewards.append(float(transition.reward))
                observation = transition.observation
                if bool(transition.done):
                    break
            if states:
                all_states.append(torch.stack(states, dim=0))
                reward_tensor = torch.tensor(rewards, dtype=torch.float32)
                all_rewards.append(reward_tensor)
                all_returns.append(_discounted_episode_returns(rewards, gamma))
                episode_lengths.append(len(states))

    if not all_states:
        raise RuntimeError("failed to collect nonempty V0 training data")
    returns = torch.cat(all_returns, dim=0)
    rewards = torch.cat(all_rewards, dim=0)
    return V0TrainingBatch(
        state_g=torch.cat(all_states, dim=0),
        returns=returns,
        metadata={
            "collector": "srvf_mappo_source_population_raw_reward_rollouts",
            "partner_blind": True,
            "target_reward_type": "raw_env_reward",
            "episodes_per_partner": int(episodes_per_partner),
            "horizon": int(horizon),
            "gamma": float(gamma),
            "partner_count": len(partners),
            "episode_lengths": episode_lengths,
            "sample_count": int(returns.numel()),
            "no_partner_id": True,
            "no_beta": True,
            "no_target_label": True,
            **_distribution("raw_reward", rewards),
            **_distribution("discounted_return", returns),
        },
    )


def train_v0_ensemble(
    batch: V0TrainingBatch,
    *,
    ensemble_size: int = 3,
    hidden_dims: Sequence[int] = (256, 256),
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.2,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[V0Ensemble, Mapping[str, Any]]:
    """Train a bootstrap partner-blind V0 ensemble."""
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target_device = _resolve_device(device)
    states = batch.state_g.to(dtype=torch.float32)
    targets = batch.returns.to(dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(states.shape[0], generator=generator)
    val_count = int(round(float(validation_fraction) * states.shape[0]))
    if states.shape[0] > 1:
        val_count = min(max(1, val_count), states.shape[0] - 1)
    else:
        val_count = 0
    val_idx = order[:val_count]
    train_idx = order[val_count:] if val_count > 0 else order
    train_states = states.index_select(0, train_idx)
    train_targets = targets.index_select(0, train_idx)
    state_mean = train_states.mean(dim=0)
    state_std = train_states.std(dim=0, unbiased=False)
    state_std = torch.where(state_std <= 1e-6, torch.ones_like(state_std), state_std)
    target_mean = train_targets.mean()
    target_std = train_targets.std(unbiased=False).clamp_min(1e-6)
    norm_states = ((states - state_mean) / state_std).to(target_device)
    norm_targets = ((targets - target_mean) / target_std).to(target_device)

    models: list[V0MLP] = []
    member_metrics: list[dict[str, float]] = []
    for member_idx in range(int(ensemble_size)):
        torch.manual_seed(int(seed) + member_idx)
        member_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 10_000 + member_idx)
        model = V0MLP(states.shape[1], hidden_dims).to(target_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        if train_idx.numel() > 1:
            bootstrap_pos = torch.randint(
                0,
                train_idx.numel(),
                size=(train_idx.numel(),),
                generator=member_generator,
            )
            member_train_idx = train_idx.index_select(0, bootstrap_pos)
        else:
            member_train_idx = train_idx
        member_train_idx = member_train_idx.to(target_device)
        for _epoch in range(int(epochs)):
            permutation = torch.randperm(member_train_idx.numel(), generator=member_generator).to(target_device)
            shuffled = member_train_idx.index_select(0, permutation)
            for start in range(0, shuffled.numel(), int(batch_size)):
                idx = shuffled[start : start + int(batch_size)]
                prediction = model(norm_states.index_select(0, idx))
                loss = F.mse_loss(prediction, norm_targets.index_select(0, idx))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            train_prediction = model(norm_states.index_select(0, train_idx.to(target_device)))
            train_loss = F.mse_loss(train_prediction, norm_targets.index_select(0, train_idx.to(target_device)))
            if val_count > 0:
                val_prediction = model(norm_states.index_select(0, val_idx.to(target_device)))
                val_loss = F.mse_loss(val_prediction, norm_targets.index_select(0, val_idx.to(target_device)))
            else:
                val_loss = train_loss
        member_metrics.append(
            {
                "train_mse_normalized": float(train_loss.detach().cpu().item()),
                "validation_mse_normalized": float(val_loss.detach().cpu().item()),
            }
        )
        models.append(model.cpu())

    metadata = {
        **dict(batch.metadata),
        "v0_partner_blind": True,
        "ensemble_size": int(ensemble_size),
        "hidden_dims": tuple(int(dim) for dim in hidden_dims),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "validation_fraction": float(validation_fraction),
        "requested_device": str(device),
        "effective_device": str(target_device),
        "train_count": int(train_idx.numel()),
        "validation_count": int(val_count),
        "target_mean": float(target_mean.item()),
        "target_std": float(target_std.item()),
        "member_metrics": member_metrics,
    }
    return (
        V0Ensemble(
            models=tuple(models),
            state_mean=state_mean.cpu(),
            state_std=state_std.cpu(),
            target_mean=target_mean.reshape(()).cpu(),
            target_std=target_std.reshape(()).cpu(),
            metadata=metadata,
        ),
        metadata,
    )


def collect_reservoir_snapshots(
    adapter: Any,
    partners: Sequence[tuple[str, Any]],
    *,
    ego_policy: Any,
    max_states: int = 256,
    episodes: int = 16,
    horizon: int = 400,
    seed: int = 0,
) -> list[Any]:
    """Collect exact classic snapshots for resettable interventions."""
    if max_states <= 0:
        raise ValueError("max_states must be positive")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not partners:
        raise ValueError("partners must be nonempty")
    snapshots: list[Any] = []
    seen: set[bytes] = set()
    for episode_idx in range(int(episodes)):
        partner_id, partner = partners[episode_idx % len(partners)]
        rollout_seed = int(seed + episode_idx)
        observation = adapter.reset(rollout_seed)
        partner_state = partner.reset(rollout_seed)
        ego_state = ego_policy.reset(rollout_seed + 10_000)
        rng = torch.Generator().manual_seed(rollout_seed + 20_000)
        for step_idx in range(int(horizon)):
            g = _adapter_public_g(adapter)
            key = torch.round(g * 1000.0).to(dtype=torch.int32).cpu().numpy().tobytes()
            if key not in seen:
                seen.add(key)
                phase = f"t{min(step_idx // 50, 7)}"
                snapshot = adapter.snapshot(phase_id=phase)
                snapshots.append(snapshot)
                if len(snapshots) >= int(max_states):
                    return snapshots
            ego_action, ego_state = ego_policy.act(observation, g, ego_state, rng)
            partner_action, partner_state = partner.act(observation, partner_state, rng)
            transition = adapter.step(int(ego_action), int(partner_action))
            observation = transition.observation
            if bool(transition.done):
                break
        _ = partner_id
    if not snapshots:
        raise RuntimeError("failed to collect reservoir snapshots")
    return snapshots


def collect_v0_residual_irf_table(
    adapter: Any,
    reservoir: Sequence[Any],
    partners: Sequence[tuple[str, Any]],
    *,
    v0: V0EnsembleProtocol,
    action_ids: Optional[Sequence[int]] = None,
    gamma: float = 0.99,
    repeats: int = 1,
    seed: int = 0,
    fail_fast: bool = True,
) -> IRFTable:
    """Collect one-step V0-residual source/target IRF table."""
    if not reservoir:
        raise ValueError("reservoir must be nonempty")
    if not partners:
        raise ValueError("partners must be nonempty")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    actions = tuple(range(adapter.num_actions)) if action_ids is None else tuple(int(x) for x in action_ids)
    state_g = torch.stack([_flatten_tensor(snapshot.state_g) for snapshot in reservoir], dim=0)
    phase_ids = tuple(str(getattr(snapshot, "phase_id", "unknown")) for snapshot in reservoir)
    partner_ids = tuple(str(partner_id) for partner_id, _partner in partners)
    s_count = len(reservoir)
    a_count = len(actions)
    p_count = len(partners)
    dz_dim = int(state_g.shape[1])
    delta_z = torch.zeros(s_count, a_count, p_count, dz_dim, dtype=torch.float32)
    q_raw = torch.zeros(s_count, a_count, p_count, dtype=torch.float32)
    valid_mask = torch.zeros(s_count, a_count, p_count, dtype=torch.bool)
    failures: list[dict[str, Any]] = []
    reward_values: list[float] = []
    next_g_records: list[torch.Tensor] = []
    reward_records: list[float] = []
    state_records: list[int] = []
    action_records: list[int] = []
    partner_records: list[int] = []
    with torch.inference_mode():
        current_v0 = v0.predict(state_g).mean.reshape(-1).to(dtype=torch.float32)

    for state_idx, snapshot in enumerate(reservoir):
        for action_pos, action_id in enumerate(actions):
            for partner_idx, (partner_id, partner) in enumerate(partners):
                local_next_g: list[torch.Tensor] = []
                local_rewards: list[float] = []
                try:
                    for repeat_idx in range(int(repeats)):
                        observation = adapter.restore(snapshot)
                        partner_state = partner.reset(seed + state_idx * 1_000_000 + action_pos * 1_000 + partner_idx * 10 + repeat_idx)
                        rng = torch.Generator().manual_seed(seed + 20_000 + state_idx * 1_000_000 + action_pos * 1_000 + partner_idx * 10 + repeat_idx)
                        partner_action, _partner_state = partner.act(observation, partner_state, rng)
                        transition = adapter.step(int(action_id), int(partner_action))
                        next_g = _step_state_g(transition)
                        reward = float(transition.reward)
                        local_next_g.append(next_g)
                        local_rewards.append(reward)
                        reward_values.append(reward)
                except Exception as exc:
                    if fail_fast:
                        raise
                    failures.append(
                        {
                            "state_idx": state_idx,
                            "action_id": int(action_id),
                            "partner_id": str(partner_id),
                            "error": repr(exc),
                        }
                    )
                    continue
                next_g_records.extend(local_next_g)
                reward_records.extend(local_rewards)
                state_records.extend([state_idx] * len(local_next_g))
                action_records.extend([action_pos] * len(local_next_g))
                partner_records.extend([partner_idx] * len(local_next_g))

    if next_g_records:
        next_g_tensor = torch.stack(next_g_records, dim=0).to(dtype=torch.float32)
        state_index = torch.tensor(state_records, dtype=torch.long)
        action_index = torch.tensor(action_records, dtype=torch.long)
        partner_index = torch.tensor(partner_records, dtype=torch.long)
        flat_index = state_index * (a_count * p_count) + action_index * p_count + partner_index
        with torch.inference_mode():
            next_v0 = v0.predict(next_g_tensor).mean.reshape(-1).to(dtype=torch.float32)
        rewards_tensor = torch.tensor(reward_records, dtype=torch.float32)
        responses = next_g_tensor - state_g.index_select(0, state_index)
        residuals = rewards_tensor + float(gamma) * next_v0 - current_v0.index_select(0, state_index)
        response_sum = torch.zeros(s_count * a_count * p_count, dz_dim, dtype=torch.float32)
        residual_sum = torch.zeros(s_count * a_count * p_count, dtype=torch.float32)
        repeat_count = torch.zeros(s_count * a_count * p_count, dtype=torch.float32)
        response_sum.index_add_(0, flat_index, responses)
        residual_sum.index_add_(0, flat_index, residuals)
        repeat_count.index_add_(0, flat_index, torch.ones_like(residuals))
        valid_flat = repeat_count > 0.0
        safe_count = repeat_count.clamp_min(1.0)
        delta_z = (response_sum / safe_count.unsqueeze(-1)).reshape(s_count, a_count, p_count, dz_dim)
        q_raw = (residual_sum / safe_count).reshape(s_count, a_count, p_count)
        valid_mask = valid_flat.reshape(s_count, a_count, p_count)

    a_raw = centered_action_residual(q_raw, valid_mask)
    return IRFTable.from_tensors(
        state_g=state_g,
        delta_z=delta_z,
        a_raw=a_raw,
        valid_mask=valid_mask,
        phase_ids=phase_ids,
        partner_ids=partner_ids,
        action_ids=actions,
        metadata={
            "collector": "srvf_mappo_resettable_v0_residual_irf_table",
            "value_residual_formula": "r + gamma * V0(g_next) - V0(g)",
            "gamma": float(gamma),
            "repeats": int(repeats),
            "failures": failures,
            "sample_count": int(valid_mask.sum().item()),
            "no_target_label_used_for_training": True,
            **_distribution("one_step_reward", torch.tensor(reward_values, dtype=torch.float32)),
        },
    )


def _batch_value(
    actor_critic: MAPPOActorCritic,
    global_state: torch.Tensor,
    g: torch.Tensor,
    belief: SRVFBelief,
    phase: torch.Tensor,
) -> torch.Tensor:
    value = actor_critic.value_head(
        actor_critic._critic_input(
            global_state.unsqueeze(0),
            g.unsqueeze(0),
            belief.mean(),
            belief.diag_covariance(),
            belief.alpha,
            phase,
        )
    ).squeeze(-1)
    return value.detach().reshape(())


def collect_classic_rollout_batch(
    adapter: Any,
    partners: Sequence[tuple[str, Any]],
    *,
    actor_critic: MAPPOActorCritic,
    srvf_heads: NeuralSRVFHeads,
    belief: SRVFBelief,
    episodes_per_partner: int = 1,
    horizon: int = 400,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    deterministic: bool = False,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> RolloutBatch:
    """Collect a flattened MAPPO `RolloutBatch` from classic Overcooked-AI.

    The collector records the belief state before the ego action. After the
    environment step, it updates `belief` only with `(g, action, Delta z)`.
    """
    if not partners:
        raise ValueError("partners must be nonempty")
    if episodes_per_partner <= 0:
        raise ValueError("episodes_per_partner must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0,1]")
    target_device = actor_critic.device if device is None else torch.device(device)
    phase_dim = actor_critic.phase_dim

    obs_rows: list[torch.Tensor] = []
    global_rows: list[torch.Tensor] = []
    g_rows: list[torch.Tensor] = []
    phase_rows: list[torch.Tensor] = []
    mean_rows: list[torch.Tensor] = []
    diag_rows: list[torch.Tensor] = []
    alpha_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    log_prob_rows: list[torch.Tensor] = []
    advantage_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    monitor_episode_returns: list[float] = []
    monitor_discounted_returns: list[float] = []
    monitor_episode_lengths: list[int] = []
    monitor_done_flags: list[bool] = []
    monitor_step_rewards: list[float] = []
    monitor_action_ids: list[int] = []
    monitor_alphas: list[float] = []
    monitor_legal_fractions: list[float] = []
    by_partner_monitor: dict[str, dict[str, Any]] = {}

    for partner_idx, (partner_id, partner) in enumerate(partners):
        partner_key = str(partner_id)
        partner_monitor = by_partner_monitor.setdefault(
            partner_key,
            {
                "episode_returns": [],
                "discounted_returns": [],
                "episode_lengths": [],
                "done_flags": [],
                "step_rewards": [],
                "action_ids": [],
                "alphas": [],
                "legal_fractions": [],
            },
        )
        for episode_idx in range(int(episodes_per_partner)):
            rollout_seed = int(seed + partner_idx * 100_000 + episode_idx)
            observation = adapter.reset(rollout_seed)
            partner_state = partner.reset(rollout_seed)
            rng = torch.Generator().manual_seed(rollout_seed + 20_000)
            belief.reset(batch_size=1)

            episode_obs: list[torch.Tensor] = []
            episode_global: list[torch.Tensor] = []
            episode_g: list[torch.Tensor] = []
            episode_phase: list[torch.Tensor] = []
            episode_mean: list[torch.Tensor] = []
            episode_diag: list[torch.Tensor] = []
            episode_alpha: list[torch.Tensor] = []
            episode_mask: list[torch.Tensor] = []
            episode_actions: list[torch.Tensor] = []
            episode_log_probs: list[torch.Tensor] = []
            episode_values: list[torch.Tensor] = []
            rewards: list[float] = []
            dones: list[bool] = []
            episode_action_ids: list[int] = []
            episode_alphas: list[float] = []
            episode_legal_fractions: list[float] = []

            done = False
            for _step_idx in range(int(horizon)):
                g = _adapter_public_g(adapter)
                obs_ego = _adapter_ego_obs(adapter, actor_critic.obs_dim)
                global_state = _adapter_global_state(adapter, actor_critic.global_state_dim)
                phase = torch.zeros(1, phase_dim, dtype=torch.float32, device=target_device)
                legal = _adapter_legal_mask(adapter, actor_critic.num_actions)
                mean = belief.mean().detach().reshape(actor_critic.factor_dim)
                diag = belief.diag_covariance().detach().reshape(actor_critic.factor_dim)
                alpha = belief.alpha.detach().reshape(())

                with torch.inference_mode():
                    output = actor_critic(
                        obs_ego.unsqueeze(0).to(target_device),
                        global_state.unsqueeze(0).to(target_device),
                        g.unsqueeze(0).to(target_device),
                        mean.unsqueeze(0).to(target_device),
                        diag.unsqueeze(0).to(target_device),
                        alpha.unsqueeze(0).to(target_device),
                        phase,
                        legal.unsqueeze(0).to(target_device),
                        srvf_heads,
                    )
                    if deterministic:
                        action = torch.argmax(output.logits, dim=-1)
                    else:
                        action = output.dist.sample()
                    log_prob = output.dist.log_prob(action).reshape(())
                    value = output.value.reshape(())

                partner_action, partner_state = partner.act(observation, partner_state, rng)
                action_int = int(action.item())
                action_for_update = action.detach().clone().to(target_device)
                transition = adapter.step(action_int, int(partner_action))
                next_g = _step_state_g(transition)
                delta_z = next_g - g
                with torch.no_grad():
                    belief.update(
                        g.unsqueeze(0).to(target_device),
                        action_for_update,
                        delta_z.unsqueeze(0).to(target_device),
                        srvf_heads,
                    )

                episode_obs.append(obs_ego)
                episode_global.append(global_state)
                episode_g.append(g)
                episode_phase.append(phase.reshape(phase_dim).detach())
                episode_mean.append(mean.detach())
                episode_diag.append(diag.detach())
                episode_alpha.append(alpha.detach())
                episode_mask.append(legal.detach().cpu())
                episode_actions.append(action.detach().clone().reshape(()))
                episode_log_probs.append(log_prob.detach().clone())
                episode_values.append(value.detach().clone())
                rewards.append(float(transition.reward))
                episode_action_ids.append(action_int)
                episode_alphas.append(float(alpha.detach().item()))
                episode_legal_fractions.append(float(legal.to(dtype=torch.float32).mean().item()))
                done = bool(transition.done)
                dones.append(done)
                observation = transition.observation
                if done:
                    break

            if not episode_values:
                continue

            if done:
                bootstrap = torch.zeros((), dtype=torch.float32, device=target_device)
            else:
                g = _adapter_public_g(adapter)
                global_state = _adapter_global_state(adapter, actor_critic.global_state_dim)
                phase = torch.zeros(1, phase_dim, dtype=torch.float32, device=target_device)
                with torch.inference_mode():
                    bootstrap = _batch_value(
                        actor_critic,
                        global_state.to(target_device),
                        g.to(target_device),
                        belief,
                        phase,
                    ).clone()

            values = torch.stack(episode_values, dim=0).to(device=target_device, dtype=torch.float32)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=target_device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=target_device)
            next_values = torch.cat([values[1:], bootstrap.reshape(1)])
            deltas = rewards_t + float(gamma) * next_values * (1.0 - dones_t) - values
            advantages = _discounted_cumsum(deltas, float(gamma) * float(gae_lambda))
            targets = advantages + values
            discounted_return = (
                float(_discounted_episode_returns(rewards, gamma)[0].item())
                if rewards
                else 0.0
            )
            episode_return = float(sum(rewards))
            episode_length = int(len(rewards))

            obs_rows.extend(episode_obs)
            global_rows.extend(episode_global)
            g_rows.extend(episode_g)
            phase_rows.extend(episode_phase)
            mean_rows.extend(episode_mean)
            diag_rows.extend(episode_diag)
            alpha_rows.extend(episode_alpha)
            mask_rows.extend(episode_mask)
            action_rows.extend(episode_actions)
            log_prob_rows.extend(episode_log_probs)
            advantage_rows.extend(advantages.detach().unbind(0))
            target_rows.extend(targets.detach().unbind(0))
            monitor_episode_returns.append(episode_return)
            monitor_discounted_returns.append(discounted_return)
            monitor_episode_lengths.append(episode_length)
            monitor_done_flags.append(done)
            monitor_step_rewards.extend(rewards)
            monitor_action_ids.extend(episode_action_ids)
            monitor_alphas.extend(episode_alphas)
            monitor_legal_fractions.extend(episode_legal_fractions)
            partner_monitor["episode_returns"].append(episode_return)
            partner_monitor["discounted_returns"].append(discounted_return)
            partner_monitor["episode_lengths"].append(episode_length)
            partner_monitor["done_flags"].append(done)
            partner_monitor["step_rewards"].extend(rewards)
            partner_monitor["action_ids"].extend(episode_action_ids)
            partner_monitor["alphas"].extend(episode_alphas)
            partner_monitor["legal_fractions"].extend(episode_legal_fractions)

    if not obs_rows:
        raise RuntimeError("classic rollout collector produced an empty RolloutBatch")

    monitor_by_partner: dict[str, Any] = {}
    for partner_id, payload in by_partner_monitor.items():
        legal_values = torch.tensor(payload["legal_fractions"], dtype=torch.float32)
        monitor_by_partner[partner_id] = {
            **_reward_monitor(
                episode_returns=payload["episode_returns"],
                discounted_returns=payload["discounted_returns"],
                episode_lengths=payload["episode_lengths"],
                done_flags=payload["done_flags"],
                step_rewards=payload["step_rewards"],
            ),
            **_alpha_monitor(payload["alphas"]),
            **_action_distribution(torch.tensor(payload["action_ids"], dtype=torch.long), actor_critic.num_actions),
            "legal_action_fraction": float(legal_values.mean().item()) if legal_values.numel() else 0.0,
        }
    legal_values = torch.tensor(monitor_legal_fractions, dtype=torch.float32)
    monitor = {
        **_reward_monitor(
            episode_returns=monitor_episode_returns,
            discounted_returns=monitor_discounted_returns,
            episode_lengths=monitor_episode_lengths,
            done_flags=monitor_done_flags,
            step_rewards=monitor_step_rewards,
        ),
        **_alpha_monitor(monitor_alphas),
        **_action_distribution(torch.tensor(monitor_action_ids, dtype=torch.long), actor_critic.num_actions),
        "legal_action_fraction": float(legal_values.mean().item()) if legal_values.numel() else 0.0,
        "by_partner": monitor_by_partner,
        "monitor_source": "source_training_rollout",
        "target_data_used": False,
    }

    return RolloutBatch(
        obs_ego=torch.stack(obs_rows, dim=0).to(device=target_device, dtype=torch.float32),
        global_state=torch.stack(global_rows, dim=0).to(device=target_device, dtype=torch.float32),
        g=torch.stack(g_rows, dim=0).to(device=target_device, dtype=torch.float32),
        phase=torch.stack(phase_rows, dim=0).to(device=target_device, dtype=torch.float32),
        belief_mean=torch.stack(mean_rows, dim=0).to(device=target_device, dtype=torch.float32),
        belief_diag_cov=torch.stack(diag_rows, dim=0).to(device=target_device, dtype=torch.float32),
        belief_alpha=torch.stack(alpha_rows, dim=0).to(device=target_device, dtype=torch.float32),
        legal_mask=torch.stack(mask_rows, dim=0).to(device=target_device, dtype=torch.bool),
        actions=torch.stack(action_rows, dim=0).to(device=target_device, dtype=torch.long),
        old_log_probs=torch.stack(log_prob_rows, dim=0).to(device=target_device, dtype=torch.float32),
        advantages=torch.stack(advantage_rows, dim=0).to(device=target_device, dtype=torch.float32),
        gae_targets=torch.stack(target_rows, dim=0).to(device=target_device, dtype=torch.float32),
        monitor=monitor,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _estimated_updates_remaining(
    cumulative_env_steps: int,
    target_env_steps: int,
    updates_completed: int,
) -> int:
    if target_env_steps <= 0 or updates_completed <= 0 or cumulative_env_steps <= 0:
        return 0
    remaining = max(int(target_env_steps) - int(cumulative_env_steps), 0)
    if remaining == 0:
        return 0
    average_rows = float(cumulative_env_steps) / float(updates_completed)
    return int(math.ceil(float(remaining) / max(average_rows, 1e-9)))


def _env_step_progress(cumulative_env_steps: int, target_env_steps: int) -> float:
    if target_env_steps <= 0:
        return 0.0
    return min(float(cumulative_env_steps) / float(target_env_steps), 1.0)


def _normalize_rollout_advantages(batch: RolloutBatch) -> RolloutBatch:
    advantages = batch.advantages.to(dtype=torch.float32)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    return RolloutBatch(
        obs_ego=batch.obs_ego,
        global_state=batch.global_state,
        g=batch.g,
        phase=batch.phase,
        belief_mean=batch.belief_mean,
        belief_diag_cov=batch.belief_diag_cov,
        belief_alpha=batch.belief_alpha,
        legal_mask=batch.legal_mask,
        actions=batch.actions,
        old_log_probs=batch.old_log_probs,
        advantages=advantages,
        gae_targets=batch.gae_targets,
        monitor=batch.monitor,
    )


def evaluate_closed_loop_classic(
    adapter: Any,
    partners: Sequence[tuple[str, Any]],
    *,
    actor_critic: MAPPOActorCritic,
    srvf_heads: NeuralSRVFHeads,
    belief_template: SRVFBelief,
    episodes_per_partner: int = 20,
    horizon: int = 400,
    gamma: float = 0.99,
    seed: int = 0,
) -> Mapping[str, Any]:
    """Evaluate deterministic closed-loop target returns."""
    if not partners:
        raise ValueError("partners must be nonempty")
    by_partner: dict[str, Any] = {}
    all_returns: list[float] = []
    for partner_idx, (partner_id, partner) in enumerate(partners):
        returns: list[float] = []
        lengths: list[int] = []
        alphas: list[float] = []
        fallback_count = 0
        action_count = 0
        for episode_idx in range(int(episodes_per_partner)):
            rollout_seed = int(seed + partner_idx * 100_000 + episode_idx)
            observation = adapter.reset(rollout_seed)
            partner_state = partner.reset(rollout_seed)
            rng = torch.Generator().manual_seed(rollout_seed + 20_000)
            belief = SRVFBelief(
                factor_dim=belief_template.factor_dim,
                dz_dim=belief_template.dz_dim,
                prior_mean=belief_template.prior_mean.detach().cpu(),
                prior_covariance=belief_template.prior_covariance.detach().cpu(),
                response_noise_var=belief_template.response_noise_var,
                source_beta=belief_template.source_beta.detach().cpu(),
                feature_mean=belief_template.feature_mean.detach().cpu(),
                feature_std=belief_template.feature_std.detach().cpu(),
                alpha_low=belief_template.alpha_low,
                alpha_high=belief_template.alpha_high,
                reliability_tau=belief_template.reliability_tau,
                ood_distance_threshold=belief_template.ood_distance_threshold,
                ood_alpha_cap=belief_template.ood_alpha_cap,
                device=actor_critic.device,
            ).reset(1)
            total = 0.0
            discount = 1.0
            done = False
            for step_idx in range(int(horizon)):
                g = _adapter_public_g(adapter)
                obs = _adapter_ego_obs(adapter, actor_critic.obs_dim)
                global_state = _adapter_global_state(adapter, actor_critic.global_state_dim)
                legal = _adapter_legal_mask(adapter, actor_critic.num_actions)
                phase = torch.zeros(1, actor_critic.phase_dim, dtype=torch.float32, device=actor_critic.device)
                mean = belief.mean().detach().reshape(actor_critic.factor_dim)
                diag = belief.diag_covariance().detach().reshape(actor_critic.factor_dim)
                alpha = belief.alpha.detach().reshape(())
                alpha_value = float(alpha.item())
                alphas.append(alpha_value)
                if alpha_value <= 0.25:
                    fallback_count += 1
                action_count += 1
                with torch.inference_mode():
                    output = actor_critic(
                        obs.unsqueeze(0).to(actor_critic.device),
                        global_state.unsqueeze(0).to(actor_critic.device),
                        g.unsqueeze(0).to(actor_critic.device),
                        mean.unsqueeze(0).to(actor_critic.device),
                        diag.unsqueeze(0).to(actor_critic.device),
                        alpha.unsqueeze(0).to(actor_critic.device),
                        phase,
                        legal.unsqueeze(0).to(actor_critic.device),
                        srvf_heads,
                    )
                    action = torch.argmax(output.logits, dim=-1)
                partner_action, partner_state = partner.act(observation, partner_state, rng)
                action_for_update = action.detach().clone().to(actor_critic.device)
                transition = adapter.step(int(action.item()), int(partner_action))
                next_g = _step_state_g(transition)
                with torch.no_grad():
                    belief.update(
                        g.unsqueeze(0).to(actor_critic.device),
                        action_for_update,
                        (next_g - g).unsqueeze(0).to(actor_critic.device),
                        srvf_heads,
                    )
                total += discount * float(transition.reward)
                discount *= float(gamma)
                observation = transition.observation
                done = bool(transition.done)
                if done:
                    lengths.append(step_idx + 1)
                    break
            if not done:
                lengths.append(int(horizon))
            returns.append(total)
            all_returns.append(total)
        values = torch.tensor(returns, dtype=torch.float32)
        by_partner[str(partner_id)] = {
            "episodes": int(episodes_per_partner),
            "mean_return": float(values.mean().item()),
            "std_return": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
            "min_return": float(values.min().item()),
            "max_return": float(values.max().item()),
            "nonzero_return_fraction": float((values.abs() > 1e-8).to(dtype=torch.float32).mean().item()),
            "mean_episode_length": float(torch.tensor(lengths, dtype=torch.float32).mean().item()),
            "mean_alpha": float(torch.tensor(alphas, dtype=torch.float32).mean().item()) if alphas else 0.0,
            "fallback_rate": 0.0 if action_count == 0 else float(fallback_count / action_count),
        }
    aggregate = torch.tensor(all_returns, dtype=torch.float32)
    return {
        "by_partner": by_partner,
        "aggregate": {
            "episodes": int(aggregate.numel()),
            "mean_return": float(aggregate.mean().item()) if aggregate.numel() else 0.0,
            "std_return": float(aggregate.std(unbiased=False).item()) if aggregate.numel() > 1 else 0.0,
            "nonzero_return_fraction": float((aggregate.abs() > 1e-8).to(dtype=torch.float32).mean().item()) if aggregate.numel() else 0.0,
        },
    }


def evaluate_offline_target_regret(
    table: IRFTable,
    *,
    actor_critic: MAPPOActorCritic,
    srvf_heads: NeuralSRVFHeads,
    belief_template: SRVFBelief,
) -> Mapping[str, Any]:
    """Evaluate target action regret using response-only posterior adaptation."""
    del actor_critic
    by_partner: dict[str, Any] = {}
    all_regrets: list[float] = []
    all_population_regrets: list[float] = []
    for partner_idx, partner_id in enumerate(table.partner_ids):
        device = srvf_heads.device
        belief = SRVFBelief(
            factor_dim=belief_template.factor_dim,
            dz_dim=belief_template.dz_dim,
            prior_mean=belief_template.prior_mean.detach().cpu(),
            prior_covariance=belief_template.prior_covariance.detach().cpu(),
            response_noise_var=belief_template.response_noise_var,
            source_beta=belief_template.source_beta.detach().cpu(),
            feature_mean=belief_template.feature_mean.detach().cpu(),
            feature_std=belief_template.feature_std.detach().cpu(),
            alpha_low=belief_template.alpha_low,
            alpha_high=belief_template.alpha_high,
            reliability_tau=belief_template.reliability_tau,
            ood_distance_threshold=belief_template.ood_distance_threshold,
            ood_alpha_cap=belief_template.ood_alpha_cap,
            device=device,
        ).reset(1)
        partner_valid = table.valid_mask[:, :, partner_idx].to(dtype=torch.bool)
        valid_state_idx, valid_action_idx = partner_valid.nonzero(as_tuple=True)
        if valid_state_idx.numel() > 0:
            g_batch = table.state_g.index_select(0, valid_state_idx).to(device=device, dtype=torch.float32)
            action_batch = valid_action_idx.to(device=device, dtype=torch.long)
            delta_batch = table.delta_z[
                valid_state_idx,
                valid_action_idx,
                partner_idx,
            ].to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                head_out = srvf_heads(g_batch)
                batch_index = torch.arange(valid_state_idx.numel(), dtype=torch.long, device=device)
                r0_selected = head_out.r0[batch_index, action_batch]
                rc_selected = head_out.r_c[batch_index, action_batch]
                residual = delta_batch - r0_selected
                inv_noise = 1.0 / belief.response_noise_var
                precision_inc = torch.einsum("bdk,bdl->bkl", rc_selected, rc_selected) * inv_noise
                eta_inc = torch.einsum("bdk,bd->bk", rc_selected, residual) * inv_noise
                belief.Lambda = belief.prior_precision.unsqueeze(0) + precision_inc.sum(dim=0, keepdim=True)
                belief.eta = belief.prior_eta.unsqueeze(0) + eta_inc.sum(dim=0, keepdim=True)
                belief.count = torch.tensor([int(valid_state_idx.numel())], dtype=torch.long, device=device)
                mu = belief.mean()
                covariance = belief.covariance()
                response_mse = residual[-1].square().mean().reshape(1)
                beta_dist = belief._beta_support_distance(mu)
                contraction = belief._posterior_contraction(covariance)
                belief.alpha = belief.calibrated_alpha(response_mse, beta_dist, contraction)
        mean = belief.mean().detach().expand(table.num_states, -1)
        alpha = belief.alpha.detach().expand(table.num_states)
        with torch.inference_mode():
            scores = SRVFBelief.score_from_posterior(
                table.state_g.to(device),
                srvf_heads,
                mean.to(device),
                alpha.to(device),
            ).detach().cpu()
            population_scores = SRVFBelief.score_from_posterior(
                table.state_g.to(device),
                srvf_heads,
                torch.zeros_like(mean).to(device),
                torch.zeros_like(alpha).to(device),
            ).detach().cpu()
        valid = table.valid_mask[:, :, partner_idx].to(dtype=torch.bool)
        values = table.a_raw[:, :, partner_idx].to(dtype=torch.float32)
        masked_values = values.masked_fill(~valid, -1.0e9)
        oracle = masked_values.max(dim=1).values
        chosen = scores.masked_fill(~valid, -1.0e9).argmax(dim=1)
        pop_chosen = population_scores.masked_fill(~valid, -1.0e9).argmax(dim=1)
        row = torch.arange(table.num_states)
        regrets = oracle - values[row, chosen]
        population_regrets = oracle - values[row, pop_chosen]
        eval_mask = valid.any(dim=1)
        regrets = regrets[eval_mask]
        population_regrets = population_regrets[eval_mask]
        all_regrets.extend(float(x) for x in regrets)
        all_population_regrets.extend(float(x) for x in population_regrets)
        by_partner[str(partner_id)] = {
            "action_regret": float(regrets.mean().item()) if regrets.numel() else 0.0,
            "population_action_regret": float(population_regrets.mean().item()) if population_regrets.numel() else 0.0,
            "evaluated_states": int(eval_mask.sum().item()),
            "final_alpha": float(belief.alpha.detach().cpu().reshape(()).item()),
            "posterior_observations": int(belief.count.detach().cpu().reshape(()).item()),
        }
    aggregate_regrets = torch.tensor(all_regrets, dtype=torch.float32)
    aggregate_population = torch.tensor(all_population_regrets, dtype=torch.float32)
    return {
        "by_partner": by_partner,
        "aggregate": {
            "action_regret": float(aggregate_regrets.mean().item()) if aggregate_regrets.numel() else 0.0,
            "population_action_regret": float(aggregate_population.mean().item()) if aggregate_population.numel() else 0.0,
            "evaluated_rows": int(aggregate_regrets.numel()),
        },
        "leakage_guard": {
            "target_rewards_used_for_training": False,
            "target_action_labels_used_for_training": False,
            "target_partner_ids_used_for_alpha": False,
            "target_responses_used_for_posterior": True,
        },
    }


def _make_belief_template(
    init: SourceFactorInit,
    *,
    dz_dim: int,
    device: torch.device,
    response_noise_var: float,
    alpha_low: float,
    alpha_high: float,
    reliability_tau: float,
    ood_alpha_cap: float,
) -> SRVFBelief:
    beta = init.beta_source.to(dtype=torch.float32)
    if beta.shape[0] > 1:
        distances = torch.cdist(beta, beta)
        distances = distances + torch.eye(beta.shape[0]) * 1.0e9
        threshold = float(distances.min(dim=1).values.quantile(0.75).item())
    else:
        threshold = float("inf")
    return SRVFBelief(
        factor_dim=beta.shape[1],
        dz_dim=dz_dim,
        prior_mean=init.prior_mean,
        prior_covariance=init.prior_covariance,
        response_noise_var=response_noise_var,
        source_beta=beta,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        reliability_tau=reliability_tau,
        ood_distance_threshold=threshold,
        ood_alpha_cap=ood_alpha_cap,
        device=device,
    )


_PARALLEL_ROLLOUT_CONTEXT: dict[str, Any] = {}


def _parallel_rollout_worker_init(config: Mapping[str, Any]) -> None:
    torch.set_num_threads(max(1, int(config.get("torch_num_threads", 1))))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(config.get("torch_num_threads", 1)))))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, int(config.get("torch_num_threads", 1)))))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, int(config.get("torch_num_threads", 1)))))

    from raob.benchmarks.overcooked_classic import (
        ClassicOvercookedBenchmarkAdapter,
        canonical_classic_layout_name,
    )

    adapter = ClassicOvercookedBenchmarkAdapter(
        layout=canonical_classic_layout_name(str(config["layout"])),
        horizon=int(config["horizon"]),
        old_dynamics=bool(config["old_dynamics"]),
    )
    _PARALLEL_ROLLOUT_CONTEXT.clear()
    _PARALLEL_ROLLOUT_CONTEXT.update(
        {
            "adapter": adapter,
            "external_root": str(config["external_root"]),
            "worker_policy_device": str(config.get("worker_policy_device", "cpu")),
            "deterministic_partners": bool(config.get("deterministic_partners", True)),
            "cached_partner_key": None,
            "cached_source_partners": None,
        }
    )


def _worker_source_partners(spec_payloads: Sequence[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    partner_key = tuple(str(payload["partner_id"]) for payload in spec_payloads)
    if _PARALLEL_ROLLOUT_CONTEXT.get("cached_partner_key") == partner_key:
        cached = _PARALLEL_ROLLOUT_CONTEXT.get("cached_source_partners")
        if cached is not None:
            return cached

    from raob.benchmarks.goat_classic import make_goat_partners
    from raob.benchmarks.partners import PartnerSpec

    specs = [PartnerSpec.from_mapping(payload) for payload in spec_payloads]
    partners = make_goat_partners(
        specs,
        external_root=str(_PARALLEL_ROLLOUT_CONTEXT["external_root"]),
        device=str(_PARALLEL_ROLLOUT_CONTEXT["worker_policy_device"]),
        agent_index=1,
        deterministic=bool(_PARALLEL_ROLLOUT_CONTEXT["deterministic_partners"]),
    )
    _PARALLEL_ROLLOUT_CONTEXT["cached_partner_key"] = partner_key
    _PARALLEL_ROLLOUT_CONTEXT["cached_source_partners"] = partners
    return partners


def _parallel_rollout_worker_collect(task: Mapping[str, Any]) -> RolloutBatch:
    if "adapter" not in _PARALLEL_ROLLOUT_CONTEXT:
        raise RuntimeError("parallel rollout worker was not initialized")
    device = torch.device("cpu")
    dims = dict(task["dims"])
    heads = NeuralSRVFHeads(
        g_dim=int(dims["g_dim"]),
        num_actions=int(dims["num_actions"]),
        factor_dim=int(dims["factor_dim"]),
        hidden_dim=int(dims["hidden_dim"]),
        device=device,
    )
    heads.load_state_dict(task["heads_state"])
    heads.eval()
    actor_critic = MAPPOActorCritic(
        obs_dim=int(dims["obs_dim"]),
        global_state_dim=int(dims["global_state_dim"]),
        g_dim=int(dims["g_dim"]),
        num_actions=int(dims["num_actions"]),
        factor_dim=int(dims["factor_dim"]),
        hidden_dim=int(dims["hidden_dim"]),
        tau=float(dims["tau"]),
        phase_dim=0,
        detach_srvf_in_actor=True,
        device=device,
    )
    actor_critic.load_state_dict(task["actor_state"])
    actor_critic.eval()
    init = SourceFactorInit(
        beta_source=task["beta_source"].to(dtype=torch.float32),
        prior_mean=task["prior_mean"].to(dtype=torch.float32),
        prior_covariance=task["prior_covariance"].to(dtype=torch.float32),
        partner_ids=tuple(str(x) for x in task["partner_ids"]),
        diagnostics={},
    )
    belief = _make_belief_template(
        init,
        dz_dim=int(dims["g_dim"]),
        device=device,
        response_noise_var=float(task["response_noise_var"]),
        alpha_low=float(task["alpha_low"]),
        alpha_high=float(task["alpha_high"]),
        reliability_tau=float(task["reliability_tau"]),
        ood_alpha_cap=float(task["ood_alpha_cap"]),
    )
    partners = _worker_source_partners(task["source_specs"])
    rollout = collect_classic_rollout_batch(
        _PARALLEL_ROLLOUT_CONTEXT["adapter"],
        partners,
        actor_critic=actor_critic,
        srvf_heads=heads,
        belief=belief,
        episodes_per_partner=int(task["episodes_per_partner"]),
        horizon=int(task["horizon"]),
        gamma=float(task["gamma"]),
        gae_lambda=float(task["gae_lambda"]),
        deterministic=False,
        seed=int(task["seed"]),
        device=device,
    )
    monitor = dict(rollout.monitor)
    monitor["worker_index"] = int(task["worker_index"])
    monitor["worker_partner_ids"] = [str(payload["partner_id"]) for payload in task["source_specs"]]
    return replace(rollout, monitor=monitor)


def _worker_spec_slices(
    source_specs: Sequence[Any],
    worker_count: int,
) -> list[list[Mapping[str, Any]]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not source_specs:
        raise ValueError("source_specs must be nonempty")
    json_specs = [spec.to_json() if hasattr(spec, "to_json") else dict(spec) for spec in source_specs]
    return [[json_specs[idx % len(json_specs)]] for idx in range(int(worker_count))]


def _collect_parallel_rollout_batch(
    *,
    executor: ProcessPoolExecutor,
    worker_count: int,
    source_specs: Sequence[Any],
    actor_critic: MAPPOActorCritic,
    srvf_heads: NeuralSRVFHeads,
    init: SourceFactorInit,
    args: argparse.Namespace,
    seed: int,
    obs_dim: int,
    global_state_dim: int,
    device: torch.device,
) -> RolloutBatch:
    dims = {
        "obs_dim": int(obs_dim),
        "global_state_dim": int(global_state_dim),
        "g_dim": int(srvf_heads.g_dim),
        "num_actions": int(srvf_heads.num_actions),
        "factor_dim": int(srvf_heads.factor_dim),
        "hidden_dim": int(args.hidden_dim),
        "tau": float(args.tau),
    }
    actor_state = _cpu_state_dict(actor_critic)
    heads_state = _cpu_state_dict(srvf_heads)
    spec_slices = _worker_spec_slices(source_specs, worker_count)
    tasks = []
    for worker_idx, specs in enumerate(spec_slices):
        tasks.append(
            {
                "worker_index": int(worker_idx),
                "source_specs": specs,
                "actor_state": actor_state,
                "heads_state": heads_state,
                "dims": dims,
                "beta_source": init.beta_source.detach().cpu(),
                "prior_mean": init.prior_mean.detach().cpu(),
                "prior_covariance": init.prior_covariance.detach().cpu(),
                "partner_ids": init.partner_ids,
                "response_noise_var": float(args.response_noise_var),
                "alpha_low": float(args.alpha_low),
                "alpha_high": float(args.alpha_high),
                "reliability_tau": float(args.reliability_tau),
                "ood_alpha_cap": float(args.ood_alpha_cap),
                "episodes_per_partner": int(args.rollout_episodes_per_partner),
                "horizon": int(args.horizon),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "seed": int(seed) + worker_idx * 10_000_000,
            }
        )
    batches = list(executor.map(_parallel_rollout_worker_collect, tasks))
    return _concat_rollout_batches(batches, device=device)


def train_one_seed(
    *,
    seed: int,
    adapter: Any,
    source_specs: Sequence[Any],
    source_partners: Sequence[tuple[str, Any]],
    target_partners: Sequence[tuple[str, Any]],
    source_table: IRFTable,
    target_table: IRFTable,
    init: SourceFactorInit,
    args: argparse.Namespace,
    output_root: Path,
    obs_dim: int,
    global_state_dim: int,
    device: torch.device,
) -> Mapping[str, Any]:
    torch.manual_seed(int(seed))
    heads = NeuralSRVFHeads(
        g_dim=source_table.dz_dim,
        num_actions=source_table.num_actions,
        factor_dim=int(args.factor_dim),
        hidden_dim=int(args.hidden_dim),
        device=device,
    )
    actor_critic = MAPPOActorCritic(
        obs_dim=obs_dim,
        global_state_dim=global_state_dim,
        g_dim=source_table.dz_dim,
        num_actions=source_table.num_actions,
        factor_dim=int(args.factor_dim),
        hidden_dim=int(args.hidden_dim),
        tau=float(args.tau),
        phase_dim=0,
        detach_srvf_in_actor=True,
        device=device,
    )
    unified = UnifiedLoss(
        actor_critic,
        heads,
        clip_epsilon=float(args.clip_epsilon),
        value_coef=float(args.value_coef),
        lambda_delta=float(args.lambda_delta),
        lambda_a=float(args.lambda_a),
        entropy_coef=float(args.entropy_coef),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        list(actor_critic.parameters()) + list(heads.parameters()),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    source_batches = iter_source_batches(
        source_table,
        init.beta_source,
        batch_size=int(args.source_batch_size),
        shuffle=True,
        seed=int(seed),
        device=device,
    )
    worker_count = max(1, int(getattr(args, "workers", 1)))
    learner_epochs = max(1, int(getattr(args, "learner_epochs", 1)))
    min_rows = max(0, int(getattr(args, "min_rollout_rows_per_update", 0)))
    executor: ProcessPoolExecutor | None = None
    if worker_count > 1:
        worker_config = {
            "layout": args.layout,
            "horizon": int(args.horizon),
            "old_dynamics": bool(args.old_dynamics),
            "external_root": args.external_root,
            "worker_policy_device": str(getattr(args, "worker_policy_device", "cpu")),
            "deterministic_partners": bool(args.deterministic_partners),
            "torch_num_threads": int(getattr(args, "torch_num_threads", 1)),
        }
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_parallel_rollout_worker_init,
            initargs=(worker_config,),
        )
    metrics_path = output_root / f"train_metrics_seed{seed}.jsonl"
    monitor_path = output_root / f"rollout_monitor_seed{seed}.jsonl"
    start_time = time.time()
    fixed_update_limit = int(getattr(args, "updates", 0))
    target_env_steps = int(getattr(args, "target_env_steps", 0))
    if fixed_update_limit < 0:
        raise ValueError("updates must be nonnegative")
    if target_env_steps < 0:
        raise ValueError("target_env_steps must be nonnegative")
    step_budget_mode = fixed_update_limit <= 0
    if step_budget_mode and target_env_steps <= 0:
        raise ValueError("target_env_steps must be positive when updates is 0")
    update_budget_mode = "target_env_steps" if step_budget_mode else "fixed_updates"
    if not step_budget_mode:
        target_env_steps = 0
    cumulative_env_steps = 0
    update_idx = 0
    try:
        while True:
            if step_budget_mode:
                if cumulative_env_steps >= target_env_steps:
                    break
            elif update_idx >= fixed_update_limit:
                break
            update_start = time.time()
            collector_start = time.time()
            if executor is None:
                rollout_belief = _make_belief_template(
                    init,
                    dz_dim=source_table.dz_dim,
                    device=device,
                    response_noise_var=float(args.response_noise_var),
                    alpha_low=float(args.alpha_low),
                    alpha_high=float(args.alpha_high),
                    reliability_tau=float(args.reliability_tau),
                    ood_alpha_cap=float(args.ood_alpha_cap),
                )
                rollout = collect_classic_rollout_batch(
                    adapter,
                    source_partners,
                    actor_critic=actor_critic,
                    srvf_heads=heads,
                    belief=rollout_belief,
                    episodes_per_partner=int(args.rollout_episodes_per_partner),
                    horizon=int(args.horizon),
                    gamma=float(args.gamma),
                    gae_lambda=float(args.gae_lambda),
                    deterministic=False,
                    seed=int(seed) * 1_000_000 + update_idx,
                    device=device,
                )
                rollout = replace(
                    rollout,
                    monitor={
                        **dict(rollout.monitor),
                        "worker_count": 1,
                        "resource_mode": "single_rollout_gpu_learner",
                    },
                )
            else:
                rollout = _collect_parallel_rollout_batch(
                    executor=executor,
                    worker_count=worker_count,
                    source_specs=source_specs,
                    actor_critic=actor_critic,
                    srvf_heads=heads,
                    init=init,
                    args=args,
                    seed=int(seed) * 1_000_000 + update_idx,
                    obs_dim=obs_dim,
                    global_state_dim=global_state_dim,
                    device=device,
                )
                rollout = replace(
                    rollout,
                    monitor={
                        **dict(rollout.monitor),
                        "resource_mode": "parallel_rollout_gpu_learner",
                    },
                )
            collector_elapsed = time.time() - collector_start
            if min_rows > 0 and int(rollout.actions.numel()) < min_rows:
                raise RuntimeError(
                    f"parallel rollout rows {int(rollout.actions.numel())} below required minimum {min_rows}"
                )
            rollout = _normalize_rollout_advantages(rollout)
            learner_start = time.time()
            logs: Mapping[str, torch.Tensor] = {}
            loss = torch.zeros((), dtype=torch.float32, device=device)
            for learner_epoch in range(learner_epochs):
                source_batch = source_batches[(update_idx * learner_epochs + learner_epoch) % len(source_batches)]
                loss, logs = unified.compute(rollout, source_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor_critic.parameters()) + list(heads.parameters()),
                    float(args.max_grad_norm),
                )
                optimizer.step()
            learner_elapsed = time.time() - learner_start
            update_elapsed = time.time() - update_start
            rollout_rows = int(rollout.actions.numel())
            cumulative_env_steps += rollout_rows
            updates_completed = update_idx + 1
            rows_per_sec = float(rollout_rows) / max(collector_elapsed, 1e-9)
            estimated_updates_remaining = (
                _estimated_updates_remaining(
                    cumulative_env_steps,
                    target_env_steps,
                    updates_completed,
                )
                if step_budget_mode
                else max(fixed_update_limit - updates_completed, 0)
            )
            env_step_progress = _env_step_progress(cumulative_env_steps, target_env_steps)
            is_final_update = (
                cumulative_env_steps >= target_env_steps
                if step_budget_mode
                else updates_completed >= fixed_update_limit
            )
            progress_payload = {
                "update_budget_mode": update_budget_mode,
                "updates_completed": updates_completed,
                "target_env_steps": target_env_steps,
                "cumulative_env_steps": cumulative_env_steps,
                "env_step_progress": env_step_progress,
                "estimated_updates_remaining": estimated_updates_remaining,
            }
            monitor_every = max(1, int(getattr(args, "monitor_every", 1)))
            log_every = max(1, int(args.log_every))
            if update_idx % monitor_every == 0 or is_final_update:
                _append_jsonl(
                    monitor_path,
                    {
                        "seed": int(seed),
                        "update": int(update_idx),
                        **progress_payload,
                        "elapsed_sec": time.time() - start_time,
                        "collector_elapsed_sec": collector_elapsed,
                        "learner_elapsed_sec": learner_elapsed,
                        "optimizer_elapsed_sec": learner_elapsed,
                        "update_elapsed_sec": update_elapsed,
                        "rows_per_sec": rows_per_sec,
                        "gpu_learner_epochs": learner_epochs,
                        **dict(rollout.monitor),
                    },
                )
                _write_json(
                    output_root / "status.json",
                    {
                        "status": "running",
                        "stage": "train_eval_seed",
                        "active_seed": int(seed),
                        "completed_seeds": [
                            int(value) for value in getattr(args, "completed_seeds", [])
                        ],
                        **progress_payload,
                        "resource_mode": rollout.monitor.get("resource_mode", "unknown"),
                        "worker_count": worker_count,
                        "learner_epochs": learner_epochs,
                        "run_dir": str(output_root),
                    },
                )
            if update_idx % log_every == 0 or is_final_update:
                _append_jsonl(
                    metrics_path,
                    {
                        "seed": int(seed),
                        "update": int(update_idx),
                        **progress_payload,
                        "elapsed_sec": time.time() - start_time,
                        "rollout_rows": rollout_rows,
                        "collector_elapsed_sec": collector_elapsed,
                        "learner_elapsed_sec": learner_elapsed,
                        "optimizer_elapsed_sec": learner_elapsed,
                        "update_elapsed_sec": update_elapsed,
                        "rows_per_sec": rows_per_sec,
                        "worker_count": worker_count,
                        "gpu_learner_epochs": learner_epochs,
                        **{key: value for key, value in logs.items()},
                        "mean_advantage": float(rollout.advantages.mean().detach().cpu().item()),
                        "mean_target": float(rollout.gae_targets.mean().detach().cpu().item()),
                        "mean_alpha": float(rollout.belief_alpha.mean().detach().cpu().item()),
                        "mean_episode_return": rollout.monitor.get("episode_return_mean", 0.0),
                        "mean_step_reward": rollout.monitor.get("step_reward_mean", 0.0),
                        "nonzero_step_reward_fraction": rollout.monitor.get("step_reward_nonzero_fraction", 0.0),
                        "mean_episode_length": rollout.monitor.get("mean_episode_length", 0.0),
                        "action_entropy": rollout.monitor.get("action_entropy", 0.0),
                        "fallback_rate": rollout.monitor.get("fallback_rate", 0.0),
                    },
                )
            update_idx += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    checkpoint_path = output_root / f"checkpoint_seed{seed}.pt"
    torch.save(
        {
            "kind": "srvf_mappo_checkpoint",
            "seed": int(seed),
            "actor_critic": actor_critic.cpu().state_dict(),
            "srvf_heads": heads.cpu().state_dict(),
            "dims": {
                "obs_dim": obs_dim,
                "global_state_dim": global_state_dim,
                "g_dim": source_table.dz_dim,
                "num_actions": source_table.num_actions,
                "factor_dim": int(args.factor_dim),
                "hidden_dim": int(args.hidden_dim),
            },
            "source_factor_init": _jsonable(init.diagnostics),
            "config": vars(args),
        },
        checkpoint_path,
    )
    actor_critic.to(device)
    heads.to(device)
    eval_belief = _make_belief_template(
        init,
        dz_dim=source_table.dz_dim,
        device=device,
        response_noise_var=float(args.response_noise_var),
        alpha_low=float(args.alpha_low),
        alpha_high=float(args.alpha_high),
        reliability_tau=float(args.reliability_tau),
        ood_alpha_cap=float(args.ood_alpha_cap),
    )
    closed_loop = evaluate_closed_loop_classic(
        adapter,
        target_partners,
        actor_critic=actor_critic,
        srvf_heads=heads,
        belief_template=eval_belief,
        episodes_per_partner=int(args.eval_episodes_per_partner),
        horizon=int(args.horizon),
        gamma=float(args.gamma),
        seed=int(seed) + 5_000_000,
    )
    offline = evaluate_offline_target_regret(
        target_table,
        actor_critic=actor_critic,
        srvf_heads=heads,
        belief_template=eval_belief,
    )
    result = {
        "seed": int(seed),
        "checkpoint": str(checkpoint_path),
        "training": {
            "update_budget_mode": update_budget_mode,
            "updates_completed": int(update_idx),
            "target_env_steps": int(target_env_steps),
            "cumulative_env_steps": int(cumulative_env_steps),
        },
        "closed_loop": closed_loop,
        "offline_regret": offline,
    }
    _write_json(output_root / f"eval_seed{seed}.json", result)
    return result


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# SRVF-MAPPO Classic Formal Run",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Layout: `{summary.get('layout')}`",
        f"- Seeds: `{summary.get('seeds')}`",
        f"- Update budget mode: `{summary.get('update_budget_mode')}`",
        f"- Target env steps per seed: `{summary.get('target_env_steps')}`",
        f"- Seed training: `{summary.get('seed_training')}`",
        f"- Source partners: `{summary.get('source_partner_ids')}`",
        f"- Target partners: `{summary.get('target_partner_ids')}`",
        "",
        "## Aggregate",
        "",
        f"- Closed-loop mean return: `{summary.get('closed_loop_mean_return')}`",
        f"- Closed-loop nonzero fraction: `{summary.get('closed_loop_nonzero_return_fraction')}`",
        f"- Offline action regret: `{summary.get('offline_action_regret')}`",
        f"- Offline population action regret: `{summary.get('offline_population_action_regret')}`",
        "",
        "## Boundary",
        "",
        "This run reports the SRVF-MAPPO method only. Target rewards, target action labels, "
        "target identity, and target partner ids are not used for training or alpha calibration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_formal_classic(args: argparse.Namespace) -> Mapping[str, Any]:
    from raob.benchmarks.goat_classic import (
        GOATClassicEgoPolicy,
        make_goat_partners,
        select_goat_partner_specs,
    )
    from raob.benchmarks.overcooked_classic import (
        ClassicOvercookedBenchmarkAdapter,
        canonical_classic_layout_name,
    )

    if bool(args.smoke):
        args.max_states = min(int(args.max_states), 8)
        args.reservoir_episodes = min(int(args.reservoir_episodes), 2)
        args.v0_episodes_per_partner = min(int(args.v0_episodes_per_partner), 1)
        args.v0_epochs = min(int(args.v0_epochs), 1)
        args.updates = 2
        args.target_env_steps = 0
        args.eval_episodes_per_partner = min(int(args.eval_episodes_per_partner), 1)
        args.horizon = min(int(args.horizon), 20)
        args.source_count = int(args.source_count or 2)
        args.target_count = int(args.target_count or 1)
        args.seeds = str(args.seeds).split(",")[0]

    torch.set_num_threads(max(1, int(getattr(args, "torch_num_threads", 1))))
    requested_layout = str(args.layout)
    canonical_layout = canonical_classic_layout_name(requested_layout)
    args.layout = canonical_layout
    if requested_layout != canonical_layout:
        setattr(args, "requested_layout", requested_layout)
    device = _resolve_device(args.device)
    run_id = args.run_id or f"srvf_mappo_classic_formal_{time.strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "status.json",
        {
            "status": "running",
            "stage": "partner_loading",
            "run_dir": str(output_root),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    source_specs, target_specs = select_goat_partner_specs(
        args.population,
        external_root=args.external_root,
        layout=args.layout,
        source_count=args.source_count,
        target_count=args.target_count,
    )
    source_partners = make_goat_partners(
        source_specs,
        external_root=args.external_root,
        device=args.policy_device,
        agent_index=1,
        deterministic=bool(args.deterministic_partners),
    )
    target_partners = make_goat_partners(
        target_specs,
        external_root=args.external_root,
        device=args.policy_device,
        agent_index=1,
        deterministic=bool(args.deterministic_partners),
    )
    ego_spec = next(
        (spec for spec in source_specs if str(spec.metadata.get("level", "")) == "final"),
        source_specs[0],
    )
    ego_policy = GOATClassicEgoPolicy(
        ego_spec,
        external_root=args.external_root,
        device=args.policy_device,
        agent_index=0,
        deterministic=True,
    )
    adapter = ClassicOvercookedBenchmarkAdapter(
        layout=args.layout,
        horizon=int(args.horizon),
        old_dynamics=bool(args.old_dynamics),
    )
    adapter.reset(int(args.seed))
    obs_dim = int(adapter.ego_observation_tensor(agent_index=0).numel())
    global_state_dim = int(adapter.global_state_tensor().numel())
    config_payload = {
        **vars(args),
        "effective_device": str(device),
        "obs_dim": obs_dim,
        "global_state_dim": global_state_dim,
        "source_partner_ids": [spec.partner_id for spec in source_specs],
        "target_partner_ids": [spec.partner_id for spec in target_specs],
    }
    _write_json(output_root / "config.json", config_payload)
    _write_json(
        output_root / "partner_split.json",
        {
            "source": [spec.to_json() for spec in source_specs],
            "target": [spec.to_json() for spec in target_specs],
        },
    )

    _write_json(output_root / "status.json", {"status": "running", "stage": "v0_data", "run_dir": str(output_root)})
    v0_batch = collect_v0_training_batch(
        adapter,
        source_partners,
        ego_policy=ego_policy,
        episodes_per_partner=int(args.v0_episodes_per_partner),
        horizon=int(args.horizon),
        gamma=float(args.gamma),
        seed=int(args.seed),
    )
    _write_json(output_root / "status.json", {"status": "running", "stage": "v0_train", "run_dir": str(output_root)})
    v0, v0_metrics = train_v0_ensemble(
        v0_batch,
        ensemble_size=int(args.v0_ensemble_size),
        epochs=int(args.v0_epochs),
        batch_size=int(args.v0_batch_size),
        learning_rate=float(args.v0_learning_rate),
        seed=int(args.seed),
        device=device,
    )
    torch.save(v0.state_dict_payload(), output_root / "v0_checkpoint.pt")
    _write_json(output_root / "v0_metrics.json", v0_metrics)

    _write_json(output_root / "status.json", {"status": "running", "stage": "reservoir", "run_dir": str(output_root)})
    reservoir = collect_reservoir_snapshots(
        adapter,
        source_partners,
        ego_policy=ego_policy,
        max_states=int(args.max_states),
        episodes=int(args.reservoir_episodes),
        horizon=int(args.horizon),
        seed=int(args.seed) + 100_000,
    )
    _write_json(output_root / "status.json", {"status": "running", "stage": "source_irf", "run_dir": str(output_root)})
    source_table = collect_v0_residual_irf_table(
        adapter,
        reservoir,
        source_partners,
        v0=v0,
        gamma=float(args.gamma),
        repeats=int(args.irf_repeats),
        seed=int(args.seed) + 200_000,
    )
    _write_json(output_root / "status.json", {"status": "running", "stage": "target_irf_eval_table", "run_dir": str(output_root)})
    target_table = collect_v0_residual_irf_table(
        adapter,
        reservoir,
        target_partners,
        v0=v0,
        gamma=float(args.gamma),
        repeats=int(args.irf_repeats),
        seed=int(args.seed) + 300_000,
    )
    torch.save({"kind": "srvf_mappo_irf_table", "table": source_table}, output_root / "source_table.pt")
    torch.save({"kind": "srvf_mappo_irf_table", "table": target_table}, output_root / "target_table.pt")
    init = initialize_source_beta(
        source_table,
        factor_dim=int(args.factor_dim),
        response_weight=float(args.response_weight),
        value_weight=float(args.value_weight),
    )
    _write_json(
        output_root / "source_factor_init.json",
        {
            "partner_ids": init.partner_ids,
            "diagnostics": init.diagnostics,
            "prior_mean": init.prior_mean,
            "prior_covariance": init.prior_covariance,
        },
    )

    seed_results = []
    for seed in [int(x) for x in str(args.seeds).split(",") if str(x).strip()]:
        update_budget_mode = "target_env_steps" if int(getattr(args, "updates", 0)) <= 0 else "fixed_updates"
        effective_target_env_steps = (
            int(getattr(args, "target_env_steps", 0))
            if update_budget_mode == "target_env_steps"
            else 0
        )
        completed_seed_ids = [int(result["seed"]) for result in seed_results]
        setattr(args, "completed_seeds", completed_seed_ids)
        _write_json(
            output_root / "status.json",
            {
                "status": "running",
                "stage": "train_eval_seed",
                "active_seed": int(seed),
                "completed_seeds": completed_seed_ids,
                "update_budget_mode": update_budget_mode,
                "target_env_steps": effective_target_env_steps,
                "resource_mode": (
                    "parallel_rollout_gpu_learner"
                    if int(getattr(args, "workers", 1)) > 1
                    else "single_rollout_gpu_learner"
                ),
                "worker_count": int(getattr(args, "workers", 1)),
                "learner_epochs": int(getattr(args, "learner_epochs", 1)),
                "run_dir": str(output_root),
            },
        )
        seed_results.append(
            train_one_seed(
                seed=seed,
                adapter=adapter,
                source_specs=source_specs,
                source_partners=source_partners,
                target_partners=target_partners,
                source_table=source_table,
                target_table=target_table,
                init=init,
                args=args,
                output_root=output_root,
                obs_dim=obs_dim,
                global_state_dim=global_state_dim,
                device=device,
            )
        )

    closed_returns = torch.tensor(
        [float(result["closed_loop"]["aggregate"]["mean_return"]) for result in seed_results],
        dtype=torch.float32,
    )
    closed_nonzero = torch.tensor(
        [float(result["closed_loop"]["aggregate"]["nonzero_return_fraction"]) for result in seed_results],
        dtype=torch.float32,
    )
    offline_regrets = torch.tensor(
        [float(result["offline_regret"]["aggregate"]["action_regret"]) for result in seed_results],
        dtype=torch.float32,
    )
    offline_population = torch.tensor(
        [float(result["offline_regret"]["aggregate"]["population_action_regret"]) for result in seed_results],
        dtype=torch.float32,
    )
    summary = {
        "status": "complete",
        "run_id": run_id,
        "layout": args.layout,
        "seeds": [int(result["seed"]) for result in seed_results],
        "update_budget_mode": "target_env_steps" if int(getattr(args, "updates", 0)) <= 0 else "fixed_updates",
        "target_env_steps": (
            int(getattr(args, "target_env_steps", 0))
            if int(getattr(args, "updates", 0)) <= 0
            else 0
        ),
        "seed_training": [result["training"] for result in seed_results],
        "source_partner_ids": [spec.partner_id for spec in source_specs],
        "target_partner_ids": [spec.partner_id for spec in target_specs],
        "closed_loop_mean_return": float(closed_returns.mean().item()),
        "closed_loop_nonzero_return_fraction": float(closed_nonzero.mean().item()),
        "offline_action_regret": float(offline_regrets.mean().item()),
        "offline_population_action_regret": float(offline_population.mean().item()),
        "seed_results": seed_results,
        "leakage_guard": {
            "target_rewards_used_for_training": False,
            "target_action_labels_used_for_training": False,
            "target_identity_used_for_training": False,
            "target_partner_ids_used_for_alpha": False,
            "target_responses_used_for_evaluation_posterior": True,
        },
    }
    if hasattr(args, "requested_layout"):
        summary["requested_layout"] = str(args.requested_layout)
    _write_json(output_root / "summary.json", summary)
    _write_report(output_root / "report.md", summary)
    _write_json(output_root / "status.json", {"status": "complete", "run_dir": str(output_root)})
    return summary


def _add_common_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--population", default="runs/goat_classic_population_cramped_room_lightformal_20260601/population.json")
    parser.add_argument("--external-root", default="external/goat_overcooked")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--source-count", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--worker-policy-device", default="cpu")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--old-dynamics", action="store_true")
    parser.set_defaults(deterministic_partners=True)
    parser.add_argument("--deterministic-partners", dest="deterministic_partners", action="store_true")
    parser.add_argument("--stochastic-partners", dest="deterministic_partners", action="store_false")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-states", type=int, default=256)
    parser.add_argument("--reservoir-episodes", type=int, default=16)
    parser.add_argument("--v0-episodes-per-partner", type=int, default=8)
    parser.add_argument("--v0-ensemble-size", type=int, default=3)
    parser.add_argument("--v0-epochs", type=int, default=100)
    parser.add_argument("--v0-batch-size", type=int, default=256)
    parser.add_argument("--v0-learning-rate", type=float, default=3e-4)
    parser.add_argument("--irf-repeats", type=int, default=1)
    parser.add_argument("--factor-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--response-noise-var", type=float, default=1.0)
    parser.add_argument("--alpha-low", type=float, default=0.0)
    parser.add_argument("--alpha-high", type=float, default=1.0)
    parser.add_argument("--reliability-tau", type=float, default=0.0)
    parser.add_argument("--ood-alpha-cap", type=float, default=0.25)
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--target-env-steps", type=int, default=100_000_000)
    parser.add_argument("--rollout-episodes-per-partner", type=int, default=1)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--learner-epochs", type=int, default=4)
    parser.add_argument("--min-rollout-rows-per-update", type=int, default=0)
    parser.add_argument("--source-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--lambda-delta", type=float, default=1.0)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--monitor-every", type=int, default=1)
    parser.add_argument("--eval-episodes-per-partner", type=int, default=20)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SRVF-MAPPO classic Overcooked training and evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command")
    formal = subparsers.add_parser(
        "formal-classic",
        help="Run SRVF-MAPPO training and held-out classic Overcooked evaluation.",
    )
    _add_common_training_args(formal)
    subparsers.add_parser("self-test", help="Run lightweight module self-tests.")
    return parser


def run_self_tests() -> Mapping[str, Any]:
    self_test_NeuralSRVFHeads()
    self_test_SRVFBelief()
    self_test_MAPPOActorCritic()
    self_test_UnifiedLoss()
    audit = gradient_audit()
    required_true = [
        "policy_reaches_actor",
        "policy_blocked_from_srvf",
        "value_reaches_critic",
        "delta_reaches_srvf",
        "delta_blocked_from_actor",
        "A_reaches_srvf",
        "A_blocked_from_actor",
    ]
    for key in required_true:
        assert bool(audit[key]), f"gradient audit failed: {key} -> {audit[key]}"
    return {"status": "passed", "gradient_audit": audit}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "formal-classic":
        summary = run_formal_classic(args)
        print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
        return
    if args.command == "self-test":
        result = run_self_tests()
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return
    parser.print_help()


# ── classic Overcooked execution notes ──

CLASSIC_OVERCOOKED_DEFAULTS: Mapping[str, int | float | str] = {
    "num_agents": 2,
    "num_actions": 6,
    "default_response_summary": "delta_z = g_next - g",
    "default_gamma": 0.99,
    "default_rollout_horizon": 400,
}


def classic_overcooked_rollout_step_contract() -> Mapping[str, str]:
    """Non-module interface contract for integrating with classic Overcooked-AI.

    This function intentionally does not implement an environment adapter. It states
    the tensors that a classic Overcooked rollout collector must provide to the five
    theorem-derived modules above.
    """
    return {
        "reset": "env.reset(layout_name, seed) -> observation, global_state, public chart g",
        "actor_action": "dist = MAPPOActorCritic(...).dist; ego_action = dist.sample()",
        "partner_action": "held-out partner policy acts from the same environment observation",
        "step": "env.step((ego_action, partner_action)) -> reward, done, next observation/state/g",
        "delta_z": "next_g - g, unless a richer public response summary is substituted consistently",
        "belief_update": "SRVFBelief.update(g, ego_action, delta_z, NeuralSRVFHeads)",
        "source_batch": "static V0-residual IRF table slice collected with resettable interventions",
    }


__all__ = [
    "ActorCriticOutput",
    "CLASSIC_OVERCOOKED_DEFAULTS",
    "IRFTable",
    "MAPPOActorCritic",
    "NeuralSRVFHeads",
    "RolloutBatch",
    "SRVFBelief",
    "SRVFHeadOutput",
    "SourceBatch",
    "SourceFactorInit",
    "UnifiedLoss",
    "V0Ensemble",
    "V0EnsembleProtocol",
    "V0MLP",
    "V0Prediction",
    "V0TrainingBatch",
    "build_arg_parser",
    "centered_action_residual",
    "classic_overcooked_rollout_step_contract",
    "collect_classic_rollout_batch",
    "collect_reservoir_snapshots",
    "collect_v0_residual_irf_table",
    "collect_v0_training_batch",
    "evaluate_closed_loop_classic",
    "evaluate_offline_target_regret",
    "gradient_audit",
    "initialize_source_beta",
    "iter_source_batches",
    "main",
    "run_formal_classic",
    "run_self_tests",
    "source_table_to_batch",
    "train_one_seed",
    "train_v0_ensemble",
]


if __name__ == "__main__":
    main()
