from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorLocalBeliefModel(nn.Module):
    def __init__(
        self,
        evidence_dim: int,
        hidden_dim: int = 128,
        factor_embed_dim: int = 16,
        max_factors: int = 1,
        max_modes: int = 1,
        factor_feature_dim: int = 0,
    ):
        super().__init__()
        self.evidence_dim = evidence_dim
        self.hidden_dim = hidden_dim
        self.factor_embed_dim = factor_embed_dim
        self.max_factors = max(1, int(max_factors))
        self.max_modes = max(1, int(max_modes))
        self.factor_feature_dim = int(factor_feature_dim)

        self.factor_embedding = nn.Embedding(self.max_factors, factor_embed_dim)
        self.factor_feature_proj = (
            nn.Linear(self.factor_feature_dim, factor_embed_dim)
            if self.factor_feature_dim > 0
            else None
        )
        self.input_proj = nn.Sequential(
            nn.Linear(evidence_dim + factor_embed_dim, hidden_dim),
            nn.ReLU(),
        )
        self.filter_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.unary_head = nn.Sequential(
            nn.Linear(hidden_dim + factor_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_modes),
        )

    def initial_hidden(
        self,
        batch_size: int,
        num_factors: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_factors > self.max_factors:
            raise ValueError(
                f"Received {num_factors} factors, but model supports {self.max_factors}."
            )
        return torch.zeros(batch_size, num_factors, self.hidden_dim, device=device)

    def step_history(
        self,
        evidence_t: torch.Tensor,
        hidden: torch.Tensor | None = None,
        factor_features: torch.Tensor | None = None,
        factor_mask: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if evidence_t.dim() == 2:
            evidence_t = evidence_t.unsqueeze(0)
        if evidence_t.dim() != 3:
            raise ValueError("evidence_t must have shape [B, F, D_evid].")

        batch_size, num_factors, _ = evidence_t.shape
        if hidden is None:
            hidden = self.initial_hidden(batch_size, num_factors, evidence_t.device)
        if hidden.shape[:2] != (batch_size, num_factors):
            raise ValueError("hidden must have shape [B, F, hidden_dim].")
        if num_factors == 0:
            return hidden

        factor_context = self._factor_context(
            batch_size,
            num_factors,
            evidence_t.device,
            factor_features,
        )
        x = self.input_proj(torch.cat([evidence_t, factor_context], dim=-1))
        next_hidden = self.filter_cell(
            x.reshape(batch_size * num_factors, -1),
            hidden.reshape(batch_size * num_factors, -1),
        ).view(batch_size, num_factors, self.hidden_dim)

        update_mask = torch.ones(
            batch_size,
            num_factors,
            1,
            dtype=next_hidden.dtype,
            device=next_hidden.device,
        )
        if factor_mask is not None:
            update_mask = update_mask * factor_mask[:, :, None].to(next_hidden.dtype)
        if active_mask is not None:
            if active_mask.dim() == 1:
                active_mask = active_mask[:, None]
            update_mask = update_mask * active_mask[:, :, None].to(next_hidden.dtype)
        return next_hidden * update_mask + hidden * (1.0 - update_mask)

    def encode_history(
        self,
        evidence_seq: torch.Tensor,
        factor_features: torch.Tensor | None = None,
        factor_mask: torch.Tensor | None = None,
        initial_hidden: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if evidence_seq.dim() != 4:
            raise ValueError("evidence_seq must have shape [B, F, T, D_evid].")
        batch_size, num_factors, max_time = evidence_seq.shape[:3]
        hidden = initial_hidden
        if hidden is None:
            hidden = self.initial_hidden(batch_size, num_factors, evidence_seq.device)

        hidden_after_steps = []
        for timestep in range(max_time):
            hidden = self.step_history(
                evidence_seq[:, :, timestep],
                hidden,
                factor_features=factor_features,
                factor_mask=factor_mask,
            )
            if return_sequence:
                hidden_after_steps.append(hidden)

        if not return_sequence:
            return hidden
        if hidden_after_steps:
            sequence = torch.stack(hidden_after_steps, dim=2)
        else:
            sequence = hidden.new_zeros(batch_size, num_factors, 0, self.hidden_dim)
        return hidden, sequence

    def logits_from_hidden(
        self,
        hidden: torch.Tensor,
        factor_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError("hidden must have shape [B, F, hidden_dim].")
        batch_size, num_factors = hidden.shape[:2]
        if num_factors == 0:
            return hidden.new_zeros(batch_size, 0, self.max_modes)
        factor_context = self._factor_context(
            batch_size,
            num_factors,
            hidden.device,
            factor_features,
        )
        logits = self.unary_head(torch.cat([hidden, factor_context], dim=-1))
        return logits

    def forward(
        self,
        evidence_seq: torch.Tensor,
        factor_features: torch.Tensor | None,
        factor_mask: torch.Tensor,
        mode_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode_history(
            evidence_seq,
            factor_features=factor_features,
            factor_mask=factor_mask,
        )
        logits = self.logits_from_hidden(hidden, factor_features=factor_features)
        return masked_softmax(logits[:, :, : mode_mask.shape[-1]], mode_mask, factor_mask)

    def _factor_context(
        self,
        batch_size: int,
        num_factors: int,
        device: torch.device,
        factor_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if num_factors > self.max_factors:
            raise ValueError(
                f"Received {num_factors} factors, but model supports {self.max_factors}."
            )
        factor_ids = torch.arange(num_factors, device=device)
        context = self.factor_embedding(factor_ids)
        context = context.unsqueeze(0).expand(batch_size, -1, -1)
        if factor_features is not None and factor_features.shape[-1] > 0:
            if self.factor_feature_proj is None:
                raise ValueError("Construct with factor_feature_dim to use factor_features.")
            context = context + self.factor_feature_proj(factor_features.to(device))
        return context


class FactorBeliefModel(nn.Module):
    def __init__(
        self,
        evidence_dim: int,
        n_factors: int,
        factor_modes: list[int],
        hidden_dim: int = 128,
        factor_embed_dim: int = 16,
    ):
        super().__init__()
        self.evidence_dim = evidence_dim
        self.n_factors = n_factors
        self.factor_modes = factor_modes
        self.hidden_dim = hidden_dim
        self.factor_embed_dim = factor_embed_dim

        self.factor_embedding = nn.Embedding(max(n_factors, 1), factor_embed_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(evidence_dim + factor_embed_dim, hidden_dim),
            nn.ReLU(),
        )
        self.filter_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.unary_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + factor_embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_modes),
                )
                for n_modes in factor_modes
            ]
        )

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.n_factors, self.hidden_dim, device=device)

    def _factor_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.n_factors == 0:
            return torch.zeros(batch_size, 0, self.factor_embed_dim, device=device)
        ids = torch.arange(self.n_factors, device=device)
        emb = self.factor_embedding(ids)
        return emb.unsqueeze(0).expand(batch_size, -1, -1)

    def step_history(
        self,
        evidence_t: torch.Tensor,
        hidden: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if evidence_t.dim() == 2:
            evidence_t = evidence_t.unsqueeze(0)
        batch_size = evidence_t.shape[0]
        if hidden is None:
            hidden = self.initial_hidden(batch_size, evidence_t.device)

        if self.n_factors == 0:
            return hidden

        factor_emb = self._factor_embeddings(batch_size, evidence_t.device)
        x = self.input_proj(torch.cat([evidence_t, factor_emb], dim=-1))
        next_hidden = self.filter_cell(
            x.reshape(batch_size * self.n_factors, -1),
            hidden.reshape(batch_size * self.n_factors, -1),
        ).view(batch_size, self.n_factors, self.hidden_dim)

        if active_mask is None:
            return next_hidden
        mask = active_mask.to(dtype=next_hidden.dtype, device=next_hidden.device).view(
            batch_size,
            1,
            1,
        )
        return next_hidden * mask + hidden * (1.0 - mask)

    def encode_history(
        self,
        evidence_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
        initial_hidden: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if evidence_seq.dim() != 4:
            raise ValueError(
                "evidence_seq must have shape [batch, time, n_factors, evidence_dim]"
            )
        batch_size, max_time = evidence_seq.shape[:2]
        hidden = initial_hidden
        if hidden is None:
            hidden = self.initial_hidden(batch_size, evidence_seq.device)

        hidden_after_steps = []
        for timestep in range(max_time):
            active_mask = None
            if lengths is not None:
                active_mask = timestep < lengths.to(evidence_seq.device)
            hidden = self.step_history(
                evidence_seq[:, timestep],
                hidden,
                active_mask=active_mask,
            )
            if return_sequence:
                hidden_after_steps.append(hidden)

        if not return_sequence:
            return hidden
        if hidden_after_steps:
            sequence = torch.stack(hidden_after_steps, dim=1)
        else:
            sequence = hidden.new_zeros(batch_size, 0, self.n_factors, self.hidden_dim)
        return hidden, sequence

    def _marginals_from_h(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        if hidden.dim() == 4:
            raise ValueError("Pass a single hidden state [batch, n_factors, hidden_dim]")
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        batch_size = hidden.shape[0]
        if self.n_factors == 0:
            return []

        factor_emb = self._factor_embeddings(batch_size, hidden.device)
        marginals = []
        for factor_idx, head in enumerate(self.unary_heads):
            logits = head(
                torch.cat([hidden[:, factor_idx], factor_emb[:, factor_idx]], dim=-1)
            )
            marginals.append(F.softmax(logits, dim=-1))
        return marginals

    def forward(
        self,
        evidence_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        hidden = self.encode_history(evidence_seq, lengths=lengths)
        return self._marginals_from_h(hidden)

    def get_entropy(self, marginals: list[torch.Tensor]) -> torch.Tensor:
        if not marginals:
            return torch.empty(0)
        return torch.stack(
            [-(m * (m + 1e-8).log()).sum(dim=-1) for m in marginals],
            dim=-1,
        )

    def predict_factor_modes(self, marginals: list[torch.Tensor]) -> list[torch.Tensor]:
        return [m.argmax(dim=-1) for m in marginals]


def masked_softmax(
    logits: torch.Tensor,
    mode_mask: torch.Tensor,
    factor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != mode_mask.shape:
        raise ValueError("logits and mode_mask must have the same shape.")
    mask = mode_mask.bool()
    masked_logits = logits.masked_fill(~mask, -1e9)
    probs = F.softmax(masked_logits, dim=-1) * mask.to(dtype=logits.dtype)
    denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    probs = probs / denom
    if factor_mask is not None:
        probs = probs * factor_mask[:, :, None].to(dtype=probs.dtype)
    return probs
