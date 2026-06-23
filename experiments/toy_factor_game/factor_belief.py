"""Factor-local belief model for ARIS-Bellman.

Each latent interaction factor receives only its routed local evidence stream.
There is no global history encoder feeding all factor heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        mask = active_mask.to(dtype=next_hidden.dtype, device=next_hidden.device).view(batch_size, 1, 1)
        return next_hidden * mask + hidden * (1.0 - mask)

    def encode_history(
        self,
        evidence_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
        initial_hidden: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if evidence_seq.dim() != 4:
            raise ValueError("evidence_seq must have shape [batch, time, n_factors, evidence_dim]")
        batch_size, max_time = evidence_seq.shape[:2]
        hidden = initial_hidden
        if hidden is None:
            hidden = self.initial_hidden(batch_size, evidence_seq.device)

        hidden_after_steps = []
        for t in range(max_time):
            active_mask = None
            if lengths is not None:
                active_mask = t < lengths.to(evidence_seq.device)
            hidden = self.step_history(evidence_seq[:, t], hidden, active_mask=active_mask)
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
            logits = head(torch.cat([hidden[:, factor_idx], factor_emb[:, factor_idx]], dim=-1))
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
