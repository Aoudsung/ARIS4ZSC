"""Post-hoc belief-information diagnostics.

G-TVOI and MI are not selectors in ARIS-Bellman. They are computed after a
real transition using the observed belief update.
"""

import torch

def compute_mi(
    marginals_before: list[torch.Tensor],
    marginals_after: list[torch.Tensor],
) -> torch.Tensor:
    if not marginals_before:
        return torch.zeros(0)
    mi = torch.zeros(marginals_before[0].shape[0], device=marginals_before[0].device)
    for before, after in zip(marginals_before, marginals_after):
        ent_before = -(before * (before + 1e-8).log()).sum(dim=-1)
        ent_after = -(after * (after + 1e-8).log()).sum(dim=-1)
        mi = mi + (ent_before - ent_after).clamp(min=0.0)
    return mi


def bellman_delta_info(
    agent,
    next_obs: torch.Tensor,
    marginals_before: list[torch.Tensor],
    marginals_after: list[torch.Tensor],
    valid_mask: torch.Tensor | None = None,
    gamma: float = 0.99,
) -> torch.Tensor:
    q_hold = agent.q_values(next_obs, marginals_before)
    q_after = agent.q_values(next_obs, marginals_after)
    if valid_mask is not None:
        q_hold = q_hold.masked_fill(~valid_mask.bool(), -1e9)
        q_after = q_after.masked_fill(~valid_mask.bool(), -1e9)
    return gamma * (q_after.max(dim=-1).values - q_hold.max(dim=-1).values)
