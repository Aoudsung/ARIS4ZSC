from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def aris_td_loss(
    q_net,
    target_q_net,
    obs_feat_t: torch.Tensor,
    belief_t: torch.Tensor,
    option_id: torch.Tensor,
    reward_sum: torch.Tensor,
    realized_cost: torch.Tensor,
    duration: torch.Tensor,
    obs_feat_next: torch.Tensor,
    belief_next: torch.Tensor,
    done: torch.Tensor,
    graph_batch: dict[str, Any],
    gamma: float,
    cost_coef: float,
) -> torch.Tensor:
    q_all = q_net(obs_feat_t, belief_t, **_graph_kwargs(graph_batch, next_step=False))
    q_pred = q_all.gather(1, option_id.long()[:, None]).squeeze(1)

    with torch.no_grad():
        next_kwargs = _graph_kwargs(graph_batch, next_step=True)
        q_next = target_q_net(obs_feat_next, belief_next, **next_kwargs)
        option_mask_next = graph_batch.get(
            "option_mask_next",
            next_kwargs.get("option_mask"),
        )
        if option_mask_next is not None:
            q_next = q_next.masked_fill(~option_mask_next.bool(), -1e9)
        max_next = q_next.max(dim=1).values
        target = (
            reward_sum
            - cost_coef * realized_cost
            + (gamma ** duration) * (1.0 - done.float()) * max_next
        )

    return F.mse_loss(q_pred, target)


def _graph_kwargs(graph_batch: dict[str, Any], *, next_step: bool) -> dict[str, Any]:
    option_key = "option_mask_next" if next_step else "option_mask"
    return {
        "option_mask": graph_batch.get(option_key, graph_batch.get("option_mask")),
        "factor_mask": graph_batch["factor_mask"],
        "mode_mask": graph_batch["mode_mask"],
        "relevance_mask": graph_batch["relevance_mask"],
        "option_features": graph_batch.get("option_features"),
        "factor_features": graph_batch.get("factor_features"),
    }
