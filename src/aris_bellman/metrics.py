from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .specs import GraphSpec


def masked_entropy(
    belief: torch.Tensor,
    mode_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if belief.shape != mode_mask.shape:
        raise ValueError("belief and mode_mask must have the same shape.")
    probs = belief * mode_mask.bool().to(dtype=belief.dtype)
    return -(probs * (probs + eps).log()).sum(dim=-1)


def realized_delta_info(
    q_net,
    obs_feat_next: torch.Tensor,
    belief_before: torch.Tensor,
    belief_after: torch.Tensor,
    graph: GraphSpec | dict[str, Any],
    gamma: float,
) -> torch.Tensor:
    with torch.no_grad():
        kwargs = _graph_kwargs(graph, obs_feat_next)
        v_after = q_net(obs_feat_next, belief_after, **kwargs).max(dim=-1).values
        v_before = q_net(obs_feat_next, belief_before, **kwargs).max(dim=-1).values
        return gamma * (v_after - v_before)


def mi_proxy(
    belief_before: torch.Tensor,
    belief_after: torch.Tensor,
    mode_mask: torch.Tensor,
) -> torch.Tensor:
    h_before = masked_entropy(belief_before, mode_mask).sum(dim=-1)
    h_after = masked_entropy(belief_after, mode_mask).sum(dim=-1)
    return h_before - h_after


def mutual_information_proxy(
    belief_before: torch.Tensor,
    belief_after: torch.Tensor,
    mode_mask: torch.Tensor,
) -> torch.Tensor:
    return mi_proxy(belief_before, belief_after, mode_mask)


def diagnostic_cost(
    q_base_values: torch.Tensor,
    selected_option: torch.Tensor | int,
    delta_info: torch.Tensor | float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(selected_option):
        selected_option = torch.as_tensor(
            selected_option,
            dtype=torch.long,
            device=q_base_values.device,
        )
    selected_option = selected_option.to(device=q_base_values.device, dtype=torch.long)
    if selected_option.dim() == 0:
        selected_option = selected_option.expand(q_base_values.shape[0])

    if not torch.is_tensor(delta_info):
        delta_info = torch.as_tensor(
            delta_info,
            dtype=q_base_values.dtype,
            device=q_base_values.device,
        )
    delta_info = delta_info.to(device=q_base_values.device, dtype=q_base_values.dtype)
    if delta_info.dim() == 0:
        delta_info = delta_info.expand(q_base_values.shape[0])

    task_option = q_base_values.argmax(dim=-1)
    selected_value = q_base_values.gather(1, selected_option[:, None]).squeeze(1)
    task_value = q_base_values.gather(1, task_option[:, None]).squeeze(1)
    is_diag = (delta_info > float(tau)) & (selected_option != task_option)
    cost = torch.clamp(task_value - selected_value, min=0.0)
    return is_diag, cost


def _graph_kwargs(
    graph: GraphSpec | dict[str, Any],
    obs_feat: torch.Tensor,
) -> dict[str, Any]:
    if isinstance(graph, dict):
        return graph
    batch_size = obs_feat.shape[0]
    device = obs_feat.device
    return {
        "option_mask": _expand_np_mask(graph.option_mask, batch_size, device),
        "factor_mask": _expand_np_mask(graph.factor_mask, batch_size, device),
        "mode_mask": _expand_np_mask(graph.mode_mask, batch_size, device),
        "relevance_mask": _expand_np_mask(graph.relevance, batch_size, device),
        "option_features": _expand_np_array(graph.option_features, batch_size, device),
        "factor_features": _expand_np_array(graph.factor_features, batch_size, device),
    }


def _expand_np_mask(
    value: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
    return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)


def _expand_np_array(
    value: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)
