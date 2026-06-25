from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from src.aris_bellman.metrics import (
    diagnostic_cost as shared_diagnostic_cost,
    mutual_information_proxy as shared_mi_proxy,
    realized_delta_info as shared_delta_info,
)
from src.aris_bellman.specs import GraphSpec


def realized_delta_info(
    q_net,
    obs_feat_next: torch.Tensor,
    belief_before: torch.Tensor,
    belief_after: torch.Tensor,
    graph_batch: dict[str, Any],
    gamma: float,
) -> torch.Tensor:
    return shared_delta_info(
        q_net,
        obs_feat_next,
        belief_before,
        belief_after,
        _q_forward_kwargs(graph_batch),
        gamma,
    )


def mutual_information_proxy(
    belief_before: torch.Tensor,
    belief_after: torch.Tensor,
    mode_mask: torch.Tensor,
) -> torch.Tensor:
    if belief_before.numel() == 0:
        return belief_before.new_zeros(belief_before.shape[0])
    return shared_mi_proxy(belief_before, belief_after, mode_mask)


def diagnostic_cost(
    q_base_values: torch.Tensor,
    selected_option: torch.Tensor | int,
    delta_info: torch.Tensor | float,
    tau: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return shared_diagnostic_cost(q_base_values, selected_option, delta_info, tau)


def reference_gap_closure(
    r_aris: float,
    r_base: float,
    r_ref: float,
    eps: float = 1e-8,
) -> dict[str, Any]:
    denom = float(r_ref) - float(r_base)
    if abs(denom) <= float(eps):
        return {
            "value": None,
            "status": "denominator_too_small",
            "r_aris": float(r_aris),
            "r_base": float(r_base),
            "r_ref": float(r_ref),
        }
    raw = (float(r_aris) - float(r_base)) / denom
    return {
        "value": float(np.clip(raw, 0.0, 2.0)),
        "raw_value": float(raw),
        "status": "ok",
        "r_aris": float(r_aris),
        "r_base": float(r_base),
        "r_ref": float(r_ref),
    }


def belief_swap_delta(
    q_net,
    obs_feat: torch.Tensor,
    belief: torch.Tensor,
    graph_batch: dict[str, Any],
    factor_i: int = 0,
    factor_j: int = 1,
) -> dict[str, Any]:
    if belief.shape[1] < 2:
        return {"status": "not_enough_factors", "value": None}
    if factor_i >= belief.shape[1] or factor_j >= belief.shape[1]:
        raise IndexError("factor_i and factor_j must be valid factor indexes.")

    mode_mask = graph_batch["mode_mask"].bool()
    if not _mode_masks_compatible(mode_mask, factor_i, factor_j):
        return {
            "status": "incompatible_mode_mask",
            "factor_i": int(factor_i),
            "factor_j": int(factor_j),
            "valid_modes_i": _valid_mode_counts(mode_mask, factor_i),
            "valid_modes_j": _valid_mode_counts(mode_mask, factor_j),
        }

    swapped = belief.clone()
    swapped[:, factor_i] = belief[:, factor_j]
    swapped[:, factor_j] = belief[:, factor_i]
    swapped = _mask_and_renormalize(swapped, mode_mask)
    with torch.no_grad():
        q_orig = q_net(obs_feat, belief, **_q_forward_kwargs(graph_batch))
        q_swap = q_net(obs_feat, swapped, **_q_forward_kwargs(graph_batch))
    delta = q_swap - q_orig
    return {
        "status": "ok",
        "factor_i": int(factor_i),
        "factor_j": int(factor_j),
        "mean_abs_maxq_delta": float(
            (q_swap.max(dim=-1).values - q_orig.max(dim=-1).values).abs().mean().item()
        ),
        "mean_abs_q_delta": float(delta.abs().mean().item()),
        "action_flip_rate": float(
            (q_swap.argmax(dim=-1) != q_orig.argmax(dim=-1)).float().mean().item()
        ),
    }


def belief_swap_top_pairs(
    q_net,
    obs_feat: torch.Tensor,
    belief: torch.Tensor,
    graph_batch: dict[str, Any],
    graph: GraphSpec,
    max_pairs: int = 8,
) -> dict[str, Any]:
    if graph.num_factors < 2:
        return {"status": "not_enough_factors", "value": None, "pairs": []}

    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, factor_i, factor_j in _ranked_factor_pairs(graph, graph_batch):
        row = belief_swap_delta(
            q_net,
            obs_feat,
            belief,
            graph_batch,
            factor_i=factor_i,
            factor_j=factor_j,
        )
        if row["status"] != "ok":
            skipped += 1
            continue
        rows.append(row)
        if len(rows) >= max_pairs:
            break

    if not rows:
        return {
            "status": "no_compatible_pairs",
            "value": None,
            "pairs": [],
            "num_skipped_incompatible": int(skipped),
        }

    return {
        "status": "ok",
        "num_pairs": len(rows),
        "num_skipped_incompatible": int(skipped),
        "pairs": rows,
        "mean_abs_maxq_delta": float(
            np.mean([row["mean_abs_maxq_delta"] for row in rows])
        ),
        "mean_abs_q_delta": float(np.mean([row["mean_abs_q_delta"] for row in rows])),
        "action_flip_rate": float(np.mean([row["action_flip_rate"] for row in rows])),
    }


def factor_deletion_return_drop(
    graph: GraphSpec,
    base_return: float,
    evaluate_deleted_factor: Callable[[int], float],
) -> list[dict[str, Any]]:
    drops = []
    for factor in graph.factors:
        deleted_return = float(evaluate_deleted_factor(int(factor.id)))
        drops.append(
            {
                "factor_id": int(factor.id),
                "factor_kind": factor.factor_kind,
                "option_i": int(factor.option_i),
                "option_j": int(factor.option_j),
                "base_return": float(base_return),
                "deleted_return": deleted_return,
                "return_drop": float(base_return - deleted_return),
            }
        )
    drops.sort(key=lambda row: (-row["return_drop"], row["factor_id"]))
    return drops


def graph_with_deleted_factor(graph: GraphSpec, factor_id: int) -> GraphSpec:
    if factor_id < 0 or factor_id >= graph.num_factors:
        raise IndexError(f"factor_id {factor_id} out of range for {graph.num_factors} factors.")
    relevance = np.asarray(graph.relevance, dtype=bool).copy()
    factor_mask = np.asarray(graph.factor_mask, dtype=bool).copy()
    mode_mask = np.asarray(graph.mode_mask, dtype=bool).copy()
    relevance[factor_id, :] = False
    factor_mask[factor_id] = False
    mode_mask[factor_id, :] = False
    metadata = {
        **(graph.metadata or {}),
        "ablation": "factor_deletion",
        "deleted_factor_id": int(factor_id),
    }
    return GraphSpec(
        layout_name=graph.layout_name,
        options=graph.options,
        factors=graph.factors,
        relevance=relevance,
        option_mask=np.asarray(graph.option_mask, dtype=bool).copy(),
        factor_mask=factor_mask,
        mode_mask=mode_mask,
        route_map={
            int(key): tuple(value)
            for key, value in graph.route_map.items()
            if int(key) != int(factor_id)
        },
        option_features=graph.option_features,
        factor_features=graph.factor_features,
        metadata=metadata,
    )


def _q_forward_kwargs(graph_batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "option_mask": graph_batch["option_mask"],
        "factor_mask": graph_batch["factor_mask"],
        "mode_mask": graph_batch["mode_mask"],
        "relevance_mask": graph_batch["relevance_mask"],
        "option_features": graph_batch.get("option_features"),
        "factor_features": graph_batch.get("factor_features"),
    }


def _mask_and_renormalize(belief: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
    masked = belief * mode_mask.to(dtype=belief.dtype)
    denom = masked.sum(dim=-1, keepdim=True)
    normalized = masked / denom.clamp(min=1e-8)
    return torch.where(denom > 0.0, normalized, torch.zeros_like(masked))


def _mode_masks_compatible(mode_mask: torch.Tensor, factor_i: int, factor_j: int) -> bool:
    return bool(torch.equal(mode_mask[:, factor_i, :], mode_mask[:, factor_j, :]))


def _valid_mode_counts(mode_mask: torch.Tensor, factor_idx: int) -> list[int]:
    return [
        int(value)
        for value in mode_mask[:, factor_idx, :].sum(dim=-1).detach().cpu().tolist()
    ]


def _ranked_factor_pairs(
    graph: GraphSpec,
    graph_batch: dict[str, Any],
) -> list[tuple[tuple[float, int, float, int, int], int, int]]:
    relevance = np.asarray(graph.relevance, dtype=bool)
    mode_mask = graph_batch["mode_mask"].bool()
    ranked = []
    for idx_i, factor_i in enumerate(graph.factors):
        for idx_j in range(idx_i + 1, len(graph.factors)):
            factor_j = graph.factors[idx_j]
            if not _mode_masks_compatible(mode_mask, idx_i, idx_j):
                compatible = 0
            else:
                compatible = 1
            overlap = int(np.logical_and(relevance[idx_i], relevance[idx_j]).sum())
            score = float(abs(factor_i.ce_score) + abs(factor_j.ce_score))
            ranked.append(((-compatible, -overlap, -score, idx_i, idx_j), idx_i, idx_j))
    ranked.sort(key=lambda item: item[0])
    return ranked
