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
    q_extra_t: dict[str, Any] | None = None,
    q_extra_next: dict[str, Any] | None = None,
    td_loss: str = "huber",
    huber_delta: float = 1.0,
    double_q: bool = True,
    reward_scale: float = 1.0,
    vmax: float | None = None,
    bootstrap_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if hasattr(q_net, "forward_heads"):
        return _ensemble_aris_td_loss(
            q_net,
            target_q_net,
            obs_feat_t,
            belief_t,
            option_id,
            reward_sum,
            realized_cost,
            duration,
            obs_feat_next,
            belief_next,
            done,
            graph_batch,
            gamma,
            cost_coef,
            q_extra_t=q_extra_t,
            q_extra_next=q_extra_next,
            td_loss=td_loss,
            huber_delta=huber_delta,
            double_q=double_q,
            reward_scale=reward_scale,
            vmax=vmax,
            bootstrap_mask=bootstrap_mask,
        )
    q_all = q_net(
        obs_feat_t,
        belief_t,
        **_graph_kwargs(graph_batch, next_step=False),
        **(q_extra_t or {}),
    )
    q_pred = q_all.gather(1, option_id.long()[:, None]).squeeze(1)

    with torch.no_grad():
        next_kwargs = _graph_kwargs(graph_batch, next_step=True)
        q_next_target = target_q_net(
            obs_feat_next,
            belief_next,
            **next_kwargs,
            **(q_extra_next or {}),
        )
        option_mask_next = graph_batch.get(
            "option_mask_next",
            next_kwargs.get("option_mask"),
        )
        if option_mask_next is not None:
            q_next_target = q_next_target.masked_fill(
                ~option_mask_next.bool(),
                -1e9,
            )
        if double_q:
            q_next_online = q_net(
                obs_feat_next,
                belief_next,
                **next_kwargs,
                **(q_extra_next or {}),
            )
            if option_mask_next is not None:
                q_next_online = q_next_online.masked_fill(
                    ~option_mask_next.bool(),
                    -1e9,
                )
            next_option = q_next_online.argmax(dim=1)
            max_next = q_next_target.gather(1, next_option[:, None]).squeeze(1)
        else:
            max_next = q_next_target.max(dim=1).values
        target = (
            reward_sum / reward_scale
            - (cost_coef * realized_cost) / reward_scale
            + (gamma ** duration) * (1.0 - done.float()) * max_next
        )
        # RC-1 fix: hard-clamp the bootstrapped target to the return scale so it cannot run away
        # (the bounded q_net already caps max_next; this bounds the reward+bootstrap sum too). A
        # persistently high clamp fraction means reward_scale/vmax are misset, not that it "works".
        if vmax is not None:
            target = target.clamp(min=-float(vmax), max=float(vmax))

    return _td_criterion(q_pred, target, td_loss=td_loss, huber_delta=huber_delta)


def _ensemble_aris_td_loss(
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
    q_extra_t: dict[str, Any] | None,
    q_extra_next: dict[str, Any] | None,
    td_loss: str,
    huber_delta: float,
    double_q: bool,
    reward_scale: float,
    vmax: float | None,
    bootstrap_mask: torch.Tensor | None,
) -> torch.Tensor:
    q_all = q_net.forward_heads(
        obs_feat_t,
        belief_t,
        **_graph_kwargs(graph_batch, next_step=False),
        **(q_extra_t or {}),
    )
    if q_all.ndim != 3:
        raise ValueError(f"forward_heads must return [B,H,A], got {tuple(q_all.shape)}.")
    gather_idx = option_id.long()[:, None, None].expand(-1, q_all.shape[1], 1)
    q_pred = q_all.gather(2, gather_idx).squeeze(2)

    with torch.no_grad():
        next_kwargs = _graph_kwargs(graph_batch, next_step=True)
        q_next_target = target_q_net.forward_heads(
            obs_feat_next,
            belief_next,
            **next_kwargs,
            **(q_extra_next or {}),
        )
        option_mask_next = graph_batch.get(
            "option_mask_next",
            next_kwargs.get("option_mask"),
        )
        if option_mask_next is not None:
            q_next_target = q_next_target.masked_fill(
                ~option_mask_next[:, None, :].bool(),
                -1e9,
            )
        if double_q:
            q_next_online = q_net.forward_heads(
                obs_feat_next,
                belief_next,
                **next_kwargs,
                **(q_extra_next or {}),
            )
            if option_mask_next is not None:
                q_next_online = q_next_online.masked_fill(
                    ~option_mask_next[:, None, :].bool(),
                    -1e9,
                )
            next_option = q_next_online.argmax(dim=2)
            max_next = q_next_target.gather(2, next_option[:, :, None]).squeeze(2)
        else:
            max_next = q_next_target.max(dim=2).values
        target = (
            reward_sum[:, None] / reward_scale
            - (cost_coef * realized_cost[:, None]) / reward_scale
            + (gamma ** duration[:, None]) * (1.0 - done.float()[:, None]) * max_next
        )
        if vmax is not None:
            target = target.clamp(min=-float(vmax), max=float(vmax))

    losses = _td_criterion(
        q_pred,
        target,
        td_loss=td_loss,
        huber_delta=huber_delta,
        reduction="none",
    )
    if bootstrap_mask is None:
        return losses.mean()
    mask = bootstrap_mask.to(device=losses.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask[None, :].expand_as(losses)
    if mask.shape != losses.shape:
        raise ValueError(
            f"bootstrap_mask must have shape {tuple(losses.shape)}; got {tuple(mask.shape)}."
        )
    denom = mask.to(dtype=losses.dtype).sum().clamp(min=1.0)
    return (losses * mask.to(dtype=losses.dtype)).sum() / denom


def _td_criterion(
    q_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    td_loss: str,
    huber_delta: float,
    reduction: str = "mean",
) -> torch.Tensor:
    if td_loss == "mse":
        return F.mse_loss(q_pred, target, reduction=reduction)
    if td_loss == "huber":
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive when td_loss='huber'.")
        return F.smooth_l1_loss(
            q_pred,
            target,
            beta=float(huber_delta),
            reduction=reduction,
        )
    raise ValueError("td_loss must be one of {'huber', 'mse'}.")


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
