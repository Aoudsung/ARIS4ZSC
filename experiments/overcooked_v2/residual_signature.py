from __future__ import annotations

import math
from typing import Any

import torch


_NEG_INF = -1.0e9


def normalized_advantage_signature(
    q_values: torch.Tensor,
    *,
    option_mask: torch.Tensor | None = None,
    tie_atol: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Return the primary Path C control code ``Q - max_valid(Q)``.

    ``q_values`` may have any leading dimensions and must end in the action
    dimension. A mask may omit ensemble/time dimensions as long as it broadcasts
    after singleton dimensions are inserted immediately before the action axis.
    Invalid actions are zeroed in the returned code and never participate in the
    maximum. This is the primary retained decision code; it does not use a
    separately learned public-state baseline.
    """

    if not isinstance(q_values, torch.Tensor) or q_values.ndim < 1:
        raise ValueError("q_values must be a tensor ending in an action dimension.")
    if not torch.is_floating_point(q_values):
        raise TypeError("q_values must use a floating-point dtype.")
    if q_values.shape[-1] <= 0:
        raise ValueError("q_values must contain at least one action.")
    if not math.isfinite(float(tie_atol)) or float(tie_atol) < 0.0:
        raise ValueError("tie_atol must be non-negative.")

    valid = _broadcast_action_mask(option_mask, q_values)
    if not bool(torch.isfinite(q_values[valid]).all()):
        raise ValueError("Valid-action Q values must be finite.")
    ranked = q_values.masked_fill(~valid, -torch.inf)
    has_valid = valid.any(dim=-1)
    max_q = ranked.max(dim=-1, keepdim=True).values
    safe_max = torch.where(
        has_valid.unsqueeze(-1),
        max_q,
        torch.zeros_like(max_q),
    )
    normalized = (q_values - safe_max).masked_fill(~valid, 0.0)
    best_option = ranked.argmax(dim=-1)
    best_option = torch.where(has_valid, best_option, torch.zeros_like(best_option))
    best_option_set = valid & torch.isclose(
        ranked,
        safe_max,
        atol=float(tie_atol),
        rtol=0.0,
    )
    return {
        "normalized_advantage": normalized,
        "best_option": best_option,
        "best_option_set": best_option_set,
        "has_valid_action": has_valid,
    }


def normalized_advantage_disagreement(
    head_q_values: torch.Tensor,
    *,
    option_mask: torch.Tensor | None = None,
    stat: str = "variance",
    tie_atol: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Measure per-action ensemble disagreement over normalized advantage.

    The final two dimensions of ``head_q_values`` are ``[K,A]``. The output
    ``per_option`` has the same leading dimensions followed by ``A``. This is the
    training-time Path C probe target; public-baseline residuals, response labels,
    identity, and style are intentionally absent.
    """

    if not isinstance(head_q_values, torch.Tensor) or head_q_values.ndim < 2:
        raise ValueError("head_q_values must have shape [...,K,A].")
    if head_q_values.shape[-2] <= 0:
        raise ValueError("head_q_values must contain at least one ensemble head.")
    stat = str(stat).lower()
    if stat not in {"variance", "range"}:
        raise ValueError("stat must be either 'variance' or 'range'.")

    signature = normalized_advantage_signature(
        head_q_values,
        option_mask=option_mask,
        tie_atol=tie_atol,
    )
    normalized = signature["normalized_advantage"]
    if stat == "range":
        per_option = normalized.max(dim=-2).values - normalized.min(dim=-2).values
    else:
        per_option = normalized.var(dim=-2, unbiased=False)
    valid = _broadcast_action_mask(option_mask, head_q_values).any(dim=-2)
    per_option = per_option.masked_fill(~valid, 0.0)
    scalar = per_option.max(dim=-1).values
    return {
        "per_option": per_option,
        "scalar": scalar,
        "normalized_advantage": normalized,
        "best_option_set": signature["best_option_set"],
    }


def _broadcast_action_mask(
    option_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if option_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    valid = torch.as_tensor(option_mask, dtype=torch.bool, device=reference.device)
    original_shape = tuple(valid.shape)
    if valid.ndim == 0 or valid.shape[-1] != reference.shape[-1]:
        raise ValueError(
            "option_mask must end in the same action dimension as q_values; got "
            f"{tuple(valid.shape)} for {tuple(reference.shape)}."
        )
    while valid.ndim < reference.ndim:
        valid = valid.unsqueeze(-2)
    try:
        return torch.broadcast_to(valid, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"option_mask shape {original_shape} cannot broadcast to "
            f"q_values shape {tuple(reference.shape)}."
        ) from exc


def residual_control_signature(
    q_values: torch.Tensor,
    base_q_values: torch.Tensor,
    *,
    option_mask: torch.Tensor | None = None,
    tie_atol: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Compute secondary public-baseline residual diagnostics.

    ``q_values`` is the history/belief-conditioned Q estimate. ``base_q_values``
    is the public-context estimate from ``BaseOnlyQNetwork``. Both tensors must
    have shape ``[B, A]``. The result is a secondary visualization or incremental
    diagnostic only. It must not define the primary retained code, drive the
    training-time probe selector, or participate in a training loss.

    Invariant: every statistic below is computed from ``q_values - base_q_values``
    and must be labeled as a residual diagnostic. The primary control code is
    :func:`normalized_advantage_signature`.
    """
    if q_values.shape != base_q_values.shape:
        raise ValueError(
            "q_values and base_q_values must have the same shape; got "
            f"{tuple(q_values.shape)} and {tuple(base_q_values.shape)}."
        )
    if q_values.ndim != 2:
        raise ValueError(f"Expected [B,A] Q tensors; got shape {tuple(q_values.shape)}.")

    residual_q = q_values - base_q_values
    if option_mask is None:
        valid = torch.ones_like(residual_q, dtype=torch.bool)
    else:
        valid = option_mask.bool()
        if valid.shape != residual_q.shape:
            raise ValueError(
                "option_mask must have the same shape as q_values; got "
                f"{tuple(valid.shape)} and {tuple(residual_q.shape)}."
            )

    rank_residual = residual_q.masked_fill(~valid, _NEG_INF)
    any_valid = valid.any(dim=1)
    max_residual = rank_residual.max(dim=1, keepdim=True).values
    safe_max = torch.where(
        any_valid.unsqueeze(1),
        max_residual,
        torch.zeros_like(max_residual),
    )
    advantage = residual_q - safe_max
    advantage = advantage.masked_fill(~valid, 0.0)
    residual_q_out = residual_q.masked_fill(~valid, 0.0)

    sorted_residual = torch.sort(rank_residual, dim=1, descending=True).values
    valid_count = valid.sum(dim=1)
    if sorted_residual.shape[1] >= 2:
        raw_gap = sorted_residual[:, 0] - sorted_residual[:, 1]
        gap = torch.where(valid_count >= 2, raw_gap, torch.zeros_like(raw_gap))
    else:
        gap = torch.zeros(q_values.shape[0], dtype=q_values.dtype, device=q_values.device)

    best_option = rank_residual.argmax(dim=1)
    best_option = torch.where(
        any_valid,
        best_option,
        torch.zeros_like(best_option),
    )
    best_option_set = valid & torch.isclose(
        rank_residual,
        safe_max,
        atol=float(tie_atol),
        rtol=0.0,
    )

    return {
        "residual_q": residual_q_out,
        "advantage": advantage,
        "gap": gap,
        "best_option": best_option,
        "best_option_set": best_option_set,
    }


def residual_signature_disagreement(
    head_q_values: torch.Tensor,
    base_q_values: torch.Tensor,
    *,
    option_mask: torch.Tensor | None = None,
    stat: str = "variance",
    tie_atol: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Secondary disagreement over a public-baseline residual signature.

    ``head_q_values`` has shape ``[B,H,A]`` and ``base_q_values`` has shape
    ``[B,A]``. The returned ``per_option`` tensor combines residual-advantage
    disagreement, best-option-set instability, and a broadcast residual-gap term.
    It is retained for secondary diagnostics and backward artifact reading only.
    The training-time selector must use
    :func:`normalized_advantage_disagreement`.
    """
    if head_q_values.ndim != 3:
        raise ValueError(
            f"Expected head_q_values with shape [B,H,A]; got {tuple(head_q_values.shape)}."
        )
    if base_q_values.ndim != 2:
        raise ValueError(
            f"Expected base_q_values with shape [B,A]; got {tuple(base_q_values.shape)}."
        )
    if head_q_values.shape[0] != base_q_values.shape[0] or head_q_values.shape[2] != base_q_values.shape[1]:
        raise ValueError(
            "head_q_values and base_q_values have incompatible batch/action shapes: "
            f"{tuple(head_q_values.shape)} vs {tuple(base_q_values.shape)}."
        )
    stat = str(stat).lower()
    if stat not in {"variance", "range"}:
        raise ValueError("stat must be either 'variance' or 'range'.")

    residual = head_q_values - base_q_values.unsqueeze(1)
    if option_mask is None:
        valid = torch.ones(
            residual.shape[0],
            residual.shape[2],
            dtype=torch.bool,
            device=residual.device,
        )
    else:
        valid = option_mask.bool()
        if valid.ndim == 1:
            valid = valid.unsqueeze(0).expand(residual.shape[0], -1)
        if valid.shape != (residual.shape[0], residual.shape[2]):
            raise ValueError(
                "option_mask must have shape [B,A] or [A]; got "
                f"{tuple(valid.shape)} for residual shape {tuple(residual.shape)}."
            )
    valid_h = valid.unsqueeze(1)
    rank_residual = residual.masked_fill(~valid_h, _NEG_INF)
    any_valid = valid.any(dim=1)
    max_residual = rank_residual.max(dim=2, keepdim=True).values
    safe_max = torch.where(
        any_valid.view(-1, 1, 1),
        max_residual,
        torch.zeros_like(max_residual),
    )
    advantage = (residual - safe_max).masked_fill(~valid_h, 0.0)
    sorted_residual = torch.sort(rank_residual, dim=2, descending=True).values
    valid_count = valid.sum(dim=1)
    if sorted_residual.shape[2] >= 2:
        head_gap = sorted_residual[:, :, 0] - sorted_residual[:, :, 1]
        head_gap = torch.where(
            valid_count.view(-1, 1) >= 2,
            head_gap,
            torch.zeros_like(head_gap),
        )
    else:
        head_gap = torch.zeros(
            residual.shape[0],
            residual.shape[1],
            dtype=residual.dtype,
            device=residual.device,
        )
    best_sets = (
        valid_h
        & torch.isclose(
            rank_residual,
            safe_max,
            atol=float(tie_atol),
            rtol=0.0,
        )
    ).to(dtype=residual.dtype)

    if stat == "range":
        adv_disp = advantage.max(dim=1).values - advantage.min(dim=1).values
        best_disp = best_sets.max(dim=1).values - best_sets.min(dim=1).values
        gap_disp = head_gap.max(dim=1).values - head_gap.min(dim=1).values
    else:
        adv_disp = advantage.var(dim=1, unbiased=False)
        best_disp = best_sets.var(dim=1, unbiased=False)
        gap_disp = head_gap.var(dim=1, unbiased=False)
    per_option = (adv_disp + best_disp + gap_disp.unsqueeze(1)).masked_fill(~valid, 0.0)
    return {
        "per_option": per_option,
        "scalar": per_option.max(dim=1).values,
        "advantage_disagreement": adv_disp.masked_fill(~valid, 0.0),
        "best_option_set_disagreement": best_disp.masked_fill(~valid, 0.0),
        "gap_disagreement": gap_disp,
    }


def select_probe_candidate(
    per_option_disagreement: torch.Tensor,
    ensemble_mean_q: torch.Tensor,
    option_mask: torch.Tensor,
    *,
    disagreement_threshold: float | None,
    return_floor: float | None,
) -> dict[str, Any]:
    """Select disagreement first, then apply safety to that same candidate."""
    disagreement = per_option_disagreement.reshape(-1)
    mean_q = ensemble_mean_q.reshape(-1)
    valid = option_mask.reshape(-1).bool()
    if disagreement.shape != mean_q.shape or disagreement.shape != valid.shape:
        raise ValueError("Probe disagreement, mean Q, and option mask must align.")
    if not bool(valid.any()):
        return {"selected": False, "reason": "disabled", "option_id": -1}
    masked = disagreement.masked_fill(~valid, _NEG_INF)
    option_id = int(torch.argmax(masked).item())
    score = float(masked[option_id].detach().cpu().item())
    candidate_mean_q = float(mean_q[option_id].detach().cpu().item())
    if disagreement_threshold is None or score < float(disagreement_threshold):
        return {
            "selected": False,
            "reason": "threshold",
            "option_id": option_id,
            "score": score,
            "candidate_mean_q": candidate_mean_q,
        }
    if return_floor is None or candidate_mean_q < float(return_floor):
        return {
            "selected": False,
            "reason": "return_floor",
            "option_id": option_id,
            "score": score,
            "candidate_mean_q": candidate_mean_q,
        }
    return {
        "selected": True,
        "reason": "selected",
        "option_id": option_id,
        "score": score,
        "candidate_mean_q": candidate_mean_q,
    }


def compute_residual_control_signature(
    q_net: Any,
    base_q_net: Any,
    obs_feat: torch.Tensor,
    belief: torch.Tensor,
    *,
    graph_kwargs: dict[str, Any],
    q_extra: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate secondary public-baseline residual diagnostics."""
    kwargs = dict(graph_kwargs)
    extra = dict(q_extra or {})
    q_values = _mean_q_values(q_net, obs_feat, belief, kwargs, extra)
    base_extra = dict(extra)
    base_extra.pop("partner_id", None)
    base_values = base_q_net(obs_feat, belief, **kwargs, **base_extra)
    return residual_control_signature(
        q_values,
        base_values,
        option_mask=kwargs.get("option_mask"),
    )


def _mean_q_values(
    q_net: Any,
    obs_feat: torch.Tensor,
    belief: torch.Tensor,
    graph_kwargs: dict[str, Any],
    q_extra: dict[str, Any],
) -> torch.Tensor:
    if hasattr(q_net, "forward_mean"):
        return q_net.forward_mean(obs_feat, belief, **graph_kwargs, **q_extra)
    values = q_net(obs_feat, belief, **graph_kwargs, **q_extra)
    if values.ndim == 3:
        return values.mean(dim=1)
    if values.ndim != 2:
        raise ValueError(f"Expected Q values with rank 2 or 3; got shape {tuple(values.shape)}.")
    return values
