"""Training and checkpoint diagnostics for the standard Path C policy.

The helpers in this module are observational: they never draw random numbers,
change model parameters, or alter optimizer state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.overcooked_v2.path_c_standard import (
    PrimitiveRecurrentEnsembleQ,
    load_standard_checkpoint,
)


TRAINING_METRICS_SCHEMA_VERSION = "path_c_standard_training_metrics_v3"
CHECKPOINT_HEALTH_SCHEMA_VERSION = "path_c_standard_checkpoint_health_v1"


@dataclass(frozen=True)
class RawRewardEvents:
    """Observable decomposition of shared OvercookedV2 raw step rewards."""

    correct_delivery_count: int
    wrong_delivery_count: int
    indicator_activation_count: int
    ambiguous_step_count: int
    ambiguous_raw_rewards: tuple[float, ...]


def decompose_raw_reward_events(raw_step_rewards: np.ndarray) -> RawRewardEvents:
    """Decompose official +20/-20 deliveries and -5 indicator costs.

    Pure repeated events are recognized first. Mixed totals are accepted only
    when the bounded integer equation has one solution; otherwise the original
    reward is retained in ``ambiguous_raw_rewards`` instead of inventing an
    event label.
    """

    rewards = np.asarray(raw_step_rewards, dtype=np.float64)
    if rewards.ndim != 1 or not bool(np.isfinite(rewards).all()):
        raise ValueError("Raw reward events require one finite reward vector.")
    correct = wrong = indicator = ambiguous = 0
    ambiguous_values: list[float] = []
    for raw_reward in rewards:
        value = float(raw_reward)
        if math.isclose(value, 0.0, abs_tol=1e-6):
            continue
        if value > 0.0 and math.isclose(value % 20.0, 0.0, abs_tol=1e-6):
            count = int(round(value / 20.0))
            if 1 <= count <= 2:
                correct += count
                continue
        if value < 0.0 and math.isclose((-value) % 20.0, 0.0, abs_tol=1e-6):
            count = int(round((-value) / 20.0))
            if 1 <= count <= 2:
                wrong += count
                continue
        if value < 0.0 and math.isclose((-value) % 5.0, 0.0, abs_tol=1e-6):
            count = int(round((-value) / 5.0))
            if 1 <= count <= 2:
                indicator += count
                continue
        solutions = [
            (a, b, c)
            for a in range(3)
            for b in range(3)
            for c in range(3)
            if math.isclose(20.0 * a - 20.0 * b - 5.0 * c, value, abs_tol=1e-6)
        ]
        if len(solutions) == 1:
            a, b, c = solutions[0]
            correct += a
            wrong += b
            indicator += c
        else:
            ambiguous += 1
            ambiguous_values.append(value)
    return RawRewardEvents(
        correct_delivery_count=correct,
        wrong_delivery_count=wrong,
        indicator_activation_count=indicator,
        ambiguous_step_count=ambiguous,
        ambiguous_raw_rewards=tuple(ambiguous_values),
    )


def _parameter_group(name: str) -> str:
    if name.startswith("learned.observation_encoder."):
        return "observation_encoder"
    if name.startswith("learned.recurrent.shared_encoder."):
        return "shared_encoder"
    if name.startswith("learned.recurrent.head_grus."):
        return "recurrent_heads"
    if name.startswith("learned.recurrent.value_heads."):
        return "value_heads"
    if name.startswith("learned.recurrent.advantage_heads."):
        return "advantage_heads"
    if name.startswith("fixed_prior."):
        return "fixed_prior"
    raise ValueError(f"Unrecognized standard-model parameter group: {name!r}.")


def snapshot_trainable_parameters(
    model: PrimitiveRecurrentEnsembleQ,
) -> dict[str, torch.Tensor]:
    """Copy trainable parameters without touching any random-number generator."""

    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _empty_parameter_totals() -> dict[str, float | int]:
    return {
        "parameter_count": 0,
        "finite_parameter_count": 0,
        "nonfinite_count": 0,
        "sum_squares": 0.0,
        "max_abs": 0.0,
        "delta_sum_squares": 0.0,
    }


def _finish_parameter_totals(
    totals: Mapping[str, float | int],
    *,
    has_reference: bool,
) -> dict[str, Any]:
    count = int(totals["parameter_count"])
    finite_count = int(totals["finite_parameter_count"])
    sum_squares = float(totals["sum_squares"])
    result = {
        "parameter_count": count,
        "finite_parameter_count": finite_count,
        "nonfinite_count": int(totals["nonfinite_count"]),
        "l2_norm": math.sqrt(max(0.0, sum_squares)),
        "rms": (
            None
            if finite_count == 0
            else math.sqrt(max(0.0, sum_squares) / finite_count)
        ),
        "max_abs": None if finite_count == 0 else float(totals["max_abs"]),
        "delta_l2_from_reference": (
            math.sqrt(max(0.0, float(totals["delta_sum_squares"])))
            if has_reference
            else None
        ),
    }
    return result


@torch.no_grad()
def model_parameter_health(
    model: PrimitiveRecurrentEnsembleQ,
    *,
    reference: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Summarize learned parameter groups and the immutable fixed prior."""

    learned_totals = _empty_parameter_totals()
    fixed_totals = _empty_parameter_totals()
    grouped = {
        name: _empty_parameter_totals()
        for name in (
            "observation_encoder",
            "shared_encoder",
            "recurrent_heads",
            "value_heads",
            "advantage_heads",
        )
    }
    reference_names = set() if reference is None else set(reference)
    observed_trainable_names: set[str] = set()
    for name, parameter in model.named_parameters():
        group = _parameter_group(name)
        value = parameter.detach()
        finite = torch.isfinite(value)
        finite_count = int(finite.sum().item())
        nonfinite_count = int(value.numel() - finite_count)
        finite_values = value[finite].float()
        sum_squares = (
            0.0
            if finite_count == 0
            else float(torch.sum(finite_values * finite_values).item())
        )
        max_abs = (
            0.0
            if finite_count == 0
            else float(torch.max(torch.abs(finite_values)).item())
        )
        target = fixed_totals if group == "fixed_prior" else learned_totals
        targets = [target]
        if group != "fixed_prior":
            if not parameter.requires_grad:
                raise ValueError(f"Learned parameter is unexpectedly frozen: {name}.")
            observed_trainable_names.add(name)
            targets.append(grouped[group])
        for totals in targets:
            totals["parameter_count"] = int(totals["parameter_count"]) + value.numel()
            totals["finite_parameter_count"] = (
                int(totals["finite_parameter_count"]) + finite_count
            )
            totals["nonfinite_count"] = (
                int(totals["nonfinite_count"]) + nonfinite_count
            )
            totals["sum_squares"] = float(totals["sum_squares"]) + sum_squares
            totals["max_abs"] = max(float(totals["max_abs"]), max_abs)
        if reference is not None and group != "fixed_prior":
            if name not in reference:
                raise ValueError(f"Parameter reference is missing {name!r}.")
            baseline = reference[name]
            if tuple(baseline.shape) != tuple(value.shape):
                raise ValueError(f"Parameter reference shape differs for {name!r}.")
            delta = value.float() - baseline.to(device=value.device, dtype=torch.float32)
            delta_sum = float(torch.sum(delta * delta).item())
            learned_totals["delta_sum_squares"] = (
                float(learned_totals["delta_sum_squares"]) + delta_sum
            )
            grouped[group]["delta_sum_squares"] = (
                float(grouped[group]["delta_sum_squares"]) + delta_sum
            )
    if reference is not None and reference_names != observed_trainable_names:
        missing = sorted(observed_trainable_names - reference_names)
        unexpected = sorted(reference_names - observed_trainable_names)
        raise ValueError(
            f"Parameter reference mismatch: missing={missing}, unexpected={unexpected}."
        )
    has_reference = reference is not None
    return {
        "learned": _finish_parameter_totals(
            learned_totals,
            has_reference=has_reference,
        ),
        "groups": {
            name: _finish_parameter_totals(totals, has_reference=has_reference)
            for name, totals in grouped.items()
        },
        "fixed_prior": _finish_parameter_totals(
            fixed_totals,
            has_reference=False,
        ),
        "fixed_prior_state_sha256": model.prior_state_sha256(),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise ValueError("Training diagnostics received a non-finite numeric value.")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _quantile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    result = _numeric_summary(values)
    if not values:
        result.update({"p10": None, "p50": None, "p90": None})
        return result
    array = np.asarray(values, dtype=np.float64)
    result.update(
        {
            "p10": float(np.quantile(array, 0.10)),
            "p50": float(np.quantile(array, 0.50)),
            "p90": float(np.quantile(array, 0.90)),
        }
    )
    return result


@dataclass
class TrainingWindowAccumulator:
    """Accumulate one fixed environment-step window without consuming RNG state."""

    batch_size: int
    n_heads: int
    gradient_clip_norm: float
    reference: Mapping[str, torch.Tensor]
    running_episode_returns: np.ndarray = field(init=False)
    environment_steps: int = field(default=0, init=False)
    raw_reward_sum: float = field(default=0.0, init=False)
    positive_reward_count: int = field(default=0, init=False)
    negative_reward_count: int = field(default=0, init=False)
    zero_reward_count: int = field(default=0, init=False)
    correct_delivery_count: int = field(default=0, init=False)
    wrong_delivery_count: int = field(default=0, init=False)
    indicator_activation_count: int = field(default=0, init=False)
    ambiguous_reward_step_count: int = field(default=0, init=False)
    training_reward_sum: float = field(default=0.0, init=False)
    training_reward_count: int = field(default=0, init=False)
    shaped_reward_sum: float = field(default=0.0, init=False)
    shaped_reward_count: int = field(default=0, init=False)
    shaped_reward_sum_by_slot: dict[str, float] = field(default_factory=dict, init=False)
    anneal_factors: list[float] = field(default_factory=list, init=False)
    completed_episode_returns: list[float] = field(default_factory=list, init=False)
    losses: list[float] = field(default_factory=list, init=False)
    per_head_losses: list[list[float]] = field(init=False)
    per_head_support: list[int] = field(init=False)
    gradient_norms: list[float] = field(default_factory=list, init=False)
    clipped_gradient_count: int = field(default=0, init=False)
    probe_count: int = field(default=0, init=False)
    probe_candidate_disagreement: list[float] = field(default_factory=list, init=False)
    probe_candidate_q: list[float] = field(default_factory=list, init=False)
    probe_candidate_regret: list[float] = field(default_factory=list, init=False)
    probe_candidate_equals_greedy_count: int = field(default=0, init=False)
    probe_disagreement_pass_count: int = field(default=0, init=False)
    probe_regret_pass_count: int = field(default=0, init=False)
    probe_budget_pass_count: int = field(default=0, init=False)
    probe_candidate_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0 or int(self.n_heads) <= 0:
            raise ValueError("Training diagnostics require positive batch and head counts.")
        if not math.isfinite(float(self.gradient_clip_norm)) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive and finite.")
        self.running_episode_returns = np.zeros(int(self.batch_size), dtype=np.float64)
        self.per_head_losses = [[] for _ in range(int(self.n_heads))]
        self.per_head_support = [0 for _ in range(int(self.n_heads))]

    def record_environment_batch(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        *,
        probe_count: int,
        training_rewards: np.ndarray | None = None,
        training_shaped_rewards: np.ndarray | None = None,
        shaped_rewards_by_slot: Mapping[str, np.ndarray] | None = None,
        anneal_factor: float | None = None,
        probe_candidate_disagreement: np.ndarray | None = None,
        probe_candidate_q: np.ndarray | None = None,
        probe_candidate_regret: np.ndarray | None = None,
        probe_candidate_equals_greedy: np.ndarray | None = None,
        probe_disagreement_pass: np.ndarray | None = None,
        probe_regret_pass: np.ndarray | None = None,
        probe_budget_pass: np.ndarray | None = None,
    ) -> None:
        reward_array = np.asarray(rewards, dtype=np.float64)
        done_array = np.asarray(dones, dtype=bool)
        if reward_array.shape != (self.batch_size,) or done_array.shape != (
            self.batch_size,
        ):
            raise ValueError("Training diagnostic batch shape differs from batch_size.")
        if not bool(np.isfinite(reward_array).all()):
            raise ValueError("Training rewards must be finite.")
        self.environment_steps += int(reward_array.size)
        self.raw_reward_sum += float(reward_array.sum())
        self.positive_reward_count += int(np.count_nonzero(reward_array > 0.0))
        self.negative_reward_count += int(np.count_nonzero(reward_array < 0.0))
        self.zero_reward_count += int(np.count_nonzero(reward_array == 0.0))
        events = decompose_raw_reward_events(reward_array)
        self.correct_delivery_count += events.correct_delivery_count
        self.wrong_delivery_count += events.wrong_delivery_count
        self.indicator_activation_count += events.indicator_activation_count
        self.ambiguous_reward_step_count += events.ambiguous_step_count
        training_array = (
            reward_array
            if training_rewards is None
            else np.asarray(training_rewards, dtype=np.float64)
        )
        if training_array.ndim != 1 or not bool(np.isfinite(training_array).all()):
            raise ValueError("Training-channel rewards must be one-dimensional and finite.")
        self.training_reward_sum += float(training_array.sum())
        self.training_reward_count += int(training_array.size)
        shaped_array = (
            np.zeros_like(training_array)
            if training_shaped_rewards is None
            else np.asarray(training_shaped_rewards, dtype=np.float64)
        )
        if shaped_array.shape != training_array.shape or not bool(
            np.isfinite(shaped_array).all()
        ):
            raise ValueError("Shaped training rewards must match the training channel.")
        self.shaped_reward_sum += float(shaped_array.sum())
        self.shaped_reward_count += int(shaped_array.size)
        if shaped_rewards_by_slot is not None:
            for slot, values in shaped_rewards_by_slot.items():
                slot_array = np.asarray(values, dtype=np.float64)
                if slot_array.shape != (self.batch_size,) or not bool(
                    np.isfinite(slot_array).all()
                ):
                    raise ValueError("Per-slot shaped reward diagnostics are misaligned.")
                key = str(slot)
                self.shaped_reward_sum_by_slot[key] = (
                    self.shaped_reward_sum_by_slot.get(key, 0.0)
                    + float(slot_array.sum())
                )
        factor = 0.0 if anneal_factor is None else float(anneal_factor)
        if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
            raise ValueError("Reward-shaping anneal factor must lie in [0,1].")
        self.anneal_factors.append(factor)
        self.running_episode_returns += reward_array
        if bool(done_array.any()):
            self.completed_episode_returns.extend(
                float(value) for value in self.running_episode_returns[done_array]
            )
            self.running_episode_returns[done_array] = 0.0
        self.probe_count += int(probe_count)
        probe_fields = (
            probe_candidate_disagreement,
            probe_candidate_q,
            probe_candidate_regret,
            probe_candidate_equals_greedy,
            probe_disagreement_pass,
            probe_regret_pass,
            probe_budget_pass,
        )
        if any(value is not None for value in probe_fields):
            if any(value is None for value in probe_fields):
                raise ValueError("Probe diagnostics must be supplied together.")
            disagreement = np.asarray(probe_candidate_disagreement, dtype=np.float64)
            candidate_q = np.asarray(probe_candidate_q, dtype=np.float64)
            candidate_regret = np.asarray(probe_candidate_regret, dtype=np.float64)
            candidate_equals_greedy = np.asarray(
                probe_candidate_equals_greedy,
                dtype=bool,
            )
            disagreement_pass = np.asarray(probe_disagreement_pass, dtype=bool)
            regret_pass = np.asarray(probe_regret_pass, dtype=bool)
            budget_pass = np.asarray(probe_budget_pass, dtype=bool)
            expected = (self.batch_size,)
            if any(
                value.shape != expected
                for value in (
                    disagreement,
                    candidate_q,
                    candidate_regret,
                    candidate_equals_greedy,
                    disagreement_pass,
                    regret_pass,
                    budget_pass,
                )
            ):
                raise ValueError("Probe diagnostics differ from batch_size.")
            if not bool(np.isfinite(disagreement).all()) or not bool(
                np.isfinite(candidate_q).all()
            ) or not bool(np.isfinite(candidate_regret).all()):
                raise ValueError("Enabled probe diagnostics must be finite.")
            self.probe_candidate_disagreement.extend(disagreement.tolist())
            self.probe_candidate_q.extend(candidate_q.tolist())
            self.probe_candidate_regret.extend(candidate_regret.tolist())
            self.probe_candidate_equals_greedy_count += int(
                candidate_equals_greedy.sum()
            )
            self.probe_disagreement_pass_count += int(disagreement_pass.sum())
            self.probe_regret_pass_count += int(regret_pass.sum())
            self.probe_budget_pass_count += int(budget_pass.sum())
            self.probe_candidate_count += int(disagreement.size)

    def record_update(
        self,
        *,
        loss: float,
        per_head_loss: Sequence[float],
        per_head_support: Sequence[int],
        gradient_norm: float,
    ) -> None:
        loss = float(loss)
        gradient_norm = float(gradient_norm)
        if not math.isfinite(loss) or not math.isfinite(gradient_norm):
            raise ValueError("Training loss and gradient norm must be finite.")
        if len(per_head_loss) != self.n_heads or len(per_head_support) != self.n_heads:
            raise ValueError("Per-head training diagnostics are misaligned.")
        self.losses.append(loss)
        self.gradient_norms.append(gradient_norm)
        if gradient_norm > float(self.gradient_clip_norm):
            self.clipped_gradient_count += 1
        for index, (head_loss, support) in enumerate(
            zip(per_head_loss, per_head_support, strict=True)
        ):
            support = int(support)
            if support < 0:
                raise ValueError("Per-head TD support must be non-negative.")
            self.per_head_support[index] += support
            if support > 0:
                value = float(head_loss)
                if not math.isfinite(value):
                    raise ValueError("Per-head TD loss must be finite when supported.")
                self.per_head_losses[index].append(value)

    def build_row(
        self,
        *,
        seed: int,
        layout: str,
        phase: str,
        phase_environment_steps: int,
        total_environment_steps: int,
        cumulative_episodes: int,
        gradient_updates: int,
        epsilon: float,
        cumulative_probe_count: int,
        model: PrimitiveRecurrentEnsembleQ,
    ) -> dict[str, Any]:
        if self.environment_steps <= 0:
            raise ValueError("Cannot write an empty training metric window.")
        returns = np.asarray(self.completed_episode_returns, dtype=np.float64)
        if returns.size and not bool(np.isfinite(returns).all()):
            raise ValueError("Completed episode returns must be finite.")
        return_summary: dict[str, Any] = {
            "count": int(returns.size),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
        if returns.size:
            return_summary.update(
                {
                    "mean": float(returns.mean()),
                    "std": float(returns.std(ddof=0)),
                    "min": float(returns.min()),
                    "max": float(returns.max()),
                    "p10": float(np.quantile(returns, 0.10)),
                    "p50": float(np.quantile(returns, 0.50)),
                    "p90": float(np.quantile(returns, 0.90)),
                }
            )
        gradient_summary = _numeric_summary(self.gradient_norms)
        gradient_summary["clipped_count"] = int(self.clipped_gradient_count)
        gradient_summary["clipped_fraction"] = (
            None
            if not self.gradient_norms
            else float(self.clipped_gradient_count / len(self.gradient_norms))
        )
        row = {
            "schema_version": TRAINING_METRICS_SCHEMA_VERSION,
            "seed": int(seed),
            "layout": str(layout),
            "phase": str(phase),
            "phase_environment_steps": int(phase_environment_steps),
            "total_environment_steps": int(total_environment_steps),
            "window_environment_steps": int(self.environment_steps),
            "cumulative_episodes": int(cumulative_episodes),
            "gradient_updates": int(gradient_updates),
            "epsilon": float(epsilon),
            "window_probe_count": int(self.probe_count),
            "cumulative_probe_count": int(cumulative_probe_count),
            "td_loss": _numeric_summary(self.losses),
            "per_head_td_loss_mean": [
                None if not values else float(np.mean(np.asarray(values)))
                for values in self.per_head_losses
            ],
            "per_head_td_support": [int(value) for value in self.per_head_support],
            "gradient_l2_norm_before_clipping": gradient_summary,
            "raw_step_reward": {
                "sum": float(self.raw_reward_sum),
                "mean_per_environment_step": float(
                    self.raw_reward_sum / self.environment_steps
                ),
                "net_positive_reward_step_count_legacy": int(
                    self.positive_reward_count
                ),
                "positive_event_count": int(self.positive_reward_count),
                "net_negative_reward_step_count_legacy": int(
                    self.negative_reward_count
                ),
                "negative_event_count": int(self.negative_reward_count),
                "legacy_count_meaning": (
                    "positive_event_count and negative_event_count count net-signed "
                    "reward steps, not decomposed environment events"
                ),
                "zero_event_count": int(self.zero_reward_count),
                "correct_delivery_count": int(self.correct_delivery_count),
                "wrong_delivery_count": int(self.wrong_delivery_count),
                "indicator_activation_count": int(
                    self.indicator_activation_count
                ),
                "ambiguous_reward_step_count": int(
                    self.ambiguous_reward_step_count
                ),
            },
            "training_step_reward": {
                "sum": float(self.training_reward_sum),
                "count": int(self.training_reward_count),
                "mean_per_trained_transition": float(
                    self.training_reward_sum / self.training_reward_count
                ),
            },
            "shaped_training_reward": {
                "annealed_sum": float(self.shaped_reward_sum),
                "count": int(self.shaped_reward_count),
                "mean_per_trained_transition": float(
                    self.shaped_reward_sum / self.shaped_reward_count
                ),
                "environment_native_sum_by_slot": dict(
                    sorted(self.shaped_reward_sum_by_slot.items())
                ),
                "anneal_factor": _numeric_summary(self.anneal_factors),
            },
            "probe_diagnostics": {
                "all_heads_agree": bool(
                    self.probe_candidate_count > 0
                    and max(self.probe_candidate_disagreement, default=math.inf)
                    <= 1.0e-8
                ),
                "candidate_count": int(self.probe_candidate_count),
                "candidate_disagreement": _quantile_summary(
                    self.probe_candidate_disagreement
                ),
                "candidate_q": _quantile_summary(self.probe_candidate_q),
                "candidate_regret": _quantile_summary(
                    self.probe_candidate_regret
                ),
                "candidate_equals_greedy_count": int(
                    self.probe_candidate_equals_greedy_count
                ),
                "disagreement_threshold_pass_count": int(
                    self.probe_disagreement_pass_count
                ),
                "disagreement_threshold_pass_fraction": (
                    None
                    if self.probe_candidate_count == 0
                    else float(
                        self.probe_disagreement_pass_count
                        / self.probe_candidate_count
                    )
                ),
                "regret_pass_count": int(
                    self.probe_regret_pass_count
                ),
                "regret_pass_fraction": (
                    None
                    if self.probe_candidate_count == 0
                    else float(
                        self.probe_regret_pass_count
                        / self.probe_candidate_count
                    )
                ),
                "budget_and_window_pass_count": int(
                    self.probe_budget_pass_count
                ),
                "budget_and_window_pass_fraction": (
                    None
                    if self.probe_candidate_count == 0
                    else float(
                        self.probe_budget_pass_count
                        / self.probe_candidate_count
                    )
                ),
                "actual_probe_count": int(self.probe_count),
            },
            "raw_episode_return": return_summary,
            "model_parameter_health": model_parameter_health(
                model,
                reference=self.reference,
            ),
        }
        self._reset_window()
        return row

    def _reset_window(self) -> None:
        self.environment_steps = 0
        self.raw_reward_sum = 0.0
        self.positive_reward_count = 0
        self.negative_reward_count = 0
        self.zero_reward_count = 0
        self.correct_delivery_count = 0
        self.wrong_delivery_count = 0
        self.indicator_activation_count = 0
        self.ambiguous_reward_step_count = 0
        self.training_reward_sum = 0.0
        self.training_reward_count = 0
        self.shaped_reward_sum = 0.0
        self.shaped_reward_count = 0
        self.shaped_reward_sum_by_slot.clear()
        self.anneal_factors.clear()
        self.completed_episode_returns.clear()
        self.losses.clear()
        self.per_head_losses = [[] for _ in range(int(self.n_heads))]
        self.per_head_support = [0 for _ in range(int(self.n_heads))]
        self.gradient_norms.clear()
        self.clipped_gradient_count = 0
        self.probe_count = 0
        self.probe_candidate_disagreement.clear()
        self.probe_candidate_q.clear()
        self.probe_candidate_regret.clear()
        self.probe_candidate_equals_greedy_count = 0
        self.probe_disagreement_pass_count = 0
        self.probe_regret_pass_count = 0
        self.probe_budget_pass_count = 0
        self.probe_candidate_count = 0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if payload.get("schema_version") != TRAINING_METRICS_SCHEMA_VERSION:
            raise ValueError(f"Unexpected training metric schema at line {line_number}.")
        rows.append(dict(payload))
    return rows


def write_training_curves(metrics_path: str | Path, output_path: str | Path) -> None:
    rows = _read_jsonl(metrics_path)
    if not rows:
        raise ValueError("Training curve generation requires at least one metric row.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([row["total_environment_steps"] for row in rows], dtype=float)
    loss = np.asarray(
        [np.nan if row["td_loss"]["mean"] is None else row["td_loss"]["mean"] for row in rows]
    )
    returns = np.asarray(
        [
            np.nan
            if row["raw_episode_return"]["mean"] is None
            else row["raw_episode_return"]["mean"]
            for row in rows
        ]
    )
    return_p10 = np.asarray(
        [
            np.nan
            if row["raw_episode_return"]["p10"] is None
            else row["raw_episode_return"]["p10"]
            for row in rows
        ]
    )
    return_p90 = np.asarray(
        [
            np.nan
            if row["raw_episode_return"]["p90"] is None
            else row["raw_episode_return"]["p90"]
            for row in rows
        ]
    )
    reward_mean = np.asarray(
        [row["raw_step_reward"]["mean_per_environment_step"] for row in rows]
    )
    training_reward_mean = np.asarray(
        [
            row["training_step_reward"]["mean_per_trained_transition"]
            for row in rows
        ]
    )
    shaped_reward_mean = np.asarray(
        [
            row["shaped_training_reward"]["mean_per_trained_transition"]
            for row in rows
        ]
    )
    grad_mean = np.asarray(
        [
            np.nan
            if row["gradient_l2_norm_before_clipping"]["mean"] is None
            else row["gradient_l2_norm_before_clipping"]["mean"]
            for row in rows
        ]
    )
    parameter_l2 = np.asarray(
        [row["model_parameter_health"]["learned"]["l2_norm"] for row in rows]
    )
    parameter_delta = np.asarray(
        [
            row["model_parameter_health"]["learned"]["delta_l2_from_reference"]
            for row in rows
        ]
    )
    probes = np.asarray([row["window_probe_count"] for row in rows], dtype=float)
    epsilon = np.asarray([row["epsilon"] for row in rows], dtype=float)
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    axes[0, 0].plot(steps, loss)
    axes[0, 0].set_title("TD loss per logging window")
    axes[0, 1].plot(steps, returns)
    axes[0, 1].fill_between(steps, return_p10, return_p90, alpha=0.2)
    axes[0, 1].set_title("Raw episode return (p10-p90)")
    axes[1, 0].plot(steps, reward_mean, label="raw team reward")
    axes[1, 0].plot(steps, training_reward_mean, label="TD training reward")
    axes[1, 0].plot(steps, shaped_reward_mean, label="annealed shaped component")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("Reward channels")
    axes[1, 1].plot(steps, grad_mean)
    axes[1, 1].set_title("Gradient L2 norm before clipping")
    axes[2, 0].plot(steps, parameter_l2, label="parameter L2")
    axes[2, 0].plot(steps, parameter_delta, label="delta from phase start")
    axes[2, 0].legend()
    axes[2, 0].set_title("Learned parameter health")
    axes[2, 1].plot(steps, probes, label="probes per window")
    epsilon_axis = axes[2, 1].twinx()
    epsilon_axis.plot(steps, epsilon, color="tab:orange", label="epsilon")
    axes[2, 1].set_title("Probes and exploration")
    for axis in axes.flat:
        axis.set_xlabel("total environment steps")
        axis.grid(alpha=0.25)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_standard_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Inspect four self-play snapshots and one final Path C checkpoint."""

    if len(checkpoint_paths) != 5:
        raise ValueError("Checkpoint health inspection requires four snapshots and one final model.")
    self_play_entries: list[dict[str, Any]] = []
    final_entries: list[dict[str, Any]] = []
    previous_reference: dict[str, torch.Tensor] | None = None
    prior_hashes: set[str | None] = set()
    for raw_path in checkpoint_paths:
        path = Path(raw_path)
        model, metadata = load_standard_checkpoint(path, device=device)
        prior_hashes.add(metadata.get("prior_state_sha256"))
        phase = str(metadata.get("phase"))
        is_self_play = phase == "self_play_partner_snapshot"
        health = model_parameter_health(
            model,
            reference=previous_reference if is_self_play else None,
        )
        entry = {
            "path": str(path),
            "checkpoint_sha256": _sha256_file(path),
            "metadata": metadata,
            "parameter_health": health,
        }
        if is_self_play:
            self_play_entries.append(entry)
            previous_reference = snapshot_trainable_parameters(model)
        elif phase == "path_c_final":
            final_entries.append(entry)
        else:
            raise ValueError(f"Unexpected checkpoint phase for health inspection: {phase!r}.")
        del model
    self_play_entries.sort(key=lambda item: int(item["metadata"]["environment_steps"]))
    if len(self_play_entries) != 4 or len(final_entries) != 1:
        raise ValueError("Checkpoint health inspection did not receive four snapshots and one final model.")
    payload = {
        "schema_version": CHECKPOINT_HEALTH_SCHEMA_VERSION,
        "checkpoint_count": 5,
        "fixed_prior_consistent": len(prior_hashes) == 1,
        "fixed_prior_state_sha256": next(iter(prior_hashes)) if len(prior_hashes) == 1 else None,
        "self_play_snapshots": self_play_entries,
        "path_c_final": final_entries[0],
        "interpretation_boundary": (
            "Self-play snapshots form one parameter trajectory; the fresh Path C "
            "model is reported separately and is not connected to that trajectory."
        ),
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    health_path = target_dir / "checkpoint_health.json"
    health_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_checkpoint_health_plot(payload, target_dir / "checkpoint_parameter_health.png")
    return payload


def _write_checkpoint_health_plot(payload: Mapping[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snapshots = list(payload["self_play_snapshots"])
    steps = np.asarray(
        [item["metadata"]["environment_steps"] for item in snapshots],
        dtype=float,
    )
    group_names = (
        "observation_encoder",
        "shared_encoder",
        "recurrent_heads",
        "value_heads",
        "advantage_heads",
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for group in group_names:
        axes[0].plot(
            steps,
            [item["parameter_health"]["groups"][group]["l2_norm"] for item in snapshots],
            marker="o",
            label=group,
        )
    axes[0].set_title("Self-play snapshot parameter L2 norms")
    axes[0].set_xlabel("self-play environment steps")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    final_groups = payload["path_c_final"]["parameter_health"]["groups"]
    axes[1].bar(
        np.arange(len(group_names)),
        [final_groups[group]["l2_norm"] for group in group_names],
    )
    axes[1].set_xticks(np.arange(len(group_names)), group_names, rotation=35, ha="right")
    axes[1].set_title("Final Path C parameter L2 norms")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_return_distribution(
    returns: Sequence[float],
    output_path: str | Path,
) -> None:
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0 or not bool(np.isfinite(values).all()):
        raise ValueError("Return-distribution plotting requires finite returns.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.hist(values, bins=min(30, max(1, int(math.sqrt(values.size)))), edgecolor="black")
    axis.axvline(float(values.mean()), color="tab:red", label=f"mean={values.mean():.3f}")
    axis.axvline(float(np.median(values)), color="tab:orange", label=f"median={np.median(values):.3f}")
    axis.set_xlabel("raw 400-step episode return")
    axis.set_ylabel("episode count")
    axis.set_title("Seed 101 self-play calibration")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
