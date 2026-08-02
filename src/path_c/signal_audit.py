"""Pure statistical audits for the r3 signal contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .qualification import GateDecision, bootstrap_interval


@dataclass(frozen=True, slots=True)
class DecisionSupportAudit:
    between_partner_variance: float
    within_partner_variance: float
    variance_difference_lcb: float
    oracle_lift_lcb: float
    stable_action_regions: int
    heldout_ordering_accuracy_lcb: float

    @property
    def passed(self) -> bool:
        return bool(
            self.variance_difference_lcb > 0.0
            and self.oracle_lift_lcb > 0.0
            and self.stable_action_regions >= 2
            and self.heldout_ordering_accuracy_lcb > 0.5
        )


@dataclass(frozen=True, slots=True)
class RawQAudit:
    selected_action_lift_lcb: float
    pairwise_concordance_lcb: float
    calibration_error: float
    monte_carlo_error_bound: float
    epoch_ordering_stable: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.selected_action_lift_lcb > 0.0
            and self.pairwise_concordance_lcb > 0.5
            and self.calibration_error <= self.monte_carlo_error_bound
            and self.epoch_ordering_stable
        )


@dataclass(frozen=True, slots=True)
class InferenceAudit:
    oracle_context_lift_lcb: float
    online_context_lift_lcb: float

    @property
    def passed(self) -> bool:
        return self.oracle_context_lift_lcb > 0.0 and self.online_context_lift_lcb > 0.0

    @property
    def recovery_ratio(self) -> float | None:
        if self.oracle_context_lift_lcb <= 0.0:
            return None
        return self.online_context_lift_lcb / self.oracle_context_lift_lcb


@dataclass(frozen=True, slots=True)
class ConditionalControlAudit:
    return_lift_lcb: float
    negative_transfer_rate: float
    conditional_to_base_kl: float
    residual_rms: float

    @property
    def passed(self) -> bool:
        return bool(
            self.return_lift_lcb > 0.0
            and self.negative_transfer_rate <= 0.05
            and self.conditional_to_base_kl <= 0.03
            and self.residual_rms <= 1.0
        )


@dataclass(frozen=True, slots=True)
class ActiveInformationAudit:
    opportunity_lcb: float
    raw_q_selection_lcb: float
    online_inference_lcb: float
    regret_calibration_lcb: float
    active_minus_passive_lcb: float

    @property
    def passed(self) -> bool:
        return bool(
            self.opportunity_lcb > 0.0
            and self.raw_q_selection_lcb > 0.0
            and self.online_inference_lcb > 0.0
            and self.regret_calibration_lcb > 0.0
            and self.active_minus_passive_lcb >= 0.0
        )


@dataclass(frozen=True, slots=True)
class GeneratorDecisionAudit:
    between_code_variance: float
    within_code_noise: float
    variance_difference_lcb: float
    oracle_code_lift_lcb: float
    stable_action_regions: int

    @property
    def passed(self) -> bool:
        return bool(
            self.variance_difference_lcb > 0.0
            and self.oracle_code_lift_lcb > 0.0
            and self.stable_action_regions >= 2
        )


def action_signature_variances(
    signatures: Any,
    partner_ids: Any,
    *,
    common_world_ids: Any | None = None,
) -> tuple[float, float]:
    values = np.asarray(signatures, dtype=np.float64)
    ids = np.asarray(partner_ids)
    if values.ndim != 2 or ids.shape != (values.shape[0],):
        raise ValueError("Signatures must be [sample, action] with one partner id per sample.")
    centered = values - np.mean(values, axis=-1, keepdims=True)
    if common_world_ids is not None:
        worlds = np.asarray(common_world_ids)
        if worlds.shape != ids.shape:
            raise ValueError("Common-world ids must align with signatures.")
        # Remove physical-state effects before attributing variance to partner
        # behavior.  This is the common-world intervention estimand used by C1.
        for world in np.unique(worlds):
            mask = worlds == world
            centered[mask] -= np.mean(centered[mask], axis=0, keepdims=True)
    partner_means = []
    within = []
    for partner in np.unique(ids):
        rows = centered[ids == partner]
        if rows.shape[0] < 2:
            raise ValueError("Each partner needs repeated Monte Carlo signature estimates.")
        partner_means.append(np.mean(rows, axis=0))
        within.append(np.mean(np.var(rows, axis=0, ddof=1)))
    between = float(np.mean(np.var(np.stack(partner_means), axis=0, ddof=1)))
    return between, float(np.mean(within))


def residualize_common_world_signatures(signatures: Any, common_world_ids: Any) -> np.ndarray:
    """Remove common physical-world effects before partner-run resampling."""

    values = np.asarray(signatures, dtype=np.float64)
    worlds = np.asarray(common_world_ids)
    if values.ndim != 2 or worlds.shape != (values.shape[0],):
        raise ValueError("Common-world signature rows are misaligned.")
    residual = values - np.mean(values, axis=-1, keepdims=True)
    for world in np.unique(worlds):
        mask = worlds == world
        residual[mask] -= np.mean(residual[mask], axis=0, keepdims=True)
    return residual


def privileged_functional_contexts(
    *, signatures: Any, partner_ids: Any, latent_dim: int
) -> np.ndarray:
    """Build a fixed diagnostic context from empirical decision signatures.

    The returned coordinate contains no checkpoint identity feature.  Partner
    IDs are used only to aggregate repeated real-return signatures and to
    index the resulting privileged diagnostic during paired C3 rollouts.
    Coordinates are the deterministic SVD embedding of centered partner mean
    action signatures, padded to the registered latent width.
    """

    values = np.asarray(signatures, dtype=np.float64)
    ids = np.asarray(partner_ids)
    dimension = int(latent_dim)
    if values.ndim != 2 or ids.shape != (values.shape[0],) or dimension <= 0:
        raise ValueError("Functional-context signature rows are malformed.")
    unique = np.unique(ids)
    if not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("Functional diagnostic partner IDs must be contiguous member indexes.")
    centered = values - np.mean(values, axis=-1, keepdims=True)
    means = np.stack([np.mean(centered[ids == partner], axis=0) for partner in unique])
    means -= np.mean(means, axis=0, keepdims=True)
    left, singular, unused_right = np.linalg.svd(means, full_matrices=False)
    del unused_right
    rank = min(dimension, left.shape[1])
    coordinates = left[:, :rank] * singular[None, :rank]
    # SVD signs are arbitrary; make the representation byte-deterministic.
    for column in range(rank):
        pivot = int(np.argmax(np.abs(coordinates[:, column])))
        if coordinates[pivot, column] < 0.0:
            coordinates[:, column] *= -1.0
    result = np.zeros((unique.size, dimension), dtype=np.float32)
    result[:, :rank] = coordinates.astype(np.float32)
    return result


def decision_support_audit(
    *,
    signatures: Any,
    partner_ids: Any,
    oracle_lift_blocks: Iterable[float],
    stable_best_actions: Iterable[int],
    heldout_ordering_blocks: Iterable[float],
    seed: int,
    common_world_ids: Any | None = None,
) -> DecisionSupportAudit:
    """Run-block C1 audit with no training-row pseudoreplication."""

    values = np.asarray(signatures, dtype=np.float64)
    ids = np.asarray(partner_ids)
    unique = np.unique(ids)
    if unique.size < 2:
        raise ValueError("C1 needs at least two independent partner runs.")
    if common_world_ids is not None:
        values = residualize_common_world_signatures(values, common_world_ids)
    between, within = action_signature_variances(values, ids)
    rng = np.random.default_rng(int(seed))
    variance_differences = []
    for _ in range(9_999):
        selected = rng.choice(unique, size=unique.size, replace=True)
        rows = []
        labels = []
        for block, partner in enumerate(selected):
            partner_rows = values[ids == partner]
            row_indexes = rng.integers(0, partner_rows.shape[0], size=partner_rows.shape[0])
            rows.append(partner_rows[row_indexes])
            labels.extend([block] * partner_rows.shape[0])
        sampled_between, sampled_within = action_signature_variances(
            np.concatenate(rows, axis=0), np.asarray(labels)
        )
        variance_differences.append(sampled_between - sampled_within)
    variance_lcb = float(np.quantile(variance_differences, 0.05))
    unused_oracle_point, oracle_lcb, unused_oracle_ucb = bootstrap_interval(
        oracle_lift_blocks, seed=seed + 1
    )
    unused_order_point, ordering_lcb, unused_order_ucb = bootstrap_interval(
        heldout_ordering_blocks, seed=seed + 2
    )
    del unused_oracle_point, unused_oracle_ucb, unused_order_point, unused_order_ucb
    return DecisionSupportAudit(
        between_partner_variance=between,
        within_partner_variance=within,
        variance_difference_lcb=variance_lcb,
        oracle_lift_lcb=oracle_lcb,
        stable_action_regions=len(set(int(value) for value in stable_best_actions)),
        heldout_ordering_accuracy_lcb=ordering_lcb,
    )


def raw_q_audit(
    *,
    selected_action_lift_blocks: Iterable[float],
    pairwise_concordance_blocks: Iterable[float],
    prediction_errors: Iterable[float],
    monte_carlo_error_bound: float,
    fit_rankings: Any,
    evaluation_rankings: Any,
    seed: int,
) -> RawQAudit:
    _, selected_lcb, _ = bootstrap_interval(selected_action_lift_blocks, seed=seed)
    _, concordance_lcb, _ = bootstrap_interval(
        pairwise_concordance_blocks, seed=seed + 1
    )
    errors = np.asarray(tuple(prediction_errors), dtype=np.float64)
    fit = np.asarray(fit_rankings)
    evaluation = np.asarray(evaluation_rankings)
    if errors.size == 0 or fit.shape != evaluation.shape:
        raise ValueError("C2 calibration/ranking audit records are incomplete.")
    return RawQAudit(
        selected_action_lift_lcb=selected_lcb,
        pairwise_concordance_lcb=concordance_lcb,
        calibration_error=float(np.max(np.abs(errors))),
        monte_carlo_error_bound=float(monte_carlo_error_bound),
        epoch_ordering_stable=bool(np.mean(fit == evaluation) > 0.5),
    )


def leave_one_partner_out_ordering_accuracy(
    *,
    features: Any,
    fit_signatures: Any,
    evaluation_signatures: Any,
    partner_ids: Any,
    minimum_action_gap: float = 1.0,
    ridge: float = 1.0,
) -> np.ndarray:
    """Legal-history readout transfer; partner identity is split-only metadata."""

    x = np.asarray(features, dtype=np.float64)
    fit = np.asarray(fit_signatures, dtype=np.float64)
    evaluation = np.asarray(evaluation_signatures, dtype=np.float64)
    ids = np.asarray(partner_ids)
    if x.ndim != 2 or fit.shape != evaluation.shape or fit.shape[0] != x.shape[0]:
        raise ValueError("LOPO decision readout arrays are misaligned.")
    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    scores = []
    for heldout in np.unique(ids):
        train = ids != heldout
        test = ids == heldout
        if np.sum(train) < 2 or not np.any(test):
            raise ValueError("LOPO fold lacks independent train/test rows.")
        gram = x[train].T @ x[train] + float(ridge) * np.eye(x.shape[1])
        weights = np.linalg.solve(gram, x[train].T @ fit[train])
        prediction = x[test] @ weights
        correct = []
        for predicted, truth in zip(prediction, evaluation[test]):
            for left in range(truth.shape[0]):
                for right in range(left + 1, truth.shape[0]):
                    difference = truth[left] - truth[right]
                    if abs(difference) < float(minimum_action_gap):
                        continue
                    correct.append(
                        float((predicted[left] - predicted[right]) * difference > 0.0)
                    )
        if not correct:
            scores.append(0.5)
        else:
            scores.append(float(np.mean(correct)))
    return np.asarray(scores, dtype=np.float64)


def top_bottom_regret_calibration(
    predicted_regret: Iterable[float],
    empirical_oracle_lift: Iterable[float],
    *,
    seed: int,
) -> tuple[float, float, float]:
    regret = np.asarray(tuple(predicted_regret), dtype=np.float64)
    lift = np.asarray(tuple(empirical_oracle_lift), dtype=np.float64)
    if regret.shape != lift.shape or regret.size < 8:
        raise ValueError("Regret calibration needs at least eight paired audit anchors.")
    order = np.argsort(regret)
    count = max(regret.size // 4, 1)
    differences = lift[order[-count:]] - lift[order[:count]]
    return bootstrap_interval(differences, seed=seed)


def active_information_audit(
    *,
    opportunity_blocks: Iterable[float],
    raw_q_selection_blocks: Iterable[float],
    online_inference_blocks: Iterable[float],
    predicted_regret: Iterable[float],
    empirical_oracle_lift: Iterable[float],
    active_minus_passive_blocks: Iterable[float],
    seed: int,
) -> ActiveInformationAudit:
    """Evaluate the five C5 subgates at independent run/block level."""

    _, opportunity_lcb, _ = bootstrap_interval(opportunity_blocks, seed=seed)
    _, raw_q_lcb, _ = bootstrap_interval(raw_q_selection_blocks, seed=seed + 1)
    _, inference_lcb, _ = bootstrap_interval(online_inference_blocks, seed=seed + 2)
    _, regret_lcb, _ = top_bottom_regret_calibration(
        predicted_regret, empirical_oracle_lift, seed=seed + 3
    )
    _, control_lcb, _ = bootstrap_interval(
        active_minus_passive_blocks, seed=seed + 4
    )
    return ActiveInformationAudit(
        opportunity_lcb=opportunity_lcb,
        raw_q_selection_lcb=raw_q_lcb,
        online_inference_lcb=inference_lcb,
        regret_calibration_lcb=regret_lcb,
        active_minus_passive_lcb=control_lcb,
    )


def generator_decision_audit(
    *,
    fit_signatures: Any,
    evaluation_signatures: Any,
    code_region_ids: Any,
    oracle_lift_blocks: Iterable[float],
    stable_best_actions: Iterable[int],
    seed: int,
) -> GeneratorDecisionAudit:
    """Noise-corrected code-region admission based on simulator signatures."""

    fit = np.asarray(fit_signatures, dtype=np.float64)
    evaluation = np.asarray(evaluation_signatures, dtype=np.float64)
    regions = np.asarray(code_region_ids)
    if fit.shape != evaluation.shape or fit.ndim != 2:
        raise ValueError("Generator signature fit/evaluation arrays differ.")
    if regions.shape != (fit.shape[0],) or np.unique(regions).size < 2:
        raise ValueError("Generator admission needs repeated rows in two code regions.")
    residual = fit - evaluation
    region_means = [np.mean(fit[regions == value], axis=0) for value in np.unique(regions)]
    between = float(np.mean(np.var(np.stack(region_means), axis=0, ddof=1)))
    within = float(np.mean(np.square(residual)) / 2.0)
    unique = np.unique(regions)
    rng = np.random.default_rng(int(seed))
    differences = []
    for _ in range(9_999):
        selected = rng.choice(unique, size=unique.size, replace=True)
        sampled_means = []
        sampled_noise = []
        for region in selected:
            rows = np.flatnonzero(regions == region)
            chosen = rng.choice(rows, size=rows.size, replace=True)
            sampled_means.append(np.mean(fit[chosen], axis=0))
            sampled_noise.append(np.mean(np.square(residual[chosen])) / 2.0)
        if len(sampled_means) < 2:
            differences.append(-np.inf)
        else:
            differences.append(
                float(np.mean(np.var(np.stack(sampled_means), axis=0, ddof=1)))
                - float(np.mean(sampled_noise))
            )
    _, oracle_lcb, _ = bootstrap_interval(oracle_lift_blocks, seed=seed + 1)
    return GeneratorDecisionAudit(
        between_code_variance=between,
        within_code_noise=within,
        variance_difference_lcb=float(np.quantile(differences, 0.05)),
        oracle_code_lift_lcb=oracle_lcb,
        stable_action_regions=len(set(int(value) for value in stable_best_actions)),
    )


def fixture_contract_outcome(
    *,
    decision_relevant: bool,
    history_identifiable: bool,
    regret_calibrated: bool,
) -> Mapping[str, bool]:
    c1 = bool(decision_relevant)
    c3 = bool(c1 and history_identifiable)
    c5 = bool(c3 and regret_calibrated)
    return {"C1": c1, "C3": c3, "C5": c5}


__all__ = [
    "ActiveInformationAudit",
    "ConditionalControlAudit",
    "DecisionSupportAudit",
    "GeneratorDecisionAudit",
    "InferenceAudit",
    "RawQAudit",
    "active_information_audit",
    "action_signature_variances",
    "decision_support_audit",
    "fixture_contract_outcome",
    "generator_decision_audit",
    "leave_one_partner_out_ordering_accuracy",
    "raw_q_audit",
    "residualize_common_world_signatures",
    "privileged_functional_contexts",
    "top_bottom_regret_calibration",
]
