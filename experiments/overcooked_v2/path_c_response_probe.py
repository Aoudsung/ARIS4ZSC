"""WP-D (§6): observable response-model ensemble and Path C v3 probe controller."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.overcooked_v2.path_c_sequence import EpisodeBootstrapRecord
from experiments.overcooked_v2.path_c_response_summary import ResponseSummarySpecV1


ACTION_RESPONSE_VOCABULARY = (
    "up", "down", "right", "left", "stay", "interact", "unseen"
)
REGISTERED_LOCAL_RESPONSE_SPEC = ResponseSummarySpecV1(
    response_classes=("visible", "unseen", "local_non_agent_change"),
    latency_bin_upper_bounds=(1,),
)
OBSERVATION_RESPONSE_VOCABULARY = REGISTERED_LOCAL_RESPONSE_SPEC.vocabulary


def _registered_local_response_token(response_class: str) -> int:
    return REGISTERED_LOCAL_RESPONSE_SPEC.encode(
        response_class=response_class,
        latency_steps=1,
    )


def partner_visible_from_default_observation(
    observations: np.ndarray,
    *,
    indicate_successful_delivery: bool,
) -> np.ndarray:
    """Read the other-agent position layer from JaxMARL v0.1.0 default observations.

    The official encoder concatenates own-agent layers followed by other-agent
    layers; each agent block is position(1), direction(4), and inventory(n+2).
    """

    array = np.asarray(observations)
    if array.ndim != 4:
        raise ValueError("Default observations must have shape [B,H,W,C].")
    extra_delivery_layer = 1 if indicate_successful_delivery else 0
    remainder = array.shape[-1] - 26 - extra_delivery_layer
    if remainder < 0 or remainder % 4:
        raise ValueError("Observation channels do not match the official default encoding.")
    num_ingredients = remainder // 4
    other_agent_position_channel = num_ingredients + 7
    return np.any(array[..., other_agent_position_channel] > 0, axis=(1, 2))


def local_non_agent_change_from_default_observation(
    previous_observations: np.ndarray,
    next_observations: np.ndarray,
    *,
    indicate_successful_delivery: bool,
) -> np.ndarray:
    """Detect any locally visible non-agent-layer change around ego.

    The signal is intentionally not attributed to the partner: the same
    official observation change can be caused by ego or by environment state.
    """

    previous = np.asarray(previous_observations)
    following = np.asarray(next_observations)
    if previous.shape != following.shape or previous.ndim != 5:
        raise ValueError("Observation differences require matching [B,T,H,W,C] arrays.")
    extra_delivery_layer = 1 if indicate_successful_delivery else 0
    remainder = previous.shape[-1] - 26 - extra_delivery_layer
    if remainder < 0 or remainder % 4:
        raise ValueError("Observation channels do not match the official default encoding.")
    num_ingredients = remainder // 4
    first_non_agent_channel = 2 * (num_ingredients + 7)
    center_y = previous.shape[2] // 2
    center_x = previous.shape[3] // 2
    previous_local = previous[
        :, :, center_y - 1 : center_y + 2, center_x - 1 : center_x + 2,
        first_non_agent_channel:
    ]
    following_local = following[
        :, :, center_y - 1 : center_y + 2, center_x - 1 : center_x + 2,
        first_non_agent_channel:
    ]
    return np.any(previous_local != following_local, axis=(2, 3, 4))


def response_tokens_from_observations(
    *,
    partner_actions: np.ndarray,
    partner_visible: np.ndarray,
    local_non_agent_change: np.ndarray,
    partner_action_channel: bool,
) -> np.ndarray:
    """Vectorized observable response-token derivation for complete rollouts."""

    actions = np.asarray(partner_actions, dtype=np.int64)
    visible = np.asarray(partner_visible, dtype=bool)
    local_change = np.asarray(local_non_agent_change, dtype=bool)
    if actions.shape != visible.shape or actions.shape != local_change.shape:
        raise ValueError("Response-token inputs must have identical shapes.")
    if partner_action_channel:
        if np.any((actions < 0) | (actions >= 6)):
            raise ValueError("Partner action tokens require six primitive actions.")
        return np.where(visible, actions, 6).astype(np.int64)
    visible_token = _registered_local_response_token("visible")
    unseen_token = _registered_local_response_token("unseen")
    local_change_token = _registered_local_response_token("local_non_agent_change")
    return np.where(
        local_change,
        local_change_token,
        np.where(visible, visible_token, unseen_token),
    ).astype(np.int64)


def response_token_from_visibility(
    *,
    partner_action: int,
    partner_visible: bool,
    local_non_agent_change: bool,
    partner_action_channel: bool,
) -> int:
    """Map observable visibility and local change to the pre-registered token."""

    if partner_action_channel:
        if not partner_visible:
            return ACTION_RESPONSE_VOCABULARY.index("unseen")
        if int(partner_action) < 0 or int(partner_action) >= 6:
            raise ValueError("A visible partner action must be one of six primitives.")
        return int(partner_action)
    if local_non_agent_change:
        return _registered_local_response_token("local_non_agent_change")
    return _registered_local_response_token(
        "visible" if partner_visible else "unseen"
    )


@dataclass(frozen=True)
class ProbeCandidate:
    """One probe candidate; the tuple permits a later pre-registered micro-script."""

    actions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.actions or len(self.actions) > 3:
            raise ValueError("A probe candidate must contain one to three actions.")
        if any(action < 0 or action >= 6 for action in self.actions):
            raise ValueError("Probe actions must belong to the six primitive actions.")


@dataclass(frozen=True)
class ResponseProbeDecision:
    actions: np.ndarray
    is_probe: np.ndarray
    candidate_score: np.ndarray
    candidate_regret: np.ndarray
    candidate_equals_greedy: np.ndarray
    candidate_equals_baseline: np.ndarray
    disagreement_pass: np.ndarray
    regret_pass: np.ndarray
    budget_pass: np.ndarray


@dataclass(frozen=True)
class FinitePrototypeTwoActionValues:
    """Values from the finite-prototype, two-action diagnostic surrogate.

    This object is deliberately not the proposal's sequential ``J_use`` and
    ``J_mask``.  It marginalizes a response token over a finite partner index
    and then chooses one next primitive action.  It has no official-history
    transition ``x``, no complete hidden execution state, and no paired
    ``B_use``/``B_mask`` continuation branches.  In particular its use value is
    mathematically at least its masked value, whereas the registered sequential
    channel can hurt a finite controller.  Formal proposal code must therefore
    reject this surrogate.
    """

    j_use: torch.Tensor
    j_mask: torch.Tensor
    v_mask: torch.Tensor
    i_response: torch.Tensor
    c_task: torch.Tensor
    s_seq: torch.Tensor


class ResponseModel(nn.Module):
    """Small classifier whose recurrent state is private to one ensemble member."""

    def __init__(
        self,
        trunk_feature_dim: int,
        *,
        n_actions: int,
        vocabulary_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Linear(
            int(trunk_feature_dim) + self.n_actions,
            self.hidden_dim,
        )
        self.recurrent = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, int(vocabulary_size))

    def forward(
        self,
        detached_features: torch.Tensor,
        candidate_actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if detached_features.requires_grad:
            raise ValueError("Response models require explicitly detached trunk features.")
        if detached_features.ndim != 3 or candidate_actions.shape != detached_features.shape[:2]:
            raise ValueError("Response inputs must have shapes [B,T,D] and [B,T].")
        action_one_hot = F.one_hot(
            candidate_actions.to(torch.long), num_classes=self.n_actions
        ).to(detached_features.dtype)
        encoded = torch.tanh(
            self.input_projection(torch.cat((detached_features, action_one_hot), dim=-1))
        )
        recurrent, next_hidden = self.recurrent(encoded, hidden)
        return self.classifier(recurrent), next_hidden


class ResponseModelEnsemble(nn.Module):
    """Response predictors that can represent distinct partner hypotheses.

    Legacy checkpoints trained every member on a bootstrap sample of the same
    pooled data.  The revised path instead routes each complete episode to the
    member representing the partner that generated it.  Keeping the container
    shape unchanged preserves checkpoint loading while changing the statistic
    used by newly trained policies from estimator disagreement to partner
    disagreement.
    """

    def __init__(
        self,
        trunk_feature_dim: int,
        *,
        n_actions: int = 6,
        vocabulary_size: int = 7,
        hidden_dim: int = 64,
        ensemble_size: int = 5,
    ) -> None:
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("Response disagreement needs at least two models.")
        self.n_actions = int(n_actions)
        self.vocabulary_size = int(vocabulary_size)
        self.ensemble_size = int(ensemble_size)
        self.models = nn.ModuleList(
            ResponseModel(
                trunk_feature_dim,
                n_actions=n_actions,
                vocabulary_size=vocabulary_size,
                hidden_dim=hidden_dim,
            )
            for _ in range(ensemble_size)
        )

    def probabilities_for_candidates(
        self,
        detached_features: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return probabilities shaped [B,A,M,V] for one current feature row."""

        if detached_features.requires_grad:
            raise ValueError("Response ensemble features must be detached.")
        if detached_features.ndim != 2:
            raise ValueError("Current response features must have shape [B,D].")
        if candidate_actions.ndim != 1:
            raise ValueError("candidate_actions must have shape [A].")
        batch_size = detached_features.shape[0]
        action_count = candidate_actions.numel()
        expanded_features = detached_features[:, None, :].expand(
            batch_size, action_count, detached_features.shape[-1]
        ).reshape(batch_size * action_count, 1, -1)
        expanded_actions = candidate_actions[None, :].expand(
            batch_size, action_count
        ).reshape(batch_size * action_count, 1)
        predictions = []
        for model in self.models:
            logits, _ = model(expanded_features, expanded_actions)
            predictions.append(
                logits[:, 0].softmax(dim=-1).reshape(
                    batch_size, action_count, self.vocabulary_size
                )
            )
        return torch.stack(predictions, dim=2)

    def probabilities_for_actions(
        self,
        detached_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-hypothesis probabilities shaped ``[B,M,V]``."""

        if detached_features.requires_grad:
            raise ValueError("Response ensemble features must be detached.")
        if detached_features.ndim != 2 or actions.shape != detached_features.shape[:1]:
            raise ValueError("Response inputs must have shapes [B,D] and [B].")
        predictions = []
        for model in self.models:
            logits, _ = model(detached_features[:, None], actions[:, None])
            predictions.append(logits[:, 0].softmax(dim=-1))
        return torch.stack(predictions, dim=1)

    def bootstrap_cross_entropy(
        self,
        detached_features: torch.Tensor,
        actions: torch.Tensor,
        response_tokens: torch.Tensor,
        bootstrap_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Compute per-model supervised losses with episode-level bootstrap masks."""

        if bootstrap_mask.shape != (detached_features.shape[0], self.ensemble_size):
            raise ValueError("bootstrap_mask must have shape [episodes, ensemble_size].")
        losses: list[torch.Tensor] = []
        batch_size, sequence_length = actions.shape
        independent_features = detached_features.detach().reshape(
            batch_size * sequence_length, 1, detached_features.shape[-1]
        )
        independent_actions = actions.reshape(batch_size * sequence_length, 1)
        for index, model in enumerate(self.models):
            logits, _ = model(independent_features, independent_actions)
            logits = logits.reshape(
                batch_size, sequence_length, self.vocabulary_size
            )
            per_token = F.cross_entropy(
                logits.reshape(-1, self.vocabulary_size),
                response_tokens.reshape(-1),
                reduction="none",
            ).reshape(response_tokens.shape)
            episode_mask = bootstrap_mask[:, index].to(per_token.dtype)[:, None]
            denominator = episode_mask.expand_as(per_token).sum().clamp_min(1.0)
            losses.append((per_token * episode_mask).sum() / denominator)
        return torch.stack(losses).mean(), tuple(losses)

    def partner_conditioned_cross_entropy(
        self,
        detached_features: torch.Tensor,
        actions: torch.Tensor,
        response_tokens: torch.Tensor,
        partner_assignments: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Train each response member only on episodes from its partner."""

        if partner_assignments.shape != detached_features.shape[:1]:
            raise ValueError("partner_assignments must have shape [episodes].")
        if (
            actions.shape != response_tokens.shape
            or actions.shape != detached_features.shape[:2]
        ):
            raise ValueError("Response actions and tokens must match [episodes,time].")
        assignments = partner_assignments.to(torch.long)
        if bool(((assignments < 0) | (assignments >= self.ensemble_size)).any()):
            raise ValueError("partner_assignments contain an unknown response hypothesis.")
        losses: list[torch.Tensor] = []
        batch_size, sequence_length = actions.shape
        independent_features = detached_features.detach().reshape(
            batch_size * sequence_length, 1, detached_features.shape[-1]
        )
        independent_actions = actions.reshape(batch_size * sequence_length, 1)
        for index, model in enumerate(self.models):
            logits, _ = model(independent_features, independent_actions)
            logits = logits.reshape(batch_size, sequence_length, self.vocabulary_size)
            per_token = F.cross_entropy(
                logits.reshape(-1, self.vocabulary_size),
                response_tokens.reshape(-1),
                reduction="none",
            ).reshape(response_tokens.shape)
            episode_mask = (assignments == index).to(per_token.dtype)[:, None]
            denominator = episode_mask.expand_as(per_token).sum().clamp_min(1.0)
            losses.append((per_token * episode_mask).sum() / denominator)
        return torch.stack(losses).mean(), tuple(losses)


class PartnerConditionedValueModel(nn.Module):
    """Predict discounted raw continuation return for one partner hypothesis."""

    def __init__(
        self,
        trunk_feature_dim: int,
        *,
        n_actions: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Linear(
            int(trunk_feature_dim) + self.n_actions,
            self.hidden_dim,
        )
        self.value_head = nn.Linear(self.hidden_dim, self.n_actions)

    def forward(
        self,
        detached_features: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        if detached_features.requires_grad:
            raise ValueError(
                "Partner-conditioned values require explicitly detached features."
            )
        if (
            detached_features.ndim != 3
            or candidate_actions.shape != detached_features.shape[:2]
        ):
            raise ValueError("Value inputs must have shapes [B,T,D] and [B,T].")
        action_one_hot = F.one_hot(
            candidate_actions.to(torch.long), num_classes=self.n_actions
        ).to(detached_features.dtype)
        hidden = torch.tanh(
            self.input_projection(
                torch.cat((detached_features, action_one_hot), dim=-1)
            )
        )
        return self.value_head(hidden)


class PartnerConditionedValueEnsemble(nn.Module):
    """One discounted-raw-return model for each represented partner."""

    def __init__(
        self,
        trunk_feature_dim: int,
        *,
        n_actions: int = 6,
        hidden_dim: int = 64,
        ensemble_size: int = 5,
    ) -> None:
        super().__init__()
        if int(ensemble_size) < 2:
            raise ValueError("Partner-conditioned values need at least two hypotheses.")
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.ensemble_size = int(ensemble_size)
        self.models = nn.ModuleList(
            PartnerConditionedValueModel(
                trunk_feature_dim,
                n_actions=self.n_actions,
                hidden_dim=self.hidden_dim,
            )
            for _ in range(self.ensemble_size)
        )

    def values_for_candidates(
        self,
        detached_features: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return values shaped ``[batch,candidate,partner,next_action]``."""

        if detached_features.requires_grad or detached_features.ndim != 2:
            raise ValueError("Current value features must be detached [B,D] values.")
        if candidate_actions.ndim != 1:
            raise ValueError("candidate_actions must have shape [candidate].")
        batch_size = detached_features.shape[0]
        candidate_count = candidate_actions.numel()
        features = detached_features[:, None, :].expand(
            batch_size, candidate_count, detached_features.shape[-1]
        )
        actions = candidate_actions[None, :].expand(batch_size, candidate_count)
        predictions = [model(features, actions) for model in self.models]
        return torch.stack(predictions, dim=2)

    def partner_conditioned_mse(
        self,
        detached_features: torch.Tensor,
        candidate_actions: torch.Tensor,
        continuation_actions: torch.Tensor,
        raw_returns_to_go: torch.Tensor,
        partner_assignments: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Fit each value model only to complete episodes from its partner."""

        expected = detached_features.shape[:2]
        for name, value in (
            ("candidate_actions", candidate_actions),
            ("continuation_actions", continuation_actions),
            ("raw_returns_to_go", raw_returns_to_go),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must match [episodes,time].")
        if partner_assignments.shape != expected[:1]:
            raise ValueError("partner_assignments must have shape [episodes].")
        assignments = partner_assignments.to(torch.long)
        if bool(((assignments < 0) | (assignments >= self.ensemble_size)).any()):
            raise ValueError("partner_assignments contain an unknown value hypothesis.")
        losses: list[torch.Tensor] = []
        for index, model in enumerate(self.models):
            all_values = model(detached_features.detach(), candidate_actions)
            selected = all_values.gather(
                -1, continuation_actions.to(torch.long).unsqueeze(-1)
            ).squeeze(-1)
            squared_error = (selected - raw_returns_to_go.detach()).square()
            episode_mask = (assignments == index).to(squared_error.dtype)[:, None]
            denominator = episode_mask.expand_as(squared_error).sum().clamp_min(1.0)
            losses.append((squared_error * episode_mask).sum() / denominator)
        return torch.stack(losses).mean(), tuple(losses)

def mean_pairwise_jsd(probabilities: torch.Tensor) -> torch.Tensor:
    """Mean Jensen-Shannon divergence over every unordered model pair."""

    if probabilities.ndim < 2 or probabilities.shape[-2] < 2:
        raise ValueError("probabilities must contain at least two model distributions.")
    if not bool(torch.isfinite(probabilities).all()) or bool(
        (probabilities < 0.0).any()
    ):
        raise ValueError("Response probabilities must be finite and non-negative.")
    if bool((probabilities.sum(dim=-1) <= 0.0).any()):
        raise ValueError("Every response distribution needs positive probability mass.")
    probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    values: list[torch.Tensor] = []
    for left, right in itertools.combinations(range(probabilities.shape[-2]), 2):
        p = probabilities[..., left, :]
        q = probabilities[..., right, :]
        mixture = 0.5 * (p + q)
        values.append(
            0.5 * (p * (p.log() - mixture.log())).sum(dim=-1)
            + 0.5 * (q * (q.log() - mixture.log())).sum(dim=-1)
        )
    return torch.stack(values, dim=-1).mean(dim=-1)


def weighted_response_information(
    probabilities: torch.Tensor,
    partner_weights: torch.Tensor,
) -> torch.Tensor:
    """Information about the partner predicted from each candidate response.

    ``probabilities`` has shape ``[..., hypotheses, vocabulary]`` and
    ``partner_weights`` has shape ``[batch, hypotheses]``.  The result is the
    weighted generalized Jensen-Shannon divergence, equivalently the mutual
    information between the partner hypothesis and the predicted response.
    """

    if probabilities.ndim < 3 or partner_weights.ndim != 2:
        raise ValueError("Partner response information requires [...,K,V] and [B,K].")
    if probabilities.shape[0] != partner_weights.shape[0]:
        raise ValueError("Response probabilities and partner weights disagree on batch size.")
    if probabilities.shape[-2] != partner_weights.shape[-1]:
        raise ValueError("Partner weights must cover every response hypothesis.")
    if not bool(torch.isfinite(probabilities).all()) or bool(
        (probabilities < 0.0).any()
    ):
        raise ValueError("Response probabilities must be finite and non-negative.")
    if bool((probabilities.sum(dim=-1) <= 0.0).any()):
        raise ValueError("Every response distribution needs positive probability mass.")
    if not bool(torch.isfinite(partner_weights).all()) or bool(
        (partner_weights < 0.0).any()
    ):
        raise ValueError("Partner weights must be finite and non-negative.")
    tiny = torch.finfo(probabilities.dtype).tiny
    distributions = probabilities.clamp_min(tiny)
    distributions = distributions / distributions.sum(dim=-1, keepdim=True)
    weights = partner_weights.to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    weight_sum = weights.sum(dim=-1, keepdim=True)
    if bool((weight_sum <= 0.0).any()):
        raise ValueError("Every row needs positive total partner weight.")
    weights = weights / weight_sum
    while weights.ndim < distributions.ndim - 1:
        weights = weights.unsqueeze(-2)
    mixture = (weights.unsqueeze(-1) * distributions).sum(dim=-2)
    mixture_entropy = -(mixture * mixture.log()).sum(dim=-1)
    component_entropy = -(distributions * distributions.log()).sum(dim=-1)
    information = mixture_entropy - (weights * component_entropy).sum(dim=-1)
    return information.clamp_min(0.0)


def finite_prototype_two_action_values(
    *,
    response_probabilities: torch.Tensor,
    continuation_values: torch.Tensor,
    partner_belief: torch.Tensor,
    valid_candidates: torch.Tensor | None = None,
    valid_continuation_actions: torch.Tensor | None = None,
) -> FinitePrototypeTwoActionValues:
    """Compute a finite-prototype two-action proxy and its masked reference.

    ``response_probabilities`` has shape ``[B,P,K,Y]`` and contains the
    registered response model for candidate ``P`` and partner hypothesis
    ``K``. ``continuation_values`` has shape ``[B,P,K,A]`` and predicts raw
    return for each subsequent action.  The calculation is prospective, but it
    omits the sequential task transition and paired response-routing branches
    required by the proposal.  All candidates and the masked reference use the
    same discounted-raw-return head, avoiding a mixture with the shaped shared
    critic.  It is not a full-episode undiscounted R015 return.
    """

    if response_probabilities.ndim != 4 or continuation_values.ndim != 4:
        raise ValueError("Decision values require response [B,P,K,Y] and value [B,P,K,A].")
    if response_probabilities.shape[:3] != continuation_values.shape[:3]:
        raise ValueError("Response and continuation tensors must share B,P,K axes.")
    batch_size, candidate_count, hypothesis_count, _ = response_probabilities.shape
    if partner_belief.shape != (batch_size, hypothesis_count):
        raise ValueError("partner_belief must have shape [B,K].")
    if not bool(torch.isfinite(response_probabilities).all()) or bool(
        (response_probabilities < 0.0).any()
    ):
        raise ValueError("Surrogate response probabilities must be finite and non-negative.")
    if bool((response_probabilities.sum(dim=-1) <= 0.0).any()):
        raise ValueError(
            "Every surrogate response distribution needs positive probability mass."
        )
    if not bool(torch.isfinite(partner_belief).all()) or bool(
        (partner_belief < 0.0).any()
    ):
        raise ValueError("Surrogate partner beliefs must be finite and non-negative.")
    if not bool(torch.isfinite(continuation_values).all()):
        raise ValueError("Surrogate continuation values must be finite.")

    tiny = torch.finfo(response_probabilities.dtype).tiny
    probabilities = response_probabilities.clamp_min(tiny)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    belief = partner_belief.to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    normalizer = belief.sum(dim=-1, keepdim=True)
    if bool((normalizer <= 0.0).any()):
        raise ValueError("Every decision row needs positive partner belief mass.")
    belief = belief / normalizer

    if valid_continuation_actions is None:
        continuation_mask = torch.ones(
            batch_size,
            candidate_count,
            continuation_values.shape[-1],
            dtype=torch.bool,
            device=continuation_values.device,
        )
    else:
        continuation_mask = valid_continuation_actions.to(
            device=continuation_values.device, dtype=torch.bool
        )
        if continuation_mask.shape == (
            batch_size,
            continuation_values.shape[-1],
        ):
            continuation_mask = continuation_mask[:, None, :].expand(
                -1, candidate_count, -1
            )
        if continuation_mask.shape != (
            batch_size,
            candidate_count,
            continuation_values.shape[-1],
        ):
            raise ValueError("valid_continuation_actions must be [B,A] or [B,P,A].")
    if not bool(continuation_mask.any(dim=-1).all()):
        raise ValueError("Every candidate requires a valid continuation action.")

    masked_expectation = torch.einsum(
        "bk,bpka->bpa", belief, continuation_values
    ).masked_fill(~continuation_mask, -torch.inf)
    j_mask = masked_expectation.max(dim=-1).values
    response_action_values = torch.einsum(
        "bk,bpky,bpka->bpya",
        belief,
        probabilities,
        continuation_values,
    ).masked_fill(~continuation_mask[:, :, None, :], -torch.inf)
    j_use = response_action_values.max(dim=-1).values.sum(dim=-1)

    if valid_candidates is None:
        candidate_mask = torch.ones(
            batch_size,
            candidate_count,
            dtype=torch.bool,
            device=j_mask.device,
        )
    else:
        candidate_mask = valid_candidates.to(device=j_mask.device, dtype=torch.bool)
        if candidate_mask.shape != (batch_size, candidate_count):
            raise ValueError("valid_candidates must have shape [B,P].")
    if not bool(candidate_mask.any(dim=-1).all()):
        raise ValueError("Every decision row requires a valid candidate.")
    masked_j_mask = j_mask.masked_fill(~candidate_mask, -torch.inf)
    v_mask = masked_j_mask.max(dim=-1).values
    i_response = j_use - j_mask
    c_task = v_mask[:, None] - j_mask
    s_seq = j_use - v_mask[:, None]
    negative_infinity = torch.full_like(j_mask, -torch.inf)
    return FinitePrototypeTwoActionValues(
        j_use=torch.where(candidate_mask, j_use, negative_infinity),
        j_mask=masked_j_mask,
        v_mask=v_mask,
        i_response=torch.where(candidate_mask, i_response, negative_infinity),
        c_task=torch.where(candidate_mask, c_task, torch.full_like(c_task, torch.inf)),
        s_seq=torch.where(candidate_mask, s_seq, negative_infinity),
    )


def select_finite_prototype_two_action_surrogate(
    *,
    actor_logits: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    continuation_values: torch.Tensor,
    partner_belief: torch.Tensor,
    valid_actions: torch.Tensor,
    surrogate_score_threshold: float,
    max_probe_task_cost: float,
    probe_allowed: np.ndarray,
) -> ResponseProbeDecision:
    """Select a candidate with the non-confirmatory two-action surrogate."""

    if actor_logits.ndim != 2 or valid_actions.shape != actor_logits.shape:
        raise ValueError("actor_logits and valid_actions must have shape [B,A].")
    if candidate_probabilities.shape[:2] != actor_logits.shape:
        raise ValueError("Candidate responses must cover each primitive action.")
    if continuation_values.shape[:2] != actor_logits.shape:
        raise ValueError("Candidate values must cover each primitive action.")
    threshold = float(surrogate_score_threshold)
    task_cost_limit = float(max_probe_task_cost)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("surrogate_score_threshold must be finite and positive.")
    if not math.isfinite(task_cost_limit) or task_cost_limit < 0.0:
        raise ValueError("max_probe_task_cost must be finite and non-negative.")
    valid = valid_actions.to(dtype=torch.bool)
    values = finite_prototype_two_action_values(
        response_probabilities=candidate_probabilities,
        continuation_values=continuation_values,
        partner_belief=partner_belief,
        valid_candidates=valid,
        valid_continuation_actions=valid,
    )
    best_mask_candidate = values.j_mask.argmax(dim=-1)
    masked_baseline = best_mask_candidate
    candidate = values.s_seq.argmax(dim=-1)
    candidate_score = values.s_seq.gather(1, candidate[:, None]).squeeze(1)
    candidate_cost = values.c_task.gather(1, candidate[:, None]).squeeze(1)
    greedy = actor_logits.masked_fill(~valid, -torch.inf).argmax(dim=-1)
    equals_greedy = candidate == greedy
    equals_baseline = candidate == masked_baseline
    score_pass = candidate_score >= threshold
    cost_pass = candidate_cost <= task_cost_limit
    allowed = torch.as_tensor(
        np.asarray(probe_allowed, dtype=bool), device=actor_logits.device
    )
    if allowed.shape != candidate.shape:
        raise ValueError("probe_allowed must have shape [B].")
    is_probe = score_pass & cost_pass & allowed & ~equals_baseline
    selected = torch.where(is_probe, candidate, masked_baseline)
    to_numpy = lambda value: value.detach().cpu().numpy().copy()
    return ResponseProbeDecision(
        actions=to_numpy(selected).astype(np.int64),
        is_probe=to_numpy(is_probe).astype(bool),
        candidate_score=to_numpy(candidate_score).astype(np.float32),
        candidate_regret=to_numpy(candidate_cost).astype(np.float32),
        candidate_equals_greedy=to_numpy(equals_greedy).astype(bool),
        candidate_equals_baseline=to_numpy(equals_baseline).astype(bool),
        disagreement_pass=to_numpy(score_pass).astype(bool),
        regret_pass=to_numpy(cost_pass).astype(bool),
        budget_pass=to_numpy(allowed).astype(bool),
    )


def initial_partner_belief(
    batch_size: int,
    hypothesis_count: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a uniform partner belief for each independent environment."""

    if int(batch_size) <= 0 or int(hypothesis_count) < 2:
        raise ValueError("Partner beliefs require a positive batch and at least two hypotheses.")
    return torch.full(
        (int(batch_size), int(hypothesis_count)),
        1.0 / float(hypothesis_count),
        device=device,
        dtype=dtype,
    )


def update_partner_belief(
    prior: torch.Tensor,
    response_probabilities: torch.Tensor,
    response_tokens: torch.Tensor,
    *,
    probability_floor: float,
    observed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one observable response likelihood to each environment belief."""

    if prior.ndim != 2 or response_probabilities.ndim != 3:
        raise ValueError("Belief update requires [B,K] priors and [B,K,V] predictions.")
    if response_probabilities.shape[:2] != prior.shape:
        raise ValueError("Belief and response-hypothesis shapes do not match.")
    if response_tokens.shape != prior.shape[:1]:
        raise ValueError("response_tokens must have shape [B].")
    if not bool(torch.isfinite(prior).all()) or bool((prior < 0.0).any()):
        raise ValueError("Partner beliefs must be finite and non-negative.")
    if not bool(torch.isfinite(response_probabilities).all()) or bool(
        (response_probabilities < 0.0).any()
    ):
        raise ValueError("Response probabilities must be finite and non-negative.")
    prior_mass = prior.sum(dim=-1)
    response_mass = response_probabilities.sum(dim=-1)
    if bool((prior_mass <= 0.0).any()) or not bool(
        torch.allclose(
            prior_mass,
            torch.ones_like(prior_mass),
            rtol=1.0e-5,
            atol=1.0e-7,
        )
    ):
        raise ValueError("Every partner belief row must sum to one.")
    if bool((response_mass <= 0.0).any()) or not bool(
        torch.allclose(
            response_mass,
            torch.ones_like(response_mass),
            rtol=1.0e-5,
            atol=1.0e-7,
        )
    ):
        raise ValueError("Every response distribution must sum to one.")
    floor = float(probability_floor)
    if not math.isfinite(floor) or floor <= 0.0 or floor >= 1.0:
        raise ValueError("probability_floor must lie strictly between zero and one.")
    tokens = response_tokens.to(device=prior.device, dtype=torch.long)
    if bool(((tokens < 0) | (tokens >= response_probabilities.shape[-1])).any()):
        raise ValueError("response_tokens contain an unknown response value.")
    likelihood = response_probabilities.to(prior.device).gather(
        -1,
        tokens[:, None, None].expand(-1, prior.shape[1], 1),
    ).squeeze(-1).clamp_min(floor)
    posterior = prior * likelihood
    posterior_mass = posterior.sum(dim=-1, keepdim=True)
    if bool((posterior_mass <= 0.0).any()) or not bool(
        torch.isfinite(posterior_mass).all()
    ):
        raise ValueError("The registered response produced no finite posterior mass.")
    posterior = posterior / posterior_mass
    if observed is None:
        return posterior
    mask = observed.to(device=prior.device, dtype=torch.bool)
    if mask.shape != prior.shape[:1]:
        raise ValueError("observed must have shape [B].")
    return torch.where(mask[:, None], posterior, prior)


def calibrate_response_disagreement_threshold(
    candidate_scores: Sequence[float],
    candidate_regrets: Sequence[float],
    candidate_equals_greedy: Sequence[bool],
    *,
    max_probe_regret: float,
    quantile: float,
    candidate_equals_baseline: Sequence[bool] | None = None,
) -> tuple[float, int]:
    """Choose a fixed threshold from safe, non-greedy calibration candidates."""

    scores = np.asarray(candidate_scores, dtype=np.float64)
    regrets = np.asarray(candidate_regrets, dtype=np.float64)
    equals = np.asarray(candidate_equals_greedy, dtype=bool)
    equals_baseline = (
        equals
        if candidate_equals_baseline is None
        else np.asarray(candidate_equals_baseline, dtype=bool)
    )
    if scores.shape != regrets.shape or scores.shape != equals.shape or scores.ndim != 1:
        raise ValueError("Probe calibration arrays must be matching one-dimensional values.")
    if equals_baseline.shape != scores.shape:
        raise ValueError("Probe baseline-equality flags must match calibration scores.")
    regret_limit = float(max_probe_regret)
    requested_quantile = float(quantile)
    if not math.isfinite(regret_limit) or regret_limit < 0.0:
        raise ValueError("max_probe_regret must be finite and non-negative.")
    if not 0.0 < requested_quantile < 1.0:
        raise ValueError("Probe calibration quantile must lie in (0,1).")
    safe = (
        np.isfinite(scores)
        & np.isfinite(regrets)
        & (regrets <= regret_limit)
        & ~equals
        & ~equals_baseline
    )
    safe_scores = scores[safe]
    if safe_scores.size == 0:
        raise ValueError("No safe non-greedy candidates are available for calibration.")
    threshold = float(np.quantile(safe_scores, requested_quantile))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            "Partner-conditioned responses contain no positive calibratable information."
        )
    return threshold, int(safe_scores.size)


def episode_bootstrap_mask(
    episode_keys: Sequence[int],
    *,
    ensemble_size: int,
    bootstrap_seed: int,
    bootstrap_p: float,
) -> torch.Tensor:
    """Deterministic, distinct episode bootstrap inclusion masks."""

    records = [
        EpisodeBootstrapRecord.sample(
            episode_id=f"adaptation:{int(episode_key)}",
            n_heads=int(ensemble_size),
            bootstrap_p=float(bootstrap_p),
            manifest_seed=int(bootstrap_seed),
            ensure_nonempty=True,
        )
        for episode_key in episode_keys
    ]
    return torch.stack([record.as_tensor() for record in records], dim=0)


def select_response_probe(
    *,
    actor_logits: torch.Tensor,
    critic_q_values: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    valid_actions: torch.Tensor,
    response_disagreement_threshold: float,
    max_probe_regret: float,
    probe_allowed: np.ndarray,
    partner_weights: torch.Tensor | None = None,
    baseline_actions: torch.Tensor | np.ndarray | None = None,
) -> ResponseProbeDecision:
    """Apply every v3 trigger condition to single-step response candidates."""

    if actor_logits.ndim != 2 or critic_q_values.ndim != 3:
        raise ValueError("actor logits and critic Q values must be [B,A] and [B,K,A].")
    if candidate_probabilities.shape[:2] != actor_logits.shape:
        raise ValueError("Candidate response predictions must cover every action.")
    threshold = float(response_disagreement_threshold)
    regret_limit = float(max_probe_regret)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("response_disagreement_threshold must be finite and positive.")
    if not math.isfinite(regret_limit) or regret_limit < 0.0:
        raise ValueError("max_probe_regret must be finite and non-negative.")
    valid = valid_actions.bool()
    probe_candidates = tuple(
        ProbeCandidate(actions=(action,)) for action in range(actor_logits.shape[1])
    )
    greedy = actor_logits.masked_fill(~valid, -torch.inf).argmax(dim=-1)
    scores = (
        weighted_response_information(candidate_probabilities, partner_weights)
        if partner_weights is not None
        else mean_pairwise_jsd(candidate_probabilities)
    ).masked_fill(~valid, -torch.inf)
    candidate = scores.argmax(dim=-1)
    if len(probe_candidates) != scores.shape[1]:
        raise RuntimeError("Probe candidate interface and score columns diverged.")
    candidate_score = scores.gather(1, candidate[:, None]).squeeze(1)
    mean_q = critic_q_values.mean(dim=1).masked_fill(~valid, -torch.inf)
    greedy_q = mean_q.gather(1, greedy[:, None]).squeeze(1)
    candidate_q = mean_q.gather(1, candidate[:, None]).squeeze(1)
    regret = greedy_q - candidate_q
    equals = candidate == greedy
    baseline = (
        greedy
        if baseline_actions is None
        else torch.as_tensor(baseline_actions, device=actor_logits.device).to(torch.long)
    )
    if baseline.shape != greedy.shape:
        raise ValueError("baseline_actions must have shape [B].")
    equals_baseline = candidate == baseline
    disagreement_pass = candidate_score >= threshold
    regret_pass = regret <= regret_limit
    allowed = torch.as_tensor(
        np.asarray(probe_allowed, dtype=bool), device=actor_logits.device
    )
    if allowed.shape != greedy.shape:
        raise ValueError("probe_allowed must have shape [B].")
    is_probe = disagreement_pass & regret_pass & allowed & ~equals & ~equals_baseline
    selected = torch.where(is_probe, candidate, baseline)
    to_numpy = lambda value: value.detach().cpu().numpy().copy()
    return ResponseProbeDecision(
        actions=to_numpy(selected).astype(np.int64),
        is_probe=to_numpy(is_probe).astype(bool),
        candidate_score=to_numpy(candidate_score).astype(np.float32),
        candidate_regret=to_numpy(regret).astype(np.float32),
        candidate_equals_greedy=to_numpy(equals).astype(bool),
        candidate_equals_baseline=to_numpy(equals_baseline).astype(bool),
        disagreement_pass=to_numpy(disagreement_pass).astype(bool),
        regret_pass=to_numpy(regret_pass).astype(bool),
        budget_pass=to_numpy(allowed).astype(bool),
    )


def validate_response_probe_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate configurable numeric slots without silently supplying thresholds."""

    required = {
        "controller", "enabled", "response_disagreement_threshold",
        "advantage_disagreement_threshold",
        "max_probe_regret", "probe_budget_per_episode",
        "probe_window_environment_steps", "response_ensemble_size",
        "response_hidden_dim", "bootstrap_seed",
        "critic_bootstrap_p",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("Missing response probe field(s): " + ", ".join(missing))
    optional = {
        "response_model_mode",
        "belief_probability_floor",
        "response_bootstrap_p",
        "surrogate_score_threshold",
        "max_probe_task_cost",
        "response_route",
        "partner_value_hidden_dim",
    }
    unknown = sorted(set(payload) - required - optional)
    if unknown:
        raise ValueError("Unknown response probe field(s): " + ", ".join(unknown))
    config = dict(payload)
    mode = str(config.get("response_model_mode", "bootstrap_legacy"))
    if mode not in {"bootstrap_legacy", "partner_conditioned"}:
        raise ValueError("response_model_mode must be bootstrap_legacy or partner_conditioned.")
    config["response_model_mode"] = mode
    if mode == "bootstrap_legacy":
        if "response_bootstrap_p" not in config:
            raise ValueError("Legacy response models require response_bootstrap_p.")
        config["belief_probability_floor"] = None
    else:
        if "belief_probability_floor" not in config:
            raise ValueError("Partner-conditioned response models require belief_probability_floor.")
        floor = float(config["belief_probability_floor"])
        if not math.isfinite(floor) or not 0.0 < floor < 1.0:
            raise ValueError("belief_probability_floor must lie in (0,1).")
        config["belief_probability_floor"] = floor
    if config["controller"] not in {
        "finite_prototype_two_action_surrogate",
        "generic_response_information",
        "response_voi",
        "advantage_disagreement",
        "random",
        "off",
    }:
        raise ValueError("Unknown probe controller.")
    response_route = str(config.get("response_route", "use_registered_response"))
    if response_route not in {
        "use_registered_response",
        "mask_current_probe_response",
    }:
        raise ValueError(
            "response_route must use or mask the current probe's registered response."
        )
    config["response_route"] = response_route
    config["partner_value_hidden_dim"] = int(
        config.get("partner_value_hidden_dim", config["response_hidden_dim"])
    )
    if config["partner_value_hidden_dim"] <= 0:
        raise ValueError("partner_value_hidden_dim must be a positive integer.")
    if config["controller"] == "finite_prototype_two_action_surrogate":
        if mode != "partner_conditioned":
            raise ValueError(
                "The finite-prototype surrogate requires partner-conditioned response models."
            )
        for name in ("surrogate_score_threshold", "max_probe_task_cost"):
            if name not in config:
                raise ValueError(f"The finite-prototype surrogate requires {name}.")
        surrogate_threshold = float(config["surrogate_score_threshold"])
        task_cost_limit = float(config["max_probe_task_cost"])
        if not math.isfinite(surrogate_threshold) or surrogate_threshold <= 0.0:
            raise ValueError("surrogate_score_threshold must be positive.")
        if not math.isfinite(task_cost_limit) or task_cost_limit < 0.0:
            raise ValueError("max_probe_task_cost must be non-negative.")
        config["surrogate_score_threshold"] = surrogate_threshold
        config["max_probe_task_cost"] = task_cost_limit
    if not isinstance(config["enabled"], bool):
        raise ValueError("probe.enabled must be boolean.")
    for name in ("probe_budget_per_episode", "probe_window_environment_steps", "response_ensemble_size", "response_hidden_dim"):
        if isinstance(config[name], bool) or int(config[name]) <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    threshold = float(config["response_disagreement_threshold"])
    advantage_threshold = float(config["advantage_disagreement_threshold"])
    regret = float(config["max_probe_regret"])
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("response_disagreement_threshold must be positive.")
    if not math.isfinite(advantage_threshold) or advantage_threshold <= 0.0:
        raise ValueError("advantage_disagreement_threshold must be positive.")
    if not math.isfinite(regret) or regret < 0.0:
        raise ValueError("max_probe_regret must be non-negative.")
    probability_fields = ["critic_bootstrap_p"]
    if mode == "bootstrap_legacy":
        probability_fields.append("response_bootstrap_p")
    for name in probability_fields:
        probability = float(config[name])
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError(f"{name} must lie in (0,1].")
    return config
