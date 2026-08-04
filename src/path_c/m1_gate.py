"""Extended M1 gate (METHOD_SPEC §7).

The action-ranking Spearman check is executed for (i) the deployed posterior
path and (ii) each independently initialized and optimized bootstrap
value-signature readout (B=3).  Every path is evaluated against the held-out
CRN all-action signature of the same anchor state.  The diagnostic passes only
if all paths reach Spearman >= 0.8 on >= 90% of anchor states.

The readout is report-only; nothing here shapes rewards or gradients.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

M1_SPEARMAN_THRESHOLD = 0.8
M1_MINIMUM_ANCHOR_FRACTION = 0.9
BOOTSTRAP_MEMBER_COUNT = 3

_BOOTSTRAP_VALUE_MEMBER: Any | None = None


def bootstrap_history_features(
    policy_states: Any, current_observations: Any | None = None
) -> Any:
    """Legal full-history carries used by every M1 bootstrap member.

    The current legal observation is included when available; initialization
    and training/evaluation must use the same feature contract.
    """

    import jax.numpy as jnp

    protocol_probabilities = policy_states.protocol_carry
    capability = policy_states.capability_carry
    parts = [
        jnp.asarray(policy_states.task_carry, dtype=jnp.float32),
        jnp.asarray(capability.hidden, dtype=jnp.float32),
        jnp.asarray(capability.published, dtype=jnp.float32),
        jnp.asarray(protocol_probabilities, dtype=jnp.float32),
        jnp.asarray(policy_states.previous_observation, dtype=jnp.float32).reshape(
            jnp.asarray(policy_states.task_carry).shape[:-1] + (-1,)
        ),
        jnp.asarray(policy_states.previous_action, dtype=jnp.float32)[..., None],
        jnp.asarray(policy_states.episode_start, dtype=jnp.float32)[..., None],
    ]
    if current_observations is not None:
        current = jnp.asarray(current_observations, dtype=jnp.float32)
        parts.append(current.reshape(current.shape[:-3] + (-1,)))
    return jnp.concatenate(tuple(parts), axis=-1)


def bootstrap_value_member_class() -> Any:
    """Small independently optimized value-signature readout."""

    global _BOOTSTRAP_VALUE_MEMBER
    if _BOOTSTRAP_VALUE_MEMBER is not None:
        return _BOOTSTRAP_VALUE_MEMBER
    import flax.linen as nn
    import jax.numpy as jnp

    class BootstrapValueMember(nn.Module):
        action_count: int
        hidden_dim: int = 64

        @nn.compact
        def __call__(self, history_features: Any) -> Any:
            hidden = nn.tanh(nn.Dense(self.hidden_dim, name="history_hidden")(history_features))
            return nn.Dense(
                self.action_count,
                kernel_init=nn.initializers.orthogonal(0.01),
                name="action_signature",
            )(hidden)

    _BOOTSTRAP_VALUE_MEMBER = BootstrapValueMember
    return BootstrapValueMember


def initialize_bootstrap_value_ensemble(
    *,
    key: Any,
    example_policy_state: Any,
    example_observation: Any,
    action_count: int = 6,
) -> tuple[Any, Any, Any, Any]:
    """Initialize three models with independent Adam states and counters."""

    import jax
    import jax.numpy as jnp
    import optax

    Member = bootstrap_value_member_class()
    models = tuple(Member(action_count=int(action_count)) for _ in range(BOOTSTRAP_MEMBER_COUNT))
    features = bootstrap_history_features(example_policy_state, example_observation)
    keys = jax.random.split(key, BOOTSTRAP_MEMBER_COUNT)
    params = tuple(
        member.init(member_key, features)["params"]
        for member, member_key in zip(models, keys, strict=True)
    )
    optimizer = optax.adam(1.0e-3)
    optimizer_states = tuple(optimizer.init(value) for value in params)
    counters = jnp.zeros((BOOTSTRAP_MEMBER_COUNT,), dtype=jnp.int32)
    return models, params, optimizer_states, counters


def train_bootstrap_value_ensemble(
    *,
    models: Any,
    params: Any,
    optimizer_states: Any,
    sampling_counters: Any,
    anchors: Any,
    key: Any,
    steps: int = 32,
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    """Train independent members on bootstrap-resampled legal-history carries."""

    import jax
    import jax.numpy as jnp
    import optax

    from .decision_geometry import centered_action_values

    features = bootstrap_history_features(
        anchors.policy_states, anchors.observations
    )
    targets = jax.lax.stop_gradient(
        centered_action_values(anchors.fit_returns_by_action)
    )
    valid = jnp.any(jnp.asarray(anchors.action_mask, dtype=jnp.bool_), axis=-1)
    valid_indexes = jnp.where(valid, size=valid.shape[0], fill_value=0)[0]
    valid_count = jnp.maximum(jnp.sum(valid.astype(jnp.int32)), 1)
    optimizer = optax.adam(1.0e-3)
    next_params = []
    next_states = []
    losses = []
    for member_index, (member, member_params, optimizer_state) in enumerate(
        zip(models, params, optimizer_states, strict=True)
    ):
        current_params = member_params
        current_state = optimizer_state
        counter = jnp.asarray(sampling_counters[member_index], dtype=jnp.int32)
        last_loss = jnp.asarray(0.0, dtype=jnp.float32)
        for local_step in range(int(steps)):
            draw_key = jax.random.fold_in(
                jax.random.fold_in(key, member_index), counter + local_step
            )
            positions = jax.random.randint(
                draw_key, (features.shape[0],), 0, valid_count
            )
            indexes = valid_indexes[positions]

            def objective(candidate: Any) -> Any:
                prediction = member.apply({"params": candidate}, features[indexes])
                return jnp.mean(jnp.square(prediction - targets[indexes]))

            last_loss, gradients = jax.value_and_grad(objective)(current_params)
            updates, current_state = optimizer.update(
                gradients, current_state, current_params
            )
            current_params = optax.apply_updates(current_params, updates)
        next_params.append(current_params)
        next_states.append(current_state)
        losses.append(last_loss)
    return (
        tuple(next_params),
        tuple(next_states),
        jnp.asarray(sampling_counters, dtype=jnp.int32) + int(steps),
        {"bootstrap_training_loss": jnp.stack(losses)},
    )


class M1GateResult(NamedTuple):
    passed: Any
    path_passing_fractions: Any
    path_mean_spearman: Any
    path_top_action_agreement: Any
    path_mean_value_regret: Any
    spearman_threshold: Any
    minimum_anchor_fraction: Any


def _average_tie_ranks(values: Any) -> Any:
    """Zero-based average ranks, including exact ties."""

    import jax.numpy as jnp

    array = jnp.asarray(values, dtype=jnp.float32)
    lower = jnp.sum(
        (array[..., None, :] < array[..., :, None]).astype(jnp.float32),
        axis=-1,
    )
    equal = jnp.sum(
        (array[..., None, :] == array[..., :, None]).astype(jnp.float32),
        axis=-1,
    )
    return lower + 0.5 * (equal - 1.0)


def spearman_rank_correlation(predicted: Any, reference: Any) -> Any:
    """Tie-aware Spearman rho using average ranks."""

    import jax.numpy as jnp

    left = jnp.asarray(predicted, dtype=jnp.float32)
    right = jnp.asarray(reference, dtype=jnp.float32)
    if left.shape != right.shape:
        raise ValueError("Spearman inputs must share shape.")
    left_ranks = _average_tie_ranks(left)
    right_ranks = _average_tie_ranks(right)
    left_centered = left_ranks - jnp.mean(left_ranks, axis=-1, keepdims=True)
    right_centered = right_ranks - jnp.mean(right_ranks, axis=-1, keepdims=True)
    numerator = jnp.sum(left_centered * right_centered, axis=-1)
    denominator = jnp.sqrt(
        jnp.sum(jnp.square(left_centered), axis=-1)
        * jnp.sum(jnp.square(right_centered), axis=-1)
    )
    return jnp.where(denominator > 1.0e-12, numerator / denominator, 0.0)


def top_action_agreement(predicted: Any, reference: Any) -> Any:
    import jax.numpy as jnp

    left = jnp.asarray(predicted, dtype=jnp.float32)
    right = jnp.asarray(reference, dtype=jnp.float32)
    if left.shape != right.shape:
        raise ValueError("Top-action inputs must share shape.")
    return (jnp.argmax(left, axis=-1) == jnp.argmax(right, axis=-1)).astype(
        jnp.float32
    )


def empirical_value_regret(predicted: Any, reference: Any) -> Any:
    """G(a*) - G(argmax predicted), the action-level empirical regret."""

    import jax.numpy as jnp

    left = jnp.asarray(predicted, dtype=jnp.float32)
    right = jnp.asarray(reference, dtype=jnp.float32)
    if left.shape != right.shape:
        raise ValueError("Value-regret inputs must share shape.")
    selected = jnp.take_along_axis(
        right, jnp.argmax(left, axis=-1)[..., None], axis=-1
    )[..., 0]
    return jnp.max(right, axis=-1) - selected


def _path_passing_fraction(
    spearman_values: Any,
    *,
    threshold: float,
    valid_mask: Any | None = None,
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(spearman_values, dtype=jnp.float32)
    mask = (
        jnp.asarray(valid_mask, dtype=jnp.float32)
        if valid_mask is not None
        else jnp.ones(values.shape, dtype=jnp.float32)
    )
    passing = ((values >= float(threshold)) * mask).astype(jnp.float32)
    total = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(passing) / total, jnp.sum(values * mask) / total


def m1_anchor_gate(
    predicted_action_values: Any,
    *,
    crn_signatures: Any,
    valid_mask: Any | None = None,
    spearman_threshold: float = M1_SPEARMAN_THRESHOLD,
    minimum_anchor_fraction: float = M1_MINIMUM_ANCHOR_FRACTION,
) -> tuple[Any, Any]:
    """Single-path gate: per-anchor-state Spearman of the action ranking
    between the context's Q values and the shared CRN signature."""

    rho = spearman_rank_correlation(predicted_action_values, crn_signatures)
    fraction, mean_rho = _path_passing_fraction(
        rho, threshold=spearman_threshold, valid_mask=valid_mask
    )
    return fraction >= float(minimum_anchor_fraction), fraction


def evaluate_extended_m1_gate(
    *,
    posterior_mean_action_values: Any,
    member_action_values: Any,
    crn_signatures: Any,
    valid_mask: Any | None = None,
    spearman_threshold: float = M1_SPEARMAN_THRESHOLD,
    minimum_anchor_fraction: float = M1_MINIMUM_ANCHOR_FRACTION,
) -> M1GateResult:
    """§6 extended gate: the posterior-mean path and *each* bootstrap member
    path must independently reach the passing fraction; pooling across
    members is forbidden."""

    import jax.numpy as jnp

    signatures = jnp.asarray(crn_signatures, dtype=jnp.float32)
    posterior_rho = spearman_rank_correlation(
        posterior_mean_action_values, signatures
    )
    posterior_fraction, posterior_mean_rho = _path_passing_fraction(
        posterior_rho, threshold=spearman_threshold, valid_mask=valid_mask
    )
    posterior_pass = posterior_fraction >= float(minimum_anchor_fraction)

    member_values = jnp.asarray(member_action_values, dtype=jnp.float32)
    member_signatures = jnp.broadcast_to(signatures[None], member_values.shape)
    member_rho = spearman_rank_correlation(member_values, member_signatures)
    if valid_mask is not None:
        member_mask = jnp.broadcast_to(
            jnp.asarray(valid_mask, dtype=jnp.float32)[None], member_values.shape[:-1]
        )
    else:
        member_mask = jnp.ones(member_values.shape[:-1], dtype=jnp.float32)
    passing = (member_rho >= float(spearman_threshold)).astype(jnp.float32)
    totals = jnp.maximum(jnp.sum(member_mask, axis=-1), 1.0)
    member_fractions = jnp.sum(passing * member_mask, axis=-1) / totals
    member_mean_rho = jnp.sum(member_rho * member_mask, axis=-1) / totals
    member_pass = jnp.all(member_fractions >= float(minimum_anchor_fraction))
    posterior_metric_mask = (
        jnp.ones(signatures.shape[:-1], dtype=jnp.float32)
        if valid_mask is None
        else jnp.asarray(valid_mask, dtype=jnp.float32)
    )
    member_metric_mask = jnp.broadcast_to(
        posterior_metric_mask[None], member_values.shape[:-1]
    )

    def masked_mean(values: Any, mask: Any, *, axis: Any = None) -> Any:
        weights = jnp.asarray(mask, dtype=jnp.float32)
        return jnp.sum(jnp.asarray(values) * weights, axis=axis) / jnp.maximum(
            jnp.sum(weights, axis=axis), 1.0
        )

    posterior_top = masked_mean(
        top_action_agreement(posterior_mean_action_values, signatures),
        posterior_metric_mask,
    )
    member_top = masked_mean(
        top_action_agreement(member_values, member_signatures),
        member_metric_mask,
        axis=-1,
    )
    posterior_regret = masked_mean(
        empirical_value_regret(posterior_mean_action_values, signatures),
        posterior_metric_mask,
    )
    member_regret = masked_mean(
        empirical_value_regret(member_values, member_signatures),
        member_metric_mask,
        axis=-1,
    )
    return M1GateResult(
        passed=jnp.asarray(posterior_pass) & jnp.asarray(member_pass),
        path_passing_fractions=jnp.concatenate(
            (jnp.asarray(posterior_fraction)[None], member_fractions)
        ),
        path_mean_spearman=jnp.concatenate(
            (jnp.asarray(posterior_mean_rho)[None], member_mean_rho)
        ),
        path_top_action_agreement=jnp.concatenate(
            (jnp.asarray(posterior_top)[None], member_top)
        ),
        path_mean_value_regret=jnp.concatenate(
            (jnp.asarray(posterior_regret)[None], member_regret)
        ),
        spearman_threshold=jnp.asarray(spearman_threshold, dtype=jnp.float32),
        minimum_anchor_fraction=jnp.asarray(
            minimum_anchor_fraction, dtype=jnp.float32
        ),
    )


def evaluate_m1_gate_on_anchor_batch(
    *,
    model: Any,
    params: Any,
    bootstrap_models: Any,
    bootstrap_params: Any,
    anchors: Any,
    spearman_threshold: float = M1_SPEARMAN_THRESHOLD,
    minimum_anchor_fraction: float = M1_MINIMUM_ANCHOR_FRACTION,
) -> M1GateResult:
    """§6 gate over one collected anchor batch.

    Path (i) uses the deployable model.  Path (ii) uses independently trained
    bootstrap value members over the anchor's complete legal recurrent carry.
    Fit replicas train those members; independent evaluation replicas supply
    the M1 reference.
    """

    import jax.numpy as jnp

    from .decision_geometry import centered_action_values

    count = int(jnp.asarray(anchors.anchor_ids).shape[0])
    dropped = jnp.zeros((count,), dtype=jnp.bool_)
    _, posterior = model.apply(
        {"params": params},
        anchors.policy_states,
        anchors.observations,
        dropped,
        method=model.step,
    )
    posterior_summary = jnp.concatenate(
        (posterior.capability, posterior.protocol_embedding), axis=-1
    )
    posterior_values = model.apply(
        {"params": params},
        posterior.task_features,
        posterior_summary,
        method=model.action_values_from_features_and_context,
    )
    if anchors.evaluation_returns_by_action is None:
        raise ValueError("M1 requires independent evaluation replicas.")
    evaluation_count = jnp.asarray(anchors.evaluation_replica_count)
    if bool(jnp.any(evaluation_count <= 0)):
        raise ValueError("M1 evaluation replica count must be positive.")
    crn_signatures = centered_action_values(anchors.evaluation_returns_by_action)
    valid_mask = jnp.any(jnp.asarray(anchors.action_mask), axis=-1)

    history_features = bootstrap_history_features(
        anchors.policy_states, anchors.observations
    )
    member_values = jnp.stack(
        [
            member.apply({"params": member_params}, history_features)
            for member, member_params in zip(
                bootstrap_models, bootstrap_params, strict=True
            )
        ],
        axis=0,
    )
    return evaluate_extended_m1_gate(
        posterior_mean_action_values=posterior_values,
        member_action_values=member_values,
        crn_signatures=crn_signatures,
        valid_mask=valid_mask,
        spearman_threshold=spearman_threshold,
        minimum_anchor_fraction=minimum_anchor_fraction,
    )


def m1_gate_report(result: M1GateResult) -> Mapping[str, Any]:
    """JSON-ready gate readout for run ledgers."""

    import numpy as np

    return {
        "m1_gate_passed": bool(np.asarray(result.passed)),
        "m1_path_passing_fractions": [
            float(value) for value in np.asarray(result.path_passing_fractions)
        ],
        "m1_path_mean_spearman": [
            float(value) for value in np.asarray(result.path_mean_spearman)
        ],
        "m1_path_top_action_agreement": [
            float(value)
            for value in np.asarray(result.path_top_action_agreement)
        ],
        "m1_path_mean_value_regret": [
            float(value) for value in np.asarray(result.path_mean_value_regret)
        ],
        "m1_spearman_threshold": float(np.asarray(result.spearman_threshold)),
        "m1_minimum_anchor_fraction": float(
            np.asarray(result.minimum_anchor_fraction)
        ),
    }


__all__ = [
    "M1_MINIMUM_ANCHOR_FRACTION",
    "M1_SPEARMAN_THRESHOLD",
    "M1GateResult",
    "bootstrap_history_features",
    "bootstrap_value_member_class",
    "evaluate_extended_m1_gate",
    "evaluate_m1_gate_on_anchor_batch",
    "initialize_bootstrap_value_ensemble",
    "m1_anchor_gate",
    "m1_gate_report",
    "empirical_value_regret",
    "spearman_rank_correlation",
    "top_action_agreement",
    "train_bootstrap_value_ensemble",
]
