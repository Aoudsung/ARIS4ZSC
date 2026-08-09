"""Two-step successor features under a probe and its response outcome.

Active VOI asks what a probe is worth.  Valuing it at the *current* state's
action values answers a different question: it prices only the posterior shift,
as if the two steps the probe costs were free and landed nowhere.  What the
probe actually buys is a decision at ``t+2``, in whatever state the probe and
the teammate's reply put the pair.

This module predicts that landing state's features,

    chi(x_t, b_t, a_t, y) -> (task features, instant partner features) at t+2,

so the belief-conditioned critic can be evaluated where the decision is really
made.  ``y`` is the compact delayed response outcome the VOI integral already
enumerates -- visibility, interface availability, interface change, and the
interface event class -- so the successor is conditioned on the same evidence
that produces the posterior it is paired with.

The head predicts a mean and reports an ensemble spread.  As with the critic
there is no trainable variance: a scale the objective can shrink for free is
not a measurement of anything.  Targets are the encoder's own features at
``t+2``, taken stop-gradiented -- the encoder belongs to PPO, and a successor
model that could reshape it would be optimising the representation to be
predictable rather than useful.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple

from .nn import init_linear, init_mlp, linear, mlp
from .observation import INTERFACE_EVENT_CLASSES

SUCCESSOR_OUTCOME_FEATURES = 3 + INTERFACE_EVENT_CLASSES
"""visibility, interface availability, interface change, then the event class.

The same decomposition ``compact_active_outcome_log_probabilities`` enumerates,
so an outcome index there and an outcome encoding here describe one event.
"""


class SuccessorFeature(NamedTuple):
    """Predicted ``t+2`` encoder features with ensemble disagreement."""

    task_mean: Any
    instant_mean: Any
    task_ensemble: Any
    instant_ensemble: Any

    def dispersion(self) -> Any:
        """Mean per-feature ensemble standard deviation across both blocks."""

        import jax.numpy as jnp

        if self.task_ensemble.shape[0] < 2:
            return jnp.zeros(self.task_mean.shape[:-1], dtype=jnp.float32)
        return 0.5 * (
            jnp.mean(jnp.std(self.task_ensemble, axis=0), axis=-1)
            + jnp.mean(jnp.std(self.instant_ensemble, axis=0), axis=-1)
        )


def outcome_encoding(
    visibility: Any,
    interface_available: Any,
    interface_changed: Any,
    interface_event: Any,
) -> Any:
    """Pack one delayed response outcome into ``[..., 34]``.

    Accepts either hard observations (for training) or probabilities (for the
    VOI enumeration, where each outcome is a known corner of the support).
    """

    import jax.numpy as jnp

    def scalar(value: Any) -> Any:
        return jnp.asarray(value, dtype=jnp.float32)[..., None]

    event = jnp.asarray(interface_event)
    if event.ndim == 0 or event.shape[-1] != INTERFACE_EVENT_CLASSES:
        event = jnp.eye(INTERFACE_EVENT_CLASSES, dtype=jnp.float32)[
            jnp.asarray(interface_event, dtype=jnp.int32)
        ]
    return jnp.concatenate(
        (
            scalar(visibility),
            scalar(interface_available),
            scalar(interface_changed),
            jnp.asarray(event, dtype=jnp.float32),
        ),
        axis=-1,
    )


@lru_cache(maxsize=1)
def enumerated_outcome_encodings() -> Any:
    """The ``[66, 34]`` encodings of the outcomes VOI integrates.

    Order matches ``compact_active_outcome_log_probabilities`` exactly: two
    unavailable outcomes (partner unseen / seen), two available-and-unchanged
    outcomes, then sixty-two changed outcomes -- for each visibility value, one
    per event class.  A mismatch here would silently pair each posterior with
    the wrong successor state, so the two orderings are asserted against each
    other in the test suite rather than trusted.

    Built with NumPy, not ``jnp``.  This is a compile-time constant, and a
    cached ``jnp`` array would be a *traced* value belonging to whichever jit
    trace happened to build it first -- which then escapes into every later
    trace and raises ``UnexpectedTracerError``, or worse, silently ties two
    graphs together.  The failure is call-order dependent, so it hides from a
    test suite that touches the function outside jit first.
    """

    import numpy

    classes = INTERFACE_EVENT_CLASSES
    # visibility varies slowest within each block, matching the VOI enumeration.
    visibility = numpy.concatenate(
        (
            numpy.asarray([0.0, 1.0]),
            numpy.asarray([0.0, 1.0]),
            numpy.repeat(numpy.asarray([0.0, 1.0]), classes),
        )
    ).astype(numpy.float32)
    available = numpy.concatenate(
        (numpy.zeros((2,)), numpy.ones((2,)), numpy.ones((2 * classes,)))
    )
    changed = numpy.concatenate(
        (numpy.zeros((4,)), numpy.ones((2 * classes,)))
    )
    events = numpy.concatenate(
        (
            numpy.zeros((4, classes)),
            numpy.tile(numpy.eye(classes), (2, 1)),
        ),
        axis=0,
    )
    return numpy.concatenate(
        (
            visibility[:, None],
            available[:, None],
            changed[:, None],
            events,
        ),
        axis=-1,
    ).astype(numpy.float32)



def expected_outcome_encoding(prediction: Any, belief: Any) -> Any:
    """The mean delayed-response outcome for each candidate probe, ``[..., P, 34]``.

    VOI integrates all sixty-six outcomes exactly for the *posterior*.  The
    landing state is a different matter: enumerating a successor per outcome
    means a ``[time, lane, probe, 66, component]`` tensor, which at the
    registered formal shape is 52 million critic rows and tens of gigabytes --
    the quantity is fine, the materialisation is not.

    Evaluating the successor at the expected outcome instead is a first-order
    approximation, ``E_y[Q(chi(y))] ~= Q(chi(E[y]))``.  It is stated here rather
    than hidden: what it assumes is that the landing state varies smoothly in
    the response, not that the response is unimportant -- the response still
    moves the posterior exactly, and the posterior is what VOI prices.
    """

    import jax.nn
    import jax.numpy as jnp

    visible = jax.nn.sigmoid(jnp.asarray(prediction.visibility_logit, jnp.float32))
    available = jax.nn.sigmoid(
        jnp.asarray(prediction.interface_availability_logit, jnp.float32)
    )
    changed = available * jax.nn.sigmoid(
        jnp.asarray(prediction.interface_change_logit, jnp.float32)
    )
    # The event head carries a component axis; marginalise it under the belief.
    event = jax.nn.softmax(
        jnp.asarray(prediction.interface_event_logits, jnp.float32), axis=-1
    )
    weights = jnp.asarray(belief, jnp.float32)
    weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1.0e-30)
    event = jnp.sum(event * weights[..., None, :, None], axis=-2)
    return jnp.concatenate(
        (
            visible[..., None],
            available[..., None],
            changed[..., None],
            event * changed[..., None],
        ),
        axis=-1,
    )


def init_successor_feature_params(
    key: Any,
    *,
    task_dim: int,
    instant_dim: int,
    behavior_dim: int,
    component_count: int,
    action_count: int,
    action_embedding_dim: int,
    hidden_dim: int,
    ensemble_size: int,
) -> dict[str, Any]:
    import jax

    if int(ensemble_size) < 1:
        raise ValueError("The successor feature model needs at least one member.")
    inputs = (
        int(task_dim)
        + int(instant_dim)
        + int(behavior_dim)
        + int(component_count)
        + int(action_embedding_dim)
        + SUCCESSOR_OUTCOME_FEATURES
    )
    keys = jax.random.split(key, 2 + 2 * int(ensemble_size))
    params: dict[str, Any] = {
        "action_embeddings": jax.random.normal(
            keys[0], (int(action_count), int(action_embedding_dim))
        )
        / max(int(action_embedding_dim), 1) ** 0.5,
        "trunk": init_mlp(keys[1], (inputs, int(hidden_dim), int(hidden_dim))),
    }
    for member in range(int(ensemble_size)):
        params[f"task_{member}"] = init_linear(
            keys[2 + 2 * member], int(hidden_dim), int(task_dim), scale=0.01
        )
        params[f"instant_{member}"] = init_linear(
            keys[3 + 2 * member], int(hidden_dim), int(instant_dim), scale=0.01
        )
    return params


def successor_feature_predict(
    params: dict[str, Any],
    task_features: Any,
    instant_partner: Any,
    behavior_features: Any,
    belief: Any,
    action: Any,
    outcome: Any,
) -> SuccessorFeature:
    """Predict the ``t+2`` features under probe ``action`` and outcome ``y``.

    The prediction is a *residual* on the current features.  Two Overcooked
    steps move the pair a short distance, so predicting the displacement rather
    than the absolute state means a freshly initialised head already answers
    "roughly where you are now", and the objective spends its capacity on the
    part the probe and the reply actually change.
    """

    import jax.numpy as jnp

    task = jnp.asarray(task_features, dtype=jnp.float32)
    instant = jnp.asarray(instant_partner, dtype=jnp.float32)
    embeddings = jnp.asarray(params["action_embeddings"], dtype=jnp.float32)
    index = jnp.asarray(action, dtype=jnp.int32)
    features = jnp.concatenate(
        (
            task,
            instant,
            jnp.asarray(behavior_features, dtype=jnp.float32),
            jnp.asarray(belief, dtype=jnp.float32),
            embeddings[index],
            jnp.asarray(outcome, dtype=jnp.float32),
        ),
        axis=-1,
    )
    hidden = mlp(params["trunk"], features, final_activation=True)
    members = sorted(
        int(name.split("_")[-1]) for name in params if name.startswith("task_")
    )
    task_ensemble = jnp.stack(
        [task + linear(params[f"task_{m}"], hidden) for m in members], axis=0
    )
    instant_ensemble = jnp.stack(
        [instant + linear(params[f"instant_{m}"], hidden) for m in members], axis=0
    )
    return SuccessorFeature(
        task_mean=jnp.mean(task_ensemble, axis=0),
        instant_mean=jnp.mean(instant_ensemble, axis=0),
        task_ensemble=task_ensemble,
        instant_ensemble=instant_ensemble,
    )


__all__ = [
    "SUCCESSOR_OUTCOME_FEATURES",
    "SuccessorFeature",
    "enumerated_outcome_encodings",
    "expected_outcome_encoding",
    "init_successor_feature_params",
    "outcome_encoding",
    "successor_feature_predict",
]
