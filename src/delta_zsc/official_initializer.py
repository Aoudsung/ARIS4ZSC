"""Initialise DELTA's base policy from a trained Official checkpoint.

DELTA's ``base_params`` own task competence and nothing else, and they are
trained by the same PPO the Official baselines use on the same observation.
Starting them from noise means the first several million steps are spent
rediscovering how to cook -- and, worse for this method, the anchor measurement
during that period asks "does the action choice matter here?" of a policy for
which it usually does not.  A random policy's continuations re-merge, the CRN
contrasts collapse to zero, and the decision channel is fitted to noise before
the task is learned.

Every Official parameter maps onto exactly one DELTA parameter, because the
encoder, recurrent cell and heads are now the pinned Official ones.  The only
DELTA inputs with no Official counterpart are the instantaneous partner
encoding and the response-only posterior; both enter the actor and value trunks
as additional input rows, and those rows are initialised to **zero**.

The *weights* are an exact copy.  The *inputs* are not, and the difference
matters: every variant except ``history_rnn`` feeds the task encoder a
partner-masked frame -- ``task_only_observation`` zeroes the teammate's
channels, because the teammate may reach the policy only through the legal
response channel -- whereas the Official checkpoint was trained on the full
frame.  A transplanted encoder therefore runs slightly off the distribution it
was fitted to.  It still carries most of its competence across (measured on a
development run: shaped return 24.8 at the first update against the partner
mixture, against -0.7 from random initialisation), but this is not the identity
and should not be described as one.

The Official convex combination is ``h' = (1-z) n + z h`` and DELTA's gate is
written the same way, so the recurrent transplant is a direct copy with no sign
correction.
"""

from __future__ import annotations

from typing import Any

OFFICIAL_CONV_KEYS = (
    "Conv_0",
    "Conv_1",
    "Conv_2",
    "Conv_3",
    "Conv_4",
    "Conv_5",
)


def load_official_parameters(checkpoint_path: str) -> dict[str, Any]:
    """Restore the ``params`` subtree of an Official PPO checkpoint."""

    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        checkpointer.metadata(str(checkpoint_path))
    )
    restored = checkpointer.restore(
        str(checkpoint_path), restore_args=restore_args
    )
    payload = restored["params"]
    # Official checkpoints are written either as ``params`` or as the nested
    # ``params/params`` Flax emits; both appear across the baseline families.
    while isinstance(payload, dict) and set(payload) == {"params"}:
        payload = payload["params"]
    return payload


def transplant_official_base_params(
    base_params: dict[str, Any], official: dict[str, Any]
) -> dict[str, Any]:
    """Overwrite DELTA's task-competence parameters with Official ones.

    Raises if any shape disagrees.  A silent partial transplant would produce a
    policy that is neither Official nor freshly initialised, and nothing
    downstream could tell which.
    """

    import jax.numpy as jnp

    cnn = official["CNN_0"]
    gru = official["ScannedRNN_0"]["GRUCell_1"]
    norm = official["LayerNorm_0"]

    def check(name: str, value: Any, expected: Any) -> Any:
        array = jnp.asarray(value, dtype=jnp.float32)
        if array.shape != jnp.asarray(expected).shape:
            raise ValueError(
                f"Official parameter {name} has shape {array.shape}, "
                f"but DELTA expects {jnp.asarray(expected).shape}."
            )
        return array

    convolutions = tuple(
        {
            "kernel": check(
                f"CNN_0/{key}/kernel",
                cnn[key]["kernel"],
                base_params["task_conv"][index]["kernel"],
            ),
            "bias": check(
                f"CNN_0/{key}/bias",
                cnn[key]["bias"],
                base_params["task_conv"][index]["bias"],
            ),
        }
        for index, key in enumerate(OFFICIAL_CONV_KEYS)
    )

    def pad_rows(official_kernel: Any, target: Any, name: str) -> Any:
        """Place an Official kernel in the task rows; zero the new inputs.

        DELTA's trunks read ``[task features | instant partner | posterior]``.
        Only the first block existed in the Official policy, so the remaining
        rows start at zero and the transplanted actor is exactly the Official
        actor until PPO moves them.
        """

        source = jnp.asarray(official_kernel, dtype=jnp.float32)
        destination = jnp.zeros_like(jnp.asarray(target, dtype=jnp.float32))
        if source.shape[0] > destination.shape[0] or (
            source.shape[1] != destination.shape[1]
        ):
            raise ValueError(
                f"Official kernel {name} of shape {source.shape} does not fit "
                f"DELTA's {destination.shape}."
            )
        return destination.at[: source.shape[0]].set(source)

    return {
        **base_params,
        "task_conv": convolutions,
        "task_dense": {
            "kernel": check(
                "CNN_0/Dense_0/kernel",
                cnn["Dense_0"]["kernel"],
                base_params["task_dense"]["kernel"],
            ),
            "bias": check(
                "CNN_0/Dense_0/bias",
                cnn["Dense_0"]["bias"],
                base_params["task_dense"]["bias"],
            ),
        },
        "task_norm": {
            "scale": check(
                "LayerNorm_0/scale", norm["scale"], base_params["task_norm"]["scale"]
            ),
            "bias": check(
                "LayerNorm_0/bias", norm["bias"], base_params["task_norm"]["bias"]
            ),
        },
        "task_gru": {
            "input_reset": {
                "kernel": jnp.asarray(gru["ir"]["kernel"], dtype=jnp.float32),
                "bias": jnp.asarray(gru["ir"]["bias"], dtype=jnp.float32),
            },
            "input_update": {
                "kernel": jnp.asarray(gru["iz"]["kernel"], dtype=jnp.float32),
                "bias": jnp.asarray(gru["iz"]["bias"], dtype=jnp.float32),
            },
            "input_candidate": {
                "kernel": jnp.asarray(gru["in"]["kernel"], dtype=jnp.float32),
                "bias": jnp.asarray(gru["in"]["bias"], dtype=jnp.float32),
            },
            # Flax gives the two gate hidden projections no bias of their own.
            "hidden_reset": {
                "kernel": jnp.asarray(gru["hr"]["kernel"], dtype=jnp.float32),
                "bias": jnp.zeros_like(
                    jnp.asarray(base_params["task_gru"]["hidden_reset"]["bias"])
                ),
            },
            "hidden_update": {
                "kernel": jnp.asarray(gru["hz"]["kernel"], dtype=jnp.float32),
                "bias": jnp.zeros_like(
                    jnp.asarray(base_params["task_gru"]["hidden_update"]["bias"])
                ),
            },
            "hidden_candidate": {
                "kernel": jnp.asarray(gru["hn"]["kernel"], dtype=jnp.float32),
                "bias": jnp.asarray(gru["hn"]["bias"], dtype=jnp.float32),
            },
        },
        "actor_trunk": {
            "kernel": pad_rows(
                official["Dense_0"]["kernel"],
                base_params["actor_trunk"]["kernel"],
                "Dense_0",
            ),
            "bias": check(
                "Dense_0/bias",
                official["Dense_0"]["bias"],
                base_params["actor_trunk"]["bias"],
            ),
        },
        "actor": {
            "kernel": check(
                "Dense_1/kernel",
                official["Dense_1"]["kernel"],
                base_params["actor"]["kernel"],
            ),
            "bias": check(
                "Dense_1/bias",
                official["Dense_1"]["bias"],
                base_params["actor"]["bias"],
            ),
        },
        "value_trunk": {
            "kernel": pad_rows(
                official["Dense_2"]["kernel"],
                base_params["value_trunk"]["kernel"],
                "Dense_2",
            ),
            "bias": check(
                "Dense_2/bias",
                official["Dense_2"]["bias"],
                base_params["value_trunk"]["bias"],
            ),
        },
        "value": {
            "kernel": check(
                "Dense_3/kernel",
                official["Dense_3"]["kernel"],
                base_params["value"]["kernel"],
            ),
            "bias": check(
                "Dense_3/bias",
                official["Dense_3"]["bias"],
                base_params["value"]["bias"],
            ),
        },
    }


def initialize_base_from_official(
    base_params: dict[str, Any], checkpoint_path: str
) -> dict[str, Any]:
    return transplant_official_base_params(
        base_params, load_official_parameters(checkpoint_path)
    )


__all__ = [
    "OFFICIAL_CONV_KEYS",
    "initialize_base_from_official",
    "load_official_parameters",
    "transplant_official_base_params",
]
