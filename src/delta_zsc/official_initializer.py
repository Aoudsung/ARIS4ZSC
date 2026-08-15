"""Bind DELTA's immutable reference actor to an Official SP checkpoint."""

from __future__ import annotations

from pathlib import Path
import re
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
    import orbax.checkpoint as ocp

    checkpoint = str(Path(checkpoint_path).resolve())
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        checkpointer.metadata(checkpoint)
    )
    restored = checkpointer.restore(checkpoint, restore_args=restore_args)
    payload = restored["params"]
    while isinstance(payload, dict) and set(payload) == {"params"}:
        payload = payload["params"]
    return payload


def _official_reference(
    expected: dict[str, Any], official: dict[str, Any]
) -> dict[str, Any]:
    import jax.numpy as jnp

    cnn = official["CNN_0"]
    gru = official["ScannedRNN_0"]["GRUCell_1"]
    norm = official["LayerNorm_0"]

    def check(name: str, value: Any, target: Any) -> Any:
        array = jnp.asarray(value, dtype=jnp.float32)
        if array.shape != jnp.asarray(target).shape:
            raise ValueError(
                f"Official parameter {name} has shape {array.shape}, "
                f"but DELTA expects {jnp.asarray(target).shape}."
            )
        return array

    return {
        "task_conv": tuple(
            {
                "kernel": check(
                    f"CNN_0/{key}/kernel",
                    cnn[key]["kernel"],
                    expected["task_conv"][index]["kernel"],
                ),
                "bias": check(
                    f"CNN_0/{key}/bias",
                    cnn[key]["bias"],
                    expected["task_conv"][index]["bias"],
                ),
            }
            for index, key in enumerate(OFFICIAL_CONV_KEYS)
        ),
        "task_dense": {
            "kernel": check(
                "CNN_0/Dense_0/kernel",
                cnn["Dense_0"]["kernel"],
                expected["task_dense"]["kernel"],
            ),
            "bias": check(
                "CNN_0/Dense_0/bias",
                cnn["Dense_0"]["bias"],
                expected["task_dense"]["bias"],
            ),
        },
        "task_norm": {
            "scale": check(
                "LayerNorm_0/scale", norm["scale"], expected["task_norm"]["scale"]
            ),
            "bias": check(
                "LayerNorm_0/bias", norm["bias"], expected["task_norm"]["bias"]
            ),
        },
        "task_gru": {
            "input_reset": {
                "kernel": check(
                    "GRUCell_1/ir/kernel",
                    gru["ir"]["kernel"],
                    expected["task_gru"]["input_reset"]["kernel"],
                ),
                "bias": check(
                    "GRUCell_1/ir/bias",
                    gru["ir"]["bias"],
                    expected["task_gru"]["input_reset"]["bias"],
                ),
            },
            "input_update": {
                "kernel": check(
                    "GRUCell_1/iz/kernel",
                    gru["iz"]["kernel"],
                    expected["task_gru"]["input_update"]["kernel"],
                ),
                "bias": check(
                    "GRUCell_1/iz/bias",
                    gru["iz"]["bias"],
                    expected["task_gru"]["input_update"]["bias"],
                ),
            },
            "input_candidate": {
                "kernel": check(
                    "GRUCell_1/in/kernel",
                    gru["in"]["kernel"],
                    expected["task_gru"]["input_candidate"]["kernel"],
                ),
                "bias": check(
                    "GRUCell_1/in/bias",
                    gru["in"]["bias"],
                    expected["task_gru"]["input_candidate"]["bias"],
                ),
            },
            "hidden_reset": {
                "kernel": check(
                    "GRUCell_1/hr/kernel",
                    gru["hr"]["kernel"],
                    expected["task_gru"]["hidden_reset"]["kernel"],
                ),
                "bias": jnp.zeros_like(expected["task_gru"]["hidden_reset"]["bias"]),
            },
            "hidden_update": {
                "kernel": check(
                    "GRUCell_1/hz/kernel",
                    gru["hz"]["kernel"],
                    expected["task_gru"]["hidden_update"]["kernel"],
                ),
                "bias": jnp.zeros_like(expected["task_gru"]["hidden_update"]["bias"]),
            },
            "hidden_candidate": {
                "kernel": check(
                    "GRUCell_1/hn/kernel",
                    gru["hn"]["kernel"],
                    expected["task_gru"]["hidden_candidate"]["kernel"],
                ),
                "bias": check(
                    "GRUCell_1/hn/bias",
                    gru["hn"]["bias"],
                    expected["task_gru"]["hidden_candidate"]["bias"],
                ),
            },
        },
        "actor_trunk": {
            "kernel": check(
                "Dense_0/kernel",
                official["Dense_0"]["kernel"],
                expected["actor_trunk"]["kernel"],
            ),
            "bias": check(
                "Dense_0/bias",
                official["Dense_0"]["bias"],
                expected["actor_trunk"]["bias"],
            ),
        },
        "actor": {
            "kernel": check(
                "Dense_1/kernel",
                official["Dense_1"]["kernel"],
                expected["actor"]["kernel"],
            ),
            "bias": check(
                "Dense_1/bias", official["Dense_1"]["bias"], expected["actor"]["bias"]
            ),
        },
    }


def transplant_official_base_params(
    base_params: dict[str, Any], official: dict[str, Any]
) -> dict[str, Any]:
    """Replace only the immutable reference; preserve zero residual/value state."""

    if set(base_params) != {"reference", "trainable"}:
        raise ValueError("DELTA v6 base parameters must contain reference/trainable trees.")
    return {
        "reference": _official_reference(base_params["reference"], official),
        "trainable": base_params["trainable"],
    }


def initializer_seed_index(checkpoint_path: str) -> int | None:
    """Read the registered ``run-<seed>/ckpt_final`` initializer convention."""

    # Inspect the path the caller supplied.  ``ckpt_final`` is commonly a
    # symlink into an Orbax directory whose target no longer contains the
    # registered ``run-<seed>`` alias; resolving first would erase the seed
    # identity we are checking.
    path = Path(checkpoint_path)
    for part in reversed(path.parts):
        match = re.fullmatch(r"run-(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def initialize_base_from_official(
    base_params: dict[str, Any], checkpoint_path: str
) -> dict[str, Any]:
    return transplant_official_base_params(
        base_params, load_official_parameters(checkpoint_path)
    )


__all__ = [
    "OFFICIAL_CONV_KEYS",
    "initialize_base_from_official",
    "initializer_seed_index",
    "load_official_parameters",
    "transplant_official_base_params",
]
