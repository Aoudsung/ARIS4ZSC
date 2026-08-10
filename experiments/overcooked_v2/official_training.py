"""Compatibility entrypoint for the pinned Official PPO trainer.

Official commit 5ce1707 stores ``RUN_BASE_DIR`` as a ``pathlib.Path`` inside
the Orbax checkpoint payload.  Supported Orbax releases do not serialize that
host object.  The training computation and checkpoint parameters remain
Official; only the copied config metadata is normalized to JSON-like strings
at the storage boundary.

The Official entrypoint also vmaps every population member resident on one GPU
at once.  The registered ten-run Wide population does not fit on an L40 in that
form.  This wrapper keeps the one root-key split and the Official train
function unchanged, but scans population members in batches of one run per
visible device.  This is an execution-layout change only: run order, PRNG keys,
budgets, parameters and checkpoint selection are identical.

Two Official modules reach the storage boundary and each holds its own
module-level reference to ``store_checkpoint``: ``ppo.main`` for the SP/OP/FCP
path, and ``ppo.state_sample_run`` for the state-augmented path, which
``ppo.main`` dispatches to whenever ``NUM_ITERATIONS`` is configured.  Both
references must be rebound -- patching only ``ppo.main`` leaves rnn-sa writing
through the unpatched function.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


def _paths_as_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _paths_as_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_paths_as_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_paths_as_strings(item) for item in value)
    return value


def _memory_bounded_population_map(function: Any, device_count: int) -> Any:
    """Map Official population runs one per visible device at a time."""

    import jax

    def mapped(*args: Any, **kwargs: Any) -> Any:
        outer_size = jax.tree_util.tree_leaves((args, kwargs))[0].shape[0]
        if int(outer_size) == int(device_count):
            return jax.pmap(function)(*args, **kwargs)

        sequential_batches = int(outer_size) // int(device_count)
        batched_args, batched_kwargs = jax.tree_util.tree_map(
            lambda value: value.reshape(
                (sequential_batches, int(device_count), *value.shape[1:])
            ),
            (args, kwargs),
        )

        def run_batch(unused: None, batch: Any) -> tuple[None, Any]:
            batch_args, batch_kwargs = batch
            return None, jax.pmap(function)(*batch_args, **batch_kwargs)

        unused, result = jax.lax.scan(
            run_batch,
            None,
            (batched_args, batched_kwargs),
        )
        del unused
        return jax.tree_util.tree_map(
            lambda value: value.reshape((int(outer_size), *value.shape[2:])),
            result,
        )

    return mapped


def main() -> None:
    import orbax.checkpoint as ocp
    from flax.training import orbax_utils
    from overcooked_v2_experiments.ppo import main as official_main
    from overcooked_v2_experiments.ppo import run as official_run
    from overcooked_v2_experiments.ppo import state_sample_run as official_state_sample
    from overcooked_v2_experiments.ppo.utils.store import _get_checkpoint_dir

    config_directory = Path(official_main.__file__).resolve().parent / "config"
    if "--config-path" not in sys.argv and not any(
        value.startswith("--config-path=") for value in sys.argv
    ):
        sys.argv[1:1] = ["--config-path", str(config_directory)]

    def store_checkpoint(
        config: dict[str, Any],
        params: Any,
        run_num: int,
        checkpoint: int,
        final: bool = False,
    ) -> None:
        checkpoint_dir = _get_checkpoint_dir(
            Path(config["RUN_BASE_DIR"]),
            run_num,
            checkpoint,
            final=final,
        )
        payload = {
            "config": _paths_as_strings(config),
            "params": params,
        }
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            checkpoint_dir,
            payload,
            save_args=orbax_utils.save_args_from_target(payload),
        )

    official_main.store_checkpoint = store_checkpoint
    official_run.mini_batch_pmap = _memory_bounded_population_map
    official_state_sample.store_checkpoint = store_checkpoint
    official_main.main()


if __name__ == "__main__":
    main()


__all__ = ["main"]
