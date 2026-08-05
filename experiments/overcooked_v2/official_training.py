"""Compatibility entrypoint for the pinned Official PPO trainer.

Official commit 5ce1707 stores ``RUN_BASE_DIR`` as a ``pathlib.Path`` inside
the Orbax checkpoint payload.  Supported Orbax releases do not serialize that
host object.  The training computation and checkpoint parameters remain
Official; only the copied config metadata is normalized to JSON-like strings
at the storage boundary.
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


def main() -> None:
    import orbax.checkpoint as ocp
    from flax.training import orbax_utils
    from overcooked_v2_experiments.ppo import main as official_main
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
    official_main.main()


if __name__ == "__main__":
    main()


__all__ = ["main"]
