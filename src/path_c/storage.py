"""Checkpoint, run identity, and lossless record I/O.

There is one identity file per run directory.  Resume and evaluation reuse it
verbatim; no second registry, content-addressed projection, or fallback path is
maintained.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence, TextIO, cast

from .experiment import (
    METHOD_VERSION,
    OFFICIAL_PROTOCOL_VERSION,
    OFFICIAL_SOURCE_COMMIT,
    PartnerManifest,
    RunConfig,
    official_training_domain_keys,
)

IDENTITY_FILE = "run_identity.json"


def _repository_state() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Active DELTA code is not inside an auditable Git checkout."
        ) from error
    return commit, dirty


def validate_registered_python_runtime() -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Formal DELTA/Official runs require the Official Python 3.10 runtime; "
            f"observed {sys.version_info.major}.{sys.version_info.minor}."
        )


def validate_formal_repository_state() -> None:
    commit, dirty = _repository_state()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Formal DELTA repository commit is not a full Git SHA.")
    if dirty:
        raise RuntimeError(
            "Formal runs require a clean committed DELTA checkout; uncommitted "
            "code cannot be reconstructed from run_identity.json."
        )


def training_identity(
    *,
    config: RunConfig,
    seed_index: int,
    jax_prng_key: Sequence[int],
    partner_manifest: PartnerManifest,
) -> Mapping[str, Any]:
    return {
        "stage": "train",
        "method": METHOD_VERSION,
        "run_kind": config.run_kind,
        "layout": config.environment.layout,
        "config": config.to_mapping(),
        "config_fingerprint": config.fingerprint,
        "seed_index": int(seed_index),
        "jax_prng_key": [int(value) for value in jax_prng_key],
        "domain_keys": {
            name: [int(value) for value in key]
            for name, key in official_training_domain_keys(seed_index).items()
        },
        "partner_manifest": partner_manifest.to_mapping(),
        "runtime": runtime_provenance(),
    }


def upstream_identity(
    *,
    config: RunConfig,
    seed_index: int,
    jax_prng_key: Sequence[int],
    algorithm: str,
) -> Mapping[str, Any]:
    return {
        "stage": "upstream",
        "method": "official_overcooked_v2_locked",
        "run_kind": config.run_kind,
        "layout": config.environment.layout,
        "config": config.to_mapping(),
        "config_fingerprint": config.fingerprint,
        "seed_index": int(seed_index),
        "jax_prng_key": [int(value) for value in jax_prng_key],
        "algorithm": str(algorithm),
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_protocol_version": OFFICIAL_PROTOCOL_VERSION,
        "runtime": runtime_provenance(),
    }


def runtime_provenance() -> Mapping[str, Any]:
    """Capture immutable software and device facts without requiring a GPU."""

    try:
        repository_commit, repository_dirty = _repository_state()
    except RuntimeError:
        repository_commit, repository_dirty = "unavailable", True
    distributions = sorted(
        {
            f"{distribution.metadata.get('Name', distribution.metadata.get('name', 'unknown'))}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
        }
    )
    dependency_payload = "\n".join(distributions)
    payload: dict[str, Any] = {
        "repository_commit": repository_commit,
        "repository_dirty": repository_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_protocol_version": OFFICIAL_PROTOCOL_VERSION,
        "installed_distribution_lock": distributions,
        "installed_distribution_lock_sha256": hashlib.sha256(
            dependency_payload.encode("utf-8")
        ).hexdigest(),
    }
    try:
        import jax

        payload.update(
            {
                "jax_version": str(jax.__version__),
                "jax_backend": str(jax.default_backend()),
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )
    except Exception as error:  # pragma: no cover - installation diagnostics
        payload["jax_runtime_error"] = f"{type(error).__name__}: {error}"
    return payload


def calibration_identity(
    *,
    config: RunConfig,
    seed_index: int,
    training_run: str | Path,
    manifest: PartnerManifest,
) -> Mapping[str, Any]:
    source = Path(training_run).resolve()
    return {
        "stage": "calibrate",
        "method": METHOD_VERSION,
        "run_kind": config.run_kind,
        "layout": config.environment.layout,
        "config": config.to_mapping(),
        "config_fingerprint": config.fingerprint,
        "seed_index": int(seed_index),
        "jax_prng_key": list(
            official_training_domain_keys(seed_index)["calibration"]
        ),
        "training_run": str(source),
        "training_identity": read_run_identity(source),
        "partner_manifest": manifest.to_mapping(),
    }


def evaluation_identity(
    *,
    config: RunConfig,
    seed: int,
    deployment_bundles: Sequence[str | Path],
    manifest: PartnerManifest,
) -> Mapping[str, Any]:
    bundles = tuple(Path(path).resolve() for path in deployment_bundles)
    return {
        "stage": "evaluate",
        "method": METHOD_VERSION,
        "run_kind": config.run_kind,
        "layout": config.environment.layout,
        "config": config.to_mapping(),
        "config_fingerprint": config.fingerprint,
        "seed": int(seed),
        "deployment_bundles": [
            {"path": str(path), "sha256": sha256_path(path)} for path in bundles
        ],
        "partner_manifest": manifest.to_mapping(),
    }


def sha256_path(path: str | Path) -> str:
    """Return a deterministic SHA-256 for one file or directory tree."""

    source = Path(path).resolve()
    digest = hashlib.sha256()
    if source.is_file():
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(item for item in source.rglob("*") if item.is_file())
    for child in files:
        relative = child.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with child.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def pytree_fingerprint(tree: Any) -> str:
    """Hash a parameter tree by structure, array dtype, shape, and bytes."""

    import numpy as np

    digest = hashlib.sha256()

    def update(value: Any, path: tuple[str, ...]) -> None:
        location = "/".join(path).encode("utf-8")
        digest.update(len(location).to_bytes(8, "big"))
        digest.update(location)
        if isinstance(value, Mapping):
            digest.update(b"mapping")
            for key in sorted(value, key=lambda item: str(item)):
                update(value[key], (*path, str(key)))
            return
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            digest.update(f"namedtuple:{type(value).__qualname__}".encode("utf-8"))
            for name in value._fields:
                update(getattr(value, name), (*path, str(name)))
            return
        if isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode("utf-8"))
            for index, child in enumerate(value):
                update(child, (*path, str(index)))
            return
        if value is None:
            digest.update(b"none")
            return
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(b"string")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            return
        if isinstance(value, bytes):
            digest.update(b"bytes")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
            return
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Object arrays cannot be fingerprinted at {'/'.join(path)}")
        contiguous = np.ascontiguousarray(array)
        digest.update(b"array")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))

    update(tree, ())
    return digest.hexdigest()


def ensure_run_identity(directory: str | Path, expected: Mapping[str, Any]) -> Path:
    """Create the run identity once, then require exact equality on reuse."""

    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / IDENTITY_FILE
    normalized = json.loads(json.dumps(dict(expected), sort_keys=True))
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != normalized:
            differing = sorted(
                key
                for key in set(observed) | set(normalized)
                if observed.get(key) != normalized.get(key)
            )
            raise RuntimeError(
                "Run directory belongs to a different experiment; "
                f"different fields={differing}."
            )
        return path
    existing = sorted(
        child.name for child in root.iterdir() if child.name != "logs"
    )
    if existing:
        raise RuntimeError(
            "Existing run data has no identity and cannot be adopted; "
            f"entries={existing}."
        )
    path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_run_identity(directory: str | Path) -> Mapping[str, Any]:
    path = Path(directory).resolve() / IDENTITY_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Run identity is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Run identity must be a JSON mapping.")
    return payload


def orbax_manager(directory: str | Path, *, create: bool = True) -> Any:
    import orbax.checkpoint as ocp

    root = Path(directory).resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    return ocp.CheckpointManager(
        str(root),
        options=ocp.CheckpointManagerOptions(create=create),
    )


def save_checkpoint(manager: Any, *, step: int, state: Any) -> None:
    import orbax.checkpoint as ocp

    saved = manager.save(int(step), args=ocp.args.PyTreeSave(state))
    if saved is False:
        raise RuntimeError(f"Orbax did not save training step {step}.")
    manager.wait_until_finished()


def restore_latest_checkpoint(
    manager: Any, *, item: Any | None = None
) -> tuple[int, Any] | None:
    """Restore the latest Orbax tree, optionally into its concrete state type."""

    import orbax.checkpoint as ocp

    step = manager.latest_step()
    if step is None:
        return None
    state = manager.restore(
        int(step), args=ocp.args.PyTreeRestore(item=item)
    )
    return int(step), state


def restore_checkpoint_step(manager: Any, *, step: int) -> Any:
    """Restore exactly one registered Orbax step without selecting another."""

    import orbax.checkpoint as ocp

    requested = int(step)
    available = {int(value) for value in manager.all_steps()}
    if requested not in available:
        raise FileNotFoundError(
            f"Orbax checkpoint step {requested} is absent; available={sorted(available)}."
        )
    return manager.restore(
        requested, args=ocp.args.PyTreeRestore()
    )


def write_run_metadata(
    path: str | Path,
    *,
    config: RunConfig,
    seed: int,
    effective_environment_steps: int,
    update_count: int,
    completed_episodes: int,
) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": METHOD_VERSION,
        "config": config.to_mapping(),
        "seed": int(seed),
        "effective_environment_steps": int(effective_environment_steps),
        "update_count": int(update_count),
        "completed_episodes": int(completed_episodes),
        "scientific_readout_allowed": False,
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a small JSON status artifact on the same filesystem."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return target


def write_parquet(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), target)
    return target


def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(Path(path).resolve()).to_pylist()


def write_array_chunks(
    directory: str | Path,
    *,
    name: str,
    values: Any,
    rows_per_chunk: int,
) -> tuple[Path, ...]:
    import numpy as np

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("Chunked arrays require a row axis.")
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive.")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, array.shape[0], int(rows_per_chunk)):
        end = min(start + int(rows_per_chunk), array.shape[0])
        target = root / f"{name}_{start:012d}_{end:012d}.npz"
        np.savez_compressed(target, values=array[start:end])
        paths.append(target)
    return tuple(paths)


def read_array_chunks(paths: Sequence[str | Path]) -> Any:
    import numpy as np

    arrays = []
    for path in paths:
        with np.load(Path(path).resolve(), allow_pickle=False) as payload:
            arrays.append(payload["values"])
    return np.concatenate(arrays, axis=0)


class Tee(TextIO):
    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self._terminal = terminal
        self._log = log

    def write(self, value: str) -> int:
        self._terminal.write(value)
        self._log.write(value)
        return len(value)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()


class CompleteConsoleLog:
    def __init__(self, directory: str | Path) -> None:
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.stdout_path = root / "stdout.log"
        self.stderr_path = root / "stderr.log"
        self._stdout_file: TextIO | None = None
        self._stderr_file: TextIO | None = None
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None

    def __enter__(self) -> "CompleteConsoleLog":
        self._stdout_file = self.stdout_path.open("a", encoding="utf-8")
        self._stderr_file = self.stderr_path.open("a", encoding="utf-8")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = Tee(sys.stdout, self._stdout_file)
        sys.stderr = Tee(sys.stderr, self._stderr_file)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        if exc_type is not None:
            import traceback as traceback_module

            traceback_module.print_exception(
                exc_type,
                exc,
                traceback_value,
                file=cast(TextIO, self._stderr_file),
            )
            cast(TextIO, self._stderr_file).flush()
        sys.stdout = cast(TextIO, self._original_stdout)
        sys.stderr = cast(TextIO, self._original_stderr)
        cast(TextIO, self._stdout_file).close()
        cast(TextIO, self._stderr_file).close()


__all__ = [
    "CompleteConsoleLog",
    "calibration_identity",
    "ensure_run_identity",
    "evaluation_identity",
    "orbax_manager",
    "pytree_fingerprint",
    "read_array_chunks",
    "read_parquet",
    "read_run_identity",
    "runtime_provenance",
    "restore_latest_checkpoint",
    "restore_checkpoint_step",
    "save_checkpoint",
    "sha256_path",
    "training_identity",
    "validate_formal_repository_state",
    "validate_registered_python_runtime",
    "upstream_identity",
    "write_array_chunks",
    "write_json",
    "write_json_atomic",
    "write_jsonl",
    "write_parquet",
    "write_run_metadata",
]
