"""Configuration, Orbax checkpoints, complete records, and resume state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence, TextIO, cast

from .method import DEPLOYMENT_MODES


CONFIG_VERSION = 1
LAYOUTS = ("test_time_simple", "test_time_wide")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    layout: str
    agent_view_size: int
    indicate_successful_delivery: bool
    episode_steps: int
    num_envs: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_dim: int
    action_embedding_dim: int
    slot_count: int
    response_count: int
    action_count: int
    prior_scale: float
    log_standard_deviation_minimum: float
    log_standard_deviation_maximum: float


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    environment_steps: int
    update_epochs: int
    minibatches_per_epoch: int
    bellman_learning_rate: float
    outcome_learning_rate: float
    gradient_clip_norm: float
    gamma: float
    responsibility_temperature: float
    bootstrap_probability: float
    polyak_coefficient: float
    codebook_decay: float
    code_replacement_rollouts: int
    checkpoint_interval_environment_steps: int


@dataclass(frozen=True, slots=True)
class KLConfig:
    target_per_step: float
    initial_temperature: float
    dual_learning_rate: float
    minimum_temperature: float
    maximum_temperature: float


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    episodes_per_pairing: int
    deployment_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UpstreamConfig:
    total_timesteps: int
    checkpoint_progress: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RunConfig:
    environment: EnvironmentConfig
    model: ModelConfig
    training: TrainingConfig
    kl: KLConfig
    evaluation: EvaluationConfig
    upstream: UpstreamConfig

    def to_mapping(self) -> dict[str, Any]:
        return {"version": CONFIG_VERSION, **asdict(self)}


def load_config(path: str | Path) -> RunConfig:
    """Load the one active configuration for a layout."""

    import yaml

    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("The configuration must be a mapping.")
    _require_fields(
        payload,
        {"version", "environment", "model", "training", "kl", "evaluation", "upstream"},
        "configuration",
    )
    if int(payload["version"]) != CONFIG_VERSION:
        raise ValueError("The active Path C configuration version is 1.")

    _require_fields(payload["environment"], set(EnvironmentConfig.__dataclass_fields__), "environment")
    _require_fields(payload["model"], set(ModelConfig.__dataclass_fields__), "model")
    _require_fields(payload["training"], set(TrainingConfig.__dataclass_fields__), "training")
    _require_fields(payload["kl"], set(KLConfig.__dataclass_fields__), "kl")
    environment = EnvironmentConfig(**payload["environment"])
    model = ModelConfig(**payload["model"])
    training = TrainingConfig(**payload["training"])
    kl = KLConfig(**payload["kl"])
    raw_evaluation = dict(payload["evaluation"])
    _require_fields(raw_evaluation, set(EvaluationConfig.__dataclass_fields__), "evaluation")
    evaluation = EvaluationConfig(
        episodes_per_pairing=int(raw_evaluation["episodes_per_pairing"]),
        deployment_modes=tuple(raw_evaluation["deployment_modes"]),
    )
    raw_upstream = dict(payload["upstream"])
    _require_fields(raw_upstream, set(UpstreamConfig.__dataclass_fields__), "upstream")
    upstream = UpstreamConfig(
        total_timesteps=int(raw_upstream["total_timesteps"]),
        checkpoint_progress=tuple(raw_upstream["checkpoint_progress"]),
    )
    result = RunConfig(
        environment=environment,
        model=model,
        training=training,
        kl=kl,
        evaluation=evaluation,
        upstream=upstream,
    )
    validate_config(result)
    return result


def _require_fields(
    payload: Any, expected: set[str], location: str
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{location} must be a mapping.")
    present = set(payload)
    if present != expected:
        raise ValueError(
            f"{location} fields differ: missing={sorted(expected - present)}, "
            f"unknown={sorted(present - expected)}."
        )


def validate_config(config: RunConfig) -> None:
    if config.environment.layout not in LAYOUTS:
        raise ValueError(f"Unknown OvercookedV2 layout: {config.environment.layout}")
    if config.environment.episode_steps != 400:
        raise ValueError("Standard OvercookedV2 evaluation uses 400-step episodes.")
    if config.environment.num_envs <= 0:
        raise ValueError("The vector environment count must be positive.")
    if config.model.action_count != 6:
        raise ValueError("JaxMARL OvercookedV2 exposes six actions.")
    if (
        config.model.hidden_dim <= 0
        or config.model.action_embedding_dim <= 0
        or config.model.slot_count < 2
        or config.model.response_count < 2
    ):
        raise ValueError("The method needs multiple slots and response codes.")
    if (
        config.model.log_standard_deviation_minimum
        >= config.model.log_standard_deviation_maximum
    ):
        raise ValueError("The outcome standard-deviation bounds are reversed.")
    if config.training.environment_steps <= 0:
        raise ValueError("Training environment steps must be positive.")
    if (
        config.training.update_epochs <= 0
        or config.training.minibatches_per_epoch <= 0
    ):
        raise ValueError("Training epochs and minibatches must be positive.")
    if (
        config.training.bellman_learning_rate <= 0.0
        or config.training.outcome_learning_rate <= 0.0
        or config.training.gradient_clip_norm <= 0.0
        or config.training.responsibility_temperature <= 0.0
    ):
        raise ValueError("Training rates, clipping, and responsibility temperature must be positive.")
    if not 0.0 <= config.training.gamma <= 1.0:
        raise ValueError("The return discount must lie in [0, 1].")
    for name, value in (
        ("bootstrap_probability", config.training.bootstrap_probability),
        ("polyak_coefficient", config.training.polyak_coefficient),
        ("codebook_decay", config.training.codebook_decay),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
    if (
        config.training.code_replacement_rollouts <= 0
        or config.training.checkpoint_interval_environment_steps <= 0
    ):
        raise ValueError("Code replacement and checkpoint intervals must be positive.")
    rollout_steps = (
        config.environment.num_envs * config.environment.episode_steps
    )
    if config.training.environment_steps % rollout_steps:
        raise ValueError("Training must contain whole vectorized episodes.")
    if config.environment.num_envs % config.training.minibatches_per_epoch:
        raise ValueError("Environment lanes must divide into whole minibatches.")
    if config.evaluation.episodes_per_pairing != 500:
        raise ValueError("Standard evaluation uses 500 episodes per pairing.")
    if tuple(config.evaluation.deployment_modes) != DEPLOYMENT_MODES:
        raise ValueError("The active evaluation contains all four deployment modes.")
    if (
        config.kl.target_per_step < 0.0
        or config.kl.initial_temperature <= 0.0
        or config.kl.dual_learning_rate <= 0.0
        or config.kl.minimum_temperature <= 0.0
        or config.kl.maximum_temperature < config.kl.minimum_temperature
    ):
        raise ValueError("Kullback–Leibler temperature settings are invalid.")
    if config.upstream.total_timesteps <= 0:
        raise ValueError("Official upstream training steps must be positive.")
    progress = config.upstream.checkpoint_progress
    if tuple(progress) != (0.0, 0.5, 1.0):
        raise ValueError("Upstream training saves start, midpoint, and final checkpoints.")


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


def save_checkpoint(
    manager: Any,
    *,
    step: int,
    state: Any,
) -> None:
    import orbax.checkpoint as ocp

    saved = manager.save(int(step), args=ocp.args.PyTreeSave(state))
    if saved is False:
        raise RuntimeError(f"Orbax did not save training step {step}.")
    manager.wait_until_finished()


def restore_latest_checkpoint(manager: Any) -> tuple[int, Any] | None:
    import orbax.checkpoint as ocp

    step = manager.latest_step()
    if step is None:
        return None
    state = manager.restore(int(step), args=ocp.args.PyTreeRestore())
    return int(step), state


def write_run_metadata(
    path: str | Path,
    *,
    config: RunConfig,
    seed: int,
    run_kind: str,
    effective_environment_steps: int,
    update_count: int,
    completed_episodes: int,
) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_kind": str(run_kind),
        "config": config.to_mapping(),
        "git_commit": current_git_commit(),
        "dependencies": dependency_versions(),
        "seed": int(seed),
        "effective_environment_steps": int(effective_environment_steps),
        "update_count": int(update_count),
        "completed_episodes": int(completed_episodes),
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def dependency_versions() -> Mapping[str, str]:
    names = (
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "orbax-checkpoint",
        "jaxmarl",
        "overcooked_v2_experiments",
        "numpy",
        "pyarrow",
    )
    return {name: importlib.metadata.version(name) for name in names}


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write every row supplied by the caller."""

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
    """Save a lossless sequence of compressed NumPy chunks."""

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
    """Write complete console output to both the terminal and one file."""

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
    "DEPLOYMENT_MODES",
    "EnvironmentConfig",
    "EvaluationConfig",
    "KLConfig",
    "ModelConfig",
    "RunConfig",
    "TrainingConfig",
    "UpstreamConfig",
    "dependency_versions",
    "load_config",
    "orbax_manager",
    "read_array_chunks",
    "read_parquet",
    "restore_latest_checkpoint",
    "save_checkpoint",
    "validate_config",
    "write_array_chunks",
    "write_json",
    "write_jsonl",
    "write_parquet",
    "write_run_metadata",
]
