"""Artifact-only real-return readout audit for the frozen Path C V4.4 run.

The audit consumes only the already-generated counterfactual continuation
artifacts.  It never restores a checkpoint, advances an environment, or changes
the frozen input directory.  Partner identity is used exclusively to construct
leave-one-partner-out folds and partner-balanced summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.path_c.storage import write_json, write_parquet


READOUT_VERSION = "path_c_v44_artifact_readout_v2"
SPLIT_VERSION = "path_c_v44_artifact_readout_v1"
SPLIT_SALT = f"{SPLIT_VERSION}:replica-fit-evaluation"
BOOTSTRAP_SEED = 20_260_729
RIDGE_GRID = tuple(10.0**power for power in range(-4, 4))
FEATURE_TIERS = ("state", "history")
LEGAL_HISTORY_FIELDS = (
    "reference_carry",
    "trainable_carry",
    "previous_action",
    "previous_team_reward",
)
EXCLUDED_MODEL_FIELDS = (
    "partner_identity_or_seed",
    "sp_op_or_training_source_label",
    "control_carry",
    "slot_belief_or_responsibility",
    "model_q_j_or_lcb",
    "full_environment_state",
    "privileged_task_phase",
    "future_continuation_fields",
    "held_out_fold_statistics",
)
INPUT_FILES = (
    "reconstruction.parquet",
    "continuations.parquet",
    "trigger_values.parquet",
    "summary.json",
    "run_metadata.json",
)
IMPLEMENTATION_FILES = (
    "experiments/overcooked_v2/artifact_readout_app.py",
    "experiments/overcooked_v2/path_c.py",
    "experiments/overcooked_v2/tests/test_path_c_artifact_readout.py",
    "src/path_c/storage.py",
)
DETERMINISTIC_OUTPUT_FILES = (
    "bootstrap.parquet",
    "configuration.json",
    "feature_schema.json",
    "report.md",
    "run_metadata.json",
    "split_manifest.parquet",
    "state_readout.parquet",
    "summary.json",
)


@dataclass(frozen=True, slots=True)
class ArtifactExpectations:
    trigger_count: int = 225
    fixed_trigger_count: int = 178
    self_trigger_count: int = 47
    partner_count: int = 4
    replicas: int = 128
    action_count: int = 6
    episode_steps: int = 400
    continuation_row_count: int = 403_200
    bootstrap_replicates: int = 10_000

    @property
    def split_replicas(self) -> int:
        if self.replicas <= 0 or self.replicas % 2:
            raise ValueError("Replica count must be a positive even integer.")
        return self.replicas // 2


REGISTERED_EXPECTATIONS = ArtifactExpectations()


@dataclass(frozen=True, slots=True)
class ArtifactState:
    trigger_id: str
    source: str
    partner_index: int | None
    episode_index: int
    episode_seed: int
    trigger_step: int
    trigger_action: int
    response_code: int
    state_features: np.ndarray
    history_features: np.ndarray
    q_fit_use: np.ndarray
    q_fit_mask: np.ndarray
    q_evaluation_use: np.ndarray
    q_evaluation_mask: np.ndarray
    q_evaluation_use_discounted: np.ndarray
    q_evaluation_mask_discounted: np.ndarray
    q_evaluation_use_correct_deliveries: np.ndarray
    q_evaluation_mask_correct_deliveries: np.ndarray
    q_evaluation_use_wrong_deliveries: np.ndarray
    q_evaluation_mask_wrong_deliveries: np.ndarray
    tau_fit: float
    tau_evaluation: float
    tau_evaluation_discounted: float
    tau_evaluation_correct_deliveries: float
    tau_evaluation_wrong_deliveries: float

    def features(self, tier: str) -> np.ndarray:
        if tier == "state":
            return self.state_features
        if tier == "history":
            return self.history_features
        raise ValueError(f"Unknown feature tier: {tier}")


@dataclass(frozen=True, slots=True)
class RidgeModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    retained: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Ridge features must have a row and feature axis.")
        standardized = (
            (values[:, self.retained] - self.feature_mean[self.retained])
            / self.feature_scale[self.retained]
        )
        return self.target_mean + standardized @ self.weights


@dataclass(frozen=True, slots=True)
class FoldModels:
    mask: RidgeModel
    delta: RidgeModel
    response_codes: tuple[int, ...]
    lambda_mask: float
    lambda_delta: float


@dataclass(frozen=True, slots=True)
class ReadoutPrediction:
    state: ArtifactState
    feature_tier: str
    evaluation_kind: str
    fold_partner: int | None
    lambda_mask: float
    lambda_delta: float
    predicted_mask: np.ndarray
    predicted_use: np.ndarray
    probe_mask_action: int
    probe_use_action: int
    oracle_mask_action: int
    oracle_use_action: int
    probe_lift: float
    oracle_lift: float


@dataclass(frozen=True, slots=True)
class LopoResult:
    predictions: tuple[ReadoutPrediction, ...]
    fold_lambdas: Mapping[int, Mapping[str, float]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_registered_inputs(directory: str | Path) -> Mapping[str, str]:
    root = Path(directory).resolve()
    missing = [name for name in INPUT_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Frozen counterfactual artifact files are missing: {missing}."
        )
    return {name: _sha256(root / name) for name in INPUT_FILES}


def implementation_provenance() -> Mapping[str, Any]:
    """Identify the exact source and runtime used for a readout bundle."""

    root = Path(__file__).resolve().parents[2]
    missing = [name for name in IMPLEMENTATION_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Readout implementation files are missing: {missing}.")
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    git_head = result.stdout.strip() if result.returncode == 0 else "unavailable"
    try:
        import pyarrow

        pyarrow_version = str(pyarrow.__version__)
    except ImportError:
        pyarrow_version = "unavailable"
    return {
        "source_sha256": {name: _sha256(root / name) for name in IMPLEMENTATION_FILES},
        "git_head": git_head,
        "source_identity": "git_head_plus_exact_file_sha256",
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": str(np.__version__),
            "pyarrow": pyarrow_version,
            "platform": sys.platform,
        },
    }


def deterministic_replica_split(
    trigger_id: str, replicas: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return a fixed, paired half split for every branch and forced action."""

    if replicas <= 0 or replicas % 2:
        raise ValueError("Replica count must be a positive even integer.")
    ranked = []
    for replica in range(int(replicas)):
        payload = f"{SPLIT_SALT}:{trigger_id}:{replica}".encode("utf-8")
        ranked.append((hashlib.sha256(payload).digest(), replica))
    ordered = [replica for unused_hash, replica in sorted(ranked)]
    middle = replicas // 2
    fit = tuple(sorted(ordered[:middle]))
    evaluation = tuple(sorted(ordered[middle:]))
    if set(fit) & set(evaluation) or len(fit) != len(evaluation):
        raise AssertionError("Replica split is not an exact partition.")
    return fit, evaluation


def _flatten_numeric(value: Any, *, prefix: str) -> tuple[tuple[str, ...], np.ndarray]:
    names: list[str] = []
    values: list[float] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for name in sorted(item):
                visit(item[name], f"{path}.{name}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if item is None:
            raise ValueError(f"Legal feature is unexpectedly null: {path}")
        if isinstance(item, (bool, int, float, np.number)):
            scalar = float(item)
            if not np.isfinite(scalar):
                raise ValueError(f"Legal feature is non-finite: {path}")
            names.append(path)
            values.append(scalar)
            return
        raise TypeError(f"Legal feature is not numeric at {path}: {type(item)!r}")

    visit(value, prefix)
    return tuple(names), np.asarray(values, dtype=np.float64)


def _agent_zero_observation(observations: Any) -> Any:
    if isinstance(observations, Mapping):
        for key in ("0", 0, "agent_0", "agent0"):
            if key in observations:
                return observations[key]
        raise ValueError("Official observations do not expose an agent-0 entry.")
    if isinstance(observations, (list, tuple)) and len(observations) == 2:
        return observations[0]
    raise ValueError("Official observations do not have the registered two-agent form.")


def extract_legal_features(
    row: Mapping[str, Any],
    *,
    action_count: int,
    episode_steps: int,
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...], np.ndarray]:
    """Extract only the predeclared deployable feature whitelist.

    Partner metadata, simulator state, task-phase summaries, Path C control
    carry, slot beliefs, and value predictions are never traversed here.
    """

    required = {
        "post_observations",
        "post_use_ego_state",
        "post_mask_ego_state",
        "trigger_step",
        "trigger_action",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Reconstruction row lacks legal feature fields: {missing}.")
    trigger_step = int(row["trigger_step"])
    trigger_action = int(row["trigger_action"])
    if not 0 <= trigger_step < int(episode_steps):
        raise ValueError(f"Trigger step is outside the registered episode: {trigger_step}.")
    if not 0 <= trigger_action < int(action_count):
        raise ValueError(f"Trigger action is outside the registered action set: {trigger_action}.")

    observation_names, observation_values = _flatten_numeric(
        _agent_zero_observation(row["post_observations"]),
        prefix="official_post_observation",
    )
    trigger_one_hot = np.zeros((action_count,), dtype=np.float64)
    trigger_one_hot[trigger_action] = 1.0
    state_names = (
        *observation_names,
        "normalized_trigger_step",
        *(f"trigger_action[{index}]" for index in range(action_count)),
    )
    denominator = max(1, int(episode_steps) - 1)
    state_values = np.concatenate(
        (
            observation_values,
            np.asarray([trigger_step / denominator], dtype=np.float64),
            trigger_one_hot,
        )
    )

    use_state = row["post_use_ego_state"]
    mask_state = row["post_mask_ego_state"]
    if not isinstance(use_state, Mapping) or not isinstance(mask_state, Mapping):
        raise ValueError("Serialized ego states must be mappings.")
    history_names: list[str] = list(state_names)
    history_parts: list[np.ndarray] = [state_values]
    for field in LEGAL_HISTORY_FIELDS:
        if field not in use_state or field not in mask_state:
            raise ValueError(f"Frozen official-history field is absent: {field}.")
        use_names, use_values = _flatten_numeric(
            use_state[field], prefix=f"official_history.{field}"
        )
        mask_names, mask_values = _flatten_numeric(
            mask_state[field], prefix=f"official_history.{field}"
        )
        if use_names != mask_names or not np.array_equal(use_values, mask_values):
            raise ValueError(
                "Use/mask branches differ in a whitelisted official-history field: "
                f"{field}."
            )
        history_names.extend(use_names)
        history_parts.append(use_values)
    return (
        tuple(state_names),
        state_values,
        tuple(history_names),
        np.concatenate(history_parts),
    )


def _read_parquet_columns(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"Parquet schema is missing registered columns: {missing}.")
    return pq.read_table(path, columns=list(columns)).to_pylist()


class _ContinuationAccumulator:
    def __init__(self, replicas: int, action_count: int) -> None:
        shape_pre = (2, replicas)
        shape_post = (2, action_count, replicas)
        self.pre_raw = np.full(shape_pre, np.nan, dtype=np.float64)
        self.pre_discounted = np.full(shape_pre, np.nan, dtype=np.float64)
        self.post_raw = np.full(shape_post, np.nan, dtype=np.float64)
        self.post_discounted = np.full(shape_post, np.nan, dtype=np.float64)
        self.pre_correct_deliveries = np.full(shape_pre, np.nan, dtype=np.float64)
        self.pre_wrong_deliveries = np.full(shape_pre, np.nan, dtype=np.float64)
        self.post_correct_deliveries = np.full(shape_post, np.nan, dtype=np.float64)
        self.post_wrong_deliveries = np.full(shape_post, np.nan, dtype=np.float64)
        self.pre_seen = np.zeros(shape_pre, dtype=np.bool_)
        self.post_seen = np.zeros(shape_post, dtype=np.bool_)


def load_registered_artifact(
    directory: str | Path,
    *,
    expectations: ArtifactExpectations = REGISTERED_EXPECTATIONS,
) -> tuple[tuple[ArtifactState, ...], tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    """Load only registered columns and validate the complete paired index."""

    root = Path(directory).resolve()
    reconstruction_columns = (
        "trigger_id",
        "source",
        "partner_index",
        "episode_index",
        "episode_seed",
        "trigger_step",
        "trigger_action",
        "response_code",
        "post_observations",
        "post_use_ego_state",
        "post_mask_ego_state",
    )
    continuation_columns = (
        "trigger_id",
        "source",
        "partner_index",
        "episode_index",
        "episode_seed",
        "trigger_step",
        "estimand",
        "forced_action",
        "replica_index",
        "branch",
        "remaining_raw_return",
        "remaining_discounted_return",
        "remaining_correct_deliveries",
        "remaining_wrong_deliveries",
    )
    reconstruction_rows = _read_parquet_columns(
        root / "reconstruction.parquet", reconstruction_columns
    )
    trigger_value_rows = _read_parquet_columns(
        root / "trigger_values.parquet", ("trigger_id", "replicas")
    )
    continuation_rows = _read_parquet_columns(
        root / "continuations.parquet", continuation_columns
    )
    if len(reconstruction_rows) != expectations.trigger_count:
        raise ValueError(
            "Reconstruction trigger count differs from the registered audit: "
            f"{len(reconstruction_rows)} != {expectations.trigger_count}."
        )
    if len(trigger_value_rows) != expectations.trigger_count:
        raise ValueError(
            "Trigger-value count differs from the registered audit: "
            f"{len(trigger_value_rows)} != {expectations.trigger_count}."
        )
    if len(continuation_rows) != expectations.continuation_row_count:
        raise ValueError(
            "Continuation row count differs from the registered audit: "
            f"{len(continuation_rows)} != {expectations.continuation_row_count}."
        )
    reconstructions: dict[str, Mapping[str, Any]] = {}
    state_names: tuple[str, ...] | None = None
    history_names: tuple[str, ...] | None = None
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in reconstruction_rows:
        trigger_id = str(row["trigger_id"])
        if trigger_id in reconstructions:
            raise ValueError(f"Duplicate reconstruction trigger: {trigger_id}.")
        current_state_names, state_values, current_history_names, history_values = (
            extract_legal_features(
                row,
                action_count=expectations.action_count,
                episode_steps=expectations.episode_steps,
            )
        )
        if state_names is None:
            state_names = current_state_names
            history_names = current_history_names
        elif state_names != current_state_names or history_names != current_history_names:
            raise ValueError("Legal feature schema changes between trigger states.")
        reconstructions[trigger_id] = row
        features[trigger_id] = (state_values, history_values)
    value_ids = [str(row["trigger_id"]) for row in trigger_value_rows]
    if len(set(value_ids)) != len(value_ids) or set(value_ids) != set(reconstructions):
        raise ValueError("Trigger-value identities differ from reconstruction identities.")
    if any(int(row["replicas"]) != expectations.replicas for row in trigger_value_rows):
        raise ValueError("Trigger-value replica counts differ from the registered audit.")

    accumulators = {
        trigger_id: _ContinuationAccumulator(
            expectations.replicas, expectations.action_count
        )
        for trigger_id in reconstructions
    }
    branch_indexes = {"use": 0, "mask": 1}
    identity_fields = (
        "source",
        "partner_index",
        "episode_index",
        "episode_seed",
        "trigger_step",
    )
    for row_number, row in enumerate(continuation_rows):
        trigger_id = str(row["trigger_id"])
        if trigger_id not in accumulators:
            raise ValueError(f"Continuation refers to unknown trigger: {trigger_id}.")
        reconstruction = reconstructions[trigger_id]
        for field in identity_fields:
            if row[field] != reconstruction[field]:
                raise ValueError(
                    f"Continuation identity differs at row {row_number}: {field}."
                )
        branch = str(row["branch"])
        if branch not in branch_indexes:
            raise ValueError(f"Unknown continuation branch: {branch}.")
        branch_index = branch_indexes[branch]
        replica = int(row["replica_index"])
        if not 0 <= replica < expectations.replicas:
            raise ValueError(f"Replica index is outside the registered range: {replica}.")
        raw_return = float(row["remaining_raw_return"])
        discounted_return = float(row["remaining_discounted_return"])
        correct_deliveries = float(row["remaining_correct_deliveries"])
        wrong_deliveries = float(row["remaining_wrong_deliveries"])
        if not np.isfinite(raw_return) or not np.isfinite(discounted_return):
            raise ValueError("Continuation return is non-finite.")
        if (
            not np.isfinite(correct_deliveries)
            or not np.isfinite(wrong_deliveries)
            or correct_deliveries < 0.0
            or wrong_deliveries < 0.0
        ):
            raise ValueError("Continuation delivery count is invalid.")
        accumulator = accumulators[trigger_id]
        estimand = str(row["estimand"])
        if estimand == "pre_response":
            if int(row["forced_action"]) != int(reconstruction["trigger_action"]):
                raise ValueError(
                    "Pre-response continuation does not preserve the registered "
                    "trigger action."
                )
            if accumulator.pre_seen[branch_index, replica]:
                raise ValueError("Duplicate pre-response continuation index.")
            accumulator.pre_seen[branch_index, replica] = True
            accumulator.pre_raw[branch_index, replica] = raw_return
            accumulator.pre_discounted[branch_index, replica] = discounted_return
            accumulator.pre_correct_deliveries[branch_index, replica] = correct_deliveries
            accumulator.pre_wrong_deliveries[branch_index, replica] = wrong_deliveries
        elif estimand == "post_response":
            if row["forced_action"] is None:
                raise ValueError("Post-response continuation lacks a forced action.")
            action = int(row["forced_action"])
            if not 0 <= action < expectations.action_count:
                raise ValueError(f"Forced action is outside the action set: {action}.")
            if accumulator.post_seen[branch_index, action, replica]:
                raise ValueError("Duplicate post-response continuation index.")
            accumulator.post_seen[branch_index, action, replica] = True
            accumulator.post_raw[branch_index, action, replica] = raw_return
            accumulator.post_discounted[branch_index, action, replica] = discounted_return
            accumulator.post_correct_deliveries[
                branch_index, action, replica
            ] = correct_deliveries
            accumulator.post_wrong_deliveries[
                branch_index, action, replica
            ] = wrong_deliveries
        else:
            raise ValueError(f"Unknown continuation estimand: {estimand}.")

    states: list[ArtifactState] = []
    split_rows: list[Mapping[str, Any]] = []
    for trigger_id in sorted(reconstructions):
        row = reconstructions[trigger_id]
        accumulator = accumulators[trigger_id]
        if not np.all(accumulator.pre_seen) or not np.all(accumulator.post_seen):
            raise ValueError(f"Continuation lattice is incomplete for {trigger_id}.")
        fit_tuple, evaluation_tuple = deterministic_replica_split(
            trigger_id, expectations.replicas
        )
        fit = np.asarray(fit_tuple, dtype=np.int64)
        evaluation = np.asarray(evaluation_tuple, dtype=np.int64)
        for replica in range(expectations.replicas):
            split_rows.append(
                {
                    "split_version": READOUT_VERSION,
                    "trigger_id": trigger_id,
                    "replica_index": replica,
                    "split": "fit" if replica in fit_tuple else "evaluation",
                }
            )
        pre_raw = accumulator.pre_raw
        pre_discounted = accumulator.pre_discounted
        post_raw = accumulator.post_raw
        post_discounted = accumulator.post_discounted
        pre_correct = accumulator.pre_correct_deliveries
        pre_wrong = accumulator.pre_wrong_deliveries
        post_correct = accumulator.post_correct_deliveries
        post_wrong = accumulator.post_wrong_deliveries
        state_values, history_values = features[trigger_id]
        partner_index = row["partner_index"]
        states.append(
            ArtifactState(
                trigger_id=trigger_id,
                source=str(row["source"]),
                partner_index=(None if partner_index is None else int(partner_index)),
                episode_index=int(row["episode_index"]),
                episode_seed=int(row["episode_seed"]),
                trigger_step=int(row["trigger_step"]),
                trigger_action=int(row["trigger_action"]),
                response_code=int(row["response_code"]),
                state_features=state_values,
                history_features=history_values,
                q_fit_use=np.mean(post_raw[0][:, fit], axis=1),
                q_fit_mask=np.mean(post_raw[1][:, fit], axis=1),
                q_evaluation_use=np.mean(post_raw[0][:, evaluation], axis=1),
                q_evaluation_mask=np.mean(post_raw[1][:, evaluation], axis=1),
                q_evaluation_use_discounted=np.mean(
                    post_discounted[0][:, evaluation], axis=1
                ),
                q_evaluation_mask_discounted=np.mean(
                    post_discounted[1][:, evaluation], axis=1
                ),
                q_evaluation_use_correct_deliveries=np.mean(
                    post_correct[0][:, evaluation], axis=1
                ),
                q_evaluation_mask_correct_deliveries=np.mean(
                    post_correct[1][:, evaluation], axis=1
                ),
                q_evaluation_use_wrong_deliveries=np.mean(
                    post_wrong[0][:, evaluation], axis=1
                ),
                q_evaluation_mask_wrong_deliveries=np.mean(
                    post_wrong[1][:, evaluation], axis=1
                ),
                tau_fit=float(np.mean(pre_raw[0, fit] - pre_raw[1, fit])),
                tau_evaluation=float(
                    np.mean(pre_raw[0, evaluation] - pre_raw[1, evaluation])
                ),
                tau_evaluation_discounted=float(
                    np.mean(
                        pre_discounted[0, evaluation]
                        - pre_discounted[1, evaluation]
                    )
                ),
                tau_evaluation_correct_deliveries=float(
                    np.mean(pre_correct[0, evaluation] - pre_correct[1, evaluation])
                ),
                tau_evaluation_wrong_deliveries=float(
                    np.mean(pre_wrong[0, evaluation] - pre_wrong[1, evaluation])
                ),
            )
        )

    fixed = [state for state in states if state.partner_index is not None]
    self_states = [state for state in states if state.partner_index is None]
    partners = sorted({int(state.partner_index) for state in fixed})
    if len(fixed) != expectations.fixed_trigger_count:
        raise ValueError(
            f"Fixed-partner trigger count differs: {len(fixed)} != "
            f"{expectations.fixed_trigger_count}."
        )
    if len(self_states) != expectations.self_trigger_count:
        raise ValueError(
            f"Self-pair trigger count differs: {len(self_states)} != "
            f"{expectations.self_trigger_count}."
        )
    if len(partners) != expectations.partner_count:
        raise ValueError(
            f"Fixed-partner fold count differs: {len(partners)} != "
            f"{expectations.partner_count}."
        )
    expected_partners = list(range(expectations.partner_count))
    if partners != expected_partners:
        raise ValueError(
            "Fixed-partner indexes differ from the registered folds: "
            f"{partners} != {expected_partners}."
        )
    if any(not any(state.partner_index == partner for state in fixed) for partner in partners):
        raise ValueError("A fixed-partner stratum is empty.")
    metadata = {
        "feature_names": {
            "state": list(state_names or ()),
            "history": list(history_names or ()),
        },
        "partner_trigger_counts": {
            str(partner): sum(state.partner_index == partner for state in fixed)
            for partner in partners
        },
        "trigger_count": len(states),
        "fixed_trigger_count": len(fixed),
        "self_trigger_count": len(self_states),
        "continuation_row_count": len(continuation_rows),
    }
    return tuple(states), tuple(split_rows), metadata


def _response_vocabulary(states: Sequence[ArtifactState]) -> tuple[int, ...]:
    return tuple(sorted({int(state.response_code) for state in states}))


def encode_response_codes(
    states: Sequence[ArtifactState], vocabulary: Sequence[int]
) -> np.ndarray:
    """Encode known response codes and one explicit, untrained UNK column."""

    lookup = {int(code): index for index, code in enumerate(vocabulary)}
    encoded = np.zeros((len(states), len(vocabulary) + 1), dtype=np.float64)
    unknown = len(vocabulary)
    for row, state in enumerate(states):
        encoded[row, lookup.get(int(state.response_code), unknown)] = 1.0
    return encoded


def _feature_matrix(states: Sequence[ArtifactState], tier: str) -> np.ndarray:
    if not states:
        raise ValueError("A readout split cannot be empty.")
    matrix = np.stack([state.features(tier) for state in states], axis=0)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Readout features contain non-finite values.")
    return matrix


def _target_matrix(states: Sequence[ArtifactState], target: str) -> np.ndarray:
    if target == "mask":
        return np.stack([state.q_fit_mask for state in states], axis=0)
    if target == "delta":
        return np.stack(
            [state.q_fit_use - state.q_fit_mask for state in states], axis=0
        )
    raise ValueError(f"Unknown ridge target: {target}")


def _model_features(
    states: Sequence[ArtifactState],
    *,
    tier: str,
    target: str,
    vocabulary: Sequence[int],
) -> np.ndarray:
    base = _feature_matrix(states, tier)
    if target == "mask":
        return base
    if target == "delta":
        return np.concatenate(
            (base, encode_response_codes(states, vocabulary)), axis=1
        )
    raise ValueError(f"Unknown ridge target: {target}")


def fit_ridge(features: np.ndarray, targets: np.ndarray, penalty: float) -> RidgeModel:
    values = np.asarray(features, dtype=np.float64)
    outcomes = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or outcomes.ndim != 2 or values.shape[0] != outcomes.shape[0]:
        raise ValueError("Ridge features and targets must share a row axis.")
    if values.shape[0] == 0 or penalty <= 0.0:
        raise ValueError("Ridge requires observations and a positive penalty.")
    feature_mean = np.mean(values, axis=0)
    feature_scale = np.std(values, axis=0)
    retained = feature_scale > 1.0e-12
    safe_scale = np.where(retained, feature_scale, 1.0)
    standardized = (
        (values[:, retained] - feature_mean[retained]) / safe_scale[retained]
    )
    target_mean = np.mean(outcomes, axis=0)
    centered_targets = outcomes - target_mean
    if standardized.shape[1] == 0:
        weights = np.zeros((0, outcomes.shape[1]), dtype=np.float64)
    elif standardized.shape[1] > standardized.shape[0]:
        gram = standardized @ standardized.T
        system = gram + float(penalty) * np.eye(
            standardized.shape[0], dtype=np.float64
        )
        dual = np.linalg.solve(system, centered_targets)
        weights = standardized.T @ dual
    else:
        left, singular, right = np.linalg.svd(standardized, full_matrices=False)
        projected = left.T @ centered_targets
        shrinkage = singular / (np.square(singular) + float(penalty))
        weights = right.T @ (shrinkage[:, None] * projected)
    return RidgeModel(
        feature_mean=feature_mean,
        feature_scale=safe_scale,
        retained=retained,
        target_mean=target_mean,
        weights=weights,
    )


def _ridge_path_predictions(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    validation_features: np.ndarray,
    penalties: Sequence[float],
) -> Mapping[float, np.ndarray]:
    """Predict an entire ridge path with one standardization and one SVD."""

    values = np.asarray(training_features, dtype=np.float64)
    outcomes = np.asarray(training_targets, dtype=np.float64)
    validation = np.asarray(validation_features, dtype=np.float64)
    if (
        values.ndim != 2
        or outcomes.ndim != 2
        or validation.ndim != 2
        or values.shape[0] != outcomes.shape[0]
        or values.shape[1] != validation.shape[1]
    ):
        raise ValueError("Ridge-path arrays have incompatible shapes.")
    feature_mean = np.mean(values, axis=0)
    feature_scale = np.std(values, axis=0)
    retained = feature_scale > 1.0e-12
    safe_scale = np.where(retained, feature_scale, 1.0)
    standardized = (
        (values[:, retained] - feature_mean[retained]) / safe_scale[retained]
    )
    standardized_validation = (
        (validation[:, retained] - feature_mean[retained])
        / safe_scale[retained]
    )
    target_mean = np.mean(outcomes, axis=0)
    centered_targets = outcomes - target_mean
    if standardized.shape[1] == 0:
        return {
            float(penalty): np.broadcast_to(
                target_mean, (validation.shape[0], outcomes.shape[1])
            ).copy()
            for penalty in penalties
        }
    predictions = {}
    if standardized.shape[1] > standardized.shape[0]:
        gram = standardized @ standardized.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected_targets = eigenvectors.T @ centered_targets
        validation_basis = (
            standardized_validation @ standardized.T @ eigenvectors
        )
        for penalty in penalties:
            if float(penalty) <= 0.0:
                raise ValueError("Ridge penalties must be positive.")
            scaled_targets = projected_targets / (
                eigenvalues[:, None] + float(penalty)
            )
            predictions[float(penalty)] = (
                target_mean + validation_basis @ scaled_targets
            )
    else:
        left, singular, right = np.linalg.svd(
            standardized, full_matrices=False
        )
        projected = left.T @ centered_targets
        for penalty in penalties:
            if float(penalty) <= 0.0:
                raise ValueError("Ridge penalties must be positive.")
            shrinkage = singular / (np.square(singular) + float(penalty))
            weights = right.T @ (shrinkage[:, None] * projected)
            predictions[float(penalty)] = (
                target_mean + standardized_validation @ weights
            )
    return predictions


def _select_lambda(
    states: Sequence[ArtifactState], *, tier: str, target: str
) -> float:
    partners = sorted({state.partner_index for state in states})
    if None in partners or len(partners) < 2:
        raise ValueError("Inner LOPO requires at least two fixed partners.")
    scores = {penalty: [] for penalty in RIDGE_GRID}
    for held_out in partners:
        training = [state for state in states if state.partner_index != held_out]
        validation = [state for state in states if state.partner_index == held_out]
        vocabulary = _response_vocabulary(training)
        train_features = _model_features(
            training, tier=tier, target=target, vocabulary=vocabulary
        )
        validation_features = _model_features(
            validation, tier=tier, target=target, vocabulary=vocabulary
        )
        train_targets = _target_matrix(training, target)
        validation_targets = _target_matrix(validation, target)
        path_predictions = _ridge_path_predictions(
            train_features,
            train_targets,
            validation_features,
            RIDGE_GRID,
        )
        for penalty, predicted in path_predictions.items():
            scores[penalty].append(float(np.mean(np.square(predicted - validation_targets))))
    mean_scores = {penalty: float(np.mean(values)) for penalty, values in scores.items()}
    best_score = min(mean_scores.values())
    tolerance = 1.0e-12 * max(1.0, abs(best_score))
    eligible = [
        penalty
        for penalty, score in mean_scores.items()
        if score <= best_score + tolerance
    ]
    return float(max(eligible))


def select_lambdas(
    states: Sequence[ArtifactState], *, tier: str
) -> tuple[float, float]:
    return (
        _select_lambda(states, tier=tier, target="mask"),
        _select_lambda(states, tier=tier, target="delta"),
    )


def fit_fold_models(
    states: Sequence[ArtifactState],
    *,
    tier: str,
    lambda_mask: float,
    lambda_delta: float,
) -> FoldModels:
    vocabulary = _response_vocabulary(states)
    mask = fit_ridge(
        _model_features(states, tier=tier, target="mask", vocabulary=vocabulary),
        _target_matrix(states, "mask"),
        lambda_mask,
    )
    delta = fit_ridge(
        _model_features(states, tier=tier, target="delta", vocabulary=vocabulary),
        _target_matrix(states, "delta"),
        lambda_delta,
    )
    return FoldModels(
        mask=mask,
        delta=delta,
        response_codes=vocabulary,
        lambda_mask=float(lambda_mask),
        lambda_delta=float(lambda_delta),
    )


def predict_fold(
    models: FoldModels,
    states: Sequence[ArtifactState],
    *,
    tier: str,
    evaluation_kind: str,
    fold_partner: int | None,
) -> tuple[ReadoutPrediction, ...]:
    mask_features = _model_features(
        states, tier=tier, target="mask", vocabulary=models.response_codes
    )
    delta_features = _model_features(
        states, tier=tier, target="delta", vocabulary=models.response_codes
    )
    predicted_mask = models.mask.predict(mask_features)
    predicted_use = predicted_mask + models.delta.predict(delta_features)
    predictions = []
    for index, state in enumerate(states):
        probe_mask_action = int(np.argmax(predicted_mask[index]))
        probe_use_action = int(np.argmax(predicted_use[index]))
        oracle_mask_action = int(np.argmax(state.q_fit_mask))
        oracle_use_action = int(np.argmax(state.q_fit_use))
        predictions.append(
            ReadoutPrediction(
                state=state,
                feature_tier=tier,
                evaluation_kind=evaluation_kind,
                fold_partner=fold_partner,
                lambda_mask=models.lambda_mask,
                lambda_delta=models.lambda_delta,
                predicted_mask=predicted_mask[index],
                predicted_use=predicted_use[index],
                probe_mask_action=probe_mask_action,
                probe_use_action=probe_use_action,
                oracle_mask_action=oracle_mask_action,
                oracle_use_action=oracle_use_action,
                probe_lift=float(
                    state.q_evaluation_use[probe_use_action]
                    - state.q_evaluation_mask[probe_mask_action]
                ),
                oracle_lift=float(
                    state.q_evaluation_use[oracle_use_action]
                    - state.q_evaluation_mask[oracle_mask_action]
                ),
            )
        )
    return tuple(predictions)


def evaluate_lopo(
    states: Sequence[ArtifactState], *, tier: str
) -> LopoResult:
    partners = sorted({state.partner_index for state in states})
    if None in partners or len(partners) != 4:
        raise ValueError("The registered outer audit requires four fixed partners.")
    predictions: list[ReadoutPrediction] = []
    lambdas: dict[int, Mapping[str, float]] = {}
    for held_out_value in partners:
        held_out = int(held_out_value)
        training = [state for state in states if state.partner_index != held_out]
        evaluation = [state for state in states if state.partner_index == held_out]
        lambda_mask, lambda_delta = select_lambdas(training, tier=tier)
        models = fit_fold_models(
            training,
            tier=tier,
            lambda_mask=lambda_mask,
            lambda_delta=lambda_delta,
        )
        predictions.extend(
            predict_fold(
                models,
                evaluation,
                tier=tier,
                evaluation_kind="lopo",
                fold_partner=held_out,
            )
        )
        lambdas[held_out] = {
            "mask": lambda_mask,
            "delta": lambda_delta,
        }
    return LopoResult(tuple(predictions), lambdas)


def evaluate_within_partner(
    states: Sequence[ArtifactState],
    *,
    tier: str,
    fold_lambdas: Mapping[int, Mapping[str, float]],
) -> tuple[ReadoutPrediction, ...]:
    predictions: list[ReadoutPrediction] = []
    for partner in sorted(fold_lambdas):
        current = [state for state in states if state.partner_index == partner]
        registered = fold_lambdas[partner]
        models = fit_fold_models(
            current,
            tier=tier,
            lambda_mask=float(registered["mask"]),
            lambda_delta=float(registered["delta"]),
        )
        predictions.extend(
            predict_fold(
                models,
                current,
                tier=tier,
                evaluation_kind="within_partner_split_replica",
                fold_partner=partner,
            )
        )
    return tuple(predictions)


def evaluate_self_pair(
    fixed_states: Sequence[ArtifactState],
    self_states: Sequence[ArtifactState],
    *,
    tier: str,
) -> tuple[ReadoutPrediction, ...]:
    lambda_mask, lambda_delta = select_lambdas(fixed_states, tier=tier)
    models = fit_fold_models(
        fixed_states,
        tier=tier,
        lambda_mask=lambda_mask,
        lambda_delta=lambda_delta,
    )
    return predict_fold(
        models,
        self_states,
        tier=tier,
        evaluation_kind="self_pair_descriptive",
        fold_partner=None,
    )


def aggregate_predictions(
    predictions: Sequence[ReadoutPrediction],
) -> Mapping[str, Any]:
    if not predictions:
        raise ValueError("Cannot aggregate an empty prediction set.")
    partner_values = sorted(
        {prediction.state.partner_index for prediction in predictions},
        key=lambda value: -1 if value is None else int(value),
    )
    by_partner: dict[str, Mapping[str, float | int]] = {}
    for partner in partner_values:
        current = [
            prediction
            for prediction in predictions
            if prediction.state.partner_index == partner
        ]
        label = "self" if partner is None else str(int(partner))
        probe_discounted = [
            value.state.q_evaluation_use_discounted[value.probe_use_action]
            - value.state.q_evaluation_mask_discounted[value.probe_mask_action]
            for value in current
        ]
        oracle_discounted = [
            value.state.q_evaluation_use_discounted[value.oracle_use_action]
            - value.state.q_evaluation_mask_discounted[value.oracle_mask_action]
            for value in current
        ]
        probe_correct = [
            value.state.q_evaluation_use_correct_deliveries[
                value.probe_use_action
            ]
            - value.state.q_evaluation_mask_correct_deliveries[
                value.probe_mask_action
            ]
            for value in current
        ]
        probe_wrong = [
            value.state.q_evaluation_use_wrong_deliveries[
                value.probe_use_action
            ]
            - value.state.q_evaluation_mask_wrong_deliveries[
                value.probe_mask_action
            ]
            for value in current
        ]
        by_partner[label] = {
            "state_count": len(current),
            "probe_lift": float(np.mean([value.probe_lift for value in current])),
            "oracle_lift": float(np.mean([value.oracle_lift for value in current])),
            "tau_response": float(
                np.mean([value.state.tau_evaluation for value in current])
            ),
            "probe_lift_discounted": float(np.mean(probe_discounted)),
            "oracle_lift_discounted": float(np.mean(oracle_discounted)),
            "probe_correct_delivery_lift": float(np.mean(probe_correct)),
            "probe_wrong_delivery_lift": float(np.mean(probe_wrong)),
            "tau_correct_delivery_lift": float(
                np.mean(
                    [
                        value.state.tau_evaluation_correct_deliveries
                        for value in current
                    ]
                )
            ),
            "tau_wrong_delivery_lift": float(
                np.mean(
                    [
                        value.state.tau_evaluation_wrong_deliveries
                        for value in current
                    ]
                )
            ),
            "probe_mask_oracle_action_agreement": float(
                np.mean(
                    [
                        value.probe_mask_action == value.oracle_mask_action
                        for value in current
                    ]
                )
            ),
            "probe_use_oracle_action_agreement": float(
                np.mean(
                    [
                        value.probe_use_action == value.oracle_use_action
                        for value in current
                    ]
                )
            ),
            "probe_both_oracle_action_agreement": float(
                np.mean(
                    [
                        value.probe_mask_action == value.oracle_mask_action
                        and value.probe_use_action == value.oracle_use_action
                        for value in current
                    ]
                )
            ),
        }
    if None in partner_values:
        aggregate_probe = float(np.mean([value.probe_lift for value in predictions]))
        aggregate_oracle = float(np.mean([value.oracle_lift for value in predictions]))
        aggregate_tau = float(
            np.mean([value.state.tau_evaluation for value in predictions])
        )
    else:
        aggregate_probe = float(
            np.mean([float(value["probe_lift"]) for value in by_partner.values()])
        )
        aggregate_oracle = float(
            np.mean([float(value["oracle_lift"]) for value in by_partner.values()])
        )
        aggregate_tau = float(
            np.mean([float(value["tau_response"]) for value in by_partner.values()])
        )
    return {
        "state_count": len(predictions),
        "partner_weighting": "equal_partner_then_equal_state",
        "probe_lift": aggregate_probe,
        "oracle_lift": aggregate_oracle,
        "tau_response": aggregate_tau,
        "auxiliary": {
            name: float(
                np.mean([float(value[name]) for value in by_partner.values()])
            )
            for name in (
                "probe_lift_discounted",
                "oracle_lift_discounted",
                "probe_correct_delivery_lift",
                "probe_wrong_delivery_lift",
                "tau_correct_delivery_lift",
                "tau_wrong_delivery_lift",
                "probe_mask_oracle_action_agreement",
                "probe_use_oracle_action_agreement",
                "probe_both_oracle_action_agreement",
            )
        },
        "partners": by_partner,
    }


def prediction_rows(
    predictions: Iterable[ReadoutPrediction],
) -> list[Mapping[str, Any]]:
    rows = []
    for prediction in predictions:
        state = prediction.state
        rows.append(
            {
                "readout_version": READOUT_VERSION,
                "feature_tier": prediction.feature_tier,
                "evaluation_kind": prediction.evaluation_kind,
                "fold_partner": prediction.fold_partner,
                "trigger_id": state.trigger_id,
                "source": state.source,
                "partner_index": state.partner_index,
                "episode_index": state.episode_index,
                "episode_seed": state.episode_seed,
                "trigger_step": state.trigger_step,
                "trigger_action": state.trigger_action,
                "response_code": state.response_code,
                "lambda_mask": prediction.lambda_mask,
                "lambda_delta": prediction.lambda_delta,
                "predicted_mask_values": prediction.predicted_mask.tolist(),
                "predicted_use_values": prediction.predicted_use.tolist(),
                "fit_mask_values": state.q_fit_mask.tolist(),
                "fit_use_values": state.q_fit_use.tolist(),
                "evaluation_mask_values": state.q_evaluation_mask.tolist(),
                "evaluation_use_values": state.q_evaluation_use.tolist(),
                "probe_mask_action": prediction.probe_mask_action,
                "probe_use_action": prediction.probe_use_action,
                "oracle_mask_action": prediction.oracle_mask_action,
                "oracle_use_action": prediction.oracle_use_action,
                "probe_lift": prediction.probe_lift,
                "oracle_lift": prediction.oracle_lift,
                "probe_lift_discounted": float(
                    state.q_evaluation_use_discounted[
                        prediction.probe_use_action
                    ]
                    - state.q_evaluation_mask_discounted[
                        prediction.probe_mask_action
                    ]
                ),
                "oracle_lift_discounted": float(
                    state.q_evaluation_use_discounted[
                        prediction.oracle_use_action
                    ]
                    - state.q_evaluation_mask_discounted[
                        prediction.oracle_mask_action
                    ]
                ),
                "probe_correct_delivery_lift": float(
                    state.q_evaluation_use_correct_deliveries[
                        prediction.probe_use_action
                    ]
                    - state.q_evaluation_mask_correct_deliveries[
                        prediction.probe_mask_action
                    ]
                ),
                "probe_wrong_delivery_lift": float(
                    state.q_evaluation_use_wrong_deliveries[
                        prediction.probe_use_action
                    ]
                    - state.q_evaluation_mask_wrong_deliveries[
                        prediction.probe_mask_action
                    ]
                ),
                "probe_mask_oracle_action_agreement": (
                    prediction.probe_mask_action == prediction.oracle_mask_action
                ),
                "probe_use_oracle_action_agreement": (
                    prediction.probe_use_action == prediction.oracle_use_action
                ),
                "tau_response": state.tau_evaluation,
                "tau_response_fit": state.tau_fit,
                "tau_response_discounted": state.tau_evaluation_discounted,
                "tau_correct_delivery_lift": (
                    state.tau_evaluation_correct_deliveries
                ),
                "tau_wrong_delivery_lift": state.tau_evaluation_wrong_deliveries,
            }
        )
    return rows


def _resample_partner_strata(
    states: Sequence[ArtifactState], rng: np.random.Generator
) -> list[ArtifactState]:
    sampled: list[ArtifactState] = []
    partners = sorted(
        {
            int(state.partner_index)
            for state in states
            if state.partner_index is not None
        }
    )
    for partner in partners:
        current = [state for state in states if state.partner_index == partner]
        indexes = rng.integers(0, len(current), size=len(current))
        sampled.extend(current[int(index)] for index in indexes)
    return sampled


def bootstrap_lopo(
    states: Sequence[ArtifactState],
    *,
    tier: str,
    replicates: int,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Mapping[str, Any], ...]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive.")
    rng = np.random.default_rng(int(seed))
    rows: list[Mapping[str, Any]] = []
    for replica in range(int(replicates)):
        sampled = _resample_partner_strata(states, rng)
        result = evaluate_lopo(sampled, tier=tier)
        aggregate = aggregate_predictions(result.predictions)
        row: dict[str, Any] = {
            "readout_version": READOUT_VERSION,
            "feature_tier": tier,
            "bootstrap_replica": replica,
            "probe_lift": float(aggregate["probe_lift"]),
            "oracle_lift": float(aggregate["oracle_lift"]),
            "tau_response": float(aggregate["tau_response"]),
        }
        partners = aggregate["partners"]
        if not isinstance(partners, Mapping):
            raise AssertionError("Partner summaries are not a mapping.")
        for partner, values in partners.items():
            if not isinstance(values, Mapping):
                raise AssertionError("Partner summary is not a mapping.")
            row[f"partner_{int(partner):02d}_probe_lift"] = float(values["probe_lift"])
            row[f"partner_{int(partner):02d}_oracle_lift"] = float(values["oracle_lift"])
            row[f"partner_{int(partner):02d}_tau_response"] = float(values["tau_response"])
        rows.append(row)
        if (replica + 1) % 100 == 0 or replica + 1 == replicates:
            print(f"{tier} bootstrap: {replica + 1}/{replicates}")
    return tuple(rows)


def _interval(point: float, values: Sequence[float]) -> Mapping[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Bootstrap interval values must be finite and non-empty.")
    return {
        "point": float(point),
        "lcb95": float(np.quantile(array, 0.05, method="linear")),
        "ucb95": float(np.quantile(array, 0.95, method="linear")),
    }


def summarize_bootstrap(
    point: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    partners = point["partners"]
    if not isinstance(partners, Mapping):
        raise ValueError("Point estimate lacks partner summaries.")
    partner_intervals = {}
    for partner, values in partners.items():
        if not isinstance(values, Mapping):
            raise ValueError("Partner point estimate is invalid.")
        prefix = f"partner_{int(partner):02d}"
        partner_intervals[str(partner)] = {
            "probe_lift": _interval(
                float(values["probe_lift"]),
                [float(row[f"{prefix}_probe_lift"]) for row in rows],
            ),
            "oracle_lift": _interval(
                float(values["oracle_lift"]),
                [float(row[f"{prefix}_oracle_lift"]) for row in rows],
            ),
            "tau_response": _interval(
                float(values["tau_response"]),
                [float(row[f"{prefix}_tau_response"]) for row in rows],
            ),
        }
    return {
        "probe_lift": _interval(
            float(point["probe_lift"]),
            [float(row["probe_lift"]) for row in rows],
        ),
        "oracle_lift": _interval(
            float(point["oracle_lift"]),
            [float(row["oracle_lift"]) for row in rows],
        ),
        "tau_response": _interval(
            float(point["tau_response"]),
            [float(row["tau_response"]) for row in rows],
        ),
        "partners": partner_intervals,
    }


def classify_verdict(
    probe: Mapping[str, float],
    oracle: Mapping[str, float],
    partner_probe: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Any]:
    positive_partner = any(float(value["lcb95"]) > 0.0 for value in partner_probe.values())
    negative_partner = any(float(value["ucb95"]) <= 0.0 for value in partner_probe.values())
    conflict = bool(positive_partner and negative_partner)
    if float(oracle["ucb95"]) <= 0.0:
        verdict = "NO-GO"
        reason = "oracle_ucb_nonpositive"
    elif float(oracle["lcb95"]) > 0.0 and float(probe["ucb95"]) <= 0.0:
        verdict = "NO-GO"
        reason = "oracle_positive_readout_ucb_nonpositive"
    elif float(probe["lcb95"]) > 0.0 and not conflict:
        verdict = "GO"
        reason = "readout_lcb_positive_without_resolved_partner_conflict"
    else:
        verdict = "INCONCLUSIVE"
        reason = "interval_or_partner_heterogeneity_not_resolved"
    return {
        "verdict": verdict,
        "reason": reason,
        "significant_partner_conflict": conflict,
    }


def _markdown_report(summary: Mapping[str, Any]) -> str:
    primary = summary["feature_tiers"]["history"]
    verdict = summary["primary_verdict"]
    probe = primary["bootstrap"]["probe_lift"]
    oracle = primary["bootstrap"]["oracle_lift"]
    tau = primary["bootstrap"]["tau_response"]
    lines = [
        "# Path C V4.4 artifact-only 真实回报读出审计",
        "",
        f"- 读出版本：`{READOUT_VERSION}`",
        f"- 主裁决：**{verdict['verdict']}**（`{verdict['reason']}`）",
        "- 统计范围：当前四个固定伙伴与当前 V4.4 触发/候选分布；不是跨训练运行总体推断。",
        "- 主尺度：evaluation-half 原始剩余回报；伙伴等权。",
        "",
        "## 主结果",
        "",
        "| 量 | 点估计 | LCB95 | UCB95 |",
        "|---|---:|---:|---:|",
        f"| `L_probe` | {probe['point']:.6f} | {probe['lcb95']:.6f} | {probe['ucb95']:.6f} |",
        f"| `L_oracle` | {oracle['point']:.6f} | {oracle['lcb95']:.6f} | {oracle['ucb95']:.6f} |",
        f"| `tau_response` | {tau['point']:.6f} | {tau['lcb95']:.6f} | {tau['ucb95']:.6f} |",
        "",
        "## 解释边界",
        "",
        "该审计比较使用合法历史与可见注册回应的选动作读出器，和保留官方历史但屏蔽注册回应的同构被动读出器。它只能裁决当前 V4.4 触发动作、回应编码和六动作候选集合，不能否定 Test Time Simple 中所有可能的主动探查。",
        "",
        "`119.65` 的表观策略库差距仍由四个同 checkpoint 对角格完全解释；本审计不改变该结论，也不把注册回应价值与跨运行策略兼容性混为同一科学对象。",
        "",
    ]
    return "\n".join(lines)


def _ensure_independent_output(artifact: Path, output: Path) -> None:
    if output == artifact or artifact in output.parents:
        raise ValueError("Readout output must be outside the frozen artifact directory.")
    existing = []
    if output.is_dir():
        existing = [child for child in output.iterdir() if child.name != "logs"]
    if existing:
        raise RuntimeError("Readout output directory is not empty.")


def analyze_artifact(
    artifact_directory: str | Path,
    output_directory: str | Path,
    *,
    expectations: ArtifactExpectations = REGISTERED_EXPECTATIONS,
) -> Mapping[str, Any]:
    artifact = Path(artifact_directory).resolve()
    output = Path(output_directory).resolve()
    _ensure_independent_output(artifact, output)
    output.mkdir(parents=True, exist_ok=True)
    before_hashes = hash_registered_inputs(artifact)
    before_implementation = implementation_provenance()
    states, split_rows, artifact_metadata = load_registered_artifact(
        artifact, expectations=expectations
    )
    fixed_states = tuple(state for state in states if state.partner_index is not None)
    self_states = tuple(state for state in states if state.partner_index is None)
    configuration = {
        "readout_version": READOUT_VERSION,
        "split": {
            "algorithm": "sha256_rank_within_trigger",
            "version": SPLIT_VERSION,
            "salt": SPLIT_SALT,
            "fit_replicas": expectations.split_replicas,
            "evaluation_replicas": expectations.split_replicas,
            "pairing_scope": "all_estimands_branches_and_actions_for_replica",
        },
        "models": {
            "mask": "multi_output_ridge_q_mask_x_a",
            "delta": "multi_output_ridge_delta_x_response_a",
            "q_use": "q_mask_plus_delta",
            "candidate_action_representation": (
                "six_action_output_index_with_action_specific_coefficients"
            ),
            "response_unknown_policy": "training_fold_vocabulary_plus_UNK",
            "ridge_grid": list(RIDGE_GRID),
            "lambda_selection": (
                "inner_leave_one_training_partner_out_equal_partner_six_action_mse"
            ),
            "lambda_tie_break": "largest_penalty",
            "action_tie_break": "smallest_action_index",
        },
        "features": {
            "tiers": list(FEATURE_TIERS),
            "state": (
                "agent0_official_post_observation_normalized_trigger_step_"
                "trigger_action"
            ),
            "history_additions": list(LEGAL_HISTORY_FIELDS),
            "excluded": list(EXCLUDED_MODEL_FIELDS),
        },
        "evaluation": {
            "outer_folds": "four_partner_LOPO",
            "partner_weighting": "equal_partner_then_equal_state",
            "primary_return": "remaining_raw_return",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": expectations.bootstrap_replicates,
            "bootstrap_unit": "trigger_state_within_fixed_partner_stratum",
            "pipeline_refit_per_bootstrap": True,
            "bounds": "one_sided_95_percentile",
        },
        "implementation": before_implementation,
    }
    all_prediction_rows: list[Mapping[str, Any]] = []
    all_bootstrap_rows: list[Mapping[str, Any]] = []
    tier_summaries: dict[str, Any] = {}
    for tier in FEATURE_TIERS:
        lopo = evaluate_lopo(fixed_states, tier=tier)
        point = aggregate_predictions(lopo.predictions)
        within = evaluate_within_partner(
            fixed_states, tier=tier, fold_lambdas=lopo.fold_lambdas
        )
        self_predictions = evaluate_self_pair(
            fixed_states, self_states, tier=tier
        )
        bootstrap_rows = bootstrap_lopo(
            fixed_states,
            tier=tier,
            replicates=expectations.bootstrap_replicates,
            seed=BOOTSTRAP_SEED,
        )
        bootstrap_summary = summarize_bootstrap(point, bootstrap_rows)
        tier_summaries[tier] = {
            "lopo": point,
            "fold_lambdas": {
                str(partner): dict(values)
                for partner, values in lopo.fold_lambdas.items()
            },
            "within_partner_split_replica": aggregate_predictions(within),
            "self_pair_descriptive": aggregate_predictions(self_predictions),
            "bootstrap": bootstrap_summary,
        }
        all_prediction_rows.extend(prediction_rows(lopo.predictions))
        all_prediction_rows.extend(prediction_rows(within))
        all_prediction_rows.extend(prediction_rows(self_predictions))
        all_bootstrap_rows.extend(bootstrap_rows)

    primary_bootstrap = tier_summaries["history"]["bootstrap"]
    partner_probe = {
        partner: values["probe_lift"]
        for partner, values in primary_bootstrap["partners"].items()
    }
    verdict = classify_verdict(
        primary_bootstrap["probe_lift"],
        primary_bootstrap["oracle_lift"],
        partner_probe,
    )
    summary = {
        "readout_version": READOUT_VERSION,
        "run_kind": "development_artifact_gate",
        "scientific_readout_allowed": False,
        "claim_scope": "conditional_on_current_four_partner_trigger_candidate_panel",
        "primary_return": "remaining_raw_return",
        "primary_feature_tier": "history",
        "partner_weighting": "equal_partner_then_equal_state",
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": expectations.bootstrap_replicates,
            "resampling_unit": "trigger_state_within_fixed_partner_stratum",
            "pipeline_refit_per_replicate": True,
            "bounds": "one_sided_95_percentile",
        },
        "configuration": configuration,
        "artifact": artifact_metadata,
        "feature_tiers": tier_summaries,
        "primary_verdict": verdict,
        "scope_guardrails": {
            "same_checkpoint_matrix_gap_119_65_is_not_transferable_opportunity": True,
            "does_not_rule_out_all_active_probes_on_test_time_simple": True,
            "does_not_estimate_cross_training_run_population_effect": True,
        },
    }
    write_parquet(output / "split_manifest.parquet", list(split_rows))
    write_parquet(output / "state_readout.parquet", all_prediction_rows)
    write_parquet(output / "bootstrap.parquet", all_bootstrap_rows)
    write_json(
        output / "feature_schema.json",
        {
            "feature_names": artifact_metadata["feature_names"],
            "candidate_action_representation": configuration["models"][
                "candidate_action_representation"
            ],
            "response_unknown_policy": configuration["models"][
                "response_unknown_policy"
            ],
        },
    )
    write_json(output / "configuration.json", configuration)
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(_markdown_report(summary), encoding="utf-8")
    after_hashes = hash_registered_inputs(artifact)
    if after_hashes != before_hashes:
        raise RuntimeError("A frozen artifact input changed during readout.")
    after_implementation = implementation_provenance()
    if after_implementation != before_implementation:
        raise RuntimeError("The readout implementation changed during execution.")
    command = (
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "python -m experiments.overcooked_v2.path_c audit-artifact-readout "
        '--artifact-directory "$ARTIFACT_DIRECTORY" --output "$OUTPUT_DIRECTORY"'
    )
    write_json(
        output / "run_metadata.json",
        {
            "readout_version": READOUT_VERSION,
            "artifact_directory": str(artifact),
            "output_directory": "$OUTPUT_DIRECTORY",
            "input_sha256": before_hashes,
            "input_sha256_before": before_hashes,
            "input_sha256_after": after_hashes,
            "frozen_inputs_unchanged": True,
            "implementation_before": before_implementation,
            "implementation_after": after_implementation,
            "implementation_unchanged": True,
            "expectations": {
                "trigger_count": expectations.trigger_count,
                "fixed_trigger_count": expectations.fixed_trigger_count,
                "self_trigger_count": expectations.self_trigger_count,
                "partner_count": expectations.partner_count,
                "replicas": expectations.replicas,
                "action_count": expectations.action_count,
                "episode_steps": expectations.episode_steps,
                "continuation_row_count": expectations.continuation_row_count,
                "bootstrap_replicates": expectations.bootstrap_replicates,
            },
            "reproduction_command": command,
        },
    )
    output_hashes = {
        name: _sha256(output / name) for name in DETERMINISTIC_OUTPUT_FILES
    }
    write_json(
        output / "output_sha256.json",
        {
            "readout_version": READOUT_VERSION,
            "sha256": output_hashes,
            "excluded_from_self_hash": ["output_sha256.json", "logs/"],
        },
    )
    print(f"Complete artifact-only readout: {output}")
    return summary


def run_artifact_value_readout(args: Any) -> None:
    analyze_artifact(
        args.artifact_directory,
        args.output,
        expectations=REGISTERED_EXPECTATIONS,
    )


__all__ = [
    "ArtifactExpectations",
    "ArtifactState",
    "BOOTSTRAP_SEED",
    "READOUT_VERSION",
    "REGISTERED_EXPECTATIONS",
    "RIDGE_GRID",
    "aggregate_predictions",
    "analyze_artifact",
    "bootstrap_lopo",
    "classify_verdict",
    "deterministic_replica_split",
    "encode_response_codes",
    "evaluate_lopo",
    "extract_legal_features",
    "fit_ridge",
    "hash_registered_inputs",
    "load_registered_artifact",
    "run_artifact_value_readout",
]
