from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.overcooked_v2.path_c_split import (
    SPLIT_ROLES,
    SplitManifestV1,
)


VALUE_CLASS_SCHEMA_VERSION = "path_c_cross_fitted_value_classes_v1"
ECOLOGICAL_VALUE_CLASS_SCHEMA_VERSION_V2 = (
    "path_c_cross_fitted_ecological_value_classes_v2"
)
ECOLOGICAL_VALUE_OUTCOME_NAME = "held_out_episode_return"
_ECOLOGICAL_ASSIGNMENT_KEYS_V2 = frozenset({
    "episode_uid",
    "fold_id",
    "held_out_episode_return",
    "public_context_stratum",
    "value_estimate",
    "value_standard_error",
    "estimation_sample_size",
    "class_id",
    "distance_to_nearest_threshold",
})
_ECOLOGICAL_FOLD_BOUND_KEYS_V2 = frozenset({"fold_id", "values"})
_ARTIFACT_KEYS = frozenset({
    "schema_version",
    "split_manifest_sha256",
    "role",
    "outcome_name",
    "n_classes",
    "quantile_method",
    "folds",
    "assignments",
    "sha256",
})
_FOLD_KEYS = frozenset({
    "schema_version",
    "role",
    "fold_id",
    "fit_episode_uids",
    "apply_episode_uids",
    "bin_upper_bounds",
    "outcome_name",
})
_ASSIGNMENT_KEYS = frozenset({
    "episode_uid",
    "fold_id",
    "held_out_episode_return",
    "class_id",
    "distance_to_nearest_threshold",
})


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    observed = set(payload)
    missing = sorted(expected.difference(observed))
    unknown = sorted(observed.difference(expected))
    if missing or unknown:
        raise ValueError(f"{name} key mismatch; missing={missing}, unknown={unknown}.")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _strict_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _strict_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string.")
    return value


def _strict_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


@dataclass(frozen=True, slots=True)
class ValueClassFoldV1:
    role: str
    fold_id: int
    fit_episode_uids: tuple[str, ...]
    apply_episode_uids: tuple[str, ...]
    bin_upper_bounds: tuple[float, ...]
    outcome_name: str
    schema_version: str = VALUE_CLASS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VALUE_CLASS_SCHEMA_VERSION:
            raise ValueError("Unknown cross-fitted value-class schema version.")
        if not str(self.role).strip() or not str(self.outcome_name).strip():
            raise ValueError("Value-class role and outcome_name are required.")
        if self.role not in SPLIT_ROLES:
            raise ValueError("Value-class role is not registered by SplitManifestV1.")
        if self.outcome_name != ECOLOGICAL_VALUE_OUTCOME_NAME:
            raise ValueError(
                "Ecological value classes must use held_out_episode_return."
            )
        _strict_integer(self.fold_id, "Value-class fold_id")
        for field_name, episode_uids in (
            ("fit_episode_uids", self.fit_episode_uids),
            ("apply_episode_uids", self.apply_episode_uids),
        ):
            if any(
                not isinstance(episode_uid, str)
                or not episode_uid
                or episode_uid != episode_uid.strip()
                for episode_uid in episode_uids
            ):
                raise ValueError(
                    f"Value-class {field_name} must contain non-empty, "
                    "whitespace-trimmed strings."
                )
            if tuple(episode_uids) != tuple(sorted(episode_uids)):
                raise ValueError(
                    f"Value-class {field_name} must be canonically sorted."
                )
        if len(self.fit_episode_uids) != len(set(self.fit_episode_uids)) or len(
            self.apply_episode_uids
        ) != len(set(self.apply_episode_uids)):
            raise ValueError("Value-class fold episode lists must be unique.")
        if set(self.fit_episode_uids).intersection(self.apply_episode_uids):
            raise ValueError("Value-class fit and apply episodes must be disjoint.")
        if not self.fit_episode_uids or not self.apply_episode_uids:
            raise ValueError("Every value-class fold needs fit and apply episodes.")
        if any(not math.isfinite(value) for value in self.bin_upper_bounds):
            raise ValueError("Value-class thresholds must be finite.")
        if any(
            right <= left
            for left, right in zip(
                self.bin_upper_bounds, self.bin_upper_bounds[1:], strict=False
            )
        ):
            raise ValueError("Value-class thresholds must be strictly increasing.")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "fold_id": int(self.fold_id),
            "fit_episode_uids": list(self.fit_episode_uids),
            "apply_episode_uids": list(self.apply_episode_uids),
            "bin_upper_bounds": list(map(float, self.bin_upper_bounds)),
            "outcome_name": self.outcome_name,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ValueClassFoldV1":
        if not isinstance(payload, Mapping):
            raise TypeError("Value-class fold must be a mapping.")
        _require_exact_keys(payload, _FOLD_KEYS, "Value-class fold")
        return cls(
            role=_strict_text(payload["role"], "Value-class fold role"),
            fold_id=_strict_integer(payload["fold_id"], "Value-class fold_id"),
            fit_episode_uids=tuple(
                _strict_text(value, "Value-class fit episode_uid")
                for value in _strict_sequence(
                    payload["fit_episode_uids"],
                    "Value-class fit_episode_uids",
                )
            ),
            apply_episode_uids=tuple(
                _strict_text(value, "Value-class apply episode_uid")
                for value in _strict_sequence(
                    payload["apply_episode_uids"],
                    "Value-class apply_episode_uids",
                )
            ),
            bin_upper_bounds=tuple(
                _finite_number(value, "Value-class threshold")
                for value in _strict_sequence(
                    payload["bin_upper_bounds"],
                    "Value-class bin_upper_bounds",
                )
            ),
            outcome_name=_strict_text(
                payload["outcome_name"], "Value-class fold outcome_name"
            ),
            schema_version=_strict_text(
                payload["schema_version"], "Value-class fold schema_version"
            ),
        )


@dataclass(frozen=True, slots=True)
class ValueClassAssignmentV1:
    episode_uid: str
    fold_id: int
    held_out_episode_return: float
    class_id: int
    distance_to_nearest_threshold: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_uid, str)
            or not self.episode_uid
            or self.episode_uid != self.episode_uid.strip()
        ):
            raise ValueError(
                "Value-class assignment episode_uid must be a non-empty, "
                "whitespace-trimmed string."
            )
        _strict_integer(self.fold_id, "Value-class assignment fold_id")
        _finite_number(
            self.held_out_episode_return,
            "Value-class assignment held_out_episode_return",
        )
        _strict_integer(self.class_id, "Value-class assignment class_id")
        distance = _finite_number(
            self.distance_to_nearest_threshold,
            "Value-class assignment distance_to_nearest_threshold",
        )
        if distance < 0.0:
            raise ValueError("Value-class threshold distance must be non-negative.")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "episode_uid": self.episode_uid,
            "fold_id": int(self.fold_id),
            "held_out_episode_return": float(self.held_out_episode_return),
            "class_id": int(self.class_id),
            "distance_to_nearest_threshold": float(
                self.distance_to_nearest_threshold
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ValueClassAssignmentV1":
        if not isinstance(payload, Mapping):
            raise TypeError("Value-class assignment must be a mapping.")
        _require_exact_keys(payload, _ASSIGNMENT_KEYS, "Value-class assignment")
        return cls(
            episode_uid=_strict_text(
                payload["episode_uid"], "Value-class assignment episode_uid"
            ),
            fold_id=_strict_integer(
                payload["fold_id"], "Value-class assignment fold_id"
            ),
            held_out_episode_return=_finite_number(
                payload["held_out_episode_return"],
                "Value-class assignment held_out_episode_return",
            ),
            class_id=_strict_integer(
                payload["class_id"], "Value-class assignment class_id"
            ),
            distance_to_nearest_threshold=_finite_number(
                payload["distance_to_nearest_threshold"],
                "Value-class assignment distance_to_nearest_threshold",
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossFittedValueClassesV1:
    folds: tuple[ValueClassFoldV1, ...]
    assignments: tuple[ValueClassAssignmentV1, ...]
    split_manifest_sha256: str
    role: str
    outcome_name: str
    n_classes: int
    quantile_method: str = "linear"
    schema_version: str = VALUE_CLASS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "folds", tuple(self.folds))
        object.__setattr__(self, "assignments", tuple(self.assignments))
        if self.schema_version != VALUE_CLASS_SCHEMA_VERSION:
            raise ValueError("Unknown cross-fitted value-class schema version.")
        if self.role not in SPLIT_ROLES:
            raise ValueError("Value-class artifact role is not registered.")
        if self.outcome_name != ECOLOGICAL_VALUE_OUTCOME_NAME:
            raise ValueError(
                "Ecological value classes must use held_out_episode_return."
            )
        if not isinstance(self.split_manifest_sha256, str) or len(
            self.split_manifest_sha256
        ) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.split_manifest_sha256
        ):
            raise ValueError("Value-class artifact requires a SHA-256 split binding.")
        _strict_integer(self.n_classes, "Value-class n_classes", minimum=2)
        if self.quantile_method != "linear":
            raise ValueError("Ecological value-class quantile_method must be linear.")
        if not self.folds or not self.assignments:
            raise ValueError("Value-class artifact needs folds and assignments.")
        episode_uids = [item.episode_uid for item in self.assignments]
        if len(episode_uids) != len(set(episode_uids)):
            raise ValueError("Value-class assignments repeat an episode_uid.")
        if any(item.class_id >= int(self.n_classes) for item in self.assignments):
            raise ValueError("A cross-fitted value-class id is out of range.")
        fold_ids = [fold.fold_id for fold in self.folds]
        if len(fold_ids) != len(set(fold_ids)):
            raise ValueError("Value-class artifact repeats a fold_id.")
        if tuple(fold_ids) != tuple(sorted(fold_ids)):
            raise ValueError("Value-class artifact folds must be canonically sorted.")
        if tuple(episode_uids) != tuple(sorted(episode_uids)):
            raise ValueError(
                "Value-class artifact assignments must be canonically sorted."
            )
        all_episodes = set(episode_uids)
        assignments_by_fold: dict[int, set[str]] = {}
        for item in self.assignments:
            assignments_by_fold.setdefault(int(item.fold_id), set()).add(
                item.episode_uid
            )
        if set(assignments_by_fold) != set(fold_ids):
            raise ValueError("Value-class fold records do not cover all assignments.")
        assignment_by_episode = self.assignment_by_episode
        for fold in self.folds:
            if fold.role != self.role or fold.outcome_name != self.outcome_name:
                raise ValueError("Value-class fold changed artifact role or outcome.")
            if len(fold.bin_upper_bounds) != int(self.n_classes) - 1:
                raise ValueError("Value-class fold threshold count changed n_classes.")
            apply = set(fold.apply_episode_uids)
            fit = set(fold.fit_episode_uids)
            if apply != assignments_by_fold[int(fold.fold_id)]:
                raise ValueError("Value-class fold apply episodes mismatch assignments.")
            if fit != all_episodes.difference(apply):
                raise ValueError("Value-class fold fit episodes are not out-of-fold.")
            for episode_uid in fold.apply_episode_uids:
                assignment = assignment_by_episode[episode_uid]
                expected_class = int(np.digitize(
                    assignment.held_out_episode_return,
                    fold.bin_upper_bounds,
                    right=True,
                ))
                expected_distance = min(
                    abs(assignment.held_out_episode_return - threshold)
                    for threshold in fold.bin_upper_bounds
                )
                if assignment.class_id != expected_class:
                    raise ValueError(
                        "Value-class assignment does not match its fold thresholds."
                    )
                if not math.isclose(
                    assignment.distance_to_nearest_threshold,
                    expected_distance,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        "Value-class threshold distance differs from its fold."
                    )

    @property
    def class_ids(self) -> np.ndarray:
        return np.asarray(
            [item.class_id for item in self.assignments], dtype=np.int16
        )

    @property
    def fold_ids(self) -> np.ndarray:
        return np.asarray(
            [item.fold_id for item in self.assignments], dtype=np.int16
        )

    @property
    def assignment_by_episode(self) -> dict[str, ValueClassAssignmentV1]:
        return {item.episode_uid: item for item in self.assignments}

    def core_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_manifest_sha256": self.split_manifest_sha256,
            "role": self.role,
            "outcome_name": self.outcome_name,
            "n_classes": int(self.n_classes),
            "quantile_method": self.quantile_method,
            "folds": [fold.to_mapping() for fold in self.folds],
            "assignments": [
                assignment.to_mapping() for assignment in self.assignments
            ],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.core_payload())

    def to_mapping(self) -> dict[str, Any]:
        return {**self.core_payload(), "sha256": self.sha256}

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        split_manifest: SplitManifestV1,
    ) -> "CrossFittedValueClassesV1":
        if not isinstance(payload, Mapping):
            raise TypeError("Value-class artifact must be a mapping.")
        _require_exact_keys(payload, _ARTIFACT_KEYS, "Value-class artifact")
        if not isinstance(split_manifest, SplitManifestV1):
            raise TypeError("split_manifest must be SplitManifestV1.")
        artifact = cls(
            folds=tuple(ValueClassFoldV1.from_mapping(item) for item in (
                _strict_sequence(payload["folds"], "Value-class folds")
            )),
            assignments=tuple(
                ValueClassAssignmentV1.from_mapping(item)
                for item in _strict_sequence(
                    payload["assignments"], "Value-class assignments"
                )
            ),
            split_manifest_sha256=_strict_text(
                payload["split_manifest_sha256"],
                "Value-class split_manifest_sha256",
            ),
            role=_strict_text(payload["role"], "Value-class role"),
            outcome_name=_strict_text(
                payload["outcome_name"], "Value-class outcome_name"
            ),
            n_classes=_strict_integer(
                payload["n_classes"], "Value-class n_classes", minimum=2
            ),
            quantile_method=_strict_text(
                payload["quantile_method"], "Value-class quantile_method"
            ),
            schema_version=_strict_text(
                payload["schema_version"], "Value-class schema_version"
            ),
        )
        if artifact.split_manifest_sha256 != split_manifest.sha256:
            raise ValueError("Value-class artifact changed the split manifest.")
        if payload["sha256"] != artifact.sha256:
            raise ValueError("Value-class artifact SHA-256 does not match its content.")
        rebuilt = fit_cross_fitted_value_classes(
            [item.held_out_episode_return for item in artifact.assignments],
            [item.episode_uid for item in artifact.assignments],
            split_manifest=split_manifest,
            role=artifact.role,
            outcome_name=artifact.outcome_name,
            n_classes=artifact.n_classes,
        )
        if rebuilt.core_payload() != artifact.core_payload():
            raise ValueError(
                "Value-class artifact does not match the frozen fold assignment, "
                "out-of-fold thresholds, and ecological outcomes."
            )
        return artifact

    def labels_for_rows(
        self,
        episode_uid: Sequence[Any],
        *,
        held_out_episode_return: Sequence[float] | None = None,
        require_complete: bool = True,
    ) -> np.ndarray:
        episodes = np.asarray(episode_uid).astype(str).reshape(-1)
        observed = set(episodes.tolist())
        registered = set(self.assignment_by_episode)
        missing = sorted(observed.difference(registered))
        extra = sorted(registered.difference(observed)) if require_complete else []
        if missing or extra:
            raise ValueError(
                "Value-class artifact episode coverage mismatch; "
                f"missing={missing}, extra={extra}."
            )
        if held_out_episode_return is not None:
            outcomes = np.asarray(
                held_out_episode_return, dtype=np.float64
            ).reshape(-1)
            if outcomes.shape != episodes.shape or not np.isfinite(outcomes).all():
                raise ValueError("Ecological outcomes must be finite and row-aligned.")
            for episode in sorted(observed):
                rows = outcomes[episodes == episode]
                expected = self.assignment_by_episode[
                    episode
                ].held_out_episode_return
                if not np.allclose(rows, expected, rtol=0.0, atol=1.0e-12):
                    raise ValueError(
                        "Value-class artifact outcome differs from dataset return."
                    )
        return np.asarray(
            [self.assignment_by_episode[item].class_id for item in episodes],
            dtype=np.int16,
        )


def fit_cross_fitted_value_classes(
    held_out_value_outcome: Sequence[float],
    episode_uid: Sequence[Any],
    *,
    split_manifest: SplitManifestV1,
    role: str,
    outcome_name: str,
    n_classes: int,
) -> CrossFittedValueClassesV1:
    """Discretize an ecological value outcome without using mechanism labels.

    Thresholds for each fold are fitted only on other episodes from the same
    registered role. The API intentionally has no mechanism, identity, or style
    argument, so those nuisance labels cannot silently define a value class.
    """

    values = np.asarray(held_out_value_outcome, dtype=np.float64).reshape(-1)
    raw_episodes = np.asarray(episode_uid, dtype=object).reshape(-1)
    if any(not isinstance(value, (str, np.str_)) for value in raw_episodes):
        raise TypeError("episode_uid must contain strings.")
    episodes = raw_episodes.astype(str)
    if any(not episode or episode != episode.strip() for episode in episodes):
        raise ValueError(
            "episode_uid must contain non-empty, whitespace-trimmed strings."
        )
    if values.shape != episodes.shape or values.size == 0:
        raise ValueError("Value outcomes and episode ids must be non-empty and aligned.")
    if not bool(np.isfinite(values).all()):
        raise ValueError("Held-out value outcomes must be finite.")
    _strict_text(role, "Registered role")
    _strict_text(outcome_name, "Registered outcome_name")
    if not isinstance(split_manifest, SplitManifestV1):
        raise TypeError("split_manifest must be SplitManifestV1.")
    if role not in SPLIT_ROLES:
        raise ValueError("Registered role is not present in SplitManifestV1.")
    if outcome_name != ECOLOGICAL_VALUE_OUTCOME_NAME:
        raise ValueError(
            "Ecological value classes must use held_out_episode_return."
        )
    classes = _strict_integer(n_classes, "n_classes", minimum=2)
    folds_count = int(split_manifest.cross_fit_folds)
    unique_episodes = tuple(sorted(np.unique(episodes).tolist()))
    if len(unique_episodes) < folds_count:
        raise ValueError("There are fewer independent episodes than cross-fit folds.")

    episode_values: dict[str, float] = {}
    for episode in unique_episodes:
        observed_values = values[episodes == episode]
        if not np.allclose(
            observed_values,
            observed_values[0],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Every episode must carry one held_out_episode_return."
            )
        episode_values[episode] = float(observed_values[0])
    episode_fold = {
        episode: int(split_manifest.cross_fit_fold(role, episode))
        for episode in unique_episodes
    }
    observed_folds = set(episode_fold.values())
    expected_folds = set(range(folds_count))
    if observed_folds != expected_folds:
        raise ValueError(
            "Dataset does not populate every frozen cross-fit fold; "
            f"missing={sorted(expected_folds.difference(observed_folds))}."
        )

    fold_records: list[ValueClassFoldV1] = []
    assignments: list[ValueClassAssignmentV1] = []
    quantiles = np.arange(1, classes, dtype=np.float64) / float(classes)
    for fold_id in range(folds_count):
        apply_episodes = tuple(
            episode for episode in unique_episodes
            if episode_fold[episode] == fold_id
        )
        fit_episodes = tuple(
            episode for episode in unique_episodes
            if episode_fold[episode] != fold_id
        )
        if not apply_episodes or not fit_episodes:
            raise ValueError("Cross-fitting produced an empty fit or apply fold.")
        fit_values = np.asarray(
            [episode_values[episode] for episode in fit_episodes],
            dtype=np.float64,
        )
        raw_bounds = np.quantile(fit_values, quantiles, method="linear")
        bounds = tuple(float(value) for value in np.unique(raw_bounds).tolist())
        if len(bounds) != classes - 1:
            raise ValueError(
                "Held-out value outcomes do not support the requested number of "
                "strictly ordered classes."
            )
        for episode in apply_episodes:
            assignments.append(ValueClassAssignmentV1(
                episode_uid=episode,
                fold_id=fold_id,
                held_out_episode_return=episode_values[episode],
                class_id=int(np.digitize(
                    episode_values[episode], bounds, right=True
                )),
                distance_to_nearest_threshold=float(min(
                    abs(episode_values[episode] - threshold)
                    for threshold in bounds
                )),
            ))
        fold_records.append(
            ValueClassFoldV1(
                role=str(role),
                fold_id=fold_id,
                fit_episode_uids=fit_episodes,
                apply_episode_uids=apply_episodes,
                bin_upper_bounds=bounds,
                outcome_name=str(outcome_name),
            )
        )
    return CrossFittedValueClassesV1(
        folds=tuple(fold_records),
        assignments=tuple(sorted(assignments, key=lambda item: item.episode_uid)),
        split_manifest_sha256=split_manifest.sha256,
        role=str(role),
        outcome_name=str(outcome_name),
        n_classes=classes,
        quantile_method="linear",
    )


def cluster_stratified_permutation(
    cluster_values: Sequence[Any],
    episode_uid: Sequence[Any],
    registered_stratum: Sequence[Any],
    *,
    seed: int,
) -> np.ndarray:
    """Permute labels at episode level within frozen analysis strata."""

    values = np.asarray(cluster_values)
    episodes = np.asarray(episode_uid).astype(str)
    strata = np.asarray(registered_stratum).astype(str)
    if len({values.shape[0], episodes.shape[0], strata.shape[0]}) != 1:
        raise ValueError("Permutation inputs must have equal row counts.")
    episode_records: dict[str, tuple[Any, str, np.ndarray]] = {}
    for episode in sorted(np.unique(episodes).tolist()):
        rows = np.flatnonzero(episodes == episode)
        episode_values = np.unique(values[rows])
        episode_strata = np.unique(strata[rows])
        if episode_values.size != 1 or episode_strata.size != 1:
            raise ValueError(
                "Each episode must carry one cluster label and one registered stratum."
            )
        episode_records[episode] = (episode_values[0], str(episode_strata[0]), rows)

    rng = np.random.default_rng(int(seed))
    output = values.copy()
    for stratum in sorted({record[1] for record in episode_records.values()}):
        keys = [
            key for key, record in episode_records.items() if record[1] == stratum
        ]
        shuffled = rng.permutation(
            np.asarray([episode_records[key][0] for key in keys], dtype=values.dtype)
        )
        for key, replacement in zip(keys, shuffled.tolist(), strict=True):
            output[episode_records[key][2]] = replacement
    return output


@dataclass(frozen=True, slots=True)
class EcologicalValueAssignmentV2:
    episode_uid: str
    fold_id: int
    held_out_episode_return: float
    public_context_stratum: str
    value_estimate: float
    value_standard_error: float
    estimation_sample_size: int
    class_id: int
    distance_to_nearest_threshold: float

    def __post_init__(self) -> None:
        _strict_text(self.episode_uid, "Ecological assignment episode_uid")
        _strict_text(
            self.public_context_stratum,
            "Ecological assignment public_context_stratum",
        )
        _strict_integer(self.fold_id, "Ecological assignment fold_id")
        _strict_integer(self.class_id, "Ecological assignment class_id")
        _strict_integer(
            self.estimation_sample_size,
            "Ecological assignment estimation_sample_size",
            minimum=2,
        )
        for name in (
            "held_out_episode_return",
            "value_estimate",
            "value_standard_error",
            "distance_to_nearest_threshold",
        ):
            value = _finite_number(getattr(self, name), f"Ecological assignment {name}")
            if name in {"value_standard_error", "distance_to_nearest_threshold"} and value < 0.0:
                raise ValueError(f"Ecological assignment {name} must be non-negative.")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "episode_uid": self.episode_uid,
            "fold_id": int(self.fold_id),
            "held_out_episode_return": float(self.held_out_episode_return),
            "public_context_stratum": self.public_context_stratum,
            "value_estimate": float(self.value_estimate),
            "value_standard_error": float(self.value_standard_error),
            "estimation_sample_size": int(self.estimation_sample_size),
            "class_id": int(self.class_id),
            "distance_to_nearest_threshold": float(
                self.distance_to_nearest_threshold
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EcologicalValueAssignmentV2":
        if not isinstance(payload, Mapping):
            raise TypeError("Ecological assignment must be a mapping.")
        _require_exact_keys(
            payload,
            _ECOLOGICAL_ASSIGNMENT_KEYS_V2,
            "Ecological assignment",
        )
        return cls(
            episode_uid=_strict_text(payload["episode_uid"], "episode_uid"),
            fold_id=_strict_integer(payload["fold_id"], "fold_id"),
            held_out_episode_return=_finite_number(
                payload["held_out_episode_return"], "held_out_episode_return"
            ),
            public_context_stratum=_strict_text(
                payload["public_context_stratum"], "public_context_stratum"
            ),
            value_estimate=_finite_number(payload["value_estimate"], "value_estimate"),
            value_standard_error=_finite_number(
                payload["value_standard_error"], "value_standard_error"
            ),
            estimation_sample_size=_strict_integer(
                payload["estimation_sample_size"],
                "estimation_sample_size",
                minimum=2,
            ),
            class_id=_strict_integer(payload["class_id"], "class_id"),
            distance_to_nearest_threshold=_finite_number(
                payload["distance_to_nearest_threshold"],
                "distance_to_nearest_threshold",
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossFittedEcologicalValueClassesV2:
    split_manifest_sha256: str
    role: str
    n_classes: int
    cross_fit_folds: int
    estimator_id: str
    fold_bin_upper_bounds: tuple[tuple[int, tuple[float, ...]], ...]
    assignments: tuple[EcologicalValueAssignmentV2, ...]
    schema_version: str = ECOLOGICAL_VALUE_CLASS_SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        if self.schema_version != ECOLOGICAL_VALUE_CLASS_SCHEMA_VERSION_V2:
            raise ValueError("Ecological value-class schema version changed.")
        if self.role not in SPLIT_ROLES:
            raise ValueError("Ecological value-class role is not registered.")
        if not isinstance(self.split_manifest_sha256, str) or len(
            self.split_manifest_sha256
        ) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.split_manifest_sha256
        ):
            raise ValueError("Ecological value classes require a split hash.")
        _strict_integer(self.n_classes, "Ecological n_classes", minimum=2)
        _strict_integer(self.cross_fit_folds, "Ecological cross_fit_folds", minimum=2)
        _strict_text(self.estimator_id, "Ecological estimator_id")
        if not self.assignments:
            raise ValueError("Ecological value classes require assignments.")
        episodes = tuple(item.episode_uid for item in self.assignments)
        if episodes != tuple(sorted(episodes)) or len(episodes) != len(set(episodes)):
            raise ValueError("Ecological assignments must be unique and sorted.")
        fold_ids = tuple(int(item[0]) for item in self.fold_bin_upper_bounds)
        if fold_ids != tuple(sorted(fold_ids)) or len(fold_ids) != len(set(fold_ids)):
            raise ValueError("Ecological threshold folds must be unique and sorted.")
        bounds_by_fold = dict(self.fold_bin_upper_bounds)
        if set(bounds_by_fold) != set(range(int(self.cross_fit_folds))):
            raise ValueError("Ecological thresholds must cover every cross-fit fold.")
        for bounds in bounds_by_fold.values():
            if any(not math.isfinite(float(value)) for value in bounds) or any(
                right <= left
                for left, right in zip(bounds, bounds[1:], strict=False)
            ):
                raise ValueError("Ecological thresholds must be finite and increasing.")
        for assignment in self.assignments:
            if int(assignment.fold_id) not in bounds_by_fold:
                raise ValueError("Ecological assignment uses an unknown fold.")
            bounds = bounds_by_fold[int(assignment.fold_id)]
            if len(bounds) != int(self.n_classes) - 1:
                raise ValueError("Ecological threshold count changed n_classes.")
            expected_class = int(np.digitize(
                assignment.value_estimate,
                bounds,
                right=True,
            ))
            if assignment.class_id != expected_class:
                raise ValueError("Ecological class does not match its out-of-fold estimate.")
            expected_distance = min(
                abs(assignment.value_estimate - threshold) for threshold in bounds
            )
            if not math.isclose(
                assignment.distance_to_nearest_threshold,
                expected_distance,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("Ecological threshold distance changed content.")

    @property
    def assignment_by_episode(self) -> dict[str, EcologicalValueAssignmentV2]:
        return {item.episode_uid: item for item in self.assignments}

    def core_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_manifest_sha256": self.split_manifest_sha256,
            "role": self.role,
            "outcome_name": ECOLOGICAL_VALUE_OUTCOME_NAME,
            "n_classes": int(self.n_classes),
            "cross_fit_folds": int(self.cross_fit_folds),
            "estimator_id": self.estimator_id,
            "fold_bin_upper_bounds": [
                {"fold_id": int(fold_id), "values": list(map(float, bounds))}
                for fold_id, bounds in self.fold_bin_upper_bounds
            ],
            "assignments": [item.to_mapping() for item in self.assignments],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.core_payload())

    def to_mapping(self) -> dict[str, Any]:
        return {**self.core_payload(), "sha256": self.sha256}

    def labels_for_rows(
        self,
        episode_uid: Sequence[Any],
        *,
        held_out_episode_return: Sequence[float] | None = None,
        require_complete: bool = True,
    ) -> np.ndarray:
        episodes = np.asarray(episode_uid).astype(str).reshape(-1)
        observed = set(episodes.tolist())
        registered = set(self.assignment_by_episode)
        missing = sorted(observed.difference(registered))
        extra = sorted(registered.difference(observed)) if require_complete else []
        if missing or extra:
            raise ValueError(
                f"Ecological assignment coverage mismatch; missing={missing}, extra={extra}."
            )
        if held_out_episode_return is not None:
            outcomes = np.asarray(held_out_episode_return, dtype=np.float64).reshape(-1)
            if outcomes.shape != episodes.shape:
                raise ValueError("Ecological outcomes are not row-aligned.")
            for episode in observed:
                expected = self.assignment_by_episode[episode].held_out_episode_return
                if not np.allclose(
                    outcomes[episodes == episode],
                    expected,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise ValueError("Ecological outcome changed after artifact creation.")
        return np.asarray(
            [self.assignment_by_episode[item].class_id for item in episodes],
            dtype=np.int16,
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        split_manifest: SplitManifestV1,
    ) -> "CrossFittedEcologicalValueClassesV2":
        if not isinstance(split_manifest, SplitManifestV1):
            raise TypeError("split_manifest must be SplitManifestV1.")
        expected = {
            "schema_version",
            "split_manifest_sha256",
            "role",
            "outcome_name",
            "n_classes",
            "cross_fit_folds",
            "estimator_id",
            "fold_bin_upper_bounds",
            "assignments",
            "sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("Ecological value-class artifact has the wrong fields.")
        assignments = tuple(
            EcologicalValueAssignmentV2.from_mapping(item)
            for item in _strict_sequence(payload["assignments"], "assignments")
        )
        fold_bounds_list: list[tuple[int, tuple[float, ...]]] = []
        for item in _strict_sequence(
            payload["fold_bin_upper_bounds"],
            "fold_bin_upper_bounds",
        ):
            if not isinstance(item, Mapping):
                raise TypeError("Ecological fold thresholds must be mappings.")
            _require_exact_keys(
                item,
                _ECOLOGICAL_FOLD_BOUND_KEYS_V2,
                "Ecological fold thresholds",
            )
            fold_bounds_list.append((
                _strict_integer(item["fold_id"], "fold threshold id"),
                tuple(
                    _finite_number(value, "fold threshold")
                    for value in _strict_sequence(
                        item["values"],
                        "Ecological fold threshold values",
                    )
                ),
            ))
        fold_bounds = tuple(fold_bounds_list)
        artifact = cls(
            split_manifest_sha256=str(payload["split_manifest_sha256"]),
            role=str(payload["role"]),
            n_classes=_strict_integer(payload["n_classes"], "n_classes", minimum=2),
            cross_fit_folds=_strict_integer(
                payload["cross_fit_folds"], "cross_fit_folds", minimum=2
            ),
            estimator_id=str(payload["estimator_id"]),
            fold_bin_upper_bounds=fold_bounds,
            assignments=tuple(sorted(assignments, key=lambda item: item.episode_uid)),
            schema_version=str(payload["schema_version"]),
        )
        if payload["outcome_name"] != ECOLOGICAL_VALUE_OUTCOME_NAME:
            raise ValueError("Ecological outcome name changed.")
        if artifact.split_manifest_sha256 != split_manifest.sha256:
            raise ValueError("Ecological artifact changed split manifest.")
        if payload["sha256"] != artifact.sha256:
            raise ValueError("Ecological artifact SHA-256 does not match content.")
        rebuilt = fit_cross_fitted_ecological_value_estimates(
            [item.held_out_episode_return for item in artifact.assignments],
            [item.episode_uid for item in artifact.assignments],
            [item.public_context_stratum for item in artifact.assignments],
            split_manifest=split_manifest,
            role=artifact.role,
            n_classes=artifact.n_classes,
        )
        if rebuilt.core_payload() != artifact.core_payload():
            raise ValueError(
                "Ecological artifact does not match its frozen cross-fitted estimator."
            )
        return artifact


def fit_cross_fitted_ecological_value_estimates(
    held_out_value_outcome: Sequence[float],
    episode_uid: Sequence[Any],
    public_context_stratum: Sequence[Any],
    *,
    split_manifest: SplitManifestV1,
    role: str,
    n_classes: int,
) -> CrossFittedEcologicalValueClassesV2:
    """Fit value estimates from other episodes; apply outcomes never enter estimates."""

    values = np.asarray(held_out_value_outcome, dtype=np.float64).reshape(-1)
    episodes = np.asarray(episode_uid).astype(str).reshape(-1)
    contexts = np.asarray(public_context_stratum).astype(str).reshape(-1)
    if not (values.shape == episodes.shape == contexts.shape) or values.size == 0:
        raise ValueError("Ecological value inputs must be non-empty and aligned.")
    if not np.isfinite(values).all():
        raise ValueError("Ecological outcomes must be finite.")
    if not isinstance(split_manifest, SplitManifestV1):
        raise TypeError("split_manifest must be SplitManifestV1.")
    if role not in SPLIT_ROLES:
        raise ValueError("Ecological value role is not registered.")
    classes = _strict_integer(n_classes, "n_classes", minimum=2)
    unique_episodes = tuple(sorted(set(episodes.tolist())))
    episode_values: dict[str, float] = {}
    episode_contexts: dict[str, str] = {}
    for episode in unique_episodes:
        rows = np.flatnonzero(episodes == episode)
        observed_values = values[rows]
        if not np.allclose(observed_values, observed_values[0], rtol=0.0, atol=1.0e-12):
            raise ValueError("Every episode must carry one ecological return.")
        context = str(contexts[int(rows[0])])
        _strict_text(context, "first-decision public context")
        episode_values[episode] = float(observed_values[0])
        episode_contexts[episode] = context
    episode_fold = {
        episode: split_manifest.cross_fit_fold(role, episode)
        for episode in unique_episodes
    }
    if set(episode_fold.values()) != set(range(split_manifest.cross_fit_folds)):
        raise ValueError("Ecological data do not populate every frozen fold.")
    quantiles = np.arange(1, classes, dtype=np.float64) / float(classes)
    assignments: list[EcologicalValueAssignmentV2] = []
    fold_bounds: list[tuple[int, tuple[float, ...]]] = []
    for fold_id in range(split_manifest.cross_fit_folds):
        fit_episodes = tuple(
            episode for episode in unique_episodes if episode_fold[episode] != fold_id
        )
        apply_episodes = tuple(
            episode for episode in unique_episodes if episode_fold[episode] == fold_id
        )
        if len(fit_episodes) < 2 or not apply_episodes:
            raise ValueError("Ecological cross-fitting produced insufficient episodes.")
        raw_bounds = np.quantile(
            np.asarray([episode_values[item] for item in fit_episodes]),
            quantiles,
            method="linear",
        )
        bounds = tuple(float(value) for value in np.unique(raw_bounds))
        if len(bounds) != classes - 1:
            raise ValueError("Ecological returns do not support strict value classes.")
        fold_bounds.append((fold_id, bounds))
        for episode in apply_episodes:
            same_context = tuple(
                item
                for item in fit_episodes
                if episode_contexts[item] == episode_contexts[episode]
            )
            donors = same_context if len(same_context) >= 2 else fit_episodes
            donor_values = np.asarray(
                [episode_values[item] for item in donors],
                dtype=np.float64,
            )
            estimate = float(np.mean(donor_values, dtype=np.float64))
            standard_error = float(
                np.std(donor_values, ddof=1, dtype=np.float64)
                / math.sqrt(donor_values.size)
            )
            assignments.append(EcologicalValueAssignmentV2(
                episode_uid=episode,
                fold_id=fold_id,
                held_out_episode_return=episode_values[episode],
                public_context_stratum=episode_contexts[episode],
                value_estimate=estimate,
                value_standard_error=standard_error,
                estimation_sample_size=int(donor_values.size),
                class_id=int(np.digitize(estimate, bounds, right=True)),
                distance_to_nearest_threshold=float(min(
                    abs(estimate - threshold) for threshold in bounds
                )),
            ))
    return CrossFittedEcologicalValueClassesV2(
        split_manifest_sha256=split_manifest.sha256,
        role=str(role),
        n_classes=classes,
        cross_fit_folds=int(split_manifest.cross_fit_folds),
        estimator_id="role_local_first_public_context_mean_v1",
        fold_bin_upper_bounds=tuple(fold_bounds),
        assignments=tuple(sorted(assignments, key=lambda item: item.episode_uid)),
    )


__all__ = (
    "VALUE_CLASS_SCHEMA_VERSION",
    "ECOLOGICAL_VALUE_OUTCOME_NAME",
    "ValueClassFoldV1",
    "ValueClassAssignmentV1",
    "CrossFittedValueClassesV1",
    "fit_cross_fitted_value_classes",
    "cluster_stratified_permutation",
    "ECOLOGICAL_VALUE_CLASS_SCHEMA_VERSION_V2",
    "EcologicalValueAssignmentV2",
    "CrossFittedEcologicalValueClassesV2",
    "fit_cross_fitted_ecological_value_estimates",
)
