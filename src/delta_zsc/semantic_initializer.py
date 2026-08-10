"""Unsupervised spectral-simplex initialization for DELTA v5 semantics.

The initializer is fitted only from episode-level residual interface-event
vectors.  Partner IDs, SP/OP labels, returns, and privileged deployment facts
are excluded.  The artifact is versioned and consists of a compact NPZ payload
plus a human-readable JSON contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SEMANTIC_INITIALIZER_SCHEMA_VERSION = 1
SEMANTIC_INITIALIZER_LOGIT_NORM = 0.5


@dataclass(frozen=True, slots=True)
class SemanticInitializer:
    """Centered component-event bias and the evidence used to construct it."""

    event_component_bias: np.ndarray
    singular_values: np.ndarray
    source: Mapping[str, Any]
    construction: str
    logit_norm: float = SEMANTIC_INITIALIZER_LOGIT_NORM
    version: int = SEMANTIC_INITIALIZER_SCHEMA_VERSION

    def to_mapping(self, *, include_bias: bool = True) -> Mapping[str, Any]:
        bias = np.asarray(self.event_component_bias, dtype=np.float32)
        payload: dict[str, Any] = {
            "version": int(self.version),
            "artifact_type": "delta_semantic_component_initializer",
            "construction": str(self.construction),
            "logit_norm": float(self.logit_norm),
            "event_component_bias_shape": list(bias.shape),
            "singular_values": np.asarray(
                self.singular_values, dtype=np.float32
            ).tolist(),
            "source": dict(self.source),
            "uses_partner_labels": False,
        }
        if include_bias:
            payload["event_component_bias"] = bias.tolist()
        return payload


def regular_simplex_vertices(component_count: int) -> np.ndarray:
    """Return K unit-norm vertices in a centered (K-1)-dimensional simplex."""

    count = int(component_count)
    if count < 2:
        raise ValueError("A semantic simplex requires at least two components.")
    centered = np.eye(count, dtype=np.float64) - np.ones((count, count)) / count
    basis, _ = np.linalg.qr(centered[:, : count - 1])
    vertices = basis * np.sqrt(count / float(count - 1))
    vertices -= np.mean(vertices, axis=0, keepdims=True)
    return vertices.astype(np.float32)


def _complete_directions(
    directions: np.ndarray,
    *,
    requested: int,
    dimension: int,
    seed: int,
) -> np.ndarray:
    """Complete an orthonormal row basis deterministically when rank is low."""

    rows = np.asarray(directions, dtype=np.float64).reshape((-1, int(dimension)))
    accepted: list[np.ndarray] = []
    for row in rows:
        candidate = row.copy()
        for previous in accepted:
            candidate -= np.dot(candidate, previous) * previous
        norm = np.linalg.norm(candidate)
        if norm > 1.0e-8:
            accepted.append(candidate / norm)
        if len(accepted) == int(requested):
            break
    rng = np.random.default_rng(int(seed))
    while len(accepted) < int(requested):
        candidate = rng.normal(size=(int(dimension),))
        for previous in accepted:
            candidate -= np.dot(candidate, previous) * previous
        norm = np.linalg.norm(candidate)
        if norm > 1.0e-8:
            accepted.append(candidate / norm)
    return np.stack(accepted, axis=0).astype(np.float32)


def spectral_simplex_bias(
    episode_residuals: Any,
    *,
    component_count: int,
    event_classes: int,
    seed: int = 0,
    logit_norm: float = SEMANTIC_INITIALIZER_LOGIT_NORM,
) -> tuple[np.ndarray, np.ndarray]:
    """Map leading residual directions onto centered simplex vertices."""

    residuals = np.asarray(episode_residuals, dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[1] != int(event_classes):
        raise ValueError("Episode residuals must have shape [episode, event_class].")
    centered = residuals - np.mean(residuals, axis=0, keepdims=True)
    if centered.shape[0] >= 2 and np.any(np.abs(centered) > 0.0):
        _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    else:
        singular_values = np.zeros((0,), dtype=np.float64)
        right = np.zeros((0, int(event_classes)), dtype=np.float64)
    directions = _complete_directions(
        right,
        requested=int(component_count) - 1,
        dimension=int(event_classes),
        seed=int(seed),
    )
    vertices = regular_simplex_vertices(component_count)
    bias = float(logit_norm) * (vertices @ directions)
    bias -= np.mean(bias, axis=0, keepdims=True)
    return bias.astype(np.float32), np.asarray(singular_values, dtype=np.float32)


def fit_spectral_simplex_initializer(
    episode_residuals: Any,
    *,
    component_count: int,
    source: Mapping[str, Any],
    seed: int = 0,
    logit_norm: float = SEMANTIC_INITIALIZER_LOGIT_NORM,
) -> SemanticInitializer:
    residuals = np.asarray(episode_residuals, dtype=np.float32)
    if residuals.ndim != 2:
        raise ValueError("Episode residual matrix must be rank two.")
    bias, singular_values = spectral_simplex_bias(
        residuals,
        component_count=component_count,
        event_classes=residuals.shape[1],
        seed=seed,
        logit_norm=logit_norm,
    )
    return SemanticInitializer(
        event_component_bias=bias,
        singular_values=singular_values,
        source=dict(source),
        construction="cross_fitted_episode_residual_spectral_simplex",
        logit_norm=float(logit_norm),
    )


def deterministic_simplex_initializer(
    component_count: int,
    event_count: int,
    *,
    seed: int = 0,
    logit_norm: float = SEMANTIC_INITIALIZER_LOGIT_NORM,
) -> SemanticInitializer:
    """Mechanical/test fallback with the same geometry but no fitted directions."""

    empty = np.zeros((1, int(event_count)), dtype=np.float32)
    bias, singular_values = spectral_simplex_bias(
        empty,
        component_count=component_count,
        event_classes=event_count,
        seed=seed,
        logit_norm=logit_norm,
    )
    return SemanticInitializer(
        event_component_bias=bias,
        singular_values=singular_values,
        source={"kind": "deterministic_unfitted_fallback", "seed": int(seed)},
        construction="deterministic_orthogonal_simplex_fallback",
        logit_norm=float(logit_norm),
    )


def random_orthogonal_simplex_bias(
    key: Any,
    *,
    component_count: int,
    event_classes: int,
    logit_norm: float = SEMANTIC_INITIALIZER_LOGIT_NORM,
) -> Any:
    """JAX fallback used when no external initializer artifact is supplied."""

    import jax
    import jax.numpy as jnp

    count = int(component_count)
    classes = int(event_classes)
    centered = jnp.eye(count, dtype=jnp.float32) - jnp.ones(
        (count, count), dtype=jnp.float32
    ) / float(count)
    basis, _ = jnp.linalg.qr(centered[:, : count - 1])
    vertices = basis * jnp.sqrt(float(count) / float(count - 1))
    raw = jax.random.normal(key, (classes, count - 1), dtype=jnp.float32)
    directions, _ = jnp.linalg.qr(raw)
    bias = float(logit_norm) * (vertices @ directions.T)
    return bias - jnp.mean(bias, axis=0, keepdims=True)


def _validate_initializer(
    initializer: SemanticInitializer,
    *,
    expected_components: int | None = None,
    expected_event_classes: int | None = None,
) -> None:
    bias = np.asarray(initializer.event_component_bias, dtype=np.float32)
    if bias.ndim != 2:
        raise ValueError("Semantic component bias must have shape [K,event].")
    if expected_components is not None and bias.shape[0] != int(expected_components):
        raise ValueError("Semantic initializer component count differs.")
    if expected_event_classes is not None and bias.shape[1] != int(
        expected_event_classes
    ):
        raise ValueError("Semantic initializer event count differs.")
    if not np.all(np.isfinite(bias)):
        raise ValueError("Semantic initializer contains non-finite values.")
    if not np.allclose(np.mean(bias, axis=0), 0.0, atol=1.0e-5):
        raise ValueError("Semantic initializer must be centered across components.")


def save_semantic_initializer(
    output: str | Path,
    initializer: SemanticInitializer | None = None,
    *,
    event_component_bias: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write the paired NPZ payload and JSON contract.

    The keyword-only bias/metadata form is retained for small standalone tools;
    the repository uses the typed :class:`SemanticInitializer` form.
    """

    if initializer is None:
        if event_component_bias is None:
            raise TypeError("A semantic initializer or event component bias is required.")
        initializer = SemanticInitializer(
            event_component_bias=np.asarray(event_component_bias, dtype=np.float32),
            singular_values=np.asarray([], dtype=np.float32),
            source=dict(metadata or {}),
            construction="external_centered_bias",
        )
    _validate_initializer(initializer)
    root = Path(output).resolve()
    if root.suffix:
        root = root.parent
    root.mkdir(parents=True, exist_ok=True)
    array_path = root / "semantic_component_initializer.npz"
    metadata_path = root / "semantic_component_initializer.json"
    bias = np.asarray(initializer.event_component_bias, dtype=np.float32)
    np.savez_compressed(array_path, event_component_bias=bias)
    metadata_path.write_text(
        json.dumps(initializer.to_mapping(include_bias=False), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return array_path, metadata_path


def load_semantic_initializer(
    source: str | Path,
    *,
    expected_components: int | None = None,
    expected_event_classes: int | None = None,
    component_count: int | None = None,
    event_count: int | None = None,
) -> SemanticInitializer:
    """Load and validate a directory, NPZ, or paired JSON path."""

    expected_components = (
        component_count if expected_components is None else expected_components
    )
    expected_event_classes = (
        event_count if expected_event_classes is None else expected_event_classes
    )
    path = Path(source).resolve()
    if path.is_dir():
        array_path = path / "semantic_component_initializer.npz"
        metadata_path = path / "semantic_component_initializer.json"
    elif path.suffix == ".json":
        metadata_path = path
        array_path = path.with_name("semantic_component_initializer.npz")
    else:
        array_path = path
        metadata_path = path.with_name("semantic_component_initializer.json")
    if not array_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Semantic initializer requires paired NPZ and JSON files.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("artifact_type") != "delta_semantic_component_initializer"
        or int(metadata.get("version", -1)) != SEMANTIC_INITIALIZER_SCHEMA_VERSION
    ):
        raise ValueError("Semantic initializer schema differs from DELTA v5.")
    if metadata.get("uses_partner_labels") is not False:
        raise ValueError(
            "Semantic initializer must explicitly certify that partner labels "
            "were not used in its construction."
        )
    with np.load(array_path, allow_pickle=False) as payload:
        if set(payload.files) != {"event_component_bias"}:
            raise ValueError("Semantic initializer NPZ contains unexpected arrays.")
        bias = np.asarray(payload["event_component_bias"], dtype=np.float32)
    initializer = SemanticInitializer(
        event_component_bias=bias,
        singular_values=np.asarray(metadata.get("singular_values", []), dtype=np.float32),
        source=dict(metadata.get("source", {})),
        construction=str(metadata.get("construction", "unknown")),
        logit_norm=float(metadata.get("logit_norm", SEMANTIC_INITIALIZER_LOGIT_NORM)),
        version=int(metadata["version"]),
    )
    _validate_initializer(
        initializer,
        expected_components=expected_components,
        expected_event_classes=expected_event_classes,
    )
    if list(bias.shape) != metadata.get("event_component_bias_shape"):
        raise ValueError("Semantic initializer JSON/NPZ shape contract differs.")
    return initializer


def validate_semantic_initializer_provenance(
    initializer: SemanticInitializer,
    *,
    method: str,
    layout: str,
    component_count: int,
    event_classes: int,
    protocol_version: str,
    official_source_commit: str,
    training_parent_ids: Any,
    expected_calibration_run_ids: Any = (),
    require_fitted: bool,
) -> None:
    """Enforce the label-free, lineage-disjoint initializer contract.

    Mechanical runs may use the deterministic orthogonal-simplex fallback.
    Development and formal semantic variants require a fitted artifact built
    from the exact ``calibration`` panel in the active manifest.  The manifest
    validator separately proves calibration/training/confirmatory parent and
    co-training disjointness; this check binds the initializer to that panel and
    also rejects an artifact copied from another method, layout, or protocol.
    """

    _validate_initializer(
        initializer,
        expected_components=int(component_count),
        expected_event_classes=int(event_classes),
    )
    source = dict(initializer.source)
    if source.get("uses_partner_labels") not in (None, False):
        raise ValueError("Semantic initializer source used privileged labels.")

    fitted = (
        initializer.construction
        == "cross_fitted_episode_residual_spectral_simplex"
    )
    if not fitted:
        if require_fitted:
            raise ValueError(
                "Development/formal DELTA semantic variants require a fitted "
                "cross-fitted spectral-simplex initializer."
            )
        if initializer.construction != "deterministic_orthogonal_simplex_fallback":
            raise ValueError("Unknown unfitted semantic initializer construction.")
        return

    expected = {
        "method": str(method),
        "layout": str(layout),
        "partner_role": "calibration",
        "component_count": int(component_count),
        "official_protocol_version": str(protocol_version),
        "official_source_commit": str(official_source_commit),
        "uses_partner_labels": False,
    }
    for name, value in expected.items():
        if source.get(name) != value:
            raise ValueError(
                f"Semantic initializer provenance field {name!r} differs: "
                f"expected={value!r}, observed={source.get(name)!r}."
            )

    observed_runs = tuple(
        sorted(str(value) for value in source.get("partner_run_ids", ()))
    )
    expected_runs = tuple(
        sorted(str(value) for value in expected_calibration_run_ids)
    )
    if require_fitted and (not expected_runs or observed_runs != expected_runs):
        raise ValueError(
            "Semantic initializer does not match the active manifest's exact "
            "calibration panel."
        )

    calibration_parents = {
        str(value) for value in source.get("parent_training_run_ids", ())
    }
    if not calibration_parents:
        raise ValueError(
            "Fitted semantic initializer lacks calibration parent lineage."
        )
    training_parents = {str(value) for value in training_parent_ids}
    overlap = sorted(calibration_parents & training_parents)
    if overlap:
        raise ValueError(
            "Semantic-initializer calibration parents overlap DELTA training "
            f"support: {overlap}."
        )
    event_count = int(source.get("event_count", 0))
    if event_count < int(component_count):
        raise ValueError(
            "Fitted semantic initializer has fewer structured events than components."
        )
    episode_ids = {int(value) for value in source.get("episode_ids", ())}
    if len(episode_ids) < int(component_count):
        raise ValueError(
            "Fitted semantic initializer has fewer event-bearing episodes than components."
        )


__all__ = [
    "SEMANTIC_INITIALIZER_LOGIT_NORM",
    "SEMANTIC_INITIALIZER_SCHEMA_VERSION",
    "SemanticInitializer",
    "deterministic_simplex_initializer",
    "fit_spectral_simplex_initializer",
    "load_semantic_initializer",
    "random_orthogonal_simplex_bias",
    "regular_simplex_vertices",
    "save_semantic_initializer",
    "spectral_simplex_bias",
    "validate_semantic_initializer_provenance",
]
