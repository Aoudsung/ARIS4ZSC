from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import re
from typing import Any, ClassVar, Mapping


RESPONSE_SUMMARY_SCHEMA_VERSION = "path_c_response_summary_v1"
STRUCTURED_MULTILABEL_ROLE = "secondary_only"

_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ResponseSummarySpecV1:
    """Frozen finite response summary used by the primary kernel theorem.

    The primary response is one categorical token. Structured multi-label
    columns may still be retained for secondary analyses, but they cannot be
    passed through this encoder or treated as the theorem response alphabet.
    """

    response_classes: tuple[str, ...]
    latency_bin_upper_bounds: tuple[int, ...]
    schema_version: str = RESPONSE_SUMMARY_SCHEMA_VERSION
    structured_multilabel_role: str = STRUCTURED_MULTILABEL_ROLE

    SPECIAL_TOKENS: ClassVar[tuple[str, ...]] = (
        "terminal",
        "censored",
        "invalid_script",
        "support_violation",
    )
    _MAPPING_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "response_classes",
            "latency_bin_upper_bounds",
            "structured_multilabel_role",
        }
    )

    def __post_init__(self) -> None:
        if isinstance(self.response_classes, (str, bytes)):
            raise TypeError("response_classes must be a sequence of labels.")
        if isinstance(self.latency_bin_upper_bounds, (str, bytes)):
            raise TypeError(
                "latency_bin_upper_bounds must be a sequence of integers."
            )
        object.__setattr__(
            self,
            "response_classes",
            tuple(self.response_classes),
        )
        object.__setattr__(
            self,
            "latency_bin_upper_bounds",
            tuple(self.latency_bin_upper_bounds),
        )
        if self.schema_version != RESPONSE_SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                "ResponseSummarySpecV1 schema_version must be "
                f"{RESPONSE_SUMMARY_SCHEMA_VERSION!r}."
            )
        if self.structured_multilabel_role != STRUCTURED_MULTILABEL_ROLE:
            raise ValueError(
                "Structured multi-label response fields are secondary-only and "
                "cannot define the primary theorem response."
            )
        if not self.response_classes:
            raise ValueError("response_classes must be non-empty.")
        if len(self.response_classes) != len(set(self.response_classes)):
            raise ValueError("response_classes must be unique.")
        for label in self.response_classes:
            if not isinstance(label, str) or not _LABEL_PATTERN.fullmatch(label):
                raise ValueError(
                    "Each response class must match [a-z][a-z0-9_]*; "
                    f"got {label!r}."
                )
            if label in self.SPECIAL_TOKENS:
                raise ValueError(
                    f"Response class {label!r} is reserved for a special token."
                )

        if not self.latency_bin_upper_bounds:
            raise ValueError(
                "latency_bin_upper_bounds must be preregistered and non-empty."
            )
        previous = -1
        normalized_bounds = []
        for bound in self.latency_bin_upper_bounds:
            if isinstance(bound, bool) or not isinstance(bound, Integral):
                raise ValueError("Latency bin upper bounds must be integers.")
            value = int(bound)
            if value < 0:
                raise ValueError("Latency bin upper bounds must be non-negative.")
            if value <= previous:
                raise ValueError(
                    "Latency bin upper bounds must be strictly increasing."
                )
            previous = value
            normalized_bounds.append(value)
        object.__setattr__(
            self,
            "latency_bin_upper_bounds",
            tuple(normalized_bounds),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ResponseSummarySpecV1":
        if not isinstance(payload, Mapping):
            raise TypeError("Response summary spec must be a mapping.")
        observed = set(payload)
        unknown = sorted(observed.difference(cls._MAPPING_KEYS))
        missing = sorted(cls._MAPPING_KEYS.difference(observed))
        if unknown:
            raise ValueError(
                "Response summary spec contains unknown key(s): "
                + ", ".join(unknown)
            )
        if missing:
            raise ValueError(
                "Response summary spec is missing required key(s): "
                + ", ".join(missing)
            )
        response_classes = payload["response_classes"]
        latency_bounds = payload["latency_bin_upper_bounds"]
        if isinstance(response_classes, (str, bytes)):
            raise TypeError("response_classes must be a sequence of labels.")
        if isinstance(latency_bounds, (str, bytes)):
            raise TypeError(
                "latency_bin_upper_bounds must be a sequence of integers."
            )
        try:
            response_classes_tuple = tuple(response_classes)
            latency_bounds_tuple = tuple(latency_bounds)
        except TypeError as exc:
            raise TypeError(
                "response_classes and latency_bin_upper_bounds must be sequences."
            ) from exc
        return cls(
            response_classes=response_classes_tuple,
            latency_bin_upper_bounds=latency_bounds_tuple,
            schema_version=str(payload["schema_version"]),
            structured_multilabel_role=str(
                payload["structured_multilabel_role"]
            ),
        )

    @property
    def latency_bucket_labels(self) -> tuple[str, ...]:
        bounded = tuple(
            f"le_{int(bound)}" for bound in self.latency_bin_upper_bounds
        )
        return (*bounded, f"gt_{int(self.latency_bin_upper_bounds[-1])}")

    @property
    def vocabulary(self) -> tuple[str, ...]:
        regular = tuple(
            f"response_{response_class}__latency_{latency_bucket}"
            for response_class in self.response_classes
            for latency_bucket in self.latency_bucket_labels
        )
        return (*self.SPECIAL_TOKENS, *regular)

    @property
    def q(self) -> int:
        """Size of the frozen categorical response alphabet."""

        return len(self.vocabulary)

    @property
    def token_ids(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.vocabulary)}

    def encode(
        self,
        response_class: str | None = None,
        latency_steps: int | None = None,
        *,
        terminal: bool = False,
        censored: bool = False,
        invalid_script: bool = False,
        support_violation: bool = False,
    ) -> int:
        flags = {
            "terminal": terminal,
            "censored": censored,
            "invalid_script": invalid_script,
            "support_violation": support_violation,
        }
        for name, value in flags.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean.")
        active_special = [name for name, value in flags.items() if value]
        if len(active_special) > 1:
            raise ValueError(
                "A response can have at most one terminal/censored/invalid-script/"
                "support-violation token."
            )
        if active_special:
            if response_class is not None or latency_steps is not None:
                raise ValueError(
                    "Special response tokens cannot also carry a response class "
                    "or latency."
                )
            return self.token_ids[active_special[0]]

        if response_class is None or latency_steps is None:
            raise ValueError(
                "A regular response requires both response_class and latency_steps."
            )
        if response_class not in self.response_classes:
            raise ValueError(
                f"Unregistered response class {response_class!r}; "
                f"expected one of {self.response_classes!r}."
            )
        if isinstance(latency_steps, bool) or not isinstance(latency_steps, Integral):
            raise TypeError("latency_steps must be an integer.")
        latency = int(latency_steps)
        if latency < 0:
            raise ValueError("latency_steps must be non-negative.")
        bucket = self.latency_bucket_labels[-1]
        for upper_bound, label in zip(
            self.latency_bin_upper_bounds,
            self.latency_bucket_labels,
            strict=False,
        ):
            if latency <= int(upper_bound):
                bucket = label
                break
        token = f"response_{response_class}__latency_{bucket}"
        return self.token_ids[token]

    def decode(self, token_id: int) -> str:
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TypeError("token_id must be an integer.")
        value = int(token_id)
        if value < 0 or value >= self.q:
            raise ValueError(f"token_id must be in [0, {self.q}); got {value}.")
        return self.vocabulary[value]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "response_classes": list(self.response_classes),
            "latency_bin_upper_bounds": [
                int(value) for value in self.latency_bin_upper_bounds
            ],
            "structured_multilabel_role": self.structured_multilabel_role,
        }

    def canonical_payload(self) -> dict[str, Any]:
        return {
            **self.to_mapping(),
            "primary_summary": "canonical_finite_token_id",
            "special_tokens": list(self.SPECIAL_TOKENS),
            "latency_bucket_labels": list(self.latency_bucket_labels),
            "vocabulary": list(self.vocabulary),
            "q": self.q,
            "structured_multilabel_theorem_eligible": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload())


__all__ = [
    "RESPONSE_SUMMARY_SCHEMA_VERSION",
    "STRUCTURED_MULTILABEL_ROLE",
    "ResponseSummarySpecV1",
]
