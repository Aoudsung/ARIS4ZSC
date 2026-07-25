"""Host-side metric records written only after compiled work completes."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contracts.records import MetricRow


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class MetricWriter:
    """Accumulate a small JSON-lines ledger without device-loop input or output."""

    def __init__(self, path: str | Path, *, truncate: bool = False) -> None:
        self.path = Path(path).resolve()
        self._rows: list[dict[str, Any]] = []
        if truncate:
            _atomic_text(self.path, "")
        elif self.path.is_file():
            self._rows = [
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def append(self, row: MetricRow | Mapping[str, Any]) -> None:
        payload = row.to_mapping() if isinstance(row, MetricRow) else dict(row)
        self._rows.append(payload)
        _atomic_text(
            self.path,
            "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in self._rows),
        )

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._rows)


class JsonlLedger:
    """Append completed host records after each device segment.

    A ``.gz`` suffix selects concatenated gzip members.  Opening one member per
    device segment keeps writes recoverable without retaining a long-lived file
    handle, and standard gzip readers transparently consume the complete stream.
    """

    def __init__(self, path: str | Path, *, truncate: bool = False) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if truncate or not self.path.exists():
            if self.path.suffix == ".gz":
                temporary = self.path.with_name(f".{self.path.name}.tmp")
                with gzip.open(temporary, "wt", encoding="utf-8"):
                    pass
                temporary.replace(self.path)
            else:
                _atomic_text(self.path, "")

    def append(self, rows: Iterable[Mapping[str, Any]]) -> None:
        encoded = "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
        if not encoded:
            return
        if self.path.suffix == ".gz":
            with gzip.open(self.path, "at", encoding="utf-8") as handle:
                handle.write(encoded)
        else:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)


def effective_counts_from_rows(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, int]:
    """Read back the largest completed environment-step and episode counts."""

    materialized = list(rows)
    if not materialized:
        return {"effective_environment_steps": 0, "completed_episodes": 0}
    return {
        "effective_environment_steps": max(int(row["environment_steps"]) for row in materialized),
        "completed_episodes": max(int(row["completed_episodes"]) for row in materialized),
    }
