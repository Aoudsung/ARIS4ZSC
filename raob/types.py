"""Minimal public types used by SRVF-MAPPO benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class AffordanceBatch:
    """Current public affordance vector and optional metadata."""

    g: torch.Tensor
    metadata: Mapping[str, Any] | None = None
