"""Pairing execution and raw-row summaries for Path C."""

from .pairing import (
    PairingSpec,
    development_pairings,
    execute_formal_pairings,
    execute_pairings,
    validate_formal_pairings,
)
from .summary import summarize_rows
from .response_contrast import (
    ResponseContrastManifest,
    execute_response_contrast_pairing,
    response_contrast_pairings,
    summarize_response_contrast_rows,
)
from .standard import (
    PopulationManifest,
    execute_standard_pairing,
    standard_pairings,
    summarize_project_xp_difference,
    summarize_standard_rows,
    validate_standard_summary_payload,
)

__all__ = [
    "PairingSpec",
    "PopulationManifest",
    "ResponseContrastManifest",
    "development_pairings",
    "execute_formal_pairings",
    "execute_pairings",
    "execute_response_contrast_pairing",
    "execute_standard_pairing",
    "standard_pairings",
    "response_contrast_pairings",
    "summarize_project_xp_difference",
    "summarize_rows",
    "summarize_response_contrast_rows",
    "summarize_standard_rows",
    "validate_standard_summary_payload",
    "validate_formal_pairings",
]
