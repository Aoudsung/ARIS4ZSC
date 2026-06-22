"""Test bootstrap for the standalone Gate 3 package.

This file intentionally only scopes imports for this package's own tests. It does
not silence or skip legacy repository tests such as tests/test_serd_core.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
