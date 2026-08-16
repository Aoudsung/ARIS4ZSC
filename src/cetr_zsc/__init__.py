"""CETR-ZSC: constrained episodic tail-robust zero-shot coordination.

One partner-agnostic recurrent actor, trained on complete raw episodic
returns with a parent-level lower-half tail objective and a measured
self-play non-inferiority constraint.  The method identity lives in
``src/cetr_zsc/config.py``.
"""

from __future__ import annotations
