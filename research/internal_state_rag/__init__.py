"""Research prototype for generator-internal retrieval decisions.

This package is intentionally isolated from ``src/uncertainty_rag`` while the
method is being validated.  Importing it does not change the production RAG
pipeline.
"""

from .calibration import CalibrationResult, Trajectory, calibrate_stop_threshold
from .labels import InterventionOutcome, RetrievalState, derive_retrieval_state
from .signals import InternalTrace, QwenInternalStateExtractor

__all__ = [
    "CalibrationResult",
    "InternalTrace",
    "InterventionOutcome",
    "QwenInternalStateExtractor",
    "RetrievalState",
    "Trajectory",
    "calibrate_stop_threshold",
    "derive_retrieval_state",
]
