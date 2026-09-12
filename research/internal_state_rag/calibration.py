"""Risk calibration for a sequential internal-state stopping policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Trajectory:
    """Internal risk scores and true safety at successive retrieval rounds.

    Lower scores mean the model believes it is safer to stop.  ``unsafe[t]`` is
    true when stopping at round ``t`` would return an ungrounded or wrong answer.
    """

    scores: Sequence[float]
    unsafe: Sequence[bool]

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("A trajectory must contain at least one round.")
        if len(self.scores) != len(self.unsafe):
            raise ValueError("scores and unsafe must have equal length.")


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    empirical_anytime_risk: float
    corrected_anytime_risk: float
    alpha: float
    n_calibration: int
    certified: bool


def _anytime_false_stop(trajectory: Trajectory, threshold: float) -> int:
    """Upper-bound policy failure by any unsafe round below the threshold."""

    return int(
        any(score <= threshold and is_unsafe for score, is_unsafe in zip(
            trajectory.scores, trajectory.unsafe
        ))
    )


def calibrate_stop_threshold(
    trajectories: Iterable[Trajectory],
    *,
    alpha: float = 0.1,
) -> CalibrationResult:
    """Choose the largest threshold satisfying a finite-sample risk correction.

    The calibration unit is a complete query trajectory, rather than a
    candidate chunk or retrieval round.  Its loss is one when *any* unsafe round
    would be certified.  This monotone loss handles a later adaptive stopping
    time without testing K or L separate hypotheses.

    For bounded zero-one loss, the conformal-risk correction is
    ``(errors + 1) / (n + 1)``.  The result controls marginal anytime false-stop
    risk under exchangeability.  It does not claim conditional error among only
    the queries on which the model stops.
    """

    rows = list(trajectories)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if not rows:
        raise ValueError("At least one calibration trajectory is required.")

    candidates = sorted({float(score) for row in rows for score in row.scores})
    best_threshold = -inf
    best_errors = 0
    certified = (1.0 / (len(rows) + 1)) <= alpha

    for threshold in candidates:
        errors = sum(_anytime_false_stop(row, threshold) for row in rows)
        corrected_risk = (errors + 1.0) / (len(rows) + 1.0)
        if corrected_risk <= alpha:
            best_threshold = threshold
            best_errors = errors

    n = len(rows)
    return CalibrationResult(
        threshold=best_threshold,
        empirical_anytime_risk=best_errors / n,
        corrected_anytime_risk=(best_errors + 1.0) / (n + 1.0),
        alpha=alpha,
        n_calibration=n,
        certified=certified,
    )
