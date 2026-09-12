"""Counterfactual labels for deciding what a RAG system should do next."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RetrievalState(str, Enum):
    """Actionable state inferred from controlled context interventions."""

    GROUNDED_SUFFICIENT = "grounded_sufficient"
    PARAMETRIC_KNOWN = "parametric_known"
    RETRIEVAL_INSUFFICIENT = "retrieval_insufficient"
    CONTEXT_MISLED = "context_misled"
    REASONING_FAILURE = "reasoning_failure"


@dataclass(frozen=True)
class InterventionOutcome:
    """Correctness under four evidence interventions for one question.

    ``current_correct`` is the result under the current top-K context.
    ``no_context_correct`` removes all retrieved evidence.
    ``without_support_correct`` removes labelled supporting chunks only.
    ``with_oracle_correct`` adds missing gold evidence to the current context.
    """

    current_correct: bool
    no_context_correct: bool
    without_support_correct: bool
    with_oracle_correct: bool


def derive_retrieval_state(outcome: InterventionOutcome) -> RetrievalState:
    """Map correctness interventions to a state with a distinct next action.

    The ordering is deliberate.  A context that breaks an answer the model knew
    is a context-integration failure, even if adding oracle evidence can repair
    it.  A currently correct answer is considered grounded only when removing
    its support breaks it.
    """

    if outcome.current_correct:
        if not outcome.without_support_correct:
            return RetrievalState.GROUNDED_SUFFICIENT
        return RetrievalState.PARAMETRIC_KNOWN

    if outcome.no_context_correct:
        return RetrievalState.CONTEXT_MISLED
    if outcome.with_oracle_correct:
        return RetrievalState.RETRIEVAL_INSUFFICIENT
    return RetrievalState.REASONING_FAILURE
