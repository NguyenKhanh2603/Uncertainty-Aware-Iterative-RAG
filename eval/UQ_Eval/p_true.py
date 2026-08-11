"""Standalone p(True) prompt construction copied from RAGU's p_true.py.

The online p(True) scorer needs a server that returns the first-token log
probabilities for choices A and B.  This module has no RAGU imports.
"""

from __future__ import annotations

import math
from typing import Iterable


def make_shot(question: str, most_likely_answer: str, brainstormed_answers: Iterable[str], correct: bool) -> str:
    answers = "".join(f"{answer.strip()} \n" for answer in [most_likely_answer, *brainstormed_answers])
    label = "A" if correct else "B"
    return (
        f"Question: {question}\nBrainstormed Answers: {answers}"
        f"Possible answer: {most_likely_answer}\nIs the possible answer:\nA) True\nB) False\n"
        f"The possible answer is: {label}"
    )


def make_p_true_prompt(
    question: str, most_likely_answer: str, brainstormed_answers: Iterable[str], few_shot_prompt: str = "", hint: bool = False,
) -> str:
    prompt = (few_shot_prompt + "\n") if few_shot_prompt else ""
    answers = "".join(f"{answer.strip()}\n" for answer in [*brainstormed_answers, most_likely_answer])
    prompt += f"Question: {question}\nBrainstormed Answers: {answers}Possible answer: {most_likely_answer}\n"
    if hint:
        return prompt + "Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:"
    return prompt + "Is the possible answer:\nA) True\nB) False\nThe possible answer is:"


def p_true_uncertainty(logprob_a: float) -> float:
    """RAGU's ``p_false_fixed`` score: ``1 - p(token=' A')``.

    RAGU directly evaluates the model likelihood of the completion `` A``; it
    does *not* renormalize only over A and B. ``logprob_a`` is therefore the
    raw next-token log probability from the completion model.
    """
    return 1.0 - math.exp(logprob_a)
