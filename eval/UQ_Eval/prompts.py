"""RAGU prompt templates copied locally for reproducible baseline evaluation.

Source: related_repos/ragu/utils/utils.py (BSD 3-Clause, Laura Perez et al.).
Only templates used by the QA/UQ experiments are retained here.
"""

from __future__ import annotations


PROMPT_DICT: dict[str, dict[str, str] | str] = {
    "prompt_directRagQA_REAR": (
        "Knowledge:\n{paragraph}\n\n"
        "Answer the following question with a very short phrase, such as \"1998\", "
        "\"May 16th, 1931\", or \"James Bond\", to meet the criteria of exact match "
        "datasets. \n\nQuestion: {instruction}\n\nAnswer: "
    ),
    "chat_directRagQA_REAR": {
        "system": "You are a helpful, respectful and honest assistant. ",
        "user": (
            "Knowledge:\n{paragraph}\n\n"
            "Answer the following question with a very short phrase, such as \"1998\", "
            "\"May 16th, 1931\", or \"James Bond\", to meet the criteria of exact match "
            "datasets. \n\nQuestion: {instruction}"
        ),
    },
    "chat_directRagQA_REAR2": {
        "system": "You are a helpful, respectful and honest assistant. ",
        "user": (
            "Knowledge:\n{paragraph}\n\n"
            "Answer the following question with a very short phrase, such as \"1998\", "
            "\"May 16th, 1931\", \"James Bond\", or \"Barack Obama and Joe Biden\", "
            "to meet the criteria of exact match datasets. \n\nQuestion: {instruction}"
        ),
    },
    "chat_directRagQA_REAR3": {
        "system": "You are a helpful assistant. ",
        "user": (
            "Knowledge:\n{paragraph}\n\n"
            "Answer the following question with a very short phrase.\n\n"
            "Question: {instruction}"
        ),
    },
    "chat_directRagQA_REAR3Llama": {
        "system": "You are a helpful assistant. Answer the user question with a very short phrase. ",
        "user": "Knowledge:\n{paragraph}\n\nQuestion: {instruction}",
    },
    "chat_directRagQA_REAR4": {
        "system": "You are a helpful, respectful and honest assistant. ",
        "user": (
            "Knowledge:\n{paragraph}\n\n"
            "Given these passages, answer the following question with a very short phrase."
            "Before even answering the question, consider whether you have sufficient information "
            "in the passages to answer the question fully.\n\nQuestion: {instruction}"
        ),
    },
    "prompt_accuracy_eval": (
        "You need to check whether the prediction of a question-answering system to a question is "
        "correct. You should make the judgment based on a list of ground truth answers provided to "
        "you. Your response should be \"correct\" if the prediction is correct or \"incorrect\" if "
        "the prediction is wrong.\n\nQuestion: {instruction}\nGround truth: {answers}\n"
        "Prediction: {output}\nCorrectness:"
    ),
}


STOP_SEQUENCES = [
    "\n\n\n\n", "\n\n\n", "\n\n", "\n", "$\n\n", "#\n\n", "+\n\n", "*\n\n",
    "$\n", "#\n", "+\n", "*\n", "/\n",
]


def make_messages(model_name: str, prompt_name: str, item: dict[str, object]) -> list[dict[str, str]]:
    """Reproduce RAGU's chat-message selection for Mistral/Gemma/Llama."""
    template = PROMPT_DICT[prompt_name]
    if not isinstance(template, dict):
        raise ValueError(f"{prompt_name} is a text-completion, not a chat prompt")
    user = template["user"].format_map(item)
    if "Meta-Llama-3.1" in model_name:
        return [{"role": "system", "content": template["system"]}, {"role": "user", "content": user}]
    if "gemma-2" in model_name or "Mistral" in model_name or "mistral" in model_name:
        # This intentionally omits the system message: it is what RAGU does for Mistral.
        return [{"role": "user", "content": user}]
    raise ValueError("RAGU has no chat-template rule for this model name: " + model_name)


def make_paragraph(record: dict[str, object], top_k: int) -> str:
    """Format RAGU's retrieved passages exactly as its generation script does."""
    contexts = record.get("ctxs")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"Record {record.get('q_id', '<unknown>')} has no ctxs")
    evidences = []
    for index, context in enumerate(contexts[:top_k], start=1):
        if not isinstance(context, dict):
            raise ValueError("Every ctx must be an object")
        evidences.append(f"[{index}] {context.get('title', '')}\n{context.get('text', '')}")
    return "\n".join(evidences)
