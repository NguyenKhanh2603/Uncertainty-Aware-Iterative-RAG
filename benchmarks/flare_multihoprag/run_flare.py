"""OpenAI-compatible Direct FLARE runner for the full MultiHop-RAG corpus.

The archived official implementation targets text-davinci-003. This adapter
preserves Direct FLARE's control flow while using Chat Completions logprobs:
generate a temporary next sentence, inspect its confidence, optionally retrieve
with a masked version of that sentence, and only then accept/regenerate it.
"""

from __future__ import annotations

import argparse
import codecs
import json
import math
import os
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from common import BM25Corpus, read_json, read_jsonl


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
IMPLEMENTATION_VERSION = "direct_flare_v4"
FINAL_RE = re.compile(
    r"(?:so\s+)?(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*[\"']?(.+?)[\"']?\s*$",
    re.I,
)
ANSWER_PREFIX_RE = re.compile(
    r"^(?:so\s+)?(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*",
    re.I,
)
SENTENCE_END_RE = re.compile(r"[.!?][\"']?(?=\s|$)")


@dataclass
class Generation:
    text: str
    tokens: list[str]
    logprobs: list[float]
    finish_reason: str
    pieces: list[str] | None = None
    token_alignment_ok: bool = True


SPECIAL_TOKEN_RE = re.compile(r"^(?:</?s>|<\|[^>]+\|>|\[(?:INST|/INST)\])$")


def fallback_decode_token(token: str) -> str:
    """Decode common tokenizer display forms when API byte data is absent."""
    if SPECIAL_TOKEN_RE.fullmatch(token):
        return ""
    return token.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")


def align_pieces_to_text(text: str, pieces: list[str]) -> tuple[list[str], bool]:
    """Make token pieces concatenate to canonical message.content text.

    vLLM may display SentencePiece labels such as ``▁Sam`` in logprob tokens.
    Those labels are metadata, not generated text. API byte fields normally
    produce an exact alignment; the proportional fallback keeps output clean
    if a backend omits or reports incompatible byte data.
    """
    if not pieces:
        return [], not text
    source = "".join(pieces)
    left = len(source) - len(source.lstrip())
    right = len(source.rstrip())
    if source[left:right] == text:
        aligned = []
        position = 0
        for piece in pieces:
            piece_end = position + len(piece)
            start = max(position, left)
            end = min(piece_end, right)
            aligned.append(source[start:end] if end > start else "")
            position = piece_end
        return aligned, True

    # Conservative fallback: preserve canonical text and token count. This is
    # only an approximate probability-to-character alignment, so expose it in
    # the trace instead of silently writing tokenizer labels into the answer.
    weights = [len(piece) for piece in pieces]
    total = sum(weights)
    if total == 0:
        aligned = [""] * len(pieces)
        aligned[-1] = text
        return aligned, False
    aligned = []
    consumed_weight = 0
    previous = 0
    for weight in weights:
        consumed_weight += weight
        boundary = round(len(text) * consumed_weight / total)
        aligned.append(text[previous:boundary])
        previous = boundary
    return aligned, False


def align_token_bytes_to_text(
    text: str,
    tokens: list[str],
    byte_values: list[list[int] | None],
) -> tuple[list[str], bool]:
    """Align a complete UTF-8 token-byte stream to decoded message text.

    Decoding each token independently is incorrect when a Unicode code point is
    split across tokens. Decode incrementally across the whole stream instead.
    Special tokens such as ``</s>`` are zero-width because Chat Completions may
    expose them in logprobs while omitting them from ``message.content``.
    """
    if len(tokens) != len(byte_values):
        return [""] * len(tokens), False
    byte_pieces: list[bytes] = []
    for token, values in zip(tokens, byte_values):
        if SPECIAL_TOKEN_RE.fullmatch(token):
            byte_pieces.append(b"")
        elif values is None:
            return [""] * len(tokens), False
        else:
            byte_pieces.append(bytes(values))

    source = b"".join(byte_pieces)
    target = text.encode("utf-8")
    left = len(source) - len(source.lstrip())
    right = len(source.rstrip())
    if source[left:right] != target:
        return [""] * len(tokens), False

    trimmed: list[bytes] = []
    position = 0
    for piece in byte_pieces:
        piece_end = position + len(piece)
        start = max(position, left)
        end = min(piece_end, right)
        trimmed.append(source[start:end] if end > start else b"")
        position = piece_end

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    decoded = [decoder.decode(piece, final=False) for piece in trimmed]
    tail = decoder.decode(b"", final=True)
    if tail:
        for index in range(len(decoded) - 1, -1, -1):
            if trimmed[index]:
                decoded[index] += tail
                break
    return decoded, "".join(decoded) == text


def generation_pieces(generation: Generation) -> tuple[list[str], bool]:
    if generation.pieces is not None:
        return generation.pieces, generation.token_alignment_ok
    decoded = [fallback_decode_token(token) for token in generation.tokens]
    return align_pieces_to_text(generation.text, decoded)


def format_documents(documents: list[dict]) -> str:
    blocks = []
    for rank, document in enumerate(documents, 1):
        blocks.append(
            f"[{rank}] {document['title']}\n"
            f"Source: {document.get('source', '')}; Date: {document.get('published_at', '')}\n"
            f"{document['text']}"
        )
    return "\n\n".join(blocks)


def make_messages(
    question: str,
    partial_answer: str,
    documents: list[dict],
    allow_final: bool,
) -> list[dict]:
    evidence = format_documents(documents) if documents else "(No retrieved evidence for this look-ahead.)"
    if allow_final:
        next_step = (
            "If the intermediate facts are sufficient, output exactly "
            "\"So the answer is: <short answer>.\" Otherwise, write exactly one next "
            "intermediate reasoning sentence."
        )
    else:
        next_step = (
            "Write exactly one intermediate reasoning sentence that establishes one fact "
            "needed for the answer. Do not give the final answer yet."
        )
    user = f"""Question: {question}

Reasoning written so far:
{partial_answer or '(nothing yet)'}

Current retrieved evidence:
{evidence}

{next_step}
Do not repeat an earlier sentence. Return one sentence only."""
    return [
        {
            "role": "system",
            "content": (
                "You answer multi-hop questions by writing a short chain of evidence-based "
                "reasoning, one sentence at a time. Each intermediate sentence should state "
                "a concrete fact. Never invent facts or citations."
            ),
        },
        {"role": "user", "content": user},
    ]


def make_final_messages(
    question: str,
    partial_answer: str,
    documents: list[dict],
) -> list[dict]:
    evidence = format_documents(documents) if documents else "(No retrieved evidence.)"
    user = f"""Question: {question}

Completed intermediate reasoning:
{partial_answer or '(none)'}

Current retrieved evidence:
{evidence}

Give the shortest answer that directly answers the question.
Output exactly one line in this format: So the answer is: <short answer>.
Do not add reasoning, explanation, citations, or another sentence."""
    return [
        {
            "role": "system",
            "content": (
                "You finalize a multi-hop question after evidence-based reasoning. "
                "Return only the requested final-answer line."
            ),
        },
        {"role": "user", "content": user},
    ]


def truncate_to_first_sentence(generation: Generation) -> Generation:
    """Match paper sentence iteration without exposing tokenizer token labels."""
    if not generation.tokens:
        return generation
    pieces, alignment_ok = generation_pieces(generation)
    boundary = SENTENCE_END_RE.search(generation.text)
    if not boundary or boundary.end() >= len(generation.text):
        return Generation(
            generation.text,
            generation.tokens,
            generation.logprobs,
            generation.finish_reason,
            pieces,
            alignment_ok,
        )

    text_end = boundary.end()
    consumed = 0
    keep = 0
    kept_pieces = []
    for piece in pieces:
        remaining = text_end - consumed
        if remaining <= 0:
            break
        kept_pieces.append(piece[:remaining])
        consumed += len(piece)
        keep += 1
        if consumed >= text_end:
            break
    return Generation(
        text=generation.text[:text_end].strip(),
        tokens=generation.tokens[:keep],
        logprobs=generation.logprobs[:keep],
        finish_reason="sentence_boundary",
        pieces=kept_pieces,
        token_alignment_ok=alignment_ok,
    )


def generate_next(
    client: OpenAI,
    model: str,
    question: str,
    partial_answer: str,
    documents: list[dict],
    look_ahead_tokens: int,
    seed: int,
    allow_final: bool,
    force_final: bool = False,
) -> Generation:
    messages = (
        make_final_messages(question, partial_answer, documents)
        if force_final
        else make_messages(question, partial_answer, documents, allow_final)
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=look_ahead_tokens,
        logprobs=True,
        top_logprobs=1,
        seed=seed,
    )
    choice = response.choices[0]
    content = choice.logprobs.content if choice.logprobs and choice.logprobs.content else []
    text = (choice.message.content or "").strip()
    if text.strip() and not content:
        raise RuntimeError(
            "The model endpoint returned text without token log-probabilities; "
            "Direct FLARE cannot make retrieval decisions."
        )
    tokens = [token.token for token in content]
    byte_values = [getattr(token, "bytes", None) for token in content]
    pieces, alignment_ok = align_token_bytes_to_text(text, tokens, byte_values)
    if not alignment_ok:
        fallback_pieces = [fallback_decode_token(token) for token in tokens]
        fallback_aligned, fallback_ok = align_pieces_to_text(text, fallback_pieces)
        if fallback_ok:
            pieces, alignment_ok = fallback_aligned, True
    if text and not alignment_ok:
        raise RuntimeError(
            "Could not align endpoint logprob tokens to message.content. "
            "Refusing to run FLARE with approximate token probabilities. "
            f"Text prefix={text[:80]!r}; token prefix={tokens[:8]!r}"
        )
    generation = Generation(
        text=text,
        tokens=tokens,
        logprobs=[float(token.logprob) for token in content],
        finish_reason=choice.finish_reason or "",
        pieces=pieces,
        token_alignment_ok=alignment_ok,
    )
    if len(generation.tokens) != len(generation.logprobs):
        raise RuntimeError("Endpoint returned mismatched tokens and log-probabilities")
    return truncate_to_first_sentence(generation)


def masked_query(generation: Generation, beta: float, fallback: str) -> str:
    threshold = math.log(beta) if beta > 0 else -math.inf
    pieces, _ = generation_pieces(generation)
    kept = [
        piece if logprob > threshold else " "
        for piece, logprob in zip(pieces, generation.logprobs)
    ]
    query = " ".join("".join(kept).split())
    query = ANSWER_PREFIX_RE.sub("", query).strip().strip(".\"'")
    return query or fallback


def needs_retrieval(generation: Generation, theta: float) -> tuple[bool, list[str]]:
    if theta <= 0:
        return False, []
    threshold = math.log(theta)
    pieces, _ = generation_pieces(generation)
    low_tokens = [
        piece for piece, logprob in zip(pieces, generation.logprobs)
        if logprob <= threshold and piece.strip()
    ]
    return bool(low_tokens), low_tokens


def token_probability_trace(generation: Generation) -> list[dict]:
    pieces, _ = generation_pieces(generation)
    return [
        {"raw_token": token, "decoded_piece": piece, "probability": math.exp(logprob)}
        for token, piece, logprob in zip(generation.tokens, pieces, generation.logprobs)
    ]


def min_probability(generation: Generation) -> float | None:
    return min((math.exp(value) for value in generation.logprobs), default=None)


def extract_answer(text: str) -> str:
    match = FINAL_RE.search(text.strip().rstrip("."))
    if match:
        return match.group(1).strip().strip("\"'")
    return text.strip()


def compact_document(document: dict) -> dict:
    return {
        "id": document["id"],
        "title": document["title"],
        "url": document["url"],
        "score": document["score"],
    }


def run_jobs(items: list, worker, concurrency: int):
    """Yield every independent job outcome without cancelling sibling jobs."""
    def execute(item):
        try:
            return item, worker(item), None
        except Exception as exc:  # surfaced and re-raised by the main thread later
            return item, None, {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

    if concurrency == 1:
        yield from map(execute, items)
        return
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute, item) for item in items]
        for future in as_completed(futures):
            yield future.result()


def run_question(
    client: OpenAI,
    model: str,
    index: BM25Corpus,
    question: dict,
    top_k: int,
    theta: float,
    beta: float,
    max_steps: int,
    look_ahead_tokens: int,
    max_generation_tokens: int,
    min_reasoning_steps: int,
    seed: int,
) -> dict:
    query = question["query"]
    initial_documents = index.search(query, top_k)
    current_documents = initial_documents
    accepted_sentences: list[str] = []
    accepted_token_count = 0
    trace: list[dict] = []
    reasoning_stop_reason = "max_steps"
    finalization_text = ""

    for step in range(max_steps):
        partial = " ".join(accepted_sentences)
        allow_final = step >= min_reasoning_steps
        # The official 2Wiki configuration retrieves with the input before the
        # first look-ahead and keeps the current context (reinit_ctx=false)
        # until a later active retrieval replaces it (ctx_increase=replace).
        draft_context_documents = current_documents
        draft = generate_next(
            client,
            model,
            query,
            partial,
            draft_context_documents,
            look_ahead_tokens,
            seed + step * 2,
            allow_final,
        )
        if not draft.text:
            reasoning_stop_reason = "empty_draft"
            break

        retrieve, low_tokens = needs_retrieval(draft, theta)
        active_documents: list[dict] = []
        retrieval_query = ""
        accepted = draft
        if retrieve:
            retrieval_query = masked_query(draft, beta, fallback=query)
            active_documents = index.search(retrieval_query, top_k)
            current_documents = active_documents
            accepted = generate_next(
                client,
                model,
                query,
                partial,
                active_documents,
                look_ahead_tokens,
                seed + step * 2 + 1,
                allow_final,
            )

        trace.append({
            "step": step,
            "phase": "reasoning",
            "draft": draft.text,
            "draft_clean_text": draft.text,
            "draft_finish_reason": draft.finish_reason,
            "draft_token_alignment_ok": draft.token_alignment_ok,
            "draft_min_probability": min_probability(draft),
            "draft_token_probabilities": token_probability_trace(draft),
            "draft_context": [compact_document(document) for document in draft_context_documents],
            "low_confidence_tokens": low_tokens,
            "active_retrieval_triggered": retrieve,
            # Retained as a compatibility alias for older evaluation scripts.
            "retrieval_triggered": retrieve,
            "retrieval_query": retrieval_query,
            "retrieved": [compact_document(document) for document in active_documents],
            "accepted": accepted.text,
            "accepted_clean_text": accepted.text,
            "accepted_extracted_answer": extract_answer(accepted.text)
            if FINAL_RE.search(accepted.text.rstrip(".")) else None,
            "accepted_token_alignment_ok": accepted.token_alignment_ok,
            "accepted_after_retrieval": retrieve,
            "accepted_context": [compact_document(document) for document in current_documents],
        })
        if not accepted.text:
            reasoning_stop_reason = "empty_accepted_sentence"
            break
        accepted_sentences.append(accepted.text)
        accepted_token_count += len(accepted.tokens)

        # Crucially, termination is checked only after the temporary sentence
        # has passed through the uncertainty/retrieval decision above.
        if FINAL_RE.search(accepted.text.rstrip(".")):
            reasoning_stop_reason = "model_final_answer"
            break
        if accepted_token_count >= max_generation_tokens:
            reasoning_stop_reason = "generation_token_limit"
            break

    full_answer = " ".join(accepted_sentences).strip()
    finished_with_final_answer = bool(FINAL_RE.search(full_answer.rstrip(".")))
    forced_finalization_used = not finished_with_final_answer
    finalization_active_retrieval_triggered = False

    if forced_finalization_used:
        final_step = len(trace)
        final_seed = seed + max_steps * 2 + 100
        final_context_documents = current_documents
        final_draft = generate_next(
            client,
            model,
            query,
            full_answer,
            final_context_documents,
            look_ahead_tokens,
            final_seed,
            allow_final=True,
            force_final=True,
        )
        retrieve, low_tokens = needs_retrieval(final_draft, theta)
        finalization_active_retrieval_triggered = retrieve
        active_documents: list[dict] = []
        retrieval_query = ""
        final_accepted = final_draft
        if retrieve:
            retrieval_query = masked_query(final_draft, beta, fallback=query)
            active_documents = index.search(retrieval_query, top_k)
            current_documents = active_documents
            final_accepted = generate_next(
                client,
                model,
                query,
                full_answer,
                active_documents,
                look_ahead_tokens,
                final_seed + 1,
                allow_final=True,
                force_final=True,
            )
        finalization_text = final_accepted.text
        trace.append({
            "step": final_step,
            "phase": "finalization",
            "draft": final_draft.text,
            "draft_clean_text": final_draft.text,
            "draft_finish_reason": final_draft.finish_reason,
            "draft_token_alignment_ok": final_draft.token_alignment_ok,
            "draft_min_probability": min_probability(final_draft),
            "draft_token_probabilities": token_probability_trace(final_draft),
            "draft_context": [compact_document(document) for document in final_context_documents],
            "low_confidence_tokens": low_tokens,
            "active_retrieval_triggered": retrieve,
            "retrieval_triggered": retrieve,
            "retrieval_query": retrieval_query,
            "retrieved": [compact_document(document) for document in active_documents],
            "accepted": final_accepted.text,
            "accepted_clean_text": final_accepted.text,
            "accepted_extracted_answer": extract_answer(final_accepted.text)
            if FINAL_RE.search(final_accepted.text.rstrip(".")) else None,
            "accepted_token_alignment_ok": final_accepted.token_alignment_ok,
            "accepted_after_retrieval": retrieve,
            "accepted_context": [compact_document(document) for document in current_documents],
        })
        if final_accepted.text:
            accepted_sentences.append(final_accepted.text)
        full_answer = " ".join(accepted_sentences).strip()
        finished_with_final_answer = bool(FINAL_RE.search(full_answer.rstrip(".")))

    if reasoning_stop_reason == "model_final_answer":
        termination_reason = "model_final_answer"
    elif finished_with_final_answer:
        termination_reason = "forced_final_answer"
    else:
        termination_reason = "forced_finalization_failed"

    active_calls = sum(bool(item["active_retrieval_triggered"]) for item in trace)
    answer = extract_answer(full_answer)
    if not finished_with_final_answer and finalization_text:
        # Preserve a useful prediction even if a model disobeys the requested
        # wrapper; final_answer_rate still exposes the formatting failure.
        answer = finalization_text.strip()
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "query": query,
        "gold_answer": question["answer"],
        "question_type": question["question_type"],
        "gold_evidence": [
            {"title": evidence["title"], "url": evidence["url"], "fact": evidence["fact"]}
            for evidence in question["evidence_list"]
        ],
        "generation": full_answer,
        "answer": answer,
        "answer_extraction_debug": {
            "matched_final_answer_pattern": finished_with_final_answer,
            "extracted_answer": answer,
            "contains_tokenizer_artifacts": "▁" in full_answer or "</s>" in full_answer,
            "all_token_alignments_exact": all(
                step.get("draft_token_alignment_ok", False)
                and step.get("accepted_token_alignment_ok", False)
                for step in trace
            ),
        },
        "finished_with_final_answer": finished_with_final_answer,
        "reasoning_stop_reason": reasoning_stop_reason,
        "forced_finalization_used": forced_finalization_used,
        "finalization_active_retrieval_triggered": finalization_active_retrieval_triggered,
        "termination_reason": termination_reason,
        "initial_retrieved": [compact_document(document) for document in initial_documents],
        "trace": trace,
        "initial_retrieval_calls": 1,
        "active_retrieval_calls": active_calls,
        "retrieval_calls": 1 + active_calls,
        "model": model,
        "theta": theta,
        "beta": beta,
        "top_k": top_k,
        "look_ahead_tokens": look_ahead_tokens,
        "max_generation_tokens": max_generation_tokens,
        "min_reasoning_steps": min_reasoning_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Direct FLARE on MultiHop-RAG")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "data/multihop_rag/MultiHopRAG.json")
    parser.add_argument("--chunks", type=Path, default=HERE / "artifacts/corpus_chunks.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "results/flare_v4.jsonl")
    parser.add_argument(
        "--error-output", type=Path, default=None,
        help="JSONL for failed questions (default: <output stem>.errors.jsonl)",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--theta", type=float, default=0.8, help="Active retrieval trigger probability")
    parser.add_argument("--beta", type=float, default=0.4, help="Look-ahead query masking probability")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--look-ahead-tokens", "--max-tokens", dest="look_ahead_tokens", type=int, default=64,
        help="Maximum tokens in each temporary next-sentence prediction (paper default: 64)",
    )
    parser.add_argument("--max-generation-tokens", type=int, default=256)
    parser.add_argument(
        "--min-reasoning-steps", type=int, default=2,
        help="Require this many intermediate sentences before allowing a final answer",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Number of independent questions processed concurrently (default: 5)",
    )
    parser.add_argument("--include-null", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.theta <= 1 or not 0 <= args.beta <= 1:
        parser.error("theta and beta must be in [0, 1]")
    if min(
        args.top_k, args.max_steps, args.look_ahead_tokens,
        args.max_generation_tokens, args.concurrency,
    ) < 1:
        parser.error(
            "top-k, max-steps, look-ahead-tokens, max-generation-tokens, "
            "and concurrency must be positive"
        )
    if args.start < 0 or args.min_reasoning_steps < 0:
        parser.error("start and min-reasoning-steps must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    questions = read_json(args.queries)
    questions = questions[args.start :]
    if not args.include_null:
        questions = [question for question in questions if question["question_type"] != "null_query"]
    if args.max_examples is not None:
        questions = questions[: args.max_examples]

    completed = set()
    if args.resume and args.output.exists():
        existing = read_jsonl(args.output)
        wrong_versions = {
            row.get("implementation_version", "legacy") for row in existing
            if row.get("implementation_version") != IMPLEMENTATION_VERSION
        }
        if wrong_versions:
            raise ValueError(
                f"Cannot resume {IMPLEMENTATION_VERSION} into an old output containing {wrong_versions}; "
                "choose a new output path."
            )
        completed = {row["query"] for row in existing}
    elif args.output.exists():
        raise FileExistsError(f"Output exists; choose another path or pass --resume: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    error_output = args.error_output or args.output.with_name(f"{args.output.stem}.errors.jsonl")
    error_output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Implementation: {IMPLEMENTATION_VERSION}")
    print(
        f"Direct FLARE settings: theta={args.theta}, beta={args.beta}, top_k={args.top_k}, "
        f"look_ahead_tokens={args.look_ahead_tokens}, concurrency={args.concurrency}"
    )
    print(f"Loading BM25 index from {args.chunks}...")
    index = BM25Corpus.from_jsonl(args.chunks)
    print(f"Indexed {len(index.chunks)} chunks from the complete shared corpus.")

    pending = [
        (offset, question)
        for offset, question in enumerate(questions)
        if question["query"] not in completed
    ]
    thread_state = threading.local()

    def run_one(item: tuple[int, dict]) -> dict:
        offset, question = item
        if not hasattr(thread_state, "client"):
            thread_state.client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        return run_question(
            client=thread_state.client,
            model=args.model,
            index=index,
            question=question,
            top_k=args.top_k,
            theta=args.theta,
            beta=args.beta,
            max_steps=args.max_steps,
            look_ahead_tokens=args.look_ahead_tokens,
            max_generation_tokens=args.max_generation_tokens,
            min_reasoning_steps=args.min_reasoning_steps,
            seed=args.seed + args.start + offset,
        )

    failures = 0
    with args.output.open("a", encoding="utf-8") as output, error_output.open("a", encoding="utf-8") as errors:
        rows = run_jobs(pending, run_one, args.concurrency)
        for item, row, error in tqdm(rows, total=len(pending), desc="FLARE"):
            if error is not None:
                offset, question = item
                failure = {
                    "implementation_version": IMPLEMENTATION_VERSION,
                    "offset": args.start + offset,
                    "query": question["query"],
                    **error,
                }
                errors.write(json.dumps(failure, ensure_ascii=False) + "\n")
                errors.flush()
                failures += 1
                tqdm.write(
                    f"FAILED query {args.start + offset}: "
                    f"{error['exception_type']}: {error['message']}"
                )
                continue
            # Only the main thread writes, so every JSONL record remains
            # complete and immediately resumable after interruption.
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
    if failures:
        raise RuntimeError(
            f"{failures} question(s) failed; successful rows were saved to {args.output} "
            f"and failures to {error_output}. Re-run with --resume after fixing the cause."
        )


if __name__ == "__main__":
    main()
