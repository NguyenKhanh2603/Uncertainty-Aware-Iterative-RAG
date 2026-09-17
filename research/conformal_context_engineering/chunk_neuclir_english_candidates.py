"""Create CCE-compatible character chunks from extracted English NeuCLIR documents."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_INPUT = Path(
    "data/external/neuclir1_english/candidates/coveragebench_initial_qwen3_top100.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/external/neuclir1_english/candidates/coveragebench_initial_qwen3_top100_chunks_500_100.jsonl"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-characters", type=int, default=500)
    parser.add_argument("--overlap-characters", type=int, default=100)
    return parser.parse_args()


def chunks(text: str, width: int, overlap: int):
    """Pack complete sentences to approximately ``width`` characters.

    Overlap is measured in source characters but the next chunk always starts
    at a sentence boundary, matching CCE's stated boundary-handling rule.
    """
    text = " ".join(text.split())
    if width < 1 or not text:
        return
    if overlap >= width:
        raise ValueError("overlap must be less than chunk size")
    sentences = [match.group(0).strip() for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text) if match.group(0).strip()]
    if not sentences:
        sentences = [text]
    locations = []
    cursor = 0
    for sentence in sentences:
        location = text.find(sentence, cursor)
        locations.append(location)
        cursor = location + len(sentence)
    index = 0
    while index < len(sentences):
        first = index
        selected = []
        length = 0
        while index < len(sentences):
            candidate = len(sentences[index]) + (1 if selected else 0)
            if selected and length + candidate > width:
                break
            selected.append(sentences[index])
            length += candidate
            index += 1
        if not selected:  # A single sentence longer than the requested size.
            selected.append(sentences[index])
            index += 1
        yield locations[first], " ".join(selected)
        next_start = max(locations[first] + 1, locations[index - 1] + len(sentences[index - 1]) - overlap)
        while index < len(sentences) and locations[index] < next_start:
            index += 1


def main() -> None:
    args = arguments()
    if args.chunk_characters < 1:
        raise ValueError("chunk size must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document_count = chunk_count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            document = json.loads(line)
            document_count += 1
            docid = str(document["id"])
            for index, (start, text) in enumerate(
                chunks(str(document.get("text", "")), args.chunk_characters, args.overlap_characters)
            ):
                target.write(
                    json.dumps(
                        {
                            "chunk_id": f"{docid}::{index}",
                            "docid": docid,
                            "chunk_index": index,
                            "character_start": start,
                            "title": str(document.get("title", "")),
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                chunk_count += 1
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "documents": document_count,
        "chunks": chunk_count,
        "chunk_characters": args.chunk_characters,
        "overlap_characters": args.overlap_characters,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
