"""Table+Text modality handler for TAT-QA style hybrid contexts."""

from __future__ import annotations

from typing import Any

from uncertainty_rag.modality.base import ContextChunk, ModalityHandler


def _table_to_markdown(table: Any) -> str:
    """Convert a table (list-of-lists or dict) to markdown format."""
    if isinstance(table, str):
        return table

    if isinstance(table, dict):
        # {"headers": [...], "rows": [[...], ...]}
        headers = table.get("headers", [])
        rows = table.get("rows", [])
    elif isinstance(table, list) and len(table) > 0:
        headers = table[0] if isinstance(table[0], list) else []
        rows = table[1:] if isinstance(table[0], list) else table
    else:
        return str(table)

    if not headers:
        return str(table)

    # Build markdown table
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [str(c) for c in row]
        # Pad if needed
        while len(cells) < len(headers):
            cells.append("")
        lines.append("| " + " | ".join(cells[:len(headers)]) + " |")

    return "\n".join(lines)


def _table_to_linearized(table: Any) -> str:
    """Linearize a table into a flat text representation."""
    if isinstance(table, str):
        return table

    if isinstance(table, dict):
        headers = table.get("headers", [])
        rows = table.get("rows", [])
    elif isinstance(table, list) and len(table) > 0:
        headers = table[0] if isinstance(table[0], list) else []
        rows = table[1:] if isinstance(table[0], list) else table
    else:
        return str(table)

    parts = []
    for row in rows:
        row_parts = []
        for j, cell in enumerate(row):
            header = headers[j] if j < len(headers) else f"col{j}"
            row_parts.append(f"{header}: {cell}")
        parts.append("; ".join(row_parts))

    return " | ".join(parts)


class TableHandler(ModalityHandler):
    """Handles TAT-QA style hybrid table+text contexts.

    Tables are serialized as markdown (best for LLM comprehension) with explicit
    [TABLE] and [TEXT] markers. Row/column headers are preserved for numerical reasoning.
    """

    def __init__(self, table_format: str = "markdown") -> None:
        self.table_format = table_format

    def format_context_for_prompt(self, chunks: list[ContextChunk]) -> list[dict]:
        if not chunks:
            return [{"type": "text", "text": "[No context provided]"}]

        parts = ["Context:\n"]
        for i, chunk in enumerate(chunks):
            if chunk.modality == "table":
                if self.table_format == "markdown":
                    rendered = _table_to_markdown(chunk.content)
                elif self.table_format == "linearized":
                    rendered = _table_to_linearized(chunk.content)
                else:
                    rendered = str(chunk.content)
                parts.append(f"[TABLE {i + 1}]\n{rendered}\n")
            else:
                parts.append(f"[TEXT {i + 1}] {chunk.content}\n")

        return [{"type": "text", "text": "\n".join(parts)}]

    def get_chunk_text_repr(self, chunk: ContextChunk) -> str:
        if chunk.modality == "table":
            return _table_to_linearized(chunk.content)
        return str(chunk.content)

    def format_for_retrieval_query(self, claims: list[str]) -> str:
        return " ".join(claims)
