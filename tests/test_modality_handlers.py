"""Unit tests for modality handlers: Text, Table, Image."""

from __future__ import annotations

import pytest

from uncertainty_rag.modality.base import ContextChunk
from uncertainty_rag.modality.text_handler import TextHandler
from uncertainty_rag.modality.table_handler import TableHandler, _table_to_markdown, _table_to_linearized
from uncertainty_rag.modality.image_handler import ImageHandler


class TestTextHandler:
    """Test TextHandler formatting."""

    def setup_method(self):
        self.handler = TextHandler()

    def test_format_single_chunk(self):
        chunks = [ContextChunk(id="1", content="Paris is the capital of France.", modality="text")]
        blocks = self.handler.format_context_for_prompt(chunks)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Paris is the capital" in blocks[0]["text"]

    def test_format_multiple_chunks(self):
        chunks = [
            ContextChunk(id="1", content="Fact A.", modality="text"),
            ContextChunk(id="2", content="Fact B.", modality="text"),
        ]
        blocks = self.handler.format_context_for_prompt(chunks)
        text = blocks[0]["text"]
        assert "[Doc 1]" in text
        assert "[Doc 2]" in text

    def test_empty_context(self):
        blocks = self.handler.format_context_for_prompt([])
        assert "[No context provided]" in blocks[0]["text"]

    def test_build_prompt_messages(self):
        chunks = [ContextChunk(id="1", content="Context here.", modality="text")]
        messages = self.handler.build_prompt_messages("What is X?", chunks, "Be precise.")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Question: What is X?" in str(messages[1]["content"])

    def test_retrieval_query(self):
        query = self.handler.format_for_retrieval_query(["claim1", "claim2"])
        assert query == "claim1 claim2"


class TestTableHandler:
    """Test TableHandler with markdown and linearized formats."""

    def test_table_dict_to_markdown(self):
        table = {"headers": ["Name", "Age"], "rows": [["Alice", "30"], ["Bob", "25"]]}
        md = _table_to_markdown(table)
        assert "| Name | Age |" in md
        assert "| Alice | 30 |" in md
        assert "| --- | --- |" in md

    def test_table_dict_to_linearized(self):
        table = {"headers": ["Name", "Age"], "rows": [["Alice", "30"]]}
        lin = _table_to_linearized(table)
        assert "Name: Alice" in lin
        assert "Age: 30" in lin

    def test_table_string_passthrough(self):
        assert _table_to_markdown("already formatted") == "already formatted"
        assert _table_to_linearized("already formatted") == "already formatted"

    def test_format_mixed_chunks(self):
        handler = TableHandler(table_format="markdown")
        chunks = [
            ContextChunk(
                id="t1",
                content={"headers": ["Year", "Revenue"], "rows": [["2024", "1M"]]},
                modality="table",
            ),
            ContextChunk(id="p1", content="The company grew 20%.", modality="text"),
        ]
        blocks = handler.format_context_for_prompt(chunks)
        text = blocks[0]["text"]
        assert "[TABLE 1]" in text
        assert "[TEXT 2]" in text
        assert "| Year | Revenue |" in text

    def test_chunk_text_repr(self):
        handler = TableHandler()
        table_chunk = ContextChunk(
            id="t1",
            content={"headers": ["A", "B"], "rows": [["1", "2"]]},
            modality="table",
        )
        text_chunk = ContextChunk(id="p1", content="Some text", modality="text")
        # Table should be linearized for NLI
        assert "A: 1" in handler.get_chunk_text_repr(table_chunk)
        assert handler.get_chunk_text_repr(text_chunk) == "Some text"


class TestImageHandler:
    """Test ImageHandler for multimodal content."""

    def setup_method(self):
        self.handler = ImageHandler(image_detail="high")

    def test_url_image(self):
        chunks = [
            ContextChunk(
                id="img1",
                content="https://example.com/image.jpg",
                modality="image",
                metadata={"caption": "A building"},
            ),
        ]
        blocks = self.handler.format_context_for_prompt(chunks)
        # Should produce: text label + image_url block
        has_image_url = any(b.get("type") == "image_url" for b in blocks)
        assert has_image_url

        # Check caption is included
        texts = [b["text"] for b in blocks if b.get("type") == "text"]
        assert any("A building" in t for t in texts)

    def test_mixed_image_text(self):
        chunks = [
            ContextChunk(id="img1", content="https://example.com/img.jpg", modality="image",
                         metadata={"caption": "Photo"}),
            ContextChunk(id="txt1", content="Some text fact.", modality="text"),
        ]
        blocks = self.handler.format_context_for_prompt(chunks)
        types = [b["type"] for b in blocks]
        assert "image_url" in types
        assert "text" in types

    def test_chunk_text_repr_image(self):
        chunk = ContextChunk(id="img1", content="data:image", modality="image",
                             metadata={"caption": "A red car"})
        repr_text = self.handler.get_chunk_text_repr(chunk)
        assert "[Image: A red car]" in repr_text

    def test_chunk_text_repr_no_caption(self):
        chunk = ContextChunk(id="img1", content="data:image", modality="image")
        repr_text = self.handler.get_chunk_text_repr(chunk)
        assert repr_text == "[Image]"

    def test_empty_context(self):
        blocks = self.handler.format_context_for_prompt([])
        assert "[No context provided]" in blocks[0]["text"]


class TestContextChunk:
    """Test ContextChunk dataclass."""

    def test_equality(self):
        a = ContextChunk(id="1", content="text", modality="text")
        b = ContextChunk(id="1", content="different", modality="text")
        assert a == b  # Equality is by ID

    def test_hash(self):
        a = ContextChunk(id="1", content="text", modality="text")
        b = ContextChunk(id="1", content="other", modality="text")
        assert hash(a) == hash(b)
        # Can be used in sets
        s = {a, b}
        assert len(s) == 1

    def test_inequality(self):
        a = ContextChunk(id="1", content="text", modality="text")
        c = ContextChunk(id="2", content="text", modality="text")
        assert a != c
