from __future__ import annotations

import gzip
import inspect
import json
from types import SimpleNamespace

from eval.datasets.multimodalqa_loader import MultiModalQALoader
from uncertainty_rag.config import Config, PruningStrategy, ThresholdConfig
from uncertainty_rag.core.router import Router, RoutingDecision
from uncertainty_rag.core.uncertainty import UncertaintyProfile
from uncertainty_rag.models.llm_client import (
    HuggingFaceLocalClient,
    _find_delimited_token_spans,
)


def _write_jsonl_gz(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_multimodal_config_is_valid_yaml():
    config = Config.from_yaml("configs/multimodal.yaml")

    assert config.model.vlm_name == "Qwen/Qwen2-VL-7B-Instruct"
    assert config.pruning.strategy == PruningStrategy.ATTENTION_MASKING
    assert config.modality.type == "multimodal"


def test_local_hf_client_defaults_to_unquantized_fp16():
    default = inspect.signature(HuggingFaceLocalClient).parameters["load_in_4bit"].default

    assert default is False


def test_qwen_input_preparation_does_not_cap_image_resolution():
    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, **_kwargs):
            self.messages = messages
            return "prompt"

        def __call__(self, **_kwargs):
            return FakeInputs()

    client = HuggingFaceLocalClient.__new__(HuggingFaceLocalClient)
    client.is_qwen_vl = True
    client.processor = FakeProcessor()
    client.process_vision_info = lambda _messages: ([], [])
    client.model = SimpleNamespace(device="cpu")

    client._prepare_inputs(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/full-image.jpg"},
                    }
                ],
            }
        ]
    )

    image_block = client.processor.messages[0]["content"][0]
    assert image_block == {
        "type": "image",
        "image": "https://example.test/full-image.jpg",
    }


def test_image_spans_use_explicit_delimiters_not_repeated_tokens():
    input_ids = [1, 7, 7, 7, 10, 42, 42, 11, 2, 10, 99, 11]

    assert _find_delimited_token_spans(input_ids, 10, 11) == [(4, 8), (9, 12)]
    assert _find_delimited_token_spans([1, 7, 7, 7, 2], 10, 11) == []


def test_insufficient_evidence_forces_retrieval():
    router = Router(
        ThresholdConfig(
            tau_semantic=0.3,
            tau_token=0.5,
            tau_evidence=0.7,
        )
    )
    low_evidence = UncertaintyProfile(
        se_semantic=0.1,
        u_token=0.1,
        num_concepts=1,
        evidence_ratio=0.5,
    )
    sufficient_evidence = UncertaintyProfile(
        se_semantic=0.1,
        u_token=0.1,
        num_concepts=1,
        evidence_ratio=0.8,
    )

    assert router.decide(low_evidence) == RoutingDecision.RETRIEVE
    assert router.decide(sufficient_evidence) == RoutingDecision.STOP


def test_mmqa_loader_preserves_gold_support_and_image_metadata(tmp_path):
    _write_jsonl_gz(
        tmp_path / "MMQA_texts.jsonl.gz",
        [{"id": "txt1", "text": "A distractor."}],
    )
    _write_jsonl_gz(
        tmp_path / "MMQA_tables.jsonl.gz",
        [
            {
                "id": "tab1",
                "table": {
                    "header": [{"column_name": "Color"}],
                    "table_rows": [[{"text": "Yellow"}]],
                },
            }
        ],
    )
    _write_jsonl_gz(
        tmp_path / "MMQA_images.jsonl.gz",
        [
            {
                "id": "img1",
                "path": "flower.jpg",
                "title": "Wood Anemone",
                "url": "https://example.test/flower",
            }
        ],
    )
    (tmp_path / "final_dataset_images").mkdir()
    (tmp_path / "final_dataset_images" / "flower.jpg").write_bytes(b"test-image")

    _write_jsonl_gz(
        tmp_path / "MMQA_dev.jsonl.gz",
        [
            {
                "qid": "q1",
                "question": "What color is the center?",
                "answers": [{"answer": "Yellow", "type": "string"}],
                "metadata": {
                    "text_doc_ids": ["txt1"],
                    "table_id": "tab1",
                    "image_doc_ids": ["img1"],
                    "type": "Compose",
                    "modalities": ["table", "image"],
                },
                "supporting_context": [
                    {"doc_id": "tab1", "doc_part": "table"},
                    {"doc_id": "img1", "doc_part": "image"},
                ],
            }
        ],
    )

    loader = MultiModalQALoader(data_dir=str(tmp_path))
    example = loader.load()[0]
    chunks = {chunk.id: chunk for chunk in example.context_chunks}

    assert example.gold_answers == ["Yellow"]
    assert chunks["txt1"].metadata["is_support"] is False
    assert chunks["tab1"].metadata["is_support"] is True
    assert chunks["img1"].metadata["is_support"] is True
    assert chunks["img1"].metadata["title"] == "Wood Anemone"

    blocks = loader.get_modality_handler().format_context_for_prompt([chunks["img1"]])
    assert any(block.get("type") == "image_url" for block in blocks)
    assert "Wood Anemone" in loader.get_modality_handler().get_chunk_text_repr(chunks["img1"])
