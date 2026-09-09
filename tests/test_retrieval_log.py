import json

import numpy as np
import pytest
import torch

from scripts.generate_conformal_retrieval_log import (
    candidate_pool_top_l,
    exact_top_l,
    modality_aware_top_l,
    resolve_dtype,
)
from uncertainty_rag.core.conformal_retrieval import ConformalDataError
from uncertainty_rag.core.retrieval_log import (
    CorpusRecord,
    QueryRecord,
    frozen_query_type,
    load_bundle_records,
    retrieval_log_row,
    stable_split_role,
)


def test_stable_split_assigns_a_whole_query_deterministically():
    first = stable_split_role(
        "mmqa",
        "q1",
        seed=42,
        development_fraction=0.2,
        calibration_fraction=0.6,
    )
    second = stable_split_role(
        "mmqa",
        "q1",
        seed=42,
        development_fraction=0.2,
        calibration_fraction=0.6,
    )

    assert first == second
    assert first in {"development", "calibration", "test"}


def test_keyword_query_type_uses_question_only():
    assert frozen_query_type("What color is the flower?", "keyword_v1") == "visual_spatial"
    assert frozen_query_type("What percentage was reported?", "keyword_v1") == "number_table"
    assert frozen_query_type("Who founded the company?", "keyword_v1") == "fact_lookup"
    assert frozen_query_type("What color was shown after 1990?", "keyword_v1") == "mixed"


def test_bundle_loader_deduplicates_corpus_and_preserves_support(tmp_path):
    dataset_dir = tmp_path / "mmqa"
    dataset_dir.mkdir()
    rows = [
        {
            "qid": "q1",
            "question": "Question one?",
            "chunks": [{"id": "shared", "modality": "text", "content": "same", "is_support": True}],
            "metadata": {"source_split": "train"},
        },
        {
            "qid": "q2",
            "question": "Question two?",
            "chunks": [
                {"id": "shared", "modality": "text", "content": "same", "is_support": False}
            ],
            "metadata": {"source_split": "dev"},
        },
    ]
    questions_path = dataset_dir / "questions.jsonl"
    questions_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    corpus, queries = load_bundle_records(
        questions_path,
        bundle_root=tmp_path,
        dataset="mmqa",
    )

    assert len(corpus) == 1
    assert queries[0].support_ids == frozenset({"shared"})
    assert queries[1].support_ids == frozenset()
    assert queries[0].candidate_ids == ("shared",)


def test_normalized_official_bundle_uses_per_query_candidates(tmp_path):
    dataset_dir = tmp_path / "webqa"
    dataset_dir.mkdir()
    (dataset_dir / "corpus.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "gold", "modality": "image", "content": "https://example/gold.jpg"},
                {"id": "negative", "modality": "text", "content": "distractor"},
            ]
        ),
        encoding="utf-8",
    )
    (dataset_dir / "questions.jsonl").write_text(
        json.dumps(
            {
                "qid": "q1",
                "question": "What is shown?",
                "candidate_ids": ["negative", "gold"],
                "support_ids": ["gold"],
                "metadata": {"source_split": "dev"},
            }
        ),
        encoding="utf-8",
    )

    corpus, queries = load_bundle_records(
        dataset_dir / "questions.jsonl", bundle_root=tmp_path, dataset="webqa"
    )

    assert [record.chunk_id for record in corpus] == ["gold", "negative"]
    assert queries[0].candidate_ids == ("negative", "gold")
    assert queries[0].support_ids == frozenset({"gold"})


def test_retrieval_row_uses_explicit_closed_world_label():
    query = QueryRecord("mmqa", "q1", "Question?", frozenset({"gold"}), "train")
    negative = CorpusRecord("negative", "text", "content", "doc")

    row = retrieval_log_row(
        query=query,
        chunk=negative,
        split_role="calibration",
        query_type="pooled",
        rank=1,
        cosine_score=1.0000001,
        top_l=20,
        retriever_id="model@rev",
        corpus_revision_id="corpus@rev",
        preprocess_hash="preprocess@rev",
        query_type_rule_id="pooled-v1",
        non_support_label="false",
    )

    assert row["support_label"] == "false"
    assert row["cosine_score"] == 1.0


def test_split_fractions_are_validated():
    with pytest.raises(ConformalDataError, match="below 1"):
        stable_split_role(
            "mmqa",
            "q1",
            seed=42,
            development_fraction=0.5,
            calibration_fraction=0.5,
        )


def test_exact_top_l_merges_multiple_corpus_blocks():
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    corpus = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scores, indices = exact_top_l(
        queries,
        corpus,
        top_l=2,
        device="cpu",
        query_batch_size=1,
        corpus_block_size=1,
    )

    assert indices.tolist() == [[0, 1]]
    assert np.allclose(scores, [[1.0, 0.8]])


def test_explicit_embedding_dtype_is_reproducible_without_a_gpu():
    assert resolve_dtype("cpu", "float16") == torch.float16
    assert resolve_dtype("cpu", "auto") == torch.float32


def test_modality_aware_top_l_reserves_candidates_before_global_fill():
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    scores = np.asarray([1.0, 0.99, 0.98, 0.97, 0.5, 0.4], dtype=np.float32)
    corpus = np.stack((scores, np.sqrt(1.0 - scores**2)), axis=1)

    selected_scores, selected_indices = modality_aware_top_l(
        queries,
        corpus,
        ["text", "text", "text", "text", "image", "image"],
        top_l=3,
        min_per_modality=1,
        device="cpu",
        query_batch_size=1,
        corpus_block_size=2,
    )

    assert selected_indices.tolist() == [[0, 1, 4]]
    assert np.allclose(selected_scores, [[1.0, 0.99, 0.5]])


def test_modality_aware_top_l_rejects_an_impossible_quota():
    with pytest.raises(ValueError, match="cannot hold the modality minima"):
        modality_aware_top_l(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            ["text", "image"],
            top_l=1,
            min_per_modality=1,
            device="cpu",
            query_batch_size=1,
            corpus_block_size=2,
        )


def test_candidate_pool_top_l_is_ragged_and_never_crosses_query_pool():
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    corpus = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)

    scores, indices, stats = candidate_pool_top_l(
        queries,
        corpus,
        [[0, 1], [2, 3]],
        ["text", "text", "text", "text"],
        top_l=3,
        retrieval_mode="global",
        min_per_modality=1,
        device="cpu",
        query_batch_size=2,
    )

    assert [values.tolist() for values in indices] == [[0, 1], [2, 3]]
    assert [len(values) for values in scores] == [2, 2]
    assert stats["candidate_pairs"] == 4
