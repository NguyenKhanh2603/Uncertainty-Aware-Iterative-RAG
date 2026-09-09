import json
import sys

from scripts.prepare_conformal_reference_banks import main


def retrieval_row(dataset: str, qid: str) -> dict:
    return {
        "dataset": dataset,
        "qid": qid,
        "split_role": "calibration",
        "query_text": "What is shown?",
        "query_type": "pooled",
        "chunk_id": f"{qid}-chunk",
        "source_doc_id": f"{qid}-doc",
        "modality": "text",
        "rank": 1,
        "cosine_score": 0.5,
        "support_label": "false",
        "top_l": 1,
        "retriever_id": "retriever@rev1",
        "corpus_revision": "corpus@rev1",
        "preprocess_hash": "preprocess@rev1",
        "query_type_rule_id": "pooled-v1",
    }


def test_cli_combines_multiple_retrieval_logs(tmp_path, monkeypatch):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "banks.json"
    first.write_text(json.dumps(retrieval_row("mmqa", "q1")) + "\n", encoding="utf-8")
    second.write_text(json.dumps(retrieval_row("webqa", "q2")) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_conformal_reference_banks.py",
            "--input",
            str(first),
            str(second),
            "--output",
            str(output),
            "--rank-bins",
            "1-1",
            "--min-bank-size",
            "1",
        ],
    )

    main()

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["summary"]["input_rows"] == 2
    assert len(artifact["banks"]) == 2
    assert [item["path"] for item in artifact["source"]["files"]] == [
        "first.jsonl",
        "second.jsonl",
    ]
