from eval.run_webq_remote import auroc_for_incorrect_answers, record_to_chunks


def test_record_to_chunks_preserves_top_k_and_title():
    record = {
        "q_id": 7,
        "question": "Who?",
        "answers": ["Example"],
        "ctxs": [
            {"id": "a", "title": "First", "text": "Passage one", "score": "0.9"},
            {"id": "b", "title": "Second", "text": "Passage two", "score": "0.8"},
        ],
    }

    chunks = record_to_chunks(record, top_k=1)

    assert len(chunks) == 1
    assert chunks[0].id == "a"
    assert chunks[0].content == "First\nPassage one"
    assert chunks[0].metadata["retrieval_score"] == "0.9"


def test_auroc_uses_higher_uncertainty_for_incorrect_answers():
    # First answer is correct, remaining answers are incorrect. The score
    # orders every incorrect answer above the correct one.
    assert auroc_for_incorrect_answers([1, 0, 0], [0.1, 0.7, 0.9]) == 1.0


def test_auroc_returns_none_when_only_one_class_is_present():
    assert auroc_for_incorrect_answers([1, 1], [0.1, 0.2]) is None
