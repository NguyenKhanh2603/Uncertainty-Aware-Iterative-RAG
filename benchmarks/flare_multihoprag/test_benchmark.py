"""Dependency-light unit tests for the standalone FLARE benchmark."""

from __future__ import annotations

import math
import threading
import unittest
from unittest.mock import patch

from common import exact_match, token_f1
from evaluate import metrics
from run_flare import (
    Generation,
    align_pieces_to_text,
    align_token_bytes_to_text,
    extract_answer,
    masked_query,
    needs_retrieval,
    run_jobs,
    run_question,
    truncate_to_first_sentence,
)


class FakeIndex:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k):
        self.queries.append((query, top_k))
        return [{
            "id": str(len(self.queries)),
            "title": f"Document {len(self.queries)}",
            "url": f"https://example.test/{len(self.queries)}",
            "score": 1.0,
            "source": "test",
            "published_at": "",
            "text": "Evidence.",
        }]


QUESTION = {
    "query": "Who founded the company and where were they born?",
    "answer": "Test City",
    "question_type": "inference_query",
    "evidence_list": [],
}


class BenchmarkTests(unittest.TestCase):
    def test_retrieval_trigger(self):
        generation = Generation("A fact.", ["A", " fact", "."], [math.log(.9), math.log(.2), math.log(.9)], "stop")
        trigger, tokens = needs_retrieval(generation, theta=.8)
        self.assertTrue(trigger)
        self.assertEqual(tokens, [" fact"])

    def test_masked_query(self):
        generation = Generation("Joe guessed Paris", ["Joe", " guessed", " Paris"], [math.log(.9), math.log(.2), math.log(.9)], "stop")
        self.assertEqual(masked_query(generation, beta=.4, fallback="question"), "Joe Paris")

    def test_answer_extraction_and_metrics(self):
        self.assertEqual(extract_answer('Reason. The final answer is: "Sam Bankman-Fried"'), "Sam Bankman-Fried")
        self.assertEqual(extract_answer("Fact. So the answer is: Test City."), "Test City")
        self.assertEqual(exact_match("The Paris", "Paris"), 1.0)
        self.assertGreater(token_f1("Sam Bankman Fried", "Sam Bankman-Fried"), 0.0)

    def test_sentence_truncation_preserves_probabilities(self):
        generation = Generation(
            "First fact. Second fact.",
            ["First", " fact", ".", " Second", " fact", "."],
            [math.log(.9)] * 6,
            "stop",
        )
        result = truncate_to_first_sentence(generation)
        self.assertEqual(result.text, "First fact.")
        self.assertEqual(len(result.tokens), 3)
        self.assertEqual(len(result.logprobs), 3)

    def test_mistral_sentencepiece_labels_never_become_output_text(self):
        generation = Generation(
            "Sam Bankman-Fried is on trial. Another sentence.",
            [
                "▁Sam", "▁Bankman", "-", "Fried", "▁is", "▁on", "▁trial", ".",
                "▁Another", "▁sentence", ".", "</s>",
            ],
            [math.log(.9)] * 12,
            "stop",
        )
        result = truncate_to_first_sentence(generation)
        self.assertEqual(result.text, "Sam Bankman-Fried is on trial.")
        self.assertNotIn("▁", result.text)
        self.assertNotIn("</s>", result.text)
        self.assertTrue(result.token_alignment_ok)

    def test_mistral_masking_uses_decoded_pieces(self):
        generation = Generation(
            "Sam guessed Paris",
            ["▁Sam", "▁guessed", "▁Paris", "</s>"],
            [math.log(.9), math.log(.2), math.log(.9), math.log(.9)],
            "stop",
        )
        self.assertEqual(masked_query(generation, beta=.4, fallback="question"), "Sam Paris")

    def test_byte_piece_alignment_trims_special_outer_whitespace(self):
        pieces, ok = align_pieces_to_text("Sam Altman", [" Sam", " Altman", ""])
        self.assertTrue(ok)
        self.assertEqual("".join(pieces), "Sam Altman")

    def test_vllm_bytes_ignore_eos_payload_absent_from_content(self):
        text = "Google, as reported by The Verge, has spent billions."
        tokens = [
            "▁Google", ",", "▁as", "▁reported", "▁by", "▁The", "▁Ver", "ge",
            ",", "▁has", "▁spent", "▁billions", ".", "</s>",
        ]
        decoded = [
            " Google", ",", " as", " reported", " by", " The", " Ver", "ge",
            ",", " has", " spent", " billions", ".", "</s>",
        ]
        pieces, ok = align_token_bytes_to_text(
            text,
            tokens,
            [list(piece.encode("utf-8")) for piece in decoded],
        )
        self.assertTrue(ok)
        self.assertEqual("".join(pieces), text)
        self.assertEqual(pieces[-1], "")

    def test_vllm_bytes_decode_split_unicode_codepoint_incrementally(self):
        text = "publishers’ revenue"
        encoded = text.encode("utf-8")
        split = encoded.index("’".encode("utf-8")) + 1
        pieces, ok = align_token_bytes_to_text(
            text,
            ["publisher-fragment", "unicode-fragment"],
            [list(encoded[:split]), list(encoded[split:])],
        )
        self.assertTrue(ok)
        self.assertEqual("".join(pieces), text)

    @patch("run_flare.generate_next")
    def test_first_final_draft_is_checked_before_stopping(self, generate):
        generate.side_effect = [
            Generation("So the answer is: Wrong.", ["Wrong"], [math.log(.1)], "stop"),
            Generation("So the answer is: Test City.", ["Test City"], [math.log(.9)], "stop"),
        ]
        index = FakeIndex()
        row = run_question(
            None, "model", index, QUESTION, top_k=2, theta=.8, beta=.4,
            max_steps=3, look_ahead_tokens=64, max_generation_tokens=256,
            min_reasoning_steps=0, seed=10,
        )
        self.assertEqual(row["answer"], "Test City")
        self.assertTrue(row["trace"][0]["active_retrieval_triggered"])
        self.assertEqual(row["active_retrieval_calls"], 1)
        self.assertEqual(row["retrieval_calls"], 2)
        self.assertEqual(len(index.queries), 2)  # input retrieval + active retrieval

    @patch("run_flare.generate_next")
    def test_later_uncertain_draft_triggers_active_retrieval(self, generate):
        generate.side_effect = [
            Generation("The founder was Person.", ["The", " founder"], [math.log(.95)] * 2, "stop"),
            Generation("So the answer is: Guess.", ["Guess"], [math.log(.2)], "stop"),
            Generation("So the answer is: Test City.", ["Test City"], [math.log(.9)], "stop"),
        ]
        index = FakeIndex()
        row = run_question(
            None, "model", index, QUESTION, top_k=2, theta=.8, beta=.4,
            max_steps=3, look_ahead_tokens=64, max_generation_tokens=256,
            min_reasoning_steps=1, seed=10,
        )
        self.assertFalse(row["trace"][0]["active_retrieval_triggered"])
        self.assertTrue(row["trace"][1]["active_retrieval_triggered"])
        self.assertEqual(row["answer"], "Test City")

    def test_v3_evaluator_separates_initial_and_active_retrieval(self):
        row = {
            "answer": "Test City",
            "gold_answer": "Test City",
            "gold_evidence": [{"url": "gold-1"}, {"url": "gold-2"}],
            "initial_retrieved": [{"url": "gold-1"}, {"url": "noise"}],
            "trace": [{
                "active_retrieval_triggered": True,
                "retrieved": [{"url": "gold-2"}],
                "draft_min_probability": .2,
            }],
            "retrieval_calls": 2,
            "active_retrieval_calls": 1,
            "finished_with_final_answer": True,
            "reasoning_stop_reason": "generation_token_limit",
            "forced_finalization_used": True,
            "finalization_active_retrieval_triggered": True,
            "termination_reason": "forced_final_answer",
            "answer_extraction_debug": {
                "contains_tokenizer_artifacts": False,
                "all_token_alignments_exact": True,
                "matched_final_answer_pattern": True,
            },
        }
        result = metrics([row])
        self.assertEqual(result["initial_evidence_recall"], .5)
        self.assertEqual(result["cumulative_evidence_recall"], 1.0)
        self.assertEqual(result["avg_active_retrieval_calls"], 1.0)
        self.assertEqual(result["active_step_trigger_rate"], 1.0)
        self.assertEqual(result["examples_with_active_retrieval_rate"], 1.0)
        self.assertEqual(result["final_answer_rate"], 1.0)
        self.assertEqual(result["forced_finalization_rate"], 1.0)
        self.assertEqual(result["finalization_active_retrieval_rate"], 1.0)
        self.assertEqual(result["termination_reasons"], {"forced_final_answer": 1})
        self.assertEqual(result["clean_output_rate"], 1.0)
        self.assertEqual(result["all_token_alignments_exact_rate"], 1.0)
        self.assertEqual(result["answer_extraction_pattern_match_rate"], 1.0)

    @patch("run_flare.generate_next")
    def test_forced_finalization_is_uncertainty_checked(self, generate):
        generate.side_effect = [
            Generation("The founder was Person.", ["The", " founder"], [math.log(.95)] * 2, "stop"),
            Generation("So the answer is: Guess.", ["Guess"], [math.log(.2)], "stop"),
            Generation("So the answer is: Test City.", ["Test City"], [math.log(.9)], "stop"),
        ]
        index = FakeIndex()
        row = run_question(
            None, "model", index, QUESTION, top_k=2, theta=.8, beta=.4,
            max_steps=1, look_ahead_tokens=64, max_generation_tokens=256,
            min_reasoning_steps=2, seed=10,
        )
        self.assertTrue(row["forced_finalization_used"])
        self.assertTrue(row["finalization_active_retrieval_triggered"])
        self.assertEqual(row["reasoning_stop_reason"], "max_steps")
        self.assertEqual(row["termination_reason"], "forced_final_answer")
        self.assertEqual(row["trace"][-1]["phase"], "finalization")
        self.assertEqual(row["answer"], "Test City")

    def test_five_jobs_can_execute_concurrently(self):
        barrier = threading.Barrier(5, timeout=3)

        def worker(item):
            barrier.wait()
            return item

        outcomes = list(run_jobs(list(range(5)), worker, concurrency=5))
        self.assertCountEqual([result for _, result, error in outcomes if error is None], range(5))

    def test_worker_failure_does_not_cancel_sibling_jobs(self):
        def worker(item):
            if item == 2:
                raise ValueError("bad request")
            return item * 10

        outcomes = list(run_jobs([1, 2, 3], worker, concurrency=3))
        successes = {item: result for item, result, error in outcomes if error is None}
        failures = {item: error for item, result, error in outcomes if error is not None}
        self.assertEqual(successes, {1: 10, 3: 30})
        self.assertEqual(failures[2]["exception_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
