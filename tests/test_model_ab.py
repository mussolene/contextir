from __future__ import annotations

from argparse import Namespace
import unittest
from unittest.mock import patch

from contextir import ContextIR
from contextir.pipeline import approximate_token_count
from scripts.evaluate_model_ab import (
    Case,
    aggregate,
    aggregate_usage,
    bootstrap_mean_ci,
    build_comparisons,
    compare_modes,
    cosine_similarity,
    make_synthetic_case,
    normalize_summary,
    pack_ranked_evidence,
    render_embedding_answer_prompt,
    run_embedding_case,
    run_summary_case,
    score,
)


class ModelABHarnessTests(unittest.TestCase):
    @staticmethod
    def baseline_args() -> Namespace:
        return Namespace(
            backend="ollama",
            model="model",
            embedding_model="embed",
            ollama_base_url="http://127.0.0.1:11434",
            timeout=10,
            context_length=2048,
            max_output_tokens=64,
            prompt_overhead_tokens=32,
            summary_context_length=32768,
            summary_output_tokens=512,
        )

    @staticmethod
    def private_retrieval_case() -> Case:
        records = " ".join(f"Record {index}: Cedar is archived." for index in range(6))
        return Case(
            "private",
            "multifieldqa_en",
            "Read the following text and answer briefly. "
            + records
            + " Contact owner@example.test about Project Juniper. "
            + "Question: Who should be contacted about Project Juniper? Answer:",
            ["owner@example.test"],
            "test",
        )

    @staticmethod
    def usage(tokens: int = 10) -> dict[str, object]:
        return {
            "backend_prompt_tokens": tokens,
            "backend_output_tokens": 1,
            "backend_prompt_ms": 1.0,
            "backend_generation_ms": 1.0,
            "model_latency_ms": 2.0,
        }

    def test_bootstrap_interval_is_deterministic_and_bounded(self) -> None:
        interval = bootstrap_mean_ci([0.0, 0.5, 1.0, 1.0], samples=500, seed=7)

        self.assertEqual(interval, bootstrap_mean_ci([0.0, 0.5, 1.0, 1.0], samples=500, seed=7))
        self.assertGreaterEqual(interval[0], 0.0)
        self.assertLessEqual(interval[1], 1.0)
        self.assertEqual(bootstrap_mean_ci([0.0, 1.0], samples=0), [0.5, 0.5])

    def test_aggregate_usage_counts_all_chunk_calls(self) -> None:
        usage = aggregate_usage(
            [
                {"backend_prompt_tokens": 10, "backend_output_tokens": 2, "model_latency_ms": 3.5},
                {"backend_prompt_tokens": 12, "backend_output_tokens": 1, "model_latency_ms": 4.0},
            ]
        )

        self.assertEqual(usage["backend_prompt_tokens"], 22)
        self.assertEqual(usage["backend_output_tokens"], 3)
        self.assertEqual(usage["model_latency_ms"], 7.5)
        self.assertEqual(usage["model_calls"], 2)

    def test_aggregate_reports_failures_calls_and_confidence_interval(self) -> None:
        rows = [
            {
                "requested_mode": "chunked",
                "quality": quality,
                "error": error,
                "estimated_source_tokens": 100,
                "estimated_prompt_tokens": 40,
                "backend_prompt_tokens": 30,
                "model_latency_ms": 10.0,
                "model_calls": calls,
                "selected_mode": "hybrid",
                "decision": "retrieval_budget_packed",
                "pipeline_accepted": error is None,
                "pipeline_trace": {"attempts": [{"stage": "direct"}]},
            }
            for quality, error, calls in [(1.0, None, 1), (0.0, "rejected", 2)]
        ]

        result = aggregate(rows, bootstrap_samples=100)[0]

        self.assertEqual(result["mean_quality"], 0.5)
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["decisions"], {"retrieval_budget_packed": 2})
        self.assertEqual(result["pipeline_stages"], {"direct": 2})
        self.assertEqual(result["pipeline_accepted"], 1)
        self.assertEqual(len(result["quality_ci_95"]), 2)

    def test_passage_retrieval_scoring_requires_the_paragraph_identifier(self) -> None:
        case = Case("case", "passage_retrieval_en", "prompt", ["Paragraph 17"], "LongBench")

        self.assertEqual(score(case, "Paragraph 17"), 1.0)
        self.assertEqual(score(case, "17"), 0.0)

    def test_paired_mode_comparison_reports_regressions_and_resource_ratios(self) -> None:
        rows = []
        for case_id, raw, chunked in [("a", 0.0, 1.0), ("b", 0.5, 0.5), ("c", 1.0, 0.0)]:
            rows.extend(
                [
                    {
                        "case_id": case_id,
                        "requested_mode": "raw",
                        "quality": raw,
                        "backend_prompt_tokens": 100,
                        "model_latency_ms": 20,
                    },
                    {
                        "case_id": case_id,
                        "requested_mode": "chunked",
                        "quality": chunked,
                        "backend_prompt_tokens": 40,
                        "model_latency_ms": 10,
                    },
                ]
            )

        result = compare_modes(rows, "raw", "chunked", bootstrap_samples=100)

        self.assertIsNotNone(result)
        self.assertEqual(result["mean_quality_delta"], 0.0)
        self.assertEqual((result["improved"], result["tied"], result["regressed"]), (1, 1, 1))
        self.assertEqual(result["backend_prompt_token_ratio"], 0.4)
        self.assertEqual(result["model_latency_ratio"], 0.5)

        comparisons = build_comparisons(rows, ["raw", "chunked"])
        self.assertEqual([(item["baseline"], item["candidate"]) for item in comparisons], [("raw", "chunked")])

    def test_cosine_similarity_ranks_aligned_vectors(self) -> None:
        self.assertEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_summary_normalization_removes_only_the_leading_role_label(self) -> None:
        self.assertEqual(normalize_summary("Answer: cobalt-seven"), "cobalt-seven")
        self.assertEqual(normalize_summary("Notes: Answer: cobalt-seven"), "Answer: cobalt-seven")

    @patch("scripts.evaluate_model_ab.invoke_model")
    def test_summary_baseline_sends_only_masked_source(self, invoke_model) -> None:
        invoke_model.side_effect = [
            ("PII_EMAIL_1", self.usage()),
            ("PII_EMAIL_1", self.usage()),
        ]

        run_summary_case(self.baseline_args(), ContextIR(), self.private_retrieval_case())

        prompts = [call.args[1] for call in invoke_model.call_args_list]
        self.assertNotIn("owner@example.test", " ".join(prompts))
        self.assertIn("PII_EMAIL_1", prompts[0])

    @patch("scripts.evaluate_model_ab.invoke_model")
    @patch("scripts.evaluate_model_ab.post_json")
    def test_embedding_baseline_sends_only_masked_source(self, post_json, invoke_model) -> None:
        captured_inputs = []

        def embed(_url, payload, **_kwargs):
            captured_inputs.extend(payload["input"])
            return {
                "embeddings": [[1.0, 0.0] for _item in payload["input"]],
                "prompt_eval_count": 20,
            }

        post_json.side_effect = embed
        invoke_model.return_value = ("PII_EMAIL_1", self.usage())

        run_embedding_case(self.baseline_args(), ContextIR(), self.private_retrieval_case())

        self.assertNotIn("owner@example.test", " ".join(captured_inputs))
        self.assertIn("PII_EMAIL_1", " ".join(captured_inputs))

    def test_embedding_packer_keeps_rank_order_and_budget(self) -> None:
        query = "Which paragraph contains cobalt-seven?"
        first = "Paragraph 7: Project Juniper uses cobalt-seven."
        second = "Paragraph 9: Cedar is archived."
        one_group_budget = approximate_token_count(render_embedding_answer_prompt(query, [first]))

        prompt, selected = pack_ranked_evidence(query, [first, second], one_group_budget)

        self.assertEqual(selected, 1)
        self.assertIn(first, prompt)
        self.assertNotIn(second, prompt)

    def test_oversized_segment_fixture_keeps_answer_outside_the_raw_tail(self) -> None:
        case = make_synthetic_case(
            {
                "id": "oversized_segment",
                "dataset": "contextir_oversized_retrieval",
                "variant": "oversized_segment",
            }
        )

        answer_index = case.prompt.index("cobalt-seven")
        self.assertGreater(len(case.prompt), 20_000)
        self.assertLess(answer_index, len(case.prompt) // 3)
        self.assertEqual(score(case, "The answer is cobalt-seven."), 1.0)
        self.assertEqual(score(case, "The answer is ruby-nine."), 0.0)


if __name__ == "__main__":
    unittest.main()
