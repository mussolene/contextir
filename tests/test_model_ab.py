from __future__ import annotations

import unittest

from scripts.evaluate_model_ab import (
    Case,
    aggregate,
    aggregate_usage,
    bootstrap_mean_ci,
    compare_modes,
    make_synthetic_case,
    score,
)


class ModelABHarnessTests(unittest.TestCase):
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
