from __future__ import annotations

import unittest

from contextir import ContextIR, ContextPipeline, ContextWindowExceeded, PipelinePolicy, ResponseVerification


LONG_TRANSFORM = " ".join(["Do not send payment 42 twice."] * 30)


class ContextPipelineTests(unittest.TestCase):
    def test_pipeline_can_store_default_invoker(self) -> None:
        pipeline = ContextPipeline(invoke=lambda _prompt: "READY")

        result = pipeline.run("Reply with READY.", source_lang="en", target_lang="en")

        self.assertTrue(result.accepted)
        self.assertEqual(result.answer, "READY")

    def test_run_invoker_overrides_pipeline_default(self) -> None:
        pipeline = ContextPipeline(invoke=lambda _prompt: "default")

        result = pipeline.run("Reply briefly.", invoke=lambda _prompt: "override")

        self.assertEqual(result.answer, "override")

    def test_pipeline_requires_an_invoker(self) -> None:
        with self.assertRaisesRegex(ValueError, "invoke is required"):
            ContextPipeline().run("Reply briefly.")

    def test_successful_candidate_uses_one_compilation_pass(self) -> None:
        class CountingGateway(ContextIR):
            def __init__(self) -> None:
                super().__init__()
                self.modes = []

            def compile_private(self, *args, **kwargs):
                self.modes.append(kwargs.get("mode", args[4] if len(args) > 4 else "auto"))
                return super().compile_private(*args, **kwargs)

        gateway = CountingGateway()
        context = " ".join(f"Record {index}: Cedar value is {1000 + index}." for index in range(60))
        text = (
            f"Read the following text and answer briefly. {context} "
            "The Juniper access phrase is cobalt-seven. "
            "Question: What is the Juniper access phrase? Answer:"
        )

        prepared = ContextPipeline(gateway=gateway).prepare(text, source_lang="en", target_lang="en")

        self.assertEqual(prepared.mode, "hybrid")
        self.assertEqual(gateway.modes, ["auto"])

    def test_exhaustive_auto_result_does_not_compile_raw_twice(self) -> None:
        class CountingGateway(ContextIR):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def compile_private(self, *args, **kwargs):
                self.calls += 1
                return super().compile_private(*args, **kwargs)

        gateway = CountingGateway()
        paragraphs = " ".join(f"Paragraph {index}: value {index}." for index in range(40))

        prepared = ContextPipeline(gateway=gateway).prepare(
            f"How many unique paragraphs remain after removing duplicates? {paragraphs}",
            source_lang="en",
            target_lang="en",
        )

        self.assertEqual(prepared.mode, "raw")
        self.assertEqual(gateway.calls, 1)

    def test_short_input_stays_raw(self) -> None:
        prepared = ContextPipeline().prepare("Summarize this note.", source_lang="en", target_lang="en")

        self.assertEqual(prepared.mode, "raw")
        self.assertEqual(prepared.decision, "short_input")

    def test_semantic_mode_requires_measured_savings(self) -> None:
        prepared = ContextPipeline().prepare(LONG_TRANSFORM, source_lang="en", target_lang="en", risk="low")

        self.assertEqual(prepared.mode, "semantic")
        self.assertGreaterEqual(prepared.token_savings, 0.15)

    def test_document_qa_uses_retrieved_hybrid_context(self) -> None:
        context = " ".join(f"Record {index}: Cedar value is {1000 + index}." for index in range(60))
        text = (
            f"Read the following text and answer briefly. {context} "
            "The Juniper access phrase is cobalt-seven. "
            "Question: What is the Juniper access phrase? Answer:"
        )

        prepared = ContextPipeline().prepare(text, source_lang="en", target_lang="en")

        self.assertEqual(prepared.mode, "hybrid")
        self.assertIn("cobalt-seven", prepared.prompt)
        self.assertGreater(prepared.token_savings, 0.5)

    def test_custom_tokenizer_can_force_raw_fallback(self) -> None:
        def expensive_protocol(text: str) -> int:
            return 1000 if text.startswith("CTXIR/") else max(len(text.split()), 1)

        prepared = ContextPipeline(token_counter=expensive_protocol).prepare(
            LONG_TRANSFORM,
            source_lang="en",
            target_lang="en",
            risk="low",
        )

        self.assertEqual(prepared.mode, "raw")
        self.assertEqual(prepared.decision, "insufficient_token_savings")

    def test_model_budget_rejects_oversized_prompt_before_invocation(self) -> None:
        class TinyInvoker:
            prompt_token_budget = 3

            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, _prompt: str) -> str:
                self.calls += 1
                return "should not run"

        invoker = TinyInvoker()
        pipeline = ContextPipeline(invoke=invoker)

        with self.assertRaises(ContextWindowExceeded) as raised:
            pipeline.run("Reply with only READY.", source_lang="en", target_lang="en")

        self.assertEqual(invoker.calls, 0)
        self.assertEqual(raised.exception.prompt_budget, 3)
        self.assertNotIn("Reply with only READY", str(raised.exception))

    def test_policy_budget_is_exposed_on_prepared_context(self) -> None:
        pipeline = ContextPipeline(policy=PipelinePolicy(max_prompt_tokens=64))

        prepared = pipeline.prepare(LONG_TRANSFORM, source_lang="en", target_lang="en", risk="high")

        self.assertEqual(prepared.prompt_budget, 64)
        self.assertTrue(prepared.fits_prompt_budget)

    def test_oversized_fallback_is_not_sent_to_model(self) -> None:
        responses = iter(["Payment completed."])
        calls = []
        pipeline = ContextPipeline(policy=PipelinePolicy(max_prompt_tokens=30))

        result = pipeline.run(
            LONG_TRANSFORM,
            lambda prompt: calls.append(prompt) or next(responses),
            source_lang="en",
            target_lang="en",
            risk="high",
            task="transform",
        )

        self.assertFalse(result.accepted)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result.attempts), 1)
        self.assertIn("fallback_exceeds_prompt_budget", result.attempts[0].verification.reasons)

    def test_policy_rejects_invalid_prompt_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_prompt_tokens"):
            PipelinePolicy(max_prompt_tokens=0)
        with self.assertRaisesRegex(ValueError, "max_prompt_tokens"):
            PipelinePolicy(max_prompt_tokens=1.5)  # type: ignore[arg-type]

    def test_transform_retries_with_source_after_semantic_loss(self) -> None:
        responses = iter(["Payment completed.", "Do not send payment 42 twice."])
        pipeline = ContextPipeline()

        result = pipeline.run(
            LONG_TRANSFORM,
            lambda _prompt: next(responses),
            source_lang="en",
            target_lang="en",
            risk="low",
            task="transform",
        )

        self.assertTrue(result.accepted)
        self.assertEqual([item.mode for item in result.attempts], ["semantic", "hybrid"])
        self.assertIn("semantic_retention", result.attempts[0].verification.reasons)

    def test_reasoning_does_not_require_echoing_the_request(self) -> None:
        result = ContextPipeline().run(
            "What is the safest next step?",
            lambda _prompt: "Ask the operator to confirm.",
            source_lang="en",
            target_lang="en",
            task="reasoning",
        )

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.attempts), 1)

    def test_unknown_placeholder_and_new_pii_are_rejected(self) -> None:
        pipeline = ContextPipeline(policy=PipelinePolicy(max_attempts=1))

        unknown = pipeline.run(
            "Draft a response.",
            lambda _prompt: "Contact PII_EMAIL_99.",
            source_lang="en",
            target_lang="en",
        )
        new_pii = pipeline.run(
            "Draft a response.",
            lambda _prompt: "Contact leaked@example.test.",
            source_lang="en",
            target_lang="en",
        )

        self.assertFalse(unknown.accepted)
        self.assertIn("unknown_placeholders", unknown.attempts[0].verification.reasons)
        self.assertFalse(new_pii.accepted)
        self.assertIn("new_pii", new_pii.attempts[0].verification.reasons)

    def test_transform_requires_issued_placeholders(self) -> None:
        result = ContextPipeline(policy=PipelinePolicy(max_attempts=1)).run(
            "Translate for person@example.test.",
            lambda _prompt: "Translation without the recipient.",
            source_lang="en",
            target_lang="en",
            task="transform",
        )

        self.assertFalse(result.accepted)
        self.assertIn("missing_placeholders", result.attempts[0].verification.reasons)

    def test_restoration_is_explicit_and_trace_contains_no_payloads(self) -> None:
        pipeline = ContextPipeline()
        text = "Reply to person@example.test."

        hidden = pipeline.run(text, lambda _prompt: "Use PII_EMAIL_1.", source_lang="en", target_lang="en")
        restored = pipeline.run(
            text,
            lambda _prompt: "Use PII_EMAIL_1.",
            source_lang="en",
            target_lang="en",
            allowed_restore={"PII_EMAIL_1"},
        )
        trace = restored.public_trace()

        self.assertEqual(hidden.answer, "Use PII_EMAIL_1.")
        self.assertEqual(restored.answer, "Use person@example.test.")
        self.assertNotIn("person@example.test", str(trace))
        self.assertNotIn("Use PII_EMAIL_1", str(trace))

    def test_custom_verifier_controls_domain_acceptance(self) -> None:
        def verifier(_prepared: object, response: str) -> ResponseVerification:
            accepted = response == "approved"
            return ResponseVerification(accepted, [] if accepted else ["domain_rule"], [], [], [])

        result = ContextPipeline(policy=PipelinePolicy(max_attempts=1)).run(
            "Check the domain rule.",
            lambda _prompt: "approved",
            source_lang="en",
            target_lang="en",
            verifier=verifier,
        )

        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
