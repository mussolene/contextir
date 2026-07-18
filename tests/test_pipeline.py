from __future__ import annotations

import unittest

from contextir import ContextPipeline, PipelinePolicy, ResponseVerification


LONG_TRANSFORM = " ".join(["Do not send payment 42 twice."] * 30)


class ContextPipelineTests(unittest.TestCase):
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
