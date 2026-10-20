from __future__ import annotations

import copy
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from jsonschema import Draft202012Validator

from contextir import ContextIR, ContextPipeline, PipelinePolicy
from contextir.gateway import main, split_source
from contextir.pipeline import ContextWindowExceeded, approximate_token_count
from contextir.schemas import load_contract_schema
from contextir.sir_runtime import PresidioPrivacyScrubber, PrivacyScrubber


class ContextIRGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = ContextIR()

    def test_grouped_questions_preserve_independent_evidence_and_budget(self) -> None:
        topics = [
            f"Quasar telescope aperture calibration wavelength exposure result: run {index}."
            for index in range(8)
        ]
        table = "Botanical storage:\n\n| Seed | Temperature |\n| --- | --- |\n| Iris | 17 kelvin |"
        text = "\n\n".join(topics + [table] + ["Unrelated archive record. " * 80])
        questions = [
            "Which quasar telescope aperture calibration wavelength exposure result was recorded?",
            "What is the botanical storage temperature?",
        ]
        singles = [
            self.gateway.compile_private(
                text, source_lang="en", target_lang="en", mode="hybrid",
                context_kind="retrieval", query=question,
            )
            for question in questions
        ]
        wrapped = self.gateway.compile_private(
            text, source_lang="en", target_lang="en", mode="hybrid",
            context_kind="retrieval", query=questions[0].replace("aperture ", "aperture\n"),
        )
        self.assertEqual(wrapped.evidence_source_groups, singles[0].evidence_source_groups)
        self.assertEqual(wrapped.required_source_refs, singles[0].required_source_refs)
        for separator in ("\n", " "):
            query = separator.join(questions)
            with self.subTest(separator=repr(separator)):
                bundle = self.gateway.compile_private(
                    text, source_lang="en", target_lang="en", mode="hybrid",
                    context_kind="retrieval", query=query,
                )
                included = [item["ref"] for item in bundle.contract["source"]["included"]]
                for single in singles:
                    for group in single.evidence_source_groups:
                        self.assertTrue(set(group) <= set(included))
                        self.assertTrue(set(group) <= set(bundle.required_source_refs))
                self.assertEqual(included, [ref for ref in bundle.sources if ref in included])
                prompt = self.gateway.render_bundle(bundle)
                self.assertIn(table, prompt)
                budget = approximate_token_count(prompt)
                packed = self.gateway._pack_retrieval_prompt(bundle, budget, approximate_token_count)
                self.assertIsNotNone(packed)
                self.assertEqual(self.gateway.render_bundle(packed), prompt)
                pipeline = ContextPipeline(gateway=self.gateway, policy=PipelinePolicy(min_token_savings=0))
                prepared = pipeline.prepare(
                    text, source_lang="en", target_lang="en", risk="high",
                    context_kind="retrieval", query=query, max_prompt_tokens=budget,
                )
                self.assertLessEqual(prepared.prompt_tokens, budget)
                self.assertIn(table, prepared.prompt)
                with self.assertRaises(ContextWindowExceeded):
                    pipeline.prepare(
                        text, source_lang="en", target_lang="en", risk="high",
                        context_kind="retrieval", query=query, max_prompt_tokens=budget - 1,
                    )

    def test_structural_retrieval_preserves_atomic_blocks_and_layout(self) -> None:
        blocks = [
            ("The zircon system is approved. However, secondary certification is pending.", "zircon approved"),
            ("The zircon system is approved.\nIt must not be used without the secondary interlock.", "zircon approved"),
            ("Measurements at room temperature:\n\n| Product | Conductivity (W/m K) |\n| --- | --- |\n| Alloy A | 12 |\n| Alloy B | 17 |", "conductivity"),
            ("Allowed actions only after supervisor approval:\n\n- inspect the zircon valve\n- record pressure\n\n- close the hatch", "zircon valve"),
            ("```python\ndef zircon_check():\n    # Keep both conditions. Do not remove the second.\n\n    return ready and interlock\n```", "zircon_check"),
            ("Limits in kelvin:\n| Part | Maximum |\n| --- | --- |\n| zircon | 310 |\nRequired precautions:\n- inspect the valve\n- record pressure\n```text\nretain interlock\n```", "zircon maximum"),
        ]
        for block, query in blocks:
            with self.subTest(query=query):
                text = "Preface.\n\nArchive entry.\n\n" + block + "\n\nUnrelated tail.\n\nEnd."
                bundle = self.gateway.compile_private(
                    text, source_lang="en", target_lang="en", mode="hybrid",
                    context_kind="retrieval", query=query,
                )
                refs = tuple(ref for ref, value in split_source(text) if value in block)
                self.assertTrue(any(set(refs) <= set(group) for group in bundle.evidence_source_groups))
                self.assertEqual(bundle.sources, dict(split_source(text)))
                self.assertIn(block, self.gateway.render_bundle(bundle))
                self.assertNotIn("Unrelated tail.", self.gateway.render_bundle(bundle))
                self.assertEqual(bundle.contract["stats"]["prompt_chars"], len(self.gateway.render_bundle(bundle)))
                packed = self.gateway._pack_retrieval_prompt(bundle, 1000, len)
                self.assertIsNotNone(packed)
                self.assertIn(block, self.gateway.render_bundle(packed))
                self.assertEqual(packed.contract["stats"]["prompt_chars"], len(self.gateway.render_bundle(packed)))

    def test_short_input_uses_raw_without_wrapper_overhead(self) -> None:
        text = "Проверь архитектуру и верни краткий ответ."
        contract = self.gateway.compile(text, mode="auto")

        self.assertEqual(contract["mode"], "raw")
        self.assertEqual(self.gateway.render_prompt(contract), text)
        self.assertLessEqual(contract["stats"]["prompt_ratio"], 1.0)

    @patch("contextir.clients.OllamaClient")
    def test_run_cli_emits_answer_and_safe_trace(self, client_class) -> None:
        client_class.return_value.return_value = "READY"
        client_class.return_value.prompt_token_budget = 32512
        stdout = io.StringIO()
        argv = [
            "contextir",
            "run",
            "--model",
            "local-model",
            "--text",
            "Reply with READY.",
            "--chunked-retrieval",
            "--chunk-prompt-ratio",
            "0.8",
            "--json",
        ]

        with patch.object(sys, "argv", argv), redirect_stdout(stdout):
            main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["answer"], "READY")
        self.assertTrue(payload["trace"]["accepted"])
        self.assertNotIn("prompt", payload["trace"])

    @patch("contextir.clients.OllamaClient")
    def test_run_cli_masks_explicit_query(self, client_class) -> None:
        client_class.return_value.return_value = "READY"
        client_class.return_value.prompt_token_budget = 32512
        argv = [
            "contextir",
            "run",
            "--model",
            "local-model",
            "--text",
            "Project Juniper credential is cobalt-seven.",
            "--context-kind",
            "retrieval",
            "--query",
            "What should reviewer@example.test use?",
        ]

        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            main()

        prompt = client_class.return_value.call_args.args[0]
        self.assertNotIn("reviewer@example.test", prompt)
        self.assertIn("PII_EMAIL_1", prompt)

    @patch("contextir.clients.OllamaClient")
    def test_run_cli_reports_context_window_error_without_invoking_model(self, client_class) -> None:
        client_class.return_value.prompt_token_budget = 2
        stderr = io.StringIO()
        argv = [
            "contextir",
            "run",
            "--model",
            "tiny-model",
            "--text",
            "Reply with READY.",
        ]

        with patch.object(sys, "argv", argv), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("model budget is 2", stderr.getvalue())
        client_class.return_value.assert_not_called()

    def test_hybrid_preserves_condition_negation_number_and_privacy(self) -> None:
        secret = "person@example.test"
        bundle = self.gateway.compile_private(
            f"Если платеж 42 выполнен, не отправляй его повторно. Напиши {secret}.",
            source_lang="ru",
            target_lang="en",
            mode="hybrid",
        )
        contract = bundle.contract
        prompt = self.gateway.render_prompt(contract)

        self.assertNotIn(secret, json.dumps(contract, ensure_ascii=False))
        self.assertNotIn(secret, prompt)
        self.assertEqual(contract["privacy"]["protected"][0]["kind"], "email")
        self.assertTrue(any(event["condition"] == "if" and event["polarity"] == "negative" for event in contract["events"]))
        self.assertTrue(any(item["type"] == "number" and item["value"] == "42" for item in contract["entities"]))

    def test_restore_uses_bundle_allowlist(self) -> None:
        bundle = self.gateway.compile_private("Почта person@example.test, телефон +1 555 010-0100.", mode="raw")
        placeholders = list(bundle.vault)

        restored = self.gateway.restore(" / ".join(placeholders), bundle, allowed={placeholders[0]})

        self.assertIn(bundle.vault[placeholders[0]], restored)
        if len(placeholders) > 1:
            self.assertIn(placeholders[1], restored)

    def test_repeated_context_is_deduplicated(self) -> None:
        text = ("Агент завершил проверку и сохранил результат. " * 30) + "Если код 17 изменится, не публикуй результат."
        contract = self.gateway.compile(text, mode="hybrid")

        self.assertTrue(any(event["count"] == 30 for event in contract["events"]))
        self.assertLess(contract["stats"]["prompt_ratio"], 0.5)

    def test_repeated_pii_reuses_one_placeholder(self) -> None:
        secret = "finance@example.test"
        bundle = self.gateway.compile_private(" ".join([f"Do not email {secret}."] * 40), mode="semantic")

        self.assertEqual(bundle.vault, {"PII_EMAIL_1": secret})
        self.assertEqual(len(bundle.contract["privacy"]["protected"]), 1)
        self.assertEqual(bundle.contract["events"][0]["count"], 40)
        self.assertLess(bundle.contract["stats"]["prompt_ratio"], 0.25)

    def test_privacy_scrubber_classifies_card_before_phone(self) -> None:
        result = PrivacyScrubber().scrub("Use test card 4111 1111 1111 1111 in the sandbox.")

        self.assertEqual(result.vault, {"PII_CARD_1": "4111 1111 1111 1111"})
        self.assertEqual(result.protected_spans[0].kind, "card")

    def test_privacy_scrubber_does_not_mask_iso_date(self) -> None:
        text = "Invoice date 2026-07-18, order 12345678, amount 4200."

        result = PrivacyScrubber().scrub(text)

        self.assertEqual(result.scrubbed_text, text)
        self.assertEqual(result.vault, {})

    def test_privacy_scrubber_keeps_international_phone_that_passes_luhn(self) -> None:
        result = PrivacyScrubber().scrub("Call +447700677662 for support.")

        self.assertEqual(result.vault, {"PII_PHONE_1": "+447700677662"})
        self.assertEqual(result.protected_spans[0].kind, "phone")

    def test_privacy_scrubber_preserves_decimal_and_scientific_metrics(self) -> None:
        text = "PPL=12.345678, accuracy=0.987654321, loss=1.25e-6."

        result = PrivacyScrubber().scrub(text)

        self.assertEqual(result.scrubbed_text, text)
        self.assertEqual(result.vault, {})

    def test_privacy_scrubber_preserves_integer_measurements_with_units(self) -> None:
        text = "Artifact size: 123456789 bytes; token count: 987654321 tokens; share: 123456789%."

        result = PrivacyScrubber().scrub(text)

        self.assertEqual(result.scrubbed_text, text)
        self.assertEqual(result.vault, {})

    def test_privacy_scrubber_preserves_grouped_integer_measurements(self) -> None:
        for separator in (" ", "\u00a0", "\u202f"):
            text = f"Size: 198{separator}765{separator}432 байта."
            with self.subTest(separator=repr(separator)):
                result = PrivacyScrubber().scrub(text)

                self.assertEqual(result.scrubbed_text, text)
                self.assertEqual(result.vault, {})

    def test_privacy_scrubber_preserves_decimal_comma_and_negative_interval_bounds(self) -> None:
        text = "accuracy=0,987654321; interval=[-12.345678, -98.765432]."

        result = PrivacyScrubber().scrub(text)

        self.assertEqual(result.scrubbed_text, text)
        self.assertEqual(result.vault, {})

    def test_privacy_scrubber_does_not_join_separate_numbers_as_one_measurement(self) -> None:
        text = "Values 987654321 123456789 bytes."

        result = PrivacyScrubber().scrub(text)

        self.assertNotEqual(result.scrubbed_text, text)
        self.assertTrue(result.vault)
        self.assertTrue(all(span.kind in {"card", "phone"} for span in result.protected_spans))

    def test_privacy_scrubber_masks_ru_and_explicitly_labeled_phones(self) -> None:
        text = "Телефон: +7 921 555-01-23; phone number: 2025550147; phone: 202.5550147; 202.555.0147."

        result = PrivacyScrubber().scrub(text)

        self.assertEqual(
            list(result.vault.values()),
            ["+7 921 555-01-23", "2025550147", "202.5550147", "202.555.0147"],
        )
        self.assertEqual([span.kind for span in result.protected_spans], ["phone"] * 4)

    def test_covered_hybrid_events_render_as_plain_text(self) -> None:
        text = " ".join(
            ["Do not send payment 42 twice to finance@example.test."] * 20
            + ["Question: What action is prohibited? Answer with only the prohibited action."]
        )

        contract = self.gateway.compile(text, source_lang="en", mode="auto")
        prompt = self.gateway.render_prompt(contract)

        self.assertEqual(contract["mode"], "hybrid")
        self.assertNotIn("CTXIR/2", prompt)
        self.assertIn("What action is prohibited?", prompt)
        self.assertLess(contract["stats"]["prompt_ratio"], 0.25)

    def test_exhaustive_counting_routes_to_raw(self) -> None:
        paragraphs = " ".join(f"Paragraph {index}: value {index}." for index in range(30))
        text = f"How many unique paragraphs remain after removing duplicates? {paragraphs}"

        contract = self.gateway.compile(text, source_lang="en", mode="auto")

        self.assertEqual(contract["mode"], "raw")

    def test_document_qa_keeps_query_and_retrieved_evidence(self) -> None:
        distractors = " ".join(
            f"Record {index}: Project Cedar stores archived value {1000 + index}." for index in range(40)
        )
        text = (
            "Read the following text and answer briefly. "
            f"{distractors} The access phrase for Project Juniper is cobalt-seven. "
            "Question: What is the access phrase for Project Juniper? Answer:"
        )

        contract = self.gateway.compile(text, source_lang="en", mode="auto")
        included = " ".join(item["text"] for item in contract["source"]["included"])

        self.assertEqual(contract["mode"], "hybrid")
        self.assertIn("cobalt-seven", included)
        self.assertIn("What is the access phrase", included)
        self.assertGreater(contract["uncertainty"]["semantic_confidence"], 0.5)
        self.assertEqual(contract["events"], [])
        self.assertFalse(any(item["type"] == "number" for item in contract["entities"]))
        self.assertNotIn("CTXIR/2", self.gateway.render_prompt(contract))
        self.assertNotIn("SRC s", self.gateway.render_prompt(contract))
        self.assertLess(contract["stats"]["included_segments"], contract["stats"]["source_segments"])
        self.assertLess(contract["stats"]["prompt_ratio"], 0.5)

    def test_document_qa_without_matching_evidence_routes_to_raw(self) -> None:
        context = " ".join(f"Record {index}: Cedar stores value {1000 + index}." for index in range(30))
        text = (
            "Read the following text and answer briefly. "
            f"{context} Question: What is the lunar launch password? Answer:"
        )

        contract = self.gateway.compile(text, source_lang="en", mode="auto")

        self.assertEqual(contract["mode"], "raw")

    def test_explicit_query_routes_unlabelled_document_retrieval(self) -> None:
        text = " ".join(
            [f"Record {index}: Cedar deployment is archived." for index in range(12)]
            + ["Northern deployment operator credential is cobalt-seven."]
        )
        query = "Which credential should the operator use for the northern deployment?"

        bundle = self.gateway.compile_private(
            text,
            source_lang="en",
            target_lang="en",
            mode="auto",
            context_kind="retrieval",
            query=query,
        )
        prompt = self.gateway.render_bundle(bundle)

        self.assertEqual(bundle.contract["mode"], "hybrid")
        self.assertEqual(bundle.retrieval_query, query)
        self.assertIn("cobalt-seven", prompt)
        self.assertIn(query, prompt)
        self.assertNotIn(query, json.dumps(bundle.contract))

    def test_query_alone_implies_retrieval_in_auto_context_kind(self) -> None:
        text = " ".join(
            [f"Record {index}: Cedar deployment is archived." for index in range(12)]
            + ["Northern deployment operator credential is cobalt-seven."]
        )

        bundle = self.gateway.compile_private(
            text,
            source_lang="en",
            target_lang="en",
            mode="auto",
            query="What is the northern deployment credential?",
        )

        self.assertEqual(bundle.contract["mode"], "hybrid")
        self.assertTrue(bundle.evidence_source_groups)

    def test_document_and_query_share_one_privacy_namespace(self) -> None:
        text = "Contact owner@example.test about Project Juniper."
        query = "What should reviewer@example.test use?"
        bundle = self.gateway.compile_private(
            text,
            source_lang="en",
            target_lang="en",
            mode="raw",
            query=query,
        )
        prompt = self.gateway.render_bundle(bundle)

        self.assertEqual(
            bundle.vault,
            {
                "PII_EMAIL_1": "owner@example.test",
                "PII_EMAIL_2": "reviewer@example.test",
            },
        )
        self.assertNotIn("owner@example.test", prompt)
        self.assertNotIn("reviewer@example.test", prompt)
        self.assertIn("PII_EMAIL_1", prompt)
        self.assertIn("PII_EMAIL_2", prompt)
        self.assertEqual(bundle.contract["stats"]["source_chars"], len(text) + len(query))
        Draft202012Validator(load_contract_schema()).validate(bundle.contract)

    def test_explicit_context_kind_validation_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires query"):
            self.gateway.compile_private("Document text.", context_kind="retrieval")
        with self.assertRaisesRegex(ValueError, "only with auto or retrieval"):
            self.gateway.compile_private(
                "Document text.", context_kind="operational", query="What is relevant?"
            )

    def test_budget_packing_keeps_paragraph_owner_with_evidence(self) -> None:
        filler = " ".join(["archive"] * 35)
        paragraphs = " ".join(
            f"Paragraph {index}: Project Cedar archive. {filler}."
            for index in range(1, 8)
        )
        text = (
            "Here are paragraphs from a report. "
            f"{paragraphs} Paragraph 8: Project Juniper record. "
            "The current access phrase is cobalt-seven. "
            "Question: What is the current access phrase for Project Juniper? Answer:"
        )
        bundle = self.gateway.compile_private(text, source_lang="en", target_lang="en", mode="auto")

        packed = self.gateway._pack_retrieval_prompt(bundle, 100, approximate_token_count)

        self.assertIsNotNone(packed)
        prompt = self.gateway.render_prompt(packed.contract)  # type: ignore[union-attr]
        self.assertIn("Paragraph 8", prompt)
        self.assertIn("cobalt-seven", prompt)
        self.assertLessEqual(approximate_token_count(prompt), 100)

    def test_budget_packing_removes_privacy_metadata_for_pruned_evidence(self) -> None:
        filler = " ".join(["archive"] * 35)
        records = [
            f"Record {index}: Project Juniper access audit {filler}."
            for index in range(8)
        ]
        records.insert(1, "Project Juniper access audit owner is private@example.test.")
        records.insert(4, "The current access phrase for Project Juniper is cobalt-seven.")
        text = (
            "Read the following text and answer briefly. "
            + " ".join(records)
            + " Question: What is the current access phrase for Project Juniper? Answer:"
        )
        bundle = self.gateway.compile_private(text, source_lang="en", target_lang="en", mode="auto")

        packed = self.gateway._pack_retrieval_prompt(bundle, 80, approximate_token_count)

        self.assertIsNotNone(packed)
        prompt = self.gateway.render_prompt(packed.contract)  # type: ignore[union-attr]
        protected = packed.contract["privacy"]["protected"]  # type: ignore[union-attr]
        self.assertNotIn("private@example.test", prompt)
        self.assertFalse(protected)

        prepared = ContextPipeline(
            gateway=self.gateway,
            policy=PipelinePolicy(max_prompt_tokens=80),
        ).prepare(text, source_lang="en", target_lang="en")
        verification = ContextPipeline(gateway=self.gateway).verify_response(prepared, "PII_EMAIL_1")
        self.assertIn("PII_EMAIL_1", verification.unknown_placeholders)

    def test_compare_detects_lost_negation_and_number(self) -> None:
        expected = self.gateway.compile("Если платеж 42 выполнен, не отправляй его.", mode="semantic")
        observed = copy.deepcopy(expected)
        observed["events"][0]["polarity"] = "positive"
        observed["entities"] = []

        check = self.gateway.compare(expected, observed)

        self.assertEqual(check.event_recall, 0.0)
        self.assertEqual(check.entity_recall, 0.0)
        self.assertTrue(check.needs_source)

    def test_contract_matches_published_schema(self) -> None:
        contract = self.gateway.compile("Если задача 7 завершена, не запускай ее снова.", mode="hybrid")
        schema = load_contract_schema()

        Draft202012Validator(schema).validate(contract)

    def test_presidio_adapter_keeps_detected_values_in_local_vault(self) -> None:
        class Finding:
            start = 7
            end = 11
            score = 0.99
            entity_type = "PERSON"

        class Analyzer:
            def analyze(self, **kwargs):
                self.language = kwargs["language"]
                return [Finding()]

        analyzer = Analyzer()
        gateway = ContextIR(privacy=PresidioPrivacyScrubber(analyzer=analyzer))
        bundle = gateway.compile_private("Привет Анна", source_lang="ru", mode="raw")

        self.assertEqual(analyzer.language, "ru")
        self.assertEqual(bundle.vault["PII_PERSON_1"], "Анна")
        self.assertNotIn("Анна", gateway.render_prompt(bundle.contract))

    def test_presidio_adapter_reuses_placeholder_for_repeated_surface(self) -> None:
        class Finding:
            def __init__(self, start: int, end: int):
                self.start = start
                self.end = end
                self.score = 0.99
                self.entity_type = "PERSON"

        class Analyzer:
            def analyze(self, **_kwargs):
                return [Finding(0, 4), Finding(7, 11)]

        scrubber = PresidioPrivacyScrubber(analyzer=Analyzer())
        result = scrubber.scrub("Anna / Anna")

        self.assertEqual(result.scrubbed_text, "PII_PERSON_1 / PII_PERSON_1")
        self.assertEqual(result.vault, {"PII_PERSON_1": "Anna"})
        self.assertEqual(len(result.protected_spans), 1)


if __name__ == "__main__":
    unittest.main()
