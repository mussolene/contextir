from __future__ import annotations

import copy
import json
import unittest

from jsonschema import Draft202012Validator

from contextir import ContextIR
from contextir.schemas import load_contract_schema
from contextir.sir_runtime import PresidioPrivacyScrubber


class ContextIRGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = ContextIR()

    def test_short_input_uses_raw_without_wrapper_overhead(self) -> None:
        text = "Проверь архитектуру и верни краткий ответ."
        contract = self.gateway.compile(text, mode="auto")

        self.assertEqual(contract["mode"], "raw")
        self.assertEqual(self.gateway.render_prompt(contract), text)
        self.assertLessEqual(contract["stats"]["prompt_ratio"], 1.0)

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
