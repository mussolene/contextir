from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from contextir import ContextIR
from contextir.sir_runtime import PresidioPrivacyScrubber


ROOT = Path(__file__).resolve().parents[1]


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
        schema = json.loads((ROOT / "schemas" / "contextir_contract_v2.schema.json").read_text(encoding="utf-8"))

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


if __name__ == "__main__":
    unittest.main()
