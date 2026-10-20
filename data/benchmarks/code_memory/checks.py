"""Historical ContextIR regressions selected with CASE_ID."""

from __future__ import annotations

import os
import unittest

from contextir import ContextIR, ContextPipeline, PipelinePolicy
from contextir.pipeline import ContextWindowExceeded, approximate_token_count
from contextir.sir_runtime import PrivacyScrubber


class ContextIRMemoryCases(unittest.TestCase):
    def _compile_retrieval(self, text: str, query: str):
        gateway = ContextIR()
        bundle = gateway.compile_private(
            text,
            source_lang="en",
            target_lang="en",
            mode="hybrid",
            context_kind="retrieval",
            query=query,
        )
        return gateway, bundle, gateway.render_bundle(bundle)

    def test_paragraph_keeps_related_sentence(self) -> None:
        paragraph = (
            "The amber relay passed its cold-start inspection. "
            "Its secondary latch must stay closed until pressure stabilizes."
        )
        text = (
            "Archive entry alpha: obsolete.\n\nArchive entry beta: obsolete.\n\n"
            f"{paragraph}\n\nArchive entry gamma: obsolete.\n\nArchive entry delta: obsolete."
        )
        _gateway, _bundle, prompt = self._compile_retrieval(
            text, "What passed the amber relay cold-start inspection?"
        )

        self.assertIn("The amber relay passed its cold-start inspection.", prompt)
        self.assertIn("Its secondary latch must stay closed until pressure stabilizes.", prompt)

    def test_list_keeps_sibling_items_and_marker(self) -> None:
        items = (
            "Approved service steps:\n\n"
            "- inspect the cobalt intake valve\n"
            "- record the pressure before restart\n"
            "- leave the backup seal in place"
        )
        text = (
            "Manual page zero: archived.\n\nManual page one: archived.\n\n"
            f"{items}\n\nManual page end: archived."
        )
        _gateway, _bundle, prompt = self._compile_retrieval(
            text, "Which service step inspects the cobalt intake valve?"
        )

        self.assertIn("- inspect the cobalt intake valve", prompt)
        self.assertIn("- record the pressure before restart", prompt)
        self.assertIn("- leave the backup seal in place", prompt)

    def test_fenced_code_keeps_dependent_guard(self) -> None:
        code = (
            "```python\n"
            "# cobalt gate routine verifies pressure and the backup latch\n"
            "def verify_cobalt_gate(state):\n"
            "    if state.ready:\n"
            "        return state.pressure_ok and state.backup_latch_closed\n"
            "    return False\n"
            "```"
        )
        text = (
            "Build note red: ignored.\n\nBuild note blue: ignored.\n\n"
            f"{code}\n\nBuild note gray: ignored."
        )
        _gateway, _bundle, prompt = self._compile_retrieval(
            text, "Which cobalt gate routine verifies pressure?"
        )

        self.assertIn("def verify_cobalt_gate(state):", prompt)
        self.assertIn("return state.pressure_ok and state.backup_latch_closed", prompt)
        self.assertIn("if state.ready:", prompt)

    def test_table_keeps_header_and_neighbor_row(self) -> None:
        table = (
            "| Station | sensor | vibration |\n"
            "| --- | --- | --- |\n"
            "| north pier | amber probe | 4.8 mm/s |\n"
            "| west pier | blue probe | 2.1 mm/s |"
        )
        text = (
            "Survey note one: retired.\n\nSurvey note two: retired.\n\n"
            f"Pier vibration readings:\n\n{table}\n\nSurvey note three: retired."
        )
        _gateway, _bundle, prompt = self._compile_retrieval(
            text, "What vibration did the amber probe report at the north pier?"
        )

        self.assertIn("| Station | sensor | vibration |", prompt)
        self.assertIn("| north pier | amber probe | 4.8 mm/s |", prompt)
        self.assertIn("| west pier | blue probe | 2.1 mm/s |", prompt)

    def test_decimal_and_signed_metrics_remain_exact(self) -> None:
        text = "score=0.765432198; error=-42.375; tolerance=+0.125."

        result = PrivacyScrubber().scrub(text)

        self.assertEqual(result.scrubbed_text, text)
        self.assertEqual(result.vault, {})

    def test_scientific_measurements_remain_exact(self) -> None:
        text = "mass=4.25e-7 kg; frequency=6.02E+9 Hz; residual=3.1e-8."

        result = PrivacyScrubber().scrub(text)

        self.assertEqual(result.scrubbed_text, text)
        self.assertEqual(result.vault, {})

    def test_grouped_integer_measurements_remain_exact(self) -> None:
        text = "Size: 2\u202f345\u202f678 bytes; total: 1 234 567 records; share: 7654321%."

        result = PrivacyScrubber().scrub(text)

        self.assertEqual(result.scrubbed_text, text)
        self.assertEqual(result.vault, {})

    def test_international_phone_remains_masked(self) -> None:
        result = PrivacyScrubber().scrub("Call +49 151 2345 6789 for the service desk.")

        self.assertNotIn("+49 151 2345 6789", result.scrubbed_text)
        self.assertEqual(list(result.vault.values()), ["+49 151 2345 6789"])

    def test_labeled_phone_remains_masked(self) -> None:
        result = PrivacyScrubber().scrub("Phone number: 2025550147.")

        self.assertNotIn("2025550147", result.scrubbed_text)
        self.assertEqual(list(result.vault.values()), ["2025550147"])

    def _crowded_document(self) -> tuple[str, str, str]:
        dominant_terms = (
            "nebula turbine blade fatigue cyclic loading rotor bearing lubricant "
            "thermal stress vibration resonance inspection endurance"
        )
        records = [
            f"Archive {index}: {dominant_terms} maintenance log batch {index}."
            for index in range(1, 9)
        ]
        answer = "Iris greenhouse seedling humidity target is 58 percent."
        records.append(answer)
        text = "\n\n".join(records)
        first = (
            "Which nebula turbine blade fatigue cyclic loading rotor bearing lubricant "
            "thermal stress vibration resonance inspection endurance report is requested?"
        )
        second = "What is the iris greenhouse seedling humidity target?"
        return text, first, second

    def test_labeled_questions_keep_evidence_for_each(self) -> None:
        text, first, second = self._crowded_document()
        query = f"Question A: {first}\nQuestion B: {second}"
        _gateway, _bundle, prompt = self._compile_retrieval(text, query)

        self.assertIn("Iris greenhouse seedling humidity target is 58 percent.", prompt)

    def test_separate_question_lines_keep_both_answers(self) -> None:
        text, first, second = self._crowded_document()
        _gateway, _bundle, prompt = self._compile_retrieval(text, f"{first}\n{second}")

        self.assertIn("Iris greenhouse seedling humidity target is 58 percent.", prompt)

    def test_line_wrapped_single_question_stays_retrievable(self) -> None:
        text = (
            "Service log red: archived.\n\n"
            "The lunar greenhouse night humidity target is 64 percent.\n\n"
            "Service log blue: archived."
        )
        query = "What is the lunar greenhouse\nnight humidity target?"
        _gateway, _bundle, prompt = self._compile_retrieval(text, query)

        self.assertIn("The lunar greenhouse night humidity target is 64 percent.", prompt)

    def test_budget_keeps_all_grouped_evidence_or_fails(self) -> None:
        text, first, second = self._crowded_document()
        query = f"{first}\n{second}"
        gateway, bundle, prompt = self._compile_retrieval(text, query)
        budget = approximate_token_count(prompt)
        packed = gateway._pack_retrieval_prompt(bundle, budget, approximate_token_count)

        self.assertIsNotNone(packed)
        self.assertIn("Iris greenhouse seedling humidity target is 58 percent.", gateway.render_bundle(packed))
        pipeline = ContextPipeline(
            gateway=gateway,
            policy=PipelinePolicy(min_token_savings=0),
        )
        accepted = 0
        for limit in range(1, budget + 1):
            with self.subTest(budget=limit):
                try:
                    prepared = pipeline.prepare(
                        text,
                        source_lang="en",
                        target_lang="en",
                        risk="high",
                        context_kind="retrieval",
                        query=query,
                        max_prompt_tokens=limit,
                    )
                except ContextWindowExceeded:
                    continue
                accepted += 1
                self.assertLessEqual(prepared.prompt_tokens, limit)
                self.assertLessEqual(approximate_token_count(prepared.prompt), limit)
                self.assertTrue(
                    any(record in prepared.prompt for record in text.split("\n\n")[:-1]),
                    "Missing complete Archive evidence for the first question",
                )
                self.assertIn(
                    "Iris greenhouse seedling humidity target is 58 percent.",
                    prepared.prompt,
                )
        self.assertGreater(accepted, 0)



def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    case_id = os.environ.get("CASE_ID", "")
    methods = {
        "structural-blocks": [
            "test_paragraph_keeps_related_sentence",
            "test_list_keeps_sibling_items_and_marker",
            "test_fenced_code_keeps_dependent_guard",
            "test_table_keeps_header_and_neighbor_row",
        ],
        "numeric-masking": [
            "test_decimal_and_signed_metrics_remain_exact",
            "test_scientific_measurements_remain_exact",
            "test_grouped_integer_measurements_remain_exact",
            "test_international_phone_remains_masked",
            "test_labeled_phone_remains_masked",
        ],
        "grouped-questions": [
            "test_labeled_questions_keep_evidence_for_each",
            "test_separate_question_lines_keep_both_answers",
            "test_line_wrapped_single_question_stays_retrievable",
            "test_budget_keeps_all_grouped_evidence_or_fails",
        ],
    }
    if case_id not in methods:
        raise ValueError(f"Set CASE_ID to one of: {', '.join(methods)}")
    return unittest.TestSuite(ContextIRMemoryCases(name) for name in methods[case_id])


if __name__ == "__main__":
    unittest.main()
