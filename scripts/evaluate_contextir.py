#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextir import ContextIR, ContextPipeline, ContextWindowExceeded, PipelinePolicy
from contextir.sir_runtime import PrivacyScrubber
from contextir.sir_sources import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ContextIR compression, logic preservation, and privacy.")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "data" / "roundtrip" / "contextir_gateway_cases.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "contextir_gateway_eval.json"))
    parser.add_argument("--mode", choices=["auto", "raw", "hybrid", "semantic"], default="auto")
    parser.add_argument("--performance-iterations", type=int, default=5000)
    parser.add_argument("--max-eligible-ratio", type=float, default=0.6)
    parser.add_argument("--external-privacy-dataset", default="")
    parser.add_argument("--min-external-privacy-precision", type=float, default=0.8)
    parser.add_argument("--min-external-privacy-recall", type=float, default=0.98)
    parser.add_argument("--check", action="store_true", help="Exit non-zero when release gates fail.")
    args = parser.parse_args()

    gateway = ContextIR()
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for case in cases:
        bundle = gateway.compile_private(
            case["text"],
            source_lang=case["source_lang"],
            target_lang=case["target_lang"],
            packet_id=case["id"],
            mode=args.mode,
        )
        contract = bundle.contract
        prompt = gateway.render_prompt(contract)
        serialized = json.dumps(contract, ensure_ascii=False)
        original_values = set(bundle.vault.values())
        pii_leaked = any(value in serialized or value in prompt for value in original_values)
        critical_events = [event for event in contract["events"] if event["polarity"] == "negative" or event["condition"]]
        privacy_kinds = {item["kind"] for item in contract["privacy"]["protected"]}
        numbers = {item["value"] for item in contract["entities"] if item["type"] == "number"}
        expected_events = case.get("expected_events", [])
        event_checks = []
        for expected in expected_events:
            event_checks.append(
                any(
                    all(event.get(key) == value for key, value in expected.items())
                    for event in contract["events"]
                )
            )
        expectations_passed = (
            set(case.get("expected_privacy", [])) <= privacy_kinds
            and set(case.get("expected_numbers", [])) <= numbers
            and all(event_checks)
        )
        results.append(
            {
                "case_id": case["id"],
                "mode": contract["mode"],
                "events": len(contract["events"]),
                "critical_events": len(critical_events),
                "protected_spans": len(contract["privacy"]["protected"]),
                "pii_leaked": pii_leaked,
                "prompt_ratio": contract["stats"]["prompt_ratio"],
                "source_chars": contract["stats"]["source_chars"],
                "prompt_chars": contract["stats"]["prompt_chars"],
                "prompt": prompt,
                "expectations_passed": expectations_passed,
            }
        )
    total = max(len(results), 1)
    eligible = [row for row in results if row["source_chars"] > 240]
    aggregate = {
        "cases": len(results),
        "avg_prompt_ratio": round(sum(row["prompt_ratio"] for row in results) / total, 4),
        "compressed_cases": sum(1 for row in results if row["prompt_ratio"] < 1.0),
        "critical_events": sum(row["critical_events"] for row in results),
        "pii_leaks": sum(1 for row in results if row["pii_leaked"]),
        "expectation_failures": sum(1 for row in results if not row["expectations_passed"]),
        "compression_eligible_cases": len(eligible),
        "avg_eligible_prompt_ratio": round(sum(row["prompt_ratio"] for row in eligible) / max(len(eligible), 1), 4),
    }
    privacy_metrics = evaluate_privacy(cases)
    aggregate.update(privacy_metrics)
    pipeline_results = evaluate_pipeline(gateway)
    aggregate["pipeline_cases"] = len(pipeline_results)
    aggregate["pipeline_failures"] = sum(1 for item in pipeline_results if not item["passed"])
    aggregate["pipeline_fallbacks"] = sum(int(item.get("fallbacks", 0)) for item in pipeline_results)
    timings = []
    if args.performance_iterations > 0 and cases:
        def compile_case(index: int) -> None:
            case = cases[index % len(cases)]
            gateway.compile(
                case["text"],
                source_lang=case["source_lang"],
                target_lang=case["target_lang"],
                packet_id=case["id"],
                mode=args.mode,
            )

        for index in range(min(100, args.performance_iterations)):
            compile_case(index)
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            for index in range(args.performance_iterations):
                started = time.perf_counter()
                compile_case(index)
                timings.append((time.perf_counter() - started) * 1000)
        finally:
            if gc_enabled:
                gc.enable()
        ordered = sorted(timings)
        p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        aggregate["compile_latency_ms_p50"] = round(statistics.median(ordered), 4)
        aggregate["compile_latency_ms_p95"] = round(ordered[p95_index], 4)
        aggregate["compile_throughput_docs_s"] = round(1000 / statistics.mean(ordered), 1)
        aggregate["performance_iterations"] = len(ordered)
    external_privacy = (
        evaluate_external_privacy(Path(args.external_privacy_dataset))
        if args.external_privacy_dataset
        else None
    )
    report = {"aggregate": aggregate, "results": results, "pipeline_results": pipeline_results}
    if external_privacy:
        report["external_privacy"] = external_privacy
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    if args.check:
        failures = []
        if aggregate["pii_leaks"]:
            failures.append("PII leaked into a public contract or prompt")
        if aggregate["expectation_failures"]:
            failures.append("semantic expectations failed")
        if aggregate["pipeline_failures"]:
            failures.append("product pipeline expectations failed")
        if aggregate["privacy_precision"] < 1.0 or aggregate["privacy_recall"] < 1.0:
            failures.append(
                f"synthetic privacy precision/recall is {aggregate['privacy_precision']}/{aggregate['privacy_recall']}"
            )
        if external_privacy and external_privacy["precision"] < args.min_external_privacy_precision:
            failures.append(
                f"external privacy precision {external_privacy['precision']} is below "
                f"{args.min_external_privacy_precision}"
            )
        if external_privacy and external_privacy["recall"] < args.min_external_privacy_recall:
            failures.append(
                f"external privacy recall {external_privacy['recall']} is below "
                f"{args.min_external_privacy_recall}"
            )
        if not eligible:
            failures.append("no compression-eligible cases")
        elif aggregate["avg_eligible_prompt_ratio"] > args.max_eligible_ratio:
            failures.append(
                f"eligible prompt ratio {aggregate['avg_eligible_prompt_ratio']} exceeds {args.max_eligible_ratio}"
            )
        if failures:
            raise SystemExit("release gate failed: " + "; ".join(failures))


def evaluate_privacy(cases: list[dict[str, object]]) -> dict[str, float | int]:
    scrubber = PrivacyScrubber()
    expected: set[tuple[str, str, str]] = set()
    detected: set[tuple[str, str, str]] = set()
    annotated_cases = 0
    for case in cases:
        annotations = case.get("privacy_annotations")
        if annotations is None:
            continue
        annotated_cases += 1
        case_id = str(case["id"])
        for item in annotations:
            expected.add((case_id, str(item["kind"]), str(item["value"])))
        result = scrubber.scrub(str(case["text"]), language=str(case["source_lang"]))
        kinds = {item.placeholder: item.kind for item in result.protected_spans}
        for placeholder, value in result.vault.items():
            detected.add((case_id, kinds[placeholder], value))
    true_positives = len(expected & detected)
    precision = true_positives / max(len(detected), 1)
    recall = true_positives / max(len(expected), 1)
    return {
        "privacy_annotated_cases": annotated_cases,
        "privacy_expected_values": len(expected),
        "privacy_detected_values": len(detected),
        "privacy_true_positives": true_positives,
        "privacy_precision": round(precision, 4),
        "privacy_recall": round(recall, 4),
    }


def evaluate_external_privacy(path: Path) -> dict[str, object]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    kind_map = {"EMAIL_ADDRESS": "email", "PHONE_NUMBER": "phone", "CREDIT_CARD": "card"}
    scrubber = PrivacyScrubber()
    expected: set[tuple[int, str, str]] = set()
    detected: set[tuple[int, str, str]] = set()
    for index, row in enumerate(rows):
        for span in row["spans"]:
            kind = kind_map.get(span["entity_type"])
            if kind:
                expected.add((index, kind, span["entity_value"]))
        result = scrubber.scrub(row["full_text"], language="en")
        kinds = {item.placeholder: item.kind for item in result.protected_spans}
        for placeholder, value in result.vault.items():
            detected.add((index, kinds[placeholder], value))
    true_positives = expected & detected
    return {
        "dataset": "microsoft/presidio-research synth_dataset_v2",
        "license": "MIT",
        "rows": len(rows),
        "supported_kinds": sorted(kind_map.values()),
        "expected_values": len(expected),
        "detected_values": len(detected),
        "true_positives": len(true_positives),
        "false_positives": len(detected - expected),
        "false_negatives": len(expected - detected),
        "precision": round(len(true_positives) / max(len(detected), 1), 4),
        "recall": round(len(true_positives) / max(len(expected), 1), 4),
    }


def evaluate_pipeline(gateway: ContextIR) -> list[dict[str, object]]:
    repeated = " ".join(["Do not send payment 42 twice."] * 30)
    pipeline = ContextPipeline(gateway=gateway)
    prepared = pipeline.prepare(repeated, source_lang="en", target_lang="en", risk="low")

    reasoning = pipeline.run(
        "What is the safest next step?",
        lambda _prompt: "Ask the operator to confirm.",
        source_lang="en",
        target_lang="en",
    )
    responses = iter(["Payment completed.", "Do not send payment 42 twice."])
    transform = pipeline.run(
        repeated,
        lambda _prompt: next(responses),
        source_lang="en",
        target_lang="en",
        risk="low",
        task="transform",
    )
    privacy = ContextPipeline(gateway=gateway, policy=PipelinePolicy(max_attempts=1)).run(
        "Draft a response.",
        lambda _prompt: "Contact leaked@example.test.",
        source_lang="en",
        target_lang="en",
    )
    initial_budget_calls = 0

    def oversized_invoke(_prompt: str) -> str:
        nonlocal initial_budget_calls
        initial_budget_calls += 1
        return "unexpected"

    initial_budget_blocked = False
    try:
        ContextPipeline(
            gateway=gateway,
            policy=PipelinePolicy(max_prompt_tokens=3),
            invoke=oversized_invoke,
        ).run("Reply with only READY.", source_lang="en", target_lang="en")
    except ContextWindowExceeded:
        initial_budget_blocked = True

    fallback_budget_calls = 0

    def fallback_invoke(_prompt: str) -> str:
        nonlocal fallback_budget_calls
        fallback_budget_calls += 1
        return "Payment completed."

    fallback_budget = ContextPipeline(
        gateway=gateway,
        policy=PipelinePolicy(max_prompt_tokens=30),
    ).run(
        repeated,
        fallback_invoke,
        source_lang="en",
        target_lang="en",
        risk="high",
        task="transform",
    )
    return [
        {
            "case_id": "measured_semantic_selection",
            "passed": prepared.mode == "semantic" and prepared.token_savings >= pipeline.policy.min_token_savings,
            "mode": prepared.mode,
            "token_savings": prepared.token_savings,
        },
        {
            "case_id": "reasoning_without_false_roundtrip",
            "passed": reasoning.accepted and len(reasoning.attempts) == 1,
            "mode": reasoning.selected_mode,
            "fallbacks": len(reasoning.attempts) - 1,
        },
        {
            "case_id": "transform_bounded_fallback",
            "passed": transform.accepted and [item.mode for item in transform.attempts] == ["semantic", "hybrid"],
            "mode": transform.selected_mode,
            "fallbacks": len(transform.attempts) - 1,
        },
        {
            "case_id": "new_pii_rejected",
            "passed": not privacy.accepted and "new_pii" in privacy.attempts[0].verification.reasons,
            "mode": privacy.selected_mode,
            "fallbacks": len(privacy.attempts) - 1,
        },
        {
            "case_id": "initial_prompt_budget_blocked",
            "passed": initial_budget_blocked and initial_budget_calls == 0,
            "mode": "raw",
            "model_calls": initial_budget_calls,
        },
        {
            "case_id": "fallback_prompt_budget_blocked",
            "passed": (
                not fallback_budget.accepted
                and fallback_budget_calls == 1
                and "fallback_exceeds_prompt_budget" in fallback_budget.attempts[0].verification.reasons
            ),
            "mode": fallback_budget.selected_mode,
            "model_calls": fallback_budget_calls,
            "fallbacks": len(fallback_budget.attempts) - 1,
        },
    ]


if __name__ == "__main__":
    main()
