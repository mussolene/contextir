#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextir import ContextIR, ContextPipeline, OllamaClient, OpenAICompatibleClient, PipelinePolicy
from contextir.clients import post_json
from contextir.gateway import paragraph_owner_map
from contextir.pipeline import approximate_token_count, clean_model_output
from contextir.sir_sources import PROJECT_ROOT


LONG_BENCH_PROMPTS = {
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Carefully determine how many unique paragraphs remain after removing duplicates.\n\n{context}\n\n"
        "Output only the final count as a number.\n\nThe final answer is:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Determine which paragraph "
        "the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\n"
        "Output only the paragraph identifier, such as Paragraph 1.\n\nThe answer is:"
    ),
}


@dataclass(frozen=True)
class Case:
    id: str
    dataset: str
    prompt: str
    answers: list[str]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B test ContextIR modes against a model endpoint.")
    parser.add_argument("--backend", choices=["ollama", "lmstudio", "agent"], default="ollama")
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "data" / "benchmarks" / "context_ab_cases.jsonl"))
    parser.add_argument("--longbench-dir", default="/tmp/contextir-longbench/data")
    parser.add_argument("--modes", default="raw,auto,hybrid,semantic")
    parser.add_argument("--case-ids", default="", help="Comma-separated subset of manifest case IDs.")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--prompt-overhead-tokens", type=int, default=32)
    parser.add_argument("--max-chunk-calls", type=int, default=16)
    parser.add_argument("--summary-context-length", type=int, default=32768)
    parser.add_argument("--summary-output-tokens", type=int, default=512)
    parser.add_argument("--embedding-model", default="nomic-embed-text-v2-moe:latest")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cases(manifest: Path, longbench_dir: Path, selected: set[str]) -> list[Case]:
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    cases = []
    for spec in read_jsonl(manifest):
        if selected and spec["id"] not in selected:
            continue
        if spec["source"] == "synthetic":
            cases.append(make_synthetic_case(spec))
            continue
        dataset = spec["dataset"]
        if dataset not in rows_by_dataset:
            path = longbench_dir / f"{dataset}.jsonl"
            if not path.exists():
                raise SystemExit(f"missing LongBench data: {path}")
            rows_by_dataset[dataset] = read_jsonl(path)
        row = rows_by_dataset[dataset][int(spec["index"])]
        prompt = LONG_BENCH_PROMPTS[dataset].format(context=row["context"], input=row["input"])
        cases.append(Case(spec["id"], dataset, prompt, list(row["answers"]), "LongBench"))
    return cases


def make_synthetic_case(spec: dict[str, Any]) -> Case:
    if spec["variant"] == "middle":
        distractors = [
            f"Record {index}: project Cedar uses status code {1000 + index} for an unrelated archived item."
            for index in range(180)
        ]
        distractors.insert(90, "The access phrase for project Juniper is cobalt-seven.")
        prompt = (
            "Read the records. Answer only with the access phrase for project Juniper.\n\n"
            + "\n".join(distractors)
            + "\n\nAnswer:"
        )
        return Case(spec["id"], spec["dataset"], prompt, ["cobalt-seven"], "RULER-style diagnostic")
    if spec["variant"] == "tool_route":
        tools = [
            f"Tool archive_record_{index}: archive historical record type {1000 + index}."
            for index in range(90)
        ]
        tools.insert(45, "Tool create_invoice: create a customer invoice for completed work.")
        prompt = (
            "Read the following tool catalog and answer the question.\n\n"
            + "\n".join(tools)
            + "\n\nQuestion: Which tool should the agent call to create a customer invoice for completed work? "
            "Answer with only the tool name.\nAnswer:"
        )
        return Case(spec["id"], spec["dataset"], prompt, ["create_invoice"], "Agent tool-routing diagnostic")
    if spec["variant"] == "agent_state":
        history = [
            f"Step {index}: archived Cedar request {2000 + index} with status completed."
            for index in range(90)
        ]
        history.insert(47, "Step 47B: Juniper invoice failed validation and requires finance review via notify_finance.")
        prompt = (
            "Read the following agent history and answer the question.\n\n"
            + "\n".join(history)
            + "\n\nQuestion: Which exact next action is required for the Juniper invoice that failed validation? "
            "Answer with only the action name.\nAnswer:"
        )
        return Case(spec["id"], spec["dataset"], prompt, ["notify_finance"], "Agent-state diagnostic")
    if spec["variant"] == "oversized_segment":
        records = " ".join(
            f"Record {index}: Cedar historical status is closed."
            for index in range(4)
        )
        evidence = " ".join(
            ["archived"] * 200
            + ["Project Juniper current access phrase is cobalt-seven"]
            + ["archived"] * 2800
        )
        prompt = (
            "Read the following text and answer briefly. "
            + records
            + " "
            + evidence
            + ". Question: What is the current access phrase for Project Juniper? Answer:"
        )
        return Case(
            spec["id"],
            spec["dataset"],
            prompt,
            ["cobalt-seven"],
            "ContextIR oversized-segment diagnostic",
        )
    prompt = " ".join(
        ["Do not send payment 42 twice to contact finance@example.test."] * 80
        + ["Question: What action is prohibited? Answer with only the prohibited action."]
    )
    return Case(spec["id"], spec["dataset"], prompt, ["send payment 42 twice"], "ContextIR diagnostic")


def prepare_prompt(gateway: ContextIR, pipeline: ContextPipeline, case: Case, requested_mode: str) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    if requested_mode == "auto":
        prepared = pipeline.prepare(case.prompt, source_lang="en", target_lang="en", risk="standard", packet_id=case.id)
        bundle = prepared.bundle
        metadata = {
            "selected_mode": prepared.mode,
            "decision": prepared.decision,
            "estimated_source_tokens": prepared.source_tokens,
            "estimated_prompt_tokens": prepared.prompt_tokens,
            "estimated_token_savings": prepared.token_savings,
        }
        rendered = prepared.prompt
    else:
        bundle = gateway.compile_private(
            case.prompt, source_lang="en", target_lang="en", packet_id=case.id, mode=requested_mode
        )
        rendered = gateway.render_prompt(bundle.contract)
        source_text = " ".join(bundle.sources.values())
        source_tokens = max(approximate_token_count(source_text), 1)
        prompt_tokens = max(approximate_token_count(rendered), 1)
        metadata = {
            "selected_mode": bundle.contract["mode"],
            "decision": "forced_mode",
            "estimated_source_tokens": source_tokens,
            "estimated_prompt_tokens": prompt_tokens,
            "estimated_token_savings": round(1 - prompt_tokens / source_tokens, 4),
        }
    metadata.update(
        {
            "compile_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "source_chars": len(case.prompt),
            "prompt_chars": len(rendered),
            "protected_spans": len(bundle.contract["privacy"]["protected"]),
            "semantic_confidence": bundle.contract["uncertainty"]["semantic_confidence"],
        }
    )
    return rendered, metadata


def invoke_model(
    args: argparse.Namespace,
    prompt: str,
    *,
    context_length: int | None = None,
    max_output_tokens: int | None = None,
) -> tuple[str, dict[str, Any]]:
    context_length = context_length or args.context_length
    max_output_tokens = max_output_tokens or args.max_output_tokens
    started = time.perf_counter()
    if args.backend == "ollama":
        result = OllamaClient(
            args.model,
            base_url=args.ollama_base_url,
            timeout=args.timeout,
            context_length=context_length,
            max_output_tokens=max_output_tokens,
            prompt_overhead_tokens=args.prompt_overhead_tokens,
        ).complete(prompt)
        answer = result.text
        usage = {
            "backend_prompt_tokens": result.prompt_tokens,
            "backend_output_tokens": result.output_tokens,
            "backend_prompt_ms": result.prompt_ms,
            "backend_generation_ms": result.generation_ms,
        }
    elif args.backend == "lmstudio":
        result = OpenAICompatibleClient(
            args.model,
            timeout=args.timeout,
            context_length=context_length,
            max_output_tokens=max_output_tokens,
            prompt_overhead_tokens=args.prompt_overhead_tokens,
        ).complete(prompt)
        answer = result.text
        usage = {
            "backend_prompt_tokens": result.prompt_tokens,
            "backend_output_tokens": result.output_tokens,
        }
    else:
        command = ["agent", "--print", "--mode", "ask", "--trust", "--model", args.model, prompt]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
        if completed.returncode:
            raise RuntimeError(f"agent exited {completed.returncode}: {completed.stderr.strip()}")
        answer = completed.stdout.strip()
        usage = {"backend_prompt_tokens": None, "backend_output_tokens": None}
    answer = clean_model_output(answer)
    usage["model_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return answer, usage


def run_chunked_case(
    args: argparse.Namespace,
    gateway: ContextIR,
    case: Case,
) -> tuple[str, dict[str, Any], dict[str, Any], str | None]:
    usages: list[dict[str, Any]] = []

    def invoke(prompt: str) -> str:
        answer, usage = invoke_model(args, prompt)
        usages.append(usage)
        return answer

    prompt_budget = args.context_length - args.max_output_tokens - args.prompt_overhead_tokens
    pipeline = ContextPipeline(
        gateway=gateway,
        invoke=invoke,
        policy=PipelinePolicy(
            max_prompt_tokens=prompt_budget,
            max_chunk_calls=args.max_chunk_calls,
        ),
    )
    started = time.perf_counter()
    try:
        result = pipeline.run(
            case.prompt,
            source_lang="en",
            target_lang="en",
            risk="standard",
            packet_id=case.id,
            chunked_retrieval=True,
        )
        error = None
        if not result.accepted:
            reasons = sorted(
                {
                    reason
                    for attempt in result.attempts
                    for reason in attempt.verification.reasons
                }
            )
            error = "pipeline_rejected" + (f":{','.join(reasons)}" if reasons else "")
        metadata = {
            "selected_mode": result.selected_mode,
            "decision": result.prepared.decision,
            "estimated_source_tokens": result.prepared.source_tokens,
            "estimated_prompt_tokens": sum(attempt.prompt_tokens for attempt in result.attempts),
            "estimated_token_savings": round(
                1
                - sum(attempt.prompt_tokens for attempt in result.attempts)
                / max(result.prepared.source_tokens, 1),
                4,
            ),
            "compile_latency_ms": round(
                max(
                    (time.perf_counter() - started) * 1000
                    - sum(float(item.get("model_latency_ms") or 0) for item in usages),
                    0,
                ),
                3,
            ),
            "source_chars": len(case.prompt),
            "prompt_chars": None,
            "protected_spans": len(result.prepared.bundle.contract["privacy"]["protected"]),
            "semantic_confidence": result.prepared.bundle.contract["uncertainty"]["semantic_confidence"],
            "pipeline_accepted": result.accepted,
            "pipeline_trace": result.public_trace(),
        }
        return result.answer, metadata, aggregate_usage(usages), error
    except Exception as exc:
        bundle = gateway.compile_private(
            case.prompt,
            source_lang="en",
            target_lang="en",
            packet_id=case.id,
            mode="auto",
        )
        source_tokens = max(approximate_token_count(" ".join(bundle.sources.values())), 1)
        metadata = {
            "selected_mode": bundle.contract["mode"],
            "decision": "pipeline_exception",
            "estimated_source_tokens": source_tokens,
            "estimated_prompt_tokens": 0,
            "estimated_token_savings": 0.0,
            "compile_latency_ms": round(
                max(
                    (time.perf_counter() - started) * 1000
                    - sum(float(item.get("model_latency_ms") or 0) for item in usages),
                    0,
                ),
                3,
            ),
            "source_chars": len(case.prompt),
            "prompt_chars": None,
            "protected_spans": len(bundle.contract["privacy"]["protected"]),
            "semantic_confidence": bundle.contract["uncertainty"]["semantic_confidence"],
            "pipeline_accepted": False,
            "pipeline_trace": [],
        }
        return "", metadata, aggregate_usage(usages), f"{type(exc).__name__}: {exc}"


def aggregate_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    def total(key: str) -> int | float | None:
        values = [item[key] for item in usages if item.get(key) is not None]
        return sum(values) if values else None

    return {
        "backend_prompt_tokens": total("backend_prompt_tokens"),
        "backend_output_tokens": total("backend_output_tokens"),
        "backend_prompt_ms": total("backend_prompt_ms"),
        "backend_generation_ms": total("backend_generation_ms"),
        "model_latency_ms": round(float(total("model_latency_ms") or 0), 3),
        "model_calls": len(usages),
    }


def baseline_metadata(
    bundle: Any,
    case: Case,
    selected_mode: str,
    decision: str,
    estimated_prompt_tokens: int,
    prompt_chars: int,
    compile_latency_ms: float,
) -> dict[str, Any]:
    source_text = " ".join(bundle.sources.values())
    source_tokens = max(approximate_token_count(source_text), 1)
    return {
        "selected_mode": selected_mode,
        "decision": decision,
        "estimated_source_tokens": source_tokens,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_token_savings": round(1 - estimated_prompt_tokens / source_tokens, 4),
        "compile_latency_ms": round(compile_latency_ms, 3),
        "source_chars": len(case.prompt),
        "prompt_chars": prompt_chars,
        "protected_spans": len(bundle.contract["privacy"]["protected"]),
        "semantic_confidence": bundle.contract["uncertainty"]["semantic_confidence"],
    }


def run_summary_case(
    args: argparse.Namespace,
    gateway: ContextIR,
    case: Case,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    raw_bundle = gateway.compile_private(
        case.prompt, source_lang="en", target_lang="en", packet_id=case.id, mode="raw"
    )
    task_bundle = gateway.compile_private(
        case.prompt, source_lang="en", target_lang="en", packet_id=case.id, mode="auto"
    )
    if not task_bundle.retrieval_query:
        raise ValueError("summary baseline requires an extracted retrieval query")
    masked_source = gateway.render_prompt(raw_bundle.contract)
    summary_prompt = (
        "Compress the MASKED SOURCE into factual notes sufficient to answer the TASK. "
        "Preserve exact names, numbers, negation, and paragraph identifiers. Do not answer the task.\n\n"
        f"TASK:\n{task_bundle.retrieval_query}\n\nMASKED SOURCE:\n{masked_source}\n\nNOTES:"
    )
    summary, summary_usage = invoke_model(
        args,
        summary_prompt,
        context_length=args.summary_context_length,
        max_output_tokens=args.summary_output_tokens,
    )
    if not summary:
        raise RuntimeError("summary model returned an empty result")
    summary = normalize_summary(summary)
    answer_prompt = render_extraction_prompt(task_bundle.retrieval_query, "EVIDENCE", [summary])
    prompt_budget = args.context_length - args.max_output_tokens - args.prompt_overhead_tokens
    if approximate_token_count(answer_prompt) > prompt_budget:
        raise ValueError("summary answer prompt exceeds the target prompt budget")
    answer, answer_usage = invoke_model(args, answer_prompt)
    usages = [summary_usage, answer_usage]
    usage = aggregate_usage(usages)
    usage.update(
        {
            "preprocessor": "neural_summary",
            "preprocessor_model": args.model,
            "preprocessor_prompt_tokens": summary_usage.get("backend_prompt_tokens"),
            "preprocessor_latency_ms": summary_usage.get("model_latency_ms"),
            "answer_prompt_tokens": answer_usage.get("backend_prompt_tokens"),
        }
    )
    estimated_tokens = approximate_token_count(summary_prompt) + approximate_token_count(answer_prompt)
    metadata = baseline_metadata(
        raw_bundle,
        case,
        "summary",
        "neural_summary_then_answer",
        estimated_tokens,
        len(summary_prompt) + len(answer_prompt),
        (time.perf_counter() - started) * 1000 - float(usage["model_latency_ms"] or 0),
    )
    return answer, metadata, usage


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def normalize_summary(text: str) -> str:
    return re.sub(r"^\s*(?:answer|notes)\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def render_embedding_answer_prompt(query: str, evidence: list[str]) -> str:
    return render_extraction_prompt(query, "EVIDENCE", evidence)


def render_extraction_prompt(query: str, evidence_label: str, evidence: list[str]) -> str:
    return (
        f"Extract the answer stated in the {evidence_label}.\n"
        f"QUESTION:\n{query}\n\n{evidence_label}:\n" + "\n\n".join(evidence) + "\n\nANSWER:"
    )


def pack_ranked_evidence(query: str, ranked_evidence: list[str], prompt_budget: int) -> tuple[str, int]:
    selected: list[str] = []
    prompt = render_embedding_answer_prompt(query, selected)
    for evidence in ranked_evidence:
        trial = render_embedding_answer_prompt(query, selected + [evidence])
        if approximate_token_count(trial) <= prompt_budget:
            selected.append(evidence)
            prompt = trial
    if not selected:
        raise ValueError("embedding retrieval found no evidence group that fits the target prompt budget")
    return prompt, len(selected)


def run_embedding_case(
    args: argparse.Namespace,
    gateway: ContextIR,
    case: Case,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if args.backend != "ollama":
        raise ValueError("embedding baseline currently requires the Ollama backend")
    started = time.perf_counter()
    raw_bundle = gateway.compile_private(
        case.prompt, source_lang="en", target_lang="en", packet_id=case.id, mode="raw"
    )
    task_bundle = gateway.compile_private(
        case.prompt, source_lang="en", target_lang="en", packet_id=case.id, mode="auto"
    )
    query = task_bundle.retrieval_query
    if not query:
        raise ValueError("embedding baseline requires an extracted retrieval query")
    candidates = [
        (ref, text)
        for ref, text in raw_bundle.sources.items()
        if ref not in set(task_bundle.task_source_refs)
    ]
    if not candidates:
        raise ValueError("embedding baseline found no candidate source segments")
    embedding_inputs = [f"search_query: {query}"] + [
        f"search_document: {text}" for _ref, text in candidates
    ]
    embedding_started = time.perf_counter()
    response = post_json(
        args.ollama_base_url.rstrip("/") + "/api/embed",
        {"model": args.embedding_model, "input": embedding_inputs, "truncate": True},
        timeout=args.timeout,
    )
    embedding_latency_ms = round((time.perf_counter() - embedding_started) * 1000, 3)
    embeddings = response.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(embedding_inputs):
        raise RuntimeError("invalid Ollama embedding response")
    query_embedding = embeddings[0]
    ranked_refs = sorted(
        (
            (cosine_similarity(query_embedding, embedding), ref)
            for (ref, _text), embedding in zip(candidates, embeddings[1:])
        ),
        key=lambda item: (-item[0], int(item[1][1:])),
    )
    owner_by_ref = paragraph_owner_map(list(raw_bundle.sources.items()))
    ranked_evidence = []
    seen_groups: set[tuple[str, ...]] = set()
    for _score, ref in ranked_refs:
        owner = owner_by_ref.get(ref, "")
        refs = tuple(dict.fromkeys(item for item in (owner, ref) if item))
        if refs in seen_groups:
            continue
        seen_groups.add(refs)
        ranked_evidence.append("\n".join(raw_bundle.sources[item] for item in refs))
    prompt_budget = args.context_length - args.max_output_tokens - args.prompt_overhead_tokens
    answer_prompt, selected_groups = pack_ranked_evidence(query, ranked_evidence, prompt_budget)
    answer, answer_usage = invoke_model(args, answer_prompt)
    embedding_tokens = int(response.get("prompt_eval_count") or 0)
    answer_tokens = int(answer_usage.get("backend_prompt_tokens") or 0)
    usage = {
        **answer_usage,
        "backend_prompt_tokens": embedding_tokens + answer_tokens,
        "model_latency_ms": round(embedding_latency_ms + float(answer_usage["model_latency_ms"]), 3),
        "model_calls": 2,
        "preprocessor": "embedding_retrieval",
        "preprocessor_model": args.embedding_model,
        "preprocessor_prompt_tokens": embedding_tokens,
        "preprocessor_latency_ms": embedding_latency_ms,
        "answer_prompt_tokens": answer_tokens,
    }
    estimated_tokens = sum(approximate_token_count(item) for item in embedding_inputs)
    estimated_tokens += approximate_token_count(answer_prompt)
    metadata = baseline_metadata(
        raw_bundle,
        case,
        "embedding",
        "embedding_ranked_budget_packed",
        estimated_tokens,
        len(answer_prompt),
        (time.perf_counter() - started) * 1000 - float(usage["model_latency_ms"]),
    )
    metadata["retrieved_groups"] = selected_groups
    return answer, metadata, usage


def normalize_answer(value: str) -> list[str]:
    value = value.lower().replace("_", " ")
    value = re.sub(r"[^\w\s-]", " ", value, flags=re.UNICODE)
    return [token for token in value.split() if token not in {"a", "an", "the"}]


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction)
    expected = normalize_answer(gold)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score(case: Case, prediction: str) -> float:
    if case.dataset == "contextir_oversized_retrieval":
        normalized_prediction = " ".join(normalize_answer(prediction))
        return max(
            float(" ".join(normalize_answer(answer)) in normalized_prediction)
            for answer in case.answers
        )
    if case.dataset in {"multifieldqa_en", "contextir_operational"}:
        return max(token_f1(prediction, answer) for answer in case.answers)
    if case.dataset == "passage_retrieval_en":
        predicted = re.findall(r"paragraph\s+(\d+)", prediction, re.IGNORECASE)
        expected = re.findall(r"paragraph\s+(\d+)", case.answers[0], re.IGNORECASE)
        if not predicted or not expected:
            return 0.0
        return sum(item == expected[0] for item in predicted) / len(predicted)
    if case.dataset == "passage_count":
        predicted = re.findall(r"\d+", prediction)
        if not predicted:
            return 0.0
        return sum(item == case.answers[0] for item in predicted) / len(predicted)
    if case.dataset in {"agent_tool_routing", "agent_state"}:
        tokens = re.findall(r"[a-z][a-z0-9_]*", prediction.lower())
        return float(tokens == [case.answers[0]])
    normalized_prediction = " ".join(normalize_answer(prediction))
    return max(float(" ".join(normalize_answer(answer)) == normalized_prediction) for answer in case.answers)


def bootstrap_mean_ci(values: list[float], samples: int = 2000, seed: int = 42) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1 or samples < 2:
        value = round(statistics.fmean(values), 4)
        return [value, value]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _item in values)
        for _sample in range(samples)
    )
    low = means[int(0.025 * (samples - 1))]
    high = means[int(0.975 * (samples - 1))]
    return [round(low, 4), round(high, 4)]


def aggregate(rows: list[dict[str, Any]], bootstrap_samples: int = 2000) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["requested_mode"], []).append(row)
    output = []
    for mode, items in groups.items():
        source_tokens = sum(item["estimated_source_tokens"] for item in items)
        prompt_tokens = sum(item["estimated_prompt_tokens"] for item in items)
        backend_prompt_tokens = [item["backend_prompt_tokens"] for item in items if item["backend_prompt_tokens"]]
        qualities = [item["quality"] for item in items]
        output.append(
            {
                "requested_mode": mode,
                "cases": len(items),
                "mean_quality": round(statistics.fmean(qualities), 4),
                "quality_ci_95": bootstrap_mean_ci(qualities, bootstrap_samples),
                "exact_matches": sum(item["quality"] == 1.0 for item in items),
                "failures": sum(bool(item["error"]) for item in items),
                "estimated_prompt_ratio": round(prompt_tokens / max(source_tokens, 1), 4),
                "backend_prompt_tokens": sum(backend_prompt_tokens) if backend_prompt_tokens else None,
                "mean_model_latency_ms": round(statistics.fmean(item["model_latency_ms"] for item in items), 1),
                "model_calls": sum(item.get("model_calls", 1) for item in items),
                "selected_modes": dict(Counter(item["selected_mode"] for item in items)),
                "decisions": dict(Counter(item["decision"] for item in items)),
                "pipeline_stages": dict(
                    Counter(
                        attempt["stage"]
                        for item in items
                        for attempt in item.get("pipeline_trace", {}).get("attempts", [])
                    )
                ),
                "pipeline_accepted": sum(item.get("pipeline_accepted") is True for item in items),
            }
        )
    return output


def compare_modes(
    rows: list[dict[str, Any]],
    baseline: str,
    candidate: str,
    bootstrap_samples: int = 2000,
) -> dict[str, Any] | None:
    baseline_rows = {item["case_id"]: item for item in rows if item["requested_mode"] == baseline}
    candidate_rows = {item["case_id"]: item for item in rows if item["requested_mode"] == candidate}
    case_ids = sorted(baseline_rows.keys() & candidate_rows.keys())
    if not case_ids:
        return None
    deltas = [candidate_rows[case_id]["quality"] - baseline_rows[case_id]["quality"] for case_id in case_ids]
    baseline_tokens = sum(float(baseline_rows[case_id].get("backend_prompt_tokens") or 0) for case_id in case_ids)
    candidate_tokens = sum(float(candidate_rows[case_id].get("backend_prompt_tokens") or 0) for case_id in case_ids)
    baseline_latency = sum(float(baseline_rows[case_id].get("model_latency_ms") or 0) for case_id in case_ids)
    candidate_latency = sum(float(candidate_rows[case_id].get("model_latency_ms") or 0) for case_id in case_ids)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "paired_cases": len(case_ids),
        "mean_quality_delta": round(statistics.fmean(deltas), 4),
        "quality_delta_ci_95": bootstrap_mean_ci(deltas, bootstrap_samples),
        "improved": sum(delta > 0 for delta in deltas),
        "tied": sum(delta == 0 for delta in deltas),
        "regressed": sum(delta < 0 for delta in deltas),
        "backend_prompt_token_ratio": round(candidate_tokens / baseline_tokens, 4) if baseline_tokens else None,
        "model_latency_ratio": round(candidate_latency / baseline_latency, 4) if baseline_latency else None,
    }


def build_comparisons(
    rows: list[dict[str, Any]], modes: list[str], bootstrap_samples: int = 2000
) -> list[dict[str, Any]]:
    pairs = [("raw", candidate) for candidate in modes if candidate != "raw"]
    if "chunked" in modes:
        pairs.extend(
            ("chunked", candidate)
            for candidate in modes
            if candidate in {"summary", "embedding"}
        )
    comparisons = []
    for baseline, candidate in pairs:
        comparison = compare_modes(rows, baseline, candidate, bootstrap_samples)
        if comparison:
            comparisons.append(comparison)
    return comparisons


def main() -> None:
    args = parse_args()
    selected = {item for item in args.case_ids.split(",") if item}
    modes = [item for item in args.modes.split(",") if item]
    unsupported = set(modes) - {"raw", "auto", "hybrid", "semantic", "chunked", "summary", "embedding"}
    if unsupported:
        raise SystemExit(f"unsupported modes: {sorted(unsupported)}")
    cases = load_cases(Path(args.manifest), Path(args.longbench_dir), selected)
    gateway = ContextIR()
    prompt_budget = args.context_length - args.max_output_tokens - args.prompt_overhead_tokens
    if prompt_budget < 1:
        raise SystemExit("context length must leave a positive prompt budget")
    pipeline = ContextPipeline(
        gateway=gateway,
        policy=PipelinePolicy(max_prompt_tokens=prompt_budget, max_chunk_calls=args.max_chunk_calls),
    )
    rows = []
    for case in cases:
        for mode in modes:
            if mode in {"summary", "embedding"}:
                try:
                    if mode == "summary":
                        prediction, metadata, usage = run_summary_case(args, gateway, case)
                    else:
                        prediction, metadata, usage = run_embedding_case(args, gateway, case)
                    error = None
                except Exception as exc:
                    prediction = ""
                    error = f"{type(exc).__name__}: {exc}"
                    bundle = gateway.compile_private(
                        case.prompt,
                        source_lang="en",
                        target_lang="en",
                        packet_id=case.id,
                        mode="raw",
                    )
                    metadata = baseline_metadata(
                        bundle,
                        case,
                        mode,
                        f"{mode}_baseline_exception",
                        0,
                        0,
                        0,
                    )
                    usage = {
                        "backend_prompt_tokens": None,
                        "backend_output_tokens": None,
                        "backend_prompt_ms": None,
                        "backend_generation_ms": None,
                        "model_latency_ms": 0.0,
                        "model_calls": 0,
                    }
                row = {
                    "case_id": case.id,
                    "dataset": case.dataset,
                    "source": case.source,
                    "requested_mode": mode,
                    "gold": case.answers,
                    "prediction": prediction,
                    "quality": round(score(case, prediction), 4) if not error else 0.0,
                    "error": error,
                    **metadata,
                    **usage,
                }
                rows.append(row)
                print(
                    f"{case.id:28} {mode:8} -> {metadata['selected_mode']:8} "
                    f"score={row['quality']:.3f} calls={usage['model_calls']} "
                    f"ratio={metadata['estimated_prompt_tokens'] / max(metadata['estimated_source_tokens'], 1):.3f}"
                )
                continue
            if mode == "chunked":
                prediction, metadata, usage, error = run_chunked_case(args, gateway, case)
                row = {
                    "case_id": case.id,
                    "dataset": case.dataset,
                    "source": case.source,
                    "requested_mode": mode,
                    "gold": case.answers,
                    "prediction": prediction,
                    "quality": round(score(case, prediction), 4) if not error else 0.0,
                    "error": error,
                    **metadata,
                    **usage,
                }
                rows.append(row)
                print(
                    f"{case.id:28} {mode:8} -> {metadata['selected_mode']:8} "
                    f"score={row['quality']:.3f} calls={usage['model_calls']} "
                    f"ratio={metadata['estimated_prompt_tokens'] / max(metadata['estimated_source_tokens'], 1):.3f}"
                )
                continue
            prompt, metadata = prepare_prompt(gateway, pipeline, case, mode)
            try:
                prediction, usage = invoke_model(args, prompt)
                usage["model_calls"] = 1
                error = None
            except Exception as exc:
                prediction = ""
                usage = {
                    "backend_prompt_tokens": None,
                    "backend_output_tokens": None,
                    "backend_prompt_ms": None,
                    "backend_generation_ms": None,
                    "model_latency_ms": 0.0,
                    "model_calls": 0,
                }
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "case_id": case.id,
                "dataset": case.dataset,
                "source": case.source,
                "requested_mode": mode,
                "gold": case.answers,
                "prediction": prediction,
                "quality": round(score(case, prediction), 4) if not error else 0.0,
                "error": error,
                **metadata,
                **usage,
            }
            rows.append(row)
            print(
                f"{case.id:28} {mode:8} -> {metadata['selected_mode']:8} "
                f"score={row['quality']:.3f} ratio={metadata['estimated_prompt_tokens'] / max(metadata['estimated_source_tokens'], 1):.3f}"
            )
    comparisons = build_comparisons(rows, modes, args.bootstrap_samples)
    report = {
        "metadata": {
            "backend": args.backend,
            "backend_scope": "remote_agent" if args.backend == "agent" else "local_model",
            "model": args.model,
            "context_length": args.context_length,
            "max_output_tokens": args.max_output_tokens,
            "prompt_overhead_tokens": args.prompt_overhead_tokens,
            "max_chunk_calls": args.max_chunk_calls,
            "summary_context_length": args.summary_context_length,
            "summary_output_tokens": args.summary_output_tokens,
            "embedding_model": args.embedding_model,
            "bootstrap_samples": args.bootstrap_samples,
            "cases": len(cases),
            "modes": modes,
        },
        "aggregate": aggregate(rows, args.bootstrap_samples),
        "comparisons": comparisons,
        "results": rows,
    }
    out = Path(args.out) if args.out else PROJECT_ROOT / "reports" / "model_ab" / f"{args.backend}_{safe_name(args.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", value.lower()).strip("_")


if __name__ == "__main__":
    main()
