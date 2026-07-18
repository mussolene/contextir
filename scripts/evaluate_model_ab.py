#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from contextir import ContextIR, ContextPipeline, OllamaClient, OpenAICompatibleClient
from contextir.pipeline import approximate_token_count
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


def invoke_model(args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    if args.backend == "ollama":
        result = OllamaClient(
            args.model,
            timeout=args.timeout,
            context_length=args.context_length,
            max_output_tokens=args.max_output_tokens,
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
            max_output_tokens=args.max_output_tokens,
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
    usage["model_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return answer, usage


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


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["requested_mode"], []).append(row)
    output = []
    for mode, items in groups.items():
        source_tokens = sum(item["estimated_source_tokens"] for item in items)
        prompt_tokens = sum(item["estimated_prompt_tokens"] for item in items)
        backend_prompt_tokens = [item["backend_prompt_tokens"] for item in items if item["backend_prompt_tokens"]]
        output.append(
            {
                "requested_mode": mode,
                "cases": len(items),
                "mean_quality": round(statistics.fmean(item["quality"] for item in items), 4),
                "exact_matches": sum(item["quality"] == 1.0 for item in items),
                "estimated_prompt_ratio": round(prompt_tokens / max(source_tokens, 1), 4),
                "backend_prompt_tokens": sum(backend_prompt_tokens) if backend_prompt_tokens else None,
                "mean_model_latency_ms": round(statistics.fmean(item["model_latency_ms"] for item in items), 1),
                "selected_modes": dict(Counter(item["selected_mode"] for item in items)),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    selected = {item for item in args.case_ids.split(",") if item}
    modes = [item for item in args.modes.split(",") if item]
    unsupported = set(modes) - {"raw", "auto", "hybrid", "semantic"}
    if unsupported:
        raise SystemExit(f"unsupported modes: {sorted(unsupported)}")
    cases = load_cases(Path(args.manifest), Path(args.longbench_dir), selected)
    gateway = ContextIR()
    pipeline = ContextPipeline(gateway=gateway)
    rows = []
    for case in cases:
        for mode in modes:
            prompt, metadata = prepare_prompt(gateway, pipeline, case, mode)
            try:
                prediction, usage = invoke_model(args, prompt)
                error = None
            except Exception as exc:
                prediction, usage, error = "", {}, f"{type(exc).__name__}: {exc}"
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
    report = {
        "metadata": {
            "backend": args.backend,
            "backend_scope": "remote_agent" if args.backend == "agent" else "local_model",
            "model": args.model,
            "context_length": args.context_length,
            "max_output_tokens": args.max_output_tokens,
            "cases": len(cases),
            "modes": modes,
        },
        "aggregate": aggregate(rows),
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
