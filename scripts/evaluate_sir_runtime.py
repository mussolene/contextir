#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_core.sir_runtime import load_runtime, result_to_dict
from semantic_core.sir_sources import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the SIR runtime prototype on smoke cases.")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "data" / "roundtrip" / "sir_runtime_smoke.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "sir_runtime_smoke.json"))
    args = parser.parse_args()

    runtime = load_runtime()
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for case in cases:
        result = runtime.run(case["text"], source_lang=case["source_lang"], target_lang=case["target_lang"])
        row = result_to_dict(result, include_prompt=True)
        row["case_id"] = case["id"]
        row["pii_leaked"] = any(secret in row["model_prompt"] or secret in row["raw_answer"] for secret in ["person@example.test", "+1 555 010-0100"])
        row["model_prompt_chars"] = len(row["model_prompt"])
        row.pop("model_prompt", None)
        results.append(row)

    aggregate = {
        "cases": len(results),
        "avg_preserved_concepts": round(sum(row["answer_check"]["preserved_concepts"] for row in results) / max(len(results), 1), 4),
        "needs_revision": sum(1 for row in results if row["answer_check"]["needs_revision"]),
        "pii_leaks": sum(1 for row in results if row["pii_leaked"]),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in results) / max(len(results), 1), 4),
    }
    report = {"aggregate": aggregate, "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
