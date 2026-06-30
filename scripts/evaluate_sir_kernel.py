#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_core.sir_kernel import load_kernel
from semantic_core.sir_runtime import load_runtime
from semantic_core.sir_sources import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the public SIR kernel compile/decompile contract.")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "data" / "roundtrip" / "sir_kernel_smoke.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "sir_kernel_smoke.json"))
    args = parser.parse_args()

    kernel = load_kernel()
    runtime = load_runtime()
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for case in cases:
        contract = kernel.compile(case["text"], source_lang=case["source_lang"], target_lang=case["target_lang"], packet_id=case["id"])
        text = kernel.decompile(contract, target_lang=case["target_lang"], include_anchors=True)
        round_contract = kernel.compile(text, source_lang=case["target_lang"], target_lang=case["target_lang"], packet_id=f"{case['id']}:round")
        source_ids = {item["id"] for item in contract["concepts"]}
        round_ids = {item["id"] for item in round_contract["concepts"]}
        shared = source_ids & round_ids
        pii_leaked = "person@example.test" in json.dumps(contract, ensure_ascii=False) or "person@example.test" in text
        results.append(
            {
                "case_id": case["id"],
                "source_concepts": len(source_ids),
                "round_concepts": len(round_ids),
                "shared_concepts": len(shared),
                "preserved_concepts": round(len(shared) / max(len(source_ids), 1), 4),
                "protected_spans": len(contract["protected_spans"]),
                "pii_leaked": pii_leaked,
                "decompiled_text": text,
                "contract_stats": contract["stats"],
            }
        )
    aggregate = {
        "cases": len(results),
        "avg_preserved_concepts": round(sum(row["preserved_concepts"] for row in results) / max(len(results), 1), 4),
        "pii_leaks": sum(1 for row in results if row["pii_leaked"]),
        "avg_contract_concepts": round(sum(row["source_concepts"] for row in results) / max(len(results), 1), 2),
    }
    report = {"aggregate": aggregate, "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
