#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextir import ContextIR
from contextir.sir_sources import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ContextIR compression, logic preservation, and privacy.")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "data" / "roundtrip" / "sir_kernel_smoke.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "sir_kernel_smoke.json"))
    parser.add_argument("--mode", choices=["auto", "raw", "hybrid", "semantic"], default="auto")
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
            }
        )
    total = max(len(results), 1)
    aggregate = {
        "cases": len(results),
        "avg_prompt_ratio": round(sum(row["prompt_ratio"] for row in results) / total, 4),
        "compressed_cases": sum(1 for row in results if row["prompt_ratio"] < 1.0),
        "critical_events": sum(row["critical_events"] for row in results),
        "pii_leaks": sum(1 for row in results if row["pii_leaked"]),
    }
    report = {"aggregate": aggregate, "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
