from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextir.sir_roundtrip import load_roundtrip, read_roundtrip_rows
from contextir.sir_sources import PROJECT_ROOT


def aggregate(results: list[dict]) -> dict[str, float | int]:
    if not results:
        return {}
    keys = [
        "concept_precision",
        "concept_recall",
        "concept_f1",
        "segment_coverage",
        "unknown_token_rate",
        "compression_ratio",
        "latency_ms",
    ]
    out: dict[str, float | int] = {"texts": len(results), "total_chars": sum(row["chars"] for row in results), "total_tokens": sum(row["tokens"] for row in results)}
    for key in keys:
        out[f"avg_{key}"] = round(sum(row[key] for row in results) / len(results), 4)
    baseline_f1 = [row["direct_baseline"]["concept_f1"] for row in results]
    out["baseline_avg_concept_f1"] = round(sum(baseline_f1) / len(baseline_f1), 4)
    out["sir_vs_baseline_f1_delta"] = round(float(out["avg_concept_f1"]) - float(out["baseline_avg_concept_f1"]), 4)
    return out


def write_examples(results: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SIR Roundtrip Examples", ""]
    for row in results:
        lines.extend(
            [
                f"## {row['text_id']}",
                "",
                f"- metrics: F1={row['concept_f1']}, recall={row['concept_recall']}, coverage={row['segment_coverage']}, unknown={row['unknown_token_rate']}",
                f"- baseline F1={row['direct_baseline']['concept_f1']}",
                "",
                "Bridge:",
                "",
                row["bridge_text"] or "_empty_",
                "",
                "Reconstructed:",
                "",
                row["reconstructed_text"] or "_empty_",
                "",
            ]
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate long-text SIR compile/decompile/compile roundtrip.")
    parser.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    parser.add_argument("--texts", default=str(PROJECT_ROOT / "data" / "roundtrip" / "roundtrip_texts.jsonl"))
    parser.add_argument("--target-lang", default="en", choices=["en", "ru"])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "roundtrip_eval.json"))
    parser.add_argument("--examples", default=str(PROJECT_ROOT / "reports" / "roundtrip_examples.md"))
    args = parser.parse_args()

    benchmark = load_roundtrip(Path(args.records))
    rows = read_roundtrip_rows(Path(args.texts))
    results = [asdict(benchmark.roundtrip(row, target_lang=args.target_lang)) for row in rows]
    report = {"aggregate": aggregate(results), "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_examples(results, Path(args.examples))
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    print(f"wrote {args.examples}")


if __name__ == "__main__":
    main()
