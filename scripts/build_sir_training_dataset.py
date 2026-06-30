#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_core.sir_kernel import load_kernel
from semantic_core.sir_sources import PROJECT_ROOT


def build_examples(seed_path: Path) -> list[dict[str, Any]]:
    kernel = load_kernel()
    rows = [json.loads(line) for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples: list[dict[str, Any]] = []
    for row in rows:
        contract = kernel.compile(row["text"], source_lang=row["source_lang"], target_lang=row["target_lang"], packet_id=row["id"])
        target_text = kernel.decompile(contract, target_lang=row["target_lang"], include_anchors=True)
        examples.append(
            {
                "id": row["id"],
                "source_lang": row["source_lang"],
                "target_lang": row["target_lang"],
                "input_text": row["text"],
                "sir_contract": contract,
                "target_text": target_text,
                "tasks": ["compile", "decompile", "privacy", "roundtrip"],
            }
        )
    return examples


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SIR neural training data from seed texts using SIRKernel as teacher.")
    parser.add_argument("--seed-texts", default=str(PROJECT_ROOT / "data" / "sir_training" / "seed_texts.jsonl"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "sir_training"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = build_examples(Path(args.seed_texts))
    random.Random(args.seed).shuffle(examples)
    n = len(examples)
    train_end = max(1, int(n * 0.7))
    valid_end = max(train_end + 1, int(n * 0.85)) if n > 2 else train_end
    splits = {
        "train": examples[:train_end],
        "valid": examples[train_end:valid_end],
        "test": examples[valid_end:],
    }
    out_dir = Path(args.out_dir)
    for name, rows in splits.items():
        write_jsonl(rows, out_dir / f"{name}.jsonl")
    summary = {
        "examples": n,
        "train": len(splits["train"]),
        "valid": len(splits["valid"]),
        "test": len(splits["test"]),
        "avg_concepts": round(sum(len(row["sir_contract"]["concepts"]) for row in examples) / max(n, 1), 3),
        "privacy_examples": sum(1 for row in examples if row["sir_contract"]["protected_spans"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
