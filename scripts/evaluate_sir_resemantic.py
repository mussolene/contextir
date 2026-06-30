#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_core.sir_kernel import load_kernel
from semantic_core.sir_neural_kernel import NeuralSIRKernel
from semantic_core.sir_sources import PROJECT_ROOT


def concept_ids(contract: dict) -> set[str]:
    return {item["id"] for item in contract.get("concepts", [])}


def f1(left: set[str], right: set[str]) -> dict[str, float | int]:
    shared = left & right
    precision = len(shared) / max(len(right), 1)
    recall = len(shared) / max(len(left), 1)
    return {
        "left_concepts": len(left),
        "right_concepts": len(right),
        "shared_concepts": len(shared),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic -> resemantic experiments over longer text.")
    parser.add_argument("--texts", default=str(PROJECT_ROOT / "data" / "sir_training" / "resemantic_texts.jsonl"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "sir_neural_kernel_smoke.npz"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "sir_resemantic_eval.json"))
    parser.add_argument("--threshold", type=float, default=0.42)
    args = parser.parse_args()

    kernel = load_kernel()
    neural = NeuralSIRKernel.load(Path(args.checkpoint))
    rows = [json.loads(line) for line in Path(args.texts).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for row in rows:
        source = kernel.compile(row["text"], source_lang=row["source_lang"], target_lang=row["target_langs"][0], packet_id=row["id"])
        source_ids = concept_ids(source)
        paths = {}
        for lang in row["target_langs"]:
            text = kernel.decompile(source, target_lang=lang, include_anchors=True)
            back = kernel.compile(text, source_lang=lang, target_lang=row["source_lang"], packet_id=f"{row['id']}:{lang}")
            paths[lang] = {
                "decompiled_text": text,
                "metrics": f1(source_ids, concept_ids(back)),
                "pii_leaked": "person@example.test" in text or "person@example.test" in json.dumps(back, ensure_ascii=False),
            }
        pred = neural.predict(row["text"], threshold=args.threshold)
        neural_ids = {item["id"] for item in pred["concepts"]}
        results.append(
            {
                "id": row["id"],
                "source_concepts": len(source_ids),
                "protected_spans": len(source["protected_spans"]),
                "paths": paths,
                "neural_prediction": pred,
                "neural_vs_teacher": f1(source_ids, neural_ids),
            }
        )
    aggregate = {
        "texts": len(results),
        "avg_path_f1": round(
            sum(path["metrics"]["f1"] for row in results for path in row["paths"].values())
            / max(sum(len(row["paths"]) for row in results), 1),
            4,
        ),
        "pii_leaks": sum(path["pii_leaked"] for row in results for path in row["paths"].values()),
        "avg_neural_vs_teacher_f1": round(sum(row["neural_vs_teacher"]["f1"] for row in results) / max(len(results), 1), 4),
    }
    report = {"aggregate": aggregate, "results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
