from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from contextir.sir_graph import ConceptRelation, load_relations_jsonl
from contextir.sir_sources import PROJECT_ROOT, ConceptRecord, load_records_jsonl, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Export normalized SIR data in Hugging Face-friendly JSONL form.")
    parser.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    parser.add_argument("--graph", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_graph.jsonl"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "hf_sir_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    records = load_records_jsonl(Path(args.records))
    relations = load_relations_jsonl(Path(args.graph))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_rows = [concept_to_hf(record) for record in records]
    relation_rows = [relation_to_hf(rel) for rel in relations]
    train, valid, test = split_rows(relation_rows, seed=args.seed, valid_ratio=args.valid_ratio, test_ratio=args.test_ratio)

    write_jsonl(out_dir / "concepts.jsonl", concept_rows)
    write_jsonl(out_dir / "relations.jsonl", relation_rows)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "validation.jsonl", valid)
    write_jsonl(out_dir / "test.jsonl", test)
    write_dataset_card(out_dir / "README.md", records, relations, train, valid, test)

    print(json.dumps({
        "concepts": len(concept_rows),
        "relations": len(relation_rows),
        "train": len(train),
        "validation": len(valid),
        "test": len(test),
        "out_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))


def concept_to_hf(record: ConceptRecord) -> dict:
    return {
        "id": record.concept_id,
        "sir_type": "concept",
        "pos": record.pos,
        "lemmas": {
            "en": normalized_values(record.en),
            "ru": normalized_values(record.ru),
        },
        "definitions": {
            "en": record.definition_en,
            "ru": "",
            "ja": "",
        },
        "source": record.source,
        "text_en": " ; ".join([*record.en[:3], record.definition_en]).strip(),
        "text_ru": " ; ".join(record.ru[:3]).strip(),
    }


def relation_to_hf(rel: ConceptRelation) -> dict:
    return {
        "id": f"{rel.source}::{rel.relation}::{rel.target}",
        "sir_type": "relation_triple",
        "source": rel.source,
        "relation": rel.relation,
        "target": rel.target,
        "source_pos": rel.source_pos,
        "target_pos": rel.target_pos,
        "label": 1,
        "task": "predict_target_concept",
        "source_file": rel.source_file,
    }


def normalized_values(values: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        norm = normalize(value)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(value)
    return out


def split_rows(rows: list[dict], seed: int, valid_ratio: float, test_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    test_n = int(len(shuffled) * test_ratio)
    valid_n = int(len(shuffled) * valid_ratio)
    test = shuffled[:test_n]
    valid = shuffled[test_n : test_n + valid_n]
    train = shuffled[test_n + valid_n :]
    return train, valid, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_dataset_card(path: Path, records: list[ConceptRecord], relations: list[ConceptRelation], train: list[dict], valid: list[dict], test: list[dict]) -> None:
    relation_counts = Counter(rel.relation for rel in relations)
    lines = [
        "---",
        "language:",
        "- en",
        "- ru",
        "task_categories:",
        "- feature-extraction",
        "- sentence-similarity",
        "- text-classification",
        "pretty_name: ContextIR Research Concept Graph",
        "---",
        "",
        "# ContextIR Research Concept Graph",
        "",
        "Normalized concept and relation triples for the ContextIR research layer.",
        "",
        "## Files",
        "",
        "- `concepts.jsonl`: concept records with English/Russian lemmas and English definitions.",
        "- `relations.jsonl`: all positive concept relation triples.",
        "- `train.jsonl`, `validation.jsonl`, `test.jsonl`: relation prediction splits.",
        "",
        "## Counts",
        "",
        f"- Concepts: {len(records)}",
        f"- Relations: {len(relations)}",
        f"- Train triples: {len(train)}",
        f"- Validation triples: {len(valid)}",
        f"- Test triples: {len(test)}",
        "",
        "## Top Relations",
        "",
        "| relation | count |",
        "|---|---:|",
    ]
    for name, count in relation_counts.most_common(20):
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## Intended Uses",
        "",
        "- multilingual concept grounding;",
        "- relation prediction;",
        "- negative-sampling graph embedding training;",
        "- SIR precompiler/decompiler experiments.",
        "",
        "## Caveats",
        "",
        "Russian definitions are not yet populated. Russian lexical grounding currently comes from OMW/Wiktionary-derived lemmas.",
        "This dataset is a research artifact, not a production lexical database.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
