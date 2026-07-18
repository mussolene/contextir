from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from contextir.sir_sources import PROJECT_ROOT, WN30_DIR, load_records_jsonl


RELATION_NAMES = {
    "!": "antonym",
    "@": "hypernym",
    "@i": "instance_hypernym",
    "~": "hyponym",
    "~i": "instance_hyponym",
    "#m": "member_holonym",
    "#s": "substance_holonym",
    "#p": "part_holonym",
    "%m": "member_meronym",
    "%s": "substance_meronym",
    "%p": "part_meronym",
    "=": "attribute",
    "+": "derivationally_related",
    ";c": "domain_topic",
    "-c": "member_of_domain_topic",
    ";r": "domain_region",
    "-r": "member_of_domain_region",
    ";u": "domain_usage",
    "-u": "member_of_domain_usage",
    "*": "entailment",
    ">": "cause",
    "^": "also_see",
    "$": "verb_group",
    "&": "similar_to",
    "<": "participle_of_verb",
    "\\": "pertainym",
}


@dataclass
class ConceptRelation:
    source: str
    relation: str
    target: str
    source_pos: str
    target_pos: str
    source_file: str = "wordnet30"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SIR concept relation graph from WordNet pointers.")
    parser.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_graph.jsonl"))
    parser.add_argument("--min-relation-count", type=int, default=3)
    args = parser.parse_args()

    concept_ids = {record.concept_id for record in load_records_jsonl(Path(args.records))}
    relations = parse_wordnet_relations(WN30_DIR, concept_ids)
    counts = Counter(rel.relation for rel in relations)
    filtered = [rel for rel in relations if counts[rel.relation] >= args.min_relation_count]
    write_relations(filtered, Path(args.out))
    print(f"wrote {len(filtered)} relations to {args.out}")
    print(json.dumps(Counter(rel.relation for rel in filtered).most_common(20), ensure_ascii=False))


def parse_wordnet_relations(root: Path, allowed: set[str]) -> list[ConceptRelation]:
    dict_dir = next(root.glob("**/dict"), root / "dict") if root.exists() else root / "dict"
    files = {
        "data.noun": "n",
        "data.verb": "v",
        "data.adj": "a",
        "data.adv": "r",
    }
    relations: list[ConceptRelation] = []
    for filename, default_pos in files.items():
        path = dict_dir / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or not line[0].isdigit():
                continue
            relation_row = parse_relation_line(line, default_pos, allowed)
            relations.extend(relation_row)
    return dedupe_relations(relations)


def parse_relation_line(line: str, default_pos: str, allowed: set[str]) -> list[ConceptRelation]:
    head = line.split("|", 1)[0].strip()
    parts = head.split()
    if len(parts) < 4 or not parts[0].isdigit():
        return []
    source_pos = "a" if parts[2] == "s" else default_pos
    source = f"{parts[0]}-{source_pos}"
    if source not in allowed:
        return []
    try:
        word_count = int(parts[3], 16)
    except ValueError:
        return []
    index = 4 + word_count * 2
    if index >= len(parts):
        return []
    try:
        pointer_count = int(parts[index])
    except ValueError:
        return []
    index += 1
    rows: list[ConceptRelation] = []
    for _ in range(pointer_count):
        if index + 3 >= len(parts):
            break
        symbol, target_offset, target_pos_raw, _source_target = parts[index : index + 4]
        index += 4
        target_pos = "a" if target_pos_raw == "s" else target_pos_raw
        target = f"{target_offset}-{target_pos}"
        if target not in allowed:
            continue
        rows.append(
            ConceptRelation(
                source=source,
                relation=RELATION_NAMES.get(symbol, f"ptr:{symbol}"),
                target=target,
                source_pos=source_pos,
                target_pos=target_pos,
            )
        )
    return rows


def dedupe_relations(relations: list[ConceptRelation]) -> list[ConceptRelation]:
    seen: set[tuple[str, str, str]] = set()
    out: list[ConceptRelation] = []
    for rel in relations:
        key = (rel.source, rel.relation, rel.target)
        if key not in seen:
            seen.add(key)
            out.append(rel)
    return out


def write_relations(relations: list[ConceptRelation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for relation in relations:
            handle.write(json.dumps(asdict(relation), ensure_ascii=False, separators=(",", ":")) + "\n")


def load_relations_jsonl(path: Path) -> list[ConceptRelation]:
    return [ConceptRelation(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
