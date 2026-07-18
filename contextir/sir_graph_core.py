from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from contextir.sir_graph import ConceptRelation, load_relations_jsonl
from contextir.sir_sources import PROJECT_ROOT, ConceptRecord, LexicalSIRCore, load_records_jsonl


CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sir_graph_core.npz"


class SIRGraphCore:
    def __init__(self, records: list[ConceptRecord], relation_vectors: dict[str, np.ndarray], relation_matrices: dict[str, np.ndarray] | None = None):
        self.records = records
        self.lexical = LexicalSIRCore(records)
        self.concepts = [record.concept_id for record in records]
        self.concept_index = {concept: i for i, concept in enumerate(self.concepts)}
        self.prototype_matrix = np.array([self.lexical.vectors[concept] for concept in self.concepts], dtype=np.float32)
        self.relation_vectors = relation_vectors
        self.relation_matrices = relation_matrices or {}

    def predict(self, source: str, relation: str, k: int = 10) -> list[tuple[str, float]]:
        source_vec = self.prototype_matrix[self.concept_index[source]]
        matrix = self.relation_matrices.get(relation)
        rel_vec = self.relation_vectors.get(relation)
        if matrix is None and rel_vec is None:
            return []
        query = normalize_vector(source_vec @ matrix) if matrix is not None else normalize_vector(source_vec + rel_vec)
        scores = self.prototype_matrix @ query
        scores[self.concept_index[source]] = -np.inf
        count = min(k, len(self.concepts))
        indexes = np.argpartition(scores, -count)[-count:]
        indexes = indexes[np.argsort(scores[indexes])[::-1]]
        return [(self.concepts[int(i)], float(scores[int(i)])) for i in indexes]

    def save(self, path: Path, records: list[ConceptRecord], history: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        names = np.array(sorted(self.relation_vectors), dtype=object)
        vectors = np.stack([self.relation_vectors[name] for name in names]).astype(np.float32)
        matrix_names = np.array(sorted(self.relation_matrices), dtype=object)
        matrices = np.stack([self.relation_matrices[name] for name in matrix_names]).astype(np.float32) if len(matrix_names) else np.zeros((0, 0, 0), dtype=np.float32)
        metadata = {
            "records": [record.__dict__ for record in records],
            "history": history,
        }
        np.savez_compressed(
            path,
            relation_names=names,
            relation_vectors=vectors,
            matrix_names=matrix_names,
            relation_matrices=matrices,
            metadata=json.dumps(metadata, ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path, records: list[ConceptRecord] | None = None) -> "SIRGraphCore":
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata"]))
        loaded_records = [ConceptRecord(**row) for row in metadata["records"]]
        names = [str(name) for name in data["relation_names"]]
        vectors = data["relation_vectors"].astype(np.float32)
        relation_vectors = {name: vectors[i] for i, name in enumerate(names)}
        relation_matrices = {}
        if "matrix_names" in data.files:
            matrix_names = [str(name) for name in data["matrix_names"]]
            matrices = data["relation_matrices"].astype(np.float32)
            relation_matrices = {name: matrices[i] for i, name in enumerate(matrix_names)}
        return cls(records or loaded_records, relation_vectors, relation_matrices)


def train_relation_vectors(records: list[ConceptRecord], relations: list[ConceptRelation]) -> dict[str, np.ndarray]:
    lexical = LexicalSIRCore(records)
    buckets: dict[str, list[np.ndarray]] = {}
    for rel in relations:
        if rel.source not in lexical.vectors or rel.target not in lexical.vectors:
            continue
        source = np.array(lexical.vectors[rel.source], dtype=np.float32)
        target = np.array(lexical.vectors[rel.target], dtype=np.float32)
        buckets.setdefault(rel.relation, []).append(target - source)
    return {name: normalize_vector(np.mean(vectors, axis=0).astype(np.float32)) for name, vectors in buckets.items() if vectors}


def train_relation_matrices(records: list[ConceptRecord], relations: list[ConceptRelation], ridge: float) -> dict[str, np.ndarray]:
    lexical = LexicalSIRCore(records)
    pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for rel in relations:
        if rel.source not in lexical.vectors or rel.target not in lexical.vectors:
            continue
        source = np.array(lexical.vectors[rel.source], dtype=np.float32)
        target = np.array(lexical.vectors[rel.target], dtype=np.float32)
        pairs.setdefault(rel.relation, []).append((source, target))
    matrices: dict[str, np.ndarray] = {}
    dim = lexical.dim
    identity = np.eye(dim, dtype=np.float32)
    for name, rows in pairs.items():
        x = np.stack([row[0] for row in rows]).astype(np.float32)
        y = np.stack([row[1] for row in rows]).astype(np.float32)
        lhs = x.T @ x + ridge * identity
        rhs = x.T @ y
        matrices[name] = np.linalg.solve(lhs, rhs).astype(np.float32)
    return matrices


def evaluate(core: SIRGraphCore, relations: list[ConceptRelation]) -> dict[str, object]:
    started = time.perf_counter()
    hit1 = hit5 = hit10 = 0
    rr = 0.0
    by_relation: dict[str, dict[str, float]] = {}
    evaluated = 0
    for rel in relations:
        if rel.source not in core.concept_index or rel.target not in core.concept_index:
            continue
        preds = core.predict(rel.source, rel.relation, 10)
        if not preds:
            continue
        ranks = [concept for concept, _score in preds]
        evaluated += 1
        h1 = int(ranks[:1] == [rel.target])
        h5 = int(rel.target in ranks[:5])
        h10 = int(rel.target in ranks[:10])
        reciprocal = 1.0 / (ranks.index(rel.target) + 1) if rel.target in ranks else 0.0
        hit1 += h1
        hit5 += h5
        hit10 += h10
        rr += reciprocal
        row = by_relation.setdefault(rel.relation, {"n": 0, "hit1": 0, "hit5": 0, "hit10": 0, "mrr10": 0.0})
        row["n"] += 1
        row["hit1"] += h1
        row["hit5"] += h5
        row["hit10"] += h10
        row["mrr10"] += reciprocal
    total = max(evaluated, 1)
    return {
        "relations": evaluated,
        "hit1": round(hit1 / total, 4),
        "hit5": round(hit5 / total, 4),
        "hit10": round(hit10 / total, 4),
        "mrr10": round(rr / total, 4),
        "latency_ms_per_relation": round((time.perf_counter() - started) * 1000 / total, 4),
        "by_relation": {
            name: {
                "n": int(row["n"]),
                "hit1": round(row["hit1"] / row["n"], 4),
                "hit5": round(row["hit5"] / row["n"], 4),
                "hit10": round(row["hit10"] / row["n"], 4),
                "mrr10": round(row["mrr10"] / row["n"], 4),
            }
            for name, row in sorted(by_relation.items())
        },
    }


def split_relations(relations: list[ConceptRelation], seed: int, valid_ratio: float) -> tuple[list[ConceptRelation], list[ConceptRelation]]:
    rng = random.Random(seed)
    shuffled = list(relations)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - valid_ratio))
    return shuffled[:cut], shuffled[cut:]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def train_command(args: argparse.Namespace) -> None:
    records = load_records_jsonl(Path(args.records))
    relations = load_relations_jsonl(Path(args.graph))
    train_relations, valid_relations = split_relations(relations, args.seed, args.valid_ratio)
    vectors = train_relation_vectors(records, train_relations)
    matrices = train_relation_matrices(records, train_relations, args.ridge)
    core = SIRGraphCore(records, vectors, matrices)
    metrics = evaluate(core, valid_relations)
    metrics["train_relations"] = len(train_relations)
    metrics["valid_relations"] = len(valid_relations)
    metrics["relation_types"] = len(vectors)
    metrics["matrix_relation_types"] = len(matrices)
    core.save(Path(args.out), records, metrics)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved {args.out}")
    print(f"wrote {report}")


def eval_command(args: argparse.Namespace) -> None:
    records = load_records_jsonl(Path(args.records))
    relations = load_relations_jsonl(Path(args.graph))
    core = SIRGraphCore.load(Path(args.checkpoint), records)
    metrics = evaluate(core, relations)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


def demo_command(args: argparse.Namespace) -> None:
    records = load_records_jsonl(Path(args.records))
    core = SIRGraphCore.load(Path(args.checkpoint), records)
    preds = core.predict(args.source, args.relation, args.k)
    rows = []
    by_concept = {record.concept_id: record for record in records}
    for concept, score in preds:
        record = by_concept[concept]
        rows.append(
            {
                "concept": concept,
                "score": round(score, 6),
                "en": record.en[:3],
                "ru": record.ru[:3],
                "definition_en": record.definition_en,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a tiny SIR graph relation core.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    train.add_argument("--graph", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_graph.jsonl"))
    train.add_argument("--valid-ratio", type=float, default=0.2)
    train.add_argument("--ridge", type=float, default=0.05)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--out", default=str(CHECKPOINT))
    train.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "sir_graph_core_valid_eval.json"))

    evalp = sub.add_parser("eval")
    evalp.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    evalp.add_argument("--graph", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_graph.jsonl"))
    evalp.add_argument("--checkpoint", default=str(CHECKPOINT))
    evalp.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "sir_graph_core_eval.json"))

    demo = sub.add_parser("demo")
    demo.add_argument("--source", required=True)
    demo.add_argument("--relation", required=True)
    demo.add_argument("--k", type=int, default=5)
    demo.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    demo.add_argument("--checkpoint", default=str(CHECKPOINT))

    args = parser.parse_args()
    if args.cmd == "train":
        train_command(args)
    elif args.cmd == "eval":
        eval_command(args)
    elif args.cmd == "demo":
        demo_command(args)


if __name__ == "__main__":
    main()
