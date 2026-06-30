from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from semantic_core.sir_graph import ConceptRelation, load_relations_jsonl
from semantic_core.sir_sources import PROJECT_ROOT, ConceptRecord, LexicalSIRCore, load_records_jsonl


CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sir_graph_embedding_core.npz"


class SIRGraphEmbeddingCore:
    def __init__(self, records: list[ConceptRecord], relation_names: list[str], dim: int, seed: int = 42):
        self.records = records
        self.concepts = [record.concept_id for record in records]
        self.concept_index = {concept: i for i, concept in enumerate(self.concepts)}
        self.relation_names = relation_names
        self.relation_index = {name: i for i, name in enumerate(relation_names)}
        self.dim = dim
        lexical = LexicalSIRCore(records, dim=dim)
        rng = np.random.default_rng(seed)
        self.concept_embeddings = np.array([lexical.vectors[concept] for concept in self.concepts], dtype=np.float32)
        self.concept_embeddings += rng.normal(0, 0.01, self.concept_embeddings.shape).astype(np.float32)
        self.concept_embeddings = normalize_rows(self.concept_embeddings)
        self.relation_embeddings = rng.normal(0, 0.05, (len(relation_names), dim)).astype(np.float32)
        self.relation_embeddings = normalize_rows(self.relation_embeddings)

    def score(self, source_idx: int, relation_idx: int, target_idx: int) -> float:
        vec = self.concept_embeddings[source_idx] + self.relation_embeddings[relation_idx] - self.concept_embeddings[target_idx]
        return -float(np.linalg.norm(vec))

    def predict(self, source: str, relation: str, k: int = 10) -> list[tuple[str, float]]:
        if source not in self.concept_index or relation not in self.relation_index:
            return []
        source_idx = self.concept_index[source]
        relation_idx = self.relation_index[relation]
        query = self.concept_embeddings[source_idx] + self.relation_embeddings[relation_idx]
        distances = np.linalg.norm(self.concept_embeddings - query, axis=1)
        distances[source_idx] = np.inf
        count = min(k, len(self.concepts))
        indexes = np.argpartition(distances, count - 1)[:count]
        indexes = indexes[np.argsort(distances[indexes])]
        return [(self.concepts[int(i)], -float(distances[int(i)])) for i in indexes]

    def train_edge(self, source_idx: int, relation_idx: int, target_idx: int, negative_idx: int, lr: float, margin: float) -> float:
        source = self.concept_embeddings[source_idx]
        relation = self.relation_embeddings[relation_idx]
        target = self.concept_embeddings[target_idx]
        negative = self.concept_embeddings[negative_idx]
        pos_delta = source + relation - target
        neg_delta = source + relation - negative
        pos_dist = float(np.linalg.norm(pos_delta) + 1e-8)
        neg_dist = float(np.linalg.norm(neg_delta) + 1e-8)
        loss = margin + pos_dist - neg_dist
        if loss <= 0:
            return 0.0
        pos_grad = pos_delta / pos_dist
        neg_grad = neg_delta / neg_dist
        self.concept_embeddings[source_idx] -= lr * (pos_grad - neg_grad)
        self.relation_embeddings[relation_idx] -= lr * (pos_grad - neg_grad)
        self.concept_embeddings[target_idx] += lr * pos_grad
        self.concept_embeddings[negative_idx] -= lr * neg_grad
        self.concept_embeddings[[source_idx, target_idx, negative_idx]] = normalize_rows(self.concept_embeddings[[source_idx, target_idx, negative_idx]])
        self.relation_embeddings[relation_idx] = normalize_vector(self.relation_embeddings[relation_idx])
        return float(loss)

    def save(self, path: Path, records: list[ConceptRecord], history: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "records": [record.__dict__ for record in records],
            "relation_names": self.relation_names,
            "dim": self.dim,
            "history": history,
        }
        np.savez_compressed(
            path,
            concept_embeddings=self.concept_embeddings,
            relation_embeddings=self.relation_embeddings,
            metadata=json.dumps(metadata, ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path, records: list[ConceptRecord] | None = None) -> "SIRGraphEmbeddingCore":
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata"]))
        loaded_records = [ConceptRecord(**row) for row in metadata["records"]]
        core = cls(records or loaded_records, metadata["relation_names"], metadata["dim"])
        core.concept_embeddings = data["concept_embeddings"].astype(np.float32)
        core.relation_embeddings = data["relation_embeddings"].astype(np.float32)
        return core


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a SIR graph embedding core with negative sampling.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    add_common_args(train)
    train.add_argument("--dim", type=int, default=128)
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--lr", type=float, default=0.03)
    train.add_argument("--margin", type=float, default=0.35)
    train.add_argument("--valid-ratio", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--out", default=str(CHECKPOINT))
    train.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "sir_graph_embedding_valid_eval.json"))

    evalp = sub.add_parser("eval")
    add_common_args(evalp)
    evalp.add_argument("--checkpoint", default=str(CHECKPOINT))
    evalp.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "sir_graph_embedding_eval.json"))

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


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    parser.add_argument("--graph", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_graph.jsonl"))
    parser.add_argument("--limit-relations", type=int, default=0)


def train_command(args: argparse.Namespace) -> None:
    records = load_records_jsonl(Path(args.records))
    relations = load_relations_jsonl(Path(args.graph))
    if args.limit_relations:
        relations = relations[: args.limit_relations]
    train_relations, valid_relations = split_relations(relations, args.seed, args.valid_ratio)
    relation_names = sorted({rel.relation for rel in relations})
    core = SIRGraphEmbeddingCore(records, relation_names, args.dim, args.seed)
    rng = random.Random(args.seed)
    train_edges = indexed_edges(core, train_relations)
    valid_edges = [rel for rel in valid_relations if rel.source in core.concept_index and rel.target in core.concept_index and rel.relation in core.relation_index]
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_edges)
        losses = []
        for source_idx, relation_idx, target_idx in train_edges:
            negative_idx = sample_negative(rng, len(core.concepts), source_idx, target_idx)
            losses.append(core.train_edge(source_idx, relation_idx, target_idx, negative_idx, args.lr, args.margin))
        metrics = evaluate(core, valid_edges)
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / max(len(losses), 1), 6),
            "valid_hit1": metrics["hit1"],
            "valid_hit10": metrics["hit10"],
            "valid_mrr10": metrics["mrr10"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    metrics = evaluate(core, valid_edges)
    metrics["train_relations"] = len(train_edges)
    metrics["valid_relations"] = len(valid_edges)
    metrics["relation_types"] = len(relation_names)
    metrics["epochs"] = args.epochs
    metrics["train_seconds"] = round(time.perf_counter() - started, 3)
    metrics["history"] = history
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
    if args.limit_relations:
        relations = relations[: args.limit_relations]
    core = SIRGraphEmbeddingCore.load(Path(args.checkpoint), records)
    metrics = evaluate(core, relations)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


def demo_command(args: argparse.Namespace) -> None:
    records = load_records_jsonl(Path(args.records))
    core = SIRGraphEmbeddingCore.load(Path(args.checkpoint), records)
    by_concept = {record.concept_id: record for record in records}
    rows = []
    for concept, score in core.predict(args.source, args.relation, args.k):
        record = by_concept[concept]
        rows.append({"concept": concept, "score": round(score, 6), "en": record.en[:3], "ru": record.ru[:3], "definition_en": record.definition_en})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def indexed_edges(core: SIRGraphEmbeddingCore, relations: list[ConceptRelation]) -> list[tuple[int, int, int]]:
    edges = []
    for rel in relations:
        if rel.source in core.concept_index and rel.target in core.concept_index and rel.relation in core.relation_index:
            edges.append((core.concept_index[rel.source], core.relation_index[rel.relation], core.concept_index[rel.target]))
    return edges


def sample_negative(rng: random.Random, count: int, source_idx: int, target_idx: int) -> int:
    while True:
        idx = rng.randrange(count)
        if idx != source_idx and idx != target_idx:
            return idx


def split_relations(relations: list[ConceptRelation], seed: int, valid_ratio: float) -> tuple[list[ConceptRelation], list[ConceptRelation]]:
    rng = random.Random(seed)
    shuffled = list(relations)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - valid_ratio))
    return shuffled[:cut], shuffled[cut:]


def evaluate(core: SIRGraphEmbeddingCore, relations: list[ConceptRelation]) -> dict[str, object]:
    started = time.perf_counter()
    hit1 = hit5 = hit10 = 0
    rr = 0.0
    by_relation: dict[str, dict[str, float]] = {}
    evaluated = 0
    for rel in relations:
        if rel.source not in core.concept_index or rel.target not in core.concept_index or rel.relation not in core.relation_index:
            continue
        ranks = [concept for concept, _score in core.predict(rel.source, rel.relation, 10)]
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


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    return matrix / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


if __name__ == "__main__":
    main()

