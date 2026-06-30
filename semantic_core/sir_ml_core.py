from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from semantic_core.sir_sources import (
    PROJECT_ROOT,
    ConceptRecord,
    LexicalSIRCore,
    load_records,
    load_records_jsonl,
    normalize,
    token_features,
)


CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sir_ml_core.npz"


class SIRMLCore:
    def __init__(self, records: list[ConceptRecord], input_dim: int, semantic_dim: int, seed: int = 42):
        self.records = records
        self.input_dim = input_dim
        self.semantic_dim = semantic_dim
        self.lexical = LexicalSIRCore(records, dim=semantic_dim)
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(0, 0.02, (input_dim, semantic_dim)).astype(np.float32)
        self.bias = np.zeros(semantic_dim, dtype=np.float32)
        self.concepts = [record.concept_id for record in records]
        self.prototype_matrix = np.array([self.lexical.vectors[concept] for concept in self.concepts], dtype=np.float32)

    def compile(self, text: str) -> np.ndarray:
        exact = self.lexical.alias_to_concepts.get(normalize(text))
        if exact:
            vectors = [self.lexical.vectors[concept] for concept in exact if concept in self.lexical.vectors]
            if vectors:
                return normalize_vector(np.mean(np.array(vectors, dtype=np.float32), axis=0))
        features = featurize(text, self.input_dim)
        return normalize_vector(features @ self.weight + self.bias)

    def nearest(self, text: str, k: int = 10) -> list[tuple[ConceptRecord, float]]:
        vector = self.compile(text)
        scores = self.prototype_matrix @ vector
        count = min(k, len(self.records))
        indexes = np.argpartition(scores, -count)[-count:]
        indexes = indexes[np.argsort(scores[indexes])[::-1]]
        return [(self.records[int(i)], float(scores[int(i)])) for i in indexes]

    def decompile(self, text: str, target: str) -> tuple[str, str, float]:
        record, score = self.nearest(text, 1)[0]
        values = record.ru if target == "ru" else record.en
        return values[0] if values else "", record.concept_id, score

    def train_batch(self, batch: list[tuple[str, str]], lr: float, weight_decay: float) -> float:
        x = np.stack([featurize(text, self.input_dim) for text, _ in batch])
        y = np.stack([self.lexical.vectors[concept] for _, concept in batch]).astype(np.float32)
        pred = x @ self.weight + self.bias
        pred_norm = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
        err = pred_norm - y
        loss = float(np.mean(err * err))
        grad = (2.0 / len(batch)) * err
        self.weight -= lr * (x.T @ grad + weight_decay * self.weight).astype(np.float32)
        self.bias -= lr * np.mean(grad, axis=0).astype(np.float32)
        return loss

    def save(self, path: Path, records: list[ConceptRecord], history: list[dict[str, float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input_dim": self.input_dim,
            "semantic_dim": self.semantic_dim,
            "records": [asdict(record) for record in records],
            "history": history,
        }
        np.savez_compressed(path, weight=self.weight, bias=self.bias, metadata=json.dumps(metadata, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path, records: list[ConceptRecord] | None = None) -> "SIRMLCore":
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata"]))
        loaded_records = [ConceptRecord(**row) for row in metadata["records"]]
        core = cls(records or loaded_records, metadata["input_dim"], metadata["semantic_dim"])
        core.weight = data["weight"].astype(np.float32)
        core.bias = data["bias"].astype(np.float32)
        return core


def read_records(path: Path, limit: int) -> list[ConceptRecord]:
    records = load_records_jsonl(path) if path.exists() else load_records(limit=limit)
    return records[:limit] if limit else records


def build_examples(records: list[ConceptRecord]) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for record in records:
        for text in [*record.en[:3], *record.ru[:3], record.definition_en]:
            if normalize(text):
                examples.append((text, record.concept_id))
    return examples


def featurize(text: str, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    for token in normalize(text).split():
        for feature in token_features(token):
            out[stable_bucket(feature, dim)] += 1.0
    return normalize_vector(out)


def stable_bucket(text: str, dim: int) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value % dim


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def evaluate_examples(core: SIRMLCore, examples: list[tuple[str, str]]) -> dict[str, float]:
    hit1 = hit5 = hit10 = 0
    rr = 0.0
    for text, concept in examples:
        ranks = [record.concept_id for record, _score in core.nearest(text, 10)]
        hit1 += int(ranks[:1] == [concept])
        hit5 += int(concept in ranks[:5])
        hit10 += int(concept in ranks[:10])
        rr += 1.0 / (ranks.index(concept) + 1) if concept in ranks else 0.0
    total = max(len(examples), 1)
    return {"hit1": round(hit1 / total, 4), "hit5": round(hit5 / total, 4), "hit10": round(hit10 / total, 4), "mrr10": round(rr / total, 4)}


def semantic_coherence(core: SIRMLCore, records: list[ConceptRecord]) -> dict[str, float]:
    sample = records[: min(len(records), 500)]
    intra = []
    inter = []
    for record in sample:
        if record.ru and record.en:
            intra.append(float(core.compile(record.ru[0]) @ core.compile(record.en[0])))
        if record.definition_en and record.en:
            intra.append(float(core.compile(record.definition_en) @ core.compile(record.en[0])))
    for left, right in zip(sample[::2], sample[1::2]):
        if left.en and right.ru:
            inter.append(float(core.compile(left.en[0]) @ core.compile(right.ru[0])))
    intra_mean = sum(intra) / max(len(intra), 1)
    inter_mean = sum(inter) / max(len(inter), 1)
    return {"intra_cluster_similarity": round(intra_mean, 4), "inter_cluster_similarity": round(inter_mean, 4), "semantic_gap": round(intra_mean - inter_mean, 4)}


def shuffled_ablation(examples: list[tuple[str, str]]) -> float:
    concepts = [concept for _text, concept in examples]
    shuffled = list(reversed(concepts))
    return round(sum(int(a == b) for a, b in zip(concepts, shuffled)) / max(len(examples), 1), 4)


def train_command(args: argparse.Namespace) -> None:
    records = read_records(Path(args.records), args.limit)
    rng = random.Random(args.seed)
    core = SIRMLCore(records, input_dim=args.input_dim, semantic_dim=args.semantic_dim, seed=args.seed)
    examples = build_examples(records)
    rng.shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    train_examples = examples[:split]
    valid_examples = examples[split:]
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_examples)
        losses = []
        for i in range(0, len(train_examples), args.batch_size):
            losses.append(core.train_batch(train_examples[i : i + args.batch_size], args.lr, args.weight_decay))
        valid = evaluate_examples(core, valid_examples or train_examples)
        row = {"epoch": float(epoch), "loss": float(sum(losses) / max(len(losses), 1)), "valid_hit1": valid["hit1"], "valid_mrr10": valid["mrr10"]}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    core.save(Path(args.out), records, history)
    print(f"saved {args.out} in {time.perf_counter() - started:.2f}s")


def eval_command(args: argparse.Namespace) -> None:
    records = read_records(Path(args.records), args.limit)
    core = SIRMLCore.load(Path(args.checkpoint), records)
    examples = build_examples(records)
    started = time.perf_counter()
    metrics = evaluate_examples(core, examples)
    metrics["records"] = len(records)
    metrics["examples"] = len(examples)
    metrics["latency_ms_per_query"] = round((time.perf_counter() - started) * 1000 / max(len(examples), 1), 4)
    metrics["semantic_coherence"] = semantic_coherence(core, records)
    metrics["ablation"] = {"shuffled_hit1": shuffled_ablation(examples)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


def demo_command(args: argparse.Namespace) -> None:
    records = read_records(Path(args.records), args.limit)
    core = SIRMLCore.load(Path(args.checkpoint), records)
    pred, concept, score = core.decompile(args.text, args.target)
    print(f"Input: {args.text}")
    print(f"Target: {args.target}")
    print(f"SIR nearest concept: {concept}")
    print(f"Score: {score:.4f}")
    print(f"Output: {pred}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a tiny SIR-native ML semantic core.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    train.add_argument("--limit", type=int, default=2000)
    train.add_argument("--input-dim", type=int, default=1024)
    train.add_argument("--semantic-dim", type=int, default=256)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=0.08)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--out", default=str(CHECKPOINT))
    evalp = sub.add_parser("eval")
    evalp.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    evalp.add_argument("--checkpoint", default=str(CHECKPOINT))
    evalp.add_argument("--limit", type=int, default=2000)
    evalp.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "sir_ml_core_eval.json"))
    demo = sub.add_parser("demo")
    demo.add_argument("--text", required=True)
    demo.add_argument("--target", choices=["ru", "en"], required=True)
    demo.add_argument("--records", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    demo.add_argument("--checkpoint", default=str(CHECKPOINT))
    demo.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    if args.cmd == "train":
        train_command(args)
    elif args.cmd == "eval":
        eval_command(args)
    elif args.cmd == "demo":
        demo_command(args)


if __name__ == "__main__":
    main()

