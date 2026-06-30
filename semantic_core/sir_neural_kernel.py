from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from semantic_core.sir_ml_core import featurize
from semantic_core.sir_sources import PROJECT_ROOT


@dataclass
class NeuralSIRExample:
    text: str
    concept_ids: list[str]
    intent: str
    has_privacy: bool


class NeuralSIRKernel:
    def __init__(self, concept_ids: list[str], intents: list[str], input_dim: int = 1024, hidden_dim: int = 128, seed: int = 42):
        self.concept_ids = concept_ids
        self.intents = intents
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.04, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_concepts = rng.normal(0, 0.04, (hidden_dim, len(concept_ids))).astype(np.float32)
        self.b_concepts = np.zeros(len(concept_ids), dtype=np.float32)
        self.w_intent = rng.normal(0, 0.04, (hidden_dim, len(intents))).astype(np.float32)
        self.b_intent = np.zeros(len(intents), dtype=np.float32)
        self.w_privacy = rng.normal(0, 0.04, (hidden_dim, 1)).astype(np.float32)
        self.b_privacy = np.zeros(1, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        x = np.stack([featurize(text, self.input_dim) for text in texts]).astype(np.float32)
        return np.tanh(x @ self.w1 + self.b1), x

    def predict(self, text: str, threshold: float = 0.45) -> dict[str, Any]:
        hidden, _x = self.encode([text])
        concept_probs = sigmoid(hidden @ self.w_concepts + self.b_concepts)[0]
        intent_probs = softmax(hidden @ self.w_intent + self.b_intent)[0]
        privacy_prob = float(sigmoid(hidden @ self.w_privacy + self.b_privacy)[0, 0])
        ranked = sorted(zip(self.concept_ids, concept_probs), key=lambda item: float(item[1]), reverse=True)
        concepts = [{"id": cid, "score": round(float(score), 4)} for cid, score in ranked if score >= threshold]
        if not concepts and ranked:
            concepts = [{"id": ranked[0][0], "score": round(float(ranked[0][1]), 4)}]
        return {
            "concepts": concepts,
            "intent": {"label": self.intents[int(np.argmax(intent_probs))], "confidence": round(float(np.max(intent_probs)), 4)},
            "privacy": {"has_protected_span": privacy_prob >= 0.5, "confidence": round(privacy_prob, 4)},
        }

    def train_batch(self, batch: list[NeuralSIRExample], lr: float, weight_decay: float) -> float:
        hidden, x = self.encode([item.text for item in batch])
        concept_y = np.stack([multi_hot(item.concept_ids, self.concept_ids) for item in batch]).astype(np.float32)
        intent_y = np.stack([one_hot(item.intent, self.intents) for item in batch]).astype(np.float32)
        privacy_y = np.array([[1.0 if item.has_privacy else 0.0] for item in batch], dtype=np.float32)

        concept_logits = hidden @ self.w_concepts + self.b_concepts
        intent_logits = hidden @ self.w_intent + self.b_intent
        privacy_logits = hidden @ self.w_privacy + self.b_privacy

        concept_p = sigmoid(concept_logits)
        intent_p = softmax(intent_logits)
        privacy_p = sigmoid(privacy_logits)

        concept_loss = bce_loss(concept_p, concept_y)
        intent_loss = ce_loss(intent_p, intent_y)
        privacy_loss = bce_loss(privacy_p, privacy_y)
        loss = concept_loss + intent_loss + privacy_loss

        n = max(len(batch), 1)
        d_concept = (concept_p - concept_y) / n
        d_intent = (intent_p - intent_y) / n
        d_privacy = (privacy_p - privacy_y) / n

        grad_hidden = d_concept @ self.w_concepts.T + d_intent @ self.w_intent.T + d_privacy @ self.w_privacy.T
        grad_z1 = grad_hidden * (1.0 - hidden * hidden)

        self.w_concepts -= lr * (hidden.T @ d_concept + weight_decay * self.w_concepts)
        self.b_concepts -= lr * d_concept.sum(axis=0)
        self.w_intent -= lr * (hidden.T @ d_intent + weight_decay * self.w_intent)
        self.b_intent -= lr * d_intent.sum(axis=0)
        self.w_privacy -= lr * (hidden.T @ d_privacy + weight_decay * self.w_privacy)
        self.b_privacy -= lr * d_privacy.sum(axis=0)
        self.w1 -= lr * (x.T @ grad_z1 + weight_decay * self.w1)
        self.b1 -= lr * grad_z1.sum(axis=0)
        return float(loss)

    def save(self, path: Path, history: list[dict[str, float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "concept_ids": self.concept_ids,
            "intents": self.intents,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "history": history,
        }
        np.savez_compressed(
            path,
            w1=self.w1,
            b1=self.b1,
            w_concepts=self.w_concepts,
            b_concepts=self.b_concepts,
            w_intent=self.w_intent,
            b_intent=self.b_intent,
            w_privacy=self.w_privacy,
            b_privacy=self.b_privacy,
            metadata=json.dumps(metadata, ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path) -> "NeuralSIRKernel":
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata"]))
        model = cls(metadata["concept_ids"], metadata["intents"], metadata["input_dim"], metadata["hidden_dim"])
        for name in ["w1", "b1", "w_concepts", "b_concepts", "w_intent", "b_intent", "w_privacy", "b_privacy"]:
            setattr(model, name, data[name].astype(np.float32))
        return model


def read_examples(path: Path) -> list[NeuralSIRExample]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        contract = row["sir_contract"]
        examples.append(
            NeuralSIRExample(
                text=row["input_text"],
                concept_ids=[item["id"] for item in contract["concepts"]],
                intent=contract["intent"]["label"],
                has_privacy=bool(contract["protected_spans"]),
            )
        )
    return examples


def build_vocab(examples: Iterable[NeuralSIRExample]) -> tuple[list[str], list[str]]:
    concept_ids = sorted({concept for item in examples for concept in item.concept_ids})
    intents = sorted({item.intent for item in examples})
    return concept_ids, intents


def evaluate(model: NeuralSIRKernel, examples: list[NeuralSIRExample], threshold: float = 0.45) -> dict[str, float]:
    concept_f1 = []
    intent_hit = 0
    privacy_hit = 0
    for item in examples:
        pred = model.predict(item.text, threshold=threshold)
        pred_ids = {row["id"] for row in pred["concepts"]}
        gold_ids = set(item.concept_ids)
        shared = pred_ids & gold_ids
        precision = len(shared) / max(len(pred_ids), 1)
        recall = len(shared) / max(len(gold_ids), 1)
        concept_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        intent_hit += int(pred["intent"]["label"] == item.intent)
        privacy_hit += int(bool(pred["privacy"]["has_protected_span"]) == item.has_privacy)
    total = max(len(examples), 1)
    return {
        "examples": float(len(examples)),
        "concept_f1": round(float(sum(concept_f1) / total), 4),
        "intent_accuracy": round(intent_hit / total, 4),
        "privacy_accuracy": round(privacy_hit / total, 4),
    }


def train_command(args: argparse.Namespace) -> None:
    train_examples = read_examples(Path(args.train))
    valid_examples = read_examples(Path(args.valid)) if Path(args.valid).exists() else train_examples
    concept_ids, intents = build_vocab([*train_examples, *valid_examples])
    model = NeuralSIRKernel(concept_ids, intents, input_dim=args.input_dim, hidden_dim=args.hidden_dim, seed=args.seed)
    rng = random.Random(args.seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_examples)
        losses = []
        for i in range(0, len(train_examples), args.batch_size):
            losses.append(model.train_batch(train_examples[i : i + args.batch_size], args.lr, args.weight_decay))
        metrics = evaluate(model, valid_examples, threshold=args.threshold)
        row = {"epoch": float(epoch), "loss": round(sum(losses) / max(len(losses), 1), 4), **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    model.save(Path(args.out), history)
    report = {"train_seconds": round(time.perf_counter() - started, 3), "valid": evaluate(model, valid_examples, args.threshold), "history": history}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.out}")
    print(f"wrote {args.report}")


def eval_command(args: argparse.Namespace) -> None:
    model = NeuralSIRKernel.load(Path(args.checkpoint))
    examples = read_examples(Path(args.data))
    metrics = evaluate(model, examples, threshold=args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def demo_command(args: argparse.Namespace) -> None:
    model = NeuralSIRKernel.load(Path(args.checkpoint))
    print(json.dumps(model.predict(args.text, threshold=args.threshold), ensure_ascii=False, indent=2))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)


def bce_loss(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return float(-np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred)))


def ce_loss(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.clip(pred, 1e-6, 1.0)
    return float(-np.mean(np.sum(target * np.log(pred), axis=1)))


def multi_hot(values: list[str], vocab: list[str]) -> np.ndarray:
    index = {value: i for i, value in enumerate(vocab)}
    out = np.zeros(len(vocab), dtype=np.float32)
    for value in values:
        if value in index:
            out[index[value]] = 1.0
    return out


def one_hot(value: str, vocab: list[str]) -> np.ndarray:
    out = np.zeros(len(vocab), dtype=np.float32)
    out[vocab.index(value)] = 1.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the first neural SIR kernel.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--train", default=str(PROJECT_ROOT / "data" / "sir_training" / "train.jsonl"))
    train.add_argument("--valid", default=str(PROJECT_ROOT / "data" / "sir_training" / "valid.jsonl"))
    train.add_argument("--out", default=str(PROJECT_ROOT / "checkpoints" / "sir_neural_kernel_smoke.npz"))
    train.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "sir_neural_kernel_train.json"))
    train.add_argument("--input-dim", type=int, default=1024)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=0.08)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--threshold", type=float, default=0.45)
    train.add_argument("--seed", type=int, default=42)
    evalp = sub.add_parser("eval")
    evalp.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "sir_neural_kernel_smoke.npz"))
    evalp.add_argument("--data", default=str(PROJECT_ROOT / "data" / "sir_training" / "test.jsonl"))
    evalp.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "sir_neural_kernel_eval.json"))
    evalp.add_argument("--threshold", type=float, default=0.45)
    demo = sub.add_parser("demo")
    demo.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "sir_neural_kernel_smoke.npz"))
    demo.add_argument("--text", required=True)
    demo.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()
    if args.cmd == "train":
        train_command(args)
    elif args.cmd == "eval":
        eval_command(args)
    elif args.cmd == "demo":
        demo_command(args)


if __name__ == "__main__":
    main()
