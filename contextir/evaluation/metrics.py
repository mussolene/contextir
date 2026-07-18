from __future__ import annotations

import difflib
import time
from pathlib import Path

import numpy as np

from contextir.models.baseline_translator import BaselineTranslator
from contextir.models.semantic_encoder import SemanticEncoder
from contextir.models.semantic_translator import SemanticTranslator, cosine
from contextir.tokenizer import normalize, token_accuracy


def char_similarity(pred: str, gold: str) -> float:
    return difflib.SequenceMatcher(None, normalize(pred), normalize(gold)).ratio()


def translation_metrics(predictions: list[tuple[str, str]]) -> dict[str, float]:
    if not predictions:
        return {"exact_match": 0.0, "token_accuracy": 0.0, "char_similarity": 0.0}
    exact = [normalize(p) == normalize(g) for p, g in predictions]
    return {
        "exact_match": float(np.mean(exact)),
        "token_accuracy": float(np.mean([token_accuracy(p, g) for p, g in predictions])),
        "char_similarity": float(np.mean([char_similarity(p, g) for p, g in predictions])),
    }


def semantic_coherence(rows: list[dict], encoder: SemanticEncoder) -> dict[str, float]:
    vectors = [(row["meaning_id"], encoder.encode_text(row["source"])) for row in rows]
    intra: list[float] = []
    inter: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            score = cosine(vectors[i][1], vectors[j][1])
            if vectors[i][0] == vectors[j][0]:
                intra.append(score)
            else:
                inter.append(score)
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    return {
        "intra_cluster_similarity": intra_mean,
        "inter_cluster_similarity": inter_mean,
        "semantic_gap": intra_mean - inter_mean,
    }


def evaluate_baseline(rows: list[dict], baseline: BaselineTranslator) -> dict[str, float]:
    return translation_metrics([(baseline.translate(row["source"]), row["target"]) for row in rows])


def evaluate_semantic(rows: list[dict], encoder: SemanticEncoder, translator: SemanticTranslator) -> dict[str, float]:
    preds = []
    for row in rows:
        pred, _, _, _ = translator.translate(row["source"], row["target_lang"], encoder)
        preds.append((pred, row["target"]))
    return translation_metrics(preds)


def cycle_consistency(rows: list[dict], encoder: SemanticEncoder, translator: SemanticTranslator) -> float:
    scores: list[float] = []
    for row in rows:
        original = encoder.encode_text(row["source"])
        translated, _, _, _ = translator.translate(row["source"], row["target_lang"], encoder)
        after = encoder.encode_text(translated)
        scores.append(cosine(original, after))
    return float(np.mean(scores)) if scores else 0.0


def ablation_quality(rows: list[dict], encoder: SemanticEncoder, translator: SemanticTranslator, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vectors = [encoder.encode_text(row["source"]) for row in rows]
    rng.shuffle(vectors)
    preds = []
    for row, vector in zip(rows, vectors):
        pred, _, _ = translator.translate_vector(vector, row["target_lang"])
        preds.append((pred, row["target"]))
    return translation_metrics(preds)


def latency_ms(rows: list[dict], fn) -> float:
    start = time.perf_counter()
    for row in rows:
        fn(row)
    elapsed = time.perf_counter() - start
    return float((elapsed / max(len(rows), 1)) * 1000)


def file_size(path: str | Path) -> int:
    p = Path(path)
    return p.stat().st_size if p.exists() else 0

