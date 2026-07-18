from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contextir.models.semantic_encoder import SemanticEncoder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class SemanticTranslator:
    def __init__(self, prototypes: dict[str, np.ndarray], targets: dict[str, dict[str, str]], atoms: dict[str, list[str]]):
        self.prototypes = prototypes
        self.targets = targets
        self.atoms = atoms

    def nearest(self, vector: np.ndarray) -> tuple[str, float]:
        best_id = ""
        best_score = -2.0
        for meaning_id, proto in self.prototypes.items():
            score = cosine(vector, proto)
            if score > best_score:
                best_id = meaning_id
                best_score = score
        return best_id, best_score

    def translate_vector(self, vector: np.ndarray, target_lang: str) -> tuple[str, str, float]:
        meaning_id, score = self.nearest(vector)
        return self.targets.get(meaning_id, {}).get(target_lang, "<unk>"), meaning_id, score

    def translate(self, text: str, target_lang: str, encoder: SemanticEncoder) -> tuple[str, str, float, np.ndarray]:
        vector = encoder.encode_text(text)
        out, meaning_id, score = self.translate_vector(vector, target_lang)
        return out, meaning_id, score, vector

    def save(self, path: str | Path) -> None:
        payload = {
            "prototypes": {k: v.tolist() for k, v in self.prototypes.items()},
            "targets": self.targets,
            "atoms": self.atoms,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticTranslator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        prototypes = {k: np.array(v, dtype=np.float32) for k, v in payload["prototypes"].items()}
        return cls(prototypes, payload["targets"], payload["atoms"])


def train_semantic_translator(rows: list[dict], encoder: SemanticEncoder) -> SemanticTranslator:
    by_meaning: dict[str, list[dict]] = {}
    for row in rows:
        by_meaning.setdefault(row["meaning_id"], []).append(row)

    prototypes: dict[str, np.ndarray] = {}
    targets: dict[str, dict[str, str]] = {}
    atoms: dict[str, list[str]] = {}
    for meaning_id, group in by_meaning.items():
        vectors = [encoder.encode_text(row["source"]) for row in group]
        proto = np.mean(vectors, axis=0)
        norm = np.linalg.norm(proto)
        prototypes[meaning_id] = (proto / norm).astype(np.float32) if norm else proto.astype(np.float32)
        atoms[meaning_id] = group[0]["semantic_atoms"]
        targets[meaning_id] = {}
        for row in group:
            targets[meaning_id][row["target_lang"]] = row["target"]
    return SemanticTranslator(prototypes, targets, atoms)

