from __future__ import annotations

from pathlib import Path

import numpy as np

from contextir.tokenizer import normalize


class SemanticEncoder:
    def __init__(
        self,
        atom_vectors: dict[str, np.ndarray],
        phrase_atoms: dict[str, list[str]],
        semantic_dim: int,
        atom_aliases: dict[str, str] | None = None,
    ):
        self.atom_vectors = atom_vectors
        self.phrase_atoms = phrase_atoms
        self.semantic_dim = semantic_dim
        self.atom_aliases = atom_aliases or {}

    def encode_atoms(self, atoms: list[str]) -> np.ndarray:
        vectors = [self.atom_vectors[a] for a in atoms if a in self.atom_vectors]
        if not vectors:
            return np.zeros(self.semantic_dim, dtype=np.float32)
        vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm else vec.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        norm = normalize(text)
        atoms = self.phrase_atoms.get(norm, [])
        if not atoms:
            atoms = sorted({atom for alias, atom in self.atom_aliases.items() if alias and alias in norm})
        return self.encode_atoms(atoms)

    def save(self, path: str | Path) -> None:
        keys = np.array(list(self.atom_vectors.keys()), dtype=object)
        vectors = np.stack([self.atom_vectors[k] for k in keys]) if len(keys) else np.zeros((0, self.semantic_dim))
        phrase_keys = np.array(list(self.phrase_atoms.keys()), dtype=object)
        phrase_vals = np.array(["\t".join(self.phrase_atoms[k]) for k in phrase_keys], dtype=object)
        alias_keys = np.array(list(self.atom_aliases.keys()), dtype=object)
        alias_vals = np.array([self.atom_aliases[k] for k in alias_keys], dtype=object)
        np.savez_compressed(
            path,
            keys=keys,
            vectors=vectors,
            phrase_keys=phrase_keys,
            phrase_vals=phrase_vals,
            alias_keys=alias_keys,
            alias_vals=alias_vals,
            semantic_dim=np.array([self.semantic_dim]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SemanticEncoder":
        data = np.load(path, allow_pickle=True)
        keys = list(data["keys"])
        vectors = data["vectors"]
        atom_vectors = {str(k): vectors[i].astype(np.float32) for i, k in enumerate(keys)}
        phrase_atoms = {str(k): str(v).split("\t") if str(v) else [] for k, v in zip(data["phrase_keys"], data["phrase_vals"])}
        atom_aliases = {}
        if "alias_keys" in data.files:
            atom_aliases = {str(k): str(v) for k, v in zip(data["alias_keys"], data["alias_vals"])}
        return cls(atom_vectors, phrase_atoms, int(data["semantic_dim"][0]), atom_aliases)


def default_atom_aliases() -> dict[str, str]:
    from contextir.dataset import ACTIONS, ENTITIES, PLACES

    aliases: dict[str, str] = {}
    for eid, ru, en, ru_alts, en_alts in ENTITIES:
        for value in [ru, en, *ru_alts, *en_alts]:
            aliases[normalize(value)] = f"object:{eid}"
    for aid, ru, en, ru_alts, en_alts in ACTIONS:
        for value in [ru, en, *ru_alts, *en_alts]:
            aliases[normalize(value)] = f"state:{aid}"
    for pid, ru, en, ru_alts, en_alts in PLACES:
        for value in [ru, en, *ru_alts, *en_alts]:
            aliases[normalize(value)] = f"place:{pid}"
    return aliases


def train_encoder(rows: list[dict], semantic_dim: int, seed: int, noise: float) -> SemanticEncoder:
    rng = np.random.default_rng(seed)
    atoms = sorted({atom for row in rows for atom in row["semantic_atoms"]})
    base: dict[str, np.ndarray] = {}
    for atom in atoms:
        vec = rng.normal(0, 1, semantic_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        base[atom] = vec
    phrase_atoms = {}
    for row in rows:
        atoms_for_phrase = row["semantic_atoms"]
        phrase_atoms[normalize(row["source"])] = atoms_for_phrase
        if noise:
            for atom in atoms_for_phrase:
                base[atom] = base[atom] + rng.normal(0, noise, semantic_dim).astype(np.float32)
                base[atom] = base[atom] / np.linalg.norm(base[atom])
    return SemanticEncoder(base, phrase_atoms, semantic_dim, default_atom_aliases())
