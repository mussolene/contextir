from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tarfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
OMW_DIR = EXTERNAL_DIR / "omw-data"
EWN_DIR = EXTERNAL_DIR / "english-wordnet"
WN30_DIR = EXTERNAL_DIR / "WNdb-3.0"
WN30_ARCHIVE = EXTERNAL_DIR / "WNdb-3.0.tar.gz"
WN30_URL = "https://wordnetcode.princeton.edu/3.0/WNdb-3.0.tar.gz"


@dataclass
class ConceptRecord:
    concept_id: str
    en: list[str]
    ru: list[str]
    definition_en: str
    pos: str
    source: str = "omw+wordnet30"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[\wё-]+", text.lower(), flags=re.IGNORECASE))


def download_sources() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    clone_or_update("https://github.com/omwn/omw-data.git", OMW_DIR, ["--branch", "v2.0"])
    clone_or_update("https://github.com/globalwordnet/english-wordnet.git", EWN_DIR, [])
    download_wordnet30()


def clone_or_update(url: str, path: Path, extra: list[str]) -> None:
    if path.exists():
        subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1"], check=False)
        return
    subprocess.run(["git", "clone", "--depth", "1", *extra, url, str(path)], check=True)


def download_wordnet30() -> None:
    if (WN30_DIR / "dict" / "data.noun").exists():
        return
    if not WN30_ARCHIVE.exists():
        subprocess.run(["curl", "-L", WN30_URL, "-o", str(WN30_ARCHIVE)], check=True)
    with tarfile.open(WN30_ARCHIVE, "r:gz") as archive:
        archive.extractall(WN30_DIR)


def load_records(limit: int = 0, require_definition: bool = True) -> list[ConceptRecord]:
    records = load_omw_records(require_definition=require_definition) if OMW_DIR.exists() else fallback_records()
    return records[:limit] if limit else records


def load_omw_records(require_definition: bool = True) -> list[ConceptRecord]:
    eng = parse_omw_lemmas(OMW_DIR / "wns" / "eng" / "wn-data-eng.tab", "eng")
    rus = parse_omw_lemmas(OMW_DIR / "wns" / "wikt" / "wn-wikt-rus.tab", "rus")
    definitions = parse_wordnet30_definitions(WN30_DIR)
    if not definitions and EWN_DIR.exists():
        definitions = parse_english_wordnet_definitions(EWN_DIR / "src" / "yaml")
    records: list[ConceptRecord] = []
    for concept_id in sorted(set(eng) & set(rus)):
        definition = definitions.get(concept_id, "")
        if require_definition and not definition:
            continue
        records.append(
            ConceptRecord(
                concept_id=concept_id,
                en=dedupe(eng[concept_id])[:8],
                ru=dedupe(rus[concept_id])[:8],
                definition_en=definition,
                pos=concept_id.rsplit("-", 1)[-1],
            )
        )
    return records


def parse_omw_lemmas(path: Path, lang: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        concept_id, kind, value = parts[0], parts[1], parts[2]
        if kind in {"lemma", f"{lang}:lemma"}:
            rows[concept_id].append(value.replace("_", " "))
    return rows


def parse_wordnet30_definitions(root: Path) -> dict[str, str]:
    dict_dir = next(root.glob("**/dict"), root / "dict") if root.exists() else root / "dict"
    files = {"data.noun": "n", "data.verb": "v", "data.adj": "a", "data.adv": "r"}
    definitions: dict[str, str] = {}
    for filename, pos in files.items():
        path = dict_dir / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("  "):
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            head, gloss = parts
            fields = head.split()
            if len(fields) < 3 or not fields[0].isdigit():
                continue
            mapped_pos = "a" if fields[2] == "s" else pos
            definitions[f"{fields[0]}-{mapped_pos}"] = gloss.split(";")[0].strip()
    return definitions


def parse_english_wordnet_definitions(yaml_dir: Path) -> dict[str, str]:
    definitions: dict[str, str] = {}
    synset_re = re.compile(r"^(\d{8}-[nvar]):\s*$")
    current = ""
    in_definition = False
    for path in sorted(yaml_dir.glob("*.yaml")):
        if path.name.startswith("entries-") or path.name == "frames.yaml":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = synset_re.match(line)
            if match:
                current = match.group(1)
                in_definition = False
                continue
            if not current:
                continue
            if line.startswith("  definition:"):
                in_definition = True
            elif in_definition and line.startswith("  - "):
                definitions.setdefault(current, line[4:].strip())
            elif in_definition and line.startswith("    "):
                definitions[current] = (definitions.get(current, "") + " " + line.strip()).strip()
            elif in_definition and line.startswith("  ") and not line.startswith("    "):
                in_definition = False
    return definitions


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        norm = normalize(value)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(value)
    return out


def fallback_records() -> list[ConceptRecord]:
    return [
        ConceptRecord("02121620-n", ["cat", "true cat"], ["кошка", "кот"], "feline mammal usually having thick soft fur", "n"),
        ConceptRecord("02084071-n", ["dog", "domestic dog"], ["собака", "пёс"], "a domesticated canid", "n"),
        ConceptRecord("03082979-n", ["computer"], ["компьютер", "ЭВМ"], "a machine for performing calculations automatically", "n"),
        ConceptRecord("02691156-n", ["airplane", "aeroplane"], ["самолёт", "аэроплан"], "an aircraft that has a fixed wing", "n"),
    ]


class LexicalSIRCore:
    def __init__(self, records: list[ConceptRecord], dim: int = 256):
        self.records = records
        self.dim = dim
        self.concept_to_record = {record.concept_id: record for record in records}
        self.alias_to_concepts: dict[str, list[str]] = defaultdict(list)
        for record in records:
            for value in [*record.en, *record.ru, record.definition_en]:
                norm = normalize(value)
                if norm:
                    self.alias_to_concepts[norm].append(record.concept_id)
        self.vectors = {record.concept_id: self._concept_vector(record) for record in records}
        self._matrix = None
        if np is not None:
            self._matrix = np.array([self.vectors[record.concept_id] for record in records], dtype=np.float32)

    def compile(self, text: str) -> list[float]:
        norm = normalize(text)
        exact = self.alias_to_concepts.get(norm)
        if exact:
            return mean_vectors([self.vectors[concept] for concept in exact], self.dim)
        scores: dict[str, float] = defaultdict(float)
        terms = set(norm.split())
        for record in self.records:
            haystack = set(normalize(" ".join([*record.en, *record.ru, record.definition_en])).split())
            overlap = len(terms & haystack)
            if overlap:
                scores[record.concept_id] += overlap
        if scores:
            top = sorted(scores, key=scores.get, reverse=True)[:4]
            return mean_vectors([self.vectors[concept] for concept in top], self.dim)
        return hashed_text_vector(norm, self.dim)

    def nearest(self, text: str, k: int = 5) -> list[tuple[ConceptRecord, float]]:
        vector = self.compile(text)
        if self._matrix is not None and np is not None:
            scores = self._matrix @ np.array(vector, dtype=np.float32)
            count = min(k, len(self.records))
            indexes = np.argpartition(scores, -count)[-count:]
            indexes = indexes[np.argsort(scores[indexes])[::-1]]
            return [(self.records[int(i)], float(scores[int(i)])) for i in indexes]
        scored = [(record, cosine(vector, self.vectors[record.concept_id])) for record in self.records]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def decompile(self, text: str, target: str) -> tuple[str, str, float]:
        record, score = self.nearest(text, 1)[0]
        values = record.ru if target == "ru" else record.en
        return values[0] if values else "", record.concept_id, score

    def _concept_vector(self, record: ConceptRecord) -> list[float]:
        return hashed_text_vector(" ".join([record.concept_id, record.pos, *record.en, *record.ru, record.definition_en]), self.dim)


def hashed_text_vector(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in normalize(text).split():
        for feature in token_features(token):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[bucket] += sign
    return l2_normalize(vector)


def token_features(token: str) -> Iterable[str]:
    yield f"tok:{token}"
    padded = f"<{token}>"
    for n in (2, 3, 4):
        for i in range(max(len(padded) - n + 1, 0)):
            yield f"c{n}:{padded[i:i+n]}"


def mean_vectors(vectors: list[list[float]], dim: int) -> list[float]:
    if not vectors:
        return [0.0] * dim
    out = [0.0] * dim
    for vector in vectors:
        for i, value in enumerate(vector):
            out[i] += value
    return l2_normalize([value / len(vectors) for value in out])


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def save_records(records: list[ConceptRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def load_records_jsonl(path: Path) -> list[ConceptRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "synset" in row and "concept_id" not in row:
            row["concept_id"] = row.pop("synset")
        records.append(ConceptRecord(**row))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare SIR concept sources.")
    parser.add_argument("--download", action="store_true", help="Clone/download OMW, English WordNet, and Princeton WordNet.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))
    args = parser.parse_args()
    if args.download:
        started = time.perf_counter()
        download_sources()
        print(f"downloaded sources in {time.perf_counter() - started:.1f}s")
    records = load_records(limit=args.limit)
    save_records(records, Path(args.out))
    print(f"wrote {len(records)} concept records to {args.out}")


if __name__ == "__main__":
    main()
