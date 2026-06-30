from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict
from typing import Iterable

from semantic_core.sir_sources import PROJECT_ROOT, ConceptRecord, LexicalSIRCore, load_record_bundle, normalize


@dataclass
class ConceptHit:
    concept_id: str
    score: float
    source: str
    en: str
    ru: str
    definition_en: str


@dataclass
class SegmentPacket:
    text: str
    concepts: list[ConceptHit]


@dataclass
class SIRPacket:
    text_id: str
    source_lang: str
    chars: int
    tokens: int
    segments: list[SegmentPacket]
    concept_ids: list[str]
    unknown_tokens: list[str]


@dataclass
class RoundtripResult:
    text_id: str
    source_lang: str
    target_lang: str
    chars: int
    tokens: int
    source_concepts: int
    final_concepts: int
    intersection_concepts: int
    concept_precision: float
    concept_recall: float
    concept_f1: float
    segment_coverage: float
    unknown_token_rate: float
    compression_ratio: float
    sir_chars: int
    latency_ms: float
    bridge_text: str
    reconstructed_text: str
    direct_baseline: dict[str, float | int | str]


class SIRRoundtrip:
    def __init__(self, records: list[ConceptRecord], dim: int = 256):
        self.records = records
        self.lexical = LexicalSIRCore(records, dim=dim)
        self.by_id = {record.concept_id: record for record in records}
        self.aliases = self._build_aliases(records)
        self.aliases_by_token: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.aliases_by_prefix: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for item in self.aliases:
            _lang, alias, _concept_id = item
            for token in alias.split():
                self.aliases_by_token[token].append(item)
                if len(token) >= 5:
                    self.aliases_by_prefix[token[:6]].append(item)

    def compile_text(
        self,
        text: str,
        source_lang: str,
        text_id: str = "inline",
        top_k_per_segment: int = 6,
        fallback_k: int = 0,
        fallback_threshold: float = 0.16,
    ) -> SIRPacket:
        segments = split_segments(text)
        packets: list[SegmentPacket] = []
        ordered_concepts: list[str] = []
        matched_tokens: set[str] = set()
        all_tokens = normalize(text).split()

        for segment in segments:
            hits = self._segment_hits(segment, source_lang)
            if len(hits) < top_k_per_segment and fallback_k > 0:
                for record, score in self.lexical.nearest(segment, fallback_k):
                    if score >= fallback_threshold:
                        hits.setdefault(
                            record.concept_id,
                            ConceptHit(
                                concept_id=record.concept_id,
                                score=round(score, 4),
                                source="vector",
                                en=first(record.en),
                                ru=first(record.ru),
                                definition_en=record.definition_en,
                            ),
                        )
            concepts = sorted(hits.values(), key=lambda hit: (-hit.score, hit.concept_id))[:top_k_per_segment]
            for hit in concepts:
                if hit.concept_id not in ordered_concepts:
                    ordered_concepts.append(hit.concept_id)
                for token in normalize(" ".join([hit.en, hit.ru])).split():
                    matched_tokens.add(token)
            packets.append(SegmentPacket(text=segment, concepts=concepts))

        unknown_tokens = [token for token in all_tokens if token not in matched_tokens and len(token) > 3]
        return SIRPacket(
            text_id=text_id,
            source_lang=source_lang,
            chars=len(text),
            tokens=len(all_tokens),
            segments=packets,
            concept_ids=ordered_concepts,
            unknown_tokens=unknown_tokens,
        )

    def decompile_packet(self, packet: SIRPacket, target_lang: str, max_concepts_per_segment: int = 4) -> str:
        rendered: list[str] = []
        seen: set[str] = set()
        for segment in packet.segments:
            terms = []
            for hit in segment.concepts[:max_concepts_per_segment]:
                if hit.concept_id in seen:
                    continue
                seen.add(hit.concept_id)
                record = self.by_id.get(hit.concept_id)
                if not record:
                    continue
                values = record.ru if target_lang == "ru" else record.en
                term = first(values)
                if term:
                    terms.append(term)
            if terms:
                rendered.append(", ".join(terms))
        return ". ".join(rendered)

    def roundtrip(self, row: dict[str, str], target_lang: str = "en") -> RoundtripResult:
        text_id = row.get("id", "inline")
        source_lang = row.get("lang", "ru")
        text = row["text"]
        started = time.perf_counter()
        source_packet = self.compile_text(text, source_lang=source_lang, text_id=text_id)
        bridge_text = self.decompile_packet(source_packet, target_lang=target_lang)
        bridge_packet = self.compile_text(bridge_text, source_lang=target_lang, text_id=f"{text_id}:bridge")
        reconstructed_text = self.decompile_packet(bridge_packet, target_lang=source_lang)
        final_packet = self.compile_text(reconstructed_text, source_lang=source_lang, text_id=f"{text_id}:final")
        latency_ms = (time.perf_counter() - started) * 1000

        source_set = set(source_packet.concept_ids)
        final_set = set(final_packet.concept_ids)
        intersection = source_set & final_set
        precision = len(intersection) / max(len(final_set), 1)
        recall = len(intersection) / max(len(source_set), 1)
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        segment_coverage = sum(1 for segment in source_packet.segments if segment.concepts) / max(len(source_packet.segments), 1)
        sir_chars = len(packet_to_json(source_packet))

        baseline = self.direct_baseline(text, source_lang=source_lang, target_lang=target_lang, reference_concepts=source_set)
        return RoundtripResult(
            text_id=text_id,
            source_lang=source_lang,
            target_lang=target_lang,
            chars=len(text),
            tokens=max(source_packet.tokens, 1),
            source_concepts=len(source_set),
            final_concepts=len(final_set),
            intersection_concepts=len(intersection),
            concept_precision=round(precision, 4),
            concept_recall=round(recall, 4),
            concept_f1=round(f1, 4),
            segment_coverage=round(segment_coverage, 4),
            unknown_token_rate=round(len(source_packet.unknown_tokens) / max(source_packet.tokens, 1), 4),
            compression_ratio=round(sir_chars / max(len(text), 1), 4),
            sir_chars=sir_chars,
            latency_ms=round(latency_ms, 2),
            bridge_text=bridge_text,
            reconstructed_text=reconstructed_text,
            direct_baseline=baseline,
        )

    def direct_baseline(self, text: str, source_lang: str, target_lang: str, reference_concepts: set[str]) -> dict[str, float | int | str]:
        source_packet = self.compile_text(
            text,
            source_lang=source_lang,
            text_id="baseline",
            top_k_per_segment=1,
            fallback_k=0,
        )
        bridge = self.decompile_packet(source_packet, target_lang=target_lang, max_concepts_per_segment=1)
        back_packet = self.compile_text(bridge, source_lang=target_lang, text_id="baseline:bridge", top_k_per_segment=1, fallback_k=0)
        final = self.decompile_packet(back_packet, target_lang=source_lang, max_concepts_per_segment=1)
        final_packet = self.compile_text(final, source_lang=source_lang, text_id="baseline:final", top_k_per_segment=1, fallback_k=0)
        final_set = set(final_packet.concept_ids)
        intersection = reference_concepts & final_set
        precision = len(intersection) / max(len(final_set), 1)
        recall = len(intersection) / max(len(reference_concepts), 1)
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        return {
            "source_concepts": len(source_packet.concept_ids),
            "reference_concepts": len(reference_concepts),
            "final_concepts": len(final_set),
            "intersection_concepts": len(intersection),
            "concept_precision": round(precision, 4),
            "concept_recall": round(recall, 4),
            "concept_f1": round(f1, 4),
            "bridge_text": bridge,
            "reconstructed_text": final,
        }

    def _segment_hits(self, segment: str, source_lang: str) -> dict[str, ConceptHit]:
        normalized = normalize(segment)
        padded = f" {normalized} "
        terms = set(normalized.split())
        hits: dict[str, ConceptHit] = {}
        if not normalized:
            return hits
        candidates: dict[tuple[str, str, str], None] = {}
        for term in sorted(terms):
            for item in self.aliases_by_token.get(term, []):
                candidates[item] = None
            if len(term) >= 5:
                for item in self.aliases_by_prefix.get(term[:6], []):
                    candidates[item] = None
        for lang, alias, concept_id in candidates:
            if lang != source_lang:
                continue
            score = 0.0
            source = ""
            record = self.by_id[concept_id]
            if f" {alias} " in padded:
                score = 1.0 if " " in alias else 0.92
                source = "alias"
            elif len(alias) >= 5 and any(loose_token_match(alias, term) for term in terms):
                score = 0.72
                source = "loose_alias"
            if not score:
                continue
            if record.source.startswith("sir-domain"):
                score += 0.35
                source = f"sir_domain_{source}"
            current = hits.get(concept_id)
            if current is None or score > current.score:
                hits[concept_id] = ConceptHit(
                    concept_id=concept_id,
                    score=score,
                    source=source,
                    en=first(record.en),
                    ru=first(record.ru),
                    definition_en=record.definition_en,
                )
        return hits

    @staticmethod
    def _build_aliases(records: Iterable[ConceptRecord]) -> list[tuple[str, str, str]]:
        aliases: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for record in records:
            for lang, values in (("en", record.en), ("ru", record.ru)):
                for value in values:
                    alias = normalize(value)
                    if len(alias) < 3:
                        continue
                    key = (lang, alias, record.concept_id)
                    if key not in seen:
                        seen.add(key)
                        aliases.append(key)
        aliases.sort(key=lambda item: (len(item[1].split()), len(item[1])), reverse=True)
        return aliases


def split_segments(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?;:\n]+", text) if normalize(part)]


def loose_token_match(alias: str, term: str) -> bool:
    if len(term) < 5:
        return False
    prefix = alias[: min(6, len(alias))]
    reverse = term[: min(6, len(term))]
    return term.startswith(prefix) or alias.startswith(reverse)


def first(values: list[str]) -> str:
    return values[0] if values else ""


def packet_to_json(packet: SIRPacket) -> str:
    return json.dumps(asdict(packet), ensure_ascii=False, separators=(",", ":"))


def read_roundtrip_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_roundtrip(records_path: Path) -> SIRRoundtrip:
    domain_path = PROJECT_ROOT / "data" / "concepts" / "sir_domain_records.jsonl"
    return SIRRoundtrip(load_record_bundle([records_path, domain_path]))
