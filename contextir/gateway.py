from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from contextir.sir_runtime import PresidioPrivacyScrubber, PrivacyScrubber, SIRRuntime, SIRV1Packet, detect_constraints, guess_intent, load_runtime
from contextir.sir_sources import normalize


Mode = Literal["auto", "raw", "hybrid", "semantic"]


@dataclass
class ContextBundle:
    """Private compilation result. Vault and source text never enter the public contract."""

    contract: dict[str, Any]
    vault: dict[str, str]
    sources: dict[str, str]


@dataclass
class ContractCheck:
    event_recall: float
    constraint_recall: float
    entity_recall: float
    lost_events: list[str]
    lost_constraints: list[str]
    lost_entities: list[str]
    needs_source: bool


@dataclass(frozen=True)
class TaskProfile:
    kind: Literal["operational", "retrieval", "exhaustive"]
    query_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    retrieval_confidence: float = 0.0


class ContextIR:
    """Adaptive context compiler with an optional lexical SIR enrichment layer."""

    def __init__(self, runtime: SIRRuntime | None = None, raw_threshold: int = 240, privacy: Any | None = None):
        self.runtime = runtime
        self.scrubber = privacy or (runtime.scrubber if runtime else PrivacyScrubber())
        self.raw_threshold = raw_threshold

    def compile(
        self,
        text: str,
        source_lang: str = "ru",
        target_lang: str = "ru",
        packet_id: str = "context",
        mode: Mode = "auto",
    ) -> dict[str, Any]:
        return self.compile_private(text, source_lang, target_lang, packet_id, mode).contract

    def compile_private(
        self,
        text: str,
        source_lang: str = "ru",
        target_lang: str = "ru",
        packet_id: str = "context",
        mode: Mode = "auto",
    ) -> ContextBundle:
        if not text.strip():
            raise ValueError("text must not be empty")
        if mode not in {"auto", "raw", "hybrid", "semantic"}:
            raise ValueError(f"unsupported mode: {mode}")

        started = time.perf_counter()
        scrubbed = self.scrubber.scrub(text, language=source_lang)
        segments = split_source(scrubbed.scrubbed_text)
        sources = {segment_id: value for segment_id, value in segments}
        task_profile = analyze_task(scrubbed.scrubbed_text, segments)
        normalized_tokens = set(normalize(scrubbed.scrubbed_text).split())
        intent = asdict(guess_intent(scrubbed.scrubbed_text, normalized_tokens=normalized_tokens))
        constraints = detect_constraints(scrubbed.scrubbed_text, normalized_tokens=normalized_tokens)
        if scrubbed.protected_spans:
            constraints.insert(0, {"type": "privacy", "value": "keep_placeholders_private"})

        retrieval_mode = choose_mode(
            mode,
            len(text),
            task_profile.retrieval_confidence,
            segments,
            self.raw_threshold,
            task_profile,
        )
        if task_profile.kind == "retrieval" and retrieval_mode != "raw":
            events: list[dict[str, Any]] = []
            entities = extract_entities(segments, scrubbed.protected_spans, include_numbers=False)
            confidence = task_profile.retrieval_confidence
            selected_mode = retrieval_mode
        else:
            events = extract_events(segments, source_lang)
            entities = extract_entities(segments, scrubbed.protected_spans)
            confidence = semantic_confidence(events, segments)
            selected_mode = choose_mode(mode, len(text), confidence, segments, self.raw_threshold, task_profile)
        included_refs = select_source_refs(selected_mode, segments, task_profile)
        if task_profile.kind == "retrieval" and selected_mode != "raw":
            included_text = " ".join(sources[ref] for ref in included_refs)
            entities = [item for item in entities if item["value"] in included_text]
        concepts = self._compact_concepts(scrubbed.scrubbed_text, source_lang, packet_id, selected_mode)
        public_spans = [{"placeholder": span.placeholder, "kind": span.kind} for span in scrubbed.protected_spans]
        if task_profile.kind == "retrieval" and selected_mode != "raw":
            public_spans = [item for item in public_spans if item["placeholder"] in included_text]

        contract: dict[str, Any] = {
            "version": "contextir.v2",
            "id": packet_id,
            "language": {"source": source_lang, "target": target_lang},
            "mode": selected_mode,
            "intent": {"label": intent["label"], "confidence": intent["confidence"]},
            "entities": entities,
            "events": events,
            "constraints": constraints,
            "concepts": concepts,
            "privacy": {"protected": public_spans},
            "source": {
                "refs": list(sources),
                "included": [{"ref": ref, "text": sources[ref]} for ref in included_refs],
            },
            "uncertainty": {
                "semantic_confidence": confidence,
                "requires_source": selected_mode != "semantic",
            },
        }
        compact_chars = len(json.dumps(contract, ensure_ascii=False, separators=(",", ":")))
        prompt_chars = len(self.render_prompt(contract))
        contract["stats"] = {
            "source_chars": len(text),
            "contract_chars": compact_chars,
            "prompt_chars": prompt_chars,
            "prompt_ratio": round(prompt_chars / max(len(text), 1), 4),
            "source_segments": len(segments),
            "included_segments": len(included_refs),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return ContextBundle(contract=contract, vault=scrubbed.vault, sources=sources)

    def render_prompt(self, contract: dict[str, Any]) -> str:
        if contract.get("version") != "contextir.v2":
            return json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
        if contract.get("mode") == "raw":
            return " ".join(str(item["text"]) for item in contract["source"]["included"])
        if contract.get("mode") == "hybrid" and contract["source"]["included"]:
            included_refs = {item["ref"] for item in contract["source"]["included"]}
            events_covered = all(event["source_ref"] in included_refs for event in contract.get("events", []))
            if events_covered:
                return "\n\n".join(str(item["text"]) for item in contract["source"]["included"])
        language = contract["language"]
        lines = [
            f"CTXIR/2 mode={contract['mode']} src={language['source']} out={language['target']} intent={contract['intent']['label']}",
        ]
        protected = contract["privacy"]["protected"]
        if protected:
            lines.append("PRIV=" + ",".join(f"{item['placeholder']}:{item['kind']}" for item in protected))
        if contract["constraints"]:
            lines.append("RULE=" + ";".join(f"{item['type']}:{item['value']}" for item in contract["constraints"]))
        if contract["entities"]:
            lines.append("ENT=" + ";".join(f"{item['id']}:{item['type']}={item['value']}" for item in contract["entities"]))
        for event in contract["events"]:
            flags = []
            if event["polarity"] == "negative":
                flags.append("NOT")
            if event["modality"] != "none":
                flags.append(event["modality"].upper())
            if event.get("condition"):
                flags.append("IF")
            args = ",".join(event["arguments"])
            prefix = "+".join(flags) + ":" if flags else ""
            repeats = f"*{event['count']}" if event.get("count", 1) > 1 else ""
            lines.append(f"EV {event['source_ref']}{repeats} {prefix}{event['predicate']}({args})")
        if contract["concepts"]:
            lines.append("TOPIC=" + ",".join(item["label"] for item in contract["concepts"]))
        for item in contract["source"]["included"]:
            lines.append(f"SRC {item['ref']}={item['text']}")
        lines.append("Answer naturally. Preserve RULE, NOT, numbers, and placeholders.")
        return "\n".join(lines)

    def decompile(self, contract: dict[str, Any], target_lang: str | None = None, include_anchors: bool = False) -> str:
        if contract.get("version") == "contextir.v2":
            included = contract.get("source", {}).get("included", [])
            if included:
                return " ".join(str(item["text"]) for item in included)
            rendered = []
            for event in contract.get("events", []):
                prefix = "not " if event.get("polarity") == "negative" else ""
                rendered.append(f"{prefix}{event.get('predicate', 'state')} {' '.join(event.get('arguments', []))}".strip())
            return ". ".join(rendered)
        return decompile_v1(contract, target_lang, include_anchors)

    def restore(self, text: str, bundle: ContextBundle, allowed: set[str] | None = None) -> str:
        """Restore only placeholders issued by this compilation and explicitly allowed."""

        allowlist = set(bundle.vault) if allowed is None else set(bundle.vault) & allowed
        restored = text
        for placeholder in sorted(allowlist, key=len, reverse=True):
            restored = restored.replace(placeholder, bundle.vault[placeholder])
        return restored

    def compare(self, expected: dict[str, Any], observed: dict[str, Any], threshold: float = 0.9) -> ContractCheck:
        expected_events = {event_signature(item) for item in expected.get("events", [])}
        observed_events = {event_signature(item) for item in observed.get("events", [])}
        expected_constraints = {constraint_signature(item) for item in expected.get("constraints", [])}
        observed_constraints = {constraint_signature(item) for item in observed.get("constraints", [])}
        expected_entities = {entity_signature(item) for item in expected.get("entities", []) if item.get("type") == "number"}
        observed_entities = {entity_signature(item) for item in observed.get("entities", []) if item.get("type") == "number"}
        event_recall = overlap_recall(expected_events, observed_events)
        constraint_recall = overlap_recall(expected_constraints, observed_constraints)
        entity_recall = overlap_recall(expected_entities, observed_entities)
        return ContractCheck(
            event_recall=event_recall,
            constraint_recall=constraint_recall,
            entity_recall=entity_recall,
            lost_events=sorted(expected_events - observed_events),
            lost_constraints=sorted(expected_constraints - observed_constraints),
            lost_entities=sorted(expected_entities - observed_entities),
            needs_source=min(event_recall, constraint_recall, entity_recall) < threshold,
        )

    def compile_legacy(
        self,
        text: str,
        source_lang: str = "ru",
        target_lang: str = "ru",
        packet_id: str = "contract",
    ) -> dict[str, Any]:
        if not self.runtime:
            raise RuntimeError("legacy compilation requires lexical=True")
        packet, _vault = self.runtime.compile_request(text, source_lang, target_lang, packet_id)
        return packet_to_v1_contract(packet)

    def _compact_concepts(self, text: str, source_lang: str, packet_id: str, mode: str) -> list[dict[str, Any]]:
        if not self.runtime or mode == "raw":
            return []
        packet = self.runtime.compiler.compile_text(text, source_lang=source_lang, text_id=packet_id, top_k_per_segment=2)
        out = []
        seen: set[str] = set()
        for segment in packet.segments:
            for hit in segment.concepts:
                if hit.concept_id in seen or hit.score < 0.9:
                    continue
                seen.add(hit.concept_id)
                label = hit.ru if source_lang == "ru" else hit.en
                out.append({"id": hit.concept_id, "label": label or hit.en or hit.ru, "score": hit.score})
                if len(out) == 8:
                    return out
        return out


SIRKernel = ContextIR


ACTION_PATTERNS = [
    ("send", r"\b(send|forward|dispatch|отправ\w*|переда\w*)\b"),
    ("cancel", r"\b(cancel|revoke|отмен\w*)\b"),
    ("redact", r"\b(redact|mask|hide|скры\w*|маскир\w*)\b"),
    ("translate", r"\b(translat\w*|перев\w*)\b"),
    ("compress", r"\b(compress\w*|сжим\w*|компресс\w*)\b"),
    ("preserve", r"\b(preserv\w*|retain\w*|сохран\w*)\b"),
    ("verify", r"\b(check|verify|validat\w*|провер\w*)\b"),
    ("build", r"\b(build|create|implement|созда\w*|постро\w*|реализ\w*)\b"),
    ("answer", r"\b(answer|respond|ответ\w*)\b"),
    ("pay", r"\b(pay|payment|оплат\w*|плат[её]ж\w*)\b"),
    ("call", r"\b(call|phone|звон\w*)\b"),
    ("use", r"\b(use|using|использ\w*)\b"),
]
ACTION_RE = re.compile(
    "|".join(f"(?P<a{index}>{pattern})" for index, (_name, pattern) in enumerate(ACTION_PATTERNS)),
    re.IGNORECASE,
)
POLARITY_RE = re.compile(r"\b(no|not|never|не|нельзя|никогда)\b")
REQUIREMENT_RE = re.compile(r"\b(must|should|need|долж\w*|нужно|надо)\b")
PROHIBITION_RE = re.compile(r"\b(must not|do not|don't|нельзя|не должен\w*)\b")
CONDITION_RE = re.compile(r"\b(if|unless|when|если|когда)\b")
SOURCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\n]+")
SALIENT_TOKEN_RE = re.compile(r"PII_[A-Z_]+_\d+|[\wё-]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*(?!\w)")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "we", "with", "you",
    "а", "без", "бы", "в", "для", "до", "и", "из", "или", "как", "на", "но", "о", "по", "с", "то", "у", "это", "я", "мы", "он", "она", "они",
    "не", "not", "if", "если", "must", "should", "должен", "должна", "нужно", "надо",
}


def split_source(text: str) -> list[tuple[str, str]]:
    parts = [part.strip() for part in SOURCE_SPLIT_RE.split(text) if part.strip()]
    return [(f"s{index}", part) for index, part in enumerate(parts, 1)]


def extract_events(segments: list[tuple[str, str]], source_lang: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    by_signature: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source_ref, text in segments:
        norm = text.lower()
        action_indexes = {int(match.lastgroup[1:]) for match in ACTION_RE.finditer(norm) if match.lastgroup}
        predicate = ACTION_PATTERNS[min(action_indexes)][0] if action_indexes else "state"
        polarity = "negative" if POLARITY_RE.search(norm) else "positive"
        modality = "requirement" if REQUIREMENT_RE.search(norm) else "none"
        if PROHIBITION_RE.search(norm):
            modality = "prohibition"
        condition = "if" if CONDITION_RE.search(norm) else ""
        arguments = salient_terms(text, predicate)
        confidence = 0.82 if predicate != "state" else 0.48
        if polarity == "negative" or condition:
            confidence = max(confidence, 0.72)
        signature = (predicate, tuple(arguments), polarity, modality, condition)
        existing = by_signature.get(signature)
        if existing:
            existing["count"] += 1
            continue
        event = {
                "id": f"e{len(events) + 1}",
                "predicate": predicate,
                "arguments": arguments,
                "polarity": polarity,
                "modality": modality,
                "condition": condition,
                "source_ref": source_ref,
                "confidence": round(confidence, 2),
                "count": 1,
            }
        events.append(event)
        by_signature[signature] = event
    return events


def salient_terms(text: str, predicate: str, limit: int = 5) -> list[str]:
    tokens = SALIENT_TOKEN_RE.findall(text.lower())
    out = []
    for token in tokens:
        if token in STOPWORDS or len(token) < 3 or token.startswith(predicate):
            continue
        if token not in out:
            out.append(token)
        if len(out) == limit:
            break
    return out


def extract_entities(
    segments: list[tuple[str, str]],
    protected_spans: list[Any],
    include_numbers: bool = True,
) -> list[dict[str, str]]:
    entities = []
    for span in protected_spans:
        entities.append({"id": f"p{len(entities) + 1}", "type": span.kind, "value": span.placeholder})
    if not include_numbers:
        return entities
    seen = {item["value"] for item in entities}
    for source_ref, text in segments:
        for value in NUMBER_RE.findall(text):
            if value not in seen:
                entities.append({"id": f"n{len(entities) + 1}", "type": "number", "value": value, "source_ref": source_ref})
                seen.add(value)
    return entities


def semantic_confidence(events: list[dict[str, Any]], segments: list[tuple[str, str]]) -> float:
    if not segments:
        return 0.0
    weighted = sum(float(event["confidence"]) * int(event.get("count", 1)) for event in events)
    return round(weighted / len(segments), 4)


def choose_mode(
    requested: Mode,
    chars: int,
    confidence: float,
    segments: list[tuple[str, str]],
    raw_threshold: int,
    task_profile: TaskProfile | None = None,
) -> str:
    if requested != "auto":
        return requested
    if chars <= raw_threshold:
        return "raw"
    if task_profile and task_profile.kind == "exhaustive":
        return "raw"
    if task_profile and task_profile.kind == "retrieval":
        has_evidence = task_profile.evidence_refs and task_profile.retrieval_confidence >= 0.12
        return "hybrid" if has_evidence else "raw"
    critical = sum(1 for _ref, text in segments if is_critical_source(text))
    if confidence < 0.65 or critical:
        return "hybrid"
    return "semantic"


def select_source_refs(
    mode: str,
    segments: list[tuple[str, str]],
    task_profile: TaskProfile | None = None,
) -> list[str]:
    if mode == "raw":
        return [ref for ref, _text in segments]
    if mode == "semantic":
        return []
    if task_profile and task_profile.kind == "retrieval":
        selected = set(task_profile.query_refs) | set(task_profile.evidence_refs)
        selected.update(paragraph_owner_refs(segments, task_profile.evidence_refs))
        selected.update(ref for ref, _text in segments[:2])
        selected.update(ref for ref, _text in segments[-2:])
        edge_segments = segments[:4] + segments[-6:]
        selected.update(
            ref
            for ref, text in edge_segments
            if re.search(r"\b(answer|enter|format|output|respond)\b", text, re.IGNORECASE)
        )
        return [ref for ref, _text in segments if ref in selected]
    selected = []
    seen_source = set()
    for ref, text in segments:
        if not is_critical_source(text):
            continue
        signature = " ".join(text.lower().split())
        if signature in seen_source:
            continue
        seen_source.add(signature)
        selected.append(ref)
        if len(selected) == 6:
            break
    if not selected and segments:
        selected.append(segments[0][0])
    selected_set = set(selected)
    selected_set.update(ref for ref, _text in segments[-2:])
    return [ref for ref, _text in segments if ref in selected_set][:8]


def paragraph_owner_refs(segments: list[tuple[str, str]], evidence_refs: tuple[str, ...]) -> set[str]:
    owner_by_ref: dict[str, str] = {}
    current_owner = ""
    for ref, text in segments:
        if re.match(r"Paragraph\s+\d+\s*:", text, re.IGNORECASE):
            current_owner = ref
        owner_by_ref[ref] = current_owner
    return {owner_by_ref[ref] for ref in evidence_refs if owner_by_ref.get(ref)}


EXHAUSTIVE_PATTERNS = [
    r"how many unique paragraphs",
    r"count (?:the )?(?:number|total|occurrences)",
    r"after removing duplicates",
    r"сколько (?:уникальн|различн|всего)",
    r"подсчита\w+ (?:количество|число)",
]

RETRIEVAL_MARKERS = [
    "based on the above text",
    "based on the following text",
    "which paragraph",
    "abstract is from",
    "read the following text and answer",
    "ответьте на вопрос по тексту",
    "на основе приведенного текста",
]

RETRIEVAL_STOPWORDS = STOPWORDS | {
    "above", "abstract", "answer", "based", "briefly", "context", "determine", "following",
    "give", "only", "output", "paragraph", "paragraphs", "please", "question", "read",
    "records", "text", "which",
    "ответ", "вопрос", "текст", "только", "прочитайте", "основе",
}


def analyze_task(text: str, segments: list[tuple[str, str]], evidence_limit: int = 6) -> TaskProfile:
    normalized = text.lower()
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in EXHAUSTIVE_PATTERNS):
        return TaskProfile(kind="exhaustive")
    if len(segments) < 6:
        return TaskProfile(kind="operational")

    query = extract_query(text, segments)
    if not query:
        return TaskProfile(kind="operational")
    query_terms = content_terms(query)
    if not query_terms:
        return TaskProfile(kind="operational")
    normalized_query = " ".join(query.lower().split())
    query_refs = tuple(ref for ref, segment in segments if segment_belongs_to_query(segment, normalized_query))
    evidence, confidence = retrieve_evidence(segments, query_terms, set(query_refs), evidence_limit)
    explicit_retrieval = any(marker in normalized for marker in RETRIEVAL_MARKERS)
    if not explicit_retrieval and confidence < 0.12:
        return TaskProfile(kind="operational")
    return TaskProfile(
        kind="retrieval",
        query_refs=query_refs,
        evidence_refs=tuple(evidence),
        retrieval_confidence=confidence,
    )


def extract_query(text: str, segments: list[tuple[str, str]]) -> str:
    patterns = [
        r"Question:\s*(.+?)(?:\s*Answer:|$)",
        r"The following is an abstract\.\s*(.+?)(?:\s*Please enter|\s*The answer is:|$)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
    questions = [segment for _ref, segment in segments if "?" in segment]
    if questions:
        return " ".join(questions[-3:])
    for _ref, segment in segments[:4]:
        if re.search(r"\b(answer|respond)\b", segment, re.IGNORECASE):
            return segment
    return ""


def segment_belongs_to_query(segment: str, normalized_query: str) -> bool:
    segment_norm = " ".join(segment.lower().split())
    return len(segment_norm) >= 8 and segment_norm in normalized_query


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\wё-]+", text.lower(), flags=re.UNICODE)
        if len(token) >= 3 and token not in RETRIEVAL_STOPWORDS and not token.isdigit()
    }


def retrieve_evidence(
    segments: list[tuple[str, str]],
    query_terms: set[str],
    excluded_refs: set[str],
    limit: int,
) -> tuple[list[str], float]:
    document_frequency: Counter[str] = Counter()
    terms_by_ref: dict[str, set[str]] = {}
    for ref, segment in segments:
        terms = content_terms(segment)
        terms_by_ref[ref] = terms
        document_frequency.update(terms)

    total = max(len(segments), 1)
    scored = []
    for index, (ref, _segment) in enumerate(segments):
        if ref in excluded_refs or index < 2 or index >= len(segments) - 2:
            continue
        overlap = query_terms & terms_by_ref[ref]
        if not overlap:
            continue
        score = sum(math.log((total + 1) / (document_frequency[term] + 1)) + 1 for term in overlap)
        score *= 1 + len(overlap) / max(len(query_terms), 1)
        scored.append((score, ref, len(overlap) / max(len(query_terms), 1)))
    scored.sort(key=lambda item: (-item[0], int(item[1][1:])))
    selected = [ref for _score, ref, _coverage in scored[:limit]]
    confidence = max((coverage for _score, _ref, coverage in scored), default=0.0)
    return selected, round(confidence, 4)


def is_critical_source(text: str) -> bool:
    return bool(
        re.search(r"\b(no|not|never|must|should|if|unless|не|нельзя|долж\w*|если|когда)\b", text, re.IGNORECASE)
        or re.search(r"\d|PII_[A-Z_]+_\d+|[`\"']", text)
    )


def event_signature(event: dict[str, Any]) -> str:
    return "|".join(
        [
            str(event.get("predicate", "")),
            str(event.get("polarity", "positive")),
            str(event.get("modality", "none")),
            str(event.get("condition", "")),
        ]
    )


def constraint_signature(item: dict[str, Any]) -> str:
    return f"{item.get('type', '')}|{item.get('value', '')}"


def entity_signature(item: dict[str, Any]) -> str:
    return f"{item.get('type', '')}|{item.get('value', '')}"


def overlap_recall(expected: set[str], observed: set[str]) -> float:
    if not expected:
        return 1.0
    return round(len(expected & observed) / len(expected), 4)


def decompile_v1(contract: dict[str, Any], target_lang: str | None, include_anchors: bool) -> str:
    lang = target_lang or str(contract.get("target_lang") or contract.get("source_lang") or "en")
    segments: dict[int, list[str]] = {}
    concepts = contract.get("concepts", [])
    for item in concepts:
        if not isinstance(item, dict):
            continue
        segment = int(item.get("segment", 0) or 0)
        surface = item.get("surface", {})
        term = str(surface.get(lang) or surface.get("en") or surface.get("ru") or item.get("id", "")) if isinstance(surface, dict) else ""
        if term and term not in segments.setdefault(segment, []):
            segments[segment].append(term)
    text = ". ".join(", ".join(terms[:5]) for _segment, terms in sorted(segments.items()) if terms)
    placeholders = [span.get("placeholder", "") for span in contract.get("protected_spans", []) if isinstance(span, dict)]
    if placeholders:
        suffix = "Protected placeholders: " + ", ".join(placeholders)
        text = f"{text}. {suffix}" if text else suffix
    if include_anchors:
        anchors = " ".join(item["id"] for item in concepts if isinstance(item, dict) and item.get("id"))
        if anchors:
            text = f"{text}\nSIR anchors: {anchors}" if text else f"SIR anchors: {anchors}"
    return text


def packet_to_v1_contract(packet: SIRV1Packet) -> dict[str, Any]:
    data = asdict(packet)
    return {
        "version": data["version"],
        "packet_id": data["packet_id"],
        "source_lang": data["source_lang"],
        "target_lang": data["target_lang"],
        "intent": data["intent"],
        "text": {"scrubbed": data["scrubbed_text"]},
        "concepts": data["concepts"],
        "relations": data["relations"],
        "constraints": data["constraints"],
        "protected_spans": [{"placeholder": span["placeholder"], "kind": span["kind"]} for span in data["protected_spans"]],
        "uncertainty": data["uncertainty"],
        "stats": data["stats"],
    }


packet_to_contract = packet_to_v1_contract


def load_contextir(records: Path | None = None, lexical: bool = False, privacy: str = "regex") -> ContextIR:
    scrubber = PresidioPrivacyScrubber() if privacy == "presidio" else PrivacyScrubber()
    if lexical and records is None:
        default_records = Path(__file__).resolve().parents[1] / "data" / "concepts" / "concept_records.jsonl"
        if not default_records.exists():
            raise RuntimeError(
                "lexical enrichment data is not bundled in the package; "
                "pass records=Path(...) or run from the source checkout"
            )
        records = default_records
    return ContextIR(load_runtime(records) if lexical else None, privacy=scrubber)


def load_kernel(records: Path | None = None) -> ContextIR:
    """Compatibility loader for research scripts that require lexical concepts."""

    return load_contextir(records, lexical=True)


def read_contract(path: str) -> dict[str, Any]:
    if path == "-":
        import sys

        return json.loads(sys.stdin.read())
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="contextir",
        description="Compile context, invoke a model, or render ContextIR for an LLM.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    compile_p = sub.add_parser("compile", help="Compile text into ContextIR v2.")
    compile_p.add_argument("--text", required=True)
    compile_p.add_argument("--source-lang", default="ru")
    compile_p.add_argument("--target-lang", default="ru")
    compile_p.add_argument("--packet-id", default="context")
    compile_p.add_argument("--mode", choices=["auto", "raw", "hybrid", "semantic"], default="auto")
    compile_p.add_argument("--lexical", action="store_true", help="Load optional WordNet concept enrichment.")
    compile_p.add_argument("--privacy", choices=["regex", "presidio"], default="regex")
    compile_p.add_argument("--out", default="-")
    render_p = sub.add_parser("render", help="Render a compact model prompt from a contract.")
    render_p.add_argument("--contract", required=True, help="Path to JSON contract, or '-' for stdin.")
    render_p.add_argument("--out", default="-")
    run_p = sub.add_parser("run", help="Compile context and invoke a model endpoint.")
    run_p.add_argument("--text", required=True, help="Input text, or '-' for stdin.")
    run_p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    run_p.add_argument("--model", required=True)
    run_p.add_argument("--base-url", default="")
    run_p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_p.add_argument("--source-lang", default="en")
    run_p.add_argument("--target-lang", default="en")
    run_p.add_argument("--risk", choices=["low", "standard", "high"], default="standard")
    run_p.add_argument("--task", choices=["reasoning", "transform"], default="reasoning")
    run_p.add_argument("--timeout", type=float, default=180)
    run_p.add_argument("--context-length", type=int, default=32768)
    run_p.add_argument("--max-output-tokens", type=int, default=256)
    run_p.add_argument("--json", action="store_true", help="Emit answer and payload-free trace as JSON.")
    args = parser.parse_args()

    if args.cmd == "run":
        import os

        from contextir.clients import OllamaClient, OpenAICompatibleClient
        from contextir.pipeline import ContextPipeline

        text = sys.stdin.read() if args.text == "-" else args.text
        if args.backend == "ollama":
            client = OllamaClient(
                args.model,
                base_url=args.base_url or "http://127.0.0.1:11434",
                timeout=args.timeout,
                context_length=args.context_length,
                max_output_tokens=args.max_output_tokens,
            )
        else:
            client = OpenAICompatibleClient(
                args.model,
                base_url=args.base_url or "http://127.0.0.1:1234/v1",
                api_key=os.environ.get(args.api_key_env, ""),
                timeout=args.timeout,
                max_output_tokens=args.max_output_tokens,
            )
        result = ContextPipeline(invoke=client).run(
            text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            risk=args.risk,
            task=args.task,
        )
        if args.json:
            print(json.dumps({"answer": result.answer, "trace": result.public_trace()}, ensure_ascii=False))
        else:
            print(result.answer)
        if not result.accepted:
            raise SystemExit(2)
        return

    gateway = load_contextir(lexical=getattr(args, "lexical", False), privacy=getattr(args, "privacy", "regex"))
    if args.cmd == "compile":
        contract = gateway.compile(args.text, args.source_lang, args.target_lang, args.packet_id, args.mode)
        payload = json.dumps(contract, ensure_ascii=False, indent=2) + "\n"
    else:
        payload = gateway.render_prompt(read_contract(args.contract)) + "\n"
    if args.out == "-":
        print(payload, end="")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
