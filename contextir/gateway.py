from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from contextir.sir_runtime import PresidioPrivacyScrubber, PrivacyScrubber, SIRRuntime, SIRV1Packet, detect_constraints, guess_intent, load_runtime


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
        intent = asdict(guess_intent(scrubbed.scrubbed_text))
        events = extract_events(segments, source_lang)
        entities = extract_entities(segments, scrubbed.protected_spans)
        constraints = detect_constraints(scrubbed.scrubbed_text)
        if scrubbed.protected_spans:
            constraints.insert(0, {"type": "privacy", "value": "keep_placeholders_private"})

        confidence = semantic_confidence(events, segments)
        selected_mode = choose_mode(mode, len(text), confidence, segments, self.raw_threshold)
        included_refs = select_source_refs(selected_mode, segments)
        concepts = self._compact_concepts(scrubbed.scrubbed_text, source_lang, packet_id, selected_mode)
        public_spans = [{"placeholder": span.placeholder, "kind": span.kind} for span in scrubbed.protected_spans]

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

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "we", "with", "you",
    "а", "без", "бы", "в", "для", "до", "и", "из", "или", "как", "на", "но", "о", "по", "с", "то", "у", "это", "я", "мы", "он", "она", "они",
    "не", "not", "if", "если", "must", "should", "должен", "должна", "нужно", "надо",
}


def split_source(text: str) -> list[tuple[str, str]]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\n]+", text) if part.strip()]
    return [(f"s{index}", part) for index, part in enumerate(parts, 1)]


def extract_events(segments: list[tuple[str, str]], source_lang: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    by_signature: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source_ref, text in segments:
        norm = text.lower()
        predicate = next((name for name, pattern in ACTION_PATTERNS if re.search(pattern, norm, re.IGNORECASE)), "state")
        polarity = "negative" if re.search(r"\b(no|not|never|не|нельзя|никогда)\b", norm) else "positive"
        modality = "requirement" if re.search(r"\b(must|should|need|долж\w*|нужно|надо)\b", norm) else "none"
        if re.search(r"\b(must not|do not|don't|нельзя|не должен\w*)\b", norm):
            modality = "prohibition"
        condition = "if" if re.search(r"\b(if|unless|when|если|когда)\b", norm) else ""
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
    tokens = re.findall(r"PII_[A-Z_]+_\d+|[\wё-]+", text.lower(), flags=re.IGNORECASE)
    out = []
    for token in tokens:
        if token in STOPWORDS or len(token) < 3 or token.startswith(predicate):
            continue
        if token not in out:
            out.append(token)
        if len(out) == limit:
            break
    return out


def extract_entities(segments: list[tuple[str, str]], protected_spans: list[Any]) -> list[dict[str, str]]:
    entities = []
    for span in protected_spans:
        entities.append({"id": f"p{len(entities) + 1}", "type": span.kind, "value": span.placeholder})
    seen = {item["value"] for item in entities}
    for source_ref, text in segments:
        for value in re.findall(r"(?<!\w)\d+(?:[.,:]\d+)*(?!\w)", text):
            if value not in seen:
                entities.append({"id": f"n{len(entities) + 1}", "type": "number", "value": value, "source_ref": source_ref})
                seen.add(value)
    return entities


def semantic_confidence(events: list[dict[str, Any]], segments: list[tuple[str, str]]) -> float:
    if not segments:
        return 0.0
    weighted = sum(float(event["confidence"]) * int(event.get("count", 1)) for event in events)
    return round(weighted / len(segments), 4)


def choose_mode(requested: Mode, chars: int, confidence: float, segments: list[tuple[str, str]], raw_threshold: int) -> str:
    if requested != "auto":
        return requested
    if chars <= raw_threshold:
        return "raw"
    critical = sum(1 for _ref, text in segments if is_critical_source(text))
    if confidence < 0.65 or critical:
        return "hybrid"
    return "semantic"


def select_source_refs(mode: str, segments: list[tuple[str, str]]) -> list[str]:
    if mode == "raw":
        return [ref for ref, _text in segments]
    if mode == "semantic":
        return []
    selected = [ref for ref, text in segments if is_critical_source(text)]
    if not selected and segments:
        selected.append(segments[0][0])
    return selected[:8]


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
    parser = argparse.ArgumentParser(prog="contextir", description="Compile text into compact ContextIR or render it for an LLM.")
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
    args = parser.parse_args()

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
