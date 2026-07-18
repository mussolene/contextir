from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from contextir.sir_roundtrip import ConceptHit, SIRRoundtrip, load_roundtrip, split_segments
from contextir.sir_sources import PROJECT_ROOT, normalize


@dataclass
class ProtectedSpan:
    placeholder: str
    kind: str
    local_ref: str
    surface_hash: str


@dataclass
class PrivacyScrubResult:
    scrubbed_text: str
    protected_spans: list[ProtectedSpan]
    vault: dict[str, str]


@dataclass
class IntentGuess:
    label: str
    confidence: float
    signals: list[str]


@dataclass
class SIRV1Packet:
    version: str
    packet_id: str
    source_lang: str
    target_lang: str
    intent: IntentGuess
    scrubbed_text: str
    concepts: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    protected_spans: list[ProtectedSpan]
    uncertainty: list[dict[str, Any]]
    stats: dict[str, Any]


@dataclass
class SIRAnswerCheck:
    preserved_concepts: float
    request_concepts: int
    answer_concepts: int
    shared_concepts: int
    lost_concepts: list[str]
    new_concepts: list[str]
    lost_constraints: list[dict[str, Any]]
    needs_revision: bool


@dataclass
class SIRRuntimeResult:
    request_packet: SIRV1Packet
    model_prompt: str
    raw_answer: str
    answer_packet: SIRV1Packet
    answer_check: SIRAnswerCheck
    final_text: str
    backend: str
    latency_ms: float


class PrivacyScrubber:
    patterns = [
        ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
        ("phone", re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")),
        ("card", re.compile(r"(?<!\w)(?:\d[ -]*?){13,19}(?!\w)")),
        ("api_key", re.compile(r"\b(?:sk|pk|api|key|token)[-_]?[A-Za-z0-9]{16,}\b")),
    ]

    def scrub(self, text: str, language: str = "en") -> PrivacyScrubResult:
        protected: list[ProtectedSpan] = []
        vault: dict[str, str] = {}
        scrubbed = text
        counters: dict[str, int] = {}
        placeholders: dict[tuple[str, str], str] = {}
        for kind, pattern in self.patterns:
            while True:
                match = pattern.search(scrubbed)
                if not match:
                    break
                surface = match.group(0)
                key = (kind, surface)
                placeholder = placeholders.get(key)
                if placeholder is None:
                    counters[kind] = counters.get(kind, 0) + 1
                    placeholder = f"PII_{kind.upper()}_{counters[kind]}"
                    placeholders[key] = placeholder
                    protected.append(
                        ProtectedSpan(
                            placeholder=placeholder,
                            kind=kind,
                            local_ref=f"local:{stable_hash(kind + ':' + surface)[:16]}",
                            surface_hash=stable_hash(surface),
                        )
                    )
                    vault[placeholder] = surface
                scrubbed = scrubbed[: match.start()] + placeholder + scrubbed[match.end() :]
        return PrivacyScrubResult(scrubbed_text=scrubbed, protected_spans=protected, vault=vault)


class PresidioPrivacyScrubber:
    """Optional Presidio-backed detector with the same local-vault contract."""

    def __init__(self, analyzer: Any | None = None, score_threshold: float = 0.5):
        if analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("install ContextIR with the 'privacy' extra to use Presidio") from exc
            analyzer = AnalyzerEngine()
        self.analyzer = analyzer
        self.score_threshold = score_threshold

    def scrub(self, text: str, language: str = "en") -> PrivacyScrubResult:
        findings = self.analyzer.analyze(text=text, language=language, score_threshold=self.score_threshold)
        accepted: list[Any] = []
        occupied: list[tuple[int, int]] = []
        for item in sorted(findings, key=lambda value: (-float(value.score), value.start, value.end)):
            if any(item.start < end and item.end > start for start, end in occupied):
                continue
            accepted.append(item)
            occupied.append((item.start, item.end))
        accepted.sort(key=lambda value: value.start)

        counters: dict[str, int] = {}
        placeholders: dict[tuple[str, str], str] = {}
        protected: list[ProtectedSpan] = []
        vault: dict[str, str] = {}
        chunks: list[str] = []
        cursor = 0
        for item in accepted:
            kind = str(item.entity_type).lower()
            surface = text[item.start : item.end]
            key = (kind, surface)
            placeholder = placeholders.get(key)
            if placeholder is None:
                counters[kind] = counters.get(kind, 0) + 1
                placeholder = f"PII_{kind.upper()}_{counters[kind]}"
                placeholders[key] = placeholder
                protected.append(
                    ProtectedSpan(
                        placeholder=placeholder,
                        kind=kind,
                        local_ref=f"local:{stable_hash(kind + ':' + surface)[:16]}",
                        surface_hash=stable_hash(surface),
                    )
                )
                vault[placeholder] = surface
            chunks.extend([text[cursor : item.start], placeholder])
            cursor = item.end
        chunks.append(text[cursor:])
        return PrivacyScrubResult(scrubbed_text="".join(chunks), protected_spans=protected, vault=vault)


class SIRRuntime:
    def __init__(self, compiler: SIRRoundtrip):
        self.compiler = compiler
        self.scrubber = PrivacyScrubber()

    def compile_request(self, text: str, source_lang: str, target_lang: str, packet_id: str = "request") -> tuple[SIRV1Packet, dict[str, str]]:
        scrubbed = self.scrubber.scrub(text)
        base = self.compiler.compile_text(scrubbed.scrubbed_text, source_lang=source_lang, text_id=packet_id)
        concepts = flatten_concepts(base.segments)
        concepts = merge_anchor_concepts(concepts, extract_concept_ids(scrubbed.scrubbed_text), self.compiler.by_id)
        constraints = [{"type": "privacy", "value": "do_not_expose_protected_spans"}] if scrubbed.protected_spans else []
        constraints.extend(detect_constraints(scrubbed.scrubbed_text))
        packet = SIRV1Packet(
            version="sir.v1",
            packet_id=packet_id,
            source_lang=source_lang,
            target_lang=target_lang,
            intent=guess_intent(scrubbed.scrubbed_text),
            scrubbed_text=scrubbed.scrubbed_text,
            concepts=concepts,
            relations=infer_relations(scrubbed.scrubbed_text, concepts),
            constraints=constraints,
            protected_spans=scrubbed.protected_spans,
            uncertainty=infer_uncertainty(base.unknown_tokens, concepts),
            stats={
                "chars": base.chars,
                "tokens": base.tokens,
                "segments": len(base.segments),
                "concepts": len({item["id"] for item in concepts}),
                "unknown_token_rate": round(len(base.unknown_tokens) / max(base.tokens, 1), 4),
            },
        )
        return packet, scrubbed.vault

    def build_model_prompt(self, packet: SIRV1Packet) -> str:
        public_packet = packet_to_public_dict(packet)
        return (
            "You are reasoning over a SIR semantic packet. "
            "Do not reveal protected placeholders. "
            "Answer using the target language requested in the packet. "
            "Preserve constraints and explain only what is supported by the packet.\n\n"
            f"SIR_PACKET:\n{json.dumps(public_packet, ensure_ascii=False, indent=2)}"
        )

    def run(
        self,
        text: str,
        source_lang: str = "ru",
        target_lang: str = "ru",
        backend: str = "deterministic",
        model: str = "",
        timeout: int = 30,
    ) -> SIRRuntimeResult:
        started = time.perf_counter()
        request_packet, vault = self.compile_request(text, source_lang, target_lang)
        prompt = self.build_model_prompt(request_packet)
        raw_answer = run_backend(prompt, request_packet, backend=backend, model=model, timeout=timeout)
        answer_packet, _answer_vault = self.compile_request(raw_answer, target_lang, target_lang, packet_id="answer")
        check = check_answer(request_packet, answer_packet)
        public_answer = strip_sir_anchors(raw_answer)
        final_text = restore_placeholders(public_answer, vault) if not check.needs_revision else build_revision_notice(public_answer, check, target_lang)
        return SIRRuntimeResult(
            request_packet=request_packet,
            model_prompt=prompt,
            raw_answer=raw_answer,
            answer_packet=answer_packet,
            answer_check=check,
            final_text=final_text,
            backend=backend,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


def flatten_concepts(segments: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, segment in enumerate(segments):
        for hit in segment.concepts:
            if hit.concept_id in seen:
                continue
            seen.add(hit.concept_id)
            out.append(
                {
                    "id": hit.concept_id,
                    "role": guess_role(hit),
                    "score": hit.score,
                    "source": hit.source,
                    "surface": {"en": hit.en, "ru": hit.ru},
                    "definition_en": hit.definition_en,
                    "segment": index,
                }
            )
    return out


def extract_concept_ids(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b(?:\d{8}-[nvar]|sir:[a-z0-9_:-]+)\b", text)))


def merge_anchor_concepts(concepts: list[dict[str, Any]], anchors: list[str], by_id: dict[str, Any]) -> list[dict[str, Any]]:
    seen = {item["id"] for item in concepts}
    merged = list(concepts)
    for concept_id in anchors:
        if concept_id in seen or concept_id not in by_id:
            continue
        record = by_id[concept_id]
        merged.append(
            {
                "id": concept_id,
                "role": "anchor",
                "score": 1.0,
                "source": "sir_anchor",
                "surface": {"en": first(record.en), "ru": first(record.ru)},
                "definition_en": record.definition_en,
                "segment": -1,
            }
        )
        seen.add(concept_id)
    return merged


def guess_role(hit: ConceptHit) -> str:
    text = normalize(" ".join([hit.en, hit.ru, hit.definition_en]))
    if any(word in text.split() for word in ["do", "act", "action", "work", "run", "делать", "работать"]):
        return "action"
    if any(word in text.split() for word in ["privacy", "secret", "private", "личный", "секрет"]):
        return "constraint"
    return "concept"


def guess_intent(text: str) -> IntentGuess:
    norm = normalize(text)
    signals: list[str] = []
    label = "ask"
    if "?" in text or any(word in norm.split() for word in ["как", "что", "why", "how", "what"]):
        signals.append("question")
    if any(word in norm.split() for word in ["сделай", "запусти", "проверь", "добей", "run", "build", "check"]):
        label = "command"
        signals.append("imperative")
    if any(word in norm.split() for word in ["переведи", "translate", "язык", "language"]):
        label = "translate"
        signals.append("translation")
    if any(word in norm.split() for word in ["план", "архитектура", "architecture", "design"]):
        label = "plan" if label == "ask" else label
        signals.append("planning")
    confidence = 0.55 + min(len(signals) * 0.15, 0.4)
    return IntentGuess(label=label, confidence=round(confidence, 2), signals=signals or ["default"])


def detect_constraints(text: str) -> list[dict[str, Any]]:
    norm = normalize(text)
    constraints: list[dict[str, Any]] = []
    if any(word in norm.split() for word in ["локально", "local", "private", "приватно", "pii"]):
        constraints.append({"type": "execution", "value": "prefer_local_or_private"})
    if any(word in norm.split() for word in ["быстро", "маленький", "small", "cheap", "утюге"]):
        constraints.append({"type": "runtime", "value": "small_fast_core"})
    return constraints


def infer_relations(text: str, concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    if concepts:
        relations.append({"source": "user", "relation": "mentions", "target": concepts[0]["id"]})
    if len(concepts) > 1:
        relations.append({"source": concepts[0]["id"], "relation": "context_with", "target": concepts[1]["id"]})
    if any(marker in normalize(text).split() for marker in ["чтобы", "because", "why", "поэтому"]):
        relations.append({"source": "intent", "relation": "has_purpose_or_cause", "target": "request"})
    return relations


def infer_uncertainty(unknown_tokens: list[str], concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if unknown_tokens:
        out.append({"type": "unknown_tokens", "sample": unknown_tokens[:12], "count": len(unknown_tokens)})
    low = [item["id"] for item in concepts if float(item["score"]) < 0.8]
    if low:
        out.append({"type": "low_confidence_concepts", "sample": low[:12], "count": len(low)})
    return out


def run_backend(prompt: str, packet: SIRV1Packet, backend: str, model: str, timeout: int) -> str:
    if backend == "deterministic":
        return deterministic_answer(packet)
    try:
        if backend == "ollama":
            if not model:
                raise RuntimeError("--model is required for ollama backend")
            return run_command(["ollama", "run", model, prompt], timeout)
        if backend == "agent":
            return run_command(["agent", "--print", "--mode", "ask", "--trust", prompt], timeout)
    except Exception as exc:
        return deterministic_answer(packet) + f"\nBackend fallback: {backend} unavailable ({type(exc).__name__})."
    raise ValueError(f"unknown backend: {backend}")


def deterministic_answer(packet: SIRV1Packet) -> str:
    lang = packet.target_lang
    concept_terms = []
    for item in packet.concepts[:8]:
        surface = item["surface"].get(lang) or item["surface"].get("en") or item["id"]
        if surface and surface not in concept_terms:
            concept_terms.append(surface)
    anchors = " ".join(item["id"] for item in packet.concepts)
    anchor_line = f"\nSIR anchors: {anchors}" if anchors else ""
    if lang == "en":
        prefix = "SIR understood the request as"
        privacy = "Protected data remains local." if packet.protected_spans else "No protected data was detected."
        terms = ", ".join(concept_terms) if concept_terms else "no stable concepts"
        return f"{prefix}: intent={packet.intent.label}; key concepts={terms}. {privacy}{anchor_line}"
    prefix = "SIR понял запрос как"
    privacy = "Защищенные данные остаются локально." if packet.protected_spans else "Защищенные данные не обнаружены."
    terms = ", ".join(concept_terms) if concept_terms else "устойчивые понятия не найдены"
    return f"{prefix}: intent={packet.intent.label}; ключевые понятия={terms}. {privacy}{anchor_line}"


def check_answer(request: SIRV1Packet, answer: SIRV1Packet) -> SIRAnswerCheck:
    request_ids = {item["id"] for item in request.concepts}
    answer_ids = {item["id"] for item in answer.concepts}
    shared = request_ids & answer_ids
    preserved = len(shared) / max(len(request_ids), 1)
    lost = sorted(request_ids - answer_ids)[:20]
    new = sorted(answer_ids - request_ids)[:20]
    lost_constraints = []
    return SIRAnswerCheck(
        preserved_concepts=round(preserved, 4),
        request_concepts=len(request_ids),
        answer_concepts=len(answer_ids),
        shared_concepts=len(shared),
        lost_concepts=lost,
        new_concepts=new,
        lost_constraints=lost_constraints,
        needs_revision=preserved < 0.35 or bool(lost_constraints),
    )


def build_revision_notice(raw_answer: str, check: SIRAnswerCheck, target_lang: str) -> str:
    if target_lang == "en":
        return (
            f"{raw_answer}\n\n"
            f"[SIR reloop: answer needs revision; preserved={check.preserved_concepts}, "
            f"lost_concepts={len(check.lost_concepts)}, lost_constraints={len(check.lost_constraints)}]"
        )
    return (
        f"{raw_answer}\n\n"
        f"[SIR reloop: ответ требует уточнения; preserved={check.preserved_concepts}, "
        f"lost_concepts={len(check.lost_concepts)}, lost_constraints={len(check.lost_constraints)}]"
    )


def restore_placeholders(text: str, vault: dict[str, str]) -> str:
    restored = text
    for placeholder, surface in vault.items():
        restored = restored.replace(placeholder, surface)
    return restored


def strip_sir_anchors(text: str) -> str:
    cleaned = re.sub(r"\n?SIR anchors:[^\n]*(?:\n)?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def packet_to_public_dict(packet: SIRV1Packet) -> dict[str, Any]:
    data = asdict(packet)
    for span in data["protected_spans"]:
        span.pop("local_ref", None)
        span.pop("surface_hash", None)
    return data


def run_command(cmd: list[str], timeout: int) -> str:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"backend timed out after {exc.timeout}s") from exc
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def stable_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def first(values: list[str]) -> str:
    return values[0] if values else ""


def result_to_dict(result: SIRRuntimeResult, include_prompt: bool = False) -> dict[str, Any]:
    data = asdict(result)
    if not include_prompt:
        data.pop("model_prompt", None)
    return data


def load_runtime(records: Path | None = None) -> SIRRuntime:
    return SIRRuntime(load_roundtrip(records or PROJECT_ROOT / "data" / "concepts" / "concept_records.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SIR v1 privacy + semantic reloop prototype.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--source-lang", default="ru")
    parser.add_argument("--target-lang", default="ru")
    parser.add_argument("--backend", choices=["deterministic", "ollama", "agent"], default="deterministic")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runtime = load_runtime()
    result = runtime.run(
        args.text,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        backend=args.backend,
        model=args.model,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(result_to_dict(result, include_prompt=args.include_prompt), ensure_ascii=False, indent=2))
        return
    print(result.final_text)
    print()
    print(json.dumps(asdict(result.answer_check), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
