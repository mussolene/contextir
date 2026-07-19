from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Callable, Literal

from contextir.gateway import ContextKind, ContractCheck, ContextBundle, ContextIR, content_terms


Risk = Literal["low", "standard", "high"]
Task = Literal["reasoning", "transform"]
TokenCounter = Callable[[str], int]
PLACEHOLDER_RE = re.compile(r"\bPII_[A-Z0-9_]+_\d+\b")
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
NO_EVIDENCE = "CTXIR_NONE"


@dataclass(frozen=True)
class PipelinePolicy:
    min_token_savings: float = 0.15
    min_semantic_confidence: float = 0.72
    verification_threshold: float = 0.9
    max_attempts: int = 3
    reject_new_pii: bool = True
    max_prompt_tokens: int | None = None
    max_chunk_calls: int = 16
    chunk_overlap_words: int = 24
    chunk_prompt_ratio: float = 0.75

    def __post_init__(self) -> None:
        for name in ("min_token_savings", "min_semantic_confidence", "verification_threshold"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        if self.max_prompt_tokens is not None and (
            isinstance(self.max_prompt_tokens, bool)
            or not isinstance(self.max_prompt_tokens, int)
            or self.max_prompt_tokens < 1
        ):
            raise ValueError("max_prompt_tokens must be a positive integer")
        if isinstance(self.max_chunk_calls, bool) or not isinstance(self.max_chunk_calls, int):
            raise ValueError("max_chunk_calls must be an integer")
        if self.max_chunk_calls < 2:
            raise ValueError("max_chunk_calls must be at least 2")
        if isinstance(self.chunk_overlap_words, bool) or not isinstance(self.chunk_overlap_words, int):
            raise ValueError("chunk_overlap_words must be an integer")
        if self.chunk_overlap_words < 0:
            raise ValueError("chunk_overlap_words must be non-negative")
        if (
            isinstance(self.chunk_prompt_ratio, bool)
            or not isinstance(self.chunk_prompt_ratio, (int, float))
            or not 0 < self.chunk_prompt_ratio <= 1
        ):
            raise ValueError("chunk_prompt_ratio must be greater than 0 and at most 1")


class ContextWindowExceeded(RuntimeError):
    """Raised before invocation when a safe prompt cannot fit the model budget."""

    def __init__(self, prompt_tokens: int, prompt_budget: int, mode: str) -> None:
        self.prompt_tokens = prompt_tokens
        self.prompt_budget = prompt_budget
        self.mode = mode
        super().__init__(
            f"{mode} prompt requires {prompt_tokens} tokens but the model budget is {prompt_budget}"
        )


class ChunkLimitExceeded(RuntimeError):
    """Raised before chunk invocation when bounded retrieval needs too many calls."""

    def __init__(self, required_calls: int, max_calls: int) -> None:
        self.required_calls = required_calls
        self.max_calls = max_calls
        super().__init__(f"chunked retrieval requires {required_calls} calls but the limit is {max_calls}")


@dataclass
class PreparedContext:
    bundle: ContextBundle
    prompt: str
    risk: Risk
    source_tokens: int
    prompt_tokens: int
    token_savings: float
    decision: str
    prompt_budget: int | None = None

    @property
    def mode(self) -> str:
        return str(self.bundle.contract["mode"])

    @property
    def fits_prompt_budget(self) -> bool:
        return self.prompt_budget is None or self.prompt_tokens <= self.prompt_budget


@dataclass
class ResponseVerification:
    accepted: bool
    reasons: list[str]
    unknown_placeholders: list[str]
    missing_placeholders: list[str]
    new_pii_kinds: list[str]
    contract_check: ContractCheck | None = None


@dataclass
class PipelineAttempt:
    mode: str
    prompt_tokens: int
    response_chars: int
    verification: ResponseVerification
    stage: str = "direct"
    chunk_index: int | None = None


@dataclass
class PipelineResult:
    answer: str
    accepted: bool
    selected_mode: str
    attempts: list[PipelineAttempt]
    prepared: PreparedContext

    def public_trace(self) -> dict[str, object]:
        """Return metrics safe for normal logs: no prompt, answer, source, or vault."""

        return {
            "accepted": self.accepted,
            "selected_mode": self.selected_mode,
            "decision": self.prepared.decision,
            "risk": self.prepared.risk,
            "source_tokens": self.prepared.source_tokens,
            "prompt_budget": self.prepared.prompt_budget,
            "attempts": [
                {
                    "mode": item.mode,
                    "stage": item.stage,
                    "chunk_index": item.chunk_index,
                    "prompt_tokens": item.prompt_tokens,
                    "response_chars": item.response_chars,
                    "verification": {
                        "accepted": item.verification.accepted,
                        "reasons": item.verification.reasons,
                        "unknown_placeholders": len(item.verification.unknown_placeholders),
                        "missing_placeholders": len(item.verification.missing_placeholders),
                        "new_pii_kinds": item.verification.new_pii_kinds,
                    },
                }
                for item in self.attempts
            ],
        }


Verifier = Callable[[PreparedContext, str], ResponseVerification]
Invoker = Callable[[str], str]


class ContextPipeline:
    """Policy-driven model boundary built on top of the ContextIR compiler."""

    def __init__(
        self,
        gateway: ContextIR | None = None,
        policy: PipelinePolicy | None = None,
        token_counter: TokenCounter | None = None,
        invoke: Invoker | None = None,
    ) -> None:
        self.gateway = gateway or ContextIR()
        self.policy = policy or PipelinePolicy()
        self.token_counter = token_counter or approximate_token_count
        self.invoke = invoke

    def prepare(
        self,
        text: str,
        source_lang: str = "ru",
        target_lang: str = "ru",
        risk: Risk = "standard",
        packet_id: str = "context",
        max_prompt_tokens: int | None = None,
        context_kind: ContextKind = "auto",
        query: str = "",
    ) -> PreparedContext:
        if risk not in {"low", "standard", "high"}:
            raise ValueError(f"unsupported risk: {risk}")

        prompt_budget = self._resolve_prompt_budget(self.invoke, max_prompt_tokens)

        if len(text) <= self.gateway.raw_threshold:
            raw = self._compile_mode(
                text, source_lang, target_lang, risk, packet_id, "raw", "short_input", context_kind, query
            )
            raw.decision = "short_input"
            return self._enforce_prompt_budget(raw, prompt_budget)

        requested_mode = {"low": "semantic", "standard": "auto", "high": "hybrid"}[risk]
        candidate = self._compile_mode(
            text,
            source_lang,
            target_lang,
            risk,
            packet_id,
            requested_mode,
            f"risk_{risk}",
            context_kind,
            query,
        )
        confidence = float(candidate.bundle.contract["uncertainty"]["semantic_confidence"])
        if candidate.mode == "semantic" and confidence < self.policy.min_semantic_confidence:
            candidate = self._compile_mode(
                text,
                source_lang,
                target_lang,
                risk,
                packet_id,
                "hybrid",
                "semantic_confidence_too_low",
                context_kind,
                query,
            )

        if candidate.mode == "raw":
            candidate.decision = "compiler_selected_raw"
            return self._enforce_prompt_budget(candidate, prompt_budget)
        candidate = self._enforce_prompt_budget(candidate, prompt_budget)
        if candidate.token_savings < self.policy.min_token_savings:
            raw = self._compile_mode(
                text, source_lang, target_lang, risk, packet_id, "raw", "raw_baseline", context_kind, query
            )
            raw.decision = "insufficient_token_savings"
            return self._enforce_prompt_budget(raw, prompt_budget)
        return candidate

    def run(
        self,
        text: str,
        invoke: Invoker | None = None,
        source_lang: str = "ru",
        target_lang: str = "ru",
        risk: Risk = "standard",
        task: Task = "reasoning",
        allowed_restore: set[str] | None = None,
        verifier: Verifier | None = None,
        packet_id: str = "context",
        chunked_retrieval: bool = False,
        context_kind: ContextKind = "auto",
        query: str = "",
    ) -> PipelineResult:
        if task not in {"reasoning", "transform"}:
            raise ValueError(f"unsupported task: {task}")
        model_invoke = invoke or self.invoke
        if model_invoke is None:
            raise ValueError("invoke is required; pass it to ContextPipeline() or run()")

        prompt_budget = self._resolve_prompt_budget(model_invoke, None)
        try:
            initial = self.prepare(
                text,
                source_lang,
                target_lang,
                risk,
                packet_id,
                max_prompt_tokens=prompt_budget,
                context_kind=context_kind,
                query=query,
            )
        except ContextWindowExceeded as exc:
            if not chunked_retrieval or task != "reasoning":
                raise
            if source_lang != target_lang:
                raise ValueError("chunked retrieval requires source_lang and target_lang to match") from exc
            return self._run_chunked_retrieval(
                text,
                model_invoke,
                source_lang,
                target_lang,
                risk,
                allowed_restore or set(),
                verifier,
                packet_id,
                prompt_budget,
                exc,
                context_kind,
                query,
            )
        modes = fallback_modes(initial.mode)[: self.policy.max_attempts]
        attempts: list[PipelineAttempt] = []
        final_prepared = initial
        final_answer = ""
        accepted = False

        for index, mode in enumerate(modes):
            if index == 0:
                prepared = initial
            else:
                prepared = self._compile_mode(
                    text,
                    source_lang,
                    target_lang,
                    risk,
                    packet_id,
                    mode,
                    "verification_fallback",
                    context_kind,
                    query,
                )
                try:
                    prepared = self._enforce_prompt_budget(prepared, prompt_budget)
                except ContextWindowExceeded:
                    attempts[-1].verification.reasons.append("fallback_exceeds_prompt_budget")
                    break
            answer = model_invoke(prepared.prompt)
            verification = (
                verifier(prepared, answer)
                if verifier
                else self.verify_response(prepared, answer, preserve_input=task == "transform")
            )
            attempts.append(
                PipelineAttempt(
                    mode=prepared.mode,
                    prompt_tokens=prepared.prompt_tokens,
                    response_chars=len(answer),
                    verification=verification,
                )
            )
            final_prepared = prepared
            final_answer = clean_model_output(answer)
            if verification.accepted:
                accepted = True
                break

        if accepted:
            final_answer = self.gateway.restore(final_answer, final_prepared.bundle, allowed=allowed_restore or set())
        return PipelineResult(
            answer=final_answer,
            accepted=accepted,
            selected_mode=final_prepared.mode,
            attempts=attempts,
            prepared=final_prepared,
        )

    def _run_chunked_retrieval(
        self,
        text: str,
        model_invoke: Invoker,
        source_lang: str,
        target_lang: str,
        risk: Risk,
        allowed_restore: set[str],
        verifier: Verifier | None,
        packet_id: str,
        prompt_budget: int | None,
        original_error: ContextWindowExceeded,
        context_kind: ContextKind,
        query: str,
    ) -> PipelineResult:
        if prompt_budget is None:
            raise ValueError("chunked retrieval requires a prompt token budget")
        base = self._compile_mode(
            text,
            source_lang,
            target_lang,
            risk,
            packet_id,
            "auto",
            "chunked_retrieval",
            context_kind,
            query,
        )
        if base.mode != "hybrid" or not base.bundle.evidence_source_groups:
            raise original_error

        chunk_budget = max(1, int(prompt_budget * self.policy.chunk_prompt_ratio))
        maps = self._build_retrieval_maps(base, chunk_budget)
        worst_case_calls = len(maps) + 1
        if worst_case_calls > self.policy.max_chunk_calls:
            raise ChunkLimitExceeded(worst_case_calls, self.policy.max_chunk_calls)

        attempts: list[PipelineAttempt] = []
        candidates: list[tuple[str, ResponseVerification]] = []
        seen_candidates: set[str] = set()
        final_prepared = maps[0]
        for index, prepared in enumerate(maps, 1):
            answer = model_invoke(prepared.prompt)
            verification = (
                verifier(prepared, answer)
                if verifier
                else self.verify_response(prepared, answer, preserve_input=False)
            )
            cleaned_answer = clean_model_output(answer)
            no_evidence = self._is_no_evidence(cleaned_answer)
            if not no_evidence and "CTXIR_RETRIEVAL_" in cleaned_answer.upper():
                verification.accepted = False
                verification.reasons.append("protocol_output")
            attempts.append(
                PipelineAttempt(
                    mode=prepared.mode,
                    prompt_tokens=prepared.prompt_tokens,
                    response_chars=len(answer),
                    verification=verification,
                    stage="map",
                    chunk_index=index,
                )
            )
            final_prepared = prepared
            if not verification.accepted:
                return PipelineResult("", False, "hybrid", attempts, final_prepared)
            if not no_evidence and not self._is_grounded(cleaned_answer, prepared.bundle):
                verification.accepted = False
                verification.reasons.append("unsupported_candidate")
                continue
            normalized_answer = " ".join(cleaned_answer.lower().split())
            if not no_evidence and normalized_answer not in seen_candidates:
                candidates.append((cleaned_answer, verification))
                seen_candidates.add(normalized_answer)

        if not candidates:
            return PipelineResult("", False, "hybrid", attempts, final_prepared)

        final_answer, final_verification = candidates[0]
        if len(candidates) > 1:
            reduce_prompt = self._render_reduce_prompt(base.bundle, [value for value, _check in candidates])
            reduce_prepared = self._prepared_for_prompt(
                base,
                base.bundle,
                reduce_prompt,
                "chunked_retrieval_reduce",
                prompt_budget,
            )
            if not reduce_prepared.fits_prompt_budget:
                raise ContextWindowExceeded(
                    reduce_prepared.prompt_tokens,
                    prompt_budget,
                    "chunked_retrieval_reduce",
                )
            raw_final_answer = model_invoke(reduce_prompt)
            final_verification = (
                verifier(reduce_prepared, raw_final_answer)
                if verifier
                else self.verify_response(reduce_prepared, raw_final_answer, preserve_input=False)
            )
            final_answer = clean_model_output(raw_final_answer)
            if self._is_no_evidence(final_answer) or "CTXIR_RETRIEVAL_" in final_answer.upper():
                final_verification.accepted = False
                final_verification.reasons.append("no_supported_answer")
            elif not self._is_grounded(final_answer, base.bundle):
                final_verification.accepted = False
                final_verification.reasons.append("unsupported_candidate")
            attempts.append(
                PipelineAttempt(
                    mode=reduce_prepared.mode,
                    prompt_tokens=reduce_prepared.prompt_tokens,
                    response_chars=len(raw_final_answer),
                    verification=final_verification,
                    stage="reduce",
                )
            )
            final_prepared = reduce_prepared

        accepted = final_verification.accepted
        if accepted:
            final_answer = self.gateway.restore(final_answer, base.bundle, allowed=allowed_restore)
        else:
            final_answer = ""
        return PipelineResult(final_answer, accepted, "hybrid", attempts, final_prepared)

    def _build_retrieval_maps(
        self,
        base: PreparedContext,
        prompt_budget: int,
    ) -> list[PreparedContext]:
        maps: list[PreparedContext] = []
        bundle = base.bundle
        query_terms = content_terms(bundle.retrieval_query)
        for group in bundle.evidence_source_groups[:1]:
            evidence_ref = group[-1]
            fixed_refs = set(bundle.task_source_refs) | set(group[:-1])

            def prepare_piece(piece: str) -> PreparedContext:
                chunk_bundle = self.gateway._bundle_with_source_refs(
                    bundle,
                    fixed_refs | {evidence_ref},
                    overrides={evidence_ref: piece},
                )
                prompt = self._render_map_prompt(chunk_bundle)
                return self._prepared_for_prompt(
                    base,
                    chunk_bundle,
                    prompt,
                    "chunked_retrieval_map",
                    prompt_budget,
                )

            group_maps = self._split_evidence(bundle.sources[evidence_ref], prepare_piece, prompt_budget)
            relevant = [
                prepared
                for prepared in group_maps
                if query_terms & content_terms(self._included_text(prepared.bundle, evidence_ref))
            ]
            maps.extend(relevant or group_maps)
        return maps

    def _split_evidence(
        self,
        text: str,
        prepare_piece: Callable[[str], PreparedContext],
        prompt_budget: int,
    ) -> list[PreparedContext]:
        whole = prepare_piece(text)
        if whole.fits_prompt_budget:
            return [whole]
        words = text.split()
        chunks: list[PreparedContext] = []
        start = 0
        while start < len(words):
            low = start + 1
            high = len(words)
            best_end = start
            best: PreparedContext | None = None
            while low <= high:
                middle = (low + high) // 2
                candidate = prepare_piece(" ".join(words[start:middle]))
                if candidate.fits_prompt_budget:
                    best_end = middle
                    best = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            if best is None:
                smallest = prepare_piece(words[start])
                raise ContextWindowExceeded(smallest.prompt_tokens, prompt_budget, "chunked_retrieval_map")
            chunks.append(best)
            worst_case_calls = len(chunks) + 1
            if worst_case_calls > self.policy.max_chunk_calls:
                raise ChunkLimitExceeded(worst_case_calls, self.policy.max_chunk_calls)
            if best_end == len(words):
                break
            start = max(best_end - self.policy.chunk_overlap_words, start + 1)
        return chunks

    def _prepared_for_prompt(
        self,
        base: PreparedContext,
        bundle: ContextBundle,
        prompt: str,
        decision: str,
        prompt_budget: int,
    ) -> PreparedContext:
        prompt_tokens = max(self.token_counter(prompt), 1)
        return PreparedContext(
            bundle=bundle,
            prompt=prompt,
            risk=base.risk,
            source_tokens=base.source_tokens,
            prompt_tokens=prompt_tokens,
            token_savings=round(1 - (prompt_tokens / base.source_tokens), 4),
            decision=decision,
            prompt_budget=prompt_budget,
        )

    def _render_map_prompt(self, bundle: ContextBundle) -> str:
        task_refs = set(bundle.task_source_refs)
        evidence = "\n".join(
            str(item["text"])
            for item in bundle.contract["source"]["included"]
            if item["ref"] not in task_refs
        )
        return (
            "Extract the answer stated in the EVIDENCE.\n"
            f"QUESTION:\n{bundle.retrieval_query}\n\nEVIDENCE:\n{evidence}\n\nANSWER:"
        )

    def _render_reduce_prompt(self, bundle: ContextBundle, candidates: list[str]) -> str:
        candidate_text = "\n".join(f"CANDIDATE {index}: {value}" for index, value in enumerate(candidates, 1))
        return (
            "Choose the concise final answer to the QUESTION from the CANDIDATES. Do not add new facts.\n"
            f"QUESTION:\n{bundle.retrieval_query}\n\nCANDIDATES:\n{candidate_text}\n\nANSWER:"
        )

    @staticmethod
    def _is_no_evidence(answer: str) -> bool:
        normalized = " ".join(answer.lower().split())
        return NO_EVIDENCE.lower() in normalized or any(
            phrase in normalized
            for phrase in (
                "not provided in the evidence",
                "not contain the answer",
                "insufficient evidence",
                "not enough information",
                "cannot determine",
            )
        )

    @staticmethod
    def _included_text(bundle: ContextBundle, source_ref: str) -> str:
        return " ".join(
            str(item["text"])
            for item in bundle.contract["source"]["included"]
            if item["ref"] == source_ref
        )

    def _is_grounded(self, answer: str, bundle: ContextBundle) -> bool:
        answer_terms = self._grounding_terms(answer) - self._grounding_terms(bundle.retrieval_query)
        answer_terms -= {"according", "answer", "evidence", "provided", "stated"}
        evidence_terms = self._grounding_terms(
            " ".join(str(item["text"]) for item in bundle.contract["source"]["included"])
        )
        return bool(answer_terms) and answer_terms <= evidence_terms

    @staticmethod
    def _grounding_terms(text: str) -> set[str]:
        visible_text = re.sub(r"</?[A-Za-z][^>]*>", " ", text)
        return content_terms(visible_text) | set(
            re.findall(r"(?<!\w)\d+(?:[.,:]\d+)*(?!\w)", visible_text.lower())
        )

    def verify_response(
        self,
        prepared: PreparedContext,
        response: str,
        preserve_input: bool = False,
    ) -> ResponseVerification:
        if not response.strip():
            return ResponseVerification(
                accepted=False,
                reasons=["empty_response"],
                unknown_placeholders=[],
                missing_placeholders=[],
                new_pii_kinds=[],
            )
        declared = {
            item["placeholder"]
            for item in prepared.bundle.contract["privacy"]["protected"]
        }
        issued = declared & set(PLACEHOLDER_RE.findall(prepared.prompt))
        mentioned = set(PLACEHOLDER_RE.findall(response))
        unknown = sorted(mentioned - issued)
        missing = sorted(issued - mentioned) if preserve_input else []
        response_bundle = self.gateway.compile_private(
            response,
            source_lang=prepared.bundle.contract["language"]["target"],
            target_lang=prepared.bundle.contract["language"]["target"],
            packet_id="response",
            mode="semantic",
        )
        new_pii_kinds = sorted({item["kind"] for item in response_bundle.contract["privacy"]["protected"]})
        reasons = []
        if unknown:
            reasons.append("unknown_placeholders")
        if missing:
            reasons.append("missing_placeholders")
        if self.policy.reject_new_pii and new_pii_kinds:
            reasons.append("new_pii")

        check = None
        if preserve_input:
            expected = copy.deepcopy(prepared.bundle.contract)
            expected["constraints"] = [item for item in expected["constraints"] if item.get("type") != "privacy"]
            check = self.gateway.compare(expected, response_bundle.contract, threshold=self.policy.verification_threshold)
            if check.needs_source:
                reasons.append("semantic_retention")
        return ResponseVerification(
            accepted=not reasons,
            reasons=reasons,
            unknown_placeholders=unknown,
            missing_placeholders=missing,
            new_pii_kinds=new_pii_kinds,
            contract_check=check,
        )

    def _compile_mode(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        risk: Risk,
        packet_id: str,
        mode: str,
        decision: str,
        context_kind: ContextKind = "auto",
        query: str = "",
    ) -> PreparedContext:
        bundle = self.gateway.compile_private(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            packet_id=packet_id,
            mode=mode,
            context_kind=context_kind,
            query=query,
        )
        prompt = self.gateway.render_bundle(bundle)
        masked_source = " ".join(bundle.sources.values())
        normalized_query = " ".join(bundle.retrieval_query.lower().split())
        if normalized_query and normalized_query not in " ".join(masked_source.lower().split()):
            masked_source = f"{masked_source} {bundle.retrieval_query}"
        source_tokens = max(self.token_counter(masked_source), 1)
        prompt_tokens = max(self.token_counter(prompt), 1)
        savings = round(1 - (prompt_tokens / source_tokens), 4)
        return PreparedContext(
            bundle=bundle,
            prompt=prompt,
            risk=risk,
            source_tokens=source_tokens,
            prompt_tokens=prompt_tokens,
            token_savings=savings,
            decision=decision,
        )

    def _resolve_prompt_budget(self, invoke: Invoker | None, explicit: int | None) -> int | None:
        budget = explicit if explicit is not None else self.policy.max_prompt_tokens
        if budget is None and invoke is not None:
            budget = getattr(invoke, "prompt_token_budget", None)
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
            raise ValueError("prompt token budget must be a positive integer")
        return budget

    def _enforce_prompt_budget(
        self,
        prepared: PreparedContext,
        prompt_budget: int | None,
    ) -> PreparedContext:
        prepared.prompt_budget = prompt_budget
        if prompt_budget is not None and not prepared.fits_prompt_budget:
            packed = self.gateway._pack_retrieval_prompt(prepared.bundle, prompt_budget, self.token_counter)
            if packed is not None:
                prepared.bundle = packed
                prepared.prompt = self.gateway.render_bundle(packed)
                prepared.prompt_tokens = max(self.token_counter(prepared.prompt), 1)
                prepared.token_savings = round(1 - (prepared.prompt_tokens / prepared.source_tokens), 4)
                prepared.decision = "retrieval_budget_packed"
        if not prepared.fits_prompt_budget:
            raise ContextWindowExceeded(prepared.prompt_tokens, prompt_budget or 0, prepared.mode)
        return prepared


def fallback_modes(mode: str) -> list[str]:
    if mode == "semantic":
        return ["semantic", "hybrid", "raw"]
    if mode == "hybrid":
        return ["hybrid", "raw"]
    return ["raw"]


def approximate_token_count(text: str) -> int:
    """Dependency-free estimate. Supply the target model tokenizer in production."""

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def clean_model_output(text: str) -> str:
    """Remove provider reasoning wrappers only after the raw response is safety-checked."""

    return THINK_TAG_RE.sub("", THINK_BLOCK_RE.sub("", text)).strip()


__all__ = [
    "ContextPipeline",
    "ChunkLimitExceeded",
    "ContextWindowExceeded",
    "PipelineAttempt",
    "PipelinePolicy",
    "PipelineResult",
    "PreparedContext",
    "ResponseVerification",
    "approximate_token_count",
]
