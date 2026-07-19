from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Callable, Literal

from contextir.gateway import ContractCheck, ContextBundle, ContextIR


Risk = Literal["low", "standard", "high"]
Task = Literal["reasoning", "transform"]
TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class PipelinePolicy:
    min_token_savings: float = 0.15
    min_semantic_confidence: float = 0.72
    verification_threshold: float = 0.9
    max_attempts: int = 3
    reject_new_pii: bool = True
    max_prompt_tokens: int | None = None

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


class ContextWindowExceeded(RuntimeError):
    """Raised before invocation when a safe prompt cannot fit the model budget."""

    def __init__(self, prompt_tokens: int, prompt_budget: int, mode: str) -> None:
        self.prompt_tokens = prompt_tokens
        self.prompt_budget = prompt_budget
        self.mode = mode
        super().__init__(
            f"{mode} prompt requires {prompt_tokens} tokens but the model budget is {prompt_budget}"
        )


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
    ) -> PreparedContext:
        if risk not in {"low", "standard", "high"}:
            raise ValueError(f"unsupported risk: {risk}")

        prompt_budget = self._resolve_prompt_budget(self.invoke, max_prompt_tokens)

        if len(text) <= self.gateway.raw_threshold:
            raw = self._compile_mode(text, source_lang, target_lang, risk, packet_id, "raw", "short_input")
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
            )

        if candidate.mode == "raw":
            candidate.decision = "compiler_selected_raw"
            return self._enforce_prompt_budget(candidate, prompt_budget)
        candidate = self._enforce_prompt_budget(candidate, prompt_budget)
        if candidate.token_savings < self.policy.min_token_savings:
            raw = self._compile_mode(text, source_lang, target_lang, risk, packet_id, "raw", "raw_baseline")
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
    ) -> PipelineResult:
        if task not in {"reasoning", "transform"}:
            raise ValueError(f"unsupported task: {task}")
        model_invoke = invoke or self.invoke
        if model_invoke is None:
            raise ValueError("invoke is required; pass it to ContextPipeline() or run()")

        prompt_budget = self._resolve_prompt_budget(model_invoke, None)
        initial = self.prepare(
            text,
            source_lang,
            target_lang,
            risk,
            packet_id,
            max_prompt_tokens=prompt_budget,
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
            final_answer = answer
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
        issued = {
            item["placeholder"]
            for item in prepared.bundle.contract["privacy"]["protected"]
        }
        mentioned = set(re.findall(r"\bPII_[A-Z0-9_]+_\d+\b", response))
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
    ) -> PreparedContext:
        bundle = self.gateway.compile_private(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            packet_id=packet_id,
            mode=mode,
        )
        prompt = self.gateway.render_prompt(bundle.contract)
        masked_source = " ".join(bundle.sources.values())
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
                prepared.prompt = self.gateway.render_prompt(packed.contract)
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


__all__ = [
    "ContextPipeline",
    "ContextWindowExceeded",
    "PipelineAttempt",
    "PipelinePolicy",
    "PipelineResult",
    "PreparedContext",
    "ResponseVerification",
    "approximate_token_count",
]
