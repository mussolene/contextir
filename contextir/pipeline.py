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

    def __post_init__(self) -> None:
        for name in ("min_token_savings", "min_semantic_confidence", "verification_threshold"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")


@dataclass
class PreparedContext:
    bundle: ContextBundle
    prompt: str
    risk: Risk
    source_tokens: int
    prompt_tokens: int
    token_savings: float
    decision: str

    @property
    def mode(self) -> str:
        return str(self.bundle.contract["mode"])


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
            "risk": self.prepared.risk,
            "source_tokens": self.prepared.source_tokens,
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
    ) -> None:
        self.gateway = gateway or ContextIR()
        self.policy = policy or PipelinePolicy()
        self.token_counter = token_counter or approximate_token_count

    def prepare(
        self,
        text: str,
        source_lang: str = "ru",
        target_lang: str = "ru",
        risk: Risk = "standard",
        packet_id: str = "context",
    ) -> PreparedContext:
        if risk not in {"low", "standard", "high"}:
            raise ValueError(f"unsupported risk: {risk}")

        if len(text) <= self.gateway.raw_threshold:
            raw = self._compile_mode(text, source_lang, target_lang, risk, packet_id, "raw", "short_input")
            raw.decision = "short_input"
            return raw

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
            return candidate
        if candidate.token_savings < self.policy.min_token_savings:
            raw = self._compile_mode(text, source_lang, target_lang, risk, packet_id, "raw", "raw_baseline")
            raw.decision = "insufficient_token_savings"
            return raw
        return candidate

    def run(
        self,
        text: str,
        invoke: Invoker,
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

        initial = self.prepare(text, source_lang, target_lang, risk, packet_id)
        modes = fallback_modes(initial.mode)[: self.policy.max_attempts]
        attempts: list[PipelineAttempt] = []
        final_prepared = initial
        final_answer = ""
        accepted = False

        for index, mode in enumerate(modes):
            prepared = initial if index == 0 else self._compile_mode(
                text,
                source_lang,
                target_lang,
                risk,
                packet_id,
                mode,
                "verification_fallback",
            )
            answer = invoke(prepared.prompt)
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
        issued = set(prepared.bundle.vault)
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
    "PipelineAttempt",
    "PipelinePolicy",
    "PipelineResult",
    "PreparedContext",
    "ResponseVerification",
    "approximate_token_count",
]
