from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from semantic_core.sir_runtime import SIRRuntime, SIRV1Packet, load_runtime
from semantic_core.sir_sources import PROJECT_ROOT


class SIRKernel:
    """Minimal public SIR kernel: text -> semantic contract, contract -> text."""

    def __init__(self, runtime: SIRRuntime):
        self.runtime = runtime

    def compile(self, text: str, source_lang: str = "ru", target_lang: str = "ru", packet_id: str = "contract") -> dict[str, Any]:
        started = time.perf_counter()
        packet, _vault = self.runtime.compile_request(text, source_lang=source_lang, target_lang=target_lang, packet_id=packet_id)
        contract = packet_to_contract(packet)
        contract["kernel"] = {
            "name": "sir-kernel",
            "operation": "compile",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return contract

    def decompile(self, contract: dict[str, Any], target_lang: str | None = None, include_anchors: bool = False) -> str:
        lang = target_lang or str(contract.get("target_lang") or contract.get("source_lang") or "en")
        concepts = contract.get("concepts", [])
        segments: dict[int, list[str]] = {}
        for item in concepts:
            if not isinstance(item, dict):
                continue
            segment = int(item.get("segment", 0) if item.get("segment", 0) is not None else 0)
            surface = item.get("surface", {})
            term = ""
            if isinstance(surface, dict):
                term = str(surface.get(lang) or surface.get("en") or surface.get("ru") or "")
            if not term:
                term = str(item.get("id", ""))
            if term and term not in segments.setdefault(segment, []):
                segments[segment].append(term)
        phrases = []
        for _segment, terms in sorted(segments.items()):
            if terms:
                phrases.append(", ".join(terms[:5]))
        text = ". ".join(phrases)
        placeholders = [span.get("placeholder", "") for span in contract.get("protected_spans", []) if isinstance(span, dict)]
        if placeholders:
            suffix = "Protected placeholders: " + ", ".join(placeholders)
            text = f"{text}. {suffix}" if text else suffix
        if include_anchors:
            anchors = " ".join(item["id"] for item in concepts if isinstance(item, dict) and item.get("id"))
            if anchors:
                text = f"{text}\nSIR anchors: {anchors}" if text else f"SIR anchors: {anchors}"
        return text


def packet_to_contract(packet: SIRV1Packet) -> dict[str, Any]:
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
        "protected_spans": public_protected_spans(data["protected_spans"]),
        "uncertainty": data["uncertainty"],
        "stats": data["stats"],
    }


def public_protected_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public = []
    for span in spans:
        public.append(
            {
                "placeholder": span["placeholder"],
                "kind": span["kind"],
                "surface_hash": span["surface_hash"],
            }
        )
    return public


def load_kernel(records: Path | None = None) -> SIRKernel:
    return SIRKernel(load_runtime(records))


def read_contract(path: str) -> dict[str, Any]:
    if path == "-":
        import sys

        return json.loads(sys.stdin.read())
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="SIR kernel: text <-> semantic contract.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    compile_p = sub.add_parser("compile", help="Compile text into a SIR semantic contract.")
    compile_p.add_argument("--text", required=True)
    compile_p.add_argument("--source-lang", default="ru")
    compile_p.add_argument("--target-lang", default="ru")
    compile_p.add_argument("--packet-id", default="contract")
    compile_p.add_argument("--out", default="-")

    decompile_p = sub.add_parser("decompile", help="Decompile a SIR semantic contract into text.")
    decompile_p.add_argument("--contract", required=True, help="Path to JSON contract, or '-' for stdin.")
    decompile_p.add_argument("--target-lang", default="")
    decompile_p.add_argument("--include-anchors", action="store_true")
    decompile_p.add_argument("--out", default="-")
    args = parser.parse_args()

    kernel = load_kernel()
    if args.cmd == "compile":
        contract = kernel.compile(args.text, source_lang=args.source_lang, target_lang=args.target_lang, packet_id=args.packet_id)
        payload = json.dumps(contract, ensure_ascii=False, indent=2) + "\n"
    else:
        contract = read_contract(args.contract)
        payload = kernel.decompile(contract, target_lang=args.target_lang or None, include_anchors=args.include_anchors) + "\n"
    if args.out == "-":
        print(payload, end="")
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
