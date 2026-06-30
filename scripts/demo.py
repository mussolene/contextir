#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_core.models.semantic_encoder import SemanticEncoder
from semantic_core.models.semantic_translator import SemanticTranslator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--target", choices=["ru", "en"], required=True)
    args = parser.parse_args()

    encoder = SemanticEncoder.load("checkpoints/semantic_encoder.npz")
    translator = SemanticTranslator.load("checkpoints/semantic_translator.json")
    start = time.perf_counter()
    translation, meaning_id, score, _ = translator.translate(args.text, args.target, encoder)
    latency = (time.perf_counter() - start) * 1000
    print(f"Input: {args.text}")
    print(f"Target language: {args.target}")
    print(f"Semantic nearest meaning: {meaning_id}")
    print(f"Translation: {translation}")
    print(f"Coherence score: {score:.4f}")
    print(f"Latency ms: {latency:.3f}")


if __name__ == "__main__":
    main()
