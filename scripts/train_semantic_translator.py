#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_core.dataset import read_jsonl
from semantic_core.models.semantic_encoder import SemanticEncoder
from semantic_core.models.semantic_translator import train_semantic_translator
from semantic_core.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    load_config(args.config)
    ensure_dirs("checkpoints")
    rows = read_jsonl("data/processed/train.jsonl") + read_jsonl("data/processed/valid.jsonl")
    encoder = SemanticEncoder.load("checkpoints/semantic_encoder.npz")
    translator = train_semantic_translator(rows, encoder)
    translator.save("checkpoints/semantic_translator.json")
    print(f"saved checkpoints/semantic_translator.json meanings={len(translator.prototypes)}")


if __name__ == "__main__":
    main()
