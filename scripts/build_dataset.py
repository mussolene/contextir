#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_core.dataset import build_meanings, paired_examples, split_examples, write_jsonl
from semantic_core.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs("data/processed", "data/raw", "data/dictionary")
    rows = paired_examples(build_meanings())
    train, valid, test = split_examples(rows, cfg["seed"], cfg["data"]["train_size"], cfg["data"]["valid_size"], cfg["data"]["test_size"])
    write_jsonl("data/processed/train.jsonl", train)
    write_jsonl("data/processed/valid.jsonl", valid)
    write_jsonl("data/processed/test.jsonl", test)
    print(f"wrote train={len(train)} valid={len(valid)} test={len(test)} to {Path('data/processed').resolve()}")


if __name__ == "__main__":
    main()
