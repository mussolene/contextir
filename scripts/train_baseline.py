#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextir.dataset import read_jsonl
from contextir.models.baseline_translator import train_baseline
from contextir.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    load_config(args.config)
    ensure_dirs("checkpoints")
    baseline = train_baseline(read_jsonl("data/processed/train.jsonl"))
    baseline.save("checkpoints/baseline_translator.json")
    print(f"saved checkpoints/baseline_translator.json entries={len(baseline.table)}")


if __name__ == "__main__":
    main()
