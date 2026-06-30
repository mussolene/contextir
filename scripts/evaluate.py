#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import zlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_core.dataset import read_jsonl
from semantic_core.evaluation.metrics import (
    ablation_quality,
    cycle_consistency,
    evaluate_baseline,
    evaluate_semantic,
    file_size,
    latency_ms,
    semantic_coherence,
)
from semantic_core.models.baseline_translator import BaselineTranslator
from semantic_core.models.semantic_encoder import SemanticEncoder
from semantic_core.models.semantic_translator import SemanticTranslator
from semantic_core.utils.config import ensure_dirs, load_config


def maybe_plot_clusters(rows, encoder: SemanticEncoder) -> None:
    points = []
    csv_lines = ["meaning_id,x,y"]
    for row in rows:
        vec = encoder.encode_text(row["source"])
        points.append((float(vec[0]), float(vec[1])))
        csv_lines.append(f"{row['meaning_id']},{float(vec[0]):.6f},{float(vec[1]):.6f}")
    Path("reports/semantic_clusters.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except Exception:
        write_scatter_png(points, "reports/semantic_clusters.png")
        return
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=12, alpha=0.7)
    plt.title("Semantic vectors, first two dimensions")
    plt.tight_layout()
    plt.savefig("reports/semantic_clusters.png")
    plt.close()


def write_scatter_png(points: list[tuple[float, float]], path: str) -> None:
    width, height, pad = 720, 520, 32
    pixels = bytearray([255, 255, 255] * width * height)
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        for x, y in points:
            px = int(pad + ((x - min_x) / span_x) * (width - pad * 2))
            py = int(height - pad - ((y - min_y) / span_y) * (height - pad * 2))
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    qx, qy = px + dx, py + dy
                    if 0 <= qx < width and 0 <= qy < height:
                        idx = (qy * width + qx) * 3
                        pixels[idx : idx + 3] = b"\x1f\x77\xb4"
    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    Path(path).write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs("reports")

    rows = read_jsonl("data/processed/test.jsonl")
    sample_rows = rows[: cfg["evaluation"]["num_examples"]]
    encoder = SemanticEncoder.load("checkpoints/semantic_encoder.npz")
    baseline = BaselineTranslator.load("checkpoints/baseline_translator.json")
    translator = SemanticTranslator.load("checkpoints/semantic_translator.json")

    baseline_metrics = evaluate_baseline(rows, baseline)
    semantic_metrics = evaluate_semantic(rows, encoder, translator)
    ablation_metrics = ablation_quality(rows, encoder, translator, cfg["seed"])
    coherence = semantic_coherence(rows, encoder)
    cycle = cycle_consistency(rows, encoder, translator)
    metrics = {
        "baseline": baseline_metrics,
        "semantic_translator": semantic_metrics,
        "semantic_coherence": coherence,
        "cycle_consistency": cycle,
        "ablation": ablation_metrics,
        "efficiency": {
            "baseline_latency_ms": latency_ms(rows, lambda row: baseline.translate(row["source"])),
            "semantic_latency_ms": latency_ms(rows, lambda row: translator.translate(row["source"], row["target_lang"], encoder)),
            "semantic_encoder_size_bytes": file_size("checkpoints/semantic_encoder.npz"),
            "baseline_size_bytes": file_size("checkpoints/baseline_translator.json"),
            "semantic_translator_size_bytes": file_size("checkpoints/semantic_translator.json"),
        },
        "success_criteria": {
            "semantic_gap_ge_0_15": coherence["semantic_gap"] >= 0.15,
            "cycle_consistency_ge_0_75": cycle >= 0.75,
            "ablation_drops_quality": ablation_metrics["exact_match"] < semantic_metrics["exact_match"] * 0.75,
        },
    }
    Path("reports/metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Translation Examples", ""]
    for row in sample_rows:
        pred, meaning_id, score, _ = translator.translate(row["source"], row["target_lang"], encoder)
        lines.append(f"- `{row['source']}` -> `{pred}` | gold: `{row['target']}` | nearest: `{meaning_id}` | score: {score:.3f}")
    Path("reports/examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    maybe_plot_clusters(rows, encoder)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
