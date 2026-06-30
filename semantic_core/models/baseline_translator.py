from __future__ import annotations

import json
from pathlib import Path

from semantic_core.tokenizer import normalize


class BaselineTranslator:
    def __init__(self, table: dict[str, str]):
        self.table = table

    def translate(self, text: str) -> str:
        return self.table.get(normalize(text), "<unk>")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"table": self.table}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BaselineTranslator":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["table"])


def train_baseline(rows: list[dict]) -> BaselineTranslator:
    return BaselineTranslator({normalize(row["source"]): row["target"] for row in rows})

