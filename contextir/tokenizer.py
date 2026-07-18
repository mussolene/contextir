from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[\wё]+", re.IGNORECASE)


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def token_accuracy(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    matches = sum(1 for p, g in zip(pred_tokens, gold_tokens) if p == g)
    return matches / max(len(gold_tokens), len(pred_tokens), 1)

