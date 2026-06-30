from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from semantic_core.tokenizer import normalize


ENTITIES = [
    ("cat", "кошка", "the cat", ["кошка", "животное", "питомец"], ["the cat", "an animal", "a pet"]),
    ("dog", "собака", "the dog", ["собака", "пёс", "животное"], ["the dog", "a dog", "an animal"]),
    ("child", "ребёнок", "the child", ["ребёнок", "мальчик", "девочка"], ["the child", "a kid", "the youngster"]),
    ("robot", "робот", "the robot", ["робот", "машина", "автомат"], ["the robot", "a machine", "the automaton"]),
    ("teacher", "учитель", "the teacher", ["учитель", "преподаватель", "наставник"], ["the teacher", "the instructor", "the mentor"]),
    ("bird", "птица", "the bird", ["птица", "пернатое", "животное"], ["the bird", "a bird", "the animal"]),
]

ACTIONS = [
    ("sitting", "сидит", "is sitting", ["сидит", "находится", "расположена"], ["is sitting", "is located", "is"]),
    ("standing", "стоит", "is standing", ["стоит", "находится", "расположена"], ["is standing", "is located", "is"]),
    ("sleeping", "спит", "is sleeping", ["спит", "дремлет", "отдыхает"], ["is sleeping", "is napping", "is resting"]),
    ("looking", "смотрит", "is looking", ["смотрит", "наблюдает", "глядит"], ["is looking", "is watching", "is gazing"]),
    ("walking", "идёт", "is walking", ["идёт", "шагает", "движется"], ["is walking", "is moving", "is going"]),
]

PLACES = [
    ("table", "на столе", "on the table", ["на столе", "сверху стола", "у стола"], ["on the table", "at the table", "near the table"]),
    ("window", "у окна", "by the window", ["у окна", "возле окна", "рядом с окном"], ["by the window", "near the window", "at the window"]),
    ("garden", "в саду", "in the garden", ["в саду", "среди растений", "на участке"], ["in the garden", "among plants", "outside"]),
    ("room", "в комнате", "in the room", ["в комнате", "в помещении", "дома"], ["in the room", "indoors", "inside"]),
    ("street", "на улице", "on the street", ["на улице", "на дороге", "снаружи"], ["on the street", "on the road", "outside"]),
]


def _record(entity: tuple, action: tuple, place: tuple) -> dict:
    eid, ru_entity, en_entity, ru_entity_alts, en_entity_alts = entity
    aid, ru_action, en_action, ru_action_alts, en_action_alts = action
    pid, ru_place, en_place, ru_place_alts, en_place_alts = place
    meaning_id = f"m_{eid}_{aid}_{pid}"
    ru = f"{ru_entity} {ru_action} {ru_place}"
    en = f"{en_entity} {en_action} {en_place}"
    ru_paraphrases = [
        f"{ru_place_alts[1]} {ru_action_alts[1]} {ru_entity_alts[0]}",
        f"{ru_entity_alts[1]} {ru_action_alts[2]} {ru_place_alts[2]}",
        f"{ru_entity_alts[2]} {ru_action_alts[0]} {ru_place_alts[0]}",
    ]
    en_paraphrases = [
        f"{en_entity_alts[1]} {en_action_alts[1]} {en_place_alts[1]}",
        f"{en_entity_alts[2]} {en_action_alts[2]} {en_place_alts[2]}",
        f"{en_entity_alts[0]} {en_action_alts[0]} {en_place_alts[0]}",
    ]
    atoms = [f"object:{eid}", f"state:{aid}", f"place:{pid}"]
    return {
        "meaning_id": meaning_id,
        "ru": ru,
        "en": en,
        "ru_paraphrases": ru_paraphrases,
        "en_paraphrases": en_paraphrases,
        "semantic_atoms": atoms,
        "definition_ru": f"смысл: {ru_entity} {ru_action} {ru_place}",
        "definition_en": f"meaning: {en_entity} {en_action} {en_place}",
    }


def build_meanings() -> list[dict]:
    return [_record(e, a, p) for e in ENTITIES for a in ACTIONS for p in PLACES]


def variants(record: dict, lang: str) -> list[str]:
    if lang == "ru":
        return [record["ru"], *record["ru_paraphrases"], record["definition_ru"]]
    if lang == "en":
        return [record["en"], *record["en_paraphrases"], record["definition_en"]]
    raise ValueError(f"unsupported language: {lang}")


def paired_examples(records: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        for ru_text in variants(rec, "ru"):
            rows.append({"meaning_id": rec["meaning_id"], "source_lang": "ru", "target_lang": "en", "source": ru_text, "target": rec["en"], "semantic_atoms": rec["semantic_atoms"]})
        for en_text in variants(rec, "en"):
            rows.append({"meaning_id": rec["meaning_id"], "source_lang": "en", "target_lang": "ru", "source": en_text, "target": rec["ru"], "semantic_atoms": rec["semantic_atoms"]})
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_examples(rows: list[dict], seed: int, train_size: int, valid_size: int, test_size: int) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["meaning_id"], []).append(row)
    rng = random.Random(seed)
    train: list[dict] = []
    valid: list[dict] = []
    test: list[dict] = []
    for group in grouped.values():
        rng.shuffle(group)
        train.extend(group[:6])
        valid.extend(group[6:8])
        test.extend(group[8:])
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train[:train_size], valid[:valid_size], test[:test_size]


def phrase_index(rows: Iterable[dict]) -> dict[str, dict]:
    return {normalize(row["source"]): row for row in rows}

