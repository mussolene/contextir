from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_contract_schema() -> dict[str, Any]:
    resource = files(__package__).joinpath("contextir_contract_v2.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


__all__ = ["load_contract_schema"]
