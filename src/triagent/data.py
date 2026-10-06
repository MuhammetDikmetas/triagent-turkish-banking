"""Dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

from triagent.config import get_settings
from triagent.pii import mask


def load_split(split: str, data_dir: Path | None = None) -> list[dict]:
    data_dir = data_dir or get_settings().data_dir
    path = data_dir / "processed" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python scripts/build_dataset.py")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    # The model only ever sees masked text, in training and in production alike.
    for r in rows:
        r["masked_text"] = mask(r["text"]).text
    return rows
