"""Turn data/raw/*.txt into data/processed/*.jsonl.

Placeholders such as {TCKN} are replaced with fake but checksum-valid identifiers and the
exact character span is recorded, giving us gold labels to evaluate the PII masker.
{DECOY} inserts look-alike numbers (order/reference numbers) that must NOT be masked.

Usage: python scripts/build_dataset.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from triagent.synthetic import (
    decoy_number,
    fake_card,
    fake_email,
    fake_iban,
    fake_phone,
    fake_tckn,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = {
    "TCKN": fake_tckn,
    "IBAN": fake_iban,
    "CARD": fake_card,
    "PHONE": fake_phone,
    "EMAIL": fake_email,
    "DECOY": decoy_number,
}
PLACEHOLDER = re.compile(r"\{(TCKN|IBAN|CARD|PHONE|EMAIL|DECOY)\}")


def fill(template: str, rng: random.Random) -> tuple[str, list[dict]]:
    out, spans, cursor = "", [], 0
    for m in PLACEHOLDER.finditer(template):
        out += template[cursor : m.start()]
        kind = m.group(1)
        value = GENERATORS[kind](rng)
        if kind != "DECOY":
            spans.append({"type": kind, "start": len(out), "end": len(out) + len(value)})
        out += value
        cursor = m.end()
    return out + template[cursor:], spans


def build(split: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for line in (ROOT / "data" / "raw" / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        category, priority, template = line.split("|", 2)
        text, spans = fill(template.strip(), rng)
        rows.append(
            {
                "id": f"{split}-{len(rows):04d}",
                "text": text,
                "category": category,
                "priority": priority,
                "pii": spans,
            }
        )
    return rows


def main() -> None:
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, seed in (("train", 13), ("test", 42)):
        rows = build(split, seed)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cats = Counter(r["category"] for r in rows)
        prios = Counter(r["priority"] for r in rows)
        n_pii = sum(len(r["pii"]) for r in rows)
        print(f"{split}: {len(rows)} rows | {dict(cats)} | {dict(prios)} | PII spans: {n_pii}")


if __name__ == "__main__":
    main()