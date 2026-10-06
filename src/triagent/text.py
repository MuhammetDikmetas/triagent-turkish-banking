"""Turkish-aware text normalisation.

Two Turkish-specific pitfalls handled here:

1. ``str.lower()`` is locale-unaware: ``"KARTIM".lower()`` gives ``"kartim"`` (should be
   ``"kartım"``) and ``"İPTAL".lower()`` gives ``"i̇ptal"`` with a combining dot.
2. Customers often type without Turkish characters ("kartim calindi"), so keyword
   rules must match both spellings. We fold everything to ASCII before matching.
"""

from __future__ import annotations

import re

_ASCII_FOLD = str.maketrans("ıiğüşöçâîû", "iigusocaiu")


def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def fold(text: str) -> str:
    """Lower-case (Turkish rules) and strip diacritics: 'Kartım ÇALINDI' -> 'kartim calindi'."""
    return re.sub(r"\s+", " ", tr_lower(text).translate(_ASCII_FOLD)).strip()


_AMOUNT_RE = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{1,2}))?\s*(?:tl|try|lira|₺)(?!\w)",
    re.IGNORECASE,
)


def extract_amounts(text: str) -> list[float]:
    """Find TL amounts written the Turkish way: '1.250,50 TL', '500 lira', '3000 TL'."""
    out = []
    for whole, frac in _AMOUNT_RE.findall(tr_lower(text)):
        value = float(whole.replace(".", ""))
        if frac:
            value += float(f"0.{frac}")
        out.append(value)
    return out
