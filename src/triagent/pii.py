"""KVKK-aware PII masking that runs locally, *before* any text reaches an LLM.

Design choice: pattern matching alone produces many false positives in banking text
(order numbers, reference codes, customer numbers are all long digit strings). So every
candidate is checked with the identifier's own validation algorithm:

* T.C. Kimlik No  -> official 10th/11th check-digit rules
* TR IBAN         -> ISO 13616 mod-97 check
* Card number     -> issuer prefix + Luhn checksum (Visa, Mastercard, Troy, Amex)
* Mobile phone    -> Turkish GSM format (5xx), with optional +90 / 0 prefix
* E-mail          -> simple RFC-ish pattern

Known limitation: person names and addresses are not masked (would need a Turkish NER
model). See README > Limitations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from triagent.schemas import PIIEntity

MASK_TOKENS = {
    "IBAN": "[IBAN]",
    "CARD": "[KART_NO]",
    "TCKN": "[TCKN]",
    "PHONE": "[TELEFON]",
    "EMAIL": "[EPOSTA]",
}


# --------------------------------------------------------------------------- validators
def is_valid_tckn(value: str) -> bool:
    if not re.fullmatch(r"[1-9]\d{10}", value):
        return False
    d = [int(c) for c in value]
    odd = d[0] + d[2] + d[4] + d[6] + d[8]
    even = d[1] + d[3] + d[5] + d[7]
    if (odd * 7 - even) % 10 != d[9]:
        return False
    return sum(d[:10]) % 10 == d[10]


def is_valid_iban_tr(value: str) -> bool:
    iban = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"TR\d{24}", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)  # T->29, R->27
    return int(numeric) % 97 == 1


def luhn_ok(value: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19:
        return False
    # Issuer prefix: Amex 3, Visa 4, Mastercard 2/5, Discover 6, Troy 9792.
    if digits[0] not in (2, 3, 4, 5, 6) and digits[:4] != [9, 7, 9, 2]:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# --------------------------------------------------------------------------- patterns
# Order matters: longer / more specific identifiers are claimed first so that,
# e.g., the digits inside an IBAN are never re-detected as a TCKN.
_PATTERNS: list[tuple[str, re.Pattern[str], object]] = [
    ("IBAN", re.compile(r"\bTR\s?\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE), is_valid_iban_tr),
    ("CARD", re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"), luhn_ok),
    ("TCKN", re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)"), is_valid_tckn),
    (
        "PHONE",
        re.compile(
            r"(?<![\d+])(?:\+90[\s-]?|0090[\s-]?|0)?\(?5\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
        ),
        None,
    ),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), None),
]


@dataclass
class MaskResult:
    text: str
    entities: list[PIIEntity]

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entities:
            out[e.type] = out.get(e.type, 0) + 1
        return out


def detect(text: str) -> list[PIIEntity]:
    """Return non-overlapping PII spans in the *original* text."""
    taken: list[tuple[int, int]] = []
    found: list[PIIEntity] = []
    for pii_type, pattern, validator in _PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < e and end > s for s, e in taken):
                continue
            if validator is not None and not validator(m.group()):
                continue
            taken.append((start, end))
            found.append(PIIEntity(type=pii_type, start=start, end=end))
    return sorted(found, key=lambda e: e.start)


def mask(text: str) -> MaskResult:
    entities = detect(text)
    out, cursor = [], 0
    for e in entities:
        out.append(text[cursor : e.start])
        out.append(MASK_TOKENS[e.type])
        cursor = e.end
    out.append(text[cursor:])
    return MaskResult(text="".join(out), entities=entities)