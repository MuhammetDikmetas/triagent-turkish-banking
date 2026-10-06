"""Generators for *fake but checksum-valid* identifiers.

Used to inject realistic PII into the synthetic dataset and tests. No real person's
data is ever used. Decoy generators produce look-alike numbers that must NOT be masked
(order numbers, reference codes) so we can measure false positives too.
"""

from __future__ import annotations

import random


def fake_tckn(rng: random.Random) -> str:
    d = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(8)]
    d10 = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    d.append(d10)
    d.append(sum(d) % 10)
    return "".join(map(str, d))


def fake_iban(rng: random.Random, spaced: bool = True) -> str:
    bban = f"{rng.randint(10, 99999):05d}0" + "".join(str(rng.randint(0, 9)) for _ in range(16))
    check = 98 - int(bban + "292700") % 97  # "TR00" -> T=29, R=27
    iban = f"TR{check:02d}{bban}"
    if not spaced:
        return iban
    return " ".join(iban[i : i + 4] for i in range(0, len(iban), 4))


def fake_card(rng: random.Random, spaced: bool = True) -> str:
    prefix = rng.choice(["4", "5", "9792"])  # Visa, Mastercard, Troy
    body = [int(c) for c in prefix] + [rng.randint(0, 9) for _ in range(15 - len(prefix))]
    total = 0
    for i, digit in enumerate(reversed(body)):
        if i % 2 == 0:  # positions that get doubled once the check digit is appended
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    body.append((10 - total % 10) % 10)
    s = "".join(map(str, body))
    return " ".join(s[i : i + 4] for i in range(0, 16, 4)) if spaced else s


def fake_phone(rng: random.Random) -> str:
    op = rng.choice(["532", "533", "535", "542", "505", "553", "555", "544"])
    a, b, c = rng.randint(100, 999), rng.randint(10, 99), rng.randint(10, 99)
    style = rng.randint(0, 2)
    if style == 0:
        return f"0{op} {a} {b} {c}"
    if style == 1:
        return f"+90 {op} {a} {b}{c}"
    return f"0{op}{a}{b}{c}"


def fake_email(rng: random.Random) -> str:
    names = ["ahmet", "ayse", "mehmet", "zeynep", "mustafa", "elif", "emre", "fatma"]
    return f"{rng.choice(names)}.{rng.randint(10, 999)}@{rng.choice(['gmail.com', 'hotmail.com', 'outlook.com'])}"


def decoy_number(rng: random.Random) -> str:
    """11-digit number that fails the TCKN checksum, e.g. an order / reference number."""
    from triagent.pii import is_valid_tckn

    while True:
        n = str(rng.randint(10**10, 10**11 - 1))
        if not is_valid_tckn(n):
            return n
