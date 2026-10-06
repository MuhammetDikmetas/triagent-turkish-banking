"""Deterministic business rules: the safety net around the ML models.

Why rules at all? Two errors are not equally expensive. Mis-routing a fee question costs
a few minutes; missing an active fraud case can cost the customer their savings. So
high-risk signals are handled by transparent, auditable rules that a compliance team
can read and change, instead of trusting a probabilistic model alone.

Rules were written from the *training* split only. All matching is done on ASCII-folded
text so "kartım çalındı" and "kartim calindi" behave the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from triagent.text import fold

PRIORITY_ORDER = {"P1": 0, "P2": 1, "P3": 2}


def _rx(*phrases: str) -> re.Pattern[str]:
    return re.compile("|".join(phrases))


# Active fraud / loss of control over card or account -> FRAUD + P1
FRAUD_SIGNALS = _rx(
    r"\bcalin(di|mis)",
    r"\bcaldir",
    r"(kart|cuzdan|canta|kimli)\w* kaybet",
    r"(kart|cuzdan|kimli)\w* kayboldu",
    r"bilgim (disinda|olmadan)",
    r"benden habersiz",
    r"\bizinsiz",
    r"dolandir",
    r"tanimadigim (bir )?(site|harcama|islem|iban)",
    r"(ben )?yapmadigim",
    r"ben yapmadim",
    r"sifremi (verdim|paylastim)",
    r"kodu? (birine )?(soyledim|verdim)",
    r"ele gecir",
    r"bosaltil",
    r"kopyalan",
    r"onaylamadim",
    r"istemedim.*kod",
    r"bilmedigim bir (iban|cihaz|kredi)",
)
# Fraud *report* without loss (phishing SMS seen, link not clicked) -> P2, not P1
FRAUD_REPORT_ONLY = _rx(
    r"tiklamadim",
    r"bilgi(lerimi)? girmedim",
    r"girmeden sildim",
    r"bildirmek ist",
    r"bildiriyorum",
    r"gercek mi",
    r"yapmadim ama",
    r"talep etmedim",
)
# Customer's money is stuck / missing -> P2 (only meaningful for TRANSFER and ATM)
MONEY_STUCK = _rx(
    r"gecmedi",
    r"ulasmadi",
    r"gelmedi",
    r"dustu",
    r"ic(i|e)r(i|)de kaldi",
    r"icinde kaldi",
    r"\byuttu",
    r"sikisti",
    r"yatmadi",
    r"yatmamis",
    r"yansimadi",
    r"yanlis (iban|kisi)",
    r"yanlislikla \d",
    r"iki (kez|kere)",
    r"\beksik",
    r"kayboldu",
    r"geri vermedi",
    r"\d+ yerine",
    r"gundur bekl",
    r"para gelmedi",
    r"maas\w* (odemeleri )?(yatma|alama|yapilama)",
)
# Customer cannot access their own money right now
ACCESS_BLOCKED = re.compile(
    r"(bloke|kapatil|kapanmis|acilmasi).*(odeyemedim|odeme yapamadim|odeme yapamiyorum|acil)"
    r"|(odeyemedim|odeme yapamadim).*(bloke|kapatil)"
)
LEGAL_THREAT = _rx(
    r"\bbddk",
    r"tuketici hakem",
    r"avukat",
    r"hukuki",
    r"\bicra",
    r"dava ac",
    r"savcilig",
)
HARDSHIP = _rx(r"isimi kaybettim", r"odeyemiyorum", r"odeyemem", r"muaccel")
DEADLINE = _rx(r"son gun", r"gecikme ceza", r"yarin gun bitiyor", r"fesih")
CHURN = _rx(r"hesabimi kapatacag", r"baska bankaya", r"karti kapatacag", r"musteriniz olmak ist")
VULNERABLE = _rx(
    r"\byasli",
    r"\d{2} yasinda",
    r"engelli",
    r"tekerlekli sandalye",
    r"vefat",
    r"emekli",
)


@dataclass
class RuleOutcome:
    category_override: str | None = None
    priority: str = "P3"
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def more_urgent(a: str, b: str) -> str:
    return a if PRIORITY_ORDER[a] <= PRIORITY_ORDER[b] else b


def apply_rules(text: str, category: str) -> RuleOutcome:
    t = fold(text)
    out = RuleOutcome()

    def raise_to(level: str, reason: str) -> None:
        out.priority = more_urgent(out.priority, level)
        out.reasons.append(reason)

    fraud_hit = FRAUD_SIGNALS.search(t)
    report_only = FRAUD_REPORT_ONLY.search(t)

    if fraud_hit and not report_only:
        out.flags.append("fraud_signal")
        if category != "FRAUD":
            out.category_override = "FRAUD"
            out.reasons.append(
                f"dolandırıcılık ifadesi '{fraud_hit.group()}' → Fraud ekibine yönlendirildi"
            )
        raise_to("P1", "aktif dolandırıcılık / kart veya hesap kontrolü kaybı")
    elif category == "FRAUD" and not report_only:
        raise_to("P1", "dolandırıcılık / güvenlik olayı olarak sınıflandırıldı")
    elif category == "FRAUD" or fraud_hit:
        raise_to("P2", "zarar teyit edilmemiş dolandırıcılık bildirimi")

    final_category = out.category_override or category
    if final_category in {"TRANSFER", "ATM"} and (m := MONEY_STUCK.search(t)):
        raise_to("P2", f"müşterinin parası askıda ('{m.group()}')")
    if ACCESS_BLOCKED.search(t):
        raise_to("P2", "müşteri kendi parasına erişemiyor")
    if m := LEGAL_THREAT.search(t):
        out.flags.append("legal_threat")
        raise_to("P2", f"yasal yol / düzenleyici kurum ('{m.group()}')")
    if m := HARDSHIP.search(t):
        out.flags.append("financial_hardship")
        raise_to("P2", f"ödeme güçlüğü ('{m.group()}')")
    if m := DEADLINE.search(t):
        raise_to("P2", f"zaman kritik ('{m.group()}')")
    if CHURN.search(t):
        out.flags.append("churn_risk")
    if VULNERABLE.search(t):
        out.flags.append("vulnerable_customer")
    return out
