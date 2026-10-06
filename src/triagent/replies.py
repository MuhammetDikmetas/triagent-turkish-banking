"""Reply drafting: approved templates as fallback + an output guardrail for LLM drafts."""

from __future__ import annotations

import re

from triagent.pii import detect
from triagent.text import fold

_CLOSING = "\n\nBilgilerinize sunar, iyi günler dileriz."

TEMPLATES: dict[str, str] = {
    "FRAUD": (
        "Sayın Müşterimiz,\n\nBildiriminiz güvenlik ekibimize öncelikli olarak iletilmiştir. "
        "Kartınızın veya hesabınızın güvenliği için lütfen hemen 7/24 güvenlik hattımızı arayınız. "
        "Bankamız sizden hiçbir koşulda şifre, kart bilgisi veya SMS onay kodu talep etmez."
    ),
    "TRANSFER": (
        "Sayın Müşterimiz,\n\nPara transferinizle ilgili talebiniz Ödeme Sistemleri ekibimize "
        "iletilmiştir. İşlemin durumu incelenerek en kısa sürede tarafınıza bilgi verilecektir."
    ),
    "ATM": (
        "Sayın Müşterimiz,\n\nATM işleminizle ilgili bildiriminiz alınmıştır. İlgili ATM'nin "
        "kayıtları ekibimizce incelenecek ve sonuç hakkında tarafınıza bilgi verilecektir."
    ),
    "CARD": (
        "Sayın Müşterimiz,\n\nKart işleminizle ilgili talebiniz Kart Operasyon ekibimize "
        "iletilmiştir. Talebiniz incelenerek size dönüş yapılacaktır."
    ),
    "LOAN": (
        "Sayın Müşterimiz,\n\nKredi ile ilgili talebiniz Bireysel Krediler ekibimize iletilmiştir. "
        "Müşteri temsilcimiz size uygun seçenekler hakkında bilgi verecektir."
    ),
    "DIGITAL": (
        "Sayın Müşterimiz,\n\nDijital kanallarımızla ilgili yaşadığınız sorun için özür dileriz. "
        "Bildiriminiz Dijital Kanallar Destek ekibimize iletilmiştir."
    ),
    "FEES": (
        "Sayın Müşterimiz,\n\nÜcret ve masraflarla ilgili talebiniz incelenmek üzere ilgili "
        "ekibimize iletilmiştir. Değerlendirme sonucu tarafınıza bildirilecektir."
    ),
    "BRANCH": (
        "Sayın Müşterimiz,\n\nGeri bildiriminiz için teşekkür ederiz. Mesajınız Müşteri Deneyimi "
        "ekibimize iletilmiştir ve gerekli değerlendirme yapılacaktır."
    ),
}

# Things a bank must never write to a customer.
_FORBIDDEN = re.compile(
    r"(sifre|parola|cvv|cvc|onay kod|sms kod|dogrulama kod)\w*\s+(\w+\s+){0,3}"
    r"(paylas|gonder|ilet|yaz|bildir|soyle|ver)(?!m[ae])"  # negation = good advice
)
_PROMISES = re.compile(r"iade (edilecektir|edecegiz|yapilacaktir)|garanti (ediyoruz|ederiz)")


def template_reply(category: str) -> str:
    return TEMPLATES[category] + _CLOSING


def check_draft(draft: str) -> list[str]:
    """Return a list of guardrail violations ( empty list = safe to show the agent )."""
    problems = []
    t = fold(draft)
    if detect(draft):
        problems.append("draft contains personal data")
    if _FORBIDDEN.search(t):
        problems.append("draft asks for credentials / OTP")
    if _PROMISES.search(t):
        problems.append("draft makes a refund / guarantee promise")
    if len(draft.split()) > 160:
        problems.append("draft too long")
    return problems