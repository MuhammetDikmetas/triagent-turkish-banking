"""Domain types shared across the pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    CARD = "CARD"
    LOAN = "LOAN"
    TRANSFER = "TRANSFER"
    FRAUD = "FRAUD"
    DIGITAL = "DIGITAL"
    ATM = "ATM"
    FEES = "FEES"
    BRANCH = "BRANCH"


CATEGORY_TR: dict[str, str] = {
    "CARD": "Kart İşlemleri",
    "LOAN": "Kredi & Taksit",
    "TRANSFER": "Para Transferi (EFT/FAST/Havale)",
    "FRAUD": "Dolandırıcılık & Güvenlik",
    "DIGITAL": "Mobil & İnternet Bankacılığı",
    "ATM": "ATM",
    "FEES": "Ücret, Aidat & Komisyon",
    "BRANCH": "Şube & Müşteri Hizmetleri",
}

# Which team receives the ticket. In a real bank this comes from the org chart.
ROUTING: dict[str, str] = {
    "CARD": "Kart Operasyon",
    "LOAN": "Bireysel Krediler",
    "TRANSFER": "Ödeme Sistemleri",
    "FRAUD": "Fraud & Güvenlik (7/24)",
    "DIGITAL": "Dijital Kanallar Destek",
    "ATM": "ATM Operasyon",
    "FEES": "Ücret İade Masası",
    "BRANCH": "Müşteri Deneyimi",
}


class Priority(str, Enum):
    P1 = "P1"  # Acil  - possible financial loss happening now
    P2 = "P2"  # Yüksek - customer's money is stuck / legal escalation risk
    P3 = "P3"  # Normal


PRIORITY_TR = {"P1": "Acil", "P2": "Yüksek", "P3": "Normal"}


class PIIEntity(BaseModel):
    type: str
    start: int
    end: int


class LLMTriage(BaseModel):
    """Structured output we ask the LLM for. Validated with Pydantic, never trusted blindly."""

    category: Category
    priority: Priority
    summary: str = Field(description="One-sentence Turkish summary of the request")
    amount_tl: float | None = Field(default=None, description="Amount mentioned, in TL")
    reason: str = Field(default="", description="Short justification for the label")


class TriageResult(BaseModel):
    ticket_id: str
    masked_text: str
    pii: list[PIIEntity]
    category: Category
    category_tr: str
    team: str
    confidence: float
    decided_by: str  # "classifier" | "llm" | "classifier (llm unavailable)"
    priority: Priority
    priority_reasons: list[str]
    flags: list[str]
    summary: str
    amounts_tl: list[float]
    draft_reply: str
    status: str  # "pending_review" | "approved" | "rejected"
    final_reply: str | None = None