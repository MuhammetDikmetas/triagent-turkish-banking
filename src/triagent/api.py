"""REST API + web console.

Run: uvicorn triagent.api:app --reload
  -> web console at http://localhost:8000      API docs at http://localhost:8000/docs
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from triagent.config import ROOT, get_settings
from triagent.schemas import TriageResult
from triagent.service import get_service

app = FastAPI(
    title="Triagent API",
    description="KVKK-aware triage agent for Turkish banking customer messages",
    version="0.1.0",
)


class MessageIn(BaseModel):
    text: str = Field(min_length=3, max_length=4000, examples=["Kartım çalındı, hemen kapatın!"])


class ReviewIn(BaseModel):
    action: Literal["approve", "edit", "reject"]
    reply: str | None = None
    note: str | None = None


@app.get("/", include_in_schema=False)
def web_console() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "llm_provider": s.llm_provider if s.llm_enabled else "none (offline mode)",
        "encoder": s.encoder,
        "confidence_threshold": s.confidence_threshold,
    }


@app.post("/tickets", response_model=TriageResult)
def create_ticket(msg: MessageIn) -> TriageResult:
    return get_service().submit(msg.text)


@app.get("/tickets")
def list_tickets(status: str | None = None) -> list[dict]:
    return get_service().list(status)


@app.get("/tickets/{ticket_id}", response_model=TriageResult)
def get_ticket(ticket_id: str) -> TriageResult:
    t = get_service().get(ticket_id)
    if t is None:
        raise HTTPException(404, "ticket not found")
    return t


@app.post("/tickets/{ticket_id}/review", response_model=TriageResult)
def review_ticket(ticket_id: str, body: ReviewIn) -> TriageResult:
    try:
        return get_service().review(ticket_id, body.action, body.reply, body.note)
    except KeyError:
        raise HTTPException(404, "ticket not found") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None