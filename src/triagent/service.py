"""Application service: the single entry point used by both the API and the panel.

submit(raw_text)  -> mask PII -> run graph until human_review -> store ticket
review(id, ...)   -> resume graph with the agent's decision -> update ticket
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from triagent.classifier import FastClassifier
from triagent.config import Settings, get_settings
from triagent.graph import build_graph
from triagent.llm import LLMClient
from triagent.pii import mask
from triagent.schemas import CATEGORY_TR, ROUTING, PIIEntity, TriageResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    payload TEXT NOT NULL
)"""


class TriageService:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.s.db_path, check_same_thread=False)
        self.conn.execute(_SCHEMA)
        self.llm = LLMClient(self.s, use_cache=False)
        self.graph = build_graph(
            classifier=FastClassifier.load(self.s.model_path),
            llm=self.llm,
            threshold=self.s.confidence_threshold,
            checkpointer=SqliteSaver(self.conn),
        )

    # ------------------------------------------------------------------ helpers
    def _to_result(self, state: dict, pii: list[PIIEntity]) -> TriageResult:
        return TriageResult(
            ticket_id=state["ticket_id"],
            masked_text=state["masked_text"],
            pii=pii,
            category=state["category"],
            category_tr=CATEGORY_TR[state["category"]],
            team=ROUTING[state["category"]],
            confidence=round(state["confidence"], 3),
            decided_by=state["decided_by"],
            priority=state["priority"],
            priority_reasons=state["priority_reasons"],
            flags=state["flags"],
            summary=state.get("summary", ""),
            amounts_tl=state["amounts_tl"],
            draft_reply=state["draft_reply"],
            status=state["status"],
            final_reply=state.get("final_reply"),
        )

    def _save(self, result: TriageResult, created_at: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO tickets VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "status=excluded.status, category=excluded.category, "
                "priority=excluded.priority, payload=excluded.payload",
                (
                    result.ticket_id,
                    created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    result.status,
                    result.category.value,
                    result.priority.value,
                    result.model_dump_json(),
                ),
            )
            self.conn.commit()

    # ------------------------------------------------------------------ API
    def submit(self, raw_text: str) -> TriageResult:
        masked = mask(raw_text)  # <- the only place raw text is touched
        ticket_id = uuid.uuid4().hex[:10]
        config = {"configurable": {"thread_id": ticket_id}}
        with self._lock:
            self.graph.invoke({"ticket_id": ticket_id, "masked_text": masked.text}, config)
            state = self.graph.get_state(config).values
        result = self._to_result(state, masked.entities)
        self._save(result)
        return result

    def review(
        self, ticket_id: str, action: str, reply: str | None = None, note: str | None = None
    ):
        current = self.get(ticket_id)
        if current is None:
            raise KeyError(ticket_id)
        if current.status != "pending_review":
            raise ValueError(f"Ticket {ticket_id} is already {current.status}")
        config = {"configurable": {"thread_id": ticket_id}}
        with self._lock:
            self.graph.invoke(
                Command(resume={"action": action, "reply": reply, "note": note}), config
            )
            state = self.graph.get_state(config).values
        result = self._to_result(state, current.pii)
        self._save(result)
        return result

    def get(self, ticket_id: str) -> TriageResult | None:
        row = self.conn.execute("SELECT payload FROM tickets WHERE id=?", (ticket_id,)).fetchone()
        return TriageResult.model_validate_json(row[0]) if row else None

    def list(self, status: str | None = None) -> list[dict]:
        q = "SELECT created_at, payload FROM tickets"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY CASE priority WHEN 'P1' THEN 0 WHEN 'P2' THEN 1 ELSE 2 END, created_at"
        out = []
        for created_at, payload in self.conn.execute(q, args).fetchall():
            d = json.loads(payload)
            d["created_at"] = created_at
            out.append(d)
        return out


@lru_cache
def get_service() -> TriageService:
    return TriageService()