"""The Triagent workflow as a LangGraph state machine.

    classify ──(confident)──────────────┐
        └──(unsure)──> llm_classify ────┤
                                        v
                                  apply_rules ─> draft_reply ─> human_review ─> finalize
                                                                  (interrupt)

Important: the graph state only ever contains *masked* text. Raw text is masked in
``service.py`` before the graph is invoked, so no PII reaches the checkpoint database.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from triagent.classifier import FastClassifier
from triagent.llm import LLMClient
from triagent.replies import check_draft, template_reply
from triagent.rules import apply_rules, more_urgent
from triagent.text import extract_amounts


class TriageState(TypedDict, total=False):
    ticket_id: str
    masked_text: str
    # classification
    category: str
    confidence: float
    probabilities: dict[str, float]
    decided_by: str
    llm_priority: str | None
    summary: str
    # rules
    priority: str
    priority_reasons: list[str]
    flags: list[str]
    amounts_tl: list[float]
    # reply
    draft_reply: str
    draft_source: str
    guardrail_issues: list[str]
    # review
    status: str
    final_reply: str | None
    reviewer_note: str | None


def build_graph(classifier: FastClassifier, llm: LLMClient, threshold: float, checkpointer=None):
    # ------------------------------------------------------------------ nodes
    def classify(state: TriageState) -> TriageState:
        p = classifier.predict(state["masked_text"])
        return {
            "category": p.label,
            "confidence": p.confidence,
            "probabilities": p.probabilities,
            "decided_by": "classifier",
            "llm_priority": None,
            "summary": state["masked_text"][:140],
        }

    def route(state: TriageState) -> Literal["llm_classify", "apply_rules"]:
        return "apply_rules" if state["confidence"] >= threshold else "llm_classify"

    def llm_classify(state: TriageState) -> TriageState:
        if not llm.enabled:
            return {"decided_by": "classifier (LLM disabled)"}
        result = llm.triage(state["masked_text"])
        if result is None:
            return {"decided_by": "classifier (LLM failed)"}
        return {
            "category": result.category.value,
            "decided_by": "llm",
            "llm_priority": result.priority.value,
            "summary": result.summary,
        }

    def rules_node(state: TriageState) -> TriageState:
        outcome = apply_rules(state["masked_text"], state["category"])
        category = outcome.category_override or state["category"]
        priority, reasons = outcome.priority, list(outcome.reasons)
        if state.get("llm_priority"):
            # Safety first: when the LLM and the rules disagree, keep the more urgent one.
            priority = more_urgent(priority, state["llm_priority"])
            reasons.append(f"LLM önerisi: {state['llm_priority']}")
        return {
            "category": category,
            "priority": priority,
            "priority_reasons": reasons or ["yükseltme sinyali yok"],
            "flags": outcome.flags,
            "amounts_tl": extract_amounts(state["masked_text"]),
        }

    def draft_reply(state: TriageState) -> TriageState:
        draft, source, issues = None, "template", []
        if llm.enabled:
            draft = llm.draft(state["masked_text"], state["category"], state["priority"])
            if draft:
                issues = check_draft(draft)
                source = "llm"
                if issues:  # unsafe draft -> fall back to an approved template
                    draft, source = None, "template (LLM draft blocked)"
        return {
            "draft_reply": draft or template_reply(state["category"]),
            "draft_source": source,
            "guardrail_issues": issues,
            "status": "pending_review",
        }

    def human_review(state: TriageState) -> Command[Literal["finalize"]]:
        # Execution pauses here and the state is checkpointed. It resumes when an agent
        # submits a decision: {"action": "approve"|"edit"|"reject", "reply": str, "note": str}
        decision = interrupt({"ticket_id": state["ticket_id"], "draft_reply": state["draft_reply"]})
        action = decision.get("action", "approve")
        if action == "reject":
            update = {"status": "rejected", "final_reply": None}
        else:
            reply = decision.get("reply") or state["draft_reply"]
            update = {"status": "approved", "final_reply": reply}
        update["reviewer_note"] = decision.get("note")
        return Command(goto="finalize", update=update)

    def finalize(state: TriageState) -> TriageState:
        # Integration point: send the reply via the bank's CRM / messaging system.
        return {}

    # ------------------------------------------------------------------ wiring
    g = StateGraph(TriageState)
    g.add_node("classify", classify)
    g.add_node("llm_classify", llm_classify)
    g.add_node("apply_rules", rules_node)
    g.add_node("draft_reply", draft_reply)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route)
    g.add_edge("llm_classify", "apply_rules")
    g.add_edge("apply_rules", "draft_reply")
    g.add_edge("draft_reply", "human_review")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)