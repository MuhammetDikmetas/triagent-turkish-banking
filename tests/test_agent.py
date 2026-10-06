"""Agent behaviour tests. Uses a fake LLM, so they run offline in CI."""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from triagent.classifier import FastClassifier
from triagent.data import load_split
from triagent.graph import build_graph
from triagent.llm import LLMClient, PIILeakError
from triagent.replies import check_draft
from triagent.rules import apply_rules
from triagent.schemas import LLMTriage


@pytest.fixture(scope="module")
def classifier():
    rows = load_split("train")
    return FastClassifier.train([r["masked_text"] for r in rows], [r["category"] for r in rows])


class FakeLLM(LLMClient):
    def __init__(self, draft_text="Sayın Müşterimiz, talebiniz incelenecektir."):
        self.calls = 0
        self.draft_text = draft_text

    @property
    def enabled(self):
        return True

    def triage(self, masked_text):
        self.calls += 1
        return LLMTriage(category="TRANSFER", priority="P3", summary="özet", reason="test")

    def draft(self, masked_text, category, priority):
        return self.draft_text


def run(graph, text, thread="t1"):
    cfg = {"configurable": {"thread_id": thread}}
    graph.invoke({"ticket_id": thread, "masked_text": text}, cfg)
    return graph, cfg


def test_confident_message_skips_llm(classifier):
    llm = FakeLLM()
    g = build_graph(classifier, llm, threshold=0.0, checkpointer=MemorySaver())
    g, cfg = run(g, "Şubede çok bekledim, personel ilgisizdi.")
    assert llm.calls == 0
    assert g.get_state(cfg).values["decided_by"] == "classifier"


def test_unsure_message_goes_to_llm_and_rules_keep_more_urgent_priority(classifier):
    llm = FakeLLM()
    g = build_graph(classifier, llm, threshold=1.01, checkpointer=MemorySaver())
    g, cfg = run(g, "Dün yaptığım EFT karşı tarafa geçmedi")
    state = g.get_state(cfg).values
    assert llm.calls == 1 and state["decided_by"] == "llm"
    assert state["priority"] == "P2"  # LLM said P3, rule (funds stuck) says P2 -> P2 wins


def test_graph_pauses_for_human_and_resumes(classifier):
    g = build_graph(classifier, FakeLLM(), threshold=0.5, checkpointer=MemorySaver())
    g, cfg = run(g, "Kartım çalındı hemen kapatın")
    snap = g.get_state(cfg)
    assert snap.next == ("human_review",)
    assert snap.values["status"] == "pending_review"
    assert snap.values["category"] == "FRAUD" and snap.values["priority"] == "P1"

    g.invoke(Command(resume={"action": "edit", "reply": "Kartınız kapatıldı."}), cfg)
    final = g.get_state(cfg).values
    assert final["status"] == "approved" and final["final_reply"] == "Kartınız kapatıldı."


def test_unsafe_llm_draft_is_replaced_by_template(classifier):
    bad = FakeLLM(draft_text="Sayın Müşterimiz, lütfen SMS onay kodunu bize gönderin.")
    g = build_graph(classifier, bad, threshold=0.5, checkpointer=MemorySaver())
    g, cfg = run(g, "Uygulamaya giriş yapamıyorum")
    state = g.get_state(cfg).values
    assert state["draft_source"].startswith("template")
    assert "onay kodunu bize" not in state["draft_reply"]


def test_llm_client_refuses_unmasked_pii():
    client = LLMClient()
    with pytest.raises(PIILeakError):
        client._chat("sys", "TC numaram 10000000146", json_mode=False, kind="x")


@pytest.mark.parametrize(
    "text,category,expected",
    [
        ("kartim calindi hemen kapatin", "CARD", ("FRAUD", "P1")),
        ("Sahte SMS aldım, linke tıklamadım, bildirmek istedim", "FRAUD", (None, "P2")),
        ("ATM paramı vermedi hesabımdan düştü", "ATM", (None, "P2")),
        ("Aidatı iade etmezseniz BDDK'ya gideceğim", "FEES", (None, "P2")),
        ("Kredi faiz oranlarınız nedir?", "LOAN", (None, "P3")),
    ],
)
def test_rules(text, category, expected):
    out = apply_rules(text, category)
    assert (out.category_override, out.priority) == expected


def test_draft_guardrail_allows_good_security_advice():
    assert check_draft("Şifrenizi kimseyle paylaşmayınız.") == []
    assert check_draft("Lütfen şifrenizi bize iletin.") != []
