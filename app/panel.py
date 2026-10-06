"""Agent console (Streamlit).  Run: streamlit run app/panel.py"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from triagent.config import get_settings
from triagent.schemas import CATEGORY_TR, PRIORITY_TR
from triagent.service import TriageService

st.set_page_config(page_title="Triagent", page_icon="🏦", layout="wide")

PRIORITY_BADGE = {"P1": "🔴 Acil", "P2": "🟠 Yüksek", "P3": "🟢 Normal"}
FLAG_TR = {
    "fraud_signal": "🚨 Dolandırıcılık sinyali",
    "legal_threat": "⚖️ Yasal yol / BDDK",
    "financial_hardship": "💸 Ödeme güçlüğü",
    "churn_risk": "🚪 Müşteri kaybı riski",
    "vulnerable_customer": "🧓 Hassas müşteri",
}
STATUS_TR = {
    "pending_review": "⏳ Onay bekliyor",
    "approved": "✅ Gönderildi",
    "rejected": "❌ Reddedildi",
}


def tl(x: float) -> str:
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " TL"


EXAMPLES = [
    "Kartım çalındı!! TC kimlik numaram 10000000146, hemen kapatın, 2 tane 1.500 TL işlem var.",
    "Dün yaptığım EFT hala karşı tarafa geçmedi, 12.000 TL, IBAN TR33 0006 1005 1978 6457 8413 26",
    "ATM kartımı yuttu, akşam saati, şube kapalı ne yapacağım",
    "Kart aidatım iade edilmezse BDDK'ya şikayet edeceğim, 3 kere başvurdum.",
    "Mobil uygulama güncellemeden sonra açılmıyor, beyaz ekranda kalıyor.",
    "Şubede 2 saat bekledim, tek gişe açıktı, yaşlılar ayakta bekliyordu.",
]


@st.cache_resource
def service() -> TriageService:
    return TriageService()


svc = service()
settings = get_settings()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🏦 Triagent")
    st.caption("Bankacılık müşteri talepleri için KVKK uyumlu yönlendirme ajanı")
    mode = settings.llm_provider if settings.llm_enabled else "offline"
    st.markdown(
        f"**LLM:** `{mode}` &nbsp; **Encoder:** `{settings.encoder}`  \n"
        f"**Güven eşiği:** `{settings.confidence_threshold}`"
    )
    st.divider()
    st.subheader("Yeni müşteri mesajı")
    example = st.selectbox("Örnek seç (isteğe bağlı)", ["—"] + EXAMPLES)
    text = st.text_area("Mesaj", value="" if example == "—" else example, height=140)
    if st.button("Analiz et", type="primary", use_container_width=True, disabled=len(text) < 3):
        with st.spinner("Maskeleniyor, sınıflandırılıyor, taslak hazırlanıyor..."):
            ticket = svc.submit(text)
        st.session_state["selected"] = ticket.ticket_id
        st.success(f"Talep oluşturuldu: {ticket.ticket_id}")

tickets = svc.list()
tab_queue, tab_detail, tab_stats = st.tabs(["📥 Kuyruk", "📝 İnceleme", "📊 İstatistik"])

# ------------------------------------------------------------------ queue
with tab_queue:
    pending = [t for t in tickets if t["status"] == "pending_review"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Onay bekleyen", len(pending))
    c2.metric("Acil (P1)", sum(t["priority"] == "P1" for t in pending))
    c3.metric("Toplam talep", len(tickets))
    if not tickets:
        st.info("Henüz talep yok. Soldan bir mesaj gönderin.")
    for t in tickets:
        cols = st.columns([1.2, 2, 5, 1.4, 1])
        cols[0].markdown(PRIORITY_BADGE[t["priority"]])
        cols[1].markdown(f"**{CATEGORY_TR[t['category']]}**")
        cols[2].write(t["masked_text"][:110] + ("…" if len(t["masked_text"]) > 110 else ""))
        cols[3].caption(STATUS_TR[t["status"]])
        if cols[4].button("Aç", key=f"open-{t['ticket_id']}"):
            st.session_state["selected"] = t["ticket_id"]
            st.rerun()

# ------------------------------------------------------------------ detail / review
with tab_detail:
    sel = st.session_state.get("selected")
    t = svc.get(sel) if sel else None
    if t is None:
        st.info("Kuyruktan bir talep seçin.")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.subheader(f"Talep {t.ticket_id}")
            st.markdown("**Maskelenmiş mesaj** (LLM'e yalnızca bu metin gider)")
            st.code(t.masked_text, language=None, wrap_lines=True)
            if t.pii:
                found = ", ".join(f"{e.type}" for e in t.pii)
                st.caption(f"🔒 Maskelenen kişisel veri: {found}")
            if t.decided_by == "llm":
                st.markdown(f"**LLM özeti:** {t.summary}")
        with right:
            st.markdown(f"### {PRIORITY_BADGE[t.priority.value]}")
            st.markdown(f"**Kategori:** {t.category_tr}  \n**Yönlendirilen ekip:** {t.team}")
            st.progress(t.confidence, text=f"Sınıflandırıcı güveni: {t.confidence:.0%}")
            st.caption(f"Karar veren: `{t.decided_by}`")
            st.markdown("**Öncelik gerekçesi**")
            for r in t.priority_reasons:
                st.markdown(f"- {r}")
            if t.flags:
                st.markdown(" ".join(FLAG_TR.get(f, f) for f in t.flags))
            if t.amounts_tl:
                st.markdown("**Tutar:** " + ", ".join(tl(a) for a in t.amounts_tl))

        st.divider()
        if t.status == "pending_review":
            st.markdown("**Cevap taslağı** — onaylamadan müşteriye gönderilmez")
            reply = st.text_area("Taslak", t.draft_reply, height=180, key=f"reply-{t.ticket_id}")
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("✅ Onayla ve gönder", type="primary"):
                action = "approve" if reply == t.draft_reply else "edit"
                svc.review(t.ticket_id, action, reply=reply)
                st.rerun()
            if b2.button("❌ Reddet"):
                svc.review(t.ticket_id, "reject")
                st.rerun()
        else:
            st.markdown(f"**Durum:** {STATUS_TR[t.status]}")
            if t.final_reply:
                st.text(t.final_reply)

# ------------------------------------------------------------------ stats
with tab_stats:
    if not tickets:
        st.info("İstatistik için talep oluşturun.")
    else:
        df = pd.DataFrame(tickets)
        df["Kategori"] = df["category"].map(CATEGORY_TR)
        df["Öncelik"] = df["priority"].map(PRIORITY_TR)
        llm_share = (df["decided_by"] == "llm").mean()
        pii_total = int(df["pii"].map(len).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("LLM'e giden talep oranı", f"{llm_share:.0%}")
        m2.metric("Maskelenen kişisel veri", pii_total)
        m3.metric("Onaylanan cevap", int((df["status"] == "approved").sum()))
        a, b = st.columns(2)
        a.markdown("**Kategoriye göre**")
        a.bar_chart(df["Kategori"].value_counts(), horizontal=True)
        b.markdown("**Önceliğe göre**")
        b.bar_chart(df["Öncelik"].value_counts())
