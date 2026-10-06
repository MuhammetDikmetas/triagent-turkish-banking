"""Evaluate Triagent on the held-out test set.

Measures
  1. Category classification : accuracy, macro-F1, FRAUD recall, LLM call rate, latency
  2. Priority                : accuracy, P1 recall (missing an urgent case is the costly error)
  3. Routing threshold curve : accuracy vs share of messages sent to the LLM
  4. PII masking             : precision / recall per identifier type (+ decoys)

Systems compared (whatever is available is run; LLM rows need an API key in .env):
  classifier            fast model only
  classifier + rules    offline mode (no LLM)
  llm only              every message goes to the LLM
  hybrid + rules        Triagent: LLM only below the confidence threshold

Usage: python eval/run_eval.py [--encoder tfidf|e5] [--threshold 0.55]
Outputs: eval/results/results.md, metrics.json, threshold_curve.png, confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    recall_score,
)

from triagent.classifier import FastClassifier  # noqa: E402
from triagent.config import ROOT, get_settings  # noqa: E402
from triagent.data import load_split  # noqa: E402
from triagent.llm import LLMClient  # noqa: E402
from triagent.pii import detect  # noqa: E402
from triagent.rules import apply_rules, more_urgent  # noqa: E402
from triagent.synthetic import (  # noqa: E402
    decoy_number,
    fake_card,
    fake_email,
    fake_iban,
    fake_phone,
    fake_tckn,
)

OUT = ROOT / "eval" / "results"
LABELS = ["CARD", "LOAN", "TRANSFER", "FRAUD", "DIGITAL", "ATM", "FEES", "BRANCH"]


# --------------------------------------------------------------------------- helpers
def cat_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS),
        "fraud_recall": recall_score(y_true, y_pred, labels=["FRAUD"], average="macro"),
    }


def prio_metrics(y_true, y_pred) -> dict:
    return {
        "priority_accuracy": accuracy_score(y_true, y_pred),
        "p1_recall": recall_score(y_true, y_pred, labels=["P1"], average="macro"),
    }


def run_system(rows, cat_pred, llm_prio=None, use_rules=True):
    """Apply the same post-processing as graph.py (rules + 'most urgent wins')."""
    cats, prios = [], []
    for i, r in enumerate(rows):
        cat = cat_pred[i]
        if use_rules:
            o = apply_rules(r["masked_text"], cat)
            cat = o.category_override or cat
            prio = o.priority
        else:  # naive baseline: priority derived from the category alone
            prio = "P1" if cat == "FRAUD" else "P3"
        if llm_prio and llm_prio[i]:
            prio = more_urgent(prio, llm_prio[i])
        cats.append(cat)
        prios.append(prio)
    return cats, prios


# --------------------------------------------------------------------------- PII benchmark
def pii_benchmark(n: int = 400, seed: int = 0) -> dict:
    """Synthetic sentences mixing real-format PII with look-alike decoys."""
    rng = random.Random(seed)
    makers = {
        "TCKN": lambda: fake_tckn(rng),
        "IBAN": lambda: fake_iban(rng, spaced=rng.random() < 0.6),
        "CARD": lambda: fake_card(rng, spaced=rng.random() < 0.6),
        "PHONE": lambda: fake_phone(rng),
        "EMAIL": lambda: fake_email(rng),
    }
    decoys = [
        lambda: f"sipariş no {decoy_number(rng)}",
        lambda: f"referans {rng.randint(10**15, 10**16 - 1)}",
        lambda: f"müşteri no {rng.randint(10**7, 10**8 - 1)}",
        lambda: f"tutar {rng.randint(1, 99)}.{rng.randint(100, 999)} TL",
        lambda: f"tarih {rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.2026",
    ]
    tp, fp, fn = Counter(), Counter(), Counter()
    for _ in range(n):
        parts, gold, text = [], [], ""
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.6:
                kind = rng.choice(list(makers))
                parts.append((kind, makers[kind]()))
            else:
                parts.append((None, rng.choice(decoys)()))
        for kind, value in parts:
            text += rng.choice(["Merhaba, ", "Ayrıca ", "", "bilginize: "])
            if kind:
                gold.append((kind, len(text), len(text) + len(value)))
            text += value + rng.choice([". ", ", ", " ve "])
        pred = {(e.type, e.start, e.end) for e in detect(text)}
        gold_set = set(gold)
        for g in gold_set:
            (tp if g in pred else fn)[g[0]] += 1
        for p in pred - gold_set:
            fp[p[0]] += 1
    out = {}
    for k in makers:
        prec = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 1.0
        rec = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 1.0
        out[k] = {"precision": prec, "recall": rec, "support": tp[k] + fn[k]}
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=s.encoder, choices=["tfidf", "e5"])
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    thr_file = ROOT / "models" / f"threshold_{args.encoder}.json"
    threshold = args.threshold
    if threshold is None:
        threshold = (
            json.loads(thr_file.read_text()).get("threshold", s.confidence_threshold)
            if thr_file.exists()
            else s.confidence_threshold
        )

    train, test = load_split("train"), load_split("test")
    X_tr = [r["masked_text"] for r in train]
    X_te = [r["masked_text"] for r in test]
    y_cat = [r["category"] for r in test]
    y_pri = [r["priority"] for r in test]

    clf = FastClassifier.train(X_tr, [r["category"] for r in train], encoder=args.encoder)
    t0 = time.perf_counter()
    proba = clf.predict_proba(X_te)
    clf_ms = (time.perf_counter() - t0) / len(X_te) * 1000
    classes = np.array(clf.classes)
    clf_pred = list(classes[proba.argmax(1)])
    conf = proba.max(1)

    rows, details = [], {}

    def add(name, cats, prios, llm_rate, latency_ms):
        m = {**cat_metrics(y_cat, cats), **prio_metrics(y_pri, prios)}
        m.update(llm_call_rate=llm_rate, latency_ms=latency_ms)
        rows.append((name, m))
        details[name] = cats

    c, p = run_system(test, clf_pred, use_rules=False)
    add(f"classifier ({args.encoder})", c, p, 0.0, clf_ms)
    c, p = run_system(test, clf_pred)
    add(f"classifier ({args.encoder}) + rules", c, p, 0.0, clf_ms)

    # ---------------------------------------------------------------- LLM systems
    llm = LLMClient(s)
    llm_cat, llm_pri, llm_lat = None, None, None
    if llm.enabled:
        print(f"Calling LLM ({s.llm_provider}/{s.llm_model}) on {len(test)} messages (cached)...")
        llm_cat, llm_pri, lat = [], [], []
        for i, r in enumerate(test):
            t0 = time.perf_counter()
            res = llm.triage(r["masked_text"])
            lat.append(time.perf_counter() - t0)
            llm_cat.append(res.category.value if res else clf_pred[i])
            llm_pri.append(res.priority.value if res else None)
            print(f"\r  {i + 1}/{len(test)}", end="", file=sys.stderr)
        print(file=sys.stderr)
        # latency from cache files = real API latency even on cached re-runs
        cached = [
            json.loads(f.read_text(encoding="utf-8"))["latency_s"]
            for f in (ROOT / "eval/cache").glob("*")
        ]
        llm_lat = float(np.median(cached) * 1000) if cached else float(np.median(lat) * 1000)

        c, p = run_system(test, llm_cat, llm_pri, use_rules=False)
        add("llm only", c, p, 1.0, llm_lat)
        c, p = run_system(test, llm_cat, llm_pri)
        add("llm + rules", c, p, 1.0, llm_lat)

        routed = conf < threshold
        hy_cat = [llm_cat[i] if routed[i] else clf_pred[i] for i in range(len(test))]
        hy_pri = [llm_pri[i] if routed[i] else None for i in range(len(test))]
        c, p = run_system(test, hy_cat, hy_pri)
        add(
            f"hybrid + rules (τ={threshold:.2f})  ← Triagent",
            c,
            p,
            float(routed.mean()),
            clf_ms + routed.mean() * llm_lat,
        )
    else:
        print("LLM disabled -> skipping LLM rows. Set LLM_PROVIDER / LLM_API_KEY in .env.")

    # ---------------------------------------------------------------- threshold curve
    taus = np.round(np.arange(0.0, 1.0001, 0.02), 2)
    curve = []
    clf_correct = np.array(clf_pred) == np.array(y_cat)
    for tau in taus:
        routed = conf < tau
        row = {"tau": float(tau), "llm_rate": float(routed.mean())}
        acc_accept = clf_correct[~routed].mean() if (~routed).any() else np.nan
        row["accuracy_on_auto_accepted"] = float(acc_accept)
        if llm_cat:
            hy = [llm_cat[i] if routed[i] else clf_pred[i] for i in range(len(test))]
            row["hybrid_accuracy"] = accuracy_score(y_cat, hy)
        curve.append(row)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    curve_plot = [r for r in curve if (1 - r["llm_rate"]) * len(test) >= 10]  # >=10 msgs kept
    rates = [r["llm_rate"] * 100 for r in curve_plot]
    ax.plot(
        rates,
        [r["accuracy_on_auto_accepted"] * 100 for r in curve_plot],
        label="classifier accuracy on messages it keeps",
    )
    if llm_cat:
        ax.plot(
            rates,
            [r["hybrid_accuracy"] * 100 for r in curve_plot],
            label="hybrid overall accuracy",
            lw=2.4,
        )
    chosen = min(curve, key=lambda r: abs(r["tau"] - threshold))
    ax.axvline(chosen["llm_rate"] * 100, ls="--", c="gray", lw=1)
    ax.annotate(
        f"τ={threshold:.2f}", (chosen["llm_rate"] * 100, ax.get_ylim()[0] + 2), color="gray"
    )
    ax.set_xlabel("% of messages sent to the LLM")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Cost / accuracy trade-off of confidence routing")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "threshold_curve.png", dpi=150)

    # ---------------------------------------------------------------- confusion matrix
    best_name = rows[-1][0]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_cat,
        details[best_name],
        labels=LABELS,
        ax=ax,
        colorbar=False,
        xticks_rotation=45,
        cmap="Blues",
    )
    ax.set_title(best_name.split("  ")[0])
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)

    # ---------------------------------------------------------------- PII
    pii = pii_benchmark()
    gold_spans = {(r["id"], p["type"], p["start"], p["end"]) for r in test for p in r["pii"]}
    pred_spans = {(r["id"], e.type, e.start, e.end) for r in test for e in detect(r["text"])}
    pii_testset = {
        "gold": len(gold_spans),
        "found": len(gold_spans & pred_spans),
        "false_positives": len(pred_spans - gold_spans),
    }

    # ---------------------------------------------------------------- report
    md = [
        "# Evaluation results",
        "",
        f"Test set: **{len(test)} messages** (8 categories × 20), held out from training.  ",
        f"Encoder: `{args.encoder}` · routing threshold τ = **{threshold:.2f}** (chosen by 5-fold CV on train)",
        "",
        "## Classification & priority",
        "",
        "| System | Accuracy | Macro-F1 | FRAUD recall | Priority acc. | P1 recall | LLM calls | Latency/msg |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        md.append(
            f"| {name} | {m['accuracy']:.1%} | {m['macro_f1']:.3f} | {m['fraud_recall']:.1%} | "
            f"{m['priority_accuracy']:.1%} | {m['p1_recall']:.1%} | {m['llm_call_rate']:.0%} | "
            f"{m['latency_ms']:.1f} ms |"
        )
    md += [
        "",
        "## PII masking",
        "",
        "Synthetic benchmark: 400 sentences mixing checksum-valid identifiers with look-alike "
        "decoys (order numbers, 16-digit references, customer numbers, amounts, dates).",
        "",
        "| Type | Precision | Recall | Support |",
        "|---|---|---|---|",
    ]
    for k, v in pii.items():
        md.append(f"| {k} | {v['precision']:.1%} | {v['recall']:.1%} | {v['support']} |")
    md += [
        "",
        f"On the test set: {pii_testset['found']}/{pii_testset['gold']} PII spans masked, "
        f"{pii_testset['false_positives']} false positives (test set includes decoy reference numbers).",
        "",
        "![threshold](threshold_curve.png)",
        "![confusion](confusion_matrix.png)",
    ]
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "metrics.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "systems": dict(rows),
                "pii": pii,
                "pii_testset": pii_testset,
                "curve": curve,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n".join(md[: 12 + len(rows)]))
    print(f"\nSaved -> {OUT.relative_to(ROOT)}/results.md")


if __name__ == "__main__":
    main()
