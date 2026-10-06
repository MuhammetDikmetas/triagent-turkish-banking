"""Train the fast classifier and pick a routing threshold *without touching the test set*.

The threshold is chosen from 5-fold out-of-fold probabilities on the training data:
the lowest threshold whose auto-accepted messages still reach the target accuracy.
(Choosing it on the test set would leak test information into the system.)

Usage: python scripts/train.py [--encoder tfidf|e5] [--target 0.95]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from triagent.classifier import FastClassifier, build_pipeline
from triagent.config import ROOT, get_settings
from triagent.data import load_split


def choose_threshold(proba: np.ndarray, y: np.ndarray, classes: np.ndarray, target: float):
    pred = classes[proba.argmax(1)]
    conf = proba.max(1)
    best = None
    for tau in np.round(np.arange(0.0, 1.0, 0.01), 2):
        accepted = conf >= tau
        if accepted.sum() == 0:
            break
        acc = float((pred[accepted] == y[accepted]).mean())
        if acc >= target:
            best = {
                "threshold": float(tau),
                "auto_accept_rate": float(accepted.mean()),
                "accuracy_on_accepted": acc,
            }
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=get_settings().encoder, choices=["tfidf", "e5"])
    ap.add_argument(
        "--target", type=float, default=0.95, help="accuracy required on auto-accepted messages"
    )
    args = ap.parse_args()

    rows = load_split("train")
    X = [r["masked_text"] for r in rows]
    y = np.array([r["category"] for r in rows])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pipe = build_pipeline(args.encoder)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(y)))
    cv_acc = float((classes[proba.argmax(1)] == y).mean())
    print(f"[{args.encoder}] 5-fold CV accuracy on train: {cv_acc:.3f}")

    choice = choose_threshold(proba, y, classes, args.target)
    print(f"Recommended CONFIDENCE_THRESHOLD (target {args.target:.0%} accuracy): {choice}")

    model = FastClassifier.train(X, list(y), encoder=args.encoder)
    out = ROOT / "models" / f"classifier_{args.encoder}.joblib"
    model.save(out)
    (ROOT / "models" / f"threshold_{args.encoder}.json").write_text(
        json.dumps({"cv_accuracy": cv_acc, **(choice or {})}, indent=2)
    )
    print(f"Saved model -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
