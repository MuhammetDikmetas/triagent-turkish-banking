"""Fast, CPU-only category classifier.

Two interchangeable encoders:

* ``tfidf`` - word 1-2 grams + character 2-5 grams on Turkish-folded text. Character
  n-grams matter for Turkish: it is agglutinative, so "kartım", "kartımı", "kartımdan"
  share the sub-word "kart" even though they are different tokens.
* ``e5``    - multilingual-e5-small sentence embeddings (118M params, runs on CPU).

Both feed a multinomial Logistic Regression. ``predict_proba`` gives the confidence used
by the router to decide whether a message needs the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from triagent.text import fold

E5_MODEL = "intfloat/multilingual-e5-small"


def _fold_all(texts):
    return [fold(t) for t in texts]


class E5Encoder:
    """sklearn-compatible transformer around sentence-transformers (lazy-loaded)."""

    def __init__(self, model_name: str = E5_MODEL):
        self.model_name = model_name
        self._model = None

    def __getstate__(self):  # never pickle the torch model itself
        return {"model_name": self.model_name, "_model": None}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = [f"query: {t}" for t in X]  # e5 models expect this prefix
        return self._load().encode(texts, normalize_embeddings=True, batch_size=32)


def build_pipeline(encoder: str = "tfidf", C: float = 10.0) -> Pipeline:
    if encoder == "tfidf":
        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                (
                    "char",
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                ),
            ]
        )
        steps = [("fold", FunctionTransformer(_fold_all)), ("features", features)]
    elif encoder == "e5":
        steps = [("features", E5Encoder())]
    else:
        raise ValueError(f"Unknown encoder: {encoder}")
    steps.append(("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")))
    return Pipeline(steps)


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class FastClassifier:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_)

    @classmethod
    def train(cls, texts: list[str], labels: list[str], encoder: str = "tfidf") -> FastClassifier:
        pipe = build_pipeline(encoder)
        pipe.fit(texts, labels)
        return cls(pipe)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, text: str) -> Prediction:
        proba = self.predict_proba([text])[0]
        best = int(np.argmax(proba))
        return Prediction(
            label=self.classes[best],
            confidence=float(proba[best]),
            probabilities={c: round(float(p), 4) for c, p in zip(self.classes, proba, strict=True)},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> FastClassifier:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Train it first: python scripts/train.py")
        return cls(joblib.load(path))