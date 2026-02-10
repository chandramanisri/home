from __future__ import annotations

import joblib
from pathlib import Path
from typing import Iterable, List

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"


class IncrementalSentimentModel:
    """Memory-safe sentiment model for very large datasets."""

    def __init__(self) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=2**20,
            alternate_sign=False,
            ngram_range=(1, 2),
            norm="l2",
        )
        self.classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-6,
            random_state=42,
            max_iter=1,
            learning_rate="optimal",
            warm_start=True,
        )
        self._is_fitted = False

    def partial_fit(self, texts: List[str], labels: List[int]) -> None:
        if not texts:
            return
        features = self.vectorizer.transform(texts)
        classes = [0, 1]
        if not self._is_fitted:
            self.classifier.partial_fit(features, labels, classes=classes)
            self._is_fitted = True
        else:
            self.classifier.partial_fit(features, labels)

    def predict_proba(self, texts: Iterable[str]) -> List[float]:
        features = self.vectorizer.transform(texts)
        probabilities = self.classifier.predict_proba(features)
        return [float(row[1]) for row in probabilities]

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": self.classifier, "is_fitted": self._is_fitted}, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "IncrementalSentimentModel":
        data = joblib.load(path)
        model = cls()
        model.classifier = data["classifier"]
        model._is_fitted = data.get("is_fitted", True)
        return model


def sentiment_label(score: float) -> str:
    if score >= 0.7:
        return "very_positive"
    if score >= 0.55:
        return "positive"
    if score <= 0.3:
        return "very_negative"
    if score <= 0.45:
        return "negative"
    return "neutral"
