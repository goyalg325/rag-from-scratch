from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        if not texts:
            texts = ["empty corpus"]
        self.vectorizer.fit(texts)
        self._fitted = True

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            self.fit(["empty corpus"])

    def _dense(self, matrix: np.ndarray) -> list[list[float]]:
        return matrix.astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._ensure_fitted()
        matrix = self.vectorizer.transform(texts).toarray()
        return self._dense(matrix)

    def embed_query(self, text: str) -> list[float]:
        self._ensure_fitted()
        vector = self.vectorizer.transform([text]).toarray()[0]
        return vector.astype(np.float32).tolist()
