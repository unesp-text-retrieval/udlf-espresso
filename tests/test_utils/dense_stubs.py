"""Helpers to stub dense retrieval dependencies for tests."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Sequence

import numpy as np  # type: ignore[import-not-found]

CATEGORY_KEYWORDS = {
    "energy": ("solar", "wind", "renewable", "energy", "battery"),
    "health": ("cancer", "therapy", "medical", "health", "clinic"),
    "space": ("space", "mars", "satellite", "mission", "rocket"),
}


def _keyword_vector(text: str, model_name: str) -> np.ndarray:
    text_l = text.lower()
    counts = [float(sum(text_l.count(word) for word in words)) for words in CATEGORY_KEYWORDS.values()]
    tokens = text_l.split()
    total_words = float(len(tokens) or 1)
    unique_words = float(len(set(tokens)) or 1)
    model_bias = float(abs(hash(model_name)) % 7) / 100.0 + 1.0
    return np.array(counts + [total_words, unique_words, model_bias], dtype=np.float32)


class StubSentenceTransformer:
    def __init__(self, model_name: str, **_: object) -> None:
        self.model_name = model_name

    def encode(self, texts: Sequence[str], **_: object) -> np.ndarray:
        return np.vstack([_keyword_vector(t, self.model_name) for t in texts]).astype(np.float32)


class _BaseFakeIndex:
    def __init__(self, d: int, metric: str) -> None:
        self.metric = metric
        self.d = d
        self.embeddings = np.zeros((0, d), dtype=np.float32)
        self.ntotal = 0

    def add(self, embeddings: np.ndarray) -> None:
        arr = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if arr.shape[1] != self.d:
            raise ValueError("Embedding dimension mismatch")
        if self.ntotal == 0:
            self.embeddings = arr.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, arr])
        self.ntotal = self.embeddings.shape[0]

    def search(self, queries: np.ndarray, top_k: int):
        q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        if self.metric == "ip":
            scores = q @ self.embeddings.T
            order = np.argsort(-scores, axis=1)[:, :top_k]
            distances = np.take_along_axis(scores, order, axis=1)
        else:
            diff = q[:, None, :] - self.embeddings[None, :, :]
            sq = np.sum(diff**2, axis=2)
            order = np.argsort(sq, axis=1)[:, :top_k]
            distances = np.take_along_axis(sq, order, axis=1)
        return distances, order


class FakeFaissModule:
    def __init__(self) -> None:
        self._threads: int | None = None

    class IndexFlatIP(_BaseFakeIndex):
        def __init__(self, d: int) -> None:
            super().__init__(d, metric="ip")

    class IndexFlatL2(_BaseFakeIndex):
        def __init__(self, d: int) -> None:
            super().__init__(d, metric="l2")

    @staticmethod
    def normalize_L2(arr: np.ndarray) -> None:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        arr /= norms

    def write_index(self, index: _BaseFakeIndex, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"embeddings": index.embeddings, "metric": index.metric}, fh)

    def read_index(self, path: str) -> _BaseFakeIndex:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        emb = np.asarray(payload["embeddings"], dtype=np.float32)
        metric = str(payload["metric"])
        idx = self.IndexFlatIP(emb.shape[1]) if metric == "ip" else self.IndexFlatL2(emb.shape[1])
        idx.add(emb)
        return idx

    def omp_set_num_threads(self, threads: int) -> None:  # pragma: no cover
        self._threads = int(threads)


@dataclass
class DenseTestEnvironment:
    fake_faiss: FakeFaissModule


def patch_dense_environment(monkeypatch) -> DenseTestEnvironment:
    fake_faiss = FakeFaissModule()
    monkeypatch.setattr("udlf_text_espresso.retrieval.dense.SentenceTransformer", StubSentenceTransformer)
    monkeypatch.setattr("udlf_text_espresso.retrieval.dense.get_faiss", lambda: fake_faiss)
    monkeypatch.setattr("udlf_text_espresso.retrieval.dense._faiss_module", fake_faiss, raising=False)
    monkeypatch.setattr("udlf_text_espresso.runner.get_faiss", lambda: fake_faiss)
    return DenseTestEnvironment(fake_faiss=fake_faiss)