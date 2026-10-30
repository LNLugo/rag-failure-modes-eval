"""Retrievers: dense, sparse (BM25) and hybrid (reciprocal rank fusion).

Dense backend
  Default is LSA (TF-IDF + truncated SVD) so the lab runs fully offline. LSA has the
  same weakness as neural embeddings that this lab is built to show: compressing
  text into a few hundred dimensions blurs rare tokens such as form codes and
  policy numbers. If sentence-transformers is installed and a model can be
  downloaded, set RAG_LAB_EMBEDDER=st:BAAI/bge-small-en-v1.5 to use a real one.
"""
from __future__ import annotations

import os
import re

import numpy as np

TOKEN = re.compile(r"[a-z0-9$]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dims: int = 24):
        self.dims = dims

    def fit(self, texts: list[str]):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Letters-only tokens: the offline stand-in deliberately cannot see "7b" in
        # "KYC-7B", imitating how embedding models blur alphanumeric identifiers.
        # Run with a real model (RAG_LAB_EMBEDDER=st:...) to see the real-world effect.
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.dims, X.shape[1] - 1, X.shape[0] - 1), random_state=0)
        return self._norm(self.svd.fit_transform(X))

    def encode(self, texts: list[str]):
        return self._norm(self.svd.transform(self.vec.transform(texts)))

    @staticmethod
    def _norm(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


class STEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model}"
        self.m = SentenceTransformer(model)

    def fit(self, texts):
        return self.encode(texts)

    def encode(self, texts):
        return self.m.encode(texts, normalize_embeddings=True)


def make_embedder():
    spec = os.environ.get("RAG_LAB_EMBEDDER", "lsa")
    return STEmbedder(spec[3:]) if spec.startswith("st:") else LSAEmbedder()


class DenseRetriever:
    def __init__(self, texts: list[str]):
        self.emb = make_embedder()
        self.M = self.emb.fit(texts)

    def scores(self, query: str) -> np.ndarray:
        return self.M @ self.emb.encode([query])[0]


class BM25Retriever:
    def __init__(self, texts: list[str]):
        from rank_bm25 import BM25Okapi

        self.bm = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm.get_scores(tokenize(query)))


class HybridRetriever:
    """Reciprocal rank fusion: robust because it ignores the two scorers' different scales."""

    def __init__(self, texts: list[str], k: int = 60):
        self.dense, self.sparse, self.k = DenseRetriever(texts), BM25Retriever(texts), k

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(self.dense.M))
        for s in (self.dense.scores(query), self.sparse.scores(query)):
            ranks = np.empty_like(s)
            ranks[np.argsort(-s)] = np.arange(len(s))
            fused += 1.0 / (self.k + ranks + 1)
        return fused
