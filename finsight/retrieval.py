"""Text search: dense embeddings, keyword search (BM25), and a hybrid of both."""
import json
import re

import numpy as np

import config

_STOP = set("the a an of to in for and or on at by is was were be as with from that this what how much did its it".split())


def tokenize(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower()) if t not in _STOP]


class Embedder:
    def __init__(self, name: str = config.EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)

    def docs(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=True).astype(np.float32)

    def query(self, q: str) -> np.ndarray:
        return self.model.encode([config.EMBED_QUERY_PREFIX + q], normalize_embeddings=True)[0].astype(np.float32)


def rrf(rankings: list, k: int = 60) -> dict:
    """Reciprocal rank fusion: combine ranked lists without tuning score scales."""
    fused = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + r + 1)
    return fused


class TextIndex:
    """One index per (chunking scheme, company)."""

    def __init__(self, chunks: list, emb: np.ndarray):
        from rank_bm25 import BM25Okapi
        self.chunks, self.emb = chunks, emb
        self.bm25 = BM25Okapi([tokenize(c["text"]) or ["_"] for c in chunks])

    @classmethod
    def build(cls, chunks: list, embedder) -> "TextIndex":
        return cls(chunks, embedder.docs([c["text"] for c in chunks]))

    def save(self, scheme: str, ticker: str) -> None:
        d = config.INDEX / scheme
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{ticker}.json").write_text(json.dumps(self.chunks))
        np.save(d / f"{ticker}.npy", self.emb)

    @classmethod
    def load(cls, scheme: str, ticker: str) -> "TextIndex":
        d = config.INDEX / scheme
        return cls(json.loads((d / f"{ticker}.json").read_text()), np.load(d / f"{ticker}.npy"))

    def search(self, query: str, embedder, k: int = 20, mode: str = "dense", depth: int = 100) -> list:
        dense = self.emb @ embedder.query(query)
        dense_rank = list(np.argsort(-dense)[:depth])
        if mode == "dense":
            order = dense_rank
        elif mode == "hybrid":
            bm = self.bm25.get_scores(tokenize(query))
            fused = rrf([dense_rank, list(np.argsort(-bm)[:depth])])
            order = sorted(fused, key=fused.get, reverse=True)
        else:
            raise ValueError(mode)
        return [{**self.chunks[i], "score": float(dense[i])} for i in order[:k]]


def ranked_pages(hits: list) -> list:
    """Unique pages in the order they first appear."""
    seen, pages = set(), []
    for h in hits:
        if h["page"] not in seen:
            seen.add(h["page"])
            pages.append(h["page"])
    return pages


def fill_context(hits: list, budget: int = config.CONTEXT_CHARS) -> list:
    """Take hits in rank order until the shared character budget is used."""
    out, used = [], 0
    for h in hits:
        text = h["text"]
        if used + len(text) > budget:
            text = text[: max(0, budget - used)]
        if not text:
            break
        out.append({**h, "text": text})
        used += len(text)
    return out
