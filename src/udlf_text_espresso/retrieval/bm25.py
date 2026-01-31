from __future__ import annotations
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi  # type: ignore
import logging
import time
import hashlib
import pickle
from pathlib import Path

from .base import Retriever, RetrievalResult

logger = logging.getLogger("udlf_text_espresso")


class BM25Retriever(Retriever):
    """
    Simple BM25 retriever using rank_bm25 with optional persistence.

    Corpus format: list of {"id": str, "text": str}
    Cache format: pickle of {"tokenized": List[List[str]], "bm25": BM25Okapi, "doc_ids": List[str]}
    """

    def __init__(self, corpus: List[Dict[str, str]], k: int = 1000, cache_path: Optional[str] = None):
        self.docs = corpus
        self.doc_ids = [d["id"] for d in self.docs]
        self.k = k
        self.cache_path = cache_path

        # Attempt to load from cache if provided and matches corpus hash
        loaded = False
        if cache_path:
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                if self._corpus_hash(self.docs) == data.get("corpus_hash"):
                    self.bm25 = data["bm25"]
                    self.tokenized_corpus = data["tokenized"]
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            self.tokenized_corpus = [d["text"].split() for d in self.docs]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            if cache_path:
                try:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "wb") as f:
                        pickle.dump({
                            "bm25": self.bm25,
                            "tokenized": self.tokenized_corpus,
                            "doc_ids": self.doc_ids,
                            "corpus_hash": self._corpus_hash(self.docs),
                        }, f)
                except Exception:
                    # best-effort caching; ignore failures
                    pass

    def _corpus_hash(self, corpus: List[Dict[str, str]]) -> str:
        h = hashlib.sha256()
        for d in corpus:
            h.update(d["id"].encode())
            h.update(d["text"].encode())
        return h.hexdigest()

    def retrieve(self, queries: List[Dict[str, str]]) -> List[RetrievalResult]:
        """Rank every document for each query and return the top ``k`` hits.

        The underlying BM25 scorer evaluates the full corpus for every query. As a
        result we always slice the ranked list to ``k`` entries, even when none of
        the documents share vocabulary with the query; those hits simply carry
        zero-valued scores. Downstream callers can therefore rely on a
        deterministic ``k``-sized result set per query.
        """
        results: List[RetrievalResult] = []
        total_queries = len(queries)
        logger.info(
            "BM25Retriever: scoring %d queries against %d documents (k=%d)",
            total_queries,
            len(self.doc_ids),
            self.k,
        )
        start = time.perf_counter()
        log_interval = total_queries // 10 or 1
        log_interval = min(log_interval, 2000)
        for idx, q in enumerate(queries, start=1):
            tokens = q["text"].split()
            scores = self.bm25.get_scores(tokens)
            ranked = sorted(
                ((self.doc_ids[i], float(scores[i])) for i in range(len(self.doc_ids))),
                key=lambda x: x[1],
                reverse=True,
            )
            top = ranked[: self.k]
            hit_list = [{"doc_id": doc_id, "score": score} for doc_id, score in top]
            results.append(RetrievalResult(query_id=q["id"], hits=hit_list))
            if idx % log_interval == 0 or idx == total_queries:
                elapsed = time.perf_counter() - start
                pct = (idx / total_queries * 100.0) if total_queries else 100.0
                logger.info(
                    "BM25Retriever: processed %d/%d queries (%.1f%%) in %.1fs",
                    idx,
                    total_queries,
                    pct,
                    elapsed,
                )
        return results
