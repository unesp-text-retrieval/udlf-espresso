from __future__ import annotations
from typing import List, Dict, Any


class RetrievalResult:
    def __init__(self, query_id: str, hits: List[Dict[str, Any]]):
        self.query_id = query_id
        self.hits = hits  # each hit: {"doc_id": str, "score": float}


class Retriever:
    def retrieve(self, queries: List[Dict[str, str]]) -> List[RetrievalResult]:  # queries: {id,text}
        raise NotImplementedError
