from __future__ import annotations
from typing import List, Dict, Any
from ..retrieval.base import RetrievalResult


def to_udlf_format(results: List[RetrievalResult]) -> List[Dict[str, Any]]:
    unified: List[Dict[str, Any]] = []
    for rr in results:
        for rank, hit in enumerate(rr.hits, start=1):
            unified.append({
                "query_id": rr.query_id,
                "doc_id": hit["doc_id"],
                "rank": rank,
                "score": hit["score"],
                # placeholder for features vector expected by UDLF if needed
                "features": []
            })
    return unified
