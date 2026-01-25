from __future__ import annotations
from typing import List, Dict, Optional, Callable
import math

def _extract_category(doc_id: str) -> str:
    """Extract category from document ID like 'sport_001' -> 'sport' or 'doc::sport_001' -> 'sport'"""
    # Strip 'doc::' prefix if present (used in some category-based datasets)
    if doc_id.startswith("doc::"):
        doc_id = doc_id[5:]  # Remove 'doc::' prefix
    if "_" in doc_id:
        return doc_id.split("_", 1)[0]
    return "unknown"

def _is_relevant_category_based(query_id: str, doc_id: str) -> bool:
    """Check if doc is relevant to query based on category matching (excluding self)"""
    if query_id == doc_id:
        return False
    return _extract_category(query_id) == _extract_category(doc_id)

def _group_rows(rows: List[Dict]) -> Dict[str, List[Dict]]:
    by_q: Dict[str, List[Dict]] = {}
    for r in rows:
        qid = r.get("query_id") or r.get("qid")
        did = r.get("doc_id") or r.get("did")
        score = r.get("rerank_score", r.get("score", 0.0))
        if qid is None or did is None:
            continue
        by_q.setdefault(str(qid), []).append({"doc_id": str(did), "score": float(score)})
    return by_q

def compute_map(rows: List[Dict], qrels: Optional[Dict[str, set]] = None, k: int = 1000, category_based: bool = False, category_counts: Optional[Dict[str, int]] = None) -> float:
    by_q = _group_rows(rows)
    aps: List[float] = []
    for qid, hits in by_q.items():
        if category_based:
            # Compute relevance on-the-fly based on categories
            hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:k]
            num_rel = 0
            precisions: List[float] = []
            for i, h in enumerate(hits, start=1):
                if _is_relevant_category_based(qid, h["doc_id"]):
                    num_rel += 1
                    precisions.append(num_rel / i)
            # Get total relevant docs from category_counts (excluding self)
            query_category = _extract_category(qid)
            if category_counts and query_category in category_counts:
                total_relevant = category_counts[query_category] - 1  # Exclude self
            else:
                # Fallback: use number found as proxy (will inflate score)
                total_relevant = num_rel if num_rel > 0 else 1
            if precisions:
                aps.append(sum(precisions) / max(1, total_relevant))
        else:
            rel_docs = qrels.get(qid)
            if rel_docs is None:
                continue
            hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:k]
            num_rel = 0
            precisions: List[float] = []
            for i, h in enumerate(hits, start=1):
                if h["doc_id"] in rel_docs:
                    num_rel += 1
                    precisions.append(num_rel / i)
            aps.append(sum(precisions) / max(1, len(rel_docs)))
    return sum(aps) / max(1, len(aps))

def compute_recall(rows: List[Dict], qrels: Optional[Dict[str, set]] = None, k: int = 1000, category_based: bool = False, category_counts: Optional[Dict[str, int]] = None) -> float:
    by_q = _group_rows(rows)
    recalls: List[float] = []
    for qid, hits in by_q.items():
        if category_based:
            hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:k]
            true_pos = sum(1 for h in hits if _is_relevant_category_based(qid, h["doc_id"]))
            # Get total relevant docs from category_counts (excluding self)
            # If category_counts not provided, use true_pos as proxy (assumes we retrieved all relevant)
            query_category = _extract_category(qid)
            if category_counts:
                total_relevant = category_counts.get(query_category, 1) - 1
            else:
                # No category_counts: use number of relevant docs found as denominator
                # This makes recall = 1.0 when we find all relevant docs in top-k
                total_relevant = true_pos if true_pos > 0 else 1
            recalls.append(true_pos / max(1, total_relevant))
        else:
            rel_docs = qrels.get(qid)
            if rel_docs is None:
                continue
            hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:k]
            hit_docs = {h["doc_id"] for h in hits}
            true_pos = len(rel_docs & hit_docs)
            recalls.append(true_pos / max(1, len(rel_docs)))
    return sum(recalls) / max(1, len(recalls))

def compute_precision(rows: List[Dict], qrels: Optional[Dict[str, set]] = None, k: int = 10, category_based: bool = False) -> float:
    by_q = _group_rows(rows)
    precs: List[float] = []
    for qid, hits in by_q.items():
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:k]
        if category_based:
            tp = sum(1 for h in hits if _is_relevant_category_based(qid, h["doc_id"]))
            precs.append(tp / max(1, len(hits)))
        else:
            rel_docs = qrels.get(qid)
            if rel_docs is None:
                continue
            hit_docs = [h["doc_id"] for h in hits]
            tp = sum(1 for d in hit_docs if d in rel_docs)
            precs.append(tp / max(1, len(hits)))
    return sum(precs) / max(1, len(precs))

def compute_ndcg(rows: List[Dict], qrels: Optional[Dict[str, set]] = None, k: int = 10, category_based: bool = False, category_counts: Optional[Dict[str, int]] = None) -> float:
    by_q = _group_rows(rows)
    ndcgs: List[float] = []
    for qid, hits in by_q.items():
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:k]
        if category_based:
            dcg = 0.0
            relevant_count = 0
            for i, h in enumerate(hits, start=1):
                rel = 1.0 if _is_relevant_category_based(qid, h["doc_id"]) else 0.0
                dcg += (2**rel - 1) / math.log2(i + 1)
                if rel > 0:
                    relevant_count += 1
            # Ideal DCG: assume we could retrieve all relevant docs in category (minus self)
            query_category = _extract_category(qid)
            total_relevant = category_counts.get(query_category, 1) - 1 if category_counts else relevant_count
            ideal_rel_count = min(total_relevant, k)
            idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, ideal_rel_count + 1))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        else:
            rel_docs = qrels.get(qid)
            if rel_docs is None:
                continue
            dcg = 0.0
            for i, h in enumerate(hits, start=1):
                rel = 1.0 if h["doc_id"] in rel_docs else 0.0
                dcg += (2**rel - 1) / math.log2(i + 1)
            ideal_rel_count = min(len(rel_docs), k)
            idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, ideal_rel_count + 1))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(ndcgs) / max(1, len(ndcgs))


# Convenience wrappers for category-based evaluation
def compute_map_category_based(rows: List[Dict], k: int = 1000, category_counts: Optional[Dict[str, int]] = None) -> float:
    """Compute MAP using category-based relevance (no qrels file needed)"""
    return compute_map(rows, qrels=None, k=k, category_based=True, category_counts=category_counts)

def compute_recall_category_based(rows: List[Dict], k: int = 1000, category_counts: Optional[Dict[str, int]] = None) -> float:
    """Compute Recall using category-based relevance (no qrels file needed)"""
    return compute_recall(rows, qrels=None, k=k, category_based=True, category_counts=category_counts)

def compute_precision_category_based(rows: List[Dict], k: int = 10, category_counts: Optional[Dict[str, int]] = None) -> float:
    """Compute Precision using category-based relevance (no qrels file needed)"""
    return compute_precision(rows, qrels=None, k=k, category_based=True)

def compute_ndcg_category_based(rows: List[Dict], k: int = 10, category_counts: Optional[Dict[str, int]] = None) -> float:
    """Compute nDCG using category-based relevance (no qrels file needed)"""
    return compute_ndcg(rows, qrels=None, k=k, category_based=True, category_counts=category_counts)

