"""BM25 retriever tests with descriptive names.

Each assertion is split into its own test for clearer reporting.
"""
import json
from udlf_text_espresso.retrieval.bm25 import BM25Retriever


CORPUS = [
    {"id": "d1", "text": "the cat sat on the mat"},
    {"id": "d2", "text": "the dog chased the cat"},
    {"id": "d3", "text": "cats and dogs are animals"},
]
QUERIES = [
    {"id": "q1", "text": "cat on mat"},
]


def _run_bm25(k: int = 3):
    retriever = BM25Retriever(corpus=CORPUS, k=k)
    return retriever.retrieve(QUERIES)


def test_bm25_returns_one_result_per_query():
    """BM25 returns one `RetrievalResult` per input query."""
    results = _run_bm25()
    assert len(results) == 1, "BM25 should return one result per query"


def test_bm25_preserves_query_id_in_results():
    """Re    python -m pytestturned result keeps the original query id."""
    results = _run_bm25()
    assert results[0].query_id == "q1", "Query id should be preserved in results"


def test_bm25_outputs_k_hits_equal_to_request():
    """Outputs exactly k hits when k equals corpus size."""
    results = _run_bm25(k=3)
    assert len(results[0].hits) == 3, "Should produce k hits when k equals corpus size"


def test_bm25_hit_schema_contains_doc_id_and_score():
    """Each hit contains `doc_id` and `score` fields."""
    results = _run_bm25()
    for h in results[0].hits:
        assert "doc_id" in h and "score" in h, "Each hit must contain doc_id and score"


def test_bm25_scores_are_floats():
    """Scores are floats suitable for sorting and metrics."""
    results = _run_bm25()
    for h in results[0].hits:
        assert isinstance(h["score"], float), "Scores must be floats"


def test_bm25_top_doc_is_relevant_to_query_terms():
    """Top-ranked document is plausibly relevant to query terms."""
    results = _run_bm25()
    top_doc = results[0].hits[0]["doc_id"]
    assert top_doc in {"d1", "d2"}, "Top doc should be relevant to query terms"
