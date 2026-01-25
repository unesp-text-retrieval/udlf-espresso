"""Unit tests for category-based evaluation (non-BEIR datasets)."""
from __future__ import annotations

import pytest

from udlf_text_expresso.metrics.eval import (
    compute_map_category_based,
    compute_recall_category_based,
    compute_precision_category_based,
    compute_ndcg_category_based,
    _extract_category,
)


@pytest.fixture
def category_dataset():
    """
    Mock category-based dataset with 5 categories.
    Each category has 3 documents.
    """
    return {
        "docs": [
            # Sport category (3 docs)
            {"id": "sport_001", "text": "Football match highlights"},
            {"id": "sport_002", "text": "Basketball championship"},
            {"id": "sport_003", "text": "Tennis tournament results"},
            # Business category (3 docs)
            {"id": "business_001", "text": "Stock market analysis"},
            {"id": "business_002", "text": "Quarterly earnings report"},
            {"id": "business_003", "text": "Merger acquisition news"},
            # Tech category (3 docs)
            {"id": "tech_001", "text": "New smartphone release"},
            {"id": "tech_002", "text": "AI breakthrough announcement"},
            {"id": "tech_003", "text": "Cloud computing trends"},
            # Politics category (2 docs)
            {"id": "politics_001", "text": "Election results analysis"},
            {"id": "politics_002", "text": "Policy reform proposal"},
            # Entertainment category (2 docs)
            {"id": "entertainment_001", "text": "Movie box office numbers"},
            {"id": "entertainment_002", "text": "Music awards ceremony"},
        ],
        "categories": {
            "sport": ["sport_001", "sport_002", "sport_003"],
            "business": ["business_001", "business_002", "business_003"],
            "tech": ["tech_001", "tech_002", "tech_003"],
            "politics": ["politics_001", "politics_002"],
            "entertainment": ["entertainment_001", "entertainment_002"],
        },
    }


@pytest.fixture
def category_retrieval_rows(category_dataset):
    """
    Simulated retrieval results:
    - Query: sport_001 retrieves [sport_002 (relevant), business_001 (not relevant), sport_003 (relevant)]
    - Query: business_002 retrieves [business_001 (relevant), business_003 (relevant), tech_001 (not relevant)]
    - Query: tech_001 retrieves [tech_002 (relevant), tech_003 (relevant), sport_001 (not relevant)]
    """
    return [
        # Query sport_001
        {"query_id": "sport_001", "doc_id": "sport_002", "score": 0.95},
        {"query_id": "sport_001", "doc_id": "business_001", "score": 0.80},
        {"query_id": "sport_001", "doc_id": "sport_003", "score": 0.75},
        {"query_id": "sport_001", "doc_id": "tech_001", "score": 0.60},
        # Query business_002
        {"query_id": "business_002", "doc_id": "business_001", "score": 0.90},
        {"query_id": "business_002", "doc_id": "business_003", "score": 0.85},
        {"query_id": "business_002", "doc_id": "tech_001", "score": 0.70},
        {"query_id": "business_002", "doc_id": "sport_002", "score": 0.55},
        # Query tech_001
        {"query_id": "tech_001", "doc_id": "tech_002", "score": 0.92},
        {"query_id": "tech_001", "doc_id": "tech_003", "score": 0.88},
        {"query_id": "tech_001", "doc_id": "sport_001", "score": 0.65},
        {"query_id": "tech_001", "doc_id": "business_001", "score": 0.50},
    ]


def test_extract_category():
    """Test category extraction from document IDs."""
    assert _extract_category("sport_001") == "sport"
    assert _extract_category("business_123") == "business"
    assert _extract_category("tech_999.txt") == "tech"
    assert _extract_category("nodash") == "unknown"
    assert _extract_category("") == "unknown"


def test_compute_map_category_based(category_retrieval_rows):
    """Test MAP computation with category-based relevance."""
    # sport_001: [sport_002 (rel), business_001 (not), sport_003 (rel)] @ k=3
    #   AP = (1/1 + 2/3) / 2 = (1.0 + 0.667) / 2 = 0.833
    # business_002: [business_001 (rel), business_003 (rel), tech_001 (not)] @ k=3
    #   AP = (1/1 + 2/2) / 2 = (1.0 + 1.0) / 2 = 1.0
    # tech_001: [tech_002 (rel), tech_003 (rel), sport_001 (not)] @ k=3
    #   AP = (1/1 + 2/2) / 2 = (1.0 + 1.0) / 2 = 1.0
    # Overall MAP = (0.833 + 1.0 + 1.0) / 3 = 0.944
    value = compute_map_category_based(category_retrieval_rows, k=3)
    assert value == pytest.approx(0.944, rel=1e-2)


def test_compute_recall_category_based(category_retrieval_rows):
    """Test Recall computation with category-based relevance."""
    # sport_001: retrieved 2 out of 2 relevant (sport_002, sport_003) -> recall = 1.0
    # business_002: retrieved 2 out of 2 relevant (business_001, business_003) -> recall = 1.0
    # tech_001: retrieved 2 out of 2 relevant (tech_002, tech_003) -> recall = 1.0
    # Overall Recall = (1.0 + 1.0 + 1.0) / 3 = 1.0
    value = compute_recall_category_based(category_retrieval_rows, k=4)
    assert value == pytest.approx(1.0, rel=1e-2)


def test_compute_precision_category_based(category_retrieval_rows):
    """Test Precision computation with category-based relevance."""
    # sport_001: 2 relevant out of 3 retrieved @ k=3 -> precision = 2/3 = 0.667
    # business_002: 2 relevant out of 3 retrieved @ k=3 -> precision = 2/3 = 0.667
    # tech_001: 2 relevant out of 3 retrieved @ k=3 -> precision = 2/3 = 0.667
    # Overall Precision = (0.667 + 0.667 + 0.667) / 3 = 0.667
    value = compute_precision_category_based(category_retrieval_rows, k=3)
    assert value == pytest.approx(0.667, rel=1e-2)


def test_compute_ndcg_category_based(category_retrieval_rows):
    """Test nDCG computation with category-based relevance."""
    # sport_001 @ k=3: [sport_002 (rel), business_001 (not), sport_003 (rel)]
    #   DCG = (2^1-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4) = 1/1 + 0 + 1/2 = 1.5
    #   IDCG = (2^1-1)/log2(2) + (2^1-1)/log2(3) = 1 + 0.631 = 1.631
    #   nDCG = 1.5 / 1.631 = 0.920
    # business_002 @ k=3: [business_001 (rel), business_003 (rel), tech_001 (not)]
    #   DCG = 1/1 + 1/log2(3) + 0 = 1 + 0.631 = 1.631
    #   IDCG = 1.631
    #   nDCG = 1.0
    # tech_001 @ k=3: [tech_002 (rel), tech_003 (rel), sport_001 (not)]
    #   DCG = 1.631
    #   IDCG = 1.631
    #   nDCG = 1.0
    # Overall nDCG = (0.920 + 1.0 + 1.0) / 3 = 0.973
    value = compute_ndcg_category_based(category_retrieval_rows, k=3)
    assert value == pytest.approx(0.973, rel=1e-2)


def test_category_based_with_no_results():
    """Test that category-based metrics return 0 when no results."""
    assert compute_map_category_based([], k=10) == 0.0
    assert compute_recall_category_based([], k=10) == 0.0
    assert compute_precision_category_based([], k=10) == 0.0
    assert compute_ndcg_category_based([], k=10) == 0.0


def test_category_based_with_single_doc_category():
    """Test that queries from single-doc categories have no relevant results (except self, which is excluded)."""
    rows = [
        {"query_id": "unique_001", "doc_id": "sport_001", "score": 0.9},
        {"query_id": "unique_001", "doc_id": "business_001", "score": 0.8},
    ]
    # unique_001 has no other docs in "unique" category, so all results are non-relevant
    # MAP = 0/1 = 0.0 (no relevant docs found)
    value = compute_map_category_based(rows, k=2)
    assert value == pytest.approx(0.0, rel=1e-2)


def test_category_based_excludes_self():
    """Test that a document doesn't consider itself as relevant when used as query."""
    rows = [
        {"query_id": "sport_001", "doc_id": "sport_001", "score": 1.0},  # Self (should be excluded)
        {"query_id": "sport_001", "doc_id": "sport_002", "score": 0.9},  # Relevant
        {"query_id": "sport_001", "doc_id": "business_001", "score": 0.8},  # Not relevant
    ]
    # Should only count sport_002 as relevant, not sport_001 (self)
    # Precision@3 = 1/3 = 0.333
    value = compute_precision_category_based(rows, k=3)
    assert value == pytest.approx(0.333, rel=1e-2)
