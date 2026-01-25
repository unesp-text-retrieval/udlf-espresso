"""Unit tests for `compute_ndcg` using the shared synthetic dataset."""
from __future__ import annotations

import pytest

from udlf_text_expresso.metrics.eval import compute_ndcg


def test_compute_ndcg_on_sample_dataset(sample_dataset, sample_retrieval_rows) -> None:
    value = compute_ndcg(sample_retrieval_rows, sample_dataset.qrels, k=3)
    assert value == pytest.approx(0.6, rel=1e-4)


def test_compute_ndcg_with_empty_rows(sample_dataset) -> None:
    assert compute_ndcg([], sample_dataset.qrels, k=3) == pytest.approx(0.0)


def test_compute_ndcg_skips_queries_without_qrels(sample_dataset, sample_retrieval_rows) -> None:
    baseline = compute_ndcg(sample_retrieval_rows, sample_dataset.qrels, k=3)
    extra_rows = sample_retrieval_rows + [
        {"query_id": sample_dataset.docs[2]["id"], "doc_id": sample_dataset.docs[0]["id"], "score": 0.2}
    ]
    assert compute_ndcg(extra_rows, sample_dataset.qrels, k=3) == pytest.approx(baseline)
