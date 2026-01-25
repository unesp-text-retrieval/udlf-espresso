"""Unit tests for `compute_recall` using the shared synthetic dataset."""
from __future__ import annotations

import pytest

from udlf_text_expresso.metrics.eval import compute_recall


def test_compute_recall_on_sample_dataset(sample_dataset, sample_retrieval_rows) -> None:
    value = compute_recall(sample_retrieval_rows, sample_dataset.qrels, k=3)
    assert value == pytest.approx(0.18, rel=1e-4)


def test_compute_recall_empty_rows(sample_dataset) -> None:
    assert compute_recall([], sample_dataset.qrels, k=3) == pytest.approx(0.0)


def test_compute_recall_skips_queries_without_qrels(sample_dataset, sample_retrieval_rows) -> None:
    baseline = compute_recall(sample_retrieval_rows, sample_dataset.qrels, k=3)
    extra_rows = sample_retrieval_rows + [
        {"query_id": sample_dataset.docs[0]["id"], "doc_id": sample_dataset.docs[2]["id"], "score": 0.4}
    ]
    assert compute_recall(extra_rows, sample_dataset.qrels, k=3) == pytest.approx(baseline)
