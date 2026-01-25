"""Unit tests for `compute_precision` using the shared synthetic dataset."""
from __future__ import annotations

import pytest

from udlf_text_expresso.metrics.eval import compute_precision


def test_compute_precision_on_sample_dataset(sample_dataset, sample_retrieval_rows) -> None:
    value = compute_precision(sample_retrieval_rows, sample_dataset.qrels, k=3)
    assert value == pytest.approx(0.6, rel=1e-4)


def test_compute_precision_no_rows(sample_dataset) -> None:
    assert compute_precision([], sample_dataset.qrels, k=3) == pytest.approx(0.0)


def test_compute_precision_ignores_missing_qrels(sample_dataset, sample_retrieval_rows) -> None:
    baseline = compute_precision(sample_retrieval_rows, sample_dataset.qrels, k=3)
    extra_rows = sample_retrieval_rows + [
        {"query_id": sample_dataset.docs[1]["id"], "doc_id": sample_dataset.docs[0]["id"], "score": 0.3}
    ]
    assert compute_precision(extra_rows, sample_dataset.qrels, k=3) == pytest.approx(baseline)
