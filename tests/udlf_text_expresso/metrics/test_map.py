"""Unit tests for `compute_map` using the shared synthetic dataset."""
from __future__ import annotations

import pytest

from udlf_text_expresso.metrics.eval import compute_map


def test_compute_map_on_sample_dataset(sample_dataset, sample_retrieval_rows) -> None:
    value = compute_map(sample_retrieval_rows, sample_dataset.qrels, k=3)
    assert value == pytest.approx(0.18, rel=1e-4)


def test_compute_map_no_hits_yields_zero(sample_dataset) -> None:
    assert compute_map([], sample_dataset.qrels, k=3) == pytest.approx(0.0)


def test_compute_map_ignores_queries_without_qrels(sample_dataset, sample_retrieval_rows) -> None:
    baseline = compute_map(sample_retrieval_rows, sample_dataset.qrels, k=3)
    extra_rows = sample_retrieval_rows + [
        {"query_id": sample_dataset.docs[0]["id"], "doc_id": sample_dataset.docs[1]["id"], "score": 0.5}
    ]
    assert compute_map(extra_rows, sample_dataset.qrels, k=3) == pytest.approx(baseline)
