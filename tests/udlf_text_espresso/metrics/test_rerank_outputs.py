from __future__ import annotations

from pathlib import Path

import pytest

from udlf_text_espresso.metrics.rerank_eval import evaluate_rows, write_rerank_outputs


def test_write_rerank_outputs_respects_limit(tmp_path):
    rows = [
        {"query_id": "q1", "doc_id": "d1", "score": 3.0},
        {"query_id": "q1", "doc_id": "d2", "score": 2.0},
        {"query_id": "q1", "doc_id": "d3", "score": 1.0},
        {"query_id": "q2", "doc_id": "d4", "score": 4.0},
        {"query_id": "q2", "doc_id": "d5", "score": 3.5},
    ]

    horizontal_path = Path(tmp_path) / "data.txt"
    trec_path = Path(tmp_path) / "data.trec"
    ranks_path = Path(tmp_path) / "ranks.tsv"

    limited = write_rerank_outputs(
        rows,
        method_label="udlf",
        horizontal_path=horizontal_path,
        trec_path=trec_path,
        ranks_path=ranks_path,
        limit=2,
    )

    assert len(limited) == 4  # 2 per query
    assert horizontal_path.read_text(encoding="utf-8").strip().splitlines() == [
        "q1 d1 d2",
        "q2 d4 d5",
    ]

    trec_lines = trec_path.read_text(encoding="utf-8").strip().splitlines()
    assert trec_lines[0].startswith("q1 Q0 d1 1")
    assert trec_lines[1].startswith("q1 Q0 d2 2")

    rank_lines = ranks_path.read_text(encoding="utf-8").strip().splitlines()
    assert rank_lines == [
        "q1\td1\t3.0",
        "q1\td2\t2.0",
        "q2\td4\t4.0",
        "q2\td5\t3.5",
    ]


def test_evaluate_rows_returns_expected_scores():
    rows = [
        {"query_id": "q1", "doc_id": "d1", "score": 2.0},
    ]
    qrels = {"q1": {"d1"}}

    metrics = evaluate_rows(
        rows,
        qrels,
        map_k=[1],
        recall_k=[1],
        precision_k=[1],
        ndcg_k=[1],
    )

    assert metrics["MAP@1"] == pytest.approx(1.0)
    assert metrics["Recall@1"] == pytest.approx(1.0)
    assert metrics["Precision@1"] == pytest.approx(1.0)
    assert metrics["nDCG@1"] == pytest.approx(1.0)
