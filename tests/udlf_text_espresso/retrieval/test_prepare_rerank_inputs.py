from __future__ import annotations
from pathlib import Path

from udlf_text_espresso.runner import _prepare_rerank_inputs


def test_prepare_rerank_inputs_writes_ranked_and_query_lists(tmp_path):
    base_out = tmp_path
    rows = [
        {"query_id": "q1", "doc_id": "d1", "score": 0.2},
        {"query_id": "q1", "doc_id": "d2", "score": 0.9},
        {"query_id": "q2", "doc_id": "d3", "score": 0.1},
        {"query_id": "q1", "doc_id": "d3", "score": 0.5},
        {"query_id": "q2", "doc_id": "d4", "score": 0.7},
    ]

    result = _prepare_rerank_inputs(base_out, "scifact", "miniLM", rows, limit=2)
    assert result is not None

    ranked_path, lists_path = result
    expected_ranked = Path(base_out) / "dataset/scifact/rerank/input/miniLM/ranked-list/data.txt"
    expected_lists = Path(base_out) / "dataset/scifact/rerank/input/miniLM/lists/data.txt"

    assert ranked_path == expected_ranked
    assert lists_path == expected_lists

    ranked_lines = ranked_path.read_text(encoding="utf-8").strip().splitlines()
    lists_lines = lists_path.read_text(encoding="utf-8").strip().splitlines()

    assert ranked_lines == ["q1 d2 d3", "q2 d4 d3"]
    assert lists_lines == ["q1", "q2", "d2", "d3", "d4"]
