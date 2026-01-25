import pytest

from udlf_text_expresso.rerank.udlf_runner import prepare_udlf_input_files


def test_prepare_udlf_input_files_creates_expected_outputs(tmp_path):
    # Input rows are assumed to be pre-sorted by score (highest first)
    rows = [
        {"query_id": "q1", "doc_id": "d2", "score": 0.7},
        {"query_id": "q1", "doc_id": "d1", "score": 0.5, "payload": {"text": "foo"}},
        {"query_id": "q2", "doc_id": "d3", "score": 0.2},
    ]

    artifacts = prepare_udlf_input_files(rows, tmp_path)

    assert artifacts.ranked_list_path.exists()
    assert artifacts.lists_path.exists()
    assert artifacts.num_queries == 2
    assert artifacts.num_docs == 3

    ranked_contents = artifacts.ranked_list_path.read_text(encoding="utf-8").strip().splitlines()
    assert ranked_contents == ["q1 d2 d1", "q2 d3"]

    lists_contents = artifacts.lists_path.read_text(encoding="utf-8").strip().splitlines()
    assert lists_contents == ["d1", "d2", "d3"]


def test_prepare_udlf_input_files_requires_documents(tmp_path):
    with pytest.raises(ValueError):
        prepare_udlf_input_files([], tmp_path)
