"""Test TREC format conversion for UDLF reranking pipeline."""
import pytest
from pathlib import Path

from udlf_text_expresso.rerank.udlf_runner import prepare_udlf_input_files, parse_udlf_output_to_trec


def test_trec_to_udlf_format_conversion(tmp_path):
    """Test that retrieval.tsv rows are correctly converted to UDLF ranked-list and lists format."""
    # Input: TREC format (like retrieval.tsv)
    rows = [
        {"query_id": "PLAIN-2", "doc_id": "MED-10", "score": 0.688},
        {"query_id": "PLAIN-2", "doc_id": "MED-14", "score": 0.6859},
        {"query_id": "PLAIN-2", "doc_id": "MED-2429", "score": 0.6856},
    ]

    artifacts = prepare_udlf_input_files(rows, tmp_path)

    # Verify ranked-list.txt format (A): sorted by score descending
    ranked_content = artifacts.ranked_list_path.read_text(encoding="utf-8").strip()
    assert ranked_content == "PLAIN-2 MED-10 MED-14 MED-2429"

    # Verify lists.txt format (B): all unique doc IDs sorted
    lists_content = artifacts.lists_path.read_text(encoding="utf-8").strip().splitlines()
    assert lists_content == ["MED-10", "MED-14", "MED-2429"]

    assert artifacts.num_queries == 1
    assert artifacts.num_docs == 3


def test_udlf_output_to_trec_format():
    """Test that UDLF reranked output is correctly converted back to TREC format with linear scores."""
    # Simulated UDLF output (ranked-list format with reordered documents)
    udlf_output_line = "PLAIN-2 MED-14 MED-10 MED-2429"
    
    # Original retrieval rows for metadata lookup
    original_rows = [
        {"query_id": "PLAIN-2", "doc_id": "MED-10", "score": 0.688, "text": "doc 10"},
        {"query_id": "PLAIN-2", "doc_id": "MED-14", "score": 0.6859, "text": "doc 14"},
        {"query_id": "PLAIN-2", "doc_id": "MED-2429", "score": 0.6856, "text": "doc 2429"},
    ]
    
    row_lookup = {(str(r["query_id"]), str(r["doc_id"])): r for r in original_rows}
    
    # Parse UDLF output and convert to TREC format
    parts = udlf_output_line.strip().split()
    qid = parts[0]
    doc_list = parts[1:]
    total_docs = len(doc_list)
    
    reranked_rows = []
    for rank, did in enumerate(doc_list, start=1):
        original = row_lookup[(qid, did)]
        baseline_score = float(original["score"])
        # Linear score: 200, 199, 198, ... (for 200 items total)
        rerank_score = float(total_docs - rank + 1)
        reranked_rows.append({
            **original,
            "baseline_score": baseline_score,
            "score": rerank_score,
            "rank": rank,
        })
    
    # Verify TREC format output
    assert len(reranked_rows) == 3
    
    # First result: MED-14 at rank 1 with score 3
    assert reranked_rows[0]["query_id"] == "PLAIN-2"
    assert reranked_rows[0]["doc_id"] == "MED-14"
    assert reranked_rows[0]["rank"] == 1
    assert reranked_rows[0]["score"] == 3.0
    assert reranked_rows[0]["baseline_score"] == 0.6859
    
    # Second result: MED-10 at rank 2 with score 2
    assert reranked_rows[1]["doc_id"] == "MED-10"
    assert reranked_rows[1]["rank"] == 2
    assert reranked_rows[1]["score"] == 2.0
    assert reranked_rows[1]["baseline_score"] == 0.688
    
    # Third result: MED-2429 at rank 3 with score 1
    assert reranked_rows[2]["doc_id"] == "MED-2429"
    assert reranked_rows[2]["rank"] == 3
    assert reranked_rows[2]["score"] == 1.0
    assert reranked_rows[2]["baseline_score"] == 0.6856


def test_parse_udlf_output_to_trec(tmp_path):
    """Test parsing UDLF output file to TREC format."""
    # Create a mock UDLF output file
    output_file = tmp_path / "data.txt"
    output_file.write_text(
        "PLAIN-2 MED-14 MED-10 MED-2429\n"
        "PLAIN-3 MED-1 MED-2\n"
    )
    
    result = parse_udlf_output_to_trec(output_file)
    
    # Verify first query results
    assert result[0] == {"query_id": "PLAIN-2", "doc_id": "MED-14", "score": 3.0, "rank": 1}
    assert result[1] == {"query_id": "PLAIN-2", "doc_id": "MED-10", "score": 2.0, "rank": 2}
    assert result[2] == {"query_id": "PLAIN-2", "doc_id": "MED-2429", "score": 1.0, "rank": 3}
    
    # Verify second query results
    assert result[3] == {"query_id": "PLAIN-3", "doc_id": "MED-1", "score": 2.0, "rank": 1}
    assert result[4] == {"query_id": "PLAIN-3", "doc_id": "MED-2", "score": 1.0, "rank": 2}


def test_parse_udlf_output_missing_file(tmp_path):
    """Test error handling when UDLF output file is missing."""
    missing_file = tmp_path / "nonexistent.txt"
    
    with pytest.raises(FileNotFoundError, match="UDLF output file not found"):
        parse_udlf_output_to_trec(missing_file)


def test_parse_udlf_output_empty_lines(tmp_path):
    """Test parsing UDLF output with empty lines."""
    output_file = tmp_path / "data.txt"
    output_file.write_text(
        "Q1 D1 D2\n"
        "\n"
        "Q2 D3\n"
        "  \n"
    )
    
    result = parse_udlf_output_to_trec(output_file)
    
    assert len(result) == 3
    assert result[0]["query_id"] == "Q1"
    assert result[2]["query_id"] == "Q2"


def test_multi_query_udlf_format(tmp_path):
    """Test UDLF format generation with multiple queries."""
    rows = [
        {"query_id": "Q1", "doc_id": "D1", "score": 0.9},
        {"query_id": "Q1", "doc_id": "D2", "score": 0.8},
        {"query_id": "Q2", "doc_id": "D3", "score": 0.7},
        {"query_id": "Q2", "doc_id": "D1", "score": 0.6},
    ]

    artifacts = prepare_udlf_input_files(rows, tmp_path)

    ranked_lines = artifacts.ranked_list_path.read_text(encoding="utf-8").strip().splitlines()
    assert ranked_lines == [
        "Q1 D1 D2",
        "Q2 D3 D1",
    ]

    lists_lines = artifacts.lists_path.read_text(encoding="utf-8").strip().splitlines()
    assert sorted(lists_lines) == ["D1", "D2", "D3"]

    assert artifacts.num_queries == 2
    assert artifacts.num_docs == 3
