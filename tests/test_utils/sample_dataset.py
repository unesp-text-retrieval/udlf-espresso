"""Shared helpers for constructing a synthetic retrieval dataset used in tests."""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from udlf_text_expresso.core.paths import (
    dataset_root,
    transformed_documents_and_topics_tsv,
    transformed_documents_jsonl,
    extracted_path,
    transformed_path,
)
from udlf_text_expresso.core.text import normalize_query_text


@dataclass
class SampleDataset:
    docs: List[Dict[str, str]]
    queries: List[Tuple[str, str]]
    qrels: Dict[str, set[str]]
    documents_path: Path
    topics_path: Path
    qrels_path: Path
    corpus_path: Path


def _write_gzip_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _make_docs(prefix: str, count: int, body: Sequence[str]) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    joined = " ".join(body)
    for i in range(count):
        docs.append({
            "id": f"{prefix}-{i:02d}",
            "contents": f"{joined} sample {i}",
        })
    return docs


def build_sample_dataset(base_dir: Path, dataset_name: str) -> SampleDataset:
    energy_docs = _make_docs("energy", 10, ["renewable", "solar", "wind", "battery", "energy"])
    health_docs = _make_docs("health", 10, ["cancer", "therapy", "medical", "health", "clinic"])
    space_docs = _make_docs("space", 10, ["space", "mars", "satellite", "mission", "rocket"])
    docs = energy_docs + health_docs + space_docs
    assert len(docs) >= 30, "Synthetic dataset should contain at least 30 documents"

    queries: List[Tuple[str, str]] = [
        ("q-energy", "renewable solar wind energy progress"),
        ("q-health", "cancer therapy medical advances"),
        ("q-space", "mars mission satellite operations"),
        ("q-empty", "ancient philosophy studies"),
        ("q-confused", "renewable energy storage"),
    ]

    qrels: Dict[str, set[str]] = {
        "q-energy": {d["id"] for d in energy_docs},
        "q-health": {d["id"] for d in health_docs},
        "q-space": {d["id"] for d in space_docs},
        "q-empty": set(),
        "q-confused": {health_docs[0]["id"]},
    }

    run_root = base_dir
    root_dir = run_root / dataset_root(dataset_name)
    transformed_dir = run_root / transformed_path(dataset_name, "documents", fmt="jsonl").parent
    transformed_dir.mkdir(parents=True, exist_ok=True)
    documents_path = run_root / transformed_documents_jsonl(dataset_name)
    with open(documents_path, "w", encoding="utf-8") as handle:
        for doc in docs:
            json.dump({"id": doc["id"], "contents": doc["contents"]}, handle)
            handle.write("\n")

    merged_path = run_root / transformed_documents_and_topics_tsv(dataset_name)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as handle:
        for qid, text in queries:
            handle.write(f"{qid}\t{normalize_query_text(text)}\n")
        for doc in docs:
            handle.write(f"{doc['id']}\t{normalize_query_text(doc['contents'])}\n")

    topics_lines = [f"{qid}\t{text}" for qid, text in queries]
    qrels_lines = [f"{qid}\t{did}" for qid, dids in qrels.items() for did in dids]
    corpus_lines = [f"{doc['id']}\t{doc['contents']}" for doc in docs]

    topics_path = run_root / extracted_path(dataset_name, "topics")
    qrels_path = run_root / extracted_path(dataset_name, "qrels")
    corpus_path = run_root / extracted_path(dataset_name, "documents")
    _write_gzip_lines(topics_path, topics_lines)
    _write_gzip_lines(qrels_path, qrels_lines)
    _write_gzip_lines(corpus_path, corpus_lines)

    return SampleDataset(
        docs=docs,
        queries=queries,
        qrels=qrels,
        documents_path=documents_path,
        topics_path=topics_path,
        qrels_path=qrels_path,
        corpus_path=corpus_path,
    )


def make_sample_retrieval_rows(sample: SampleDataset) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = [
        {"query_id": "q-energy", "doc_id": "energy-00", "score": 3.0},
        {"query_id": "q-energy", "doc_id": "energy-01", "score": 2.5},
        {"query_id": "q-energy", "doc_id": "energy-02", "score": 2.0},
        {"query_id": "q-energy", "doc_id": "health-00", "score": 1.0},
        {"query_id": "q-health", "doc_id": "health-00", "score": 3.0},
        {"query_id": "q-health", "doc_id": "health-01", "score": 2.5},
        {"query_id": "q-health", "doc_id": "health-02", "score": 2.0},
        {"query_id": "q-health", "doc_id": "energy-00", "score": 1.0},
        {"query_id": "q-space", "doc_id": "space-00", "score": 3.0},
        {"query_id": "q-space", "doc_id": "space-01", "score": 2.5},
        {"query_id": "q-space", "doc_id": "space-02", "score": 2.0},
        {"query_id": "q-space", "doc_id": "energy-00", "score": 1.0},
        {"query_id": "q-empty", "doc_id": "energy-00", "score": 1.0},
        {"query_id": "q-confused", "doc_id": "energy-00", "score": 1.5},
        {"query_id": "q-confused", "doc_id": "energy-01", "score": 1.0},
    ]
    # Basic sanity checks to ensure ids exist in the dataset.
    doc_ids = {doc["id"] for doc in sample.docs}
    for row in rows:
        assert row["doc_id"] in doc_ids, f"Unknown document id in retrieval rows: {row['doc_id']}"
    return rows
