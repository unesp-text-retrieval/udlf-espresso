"""Unit tests for dense retrieval models using synthetic data."""
from __future__ import annotations

from typing import Dict

import pytest  # type: ignore[import-not-found]

from udlf_text_espresso.retrieval.dense import DenseIndexer
from test_utils.dense_stubs import patch_dense_environment
from test_utils.sample_dataset import SampleDataset


@pytest.mark.parametrize(
    "model_name",
    ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
)
def test_dense_model_prefers_relevant_category(
    sample_dataset: SampleDataset,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    patch_dense_environment(monkeypatch)
    indexer = DenseIndexer(
        model_name=model_name,
        normalize=True,
        device=None,
        batch_size=8,
        use_subprocess=False,
        omp_threads=1,
    )
    doc_ids = [doc["id"] for doc in sample_dataset.docs]
    doc_texts = [doc["contents"] for doc in sample_dataset.docs]
    embeddings = indexer.encode_texts(doc_texts)
    index = indexer.build_index(embeddings)

    queries: Dict[str, str] = {qid: text for qid, text in sample_dataset.queries}
    q_texts = [queries["q-energy"], queries["q-health"], queries["q-space"]]
    q_embs = indexer.encode_texts(q_texts)
    indexer.faiss.normalize_L2(q_embs)
    _, hits = index.search(q_embs, top_k=1)

    top_energy = doc_ids[hits[0][0]]
    top_health = doc_ids[hits[1][0]]
    top_space = doc_ids[hits[2][0]]

    assert top_energy.startswith("energy-"), f"Expected energy doc, got {top_energy}"
    assert top_health.startswith("health-"), f"Expected health doc, got {top_health}"
    assert top_space.startswith("space-"), f"Expected space doc, got {top_space}"

    indexer.cleanup()
