"""End-to-end dense indexing and retrieval tests using a synthetic dataset.

The synthetic dataset contains 30+ documents with known relevance relationships so that
indexing, retrieval, and metric computations can be validated deterministically without
external model downloads. Dense encoders and FAISS are monkeypatched with light-weight
stubs that emulate the real interfaces.
"""
from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Dict, List

import pytest  # type: ignore[import-not-found]
from omegaconf import OmegaConf  # type: ignore[import-not-found]

from udlf_text_expresso.core.artifacts import Artifact
from udlf_text_expresso.core.paths import metrics_file_path, ranks_root_dir
from udlf_text_expresso.core.pipeline import PipelineContext
from udlf_text_expresso.core.storage import LocalStorage
from udlf_text_expresso.metrics.eval import (
    compute_map,
    compute_ndcg,
    compute_precision,
    compute_recall,
)
from udlf_text_expresso.runner import DenseRetrievalStep, IndexDenseStep, MetricsStep
from test_utils.dense_stubs import DenseTestEnvironment, patch_dense_environment
from test_utils.sample_dataset import build_sample_dataset


@pytest.fixture()
def stubbed_environment(monkeypatch: pytest.MonkeyPatch) -> DenseTestEnvironment:
    return patch_dense_environment(monkeypatch)


def _extract_variants() -> List[Dict[str, str]]:
    project_root = Path(__file__).resolve().parent.parent
    cfg = OmegaConf.load(project_root / "conf" / "config.yaml")
    # Set dataset as a dict with name to satisfy dense_indexing.dataset_name interpolation
    OmegaConf.update(cfg, "dataset", {"name": "synthetic-sample"}, merge=False)
    dense_cfg = OmegaConf.to_container(cfg.dense_indexing, resolve=True)  # type: ignore[arg-type]
    variants_cfg = OmegaConf.to_container(cfg.dense_indexing.variants, resolve=True)  # type: ignore[arg-type]
    candidates: List[Dict[str, str]] = [dense_cfg] + list(variants_cfg or [])
    seen: Dict[str, Dict[str, str]] = {}
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        alias = str(cand.get("model_alias") or cand.get("model_name", "").split("/")[-1])
        seen[alias] = {"model_name": cand["model_name"], "model_alias": alias}
    return list(seen.values())


def _load_retrieval_rows(tsv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in tsv_path.read_text(encoding="utf-8").strip().splitlines():
        if not line:
            continue
        qid, did, score = line.split("\t")
        rows.append({"query_id": qid, "doc_id": did, "score": float(score)})
    return rows


@pytest.mark.parametrize("precision_k", [3])
def test_dense_pipeline_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stubbed_environment: DenseTestEnvironment,
    precision_k: int,
) -> None:
    dataset_name = "synthetic-sample"
    run_id = "unit-test-dense"
    monkeypatch.chdir(tmp_path)
    base_dir = Path("outputs") / run_id
    sample = build_sample_dataset(base_dir, dataset_name)
    qrels_for_metrics: Dict[str, set[str]] = {}
    with gzip.open(sample.qrels_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            qid, did = line.split("\t", 1)
            qrels_for_metrics.setdefault(qid, set()).add(did)

    ctx = PipelineContext(run_id=run_id, storage=LocalStorage(), force=True)
    ctx.register(Artifact(name="topics", uri=str(sample.topics_path), kind="topics"))
    ctx.register(Artifact(name="qrels", uri=str(sample.qrels_path), kind="qrels"))
    ctx.register(Artifact(name="corpus", uri=str(sample.corpus_path), kind="corpus"))

    variants = _extract_variants()
    _ = stubbed_environment  # ensure fixture is used
    base_dense_cfg = {
        "output_dir": str(base_dir),
        "dataset_name": dataset_name,
        "normalize": True,
        "device": None,
        "batch_size": 8,
        "use_subprocess": False,
        "k": 5,
        "omp_threads": 1,
    }
    metrics_cutoffs = {
        "map": [5],
        "recall": [5],
        "precision": [precision_k],
        "ndcg": [5],
    }

    for variant in variants:
        model_name = variant["model_name"]
        alias = variant["model_alias"]
        dense_cfg = OmegaConf.create({**base_dense_cfg, "model_name": model_name, "model_alias": alias})
        index_step = IndexDenseStep(dense_cfg)
        index_step.execute(ctx)

        retrieval_step = DenseRetrievalStep(dense_cfg)
        retrieval_step.execute(ctx)

        metrics_cfg = OmegaConf.create(
            {
                "output_dir": str(metrics_file_path(base_dir, dataset_name, alias).parent),
                "dataset_name": dataset_name,
                "index_name": alias,
                "k": metrics_cutoffs["map"],
                "cutoffs": metrics_cutoffs,
                "run_id": run_id,
            }
        )
        metrics_step = MetricsStep(metrics_cfg)
        metrics_step.execute(ctx)

        ranks_dir = ranks_root_dir(base_dir, dataset_name, alias)
        metrics_path = metrics_file_path(base_dir, dataset_name, alias)
        retrieval_rows = _load_retrieval_rows(ranks_dir / "retrieval.tsv")
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))[alias]

        expected = {}
        for cutoff in metrics_cutoffs["map"]:
            expected[f"MAP@{cutoff}"] = compute_map(retrieval_rows, qrels_for_metrics, k=cutoff)
        for cutoff in metrics_cutoffs["recall"]:
            expected[f"Recall@{cutoff}"] = compute_recall(retrieval_rows, qrels_for_metrics, k=cutoff)
        for cutoff in metrics_cutoffs["precision"]:
            expected[f"Precision@{cutoff}"] = compute_precision(retrieval_rows, qrels_for_metrics, k=cutoff)
        for cutoff in metrics_cutoffs["ndcg"]:
            expected[f"nDCG@{cutoff}"] = compute_ndcg(retrieval_rows, qrels_for_metrics, k=cutoff)
        for metric_name, metric_value in expected.items():
            assert metrics_data[metric_name] == pytest.approx(metric_value), f"Mismatch for {metric_name} in {alias}"

        # Edge-case verification per query.
        for qid in ("q-confused", "q-empty"):
            per_rows = [row for row in retrieval_rows if row["query_id"] == qid]
            qrels_single = {qid: sample.qrels[qid]}
            for cutoff in metrics_cutoffs["map"]:
                assert compute_map(per_rows, qrels_single, k=cutoff) == pytest.approx(0.0)
            for cutoff in metrics_cutoffs["recall"]:
                assert compute_recall(per_rows, qrels_single, k=cutoff) == pytest.approx(0.0)
            for cutoff in metrics_cutoffs["precision"]:
                assert compute_precision(per_rows, qrels_single, k=cutoff) == pytest.approx(0.0)
            for cutoff in metrics_cutoffs["ndcg"]:
                assert compute_ndcg(per_rows, qrels_single, k=cutoff) == pytest.approx(0.0)

    assert len(sample.docs) >= 30

    dataset_root_dir = base_dir / "dataset" / dataset_name
    assert dataset_root_dir.exists(), "Dataset root directory should exist"
    actual_dirs = {p.name for p in dataset_root_dir.iterdir() if p.is_dir()}
    expected_dirs = {"extracted", "transformed", "indexes", "ranks", "rerank"}
    assert expected_dirs.issubset(actual_dirs), f"Missing expected subdirectories: {expected_dirs - actual_dirs}"
    assert not (dataset_root_dir / "dataset").exists(), "Nested 'dataset' directory should not be created"
    assert not (Path("outputs") / "dataset").exists(), "Global outputs directory should not contain dangling 'dataset' root"
