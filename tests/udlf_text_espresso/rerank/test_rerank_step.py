from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import List

import pytest
from omegaconf import OmegaConf

from udlf_text_espresso.core.artifacts import Artifact
from udlf_text_espresso.core.paths import (
    dataset_extracted_dir,
    rerank_experiment_config_path,
    rerank_experiment_dir,
    rerank_experiment_metrics_path,
    rerank_experiment_ranks_path,
)
from udlf_text_espresso.core.pipeline import PipelineContext
from udlf_text_espresso.core.storage import LocalStorage
from udlf_text_espresso.runner import ReRankStep
from udlf_text_espresso.rerank.udlf_runner import UDLFReranker
from test_utils.sample_dataset import build_sample_dataset, make_sample_retrieval_rows


@pytest.fixture(autouse=True)
def _ensure_udlf_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guarantee placeholder reranker path is used during tests.
    def mock_rerank(self, rows, ranked_list_path=None, lists_path=None):
        return [
            {**row, "rerank_score": float(row.get("score", 0.0)) * 1.1}
            for row in rows
        ]
    monkeypatch.setattr(UDLFReranker, "rerank", mock_rerank)


def _write_retrieval(tsv_path: Path, rows: List[dict]) -> None:
    lines = [f"{row['query_id']}\t{row['doc_id']}\t{row['score']}" for row in rows]
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_rerank_step_runs_multiple_experiments(tmp_path: Path) -> None:
    dataset_name = "synthetic-rerank"
    index_name = "miniLM"
    method_root = "udlf"
    run_root = tmp_path / "outputs" / "rerank-run"
    sample = build_sample_dataset(run_root, dataset_name)

    retrieval_rows = make_sample_retrieval_rows(sample)
    ranks_dir = run_root / "dataset" / dataset_name / "ranks" / index_name
    retrieval_path = ranks_dir / "retrieval.tsv"
    _write_retrieval(retrieval_path, retrieval_rows)

    ctx = PipelineContext(run_id="rerank-run", storage=LocalStorage(), force=True)
    ctx.register(Artifact(name="retrieval", uri=str(retrieval_path), kind="retrieval"))

    rerank_cfg = OmegaConf.create(
        {
            "dataset_name": dataset_name,
            "index_name": index_name,
            "method": method_root,
            "base_output_dir": str(run_root),
            "limit": None,
            "cutoffs": {"map": 5, "recall": 5, "precision": 5, "ndcg": 5},
            "experiments": [
                {"name": "scale", "method": method_root, "config": {"score_scale": 1.2}},
                {"name": "bias", "method": f"{method_root}-bias", "config": {"score_bias": 0.5}, "limit": 1},
            ],
        }
    )

    rerank_step = ReRankStep(rerank_cfg)
    artifacts = rerank_step.execute(ctx)

    assert any(art.name == "rerank:udlf:scale" for art in artifacts)
    assert any(art.name == "rerank:udlf-bias:bias" for art in artifacts)

    method_for_experiment = {
        "scale": method_root,
        "bias": f"{method_root}-bias",
    }

    # Verify experiment directories and metrics were written.
    for exp_name in ("scale", "bias"):
        exp_method = method_for_experiment[exp_name]
        exp_dir = rerank_experiment_dir(run_root, dataset_name, index_name, exp_method, exp_name)
        assert exp_dir.exists()
        horiz_path = exp_dir / "data.txt"
        trec_path = exp_dir / "data.trec"
        ranks_path = rerank_experiment_ranks_path(run_root, dataset_name, index_name, exp_method, exp_name)
        metrics_path = rerank_experiment_metrics_path(run_root, dataset_name, index_name, exp_method, exp_name)
        config_path = rerank_experiment_config_path(run_root, dataset_name, index_name, exp_method, exp_name)

        assert horiz_path.exists()
        assert trec_path.exists()
        assert ranks_path.exists()
        assert metrics_path.exists()
        assert config_path.exists()

        lines = [line.strip() for line in horiz_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if exp_name == "bias":
            # Limit 1 should enforce single document per query.
            assert all(len(line.split()) == 2 for line in lines)

        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert exp_name in metrics_payload
        metrics = metrics_payload[exp_name]
        expected_metric_keys = {"MAP@5", "Recall@5", "Precision@5", "nDCG@5"}
        assert expected_metric_keys.issubset(metrics.keys())

        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        assert config_payload["method"] in (method_root, f"{method_root}-bias")
        assert config_payload["experiment"] == exp_name
        assert "params" in config_payload
        assert "query_count" in config_payload

    # Ensure qrels file was consumed by verifying it still exists and contains lines.
    qrels_path = dataset_extracted_dir(run_root, dataset_name) / "qrels" / "data.tsv.gz"
    assert qrels_path.exists()
    with gzip.open(qrels_path, "rt", encoding="utf-8") as handle:
        assert any(line.strip() for line in handle)
