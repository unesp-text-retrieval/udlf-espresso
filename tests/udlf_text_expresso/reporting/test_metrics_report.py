from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from udlf_text_expresso.reporting import metrics_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.usefixtures("tmp_path")
def test_metrics_report_dataset_filter_writes_to_dataset_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs_root = tmp_path / "outputs"
    run_id = "paper-assets"
    dataset_name = "scidocs"
    index_name = "miniLM-L6-v2"

    run_root = outputs_root / run_id

    baseline_payload = {"retrieval": {"MAP@10": 0.1234, "Recall@10": 0.5678}}
    rerank_payload = {"UDLF-default": {"MAP@10": 0.1500, "Recall@10": 0.6000}}
    config_payload = {"method": "UDLF", "experiment": "UDLF-default"}

    baseline_path = run_root / "dataset" / dataset_name / "ranks" / index_name / "metrics.json"
    rerank_metrics_path = run_root / "dataset" / dataset_name / "rerank" / "output" / index_name / "UDLF-default" / "metrics.json"
    rerank_config_path = rerank_metrics_path.with_name("config.json")

    _write_json(baseline_path, baseline_payload)
    _write_json(rerank_metrics_path, rerank_payload)
    _write_json(rerank_config_path, config_payload)

    argv = [
        "metrics_report",
        run_id,
        "--outputs-root",
        str(outputs_root),
        "--dataset",
        dataset_name,
        "--storage-kind",
        "local",
    ]

    monkeypatch.setattr(sys, "argv", argv)

    metrics_report.main()

    report_path = run_root / "dataset" / dataset_name / "reports" / "udlf_metrics_report.html"
    assert report_path.exists()
    contents = report_path.read_text(encoding="utf-8")
    assert dataset_name in contents
    assert index_name in contents
    assert "UDLF default" in contents
