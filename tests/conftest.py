from __future__ import annotations

from pathlib import Path

import pytest

from test_utils.sample_dataset import SampleDataset, build_sample_dataset, make_sample_retrieval_rows


@pytest.fixture()
def sample_dataset(tmp_path: Path) -> SampleDataset:
    base_dir = tmp_path / "outputs" / "sample-run"
    dataset = build_sample_dataset(base_dir, "synthetic-sample")
    assert len(dataset.docs) >= 30
    return dataset


@pytest.fixture()
def sample_retrieval_rows(sample_dataset: SampleDataset):
    return make_sample_retrieval_rows(sample_dataset)
