from __future__ import annotations
from pathlib import Path


def dataset_root(dataset: str) -> Path:
    return Path(f"dataset/{dataset}")


def run_root_dir(run_id: str | Path) -> Path:
    rid = str(run_id).strip()
    if not rid:
        raise ValueError("run_id must be a non-empty string")
    return Path("outputs") / rid

# Convenience directory builders used by runner
def dataset_extracted_dir(base_out: str | Path, dataset: str) -> Path:
    return Path(base_out) / dataset_root(dataset) / "extracted"

def dataset_transformed_dir(base_out: str | Path, dataset: str) -> Path:
    return Path(base_out) / dataset_root(dataset) / "transformed"

def index_root_dir(base_out: str | Path, dataset: str, index_name: str) -> Path:
    return Path(base_out) / dataset_root(dataset) / Path(f"indexes/{index_name}")

def ranks_root_dir(base_out: str | Path, dataset: str, index_name: str) -> Path:
    return Path(base_out) / dataset_root(dataset) / Path(f"ranks/{index_name}")

def rerank_root_dir(base_out: str | Path, dataset: str, method_name: str) -> Path:
    return Path(base_out) / dataset_root(dataset) / Path(f"rerank/{method_name}")

def metrics_file_path(base_out: str | Path, dataset: str, index_name: str) -> Path:
    return Path(base_out) / dataset_root(dataset) / Path(f"ranks/{index_name}/metrics.json")


def rerank_input_dir(base_out: str | Path, dataset: str, index_name: str) -> Path:
    return Path(base_out) / dataset_root(dataset) / Path(f"rerank/input/{index_name}")


def _slugify_segment(value: str, *, to_lower: bool = False) -> str:
    text = str(value).strip()
    if to_lower:
        text = text.lower()
    for ch in (" ", "\\", "/"):
        text = text.replace(ch, "-")
    return text


def rerank_output_root_dir(base_out: str | Path, dataset: str, index_name: str) -> Path:
    """Return the directory that stores rerank experiment outputs for a retrieval model."""
    return Path(base_out) / dataset_root(dataset) / Path(f"rerank/output/{_slugify_segment(index_name)}")


def rerank_experiment_dir(
    base_out: str | Path,
    dataset: str,
    index_name: str,
    method_name: str,
    experiment_name: str,
) -> Path:
    """Directory containing artifacts for a single rerank experiment."""
    method_slug = _slugify_segment(method_name, to_lower=True)
    experiment_slug = _slugify_segment(experiment_name)
    if experiment_slug:
        if experiment_slug.lower().startswith(method_slug):
            folder = experiment_slug
        else:
            folder = f"{method_slug}-{experiment_slug}"
    else:
        folder = method_slug
    return rerank_output_root_dir(base_out, dataset, index_name) / folder


def rerank_experiment_horizontal_path(
    base_out: str | Path,
    dataset: str,
    index_name: str,
    method_name: str,
    experiment_name: str,
) -> Path:
    return rerank_experiment_dir(base_out, dataset, index_name, method_name, experiment_name) / "data.txt"


def rerank_experiment_trec_path(
    base_out: str | Path,
    dataset: str,
    index_name: str,
    method_name: str,
    experiment_name: str,
) -> Path:
    return rerank_experiment_dir(base_out, dataset, index_name, method_name, experiment_name) / "data.trec"


def rerank_experiment_metrics_path(
    base_out: str | Path,
    dataset: str,
    index_name: str,
    method_name: str,
    experiment_name: str,
) -> Path:
    return rerank_experiment_dir(base_out, dataset, index_name, method_name, experiment_name) / "metrics.json"


def rerank_experiment_ranks_path(
    base_out: str | Path,
    dataset: str,
    index_name: str,
    method_name: str,
    experiment_name: str,
) -> Path:
    return rerank_experiment_dir(base_out, dataset, index_name, method_name, experiment_name) / "ranks.tsv"


def rerank_experiment_config_path(
    base_out: str | Path,
    dataset: str,
    index_name: str,
    method_name: str,
    experiment_name: str,
) -> Path:
    return rerank_experiment_dir(base_out, dataset, index_name, method_name, experiment_name) / "config.json"


def rerank_ranked_list_path(base_out: str | Path, dataset: str, index_name: str) -> Path:
    return rerank_input_dir(base_out, dataset, index_name) / "ranked-list" / "data.txt"


def rerank_lists_path(base_out: str | Path, dataset: str, index_name: str) -> Path:
    return rerank_input_dir(base_out, dataset, index_name) / "lists" / "data.txt"


def extracted_path(dataset: str, kind: str) -> Path:
    # kind: topics | documents | qrels
    return dataset_root(dataset) / "extracted" / kind / "data.tsv.gz"


def transformed_path(dataset: str, kind: str, fmt: str = "tsv.gz") -> Path:
    # kind: topics | documents | qrels
    return dataset_root(dataset) / "transformed" / kind / f"data.{fmt}"


def transformed_documents_jsonl(dataset: str) -> Path:
    return dataset_root(dataset) / "transformed" / "documents" / "data.jsonl"


def transformed_documents_and_topics_tsv(dataset: str) -> Path:
    return dataset_root(dataset) / "transformed" / "documents-and-topics" / "data.tsv"


def index_bm25_path(index_name: str) -> Path:
    return Path(f"indexes/{index_name}/bm25.pkl")

def index_bm25_cache_file(index_root: str | Path) -> Path:
    return Path(index_root) / "bm25.pkl"


def ranks_input_path(index_name: str, fmt: str = "tsv.gz") -> Path:
    return Path(f"ranks/{index_name}/input/{index_name}/data.{fmt}")


def method_config_path(index_name: str, method_name: str) -> Path:
    return Path(f"output/{index_name}/{method_name}/config/config.ini")


def method_ranks_raw_path(index_name: str, method_name: str) -> Path:
    return Path(f"output/{index_name}/{method_name}/ranks/raw/data.tsv")


def metrics_path(index_name: str) -> Path:
    return Path(f"ranks/{index_name}/metrics/data.json")
