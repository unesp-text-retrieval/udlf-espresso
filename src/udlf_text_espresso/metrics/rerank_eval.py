from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .eval import compute_map, compute_ndcg, compute_precision, compute_recall

_DEFAULT_MAP_K = 1000
_DEFAULT_RECALL_K = 1000
_DEFAULT_PRECISION_K = 10
_DEFAULT_NDCG_K = 10


def _normalize_cutoffs(
    values: Union[int, Sequence[int], None],
    *,
    default: int,
    fallback: Optional[Sequence[int]] = None,
) -> List[int]:
    if isinstance(values, (list, tuple, set)):
        candidates = [int(v) for v in values if isinstance(v, (int, float, str))]
    elif values is None:
        candidates = []
    else:
        try:
            candidates = [int(values)]
        except Exception:
            candidates = []

    cleaned = sorted({int(v) for v in candidates if int(v) > 0})
    if cleaned:
        return cleaned
    if fallback is not None and len(fallback) > 0:
        return sorted({int(v) for v in fallback if int(v) > 0})
    return [default]


def load_qrels(qrels_path: Path) -> Dict[str, set]:
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
    qrels: Dict[str, set] = {}
    with gzip.open(qrels_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            qid, doc_id = line.split("\t", 1)
            qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def horizontal_file_to_rows(path: Path, method_name: str, trec_path: Optional[Path] = None) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Horizontal rerank file not found: {path}")

    rows: List[Dict[str, object]] = []
    trec_handle = None
    if trec_path is not None:
        trec_path.parent.mkdir(parents=True, exist_ok=True)
        trec_handle = trec_path.open("w", encoding="utf-8")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            query_id, *doc_ids = parts
            if not doc_ids:
                continue
            for rank, doc_id in enumerate(doc_ids, start=1):
                score = 1.0 / rank  # reciprocal-rank surrogate
                rows.append({
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "score": score,
                    "rank": rank,
                })
                if trec_handle is not None:
                    trec_handle.write(f"{query_id} Q0 {doc_id} {rank} {score:.8f} {method_name}\n")

    if trec_handle is not None:
        trec_handle.close()

    return rows


def _group_rows(
    rows: Iterable[Dict[str, object]],
    *,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, List[Tuple[float, str, Dict[str, object]]]], List[Dict[str, object]]]:
    grouped: Dict[str, List[Tuple[float, str, Dict[str, object]]]] = {}
    for row in rows:
        qid = str(row.get("query_id"))
        doc_id = str(row.get("doc_id"))
        score = float(row.get("rerank_score", row.get("score", 0.0)))
        grouped.setdefault(qid, []).append((score, doc_id, row))

    limited_rows: List[Dict[str, object]] = []
    for qid, entries in grouped.items():
        entries.sort(key=lambda item: item[0], reverse=True)
        if limit is not None and limit > 0:
            entries[:] = entries[:limit]
        next_entries: List[Tuple[float, str, Dict[str, object]]] = []
        for rank, (score, doc_id, original_row) in enumerate(entries, start=1):
            copy_row = {**original_row}
            copy_row["score"] = score
            copy_row["rank"] = rank
            copy_row.setdefault("rerank_score", score)
            limited_rows.append(copy_row)
            next_entries.append((score, doc_id, copy_row))
        grouped[qid] = next_entries
    return grouped, limited_rows


def write_rerank_outputs(
    rows: Iterable[Dict[str, object]],
    *,
    method_label: str,
    horizontal_path: Path,
    trec_path: Path,
    ranks_path: Path,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    horizontal_path.parent.mkdir(parents=True, exist_ok=True)
    trec_path.parent.mkdir(parents=True, exist_ok=True)
    ranks_path.parent.mkdir(parents=True, exist_ok=True)

    grouped, limited_rows = _group_rows(rows, limit=limit)

    with horizontal_path.open("w", encoding="utf-8") as h_out:
        for qid, entries in grouped.items():
            doc_ids = [doc_id for _, doc_id, _ in entries]
            line = " ".join([qid] + doc_ids) if doc_ids else qid
            h_out.write(f"{line}\n")

    with trec_path.open("w", encoding="utf-8") as trec_out:
        for qid, entries in grouped.items():
            for rank, (score, doc_id, _) in enumerate(entries, start=1):
                trec_out.write(f"{qid} Q0 {doc_id} {rank} {score:.8f} {method_label}\n")

    with ranks_path.open("w", encoding="utf-8") as ranks_out:
        for row in limited_rows:
            ranks_out.write(f"{row['query_id']}\t{row['doc_id']}\t{row['score']}\n")

    return limited_rows


def evaluate_rows(
    rows: List[Dict[str, object]],
    qrels: Mapping[str, Iterable[str]],
    *,
    map_k: Union[int, Sequence[int]] = _DEFAULT_MAP_K,
    recall_k: Union[int, Sequence[int]] = _DEFAULT_RECALL_K,
    precision_k: Union[int, Sequence[int]] = _DEFAULT_PRECISION_K,
    ndcg_k: Union[int, Sequence[int]] = _DEFAULT_NDCG_K,
    category_based: bool = False,
    category_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    if not rows:
        raise ValueError("No rows supplied for evaluation")

    # ensure qrels values are sets for faster lookup (not used if category_based=True)
    qrels_sets: Optional[Dict[str, set]] = None
    if not category_based:
        qrels_sets = {
            qid: set(docs if isinstance(docs, set) else set(docs)) for qid, docs in qrels.items()
        }

    map_cutoffs = _normalize_cutoffs(map_k, default=_DEFAULT_MAP_K)
    recall_cutoffs = _normalize_cutoffs(recall_k, default=_DEFAULT_RECALL_K, fallback=map_cutoffs)
    precision_cutoffs = _normalize_cutoffs(precision_k, default=_DEFAULT_PRECISION_K)
    ndcg_cutoffs = _normalize_cutoffs(ndcg_k, default=_DEFAULT_NDCG_K, fallback=precision_cutoffs)

    metrics: Dict[str, float] = {}
    for cutoff in map_cutoffs:
        metrics[f"MAP@{cutoff}"] = compute_map(rows, qrels_sets, k=cutoff, category_based=category_based, category_counts=category_counts)
    for cutoff in recall_cutoffs:
        metrics[f"Recall@{cutoff}"] = compute_recall(rows, qrels_sets, k=cutoff, category_based=category_based, category_counts=category_counts)
    for cutoff in precision_cutoffs:
        metrics[f"Precision@{cutoff}"] = compute_precision(rows, qrels_sets, k=cutoff, category_based=category_based)
    for cutoff in ndcg_cutoffs:
        metrics[f"nDCG@{cutoff}"] = compute_ndcg(rows, qrels_sets, k=cutoff, category_based=category_based, category_counts=category_counts)

    return metrics
    for cutoff in ndcg_cutoffs:
        metrics[f"nDCG@{cutoff}"] = compute_ndcg(rows, qrels_sets, k=cutoff)

    return metrics


def evaluate_rerank_directory(
    run_id: str,
    dataset: str,
    method: str,
    *,
    base_outputs: Optional[Path] = None,
    map_k: int = _DEFAULT_MAP_K,
    recall_k: int = _DEFAULT_RECALL_K,
    precision_k: int = _DEFAULT_PRECISION_K,
    ndcg_k: int = _DEFAULT_NDCG_K,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all experiment subdirectories for a rerank method.

    Each experiment directory is expected to contain a ``data.txt`` file with
    horizontal ranked lists (query followed by document ids). This function
    converts each ``data.txt`` into a TREC-style ``data.trec`` file and writes a
    ``metrics.json`` containing MAP/Prec/Recall/nDCG for that experiment.
    """

    base_outputs = base_outputs or Path("outputs")
    base_outputs = base_outputs.resolve()

    method_root = base_outputs / run_id / "dataset" / dataset / "rerank" / "output" / method
    if not method_root.exists():
        raise FileNotFoundError(f"Method directory not found: {method_root}")

    # Check if this is a category-based dataset
    metadata_path = base_outputs / run_id / "dataset" / dataset / "extracted" / "qrels" / "category_metadata.json"
    is_category_based = False
    category_counts: Optional[Dict[str, int]] = None
    
    if metadata_path.exists():
        try:
            import json
            with open(metadata_path, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)
                if metadata.get("type") == "category_based":
                    is_category_based = True
                    category_counts = metadata.get("categories", {})
        except Exception:
            pass

    qrels_path = base_outputs / run_id / "dataset" / dataset / "extracted" / "qrels" / "data.tsv.gz"
    qrels: Dict[str, set] = {}
    if not is_category_based and qrels_path.exists():
        qrels = load_qrels(qrels_path)

    results: Dict[str, Dict[str, float]] = {}

    for experiment_dir in sorted(d for d in method_root.iterdir() if d.is_dir()):
        horizontal_path = experiment_dir / "data.txt"
        trec_path = experiment_dir / "data.trec"
        metrics_path = experiment_dir / "metrics.json"

        rows = horizontal_file_to_rows(horizontal_path, method, trec_path=trec_path)
        metrics = evaluate_rows(
            rows,
            qrels,
            map_k=map_k,
            recall_k=recall_k,
            precision_k=precision_k,
            ndcg_k=ndcg_k,
            category_based=is_category_based,
            category_counts=category_counts,
        )

        metrics_path.write_text(json.dumps({experiment_dir.name: metrics}, indent=2), encoding="utf-8")
        results[experiment_dir.name] = metrics

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rerank experiment outputs.")
    parser.add_argument("run_id", help="Run identifier, e.g. 'paper-assets'")
    parser.add_argument("dataset", help="Dataset name, e.g. 'scifact'")
    parser.add_argument("method", help="Method/reranker identifier, e.g. 'bge-small-en-v1.5'")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Root directory that holds run outputs")
    parser.add_argument("--map-k", type=int, default=_DEFAULT_MAP_K, help="Cutoff for MAP evaluation")
    parser.add_argument("--recall-k", type=int, default=_DEFAULT_RECALL_K, help="Cutoff for Recall evaluation")
    parser.add_argument("--precision-k", type=int, default=_DEFAULT_PRECISION_K, help="Cutoff for Precision evaluation")
    parser.add_argument("--ndcg-k", type=int, default=_DEFAULT_NDCG_K, help="Cutoff for nDCG evaluation")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = evaluate_rerank_directory(
        run_id=args.run_id,
        dataset=args.dataset,
        method=args.method,
        base_outputs=args.outputs_root,
        map_k=args.map_k,
        recall_k=args.recall_k,
        precision_k=args.precision_k,
        ndcg_k=args.ndcg_k,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
