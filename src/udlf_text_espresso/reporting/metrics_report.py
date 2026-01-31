from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, cast

from hydra import compose, initialize_config_dir

from ..core.storage import StorageBackend, LocalStorage, S3Storage, GCSStorage


@dataclass
class MetricsEntry:
    dataset: str
    index: str
    experiment: str
    method: str
    metrics: Dict[str, float]

    @property
    def label(self) -> str:
        return f"{self.method}:{self.experiment}" if self.experiment else self.method


def _read_json_from_storage(path: Path, storage: Optional[StorageBackend]) -> Dict[str, Dict[str, float]]:
    if storage is None or isinstance(storage, LocalStorage):
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(storage.read_bytes(str(path)).decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metrics payload in {path}")
    return data


def _list_dir_names(base_path: Path, storage: Optional[StorageBackend]) -> List[str]:
    if base_path.exists():
        return sorted(child.name for child in base_path.iterdir() if child.is_dir())
    if storage is not None:
        try:
            return storage.list_children(str(base_path))
        except NotImplementedError:
            return []
    return []


def format_experiment_label(method: str, experiment: str) -> str:
    method_slug = method.lower()
    experiment_slug = experiment.lower()
    suffix = experiment
    if experiment_slug.startswith(method_slug + "-"):
        suffix = experiment[len(method) + 1 :]
    elif experiment_slug.startswith(method_slug):
        suffix = experiment[len(method) :]
    suffix = suffix.lstrip("-_ ")
    if suffix:
        return f"{method} {suffix}"
    return method


def load_baseline_metrics(index_dir: Path, storage: Optional[StorageBackend]) -> Optional[Dict[str, float]]:
    metrics_file = index_dir / "metrics.json"
    if not metrics_file.exists():
        if storage is None or not storage.exists(str(metrics_file)):
            return None
    payload = _read_json_from_storage(metrics_file, storage)
    if not payload:
        return None
    first_key = next(iter(payload))
    metrics = payload[first_key]
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid metrics content in {metrics_file}")
    return {str(k): float(v) for k, v in metrics.items()}


def load_rerank_metrics(dataset_name: str, index_name: str, index_dir: Path, storage: Optional[StorageBackend]) -> Iterable[MetricsEntry]:
    experiment_names = _list_dir_names(index_dir, storage)
    for experiment_name in experiment_names:
        experiment_dir = index_dir / experiment_name
        metrics_file = experiment_dir / "metrics.json"
        config_file = experiment_dir / "config.json"
        if not metrics_file.exists() and (storage is None or not storage.exists(str(metrics_file))):
            continue
        payload = _read_json_from_storage(metrics_file, storage)
        if not payload:
            continue
        first_key = next(iter(payload))
        metrics = payload[first_key]
        if not isinstance(metrics, dict):
            continue
        method = experiment_name.split("-", 1)[0]
        if config_file.exists() or (storage and storage.exists(str(config_file))):
            try:
                if config_file.exists():
                    config_payload = json.loads(config_file.read_text(encoding="utf-8"))
                else:
                    config_payload = json.loads(storage.read_bytes(str(config_file)).decode("utf-8"))
                method = str(config_payload.get("method", method))
                experiment_name = str(config_payload.get("experiment", experiment_name))
            except Exception:
                pass
        yield MetricsEntry(
            dataset=dataset_name,
            index=index_name,
            experiment=experiment_name,
            method=method,
            metrics={str(k): float(v) for k, v in metrics.items()},
        )


def collect_metrics(
    run_root: Path,
    dataset_filter: Optional[Iterable[str]] = None,
    storage: Optional[StorageBackend] = None,
) -> Dict[str, Dict[str, Dict[str, MetricsEntry]]]:
    datasets: Dict[str, Dict[str, Dict[str, MetricsEntry]]] = {}
    dataset_root = run_root / "dataset"
    if not dataset_root.exists():
        if storage is None or not _list_dir_names(dataset_root, storage):
            raise FileNotFoundError(f"Dataset directory not found for run: {dataset_root}")

    if dataset_filter is not None:
        dataset_names = list(dict.fromkeys(str(name) for name in dataset_filter))
        filter_set: Optional[Set[str]] = set(dataset_names)
    else:
        dataset_names = _list_dir_names(dataset_root, storage)
        filter_set = None

    for dataset_name in dataset_names:
        if filter_set and dataset_name not in filter_set:
            continue
        dataset_dir = dataset_root / dataset_name
        ranks_root = dataset_dir / "ranks"
        index_names = _list_dir_names(ranks_root, storage)
        if not index_names:
            continue
        rerank_root = dataset_dir / "rerank" / "output"
        datasets.setdefault(dataset_name, {})
        for index_name in sorted(index_names):
            index_dir = ranks_root / index_name
            baseline = load_baseline_metrics(index_dir, storage)
            if baseline is None:
                continue
            entries: Dict[str, MetricsEntry] = {}
            baseline_entry = MetricsEntry(
                dataset=dataset_name,
                index=index_name,
                experiment="baseline",
                method="baseline",
                metrics=baseline,
            )
            entries["baseline"] = baseline_entry
            rerank_index_dir = rerank_root / index_name
            for entry in load_rerank_metrics(dataset_name, index_name, rerank_index_dir, storage):
                entries[entry.experiment] = entry
            datasets[dataset_name][index_name] = entries
    return datasets


def format_delta(baseline: float, value: float) -> Dict[str, Optional[str]]:
    delta = value - baseline
    pct = None
    if abs(baseline) > 1e-12:
        pct = (delta / baseline) * 100.0
    sign_class = "improved" if delta > 0 else "degraded" if delta < 0 else "neutral"
    return {
        "delta": f"{delta:+.4f}",
        "pct": f"{pct:+.2f}%" if pct is not None else "n/a",
        "class": sign_class,
    }


def build_html(run_id: str, metrics: Dict[str, Dict[str, Dict[str, MetricsEntry]]]) -> str:
    css = """
    body { font-family: Arial, sans-serif; margin: 40px; }
    h1 { margin-bottom: 0.4em; }
    h2 { margin-top: 1.6em; }
    h3 { margin-top: 1.2em; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1.5em; }
    th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: right; }
    th.metric, td.metric { text-align: left; }
    tr:nth-child(even) { background-color: #f8f8f8; }
    .improved { color: #1a7f37; font-weight: 600; }
    .degraded { color: #c0392b; font-weight: 600; }
    .neutral { color: #555; }
    .baseline { background-color: #eef2f7; font-weight: 600; }
    .summary-table td { text-align: left; }
    """
    parts: List[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\" />",
        f"<title>UDLF Metrics Report — {html.escape(run_id)}</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        f"<h1>UDLF Metrics Report — {html.escape(run_id)}</h1>",
        "<p>This report aggregates baseline retrieval metrics and UDLF reranking results across datasets, retrieval models, and experiment configurations. Values reflect the metrics stored in the pipeline outputs.</p>",
    ]

    map_highlights: List[Dict[str, object]] = []

    for dataset_name, indices in metrics.items():
        parts.append(f"<h2>Dataset: {html.escape(dataset_name)}</h2>")
        for index_name, experiments in indices.items():
            parts.append(f"<h3>Retrieval Model: {html.escape(index_name)}</h3>")
            baseline_entry = experiments.get("baseline")
            if baseline_entry is None:
                parts.append("<p><em>Baseline metrics missing.</em></p>")
                continue
            metric_names = sorted(baseline_entry.metrics.keys())
            parts.append("<table>")
            parts.append("<thead><tr><th class=\"metric\">Metric</th><th>Baseline</th>")
            for exp_name, entry in experiments.items():
                if exp_name == "baseline":
                    continue
                display_label = format_experiment_label(entry.method, entry.experiment)
                parts.append(f"<th>{html.escape(display_label)}</th><th>Δ (abs)</th><th>Δ (rel)</th>")
            parts.append("</tr></thead>")
            parts.append("<tbody>")
            for metric_name in metric_names:
                baseline_value = baseline_entry.metrics.get(metric_name)
                if baseline_value is None:
                    continue
                parts.append("<tr>")
                parts.append(f"<td class=\"metric\">{html.escape(metric_name)}</td>")
                parts.append(f"<td class=\"baseline\">{baseline_value:.6f}</td>")
                for exp_name, entry in experiments.items():
                    if exp_name == "baseline":
                        continue
                    value = entry.metrics.get(metric_name)
                    if value is None:
                        parts.append("<td colspan=\"3\"><em>n/a</em></td>")
                        continue
                    delta_info = format_delta(baseline_value, value)
                    parts.append(f"<td>{value:.6f}</td>")
                    parts.append(f"<td class=\"{delta_info['class']}\">{delta_info['delta']}</td>")
                    parts.append(f"<td class=\"{delta_info['class']}\">{delta_info['pct']}</td>")
                parts.append("</tr>")
            parts.append("</tbody></table>")

            # Summary of best improvements per metric
            summary_rows: List[str] = []
            for metric_name in metric_names:
                baseline_value = baseline_entry.metrics.get(metric_name)
                if baseline_value is None:
                    continue
                best_entry: Optional[MetricsEntry] = None
                best_delta: float = 0.0
                for exp_name, entry in experiments.items():
                    if exp_name == "baseline":
                        continue
                    value = entry.metrics.get(metric_name)
                    if value is None:
                        continue
                    delta = value - baseline_value
                    if best_entry is None or delta > best_delta:
                        best_entry = entry
                        best_delta = delta
                if best_entry is None:
                    continue
                delta_info = format_delta(baseline_value, best_entry.metrics[metric_name])
                summary_rows.append(
                    "<tr>"
                    f"<td>{html.escape(metric_name)}</td>"
                    f"<td>{html.escape(format_experiment_label(best_entry.method, best_entry.experiment))}</td>"
                    f"<td class=\"{delta_info['class']}\">{delta_info['delta']}</td>"
                    f"<td class=\"{delta_info['class']}\">{delta_info['pct']}</td>"
                    "</tr>"
                )
            if summary_rows:
                parts.append("<table class=\"summary-table\">")
                parts.append("<thead><tr><th>Metric</th><th>Best Experiment</th><th>Δ (abs)</th><th>Δ (rel)</th></tr></thead>")
                parts.append("<tbody>")
                parts.extend(summary_rows)
                parts.append("</tbody></table>")

            map_metric_names = [name for name in metric_names if name.lower().startswith("map")]
            if map_metric_names:
                best_entry: Optional[MetricsEntry] = None
                best_avg_delta = float("-inf")
                best_avg_pct: Optional[float] = None
                best_deltas: Dict[str, float] = {}
                for exp_name, entry in experiments.items():
                    if exp_name == "baseline":
                        continue
                    deltas: Dict[str, float] = {}
                    pct_values: List[float] = []
                    for metric_name in map_metric_names:
                        baseline_value = baseline_entry.metrics.get(metric_name)
                        value = entry.metrics.get(metric_name)
                        if baseline_value is None or value is None:
                            continue
                        delta = value - baseline_value
                        deltas[metric_name] = delta
                        if abs(baseline_value) > 1e-12:
                            pct_values.append((delta / baseline_value) * 100.0)
                    if not deltas:
                        continue
                    avg_delta = sum(deltas.values()) / len(deltas)
                    avg_pct = sum(pct_values) / len(pct_values) if pct_values else None
                    if avg_delta <= 0:
                        continue
                    if best_entry is None or avg_delta > best_avg_delta:
                        best_entry = entry
                        best_avg_delta = avg_delta
                        best_avg_pct = avg_pct
                        best_deltas = deltas
                if best_entry is not None:
                    map_highlights.append(
                        {
                            "dataset": dataset_name,
                            "index": index_name,
                            "entry": best_entry,
                            "avg_delta": best_avg_delta,
                            "avg_pct": best_avg_pct,
                            "deltas": best_deltas,
                        }
                    )

    # Global summary tables by metric category
    summary_sections = {
        "Recall Metrics": ["Recall@5", "Recall@10", "Recall@20", "Recall@50", "Recall@100", "Recall@200"],
        "MAP Metrics": ["MAP@5", "MAP@10", "MAP@20", "MAP@50", "MAP@100", "MAP@200"],
        "Precision Metrics": ["Precision@5", "Precision@10", "Precision@20", "Precision@50", "Precision@100", "Precision@200"],
    }

    if map_highlights:
        parts.append("<h2>MAP Highlights</h2>")
        parts.append("<table class=\"summary-table\">")
        parts.append(
            "<thead><tr><th>Dataset</th><th>Retrieval Model</th><th>Best Experiment</th><th>Δ MAP (avg)</th><th>Δ MAP (avg %)</th><th>Per-metric Δ</th></tr></thead>"
        )
        parts.append("<tbody>")
        for item in sorted(map_highlights, key=lambda data: float(data["avg_delta"]), reverse=True):
            entry = cast(MetricsEntry, item["entry"])
            label = format_experiment_label(entry.method, entry.experiment)
            avg_delta = float(item["avg_delta"])
            avg_pct = item["avg_pct"]
            deltas = cast(Dict[str, float], item["deltas"])
            details = ", ".join(
                f"{metric_name} {delta:+.4f}" for metric_name, delta in sorted(deltas.items())
            )
            avg_pct_text = f"{avg_pct:+.2f}%" if isinstance(avg_pct, float) else "n/a"
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(item['dataset']))}</td>"
                f"<td>{html.escape(str(item['index']))}</td>"
                f"<td>{html.escape(label)}</td>"
                f"<td class=\"improved\">{avg_delta:+.4f}</td>"
                f"<td class=\"improved\">{avg_pct_text}</td>"
                f"<td>{html.escape(details)}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

    # Flatten all entries to look up improvements globally
    flat_entries: List[MetricsEntry] = []
    baseline_lookup: Dict[tuple[str, str], MetricsEntry] = {}
    for dataset_name, indices in metrics.items():
        for index_name, experiments in indices.items():
            baseline_entry = experiments.get("baseline")
            if baseline_entry is not None:
                baseline_lookup[(dataset_name, index_name)] = baseline_entry
            for exp_name, entry in experiments.items():
                if exp_name == "baseline":
                    continue
                flat_entries.append(entry)

    def find_best(metric_name: str) -> Optional[tuple[MetricsEntry, float, float]]:
        best: Optional[MetricsEntry] = None
        best_delta = float("-inf")
        best_pct = 0.0
        for entry in flat_entries:
            baseline_entry = baseline_lookup.get((entry.dataset, entry.index))
            if baseline_entry is None:
                continue
            baseline_value = baseline_entry.metrics.get(metric_name)
            value = entry.metrics.get(metric_name)
            if baseline_value is None or value is None:
                continue
            delta = value - baseline_value
            pct = (delta / baseline_value * 100.0) if abs(baseline_value) > 1e-12 else 0.0
            if delta > best_delta:
                best = entry
                best_delta = delta
                best_pct = pct
        if best is None or best_delta <= 0:
            return None
        return best, best_delta, best_pct

    for section_title, metric_list in summary_sections.items():
        section_rows: List[str] = []
        for metric_name in metric_list:
            result = find_best(metric_name)
            if result is None:
                continue
            entry, delta_value, pct_value = result
            label = format_experiment_label(entry.method, entry.experiment)
            section_rows.append(
                "<tr>"
                f"<td>{html.escape(metric_name)}</td>"
                f"<td>{html.escape(entry.dataset)}</td>"
                f"<td>{html.escape(entry.index)}</td>"
                f"<td>{html.escape(label)}</td>"
                f"<td class=\"improved\">{delta_value:+.4f}</td>"
                f"<td class=\"improved\">{pct_value:+.2f}%</td>"
                "</tr>"
            )
        if section_rows:
            parts.append(f"<h2>{html.escape(section_title)}</h2>")
            parts.append("<table class=\"summary-table\">")
            parts.append("<thead><tr><th>Metric</th><th>Dataset</th><th>Retrieval Model</th><th>Best Experiment</th><th>Δ (abs)</th><th>Δ (rel)</th></tr></thead>")
            parts.append("<tbody>")
            parts.extend(section_rows)
            parts.append("</tbody></table>")

    parts.append("<p style=\"margin-top: 2em; color: #555;\">Generated by udlf_text_espresso.reporting.metrics_report.</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _load_default_storage_config() -> tuple[str, Optional[str], Optional[str]]:
    config_dir = Path(__file__).resolve().parents[3] / "conf"
    with initialize_config_dir(config_dir=str(config_dir), job_name="metrics_report_config", version_base="1.3"):
        cfg = compose(config_name="config")
    kind = str(getattr(cfg.storage, "kind", "local"))
    bucket = getattr(cfg.storage, "bucket", None)
    prefix = getattr(cfg.storage, "prefix", None)
    if isinstance(prefix, str):
        prefix = prefix.strip("/")
    return kind, bucket, prefix


def _instantiate_storage(kind: Optional[str], bucket: Optional[str], prefix: Optional[str]) -> Optional[StorageBackend]:
    normalized_kind = (kind or "local").lower()
    safe_prefix = (prefix or "").strip("/")
    if normalized_kind == "local":
        return LocalStorage()
    if normalized_kind == "s3":
        if not bucket:
            raise ValueError("storage bucket is required for s3 storage")
        return S3Storage(bucket=bucket, prefix=safe_prefix)
    if normalized_kind == "gcs":
        if not bucket:
            raise ValueError("storage bucket is required for gcs storage")
        return GCSStorage(bucket=bucket, prefix=safe_prefix)
    raise ValueError(f"Unsupported storage kind '{kind}'")


def _resolve_storage(kind: Optional[str], bucket: Optional[str], prefix: Optional[str]) -> Optional[StorageBackend]:
    storage_kind = kind
    storage_bucket = bucket
    storage_prefix = prefix
    if storage_kind is None:
        try:
            default_kind, default_bucket, default_prefix = _load_default_storage_config()
            storage_kind = default_kind
            if storage_bucket is None:
                storage_bucket = default_bucket
            if storage_prefix is None:
                storage_prefix = default_prefix
        except Exception:
            storage_kind = storage_kind or "local"
    try:
        return _instantiate_storage(storage_kind, storage_bucket, storage_prefix)
    except Exception:
        if storage_kind and storage_kind.lower() != "local":
            raise
    return LocalStorage()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML report comparing baseline and UDLF rerank metrics.")
    parser.add_argument("run_id", help="Run identifier, e.g. 'paper-assets'")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Root directory that holds run outputs")
    parser.add_argument("--output", type=Path, default=None, help="Optional explicit path for the HTML report")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=None,
        help="Dataset name to include in the report. Repeat to include multiple datasets.",
    )
    parser.add_argument("--storage-kind", choices=["local", "s3", "gcs"], default=None, help="Override storage backend (defaults to project config)")
    parser.add_argument("--storage-bucket", default=None, help="Bucket name for remote storage backends")
    parser.add_argument("--storage-prefix", default=None, help="Prefix/path within the storage bucket")
    args = parser.parse_args()

    run_root = args.outputs_root / args.run_id
    storage = _resolve_storage(args.storage_kind, args.storage_bucket, args.storage_prefix)
    if not run_root.exists():
        dataset_root = run_root / "dataset"
        if storage is None or not _list_dir_names(dataset_root, storage):
            raise FileNotFoundError(f"Run outputs not found at {run_root}")

    dataset_filters: Optional[List[str]] = list(args.datasets) if args.datasets else None
    metrics = collect_metrics(run_root, dataset_filters, storage)
    if not metrics:
        dataset_msg = " for dataset(s): " + ", ".join(dataset_filters) if dataset_filters else ""
        raise SystemExit(f"No metrics found to report{dataset_msg}.")

    html_content = build_html(args.run_id, metrics)
    if args.output is None:
        if dataset_filters and len(dataset_filters) == 1:
            dataset_name = dataset_filters[0]
            report_path = run_root / "dataset" / dataset_name / "reports" / "udlf_metrics_report.html"
        else:
            report_path = run_root / "reports" / "udlf_metrics_report.html"
    else:
        report_path = args.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_content, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
