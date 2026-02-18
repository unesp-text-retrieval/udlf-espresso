#!/usr/bin/env python3
"""
Generate LaTeX tables summarising re-ranking effectiveness per dataset.

For every dataset the script:
  1. Reads experiment_progress_data.json (the same file used by other reports).
  2. For each (dataset, retrieval-model, re-ranking-method) triple it picks
     the K value that **maximises Precision@20 improvement**.
  3. Emits one LaTeX table per dataset showing baselines and best-K re-ranking
     results across MAP, Precision and Recall at the three available cutoffs
     (@20, @50, @200).

Usage:
    python scripts/generate_latex_tables.py
    python scripts/generate_latex_tables.py --mode compact      # paper-friendly small tables
    python scripts/generate_latex_tables.py --mode paper        # single-page all-datasets table
    python scripts/generate_latex_tables.py --mode complete      # full per-model detail (dissertation)
    python scripts/generate_latex_tables.py --data scripts/experiment_progress_data.json
    python scripts/generate_latex_tables.py --cutoffs 20 200
    python scripts/generate_latex_tables.py --datasets scifact bbc-news
    python scripts/generate_latex_tables.py --output outputs/paper-assets/tables.tex

Modes:
    complete  – One table per dataset with every model × method combination,
               baseline rows and all requested cutoffs (great for dissertations).
    compact   – One table per dataset with only @20, one row per model showing
               the single best re-ranking method, values include Δ% inline
               (great for space-constrained papers).
    paper     – A single unified table spanning all datasets on one page,
               one row per (dataset, model) with the best method and Δ%.

The generated .tex file can be \\input{} directly from a LaTeX article.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Pretty names ────────────────────────────────────────────────────────────
MODEL_DISPLAY: Dict[str, str] = {
    "bm25": "BM25",
    "miniLM-L6-v2": "MiniLM",
    "bge-small-en-v1.5": "BGE",
    "contriever-msmarco": "Contriever",
}

METHOD_DISPLAY: Dict[str, str] = {
    "cprr": "CPRR",
    "bfstree": "BFS-TREE",
    "rdpac": "RDPAC",
}

DATASET_DISPLAY: Dict[str, str] = {
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "scidocs": "SciDocs",
    "fiqa": "FiQA",
    "wos": "WOS",
    "bbc-news": "BBC-News",
    "huffpost-news": "HuffPost",
    "mental-health": "Mental-Health",
}

# The three cutoff families present in the data
ALL_CUTOFFS = [20, 50, 200]

# Metric families
METRIC_FAMILIES = ["map", "precision", "recall"]

METRIC_FAMILY_DISPLAY = {
    "map": "MAP",
    "precision": "P",
    "recall": "Recall",
}


# ── Data helpers ────────────────────────────────────────────────────────────

def load_data(path: str | Path) -> Dict[str, Any]:
    """Load experiment_progress_data.json."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def find_best_k(
    dataset_data: Dict,
    model: str,
    method: str,
) -> Optional[Tuple[str, float]]:
    """Return (k_str, precision@20 improvement%) for the best K, or None."""
    method_data = dataset_data.get(model, {}).get(method, {})
    if not method_data:
        return None

    best_k: Optional[str] = None
    best_improvement = -float("inf")

    for k_str, metrics in method_data.items():
        p20 = metrics.get("precision@20", {})
        improvement = p20.get("improvement_pct", -float("inf"))
        k_num = int(k_str)

        # Pick highest improvement; break ties with lower K (more efficient)
        if improvement > best_improvement or (
            improvement == best_improvement
            and best_k is not None
            and k_num < int(best_k)
        ):
            best_improvement = improvement
            best_k = k_str

    if best_k is None:
        return None
    return best_k, best_improvement


def get_metric_value(
    dataset_data: Dict,
    model: str,
    method: str,
    k_str: str,
    metric_key: str,
    field: str = "rerank",
) -> Optional[float]:
    """Retrieve a single metric value from the nested dict."""
    try:
        return dataset_data[model][method][k_str][metric_key][field]
    except (KeyError, TypeError):
        return None


def get_baseline_value(
    dataset_data: Dict,
    model: str,
    metric_key: str,
) -> Optional[float]:
    """Retrieve the baseline value for a model/metric (same across all methods/k)."""
    model_data = dataset_data.get(model, {})
    # Pick the first available method/k to read the baseline
    for _method, k_dict in model_data.items():
        for _k, metrics in k_dict.items():
            entry = metrics.get(metric_key, {})
            if "baseline" in entry:
                return entry["baseline"]
    return None


# ── LaTeX generation ────────────────────────────────────────────────────────

def _fmt(value: Optional[float], bold: bool = False) -> str:
    """Format a metric value to 4 decimal places for the table."""
    if value is None:
        return "--"
    s = f"{value:.4f}"
    if bold:
        return f"\\textbf{{{s}}}"
    return s


def _escape_latex(text: str) -> str:
    """Escape minimal LaTeX special characters in display names."""
    return text.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def generate_table_for_dataset(
    dataset: str,
    baseline_comparison: Dict,
    models: List[str],
    methods: List[str],
    cutoffs: List[int],
) -> str:
    """Build a complete LaTeX table string for one dataset."""
    dataset_data = baseline_comparison.get(dataset, {})
    if not dataset_data:
        return f"% No data found for dataset: {dataset}\n"

    # ── Determine columns ───────────────────────────────────────────────
    # Column groups: one group per cutoff, each with MAP, P, Recall
    n_metric_cols = len(cutoffs) * len(METRIC_FAMILIES)
    col_spec = "l" + "".join(
        f"{'|' if i == 0 else ''}{'c' * len(METRIC_FAMILIES)}"
        for i, _ in enumerate(cutoffs)
    )
    # We want a vertical line before each cutoff group for readability
    # e.g. l|ccc|ccc|ccc
    col_spec = "l" + "|".join(["ccc"] * len(cutoffs))

    # ── Build header ────────────────────────────────────────────────────
    # Two-row header: top row = cutoff groups, bottom row = metric names
    header_top_parts = [""]
    header_bot_parts = ["Method"]
    for cutoff in cutoffs:
        span = len(METRIC_FAMILIES)
        header_top_parts.append(
            f"\\multicolumn{{{span}}}{{c}}{{@{cutoff}}}"
        )
        for fam in METRIC_FAMILIES:
            header_bot_parts.append(METRIC_FAMILY_DISPLAY[fam])

    header_top = " & ".join(header_top_parts) + " \\\\"
    header_bot = " & ".join(header_bot_parts) + " \\\\"

    # ── Build rows ──────────────────────────────────────────────────────
    rows: List[str] = []

    for model in models:
        if model not in dataset_data:
            continue

        model_label = MODEL_DISPLAY.get(model, model)

        # --- Baseline row ---
        baseline_cells = [f"{_escape_latex(model_label)} (base)"]
        for cutoff in cutoffs:
            for fam in METRIC_FAMILIES:
                metric_key = f"{fam}@{cutoff}"
                val = get_baseline_value(dataset_data, model, metric_key)
                baseline_cells.append(_fmt(val))
        rows.append(" & ".join(baseline_cells) + " \\\\")

        # --- Re-ranking rows (one per method, using best K) ---
        for method in methods:
            best = find_best_k(dataset_data, model, method)
            if best is None:
                continue
            best_k_str, best_p20_improvement = best
            method_label = METHOD_DISPLAY.get(method, method.upper())
            k_display = best_k_str

            row_cells = [f"\\quad + {method_label} ($K$={k_display})"]

            # Collect all reranked values + baselines for bolding logic
            rerank_vals = {}
            baseline_vals = {}
            for cutoff in cutoffs:
                for fam in METRIC_FAMILIES:
                    metric_key = f"{fam}@{cutoff}"
                    rv = get_metric_value(
                        dataset_data, model, method, best_k_str, metric_key, "rerank"
                    )
                    bv = get_baseline_value(dataset_data, model, metric_key)
                    rerank_vals[metric_key] = rv
                    baseline_vals[metric_key] = bv

            for cutoff in cutoffs:
                for fam in METRIC_FAMILIES:
                    metric_key = f"{fam}@{cutoff}"
                    rv = rerank_vals[metric_key]
                    bv = baseline_vals[metric_key]
                    # Bold if the reranked value improved over baseline
                    improved = (
                        rv is not None
                        and bv is not None
                        and rv > bv
                    )
                    row_cells.append(_fmt(rv, bold=improved))

            rows.append(" & ".join(row_cells) + " \\\\")

        # Thin rule between model groups (except after last)
        rows.append("\\midrule")

    # Remove trailing midrule
    if rows and rows[-1] == "\\midrule":
        rows.pop()

    # ── Assemble table ──────────────────────────────────────────────────
    ds_display = DATASET_DISPLAY.get(dataset, dataset)
    lines = [
        f"% ── Table for {ds_display} {'─' * 50}",
        f"\\begin{{table}}[htbp]",
        f"  \\centering",
        f"  \\caption{{Re-ranking effectiveness on \\textbf{{{ds_display}}}. "
        f"Best $K$ is chosen per model/method by highest P@20 improvement. "
        f"Bold values indicate improvement over the retrieval baseline.}}",
        f"  \\label{{tab:rerank-{dataset}}}",
        f"  \\small",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        f"    \\toprule",
        f"    {header_top}",
        f"    {header_bot}",
        f"    \\midrule",
    ]
    for row in rows:
        lines.append(f"    {row}")
    lines += [
        f"    \\bottomrule",
        f"  \\end{{tabular}}",
        f"\\end{{table}}",
        "",
    ]
    return "\n".join(lines)


def generate_compact_table_for_dataset(
    dataset: str,
    baseline_comparison: Dict,
    models: List[str],
    methods: List[str],
) -> str:
    """
    Build a compact LaTeX table for one dataset (paper-friendly).

    Only @20 metrics. For each model: a baseline row followed by a single
    best-method row with Δ% inline. This keeps each cell to one number,
    avoiding wide columns.
    """
    dataset_data = baseline_comparison.get(dataset, {})
    if not dataset_data:
        return f"% No data found for dataset: {dataset}\n"

    ds_display = DATASET_DISPLAY.get(dataset, dataset)
    metric_keys = ["map@20", "precision@20", "recall@20"]
    metric_headers = ["MAP@20", "P@20", "Recall@20"]

    lines = [
        f"% ── Compact table for {ds_display} {'─' * 44}",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Best re-ranking result per model on \\textbf{{{ds_display}}} "
        "(selected by highest P@20 improvement). "
        "Bold values indicate improvement over the retrieval baseline.}",
        f"  \\label{{tab:rerank-compact-{dataset}}}",
        "  \\small",
        "  \\begin{tabular}{lccc}",
        "    \\toprule",
        "    Method & " + " & ".join(metric_headers) + " \\\\",
        "    \\midrule",
    ]

    for model in models:
        if model not in dataset_data:
            continue

        model_label = MODEL_DISPLAY.get(model, model)

        # Find the single best method for this model
        best_method: Optional[str] = None
        best_k_str: Optional[str] = None
        best_improvement = -float("inf")

        for method in methods:
            result = find_best_k(dataset_data, model, method)
            if result is None:
                continue
            k_str, improvement = result
            if improvement > best_improvement:
                best_improvement = improvement
                best_method = method
                best_k_str = k_str

        if best_method is None or best_k_str is None:
            continue

        # --- Baseline row ---
        baseline_cells = [f"{_escape_latex(model_label)} (base)"]
        for mk in metric_keys:
            base_val = get_baseline_value(dataset_data, model, mk)
            baseline_cells.append(_fmt(base_val))
        lines.append("    " + " & ".join(baseline_cells) + " \\\\")

        # --- Best re-ranking row with inline Δ% ---
        method_label = METHOD_DISPLAY.get(best_method, best_method.upper())
        rerank_cells = [f"\\quad + {method_label} ($K$={best_k_str})"]

        for mk in metric_keys:
            base_val = get_baseline_value(dataset_data, model, mk)
            rerank_val = get_metric_value(
                dataset_data, model, best_method, best_k_str, mk, "rerank"
            )
            improved = (
                rerank_val is not None
                and base_val is not None
                and rerank_val > base_val
            )
            val_str = _fmt(rerank_val, bold=improved)
            # Append inline Δ%
            if (
                base_val is not None
                and rerank_val is not None
                and base_val != 0
            ):
                delta_pct = (rerank_val - base_val) / base_val * 100
                sign = "+" if delta_pct > 0 else ""
                val_str += f" {{\\scriptsize ({sign}{delta_pct:.1f}\\%)}}"
            rerank_cells.append(val_str)

        lines.append("    " + " & ".join(rerank_cells) + " \\\\")
        lines.append("    \\midrule")

    # Remove trailing midrule
    if lines and lines[-1].strip() == "\\midrule":
        lines.pop()

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def generate_summary_table(
    baseline_comparison: Dict,
    datasets: List[str],
    models: List[str],
    methods: List[str],
) -> str:
    """
    Build a single summary table across all datasets.

    For each dataset, show baseline row + best re-ranking row (best
    model+method+K by P@20 improvement) with inline Δ%.
    """
    lines = [
        "% ── Summary table across all datasets ────────────────────────",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Best re-ranking configuration per dataset (selected by "
        "highest P@20 improvement). Bold values indicate improvement "
        "over the retrieval baseline.}",
        "  \\label{tab:rerank-summary}",
        "  \\small",
        "  \\begin{tabular}{lllccc}",
        "    \\toprule",
        "    Dataset & Model & Method ($K$) & MAP@20 & P@20 & Recall@20 \\\\",
        "    \\midrule",
    ]

    for dataset in datasets:
        ds_data = baseline_comparison.get(dataset, {})
        if not ds_data:
            continue

        # Find global best (model, method, K) for this dataset
        best_combo: Optional[Tuple[str, str, str, float]] = None
        for model in models:
            for method in methods:
                result = find_best_k(ds_data, model, method)
                if result is None:
                    continue
                k_str, improvement = result
                if best_combo is None or improvement > best_combo[3]:
                    best_combo = (model, method, k_str, improvement)

        if best_combo is None:
            continue

        bm, bmethod, bk, _ = best_combo
        ds_display = DATASET_DISPLAY.get(dataset, dataset)
        model_display = MODEL_DISPLAY.get(bm, bm)
        method_display = METHOD_DISPLAY.get(bmethod, bmethod.upper())

        # --- Baseline row ---
        base_cells = [_escape_latex(ds_display), _escape_latex(model_display), "(base)"]
        for metric_key in ["map@20", "precision@20", "recall@20"]:
            base_val = get_baseline_value(ds_data, bm, metric_key)
            base_cells.append(_fmt(base_val))
        lines.append("    " + " & ".join(base_cells) + " \\\\")

        # --- Best re-ranking row with inline Δ% ---
        rerank_cells = ["", "", f"{method_display} ({bk})"]
        for metric_key in ["map@20", "precision@20", "recall@20"]:
            base_val = get_baseline_value(ds_data, bm, metric_key)
            rerank_val = get_metric_value(ds_data, bm, bmethod, bk, metric_key, "rerank")
            improved = (
                rerank_val is not None
                and base_val is not None
                and rerank_val > base_val
            )
            val_str = _fmt(rerank_val, bold=improved)
            if base_val is not None and rerank_val is not None and base_val != 0:
                delta_pct = (rerank_val - base_val) / base_val * 100
                sign = "+" if delta_pct > 0 else ""
                val_str += f" {{\\scriptsize ({sign}{delta_pct:.1f}\\%)}}"
            rerank_cells.append(val_str)
        lines.append("    " + " & ".join(rerank_cells) + " \\\\")
        lines.append("    \\midrule")

    # Remove trailing midrule
    if lines and lines[-1].strip() == "\\midrule":
        lines.pop()

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def generate_paper_table(
    baseline_comparison: Dict,
    datasets: List[str],
    models: List[str],
    methods: List[str],
) -> str:
    """
    Build a single-page unified table with ALL results across every dataset.

    Layout (one row per dataset × model, best method only):
        Dataset | Model | Method (K) | MAP@20 | P@20 | Recall@20

    Each metric cell shows the re-ranked value with inline Δ%.
    Datasets are separated by \\midrule.  Uses \\footnotesize and tight
    column spacing to fit ~35 rows on one page.
    """
    metric_keys = ["map@20", "precision@20", "recall@20"]

    lines = [
        "% ── Single-page unified table (paper mode) ──────────────────",
        "\\begin{table*}[htbp]",
        "  \\centering",
        "  \\caption{Re-ranking effectiveness across all datasets. "
        "For each model the single best UDLF method and $K$ are selected "
        "by highest P@20 improvement. "
        "Bold indicates improvement over the retrieval baseline.}",
        "  \\label{tab:rerank-all}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{ll l ccc}",
        "    \\toprule",
        "    Dataset & Model & Method ($K$) & MAP@20 & P@20 & Recall@20 \\\\",
        "    \\midrule",
    ]

    for ds_idx, dataset in enumerate(datasets):
        ds_data = baseline_comparison.get(dataset, {})
        if not ds_data:
            continue

        ds_display = DATASET_DISPLAY.get(dataset, dataset)
        first_model_in_ds = True

        for model in models:
            if model not in ds_data:
                continue

            model_label = MODEL_DISPLAY.get(model, model)

            # Find single best method for this model
            best_method: Optional[str] = None
            best_k_str: Optional[str] = None
            best_improvement = -float("inf")

            for method in methods:
                result = find_best_k(ds_data, model, method)
                if result is None:
                    continue
                k_str, improvement = result
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_method = method
                    best_k_str = k_str

            if best_method is None or best_k_str is None:
                continue

            method_label = METHOD_DISPLAY.get(best_method, best_method.upper())

            # Only print dataset name on the first row of each group
            ds_cell = _escape_latex(ds_display) if first_model_in_ds else ""
            first_model_in_ds = False

            row_cells = [
                ds_cell,
                _escape_latex(model_label),
                f"{method_label} ({best_k_str})",
            ]

            for mk in metric_keys:
                base_val = get_baseline_value(ds_data, model, mk)
                rerank_val = get_metric_value(
                    ds_data, model, best_method, best_k_str, mk, "rerank"
                )
                improved = (
                    rerank_val is not None
                    and base_val is not None
                    and rerank_val > base_val
                )
                val_str = _fmt(rerank_val, bold=improved)
                # Append inline Δ%
                if (
                    base_val is not None
                    and rerank_val is not None
                    and base_val != 0
                ):
                    delta_pct = (rerank_val - base_val) / base_val * 100
                    sign = "+" if delta_pct > 0 else ""
                    val_str += f" {{\\scriptsize ({sign}{delta_pct:.1f}\\%)}}"
                row_cells.append(val_str)

            lines.append("    " + " & ".join(row_cells) + " \\\\")

        # Separator between dataset groups (except after last)
        if ds_idx < len(datasets) - 1:
            lines.append("    \\midrule")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table*}",
        "",
    ]
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for re-ranking effectiveness results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent / "experiment_progress_data.json"),
        help="Path to experiment_progress_data.json (default: scripts/experiment_progress_data.json)",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "paper-assets"
            / "latex_rerank_tables.tex"
        ),
        help="Output .tex file path",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to include (default: all found in data)",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=ALL_CUTOFFS,
        help="Cutoff values to include as column groups (default: 20 50 200)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the cross-dataset summary table",
    )
    parser.add_argument(
        "--mode",
        choices=["complete", "compact", "paper"],
        default="complete",
        help="Table layout mode: 'complete' shows all models × methods with baselines "
        "(dissertation); 'compact' shows one best row per model with Δ%%%% inline; "
        "'paper' generates a single unified table across all datasets (article). "
        "Default: complete",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    data = load_data(args.data)
    baseline_comparison = data.get("baseline_comparison", {})
    config = data.get("configuration", {})

    models: List[str] = config.get("models", list(MODEL_DISPLAY.keys()))
    methods: List[str] = config.get("methods", list(METHOD_DISPLAY.keys()))
    datasets: List[str] = args.datasets or config.get(
        "datasets", sorted(baseline_comparison.keys())
    )

    cutoffs = sorted(args.cutoffs)

    # Validate cutoffs exist in data
    valid_cutoffs = set()
    for c in cutoffs:
        key = f"map@{c}"
        # Check if any dataset/model/method/k has this metric
        for ds_data in baseline_comparison.values():
            for m_data in ds_data.values():
                for meth_data in m_data.values():
                    for k_data in meth_data.values():
                        if key in k_data:
                            valid_cutoffs.add(c)
                            break
    cutoffs = sorted(valid_cutoffs & set(cutoffs))
    if not cutoffs:
        print("⚠️  No valid cutoffs found in data. Available cutoffs may differ.")
        sys.exit(1)

    # ── Generate LaTeX content ──────────────────────────────────────────
    parts: List[str] = [
        "% Auto-generated by generate_latex_tables.py",
        "% Requires: \\usepackage{booktabs}",
        "",
    ]

    if args.mode == "paper":
        # Single unified table — no per-dataset loop, no summary
        parts.append(
            generate_paper_table(
                baseline_comparison, datasets, models, methods
            )
        )
    else:
        for dataset in datasets:
            if args.mode == "compact":
                table = generate_compact_table_for_dataset(
                    dataset, baseline_comparison, models, methods
                )
            else:
                table = generate_table_for_dataset(
                    dataset, baseline_comparison, models, methods, cutoffs
                )
            parts.append(table)

        if not args.no_summary:
            parts.append(
                generate_summary_table(
                    baseline_comparison, datasets, models, methods
                )
            )

    output_text = "\n".join(parts)

    # ── Write output ────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output_text, encoding="utf-8")
    print(f"✅ LaTeX tables written to {out_path}")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Cutoffs : {cutoffs}")
    print(f"   Models  : {', '.join(models)}")
    print(f"   Methods : {', '.join(methods)}")


if __name__ == "__main__":
    main()
