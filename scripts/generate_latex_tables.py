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
    python scripts/generate_latex_tables.py --mode dataset-summary  # dataset overview table
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
    dataset-summary – Overview table listing all datasets with #Docs, #Queries,
               Avg Rel/Q, evaluation type, and domain.

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


# ── Dataset metadata (static, computed from GCS category_metadata.json) ─────
# For category-based datasets: Avg Rel/Q = Σ_c [ n_c × (n_c − 1) ] / N
# where n_c is the number of documents in category c and N is total docs.
# This counts, for each query (= every document), how many other docs share
# its category (excluding itself), averaged over all queries.
#
# BEIR Avg Rel/Q values come from the official BEIR benchmark qrels.

DATASET_INFO: Dict[str, Dict[str, Any]] = {
    # --- BEIR datasets (qrels-based) ---
    # Distribution computed from actual qrels files on GCS.
    "arguana": {
        "docs": 8674, "queries": 1406, "avg_rel_q": 1.0,
        "eval_type": "Qrels", "domain": "Argumentation",
        "dist": {"mean": 1.0, "std": 0.0, "min": 1, "p25": 1, "median": 1, "p75": 1, "p90": 1, "p95": 1, "max": 1},
    },
    "scifact": {
        "docs": 5183, "queries": 300, "avg_rel_q": 1.1,
        "eval_type": "Qrels", "domain": "Scientific Claims",
        "dist": {"mean": 1.1, "std": 0.5, "min": 1, "p25": 1, "median": 1, "p75": 1, "p90": 1, "p95": 2, "max": 5},
    },
    "nfcorpus": {
        "docs": 3633, "queries": 323, "avg_rel_q": 38.2,
        "eval_type": "Qrels", "domain": "Biomedical",
        "dist": {"mean": 38.2, "std": 71.7, "min": 1, "p25": 5, "median": 16, "p75": 39, "p90": 71, "p95": 158, "max": 475},
    },
    "scidocs": {
        "docs": 25657, "queries": 1000, "avg_rel_q": 4.9,
        "eval_type": "Qrels", "domain": "Scientific Docs",
        "dist": {"mean": 4.9, "std": 0.3, "min": 2, "p25": 5, "median": 5, "p75": 5, "p90": 5, "p95": 5, "max": 5},
    },
    "fiqa": {
        "docs": 57638, "queries": 648, "avg_rel_q": 2.6,
        "eval_type": "Qrels", "domain": "Finance",
        "dist": {"mean": 2.6, "std": 2.1, "min": 1, "p25": 1, "median": 2, "p75": 3, "p90": 5, "p95": 7, "max": 15},
    },
    # --- Category-based datasets ---
    # Avg Rel/Q computed from category_metadata.json on GCS.
    # Each document acts as a query; relevant = same-category docs (excl. itself).
    # Distribution percentiles are weighted by category size:
    # a category of size n contributes n queries each with Rel/Q = n − 1.
    "bbc-news": {
        "docs": 2225, "queries": 2225, "avg_rel_q": 450.6,
        "eval_type": "Category", "n_categories": 5, "domain": "News",
        "category_based": True,
        "dist": {"mean": 450.6, "std": 55.0, "min": 385, "p25": 400, "median": 416, "p75": 509, "p90": 510, "p95": 510, "max": 510},
    },
    "wos": {
        "docs": 46985, "queries": 46985, "avg_rel_q": 389.0,
        "eval_type": "Category", "n_categories": 134, "domain": "Academic",
        "category_based": True,
        "dist": {"mean": 389.0, "std": 119.5, "min": 0, "p25": 338, "median": 383, "p75": 417, "p90": 551, "p95": 651, "max": 749},
    },
    "huffpost-news": {
        "docs": 32730, "queries": 32730, "avg_rel_q": 2044.0,
        "eval_type": "Category", "n_categories": 42, "domain": "News",
        "category_based": True,
        "dist": {"mean": 2044.0, "std": 1865.3, "min": 124, "p25": 618, "median": 1744, "p75": 3539, "p90": 5593, "p95": 5593, "max": 5593},
    },
    "mental-health": {
        "docs": 20353, "queries": 20353, "avg_rel_q": 5593.8,
        "eval_type": "Category", "n_categories": 4, "domain": "Health Forums",
        "category_based": True,
        "dist": {"mean": 5593.8, "std": 1386.8, "min": 2581, "p25": 5264, "median": 5451, "p75": 7053, "p90": 7053, "p95": 7053, "max": 7053},
    },
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

    Layout – two column groups (Baseline / Re-ranked) with one row per
    dataset × model (best method only):

        Dataset | Model | Method (K) | MAP | P@20 | Recall | MAP | P@20 | Recall
                                       ─── Baseline ───   ─── Re-ranked (Δ%) ───

    Datasets are separated by \\midrule.  Uses \\scriptsize and tight
    column spacing to fit on one page.
    """
    metric_keys = ["map@20", "precision@20", "recall@20"]
    metric_short = ["MAP", "P@20", "Recall"]

    lines = [
        "% ── Single-page unified table (paper mode) ──────────────────",
        "% Requires: \\usepackage{booktabs}, \\usepackage{caption}, \\usepackage{graphicx}",
        "\\begingroup",
        "\\renewcommand{\\arraystretch}{0.95}",
        "\\captionof{table}{Re-ranking effectiveness across all datasets. "
        "For each retrieval model the single best UDLF method and $K$ are "
        "selected by highest P@20 improvement.}",
        "\\label{tab:rerank-all}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{ll l ccc ccc}",
        "    \\toprule",
        "    & & & \\multicolumn{3}{c}{Baseline}"
        " & \\multicolumn{3}{c}{Re-ranked ($\\Delta\\%$)} \\\\",
        "    \\cmidrule(lr){4-6} \\cmidrule(lr){7-9}",
        "    Dataset & Model & Method ($K$)"
        + "".join(f" & {m}" for m in metric_short)
        + "".join(f" & {m}" for m in metric_short)
        + " \\\\",
        "    \\midrule",
    ]

    for ds_idx, dataset in enumerate(datasets):
        ds_data = baseline_comparison.get(dataset, {})
        if not ds_data:
            continue

        ds_display = DATASET_DISPLAY.get(dataset, dataset)

        # ── First pass: collect row data per model and track P@20 Δ% ──
        model_rows: list = []  # list of (model, method_label, k_str, base_vals, p20_delta)
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

            # Collect baseline values
            base_vals: list = []
            for mk in metric_keys:
                bv = get_baseline_value(ds_data, model, mk)
                base_vals.append(bv)

            # Compute P@20 Δ%
            p20_base = get_baseline_value(ds_data, model, "precision@20")
            p20_rerank = get_metric_value(
                ds_data, model, best_method, best_k_str, "precision@20", "rerank"
            )
            p20_delta = (
                (p20_rerank - p20_base) / p20_base * 100
                if p20_base and p20_rerank and p20_base != 0
                else -float("inf")
            )

            model_rows.append(
                (model, model_label, method_label, best_method, best_k_str, base_vals, p20_delta)
            )

        if not model_rows:
            continue

        # Find max P@20 Δ% across models in this dataset
        best_p20_delta = max(r[6] for r in model_rows)

        # ── Second pass: emit rows, adding † to the best P@20 model ──
        first_model_in_ds = True
        for model, model_label, method_label, best_method, best_k_str, base_vals, p20_delta in model_rows:
            ds_cell = _escape_latex(ds_display) if first_model_in_ds else ""
            first_model_in_ds = False

            # Mark best P@20 improvement with †
            dagger = "$^{\\dagger}$" if p20_delta == best_p20_delta else ""

            # --- Single row: info | baseline cols | re-ranked cols ---
            row_cells = [
                ds_cell,
                _escape_latex(model_label) + dagger,
                f"{method_label} ({best_k_str})",
            ]

            # Baseline columns
            for bv in base_vals:
                row_cells.append(_fmt(bv))

            # Re-ranked columns with inline Δ%
            for i, mk in enumerate(metric_keys):
                base_val = base_vals[i]
                rerank_val = get_metric_value(
                    ds_data, model, best_method, best_k_str, mk, "rerank"
                )
                improved = (
                    rerank_val is not None
                    and base_val is not None
                    and rerank_val > base_val
                )
                val_str = _fmt(rerank_val, bold=improved)
                if (
                    base_val is not None
                    and rerank_val is not None
                    and base_val != 0
                ):
                    delta_pct = (rerank_val - base_val) / base_val * 100
                    sign = "+" if delta_pct > 0 else ""
                    val_str += (
                        f" {{\\scriptsize ({sign}{delta_pct:.1f}\\%)}}"
                    )
                row_cells.append(val_str)

            lines.append("    " + " & ".join(row_cells) + " \\\\")

        # Separator between dataset groups (except after last)
        if ds_idx < len(datasets) - 1:
            lines.append("    \\midrule")

    lines += [
        "    \\bottomrule",
        "\\end{tabular}",
        "}% end resizebox",
        "\\par\\smallskip",
        "\\parbox{\\textwidth}{\\scriptsize",
        "\\textbf{Bold} = improvement over baseline. "
        "$^{\\dagger}$ = highest P@20 relative gain per dataset. "
        "$\\Delta\\%$ = relative change from baseline.}",
        "\\endgroup",
        "",
    ]
    return "\n".join(lines)


def generate_distribution_table(datasets: List[str]) -> str:
    """
    Build a LaTeX table showing the Rel/Q distribution for ALL datasets.

    Layout – datasets as rows, statistics as columns (compact horizontal):

        Dataset      | Type   | Mean  | Std   | Min | Median | P95 | Max
        -------------|--------|-------|-------|-----|--------|-----|----
        ArguAna      | Qrels  |  1.0  |  0.0  |  1  |   1    |  1  |  1
        ...          |        |       |       |     |        |     |
        BBC-News (5) | Cat.   | 450.6 | 55.0  | 385 |  416   | 510 | 510
        ...          |        |       |       |     |        |     |

    Includes all datasets with a "dist" key in DATASET_INFO.
    """
    ds_with_dist = [d for d in datasets if DATASET_INFO.get(d, {}).get("dist")]
    if not ds_with_dist:
        return "% No datasets with distribution data found.\n"

    def _fmt_num(v: float) -> str:
        """Format numeric value: integer if whole, else 1 decimal. LaTeX thousands sep."""
        if v == int(v):
            n = int(v)
            s = f"{n:,}"
            return s.replace(",", "{,}")
        # Float with 1 decimal
        if v >= 1000:
            int_part = int(v)
            frac = round((v - int_part) * 10)
            s = f"{int_part:,}"
            return s.replace(",", "{,}") + f".{frac}"
        return f"{v:.1f}"

    # Pick statistics that tell the story without being too wide:
    # Mean, Std, Min, Median, P75, P95, Max  (7 stat columns)
    stat_keys = [
        ("Mean", "mean"),
        ("$\\sigma$", "std"),
        ("Min", "min"),
        ("Median", "median"),
        ("P75", "p75"),
        ("P95", "p95"),
        ("Max", "max"),
    ]

    # Separate BEIR / category
    beir_ds = [d for d in ds_with_dist if not DATASET_INFO[d].get("category_based")]
    cat_ds = [d for d in ds_with_dist if DATASET_INFO[d].get("category_based")]

    lines = [
        "% ── Rel/Q distribution across all datasets ───────────────────",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\caption{Distribution of relevant documents per query (Rel/Q). "
        "For BEIR datasets, Rel/Q is the number of judged-relevant documents "
        "per query from official qrels. "
        "For category-based datasets, Rel/Q counts same-category documents "
        "excluding the query itself; percentiles are weighted by category size.}",
        "  \\label{tab:relq-distribution}",
        "  \\small",
        "  \\begin{tabular}{ll rrrrrrr}",
        "    \\toprule",
        "    Dataset & Eval.\\ Type & "
        + " & ".join(h for h, _ in stat_keys)
        + " \\\\",
        "    \\midrule",
    ]

    def _emit_row(ds: str) -> str:
        info = DATASET_INFO[ds]
        dist = info["dist"]
        ds_display = DATASET_DISPLAY.get(ds, ds)
        # Eval type column
        if info.get("category_based"):
            n_cats = info.get("n_categories", "?")
            eval_cell = f"Category ({n_cats})"
        else:
            eval_cell = info.get("eval_type", "Qrels")
        cells = [_escape_latex(ds_display), eval_cell]
        for _, key in stat_keys:
            cells.append(_fmt_num(dist[key]))
        return "    " + " & ".join(cells) + " \\\\"

    for ds in beir_ds:
        lines.append(_emit_row(ds))

    if beir_ds and cat_ds:
        lines.append("    \\midrule")

    for ds in cat_ds:
        lines.append(_emit_row(ds))

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def generate_dataset_summary_table(datasets: List[str]) -> str:
    """
    Build a LaTeX table summarising the datasets used in the experiments.

    Columns: Dataset | #Docs | #Queries | Avg Rel/Q | Eval Type | Domain

    For BEIR datasets, Avg Rel/Q comes from official qrels.
    For category-based datasets, Avg Rel/Q = Σ_c [n_c × (n_c − 1)] / N
    (each document is a query; relevant = other docs in same category).

    Category-based queries are marked with † (every document = a query).
    """
    # Separate BEIR and category-based for the midrule divider
    beir_datasets = [d for d in datasets if not DATASET_INFO.get(d, {}).get("category_based")]
    category_datasets = [d for d in datasets if DATASET_INFO.get(d, {}).get("category_based")]

    lines = [
        "% ── Dataset summary table ─────────────────────────────────────",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\caption{Summary of datasets used in the experiments. "
        "BEIR datasets use standard relevance judgments (qrels); "
        "category-based datasets treat every document as a query and "
        "define relevance by shared category membership (excluding the query itself).}",
        "  \\label{tab:datasets}",
        "  \\small",
        "  \\begin{tabular}{lrrlll}",
        "    \\toprule",
        "    Dataset & \\#Docs & \\#Queries & Avg.\\,Rel/Q & Eval.\\,Type & Domain \\\\",
        "    \\midrule",
    ]

    def _fmt_int(n: int) -> str:
        """Format integer with thousands separator using LaTeX thin space."""
        s = f"{n:,}"
        return s.replace(",", "{,}")

    def _fmt_avg_rel(val: float) -> str:
        """Format Avg Rel/Q: 1 decimal, with thousands separator if large."""
        if val >= 1000:
            # e.g. 2044.0 → "2{,}044.0"
            int_part = int(val)
            frac = val - int_part
            return f"{_fmt_int(int_part)}.{int(frac * 10)}"
        return f"{val:.1f}"

    # BEIR datasets
    for ds in beir_datasets:
        info = DATASET_INFO.get(ds)
        if info is None:
            continue
        ds_display = DATASET_DISPLAY.get(ds, ds)
        lines.append(
            f"    {_escape_latex(ds_display)}"
            f" & {_fmt_int(info['docs'])}"
            f" & {_fmt_int(info['queries'])}"
            f" & {_fmt_avg_rel(info['avg_rel_q'])}"
            f" & {info['eval_type']}"
            f" & {info['domain']} \\\\"
        )

    # Separator between BEIR and category-based
    if beir_datasets and category_datasets:
        lines.append("    \\midrule")

    # Category-based datasets
    for ds in category_datasets:
        info = DATASET_INFO.get(ds)
        if info is None:
            continue
        ds_display = DATASET_DISPLAY.get(ds, ds)
        n_cats = info.get("n_categories", "?")
        lines.append(
            f"    {_escape_latex(ds_display)}"
            f" & {_fmt_int(info['docs'])}"
            f" & {_fmt_int(info['queries'])}$^{{\\dagger}}$"
            f" & {_fmt_avg_rel(info['avg_rel_q'])}"
            f" & Category ({n_cats})"
            f" & {info['domain']} \\\\"
        )

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\par\\smallskip",
        "  \\parbox{\\textwidth}{\\scriptsize",
        "  $^{\\dagger}$ Every document serves as both query and potential result.",
        "  Avg.\\,Rel/Q for category-based datasets counts same-category documents",
        "  (excluding the query itself).}",
        "\\end{table}",
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
        choices=["complete", "compact", "paper", "dataset-summary"],
        default="complete",
        help="Table layout mode: 'complete' shows all models × methods with baselines "
        "(dissertation); 'compact' shows one best row per model with Δ%%%% inline; "
        "'paper' generates a single unified table across all datasets (article); "
        "'dataset-summary' generates the dataset overview table with statistics. "
        "Default: complete",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # ── Dataset-summary mode doesn't need experiment data ───────────────
    if args.mode == "dataset-summary":
        # Use all datasets in canonical order unless --datasets overrides
        all_ds_order = [
            "arguana", "scifact", "nfcorpus", "scidocs", "fiqa",
            "bbc-news", "wos", "huffpost-news", "mental-health",
        ]
        datasets = args.datasets or all_ds_order
        parts = [
            "% Auto-generated by generate_latex_tables.py --mode dataset-summary",
            "% Requires: \\usepackage{booktabs}",
            "",
            generate_dataset_summary_table(datasets),
            generate_distribution_table(datasets),
        ]
        output_text = "\n".join(parts)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"✅ Dataset summary table written to {out_path}")
        print(f"   Datasets: {', '.join(datasets)}")
        return

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
