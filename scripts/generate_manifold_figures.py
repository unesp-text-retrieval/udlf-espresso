#!/usr/bin/env python3
"""
Generate publication-quality UMAP / t-SNE figures illustrating how UDLF
manifold-learning re-ranking reshapes document neighbourhoods.

The script downloads data from GCS (FAISS index, ranked lists, reranked
outputs) and produces static matplotlib figures suitable for inclusion in
a LaTeX article.

Figures produced:
  1. **Embedding scatter**: all documents projected via UMAP, coloured by
     category.  Ideal for category-based datasets (bbc-news, mental-health,
     wos, huffpost-news).
  2. **Before / after overlay**: for a handful of sampled queries the top-K
     neighbourhood is highlighted before and after re-ranking, showing how
     UDLF promotes category-coherent results.
  3. **Category purity bar chart**: aggregated same-category proportion in
     the top-K before vs. after re-ranking, broken down by query category.

Usage examples:
    # BBC-News with BGE + CPRR k=75 (small, 5 categories – ideal demo)
    python scripts/generate_manifold_figures.py \\
        --dataset bbc-news \\
        --model bge-small-en-v1.5 \\
        --method cprr --k 75

    # Mental-Health with Contriever + CPRR k=75 (4 categories)
    python scripts/generate_manifold_figures.py \\
        --dataset mental-health \\
        --model contriever-msmarco \\
        --method cprr --k 75 \\
        --sample-size 5000

    # Use t-SNE instead of UMAP
    python scripts/generate_manifold_figures.py \\
        --dataset bbc-news \\
        --model bge-small-en-v1.5 \\
        --method cprr --k 75 \\
        --reducer tsne

Requires: umap-learn, matplotlib, faiss-cpu, google-cloud-storage, numpy, scikit-learn
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ── Optional heavy imports (fail gracefully) ───────────────────────────
try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore

try:
    import umap  # type: ignore
except ImportError:
    umap = None  # type: ignore

try:
    from sklearn.manifold import TSNE  # type: ignore
except ImportError:
    TSNE = None  # type: ignore

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ── GCS bucket name ────────────────────────────────────────────────────
GCS_BUCKET = "text-udlf-expresso"
GCS_BASE   = "outputs/paper-assets/dataset"

# ── Colour palette (colour-blind friendly, from Tableau 10) ───────────
CATEGORY_COLOURS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


# ═══════════════════════════════════════════════════════════════════════
#  Data helpers
# ═══════════════════════════════════════════════════════════════════════

def _extract_category(doc_id: str) -> str:
    """Extract category prefix from a document ID like 'sport_001.txt' → 'sport'."""
    # Strip optional 'doc::' prefix
    if doc_id.startswith("doc::"):
        doc_id = doc_id[5:]
    # Remove .txt suffix if present
    if doc_id.endswith(".txt"):
        doc_id = doc_id[:-4]
    # Category is everything before the last underscore + numeric suffix
    parts = doc_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    # Fallback: first token before underscore
    if "_" in doc_id:
        return doc_id.split("_", 1)[0]
    return "unknown"


class GCSReader:
    """Thin wrapper around google-cloud-storage for reading experiment files."""

    def __init__(self, bucket_name: str = GCS_BUCKET):
        from google.cloud import storage as gcs_storage
        self.client = gcs_storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self._tmp = tempfile.TemporaryDirectory()

    def read_bytes(self, path: str) -> bytes:
        blob = self.bucket.blob(path)
        return blob.download_as_bytes()

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8")

    def read_json(self, path: str) -> Any:
        return json.loads(self.read_text(path))

    def download_to_tmp(self, path: str, name: str) -> Path:
        local = Path(self._tmp.name) / name
        blob = self.bucket.blob(path)
        blob.download_to_filename(str(local))
        return local

    def exists(self, path: str) -> bool:
        return self.bucket.blob(path).exists()

    def cleanup(self):
        self._tmp.cleanup()


def load_embeddings_from_faiss(
    gcs: GCSReader,
    dataset: str,
    model: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Download FAISS index + docids from GCS and extract all vectors.

    The document-ID mapping is read from ``docids.json`` when available,
    falling back to ``rerank/input/<model>/lists/data.txt`` which
    contains the same ordered list of unique IDs used during indexing.

    Returns:
        (embeddings [N, D], doc_ids [N])
    """
    if faiss is None:
        raise RuntimeError("faiss-cpu is required. Install with: pip install faiss-cpu")

    base_idx = f"{GCS_BASE}/{dataset}/indexes/{model}"
    print(f"  ↓ Downloading FAISS index from gs://{GCS_BUCKET}/{base_idx} ...")

    # --- Resolve document ID list (primary: docids.json, fallback: lists/data.txt) ---
    docids_path = f"{base_idx}/docids.json"
    lists_path = f"{GCS_BASE}/{dataset}/rerank/input/{model}/lists/data.txt"

    if gcs.exists(docids_path):
        docids = gcs.read_json(docids_path)
        doc_ids: List[str] = docids.get("doc_ids", docids) if isinstance(docids, dict) else docids
        print(f"    (doc IDs from docids.json: {len(doc_ids):,})")
    elif gcs.exists(lists_path):
        text = gcs.read_text(lists_path)
        doc_ids = [line.strip() for line in text.strip().split("\n") if line.strip()]
        print(f"    (doc IDs from lists/data.txt fallback: {len(doc_ids):,})")
    else:
        raise FileNotFoundError(
            f"Cannot find document IDs for {dataset}/{model}. "
            f"Neither {docids_path} nor {lists_path} exist in GCS."
        )

    local_idx = gcs.download_to_tmp(f"{base_idx}/faiss.index", f"faiss_{dataset}_{model}.index")
    index = faiss.read_index(str(local_idx))
    n, d = index.ntotal, index.d
    print(f"  ✓ Loaded {n:,} vectors of dim {d}")

    if n != len(doc_ids):
        print(f"  ⚠ Warning: FAISS has {n:,} vectors but doc_ids has {len(doc_ids):,} entries. Using min.")
        count = min(n, len(doc_ids))
        doc_ids = doc_ids[:count]
    else:
        count = n

    # Extract all embeddings
    embeddings = np.zeros((count, d), dtype=np.float32)
    for i in range(count):
        embeddings[i] = index.reconstruct(i)
    return embeddings, doc_ids


def load_ranked_list(gcs: GCSReader, path: str) -> Dict[str, List[str]]:
    """
    Parse a horizontal ranked-list file.

    Format: query_id doc1 doc2 doc3 ...
    Returns: {query_id: [doc1, doc2, ...]}
    """
    text = gcs.read_text(path)
    rankings: Dict[str, List[str]] = {}
    for line in text.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        query_id = parts[0]
        doc_ids = parts[1:]  # first doc may be the query itself (self-retrieval)
        rankings[query_id] = doc_ids
    return rankings


def load_ranked_list_tsv(gcs: GCSReader, path: str) -> Dict[str, List[str]]:
    """
    Parse a vertical TSV ranked-list file (TREC-style: query_id\tdoc_id\tscore).

    Docs are assumed to be sorted by score descending per query (as produced
    by the rerank step).

    Returns: {query_id: [doc1, doc2, ...]}
    """
    text = gcs.read_text(path)
    rankings: Dict[str, List[str]] = defaultdict(list)
    for line in text.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        qid = parts[0]
        did = parts[1]
        rankings[qid].append(did)
    return dict(rankings)


def load_rankings_auto(gcs: GCSReader, data_txt_path: str, ranks_tsv_path: str) -> Dict[str, List[str]]:
    """
    Load rankings by trying data.txt first (horizontal), falling back to
    ranks.tsv (vertical TSV).  Returns the parsed rankings dict.
    """
    if gcs.exists(data_txt_path):
        print(f"    (loading from data.txt)")
        return load_ranked_list(gcs, data_txt_path)
    elif gcs.exists(ranks_tsv_path):
        print(f"    (loading from ranks.tsv fallback)")
        return load_ranked_list_tsv(gcs, ranks_tsv_path)
    else:
        raise FileNotFoundError(
            f"Neither {data_txt_path} nor {ranks_tsv_path} found in GCS."
        )


def compute_category_purity(
    rankings: Dict[str, List[str]],
    top_k: int = 20,
) -> Dict[str, float]:
    """
    For each query category, compute the average proportion of top-K docs
    that share the query's category.

    Returns: {category: purity_proportion}
    """
    cat_totals: Dict[str, List[float]] = defaultdict(list)

    for qid, docs in rankings.items():
        q_cat = _extract_category(qid)
        top_docs = docs[:top_k]
        if not top_docs:
            continue
        same = sum(1 for d in top_docs if _extract_category(d) == q_cat)
        cat_totals[q_cat].append(same / len(top_docs))

    return {cat: float(np.mean(vals)) for cat, vals in cat_totals.items()}


# ═══════════════════════════════════════════════════════════════════════
#  Dimensionality reduction
# ═══════════════════════════════════════════════════════════════════════

def compute_projection(
    embeddings: np.ndarray,
    reducer: str = "umap",
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Project embeddings to 2D using UMAP or t-SNE.

    Args:
        embeddings: array of shape (N, D)
        reducer: "umap" or "tsne"
        random_state: reproducibility seed

    Returns:
        2D array of shape (N, 2)
    """
    if reducer == "umap":
        if umap is None:
            raise RuntimeError("umap-learn is required. Install with: pip install umap-learn")
        print(f"  ◐ Computing UMAP projection ({embeddings.shape[0]:,} pts, dim={embeddings.shape[1]}) ...")
        reducer_obj = umap.UMAP(
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            n_components=2,
            metric="cosine",
            random_state=random_state,
        )
        coords = reducer_obj.fit_transform(embeddings)
    elif reducer == "tsne":
        if TSNE is None:
            raise RuntimeError("scikit-learn is required for t-SNE.")
        print(f"  ◐ Computing t-SNE projection ({embeddings.shape[0]:,} pts, dim={embeddings.shape[1]}) ...")
        reducer_obj = TSNE(
            n_components=2,
            perplexity=kwargs.get("perplexity", 30),
            random_state=random_state,
            metric="cosine",
            init="pca",
        )
        coords = reducer_obj.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reducer: {reducer}")

    print("  ✓ Projection done.")
    return coords


# ═══════════════════════════════════════════════════════════════════════
#  Figure generation
# ═══════════════════════════════════════════════════════════════════════

def _build_colour_map(categories: List[str]) -> Dict[str, str]:
    """Map unique categories to colours deterministically."""
    unique = sorted(set(categories))
    return {
        cat: CATEGORY_COLOURS[i % len(CATEGORY_COLOURS)]
        for i, cat in enumerate(unique)
    }


def figure_embedding_scatter(
    coords_2d: np.ndarray,
    categories: List[str],
    title: str = "Document Embedding Space",
    reducer_name: str = "UMAP",
    figsize: Tuple[float, float] = (7, 6),
) -> plt.Figure:
    """
    Figure 1: scatter of all documents coloured by category.
    """
    cmap = _build_colour_map(categories)
    colours = [cmap[c] for c in categories]

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=colours, s=6, alpha=0.45, edgecolors="none", rasterized=True,
    )
    # Legend
    handles = [
        mpatches.Patch(color=cmap[cat], label=cat.replace("-", " ").title())
        for cat in sorted(cmap)
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
        title="Category",
        title_fontsize=9,
    )
    ax.set_xlabel(f"{reducer_name} 1", fontsize=10)
    ax.set_ylabel(f"{reducer_name} 2", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def figure_before_after_overlay(
    coords_2d: np.ndarray,
    doc_ids: List[str],
    categories: List[str],
    before_rankings: Dict[str, List[str]],
    after_rankings: Dict[str, List[str]],
    sample_queries: List[str],
    top_k: int = 20,
    reducer_name: str = "UMAP",
    method_label: str = "CPRR",
    k_label: str = "75",
    zoom_margin: float = 0.30,
) -> plt.Figure:
    """
    Figure 2: side-by-side panels showing the top-K neighbourhood of
    sampled queries before and after re-ranking.

    Each panel auto-zooms to the bounding box of the query + its top-K
    retrieved documents (with ``zoom_margin`` relative padding) so that
    even small spatial shifts are clearly visible.
    """
    cmap = _build_colour_map(categories)
    id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    n_queries = len(sample_queries)
    fig, axes = plt.subplots(n_queries, 2, figsize=(10, 4.0 * n_queries), dpi=150)
    if n_queries == 1:
        axes = axes.reshape(1, 2)

    for row, qid in enumerate(sample_queries):
        q_cat = _extract_category(qid)
        q_idx = id_to_idx.get(qid)

        # ── Collect indices for both panels to compute a shared zoom ────
        # We use the *union* of before + after doc positions + the query
        # so both panels share the same viewport and are directly comparable.
        all_highlight_indices: List[int] = []
        if q_idx is not None:
            all_highlight_indices.append(q_idx)

        for rankings in (before_rankings, after_rankings):
            for did in rankings.get(qid, [])[:top_k]:
                idx = id_to_idx.get(did)
                if idx is not None:
                    all_highlight_indices.append(idx)

        # Compute shared bounding box with margin
        if all_highlight_indices:
            pts = coords_2d[all_highlight_indices]
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            dx = max(x_max - x_min, 0.5)  # avoid degenerate range
            dy = max(y_max - y_min, 0.5)
            pad_x = dx * zoom_margin
            pad_y = dy * zoom_margin
            xlim = (x_min - pad_x, x_max + pad_x)
            ylim = (y_min - pad_y, y_max + pad_y)
        else:
            xlim = ylim = None  # fall back to matplotlib default

        for col, (label, rankings) in enumerate([
            ("Before re-ranking", before_rankings),
            (f"After {method_label} (K={k_label})", after_rankings),
        ]):
            ax = axes[row, col]

            # Background: all documents, very faint
            bg_colours = [cmap[c] for c in categories]
            ax.scatter(
                coords_2d[:, 0], coords_2d[:, 1],
                c=bg_colours, s=3, alpha=0.10, edgecolors="none", rasterized=True,
            )

            # Highlight top-K
            ranked_docs = rankings.get(qid, [])[:top_k]
            for rank_pos, did in enumerate(ranked_docs):
                idx = id_to_idx.get(did)
                if idx is None:
                    continue
                d_cat = _extract_category(did)
                is_same = d_cat == q_cat
                ax.scatter(
                    coords_2d[idx, 0], coords_2d[idx, 1],
                    c=cmap[d_cat],
                    s=70 if is_same else 40,
                    alpha=0.90,
                    edgecolors="black" if is_same else "gray",
                    linewidths=1.0 if is_same else 0.5,
                    zorder=5,
                )

            # Mark the query itself
            if q_idx is not None:
                ax.scatter(
                    coords_2d[q_idx, 0], coords_2d[q_idx, 1],
                    marker="*", c="gold", s=280, edgecolors="black",
                    linewidths=1.2, zorder=10,
                )

            # Apply shared zoom
            if xlim is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            # Purity annotation
            same_count = sum(
                1 for d in ranked_docs if _extract_category(d) == q_cat
            )
            purity = same_count / len(ranked_docs) if ranked_docs else 0
            ax.set_title(
                f"{label}\nPurity@{top_k} = {purity:.0%}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xlabel(f"{reducer_name} 1", fontsize=8)
            ax.set_ylabel(f"{reducer_name} 2", fontsize=8)
            ax.tick_params(labelsize=7)

        # Row label with query info
        axes[row, 0].annotate(
            f"Query: {qid}\nCategory: {q_cat}",
            xy=(0.02, 0.98), xycoords="axes fraction",
            fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    # Shared legend at bottom
    handles = [
        mpatches.Patch(color=cmap[cat], label=cat.replace("-", " ").title())
        for cat in sorted(cmap)
    ]
    handles.append(Line2D(
        [0], [0], marker="*", color="w", markerfacecolor="gold",
        markeredgecolor="black", markersize=12, label="Query",
    ))
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 6),
        fontsize=8, framealpha=0.9,
    )
    fig.suptitle(
        f"Top-{top_k} Neighbourhood: Before vs. After {method_label} Re-ranking",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    return fig


def figure_purity_comparison(
    before_rankings: Dict[str, List[str]],
    after_rankings: Dict[str, List[str]],
    top_k: int = 20,
    method_label: str = "CPRR",
    k_label: str = "75",
    figsize: Tuple[float, float] = (10, 4.5),
) -> plt.Figure:
    """
    Figure 3: two-panel purity comparison.

    Left panel  – dumbbell chart (connected dots) with a zoomed y-axis so
                  that even small @20 improvements are clearly visible.
    Right panel – Δ% bar chart showing the absolute improvement per category,
                  making the direction and magnitude immediately obvious.
    """
    before_purity = compute_category_purity(before_rankings, top_k)
    after_purity = compute_category_purity(after_rankings, top_k)

    cats = sorted(set(before_purity) | set(after_purity))
    labels = [c.replace("-", " ").title() for c in cats]
    y_pos = np.arange(len(cats))

    before_vals = np.array([before_purity.get(c, 0) for c in cats])
    after_vals = np.array([after_purity.get(c, 0) for c in cats])
    deltas = after_vals - before_vals

    # Colours
    COL_BEFORE = "#4E79A7"
    COL_AFTER = "#E15759"
    COL_POS = "#2CA02C"
    COL_NEG = "#D62728"

    fig, (ax_dot, ax_delta) = plt.subplots(
        1, 2, figsize=figsize, dpi=150,
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35},
        constrained_layout=True,
    )

    # ── Left panel: dumbbell chart ──────────────────────────────────────
    for i in range(len(cats)):
        ax_dot.plot(
            [before_vals[i], after_vals[i]], [y_pos[i], y_pos[i]],
            color="#888888", linewidth=1.5, zorder=1,
        )
    ax_dot.scatter(before_vals, y_pos, s=70, color=COL_BEFORE,
                   label="Before re-ranking", zorder=2, edgecolors="white", linewidths=0.5)
    ax_dot.scatter(after_vals, y_pos, s=70, color=COL_AFTER,
                   label=f"After {method_label} (K={k_label})", zorder=2,
                   edgecolors="white", linewidths=0.5)

    # Zoom the x-axis to magnify differences instead of starting at 0
    all_vals = np.concatenate([before_vals, after_vals])
    val_min, val_max = all_vals.min(), all_vals.max()
    margin = max((val_max - val_min) * 0.35, 0.02)
    ax_dot.set_xlim(max(0, val_min - margin), min(1.0, val_max + margin))

    ax_dot.set_yticks(y_pos)
    ax_dot.set_yticklabels(labels, fontsize=9)
    ax_dot.set_xlabel(f"Same-Category Proportion in Top-{top_k}", fontsize=10)
    ax_dot.set_title(
        f"Category Purity@{top_k}",
        fontsize=11, fontweight="bold",
    )
    ax_dot.legend(fontsize=8, loc="lower right")
    ax_dot.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_dot.grid(axis="x", alpha=0.25)
    ax_dot.invert_yaxis()  # top category at top

    # ── Right panel: Δ% bar chart ───────────────────────────────────────
    bar_colors = [COL_POS if d >= 0 else COL_NEG for d in deltas]
    ax_delta.barh(y_pos, deltas, color=bar_colors, height=0.55, alpha=0.85)

    # Value annotations on each bar
    for i, d in enumerate(deltas):
        ha = "left" if d >= 0 else "right"
        offset = 0.002 if d >= 0 else -0.002
        ax_delta.text(
            d + offset, y_pos[i], f"{d:+.1%}",
            va="center", ha=ha, fontsize=8, fontweight="bold",
            color=bar_colors[i],
        )

    ax_delta.axvline(0, color="black", linewidth=0.8)
    ax_delta.set_yticks(y_pos)
    ax_delta.set_yticklabels([])  # shared with left panel
    ax_delta.set_xlabel(f"Δ Purity@{top_k}", fontsize=10)
    ax_delta.set_title(
        f"Improvement after {method_label}",
        fontsize=11, fontweight="bold",
    )
    ax_delta.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1%}"))
    ax_delta.grid(axis="x", alpha=0.25)
    ax_delta.invert_yaxis()

    fig.suptitle(
        f"{method_label} (K={k_label}) Re-Ranking Effect on Category Purity@{top_k}",
        fontsize=12, fontweight="bold",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Query sampling heuristic
# ═══════════════════════════════════════════════════════════════════════

def _query_purity(qid: str, rankings: Dict[str, List[str]], top_k: int) -> float:
    """Compute purity@top_k for a single query."""
    q_cat = _extract_category(qid)
    docs = rankings.get(qid, [])[:top_k]
    if not docs:
        return 0.0
    return sum(1 for d in docs if _extract_category(d) == q_cat) / len(docs)


def sample_best_improvement_queries(
    before_rankings: Dict[str, List[str]],
    after_rankings: Dict[str, List[str]],
    top_k: int = 20,
    n_per_category: int = 1,
    seed: int = 42,
    candidate_qids: Optional[Set[str]] = None,
) -> List[str]:
    """
    Select the queries with the largest purity@top_k improvement
    (Δ = after − before), ensuring category diversity by picking
    at most ``n_per_category`` per category.

    If ``candidate_qids`` is given, only queries in that set are
    considered (useful when embeddings have been subsampled).

    This guarantees the overlay figure shows the most compelling
    before→after examples.
    """
    # Compute per-query delta
    deltas: List[Tuple[str, float, float, float]] = []  # (qid, delta, before, after)
    pool = candidate_qids if candidate_qids is not None else set(before_rankings.keys())
    for qid in pool:
        b_docs = before_rankings.get(qid, [])[:top_k]
        a_docs = after_rankings.get(qid, [])[:top_k]
        # Skip queries with too few results
        if len(b_docs) < top_k or len(a_docs) < top_k:
            continue
        p_before = _query_purity(qid, before_rankings, top_k)
        p_after = _query_purity(qid, after_rankings, top_k)
        deltas.append((qid, p_after - p_before, p_before, p_after))

    # Sort by delta descending (best improvement first)
    deltas.sort(key=lambda t: t[1], reverse=True)

    # Pick top improvers, at most n_per_category per category
    rng = np.random.default_rng(seed)
    cat_count: Dict[str, int] = defaultdict(int)
    selected: List[str] = []
    for qid, delta, p_b, p_a in deltas:
        cat = _extract_category(qid)
        if cat_count[cat] >= n_per_category:
            continue
        cat_count[cat] += 1
        selected.append(qid)
        print(f"      ★ {qid}  P@{top_k}: {p_b:.0%} → {p_a:.0%}  (Δ={delta:+.0%})")

    return selected


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

DATASET_DISPLAY = {
    "bbc-news": "BBC-News",
    "mental-health": "Mental-Health",
    "huffpost-news": "HuffPost",
    "wos": "WOS",
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "scidocs": "SciDocs",
    "fiqa": "FiQA",
    "arguana": "ArguAna",
}

MODEL_DISPLAY = {
    "bge-small-en-v1.5": "BGE",
    "miniLM-L6-v2": "MiniLM",
    "contriever-msmarco": "Contriever",
}

METHOD_DISPLAY = {
    "cprr": "CPRR",
    "bfstree": "BFS-TREE",
    "rdpac": "RDPAC",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate UMAP/t-SNE manifold visualisation figures.",
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset name (e.g. bbc-news, mental-health).")
    p.add_argument("--model", required=True,
                   help="Dense model name (e.g. bge-small-en-v1.5). Must have a FAISS index.")
    p.add_argument("--method", required=True,
                   help="Re-ranking method (cprr, bfstree, rdpac).")
    p.add_argument("--k", type=int, required=True,
                   help="K value of the re-ranking experiment.")
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of top results to highlight (default: 20).")
    p.add_argument("--reducer", choices=["umap", "tsne"], default="umap",
                   help="Dimensionality-reduction method (default: umap).")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Subsample documents for large datasets (default: all).")
    p.add_argument("--n-sample-queries", type=int, default=1,
                   help="Number of queries to sample per category for the overlay figure (default: 1).")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for figures (default: outputs/paper-assets/figures/).")
    p.add_argument("--format", choices=["pdf", "svg", "png"], default="pdf",
                   help="Figure output format (default: pdf).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--bucket", default=GCS_BUCKET,
                   help=f"GCS bucket (default: {GCS_BUCKET}).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    ds = args.dataset
    model = args.model
    method = args.method
    k_val = args.k
    reducer_name = args.reducer.upper()

    ds_display = DATASET_DISPLAY.get(ds, ds)
    model_display = MODEL_DISPLAY.get(model, model)
    method_display = METHOD_DISPLAY.get(method, method.upper())
    experiment_tag = f"{method}-k{k_val}"

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent.parent / "outputs" / "paper-assets" / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format

    print(f"═══ Manifold Visualisation: {ds_display} / {model_display} / {method_display} K={k_val} ═══")

    # ── 1. Connect to GCS and load data ─────────────────────────────────
    gcs = GCSReader(args.bucket)
    base = f"{GCS_BASE}/{ds}"

    # Embeddings from FAISS
    embeddings, doc_ids = load_embeddings_from_faiss(gcs, ds, model)
    categories = [_extract_category(did) for did in doc_ids]

    # Optional subsampling for very large datasets
    if args.sample_size and len(doc_ids) > args.sample_size:
        print(f"  ↓ Subsampling {args.sample_size:,} / {len(doc_ids):,} documents ...")
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(len(doc_ids), size=args.sample_size, replace=False))
        sampled_ids = set(doc_ids[i] for i in indices)
        embeddings = embeddings[indices]
        doc_ids = [doc_ids[i] for i in indices]
        categories = [categories[i] for i in indices]
    else:
        sampled_ids = None  # use all

    # Ranked lists (before / after)
    before_path = f"{base}/rerank/input/{model}/ranked-list/data.txt"
    after_data_path = f"{base}/rerank/output/{model}/{experiment_tag}/data.txt"
    after_tsv_path = f"{base}/rerank/output/{model}/{experiment_tag}/ranks.tsv"

    print(f"  ↓ Loading before-reranking list ...")
    before_rankings = load_ranked_list(gcs, before_path)
    print(f"    {len(before_rankings):,} queries")

    print(f"  ↓ Loading after-reranking list ({experiment_tag}) ...")
    after_rankings = load_rankings_auto(gcs, after_data_path, after_tsv_path)
    print(f"    {len(after_rankings):,} queries")

    # ── 2. Compute 2D projection ────────────────────────────────────────
    coords_2d = compute_projection(
        embeddings,
        reducer=args.reducer,
        random_state=args.seed,
    )

    # ── 3. Generate figures ─────────────────────────────────────────────
    prefix = f"{ds}_{model}_{method}_k{k_val}"

    # Figure 1: embedding scatter
    print("  ✎ Generating embedding scatter ...")
    fig1 = figure_embedding_scatter(
        coords_2d, categories,
        title=f"{ds_display} — {model_display} Embeddings ({reducer_name} Projection)",
        reducer_name=reducer_name,
    )
    path1 = out_dir / f"{prefix}_scatter.{ext}"
    fig1.savefig(path1, bbox_inches="tight")
    plt.close(fig1)
    print(f"    → {path1}")

    # Figure 2: before / after overlay
    print("  ✎ Selecting queries with best purity improvement ...")
    sample_qs = sample_best_improvement_queries(
        before_rankings,
        after_rankings,
        top_k=args.top_k,
        n_per_category=args.n_sample_queries,
        seed=args.seed,
        candidate_qids=sampled_ids,  # restrict to subsampled docs when applicable
    )
    # Limit to avoid overly tall figure
    sample_qs = sample_qs[:8]

    fig2 = figure_before_after_overlay(
        coords_2d, doc_ids, categories,
        before_rankings, after_rankings,
        sample_queries=sample_qs,
        top_k=args.top_k,
        reducer_name=reducer_name,
        method_label=method_display,
        k_label=str(k_val),
    )
    path2 = out_dir / f"{prefix}_overlay.{ext}"
    fig2.savefig(path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"    → {path2}")

    # Figure 3: category purity comparison
    print("  ✎ Generating purity comparison ...")
    fig3 = figure_purity_comparison(
        before_rankings, after_rankings,
        top_k=args.top_k,
        method_label=method_display,
        k_label=str(k_val),
    )
    path3 = out_dir / f"{prefix}_purity.{ext}"
    fig3.savefig(path3, bbox_inches="tight")
    plt.close(fig3)
    print(f"    → {path3}")

    gcs.cleanup()
    print(f"\n✅ All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
