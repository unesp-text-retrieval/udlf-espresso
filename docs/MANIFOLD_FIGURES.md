# Manifold Visualisation Figures

## Purpose

These figures illustrate **how UDLF unsupervised re-ranking reshapes document neighbourhoods** in the embedding space. They are designed for inclusion in a research paper or dissertation and answer a key qualitative question: *"beyond the numbers, what does re-ranking actually do to the retrieval results?"*

Three figure types are produced per dataset / model / method combination:

| Figure | What it shows | Why it matters |
|--------|--------------|----------------|
| **Embedding scatter** | All documents projected to 2D via UMAP, coloured by category | Gives a global view of the corpus structure — how well-separated the categories are in the learned embedding space |
| **Before / after overlay** | Side-by-side zoomed panels of a query's top-K neighbourhood before and after re-ranking | Shows that UDLF concentrates same-category documents around the query, removing cross-category noise |
| **Category precision chart** | Dumbbell + Δ bar chart comparing same-category proportion in top-K before vs. after | Aggregates the per-category improvement across all queries — the quantitative complement to the overlay |

## Quick Start

```bash
conda activate udlf-espresso

# BBC-News (small, 5 categories — good demo)
python scripts/generate_manifold_figures.py \
    --dataset bbc-news \
    --model contriever-msmarco \
    --method cprr --k 75

# Mental-Health (4 categories, 20K docs)
python scripts/generate_manifold_figures.py \
    --dataset mental-health \
    --model contriever-msmarco \
    --method cprr --k 75

# HuffPost (42 categories, 33K docs)
python scripts/generate_manifold_figures.py \
    --dataset huffpost-news \
    --model contriever-msmarco \
    --method cprr --k 75

# With subsampling (faster UMAP, smaller PDFs)
python scripts/generate_manifold_figures.py \
    --dataset huffpost-news \
    --model contriever-msmarco \
    --method cprr --k 75 \
    --sample-size 5000
```

Output PDFs land in `outputs/paper-assets/figures/`.

### Full CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | Dataset name (`bbc-news`, `mental-health`, `huffpost-news`, `wos`, …) |
| `--model` | *(required)* | Dense retrieval model that has a FAISS index in GCS |
| `--method` | *(required)* | UDLF re-ranking method (`cprr`, `bfstree`, `rdpac`) |
| `--k` | *(required)* | K parameter of the re-ranking experiment |
| `--top-k` | `20` | Number of top results to highlight per query |
| `--reducer` | `umap` | Dimensionality reduction (`umap` or `tsne`) |
| `--sample-size` | all | Subsample N documents (useful for 20K+ corpora) |
| `--n-sample-queries` | `1` | Max queries per category in the overlay figure |
| `--output-dir` | `outputs/paper-assets/figures/` | Where to save figures |
| `--format` | `pdf` | Output format (`pdf`, `svg`, `png`) |
| `--seed` | `42` | Random seed for UMAP and subsampling |
| `--bucket` | `text-udlf-expresso` | GCS bucket name |

---

## Data Sources (GCS)

All input data is read from `gs://text-udlf-expresso/outputs/paper-assets/dataset/<dataset>/`. The script needs three pieces:

### 1. FAISS Index — the document embeddings

```
indexes/<model>/faiss.index       # binary FAISS flat index
indexes/<model>/docids.json       # ordered list of document IDs (optional)
```

If `docids.json` is absent, the script falls back to:

```
rerank/input/<model>/lists/data.txt   # one doc ID per line, same order as the FAISS index
```

Both files encode the same ordered mapping `index_position → document_id`. This was verified empirically (see "Reproducibility" section below).

### 2. Before-Reranking Ranked List

```
rerank/input/<model>/ranked-list/data.txt
```

Horizontal format — one line per query, space-separated:
```
query_id  doc_1  doc_2  doc_3  ...
```

### 3. After-Reranking Output

```
rerank/output/<model>/<method>-k<K>/data.txt     # horizontal (preferred)
rerank/output/<model>/<method>-k<K>/ranks.tsv    # vertical TSV fallback
```

The script tries `data.txt` first. If missing it reads `ranks.tsv` (TREC-style: `query_id\tdoc_id\tscore` per line, sorted by score descending).

---

## High-Level Algorithm (Reproducible Without This Script)

If the Python script is lost, here is the logic to reproduce each figure from scratch with any programming language or tool:

### Figure 1 — Embedding Scatter

1. **Load embeddings**: read all N vectors from the FAISS index (or any other source of 768-dim document embeddings).
2. **Extract categories**: parse the category prefix from each document ID (e.g. `sport_001.txt` → `sport`). For IDs with hash suffixes (e.g. `anxiety_52b22c62e7`) the category is everything before the last `_`.
3. **Project to 2D**: run UMAP with `n_neighbors=15, min_dist=0.1, metric=cosine, random_state=42`. Alternatively, t-SNE with `perplexity=30, metric=cosine`.
4. **Plot**: scatter all points, colour by category. Use a colour-blind-friendly palette (Tableau 10).

### Figure 2 — Before / After Overlay

1. **Select queries**: for every query, compute **P@K** (Precision@K) before and after re-ranking:
   ```
   P@K(q) = |{d ∈ top-K(q) : category(d) = category(q)}| / K
   ```
   Compute Δ = P_after − P_before. Sort all queries by Δ descending. Pick the top-1 per category (for diversity). Use up to 8 queries.

2. **For each selected query, draw two panels (Before / After)**:
   - Plot all documents as a faint background scatter (alpha ~0.10).
   - Highlight the top-K retrieved documents with larger, opaque markers. Same-category docs get black edges; cross-category docs get grey edges.
   - Mark the query with a gold star.
   - **Auto-zoom**: compute the bounding box of all highlighted points (union of before and after), add 30% margin, and set both panels to the same viewport. This shared zoom is critical — it makes the spatial shift visible.
   - Annotate the panel title with the P@K value.

3. **Layout**: N rows × 2 columns, one row per query.

### Figure 3 — Category Precision Chart

1. **Compute per-category average precision** before and after:
   ```
   avg_precision(cat) = mean over all queries q where category(q)=cat of P@K(q)
   ```
2. **Left panel — dumbbell chart**: horizontal connected-dot plot. For each category, draw a line connecting the before-dot (blue) and after-dot (red). Zoom the x-axis to the data range (not 0–100%) so small improvements are visible.
3. **Right panel — Δ bar chart**: horizontal bars showing `avg_precision_after − avg_precision_before` per category. Green for positive, red for negative. Annotate each bar with the signed percentage.

---

## Design Decisions

### Why UMAP over t-SNE?
- UMAP preserves both local and global structure better than t-SNE for medium-scale data (2K–50K points).
- UMAP is faster (30s vs minutes for 20K+ points).
- t-SNE is supported via `--reducer tsne` for comparison if needed.

### Why auto-zoom the overlay panels?
The original full-view showed all 2,225+ documents, compressing the top-20 neighbourhood into a tiny cluster. The improvement from 95% to 100% precision was barely noticeable. Auto-zooming to the query's neighbourhood (with 30% margin and shared viewport between before/after) makes the spatial shift immediately clear.

### Why select queries by best P@20 improvement?
Random sampling often picked queries where re-ranking had little effect (e.g. already 95% → 100%). Selecting the queries with the largest Δ improvement guarantees that the overlay figure showcases the most dramatic examples — which is the whole point of including these figures in a paper.

### Why dumbbell chart instead of grouped bars?
The original grouped bar chart started the y-axis at 0%, which compressed differences like 85% → 92% into barely distinguishable bar heights. The dumbbell chart zooms the axis to the data range and the Δ panel directly shows improvement magnitude.

### Subsampling considerations
For datasets >10K documents, UMAP computation takes 1–4 minutes and PDFs grow to 2–6 MB. Subsampling to 5K reduces both but:
- May miss the best Δ queries (they might not be in the subsample)
- Produces sparser cluster visualisations
- Precision values in Figure 3 are always computed over **all** queries regardless of subsampling

**Recommendation**: use full data for final paper figures. Use `--sample-size 5000` for rapid iteration.

---

## Known Limitations

- **Category-based datasets only**: the category colouring and precision metric require document IDs that encode categories. BEIR datasets (scifact, nfcorpus, etc.) use opaque doc IDs, so the figures would not be meaningful.
- **FAISS index required**: the script needs a FAISS flat index to extract embeddings. If only raw embeddings exist (e.g. numpy files), the `load_embeddings_from_faiss` function would need adaptation.
- **Single retrieval model per run**: each invocation visualises one model. To compare models, run the script multiple times and compose figures in LaTeX.

## Available FAISS Indexes (as of Feb 2026)

| Dataset | contriever-msmarco | bge-small-en-v1.5 | miniLM-L6-v2 |
|---------|:--:|:--:|:--:|
| bbc-news | ✅ | — | — |
| mental-health | ✅ | ✅ | ✅ |
| huffpost-news | ✅ | ✅ | ✅ |
| wos | ✅ | ✅ | ✅ |

---

## Dependencies

```
umap-learn        # UMAP dimensionality reduction
matplotlib        # static figure generation
faiss-cpu         # read FAISS indexes
google-cloud-storage  # download from GCS
numpy
scikit-learn      # optional, only for t-SNE
```

All are included in the `udlf-espresso` conda environment.

---

## Example Outputs

Running for BBC-News produces three files:

```
outputs/paper-assets/figures/
  bbc-news_contriever-msmarco_cprr_k75_scatter.pdf   (~91 KB)
  bbc-news_contriever-msmarco_cprr_k75_overlay.pdf   (~340 KB)
  bbc-news_contriever-msmarco_cprr_k75_precision.pdf    (~28 KB)
```

The overlay figure selects queries with the largest precision improvement, for example:

| Query | Before P@20 | After P@20 | Δ |
|-------|:-----------:|:----------:|:-:|
| tech_140 | 25% | 100% | +75% |
| business_052 | 30% | 100% | +70% |
| entertainment_192 | 25% | 95% | +70% |
| politics_090 | 75% | 100% | +25% |
| sport_015 | 85% | 100% | +15% |

### LaTeX Inclusion

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/bbc-news_contriever-msmarco_cprr_k75_scatter.pdf}
  \caption{UMAP projection of BBC-News document embeddings (Contriever).
           Colours represent the five news categories.}
  \label{fig:bbc-scatter}
\end{figure}
```
