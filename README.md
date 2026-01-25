# UDLF Text Expresso

<p align="center">
  <img src="assets/udlf-expresso-logo.png" alt="UDLF Expresso" width="600"/>
</p>

UDLF Text Expresso is a lightweight experimentation harness for end-to-end text retrieval and contextual re-ranking, designed around Hydra configs and pluggable storage backends (Local/S3). It bundles BEIR data collection, BM25 and dense retrieval, UDLF (Unsupervised Distance Learning Framework) re-ranking, and metric computation under a single reproducible pipeline.

## What You Get
- Deterministic pipelines: every step is a `PipelineStep` that records its outputs as `Artifact`s with hashable config snapshots.
- Canonical directory layout: `outputs/<run_id>/dataset/<dataset_name>/...` contains ranks, rerank inputs, rerank outputs, and metrics.
- Unified document+query corpus: data collection emits `transformed/documents-and-topics/data.tsv` with elements that can score both user queries and corpus documents (indices continue to build from the full document texts).
- Automatic rerank list preparation: retrieval steps materialize `ranked-list/data.txt` and `lists/data.txt` for downstream rerankers using the configured cutoff.
- Hydra-driven overrides: swap datasets, retrievers, storage, or limits from the command line with minimal boilerplate.
- In a nutshell, you are able to test UDLF (Unsupervised Distance Learning Framework) methods with text retrieval ranks, experiment and visualize reports in a very practical manner.

## Repository Layout
- `conf/` – Hydra configuration tree (`config.yaml` plus per-step overrides).
- `src/udlf_text_expresso/` – pipeline implementation, steps, and utilities.
- `tests/` – pytest suite, including rerank input pivot coverage.
- `outputs/` and `dataset/` – default local artifact roots (ignored by VCS).

## Installing
```bash
pip install -r requirements.txt
```
Optional: activate your preferred environment (Conda, venv) before installing.

### PyUDLF Setup for re-ranking
PyUDLF is a wrapper to run the UDLF (Unsupervised Distance Learning Framework) binary, it helps to organize the code in a single language and
modify parameters easily, create your conda env, install udlf-text-expresso requirements and go ahead following 
instructions from pyUDLF [github repository](https://github.com/UDLF/pyUDLF) and install it in the same environment.

Default binary path is `/usr/local/bin/udlf`. Update `conf/rerank/udlf.yaml` to set project-wide defaults.

## Running Pipelines
Hydra parameters live under `conf/config.yaml`. Override them inline to compose runs:

Scifact is a good starting point to validate your environment is working, it is fast to index, retrieve and re-rank. It should take no longer than 15 minutes to index bm25, re-rank, and generate the report in a macbook with arm processors, it was tested with M4 processor and 16GB of RAM.

- **Scifact baseline (BM25 + MiniLM + BGE + Contriever)**. Pick a stable `RUN_ID` (for example `paper-assets`) so you and your teammates can rerun retrieval/metrics without recomputing indexes. The first time you execute these commands the index steps will build FAISS files locally; subsequent runs with the same `RUN_ID` can skip them.
  ```bash
  RUN_ID="paper-assets"

  # 1. Collect corpus/topics/qrels and run BM25 baseline + metrics
  python -m udlf_text_expresso.runner \
    run_id=$RUN_ID \
    dataset=scifact \
    model=bm25 \
    pipeline.steps='[data_collection,index,retrieve,metrics]'

  # 2. Dense: MiniLM
  python -m udlf_text_expresso.runner \
    run_id=$RUN_ID \
    dataset=scifact \
    model=miniLM-L6-v2 \
    pipeline.steps='[index,retrieve,metrics]'

  # 3. Dense: BGE small
  python -m udlf_text_expresso.runner \
    run_id=$RUN_ID \
    dataset=scifact \
    model=bge-small-en-v1.5 \
    pipeline.steps='[index,retrieve,metrics]'

  # 4. Dense: Contriever
  python -m udlf_text_expresso.runner \
    run_id=$RUN_ID \
    dataset=scifact \
    model=contriever-msmarco \
    pipeline.steps='[index,retrieve,metrics]'
  ```
  After the first execution you can re-run just the retrieval and metrics phases for any model (e.g., `pipeline.steps='[retrieve,metrics]'`) to reuse the cached indexes. If you ever remove the FAISS files or want to rebuild them with new inputs, make sure to rerun `index` (optionally add `force=true`)—otherwise `retrieve` will fail looking for `faiss.index` and `docids.json` (for dense models) or `bm25.pkl` cache file (for BM25).

  The `model` parameter automatically configures the retriever:
  - `model=bm25` → BM25 retrieval
  - `model=miniLM-L6-v2` → sentence-transformers/all-MiniLM-L6-v2
  - `model=bge-small-en-v1.5` → BAAI/bge-small-en-v1.5
  - `model=contriever-msmarco` → facebook/contriever-msmarco

- **New BEIR dataset (example: Scidocs)**. Any BEIR collection can be pulled by setting `data_collection.source=beir_<dataset>` and matching `dataset.name`. Below is an end-to-end walkthrough for Scidocs that you can copy-paste and tweak for other corpora.
  1. **Collect, index, and score the BM25 baseline**
     ```bash
     RUN_ID="scidocs-paper"
     python -m udlf_text_expresso.runner \
       run_id=$RUN_ID \
       dataset.name=scidocs \
       data_collection.source=beir_scidocs \
       model=bm25 \
       pipeline.steps='[data_collection,index,retrieve,metrics]'
     ```
  2. **Optional dense retrievers** – rerun with the same `RUN_ID` so FAISS indexes are cached under the Scidocs directory. For example, to build and score BGE-small:
     ```bash
     python -m udlf_text_expresso.runner \
       run_id=$RUN_ID \
       dataset.name=scidocs \
       data_collection.source=beir_scidocs \
       model=bge-small-en-v1.5 \
       pipeline.steps='[index,retrieve,metrics]'
     # add force=true if you need to rebuild the FAISS index
     ```
  3. **Run UDLF reranking** – once `retrieval.tsv` exists for a retriever, reuse the same `run_id` to launch the UDLF reranking:
     ```bash
     python -m udlf_text_expresso.runner \
       run_id=$RUN_ID \
       dataset.name=scidocs \
       model=bge-small-en-v1.5 \
       pipeline.steps='[rerank,metrics]'
     ```
     Repeat for other retrieval models (e.g., `index.name=bm25`, `index.name=miniLM-L6-v2`, etc.).
    4. **Generate the comparative report** – the HTML summariser automatically sweeps every dataset and retriever under the `run_id` hierarchy:
      ```bash
      python -m udlf_text_expresso.reporting.metrics_report $RUN_ID
      ```
      Open `outputs/$RUN_ID/reports/udlf_metrics_report.html` to compare Scidocs baselines vs. reranks. To rebuild a single-dataset report (written alongside that dataset), add one or more `--dataset <name>` flags:
      ```bash
      python -m udlf_text_expresso.reporting.metrics_report $RUN_ID --dataset scidocs
      ```
      The dataset-scoped HTML lands in `outputs/$RUN_ID/dataset/scidocs/reports/udlf_metrics_report.html`, which makes it easy to rerun the same command after introducing new retrieval models without touching other corpora. Supplying multiple `--dataset` flags re-aggregates just those datasets, while omitting the flag retains the cross-dataset view (e.g., Scifact + Scidocs under `paper-assets`).

- **Rerank limit overrides** (cap ranked lists to 200 docs without editing configs):
  ```bash
  python -m udlf_text_expresso.runner \
      pipeline.steps='[retrieve,rerank]' \
      retrieval.rerank_list_limit=200
  ```

- **UDLF reranking with multiple experiments** (assumes `retrieval.tsv` already exists for the chosen dataset/index):
  ```bash
  RUN_ID="paper-assets"
  python -m udlf_text_expresso.runner \
      run_id=$RUN_ID \
      dataset=scifact \
      model=miniLM-L6-v2 \
      pipeline.steps='[rerank,metrics]' \
      rerank.experiments='[{"name":"cprr-k10","method":"CPRR","config":{"k":10,"t":1,"window_size":200,"task":"UDL","input_format":"RK"}},{"name":"bfstree-k50","method":"BFSTREE","config":{"k":50,"window_size":200,"correlation_metric":"RBO"}}]'
  ```
  Each experiment writes `data.txt`, `data.trec`, `ranks.tsv`, and `metrics.json` under `outputs/$RUN_ID/dataset/<dataset>/rerank/output/<method>/<experiment>/`. Metrics are computed automatically after reranking, using the MAP/Recall/Precision/nDCG cutoffs defined in `rerank.cutoffs` (defaults mirror the global metrics cutoffs).

## Reranking & Evaluation
- **Prerequisites**: run either BM25 or dense retrieval so that `outputs/<run_id>/dataset/<dataset>/ranks/<index>/retrieval.tsv` exists. The rerank step aborts early if the retrieval file is missing.
- **Configuring experiments**: edit `conf/rerank/udlf.yaml` or supply inline overrides for `rerank.experiments`. Each experiment block accepts `name`, `method`, optional `limit`, and a free-form `config` dictionary forwarded to `UDLFReranker`.
- **Automatic outputs**: rerank execution normalizes horizontal lists, emits TREC-formatted scores, and saves limited TSV ranks per experiment. Metrics are written to `<experiment>/metrics.json` right after reranking—no extra evaluation command is required.
- **Inspecting results**: combine Hydra overrides with `rerank.cutoffs` to harmonize evaluation thresholds across experiments. Metrics files contain the four core scores; additional metadata lives inside the registered artifacts (see the `Artifacts produced` log section after a run).
- **Quick verification**: after a rerank run, confirm that directories appear under `outputs/<run_id>/dataset/<dataset>/rerank/output/` before moving on to downstream analyses or copying artifacts to shared storage.

Key Hydra knobs:
- `run_id`: labels the output directory. If omitted, a timestamped ID is generated.
- `model`: unified parameter for selecting retrieval model (`bm25`, `miniLM-L6-v2`, `bge-small-en-v1.5`, `contriever-msmarco`).
- `pipeline.steps`: ordered list of step names to execute. Use unified step names that work with any model:
  - `index` – builds index (BM25 or FAISS based on model)
  - `retrieve` – runs retrieval (sparse or dense based on model)
  - `rerank` – applies UDLF reranking
  - `metrics` – computes evaluation metrics
  - Legacy model-specific names still work: `index_bm25`, `index_dense`, `retrieval`, `dense_retrieval`
- `retrieval.*` / `dense_indexing.*`: control top-k, rerank cutoffs, model selection, batch sizes, etc.
- `storage.kind`: `gcs` (default), with `local` and `s3` also available when configure bucket/prefix.

## Artifacts Produced
Each step registers `Artifact`s announcing the files it created. Typical dense retrieval run:
```
Artifacts produced:
topics outputs/dataset/scifact/extracted/topics/data.tsv.gz
corpus outputs/dataset/scifact/extracted/documents/data.tsv.gz
qrels outputs/dataset/scifact/extracted/qrels/data.tsv.gz
documents_and_topics outputs/dataset/scifact/transformed/documents-and-topics/data.tsv
dense_retrieval outputs/dataset/scifact/ranks/miniLM-L6-v2/retrieval.tsv
dense_rerank_ranked_list outputs/dataset/scifact/rerank/input/miniLM-L6-v2/ranked-list/data.txt
dense_rerank_query_list outputs/dataset/scifact/rerank/input/miniLM-L6-v2/lists/data.txt
```
The merged TSV feeds both retrieval steps: every run queries the index with user topics plus each document in the corpus, enabling self-similarity scores for UDLF. Ranked-list and query-list files are generated by `_prepare_rerank_inputs`, which pivots the flat `retrieval.tsv` into per-query doc lists respecting the configured `rerank_list_limit`.

### Text Normalization
- Document content is stored in full for indexing (falling back to `No content available for this document.` when source text is blank).
- Retrieval queries—including user topics and document-as-query entries—are normalized by stripping non-alphanumeric characters, collapsing whitespace, and truncating to 1,000 characters.

## Configuration Notes
- `conf/config.yaml` captures the canonical defaults. Update the `retrieval.rerank_list_limit` or `dense_indexing.rerank_list_limit` fields to change the global default cutoff.
- The merged TSV is registered as the `documents_and_topics` artifact; reuse it for custom rerank pipelines or additional retrieval passes where normalized query text is helpful.
- Additional method- or dataset-specific overrides can live under `conf/retrieval/`, `conf/rerank/`, etc., and are activated with Hydra’s `+` syntax or config groups.

## Testing
Run the full suite with:
```bash
pytest -q tests
```
The tests include coverage for rerank input generation to guard against regressions in the ranked-list pivot logic.

## Extending the Stack
- **New retrieval models**: subclass the appropriate retriever, register a Hydra config, and append a pipeline step.
- **New rerankers**: follow `udlf_text_expresso/rerank/udlf_runner.py`, emitting TSV scores and registering artifacts.
- **Custom metrics**: implement in `metrics/eval.py` and wire them into `MetricsStep`.

This project is intended for research and experimentation; adjust licensing or deployment specifics to fit your environment.

---

## Research Context: Re-Ranking Study

### Experiment Overview
This repository supports a comprehensive text retrieval and re-ranking research study evaluating UDLF (Unsupervised Distance Learning Framework) methods across multiple datasets and retrieval models.

**Total Experiments: 972** (9 datasets × 4 retrieval models × 3 re-ranking methods × 9 k-values)

### Experiment Parameters

#### Datasets (9)
| Dataset | Source | Queries | Documents | Has Qrels | Description |
|---------|--------|---------|-----------|-----------|-------------|
| **arguana** | BEIR | 1,406 | 8,674 | ✓ | Argument retrieval - counter-argument identification |
| **scifact** | BEIR | 300 | 5,183 | ✓ | Scientific claim verification |
| **nfcorpus** | BEIR | 323 | 3,633 | ✓ | Medical/nutrition domain |
| **scidocs** | BEIR | 1,000 | 25,657 | ✓ | Scientific document retrieval |
| **fiqa** | BEIR | 648 | 57,638 | ✓ | Financial question answering |
| **wos** | Non-BEIR | - | ~46K | ✗ | Web of Science abstracts |
| **bbc-news** | Non-BEIR | - | ~2K | ✗ | BBC News articles |
| **huffpost-news** | Non-BEIR | - | 32,964 | ✗ | HuffPost News articles (20% stratified sample from 164K, min length 50 chars) |
| **mental-health** | Non-BEIR | - | 20,458 | ✗ | Mental health forum posts (50% stratified sample from 40K: Anxiety, Depression, Normal, Suicidal) |

**Note on Query-Document Overlap:** BEIR datasets like arguana and fiqa use documents as queries (document-as-query retrieval). For example:
- **arguana**: 1,406 regular queries + 8,674 documents, with 1,298 IDs appearing as both → 8,782 unique IDs total
- **fiqa**: 648 regular queries + 57,638 documents, with 55 overlapping IDs → 58,231 unique IDs total

#### Handling Overlapping Query-Document IDs

When an ID appears as both a regular query (topic) and a document, the system must decide which text to use when querying the index.

**Unified Strategy (All Retrieval Models):**
- **Approach:** Document content is always preferred over topic content for overlapping IDs
- **Rationale:** Maximizes corpus-to-corpus similarity for UDLF methods and ensures consistent query sets across all retrieval models
- **Implementation:**
  1. Parse topics and build `topic_queries` list
  2. If a topic ID also exists in documents, **skip the topic version**
  3. Create `doc_queries` list with all document texts
  4. Merge using `_merge_unique_queries(topic_queries, doc_queries)` helper function
  5. For overlapping IDs, document text is used for retrieval

**Example (arguana with 1,298 overlapping IDs):**
```python
# Consistent behavior across BM25, miniLM, BGE, and Contriever:
topic_queries = [{"id": "query_1", "text": "topic text"}, ...]  # 108 unique topics (1,406 - 1,298)
doc_queries = [{"id": "doc_1", "text": "doc text"}, ...]        # 8,674 documents (includes 1,298 overlaps)
# Result: 108 + 8,674 = 8,782 queries (overlapping IDs use document text)

# This ensures:
# ✅ Fair comparison between retrieval models
# ✅ Consistent query sets for UDLF input
# ✅ No duplicate IDs or prefix issues
```

#### Retrieval Models (4)
1. **bm25** - BM25 sparse retrieval (rank_bm25 library)
2. **miniLM-L6-v2** - sentence-transformers/all-MiniLM-L6-v2 (dense)
3. **bge-small-en-v1.5** - BAAI/bge-small-en-v1.5 (dense)
4. **contriever-msmarco** - facebook/contriever-msmarco (dense)

#### Re-ranking Methods (3)
1. **CPRR** - Cartesian Product of Ranking References
2. **BFS-TREE** - Breadth-First Search Tree
3. **RDPAC** - Rank-based Diffusion Process with Assured Convergence

#### K Values (9)
Fine-tuning parameter for UDLF methods: `[1, 3, 5, 10, 20, 30, 40, 50, 75]`

---

## Folder Structure & Data Flow

### Directory Organization
Each dataset follows this structure under `outputs/<run_id>/dataset/<dataset_name>/`:

```
<dataset>/
├── extracted/           # Raw data from BEIR or manual sources
│   ├── documents/       # Corpus documents
│   │   └── data.tsv.gz  # Format: doc_id\tdoc_text
│   ├── topics/          # Query topics
│   │   └── data.tsv.gz  # Format: query_id\tquery_text
│   └── qrels/           # Relevance judgments
│       ├── data.tsv.gz           # Format: query_id\tdoc_id\trelevance (BEIR)
│       └── category_metadata.json # Category-based dataset metadata (non-BEIR)
│
├── transformed/         # Processed unified corpus
│   ├── documents/       # Documents in JSONL format
│   │   └── data.jsonl   # Format: {"id": "...", "contents": "..."}
│   └── documents-and-topics/
│       └── data.tsv     # Merged doc+query corpus for retrieval
│                        # Format: id\ttext (all searchable items)
│
├── indexes/             # Retrieval indexes
│   ├── bm25/           # rank_bm25 BM25 index
│   │   └── bm25.pkl    # Serialized rank_bm25 index with BM25Okapi instance
│   └── <model_name>/   # FAISS dense indexes
│       ├── faiss.index  # FAISS index file
│       └── docids.json  # Document ID mapping
│
├── ranks/              # Retrieval results
│   └── <model_name>/   # e.g., bm25, miniLM-L6-v2
│       ├── retrieval.tsv         # Format: query_id\tdoc_id\tscore
│       ├── query_embeddings.npy  # Query embeddings (dense models only)
│       └── metrics.json          # Baseline retrieval metrics
│
└── rerank/             # Re-ranking inputs and outputs
    ├── input/          # UDLF input files
    │   └── <model_name>/
    │       ├── ranked-list/
    │       │   └── data.txt  # Format: query_id doc_id1 doc_id2 ...
    │       └── lists/
    │           └── data.txt  # Format: unique_id (one per line)
    │
    └── output/         # UDLF re-ranking results
        └── <model_name>/
            └── <method>-k<value>/  # e.g., bfstree-k10
                ├── data.txt         # UDLF output (horizontal format)
                ├── data.trec        # TREC-formatted results
                ├── ranks.tsv        # Re-ranked results (TSV format)
                ├── config.json      # Experiment configuration
                └── metrics.json     # Re-ranking evaluation metrics
```

### Data Flow Pipeline (ELT)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE (ELT)                                   │
│  Pipeline Steps: data_collection → index → retrieve → rerank → metrics                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐
  │  BEIR / Manual   │
  │    Data Source   │
  └────────┬─────────┘
           │
           ▼
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  STEP: data_collection (Extract + Transform)                                     ┃
  ┃  ─────────────────────────────────────────────────────────────────────────────   ┃
  ┃  Download, extract raw data, and transform into unified corpus                   ┃
  ┃  PipelineStep: DataCollectionStep                                                ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
           │
           ├─────► extracted/documents/data.tsv.gz          (corpus)
           ├─────► extracted/topics/data.tsv.gz             (queries)
           ├─────► extracted/qrels/data.tsv.gz              (relevance judgments)
           ├─────► transformed/documents/data.jsonl         (JSONL format)
           └─────► transformed/documents-and-topics/data.tsv (unified corpus)
           │
           ▼
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  STEP: index (or index_bm25 / index_dense)                                       ┃
  ┃  ─────────────────────────────────────────────────────────────────────────────   ┃
  ┃  Build searchable indexes (BM25 or FAISS)                                        ┃
  ┃  PipelineSteps: IndexBM25Step / IndexDenseStep                                   ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
           │
           ├─────► indexes/bm25/bm25.pkl                 (BM25Okapi)
           └─────► indexes/<model>/faiss.index + docids.json  (Dense)
           │
           ▼
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  STEP: retrieve (or retrieval / dense_retrieval)                                 ┃
  ┃  ─────────────────────────────────────────────────────────────────────────────   ┃
  ┃  Query index with all items (topics + documents as queries)                     ┃
  ┃  PipelineSteps: RetrievalStep (BM25) / DenseRetrievalStep (Dense)               ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
           │
           ├─────► ranks/<model>/retrieval.tsv                 (query_id│doc_id│score)
           ├─────► ranks/<model>/query_embeddings.npy          (dense only)
           ├─────► rerank/input/<model>/ranked-list/data.txt   (horizontal format)
           └─────► rerank/input/<model>/lists/data.txt         (unique IDs)
           │
           ▼
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  STEP: rerank                                                                    ┃
  ┃  ─────────────────────────────────────────────────────────────────────────────   ┃
  ┃  Apply UDLF methods (CPRR, BFS-TREE, RDPAC) with k-values                       ┃
  ┃  PipelineStep: ReRankStep                                                        ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
           │
           └─────► rerank/output/<model>/<method>-k<value>/
                   ├── data.txt       (UDLF output)
                   ├── data.trec      (TREC format)
                   ├── ranks.tsv      (re-ranked results)
                   └── config.json    (experiment params)
           │
           ▼
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  STEP: metrics                                                                   ┃
  ┃  ─────────────────────────────────────────────────────────────────────────────   ┃
  ┃  Evaluate with qrels: MAP, nDCG, Recall, Precision @ various cutoffs            ┃
  ┃  PipelineStep: MetricsStep                                                       ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
           │
           ├─────► ranks/<model>/metrics.json                      (baseline)
           └─────► rerank/output/<model>/<method>-k<value>/metrics.json  (re-ranked)

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  AVAILABLE PIPELINE STEPS:                                                              │
│  • data_collection - Extract and transform data (combined ETL)                          │
│  • index, index_bm25, index_dense - Build indexes                                       │
│  • retrieve, retrieval, dense_retrieval - Run retrieval                                 │
│  • rerank - Apply UDLF re-ranking                                                       │
│  • metrics - Compute evaluation metrics                                                 │
│                                                                                          │
│  KEY DATA TRANSFORMATIONS:                                                              │
│  • data_collection creates both extracted/ and transformed/ outputs in one step         │
│  • Vertical (TSV) → Horizontal (space-separated) for UDLF input                         │
│  • All documents become queries for corpus-to-corpus similarity                         │
│  • Both BM25 & Dense: Prefer document text over topic text for overlapping IDs          │
│  • Consistent query sets across all models enable fair comparison                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Step: data_collection (Extract + Transform)
**Purpose:** Download, extract raw data from BEIR or GCS, and transform into unified corpus format.

**Pipeline Step:** `DataCollectionStep`

#### Step: data_collection (Extract + Transform)
**Purpose:** Download, extract raw data from BEIR or GCS, and transform into unified corpus format.

**Pipeline Step:** `DataCollectionStep`

**What it does:**
1. Downloads and extracts raw data (Extract phase)
2. Normalizes text and creates unified corpus (Transform phase)
3. Produces both extracted/ and transformed/ outputs in a single step

**For BEIR datasets (arguana, scifact, nfcorpus, scidocs, fiqa):**
- Downloads from BEIR benchmark via `beir.datasets.data_loader`
- Extracts to `extracted/documents/`, `extracted/topics/`, `extracted/qrels/`
- Creates transformed formats in parallel
- **Files created:**
  - `extracted/documents/data.tsv.gz` - Corpus documents (id, text, title)
  - `extracted/topics/data.tsv.gz` - Query topics (id, text)
  - `extracted/qrels/data.tsv.gz` - Relevance judgments (query_id, doc_id, relevance)
  - `transformed/documents/data.jsonl` - JSONL format for storage
  - `transformed/documents-and-topics/data.tsv` - Unified corpus (merged docs + queries)

**For non-BEIR datasets (wos, bbc-news, huffpost-news, mental-health):**
- Downloads corpus from GCS
- Uses category-based relevance (doc IDs like `sport_001`, `business_005`)
- **Files created:**
  - `extracted/documents/data.tsv.gz` - Corpus documents (all docs are also queries)
  - `extracted/topics/data.tsv.gz` - Placeholder file
  - `extracted/qrels/data.tsv.gz` - Placeholder file
  - `extracted/qrels/category_metadata.json` - Category counts and metadata
  - `transformed/documents/data.jsonl` - JSONL format
  - `transformed/documents-and-topics/data.tsv` - Unified corpus

**Implementation:** `data/steps.py::DataCollectionStep`, `data/collection.py`

---

#### Step: index (index_bm25 / index_dense)
#### Step: index (index_bm25 / index_dense)
**Purpose:** Build searchable indexes for retrieval.

**Pipeline Steps:** `IndexBM25Step` (for BM25) / `IndexDenseStep` (for dense models)

**For BM25 (sparse):**
- Uses rank_bm25 library (BM25Okapi implementation)
- Reads corpus from data collection artifacts
- Tokenizes documents and builds in-memory BM25 index
- **Output:** `indexes/bm25/bm25.pkl` - Serialized pickle containing BM25Okapi instance, tokenized corpus, and document IDs
- **Implementation:** `runner.py::IndexBM25Step`

**For Dense Models (miniLM-L6-v2, bge-small-en-v1.5, contriever-msmarco):**
- Reads from `transformed/documents/data.jsonl`
- Encodes all documents using sentence-transformers
- Builds FAISS index for similarity search
- **Files Created:**
  - `indexes/<model_name>/faiss.index` - FAISS index binary
  - `indexes/<model_name>/docids.json` - ID mapping (array index → doc_id)
- **Implementation:** `runner.py::IndexDenseStep`

---

#### Step: retrieve (retrieval / dense_retrieval)

#### Step: retrieve (retrieval / dense_retrieval)
**Purpose:** Query the index with all items (topics + documents as queries) and prepare UDLF input files.

**Pipeline Steps:** `RetrievalStep` (for BM25) / `DenseRetrievalStep` (for dense models)

**For BM25:**
- Uses rank_bm25 (BM25Okapi) with cached index from index step
- For overlapping IDs: prefers document content over topic content
- Automatically generates UDLF input files (ranked-list and lists)
- **Files Created:**
  - `ranks/bm25/retrieval.tsv` - Raw retrieval results (query_id\tdoc_id\tscore)
  - `rerank/input/bm25/ranked-list/data.txt` - UDLF horizontal format (auto-generated)
  - `rerank/input/bm25/lists/data.txt` - Unique ID list (auto-generated)
- **Implementation:** `runner.py::RetrievalStep`

**For Dense Models:**
- Loads FAISS index and encodes queries using same model
- For overlapping IDs: prefers document content over topic content
- Automatically generates UDLF input files
- **Files Created:**
  - `ranks/<model_name>/retrieval.tsv` - Raw retrieval results
  - `ranks/<model_name>/query_embeddings.npy` - Cached query embeddings (numpy array)
  - `rerank/input/<model_name>/ranked-list/data.txt` - UDLF horizontal format (auto-generated)
  - `rerank/input/<model_name>/lists/data.txt` - Unique ID list (auto-generated)
- **Implementation:** `runner.py::DenseRetrievalStep`

**Auto-generation of UDLF Input Files:**
Both BM25 and Dense retrieval automatically call `_prepare_rerank_inputs()` to create:
- **ranked-list/data.txt** - Horizontal format (one line per query with space-separated doc IDs)
  ```
  query_id_1 doc_id_1 doc_id_2 doc_id_3 ...
  query_id_2 doc_id_5 doc_id_1 doc_id_8 ...
  ```
- **lists/data.txt** - Vertical format (one unique ID per line, queries first, then docs)
  ```
  query_id_1
  query_id_2
  doc_id_1
  doc_id_2
  ...
  ```

---

#### Step: rerank
#### Step: rerank
**Purpose:** Apply UDLF re-ranking methods to improve retrieval results.

**Pipeline Step:** `ReRankStep`

**Process:**
1. Loads pre-generated UDLF input files from `rerank/input/<model_name>/`
   - Tries to download from GCS if not available locally
   - Falls back to generating from `retrieval.tsv` if necessary
2. For each experiment configuration (method × k-value):
   - Runs UDLF binary with specified parameters
   - Method: CPRR, BFS-TREE, or RDPAC
   - K value: controls algorithm behavior (1, 3, 5, 10, 20, 30, 40, 50, 75)
3. Writes outputs for each experiment

**Files Created per Experiment:** `rerank/output/<model_name>/<method>-k<value>/`
- `data.txt` - UDLF output (horizontal ranked list format)
- `data.trec` - TREC format (query_id Q0 doc_id rank score run_tag)
- `ranks.tsv` - Vertical format (query_id\tdoc_id\trerank_score)
- `config.json` - Experiment parameters and metadata
- `metrics.json` - Evaluation metrics (created by subsequent metrics step)

**Skip Logic:** Experiments with existing `data.txt` output are automatically skipped

**Implementation:** `runner.py::ReRankStep`

---

#### Step: metrics
**Purpose:** Evaluate retrieval and re-ranking performance using standard IR metrics.

**Pipeline Step:** `MetricsStep`

**Process:**
1. Loads retrieval results from `ranks/<model_name>/retrieval.tsv`
2. Loads relevance judgments (qrels) from `extracted/qrels/data.tsv.gz`
3. For category-based datasets: Uses `category_metadata.json` for on-the-fly relevance
4. Computes metrics at various cutoffs (k values)

**Metrics Computed:**
- **MAP** (Mean Average Precision) - @5, @10, @20, @50, @100, @200, @1000
- **Recall** - @5, @10, @20, @50, @100, @200, @1000
- **Precision** - @5, @10, @20, @50, @100, @200, @1000
- **nDCG** (Normalized Discounted Cumulative Gain) - @5, @10, @20, @50, @100, @200, @1000

**Files Created:**
- `ranks/<model_name>/metrics.json` - Baseline retrieval metrics
- `rerank/output/<model_name>/<method>-k<value>/metrics.json` - Re-ranking metrics (when run after rerank step)

**Output Format:**
```json
{
  "bm25": {
    "MAP@100": 0.2679,
    "nDCG@10": 0.3208,
    "Recall@100": 0.9090,
    "Precision@10": 0.0797,
    ...
  }
}
```

**Validation:** Skips writing metrics if all values are zero (indicates qrels/category metadata issues)

**Implementation:** `runner.py::MetricsStep`, `metrics/rerank_eval.py::evaluate_rows()`

---

## GCS Storage Organization200, 1000
- **nDCG (Normalized Discounted Cumulative Gain)** @ 5, 10, 20, 50, 100, 200, 1000
- **Recall** @ 5, 10, 20, 50, 100, 200, 1000
- **Precision** @ 5, 10, 20, 50, 100, 200, 1000

**Requires:**
- Ranked results (retrieval.tsv or reranked ranks.tsv)
- Qrels file (`extracted/qrels/data.tsv.gz`)

**Output:** `metrics.json` in respective directory
- For baselines: `ranks/<model_name>/metrics.json`
- For re-ranking: `rerank/output/<model_name>/<method>-k<value>/metrics.json`

**Implementation:** `runner.py` - `MetricsStep.run()` (lines ~1200-1300)

---

## GCS Storage Organization

**Base Path:** `gs://text-udlf-expresso/outputs/paper-assets/dataset/`

**Complete Path Structure:**
```
gs://text-udlf-expresso/outputs/paper-assets/dataset/
├── arguana/
│   ├── extracted/
│   ├── indexes/
│   ├── ranks/
│   │   ├── bm25/metrics.json
│   │   ├── miniLM-L6-v2/metrics.json
│   │   ├── bge-small-en-v1.5/metrics.json
│   │   └── contriever-msmarco/metrics.json
│   ├── rerank/
│   │   └── output/
│   │       ├── bm25/
│   │       │   ├── cprr-k1/metrics.json
│   │       │   ├── cprr-k3/metrics.json
│   │       │   ├── ...
│   │       │   ├── bfstree-k1/metrics.json
│   │       │   ├── ...
│   │       │   └── rdpac-k75/metrics.json
│   │       ├── miniLM-L6-v2/
│   │       │   └── [27 method-k combinations]
│   │       ├── bge-small-en-v1.5/
│   │       │   └── [27 method-k combinations]
│   │       └── contriever-msmarco/
│   │           └── [27 method-k combinations]
│   └── transformed/
├── scifact/
│   └── [same structure]
├── nfcorpus/
│   └── [same structure]
├── scidocs/
│   └── [same structure]
├── fiqa/
│   └── [same structure]
├── wos/
│   └── [same structure]
└── bbc-news/
    └── [same structure]
```

**Experiment Tracking:**
Each completed experiment has: `<dataset>/rerank/output/<model>/<method>-k<value>/metrics.json`

**Total Expected Files:**
- Baseline metrics: 7 datasets × 4 models = 28 files
- Re-ranking metrics: 7 datasets × 4 models × 3 methods × 9 k-values = 756 files
- **Grand Total: 784 metrics.json files**

---

## Progress Monitoring

To check progress:
```bash
python monitor_experiment_progress.py
```

This generates `experiment_progress_report.html` with:
- Overall completion percentage
- Per-dataset breakdown
- Missing experiment identification
- Estimated completion time

---

## Visual Re-ranking Comparison Report

### Overview
To better understand the practical impact of re-ranking beyond aggregate metrics, the framework includes a visual comparison report generator that showcases specific queries where re-ranking significantly improves the TOP@20 results.

### Features
- **Side-by-side comparisons**: Original ranking vs re-ranked results for each query
- **Visual relevance indicators**: Green borders for relevant documents, red for non-relevant
- **Automatic K-value selection**: Identifies the optimal K parameter for each dataset/model/method combination
- **Top improved queries**: Displays up to 5 queries with highest precision improvement per configuration
- **Comprehensive metrics**: Shows baseline vs re-ranked performance with percentage improvements
- **Category-based support**: Works with both BEIR datasets (explicit qrels) and category-based datasets

### Generate Report

**Basic usage (with GCS data):**
```bash
conda activate udlf-expresso
python generate_rerank_report.py \
  --datasets bbc-news wos mental-health \
  --models bm25 miniLM-L6-v2 \
  --methods cprr bfstree rdpac \
  --output rerank_comparison_report.html
```

**Using local data (without GCS):**
```bash
python generate_rerank_report.py \
  --datasets bbc-news \
  --models bm25 \
  --methods cprr \
  --output report.html \
  --no-gcs \
  --local-outputs outputs
```

**All available options:**
```bash
python generate_rerank_report.py --help

Options:
  --datasets        List of datasets to include (default: scifact nfcorpus arguana bbc-news wos)
  --models          List of models to analyze (default: bm25 miniLM-L6-v2)
  --methods         List of re-ranking methods (default: cprr bfstree rdpac)
  --output          Output HTML file path (default: rerank_comparison_report.html)
  --progress-data   Path to experiment_progress_data.json (default: experiment_progress_data.json)
  --run-id          Experiment run ID (default: paper-assets)
  --no-gcs          Use local files instead of GCS
  --local-outputs   Local outputs directory (default: outputs)
```

### Report Structure

The generated HTML report includes:

1. **Introduction**: Explains the purpose and value of visual comparison
2. **Experimental Setup**: Details about datasets, models, methods, and configuration
3. **Dataset Sections**: For each dataset/model/method combination:
   - Dataset type (BEIR vs Category-based)
   - Optimal K value automatically selected
   - Metric improvements (MAP@20, Precision@20)
   - Top 5 queries with highest improvements
   - Visual comparison tables showing:
     - Query document (blue border)
     - Original TOP@20 ranking (first row)
     - Re-ranked TOP@20 ranking (second row)
     - Relevant documents (green borders)
     - Non-relevant documents (red borders)
4. **Conclusion**: Summary of key observations

### Example Output

The report clearly shows how re-ranking clusters relevant documents at the top of result lists:

```
Original Ranking:  [Query] → [Rel] [Rel] [Not] [Not] [Not] [Rel] [Rel] ...
Re-ranked Ranking: [Query] → [Rel] [Rel] [Rel] [Rel] [Not] [Rel] [Not] ...
                             └─── More relevant docs at top ───┘
```

### Best Practices

1. **Dataset Selection**: 
   - Category-based datasets (bbc-news, wos, mental-health) typically show clearer improvements
   - BEIR datasets may show variable results depending on query characteristics

2. **Model Selection**:
   - Start with BM25 baseline to see how re-ranking improves sparse retrieval
   - Compare with dense models (miniLM, BGE, Contriever) to see different patterns

3. **Method Comparison**:
   - Include multiple methods (cprr, bfstree, rdpac) to see which works best per dataset
   - The script automatically selects the best K value for each combination

4. **Report Size**:
   - Each dataset/model/method combination adds 5 query comparisons
   - For large reports, consider generating separate reports per dataset type

### Technical Notes

- The report generator reads data from GCS by default (requires `google-cloud-storage` package)
- Requires `experiment_progress_data.json` to identify optimal K values
- Automatically detects category-based datasets using `category_metadata.json`
- Self-contained HTML file with embedded CSS (no external dependencies)
- File size: ~50-100KB per dataset section depending on number of queries

---
