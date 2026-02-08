#!/usr/bin/env python3
"""
UMAP Visualization for Category Clustering Analysis.

This module generates UMAP projections to visualize how query/document categories
cluster in the embedding space. It supports before/after re-ranking comparison
to show how re-ranking affects the category distribution in ranked lists.

Supports both local filesystem and Google Cloud Storage (GCS) backends.

The visualization helps answer questions like:
- Do queries from the same category cluster together in embedding space?
- How does re-ranking affect the category composition of retrieved documents?
- Which categories are more separable in the embedding space?
"""

from __future__ import annotations

import json
import gzip
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

try:
    import umap
    HAS_UMAP = True
except ImportError:
    umap = None  # type: ignore
    HAS_UMAP = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None  # type: ignore
    px = None  # type: ignore
    HAS_PLOTLY = False

try:
    from google.cloud import storage as gcs_storage
    HAS_GCS = True
except ImportError:
    gcs_storage = None  # type: ignore
    HAS_GCS = False


def _ensure_dependencies():
    """Check that required dependencies are available."""
    if not HAS_UMAP:
        raise RuntimeError(
            "UMAP visualization requires umap-learn. "
            "Install with: pip install umap-learn"
        )
    if not HAS_PLOTLY:
        raise RuntimeError(
            "UMAP visualization requires plotly. "
            "Install with: pip install plotly"
        )


def _extract_category(doc_id: str) -> str:
    """Extract category from document ID like 'sport_001' -> 'sport' or 'doc::sport_001' -> 'sport'."""
    if doc_id.startswith("doc::"):
        doc_id = doc_id[5:]
    if "_" in doc_id:
        return doc_id.split("_", 1)[0]
    return "unknown"


@dataclass
class UMAPConfig:
    """Configuration for UMAP projection."""
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 2
    metric: str = "cosine"
    random_state: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "n_components": self.n_components,
            "metric": self.metric,
            "random_state": self.random_state,
        }


@dataclass
class CategoryClusterData:
    """Data structure holding embedding and category information."""
    doc_ids: List[str]
    embeddings: np.ndarray
    categories: List[str]
    category_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def unique_categories(self) -> List[str]:
        return sorted(set(self.categories))
    
    @property
    def n_samples(self) -> int:
        return len(self.doc_ids)


@dataclass
class RankingSnapshot:
    """Snapshot of ranked results for a query (before or after re-ranking)."""
    query_id: str
    query_category: str
    ranked_doc_ids: List[str]
    ranked_categories: List[str]
    scores: List[float]


class CategoryUMAPVisualizer:
    """
    Generate UMAP visualizations for category clustering analysis.
    
    This class provides methods to:
    1. Load embeddings and category metadata (from local or GCS)
    2. Project embeddings to 2D using UMAP
    3. Compare category distributions before/after re-ranking
    4. Generate interactive HTML reports
    """
    
    def __init__(
        self,
        dataset_path: str,
        model_name: str,
        umap_config: Optional[UMAPConfig] = None,
        sample_size: Optional[int] = None,
        use_gcs: bool = True,
        bucket_name: str = "text-udlf-expresso",
    ):
        """
        Initialize the visualizer.
        
        Args:
            dataset_path: Path to the dataset directory (e.g., outputs/paper-assets/dataset/mental-health)
                         This is the path within the bucket if use_gcs=True
            model_name: Name of the embedding model (e.g., 'bge-small-en-v1.5')
            umap_config: UMAP projection configuration
            sample_size: Optional sample size for large datasets (random sampling)
            use_gcs: Whether to fetch data from GCS (default: True)
            bucket_name: GCS bucket name (if use_gcs=True)
        """
        _ensure_dependencies()
        
        self.dataset_path = dataset_path.rstrip("/")
        self.model_name = model_name
        self.umap_config = umap_config or UMAPConfig()
        self.sample_size = sample_size
        self.use_gcs = use_gcs
        self.bucket_name = bucket_name
        
        # Initialize GCS client if needed
        if use_gcs:
            if not HAS_GCS:
                raise RuntimeError(
                    "GCS support requires google-cloud-storage. "
                    "Install with: pip install google-cloud-storage"
                )
            self.gcs_client = gcs_storage.Client()
            self.bucket = self.gcs_client.bucket(bucket_name)
        else:
            self.gcs_client = None
            self.bucket = None
        
        # Paths (relative to dataset_path)
        self.index_path = f"{self.dataset_path}/indexes/{model_name}"
        self.qrels_path = f"{self.dataset_path}/extracted/qrels"
        self.ranks_path = f"{self.dataset_path}/ranks/{model_name}"
        self.rerank_input_path = f"{self.dataset_path}/rerank/input/{model_name}"
        self.rerank_output_path = f"{self.dataset_path}/rerank/output"
        
        # Cached data
        self._embeddings: Optional[np.ndarray] = None
        self._doc_ids: Optional[List[str]] = None
        self._category_metadata: Optional[Dict[str, Any]] = None
        self._umap_projection: Optional[np.ndarray] = None
        
        # Temp directory for downloaded files
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
    
    def _read_file(self, path: str) -> bytes:
        """Read file from GCS or local filesystem."""
        if self.use_gcs:
            blob = self.bucket.blob(path)
            return blob.download_as_bytes()
        else:
            return Path(path).read_bytes()
    
    def _file_exists(self, path: str) -> bool:
        """Check if file exists in GCS or local filesystem."""
        if self.use_gcs:
            blob = self.bucket.blob(path)
            return blob.exists()
        else:
            return Path(path).exists()
    
    def _download_file_to_temp(self, gcs_path: str, filename: str) -> Path:
        """Download a file from GCS to a temporary location."""
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
        
        local_path = Path(self._temp_dir.name) / filename
        
        if self.use_gcs:
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(str(local_path))
        else:
            # For local, just return the original path
            return Path(gcs_path)
        
        return local_path
    
    def load_category_metadata(self) -> Dict[str, Any]:
        """Load category metadata from the qrels directory."""
        if self._category_metadata is not None:
            return self._category_metadata
        
        metadata_path = f"{self.qrels_path}/category_metadata.json"
        
        if not self._file_exists(metadata_path):
            raise FileNotFoundError(
                f"Category metadata not found at {metadata_path}. "
                "This visualization is designed for category-based datasets."
            )
        
        metadata_bytes = self._read_file(metadata_path)
        self._category_metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        return self._category_metadata
    
    def load_embeddings_and_ids(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load embeddings from FAISS index and document IDs.
        
        Returns:
            Tuple of (embeddings array, list of document IDs)
        """
        if self._embeddings is not None and self._doc_ids is not None:
            return self._embeddings, self._doc_ids
        
        try:
            import faiss
        except ImportError:
            raise RuntimeError("Loading embeddings requires faiss. Install with: pip install faiss-cpu")
        
        # Load document IDs
        docids_path = f"{self.index_path}/docids.json"
        if not self._file_exists(docids_path):
            raise FileNotFoundError(f"Document IDs not found at {docids_path}")
        
        docids_bytes = self._read_file(docids_path)
        docids_data = json.loads(docids_bytes.decode('utf-8'))
        self._doc_ids = docids_data.get("doc_ids", [])
        
        # Download FAISS index to temp location (faiss needs a file path)
        faiss_gcs_path = f"{self.index_path}/faiss.index"
        if not self._file_exists(faiss_gcs_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_gcs_path}")
        
        print(f"Downloading FAISS index from {'GCS' if self.use_gcs else 'local'}...")
        local_faiss_path = self._download_file_to_temp(faiss_gcs_path, "faiss.index")
        
        # Load FAISS index and extract embeddings
        print("Loading FAISS index...")
        index = faiss.read_index(str(local_faiss_path))
        
        # Extract all vectors from the index
        n_vectors = index.ntotal
        d = index.d
        
        print(f"Extracting {n_vectors:,} embeddings of dimension {d}...")
        self._embeddings = np.zeros((n_vectors, d), dtype=np.float32)
        for i in range(n_vectors):
            self._embeddings[i] = index.reconstruct(i)
        
        return self._embeddings, self._doc_ids
    
    def prepare_cluster_data(self) -> CategoryClusterData:
        """
        Prepare data structure with embeddings and category labels.
        
        Returns:
            CategoryClusterData with embeddings, doc_ids, and categories
        """
        embeddings, doc_ids = self.load_embeddings_and_ids()
        metadata = self.load_category_metadata()
        
        # Extract categories from document IDs
        categories = [_extract_category(did) for did in doc_ids]
        
        # Optionally sample for large datasets
        if self.sample_size and len(doc_ids) > self.sample_size:
            print(f"Sampling {self.sample_size:,} documents from {len(doc_ids):,}...")
            rng = np.random.default_rng(self.umap_config.random_state)
            indices = rng.choice(len(doc_ids), size=self.sample_size, replace=False)
            indices = np.sort(indices)
            
            embeddings = embeddings[indices]
            doc_ids = [doc_ids[i] for i in indices]
            categories = [categories[i] for i in indices]
        
        return CategoryClusterData(
            doc_ids=doc_ids,
            embeddings=embeddings,
            categories=categories,
            category_counts=metadata.get("categories", {}),
        )
    
    def compute_umap_projection(
        self,
        cluster_data: CategoryClusterData,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """
        Compute UMAP 2D projection of embeddings.
        
        Args:
            cluster_data: CategoryClusterData with embeddings
            force_recompute: If True, recompute even if cached
            
        Returns:
            2D numpy array with UMAP coordinates
        """
        if self._umap_projection is not None and not force_recompute:
            return self._umap_projection
        
        print(f"Computing UMAP projection for {cluster_data.n_samples:,} samples...")
        reducer = umap.UMAP(**self.umap_config.to_dict())
        self._umap_projection = reducer.fit_transform(cluster_data.embeddings)
        
        return self._umap_projection
    
    def load_ranking_data(
        self,
        ranking_file: str,
        top_k: int = 100,
    ) -> Dict[str, RankingSnapshot]:
        """
        Load ranking data from a TREC-format file.
        
        Args:
            ranking_file: Path to TREC run file or ranked list (local or GCS path)
            top_k: Maximum number of results per query
            
        Returns:
            Dictionary mapping query_id to RankingSnapshot
        """
        rankings: Dict[str, RankingSnapshot] = {}
        
        if not self._file_exists(ranking_file):
            print(f"Warning: Ranking file not found: {ranking_file}")
            return rankings
        
        # Read ranking file
        ranking_bytes = self._read_file(ranking_file)
        ranking_text = ranking_bytes.decode('utf-8')
        
        # Parse TREC format: query_id Q0 doc_id rank score run_name
        # or simpler format: query_id doc_id score rank
        query_results: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
        
        for line in ranking_text.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 4:
                # Try TREC format first
                if len(parts) >= 6:
                    query_id = parts[0]
                    doc_id = parts[2]
                    rank = int(parts[3])
                    score = float(parts[4])
                else:
                    query_id = parts[0]
                    doc_id = parts[1]
                    score = float(parts[2]) if len(parts) > 2 else 0.0
                    rank = int(parts[3]) if len(parts) > 3 else len(query_results[query_id]) + 1
                
                if rank <= top_k:
                    query_results[query_id].append((doc_id, score, rank))
        
        for query_id, results in query_results.items():
            results = sorted(results, key=lambda x: x[2])[:top_k]
            doc_ids = [r[0] for r in results]
            scores = [r[1] for r in results]
            categories = [_extract_category(did) for did in doc_ids]
            
            rankings[query_id] = RankingSnapshot(
                query_id=query_id,
                query_category=_extract_category(query_id),
                ranked_doc_ids=doc_ids,
                ranked_categories=categories,
                scores=scores,
            )
        
        return rankings
    
    def compute_category_distribution_at_k(
        self,
        rankings: Dict[str, RankingSnapshot],
        k_values: List[int] = [10, 20, 50, 100],
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Compute category distribution at different cutoff values.
        
        For each query category, computes the distribution of retrieved document
        categories at various k values.
        
        Args:
            rankings: Dictionary of RankingSnapshot by query_id
            k_values: List of cutoff values to analyze
            
        Returns:
            Nested dict: query_category -> k -> {doc_category: proportion}
        """
        distributions: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        
        category_query_counts: Dict[str, int] = defaultdict(int)
        
        for query_id, snapshot in rankings.items():
            query_cat = snapshot.query_category
            category_query_counts[query_cat] += 1
            
            for k in k_values:
                cats_at_k = snapshot.ranked_categories[:k]
                for cat in cats_at_k:
                    distributions[query_cat][k][cat] += 1
        
        # Normalize to proportions
        for query_cat in distributions:
            n_queries = category_query_counts[query_cat]
            for k in distributions[query_cat]:
                for doc_cat in distributions[query_cat][k]:
                    # Average count per query, then divide by k for proportion
                    avg_count = distributions[query_cat][k][doc_cat] / n_queries
                    distributions[query_cat][k][doc_cat] = avg_count / k
        
        return dict(distributions)
    
    def generate_embedding_scatter_figure(
        self,
        cluster_data: CategoryClusterData,
        umap_coords: np.ndarray,
        title: str = "UMAP Projection of Document Embeddings by Category",
    ) -> "go.Figure":
        """
        Generate an interactive scatter plot of UMAP projection colored by category.
        
        Args:
            cluster_data: CategoryClusterData with categories
            umap_coords: 2D UMAP coordinates
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        # Create color mapping for categories
        unique_cats = cluster_data.unique_categories
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_cats)}
        
        fig = go.Figure()
        
        for category in unique_cats:
            mask = np.array([c == category for c in cluster_data.categories])
            count = cluster_data.category_counts.get(category, mask.sum())
            
            fig.add_trace(go.Scatter(
                x=umap_coords[mask, 0],
                y=umap_coords[mask, 1],
                mode='markers',
                name=f"{category} ({count:,})",
                marker=dict(
                    size=5,
                    color=color_map[category],
                    opacity=0.6,
                ),
                text=[cluster_data.doc_ids[i] for i in np.where(mask)[0]],
                hovertemplate="%{text}<br>Category: " + category + "<extra></extra>",
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            legend_title="Categories",
            hovermode='closest',
            width=1000,
            height=800,
            template="plotly_white",
        )
        
        return fig
    
    def generate_category_distribution_comparison(
        self,
        before_rankings: Dict[str, RankingSnapshot],
        after_rankings: Dict[str, RankingSnapshot],
        k: int = 20,
        title: str = "Category Distribution in Ranked Lists: Before vs After Re-ranking",
    ) -> "go.Figure":
        """
        Generate comparison visualization of category distributions before/after re-ranking.
        
        Shows for each query category, what proportion of retrieved documents
        belong to each document category.
        
        Args:
            before_rankings: Rankings before re-ranking
            after_rankings: Rankings after re-ranking
            k: Cutoff value for analysis
            title: Plot title
            
        Returns:
            Plotly Figure with subplots
        """
        before_dist = self.compute_category_distribution_at_k(before_rankings, [k])
        after_dist = self.compute_category_distribution_at_k(after_rankings, [k])
        
        # Get all categories (both query and doc categories)
        all_query_cats = sorted(set(before_dist.keys()) | set(after_dist.keys()))
        all_doc_cats = set()
        for qc in all_query_cats:
            all_doc_cats.update(before_dist.get(qc, {}).get(k, {}).keys())
            all_doc_cats.update(after_dist.get(qc, {}).get(k, {}).keys())
        all_doc_cats = sorted(all_doc_cats)
        
        # Create subplot for each query category
        n_rows = len(all_query_cats)
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=[f"Queries from '{qc}'" for qc in all_query_cats],
            vertical_spacing=0.08,
        )
        
        colors = px.colors.qualitative.Set1
        
        for row_idx, query_cat in enumerate(all_query_cats, start=1):
            before_vals = [before_dist.get(query_cat, {}).get(k, {}).get(dc, 0) for dc in all_doc_cats]
            after_vals = [after_dist.get(query_cat, {}).get(k, {}).get(dc, 0) for dc in all_doc_cats]
            
            fig.add_trace(
                go.Bar(
                    name="Before Re-ranking" if row_idx == 1 else None,
                    x=all_doc_cats,
                    y=before_vals,
                    marker_color=colors[0],
                    opacity=0.7,
                    showlegend=(row_idx == 1),
                    legendgroup="before",
                ),
                row=row_idx, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    name="After Re-ranking" if row_idx == 1 else None,
                    x=all_doc_cats,
                    y=after_vals,
                    marker_color=colors[1],
                    opacity=0.7,
                    showlegend=(row_idx == 1),
                    legendgroup="after",
                ),
                row=row_idx, col=1
            )
            
            # Add marker for same-category (relevant) proportion
            same_cat_before = before_dist.get(query_cat, {}).get(k, {}).get(query_cat, 0)
            same_cat_after = after_dist.get(query_cat, {}).get(k, {}).get(query_cat, 0)
            
            fig.add_annotation(
                x=query_cat,
                y=max(same_cat_before, same_cat_after) + 0.05,
                text=f"Δ={same_cat_after - same_cat_before:+.1%}",
                showarrow=False,
                font=dict(size=10, color="green" if same_cat_after > same_cat_before else "red"),
                row=row_idx, col=1,
            )
        
        fig.update_layout(
            title=dict(text=f"{title} (Top-{k})", x=0.5, font=dict(size=16)),
            barmode='group',
            height=300 * n_rows,
            width=1000,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        
        for i in range(1, n_rows + 1):
            fig.update_yaxes(title_text="Proportion", row=i, col=1)
            fig.update_xaxes(title_text="Document Category", row=i, col=1)
        
        return fig
    
    def generate_relevance_improvement_heatmap(
        self,
        before_rankings: Dict[str, RankingSnapshot],
        after_rankings: Dict[str, RankingSnapshot],
        k_values: List[int] = [10, 20, 50, 100],
        title: str = "Relevance Improvement by Query Category",
    ) -> "go.Figure":
        """
        Generate heatmap showing improvement in same-category retrieval.
        
        For category-based datasets, a relevant document is one from the same
        category as the query. This heatmap shows how re-ranking improves
        the proportion of relevant documents at different k values.
        
        Args:
            before_rankings: Rankings before re-ranking
            after_rankings: Rankings after re-ranking
            k_values: List of cutoff values
            title: Plot title
            
        Returns:
            Plotly Figure with heatmap
        """
        before_dist = self.compute_category_distribution_at_k(before_rankings, k_values)
        after_dist = self.compute_category_distribution_at_k(after_rankings, k_values)
        
        query_cats = sorted(set(before_dist.keys()) | set(after_dist.keys()))
        
        # Compute improvement matrix
        improvements = []
        for query_cat in query_cats:
            row = []
            for k in k_values:
                before_rel = before_dist.get(query_cat, {}).get(k, {}).get(query_cat, 0)
                after_rel = after_dist.get(query_cat, {}).get(k, {}).get(query_cat, 0)
                improvement = (after_rel - before_rel) * 100  # Convert to percentage points
                row.append(improvement)
            improvements.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=improvements,
            x=[f"@{k}" for k in k_values],
            y=query_cats,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f"{v:+.1f}%" for v in row] for row in improvements],
            texttemplate="%{text}",
            textfont=dict(size=12),
            hovertemplate="Query Category: %{y}<br>Cutoff: %{x}<br>Improvement: %{text}<extra></extra>",
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Cutoff (k)",
            yaxis_title="Query Category",
            width=800,
            height=400 + len(query_cats) * 50,
            template="plotly_white",
        )
        
        return fig
    
    def generate_query_trajectory_plot(
        self,
        cluster_data: CategoryClusterData,
        umap_coords: np.ndarray,
        before_rankings: Dict[str, RankingSnapshot],
        after_rankings: Dict[str, RankingSnapshot],
        sample_queries: Optional[List[str]] = None,
        n_sample: int = 5,
        top_k: int = 10,
        title: str = "Query Result Trajectories: Before vs After Re-ranking",
    ) -> "go.Figure":
        """
        Generate visualization showing how top-k results change in embedding space.
        
        For selected queries, shows the centroid of top-k results before and
        after re-ranking, with arrows indicating the shift.
        
        Args:
            cluster_data: CategoryClusterData with embeddings
            umap_coords: 2D UMAP coordinates
            before_rankings: Rankings before re-ranking
            after_rankings: Rankings after re-ranking
            sample_queries: Specific query IDs to visualize (optional)
            n_sample: Number of queries to sample per category if sample_queries not provided
            top_k: Number of top results to consider
            title: Plot title
            
        Returns:
            Plotly Figure
        """
        # Create doc_id to index mapping
        doc_id_to_idx = {did: i for i, did in enumerate(cluster_data.doc_ids)}
        
        # Sample queries if not provided
        if sample_queries is None:
            # Sample n_sample queries per category
            queries_by_cat: Dict[str, List[str]] = defaultdict(list)
            for qid in before_rankings.keys():
                if qid in after_rankings:
                    cat = _extract_category(qid)
                    queries_by_cat[cat].append(qid)
            
            sample_queries = []
            rng = np.random.default_rng(42)
            for cat, qids in queries_by_cat.items():
                n = min(n_sample, len(qids))
                sample_queries.extend(rng.choice(qids, size=n, replace=False).tolist())
        
        # Create figure with category scatter as background
        fig = self.generate_embedding_scatter_figure(
            cluster_data, umap_coords,
            title=title
        )
        
        # Add trajectory arrows for each sampled query
        for qid in sample_queries:
            if qid not in before_rankings or qid not in after_rankings:
                continue
            
            before_snap = before_rankings[qid]
            after_snap = after_rankings[qid]
            
            # Get coordinates of top-k docs
            before_coords = []
            for did in before_snap.ranked_doc_ids[:top_k]:
                if did in doc_id_to_idx:
                    before_coords.append(umap_coords[doc_id_to_idx[did]])
            
            after_coords = []
            for did in after_snap.ranked_doc_ids[:top_k]:
                if did in doc_id_to_idx:
                    after_coords.append(umap_coords[doc_id_to_idx[did]])
            
            if not before_coords or not after_coords:
                continue
            
            # Compute centroids
            before_centroid = np.mean(before_coords, axis=0)
            after_centroid = np.mean(after_coords, axis=0)
            
            query_cat = _extract_category(qid)
            
            # Add arrow from before to after
            fig.add_annotation(
                x=after_centroid[0],
                y=after_centroid[1],
                ax=before_centroid[0],
                ay=before_centroid[1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="black",
            )
            
            # Add query marker at before position
            fig.add_trace(go.Scatter(
                x=[before_centroid[0]],
                y=[before_centroid[1]],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='diamond',
                    color='white',
                    line=dict(color='black', width=2),
                ),
                name=f"Query: {qid}",
                showlegend=False,
                hovertemplate=f"Query: {qid}<br>Category: {query_cat}<br>Before centroid<extra></extra>",
            ))
            
            # Add marker at after position
            fig.add_trace(go.Scatter(
                x=[after_centroid[0]],
                y=[after_centroid[1]],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='star',
                    color='gold',
                    line=dict(color='black', width=2),
                ),
                showlegend=False,
                hovertemplate=f"Query: {qid}<br>Category: {query_cat}<br>After centroid<extra></extra>",
            ))
        
        return fig
    
    def generate_full_report(
        self,
        before_ranking_file: Optional[str] = None,
        after_ranking_file: Optional[str] = None,
        output_path: Optional[str] = None,
        k_values: List[int] = [10, 20, 50, 100],
    ) -> str:
        """
        Generate a complete HTML report with all visualizations.
        
        Args:
            before_ranking_file: Path to TREC file with original rankings (local or GCS)
            after_ranking_file: Path to TREC file with re-ranked results (local or GCS)
            output_path: Where to save the HTML report (local path)
            k_values: List of cutoff values for analysis
            
        Returns:
            Path to generated HTML report
        """
        print("Loading data and computing UMAP projection...")
        cluster_data = self.prepare_cluster_data()
        umap_coords = self.compute_umap_projection(cluster_data)
        
        # Generate embedding scatter plot
        fig_scatter = self.generate_embedding_scatter_figure(
            cluster_data, umap_coords,
            title=f"UMAP Projection: {self.model_name} Embeddings"
        )
        
        figures = [fig_scatter]
        
        # If we have before/after rankings, add comparison plots
        if before_ranking_file and after_ranking_file:
            print("Loading ranking data...")
            before_rankings = self.load_ranking_data(before_ranking_file)
            after_rankings = self.load_ranking_data(after_ranking_file)
            
            if before_rankings and after_rankings:
                print("Generating comparison visualizations...")
                
                # Category distribution comparison
                fig_dist = self.generate_category_distribution_comparison(
                    before_rankings, after_rankings, k=20
                )
                figures.append(fig_dist)
                
                # Relevance improvement heatmap
                fig_heatmap = self.generate_relevance_improvement_heatmap(
                    before_rankings, after_rankings, k_values=k_values
                )
                figures.append(fig_heatmap)
                
                # Query trajectory plot
                fig_trajectory = self.generate_query_trajectory_plot(
                    cluster_data, umap_coords,
                    before_rankings, after_rankings,
                    n_sample=3, top_k=10
                )
                figures.append(fig_trajectory)
        
        # Build HTML report
        html_content = self._build_html_report(figures, cluster_data)
        
        # Save report (always locally)
        if output_path is None:
            # Default local output path
            local_dataset_path = Path(self.dataset_path.replace("outputs/", "outputs/"))
            output_path = str(local_dataset_path / "reports" / f"umap_category_report_{self.model_name}.html")
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"Report saved to: {output_path}")
        return str(output_path)
    
    def _build_html_report(
        self,
        figures: List["go.Figure"],
        cluster_data: CategoryClusterData,
    ) -> str:
        """Build complete HTML report from figures."""
        
        # Convert figures to HTML divs using Plotly's built-in method
        # This is more reliable than manual JSON serialization
        figure_divs = []
        for i, fig in enumerate(figures):
            # Use Plotly's to_html with include_plotlyjs for first figure only
            include_js = 'cdn' if i == 0 else False
            fig_html = fig.to_html(
                full_html=False,
                include_plotlyjs=include_js,
                div_id=f"fig_{i}",
            )
            figure_divs.append(f'<div class="figure-container">{fig_html}</div>')
        
        # Build metadata section
        metadata_html = f'''
        <div class="metadata">
            <h2>Dataset Information</h2>
            <table>
                <tr><th>Model</th><td>{self.model_name}</td></tr>
                <tr><th>Total Documents</th><td>{cluster_data.n_samples:,}</td></tr>
                <tr><th>Categories</th><td>{len(cluster_data.unique_categories)}</td></tr>
                <tr><th>Data Source</th><td>{'GCS: ' + self.bucket_name if self.use_gcs else 'Local'}</td></tr>
            </table>
            <h3>Category Distribution</h3>
            <table>
                <tr><th>Category</th><th>Count</th><th>Proportion</th></tr>
                {''.join(f"<tr><td>{cat}</td><td>{cluster_data.category_counts.get(cat, 0):,}</td><td>{cluster_data.category_counts.get(cat, 0)/sum(cluster_data.category_counts.values())*100:.1f}%</td></tr>" for cat in cluster_data.unique_categories)}
            </table>
        </div>
        '''
        
        html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>UMAP Category Clustering Report - {self.model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #444;
            margin-top: 30px;
        }}
        .metadata {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .metadata table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .metadata th, .metadata td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .metadata th {{
            background: #e9e9e9;
            font-weight: 600;
        }}
        .figure-container {{
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: white;
        }}
        .explanation {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>UMAP Category Clustering Report</h1>
        
        {metadata_html}
        
        <div class="explanation">
            <strong>About this visualization:</strong>
            <p>This report shows how documents from different categories cluster in the embedding space using UMAP projection.
            For category-based datasets (like mental-health, bbc-news), documents are considered relevant to a query if they belong to the same category.</p>
            <p>The visualizations help understand:</p>
            <ul>
                <li>How well-separated categories are in the embedding space</li>
                <li>How re-ranking affects the category distribution of retrieved results</li>
                <li>Which categories benefit most from re-ranking</li>
            </ul>
        </div>
        
        {''.join(figure_divs)}
        
        <footer style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
            Generated by UDLF Text Espresso | UMAP Category Clustering Report
        </footer>
    </div>
</body>
</html>
'''
        return html
    
    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


def generate_category_umap_report(
    dataset_path: str,
    model_name: str,
    before_ranking: Optional[str] = None,
    after_ranking: Optional[str] = None,
    output_path: Optional[str] = None,
    sample_size: Optional[int] = 5000,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    use_gcs: bool = True,
    bucket_name: str = "text-udlf-expresso",
) -> str:
    """
    Convenience function to generate a UMAP category clustering report.
    
    Args:
        dataset_path: Path to dataset directory (within bucket if use_gcs=True)
        model_name: Name of embedding model
        before_ranking: Path to original ranking file (TREC format)
        after_ranking: Path to re-ranked file (TREC format)
        output_path: Where to save HTML report (local)
        sample_size: Max documents to include (for performance)
        umap_n_neighbors: UMAP n_neighbors parameter
        umap_min_dist: UMAP min_dist parameter
        use_gcs: Whether to fetch data from GCS
        bucket_name: GCS bucket name
        
    Returns:
        Path to generated report
    """
    config = UMAPConfig(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
    )
    
    visualizer = CategoryUMAPVisualizer(
        dataset_path=dataset_path,
        model_name=model_name,
        umap_config=config,
        sample_size=sample_size,
        use_gcs=use_gcs,
        bucket_name=bucket_name,
    )
    
    try:
        return visualizer.generate_full_report(
            before_ranking_file=before_ranking,
            after_ranking_file=after_ranking,
            output_path=output_path,
        )
    finally:
        visualizer.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate UMAP category clustering report")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset directory (e.g., outputs/paper-assets/dataset/mental-health)")
    parser.add_argument("--model", required=True, help="Embedding model name")
    parser.add_argument("--before", help="Path to original ranking file")
    parser.add_argument("--after", help="Path to re-ranked file")
    parser.add_argument("--output", help="Output HTML path (local)")
    parser.add_argument("--sample-size", type=int, default=5000, help="Max samples for UMAP")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--local", action="store_true", help="Use local filesystem instead of GCS")
    parser.add_argument("--bucket", default="text-udlf-expresso", help="GCS bucket name")
    
    args = parser.parse_args()
    
    report_path = generate_category_umap_report(
        dataset_path=args.dataset_path,
        model_name=args.model,
        before_ranking=args.before,
        after_ranking=args.after,
        output_path=args.output,
        sample_size=args.sample_size,
        umap_n_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        use_gcs=not args.local,
        bucket_name=args.bucket,
    )
    
    print(f"Report generated: {report_path}")
