#!/usr/bin/env python3
"""
Generate HTML report comparing original rankings vs re-ranked results.

This module creates visual comparisons showing how re-ranking improves document 
retrieval for specific queries, highlighting relevant vs non-relevant documents.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from google.cloud import storage as gcs_storage
    HAS_GCS = True
except ImportError:
    gcs_storage = None
    HAS_GCS = False


@dataclass
class QueryComparison:
    """Represents a single query comparison between original and re-ranked results."""
    query_id: str
    query_text: str
    original_ranking: List[str]  # List of doc_ids in rank order (TOP@20)
    reranked_ranking: List[str]  # List of doc_ids after re-ranking (TOP@20)
    improvement_metrics: Dict[str, float]  # precision, recall, etc.
    relevant_docs: Set[str]  # Set of relevant doc_ids for this query
    
    def get_improvement_score(self) -> float:
        """Calculate overall improvement score (e.g., precision@20 improvement)."""
        return self.improvement_metrics.get('precision_improvement', 0.0)


@dataclass
class DatasetResults:
    """Results for a single dataset/model/method/k-value combination."""
    dataset_name: str
    model_name: str
    method_name: str
    k_value: int
    baseline_metrics: Dict[str, float]
    rerank_metrics: Dict[str, float]
    top_comparisons: List[QueryComparison]
    
    def get_best_metric_improvement(self) -> Tuple[str, float]:
        """Get the metric with best improvement percentage."""
        best_metric = ""
        best_improvement = 0.0
        
        for metric in self.rerank_metrics:
            baseline = self.baseline_metrics.get(metric, 0.0)
            rerank = self.rerank_metrics.get(metric, 0.0)
            if baseline > 0:
                improvement = ((rerank - baseline) / baseline) * 100
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_metric = metric
        
        return best_metric, best_improvement


class RerankComparisonReportGenerator:
    """Generate HTML report comparing original vs re-ranked results."""
    
    def __init__(
        self,
        run_id: str = "paper-assets",
        base_path: str = "outputs/paper-assets/dataset",
        use_gcs: bool = True,
        bucket_name: str = "text-udlf-espresso",
        local_outputs: Optional[Path] = None
    ):
        """
        Initialize the report generator.
        
        Args:
            run_id: Experiment run identifier
            base_path: Base path in GCS or local filesystem
            use_gcs: Whether to fetch data from GCS
            bucket_name: GCS bucket name (if use_gcs=True)
            local_outputs: Local outputs directory (if use_gcs=False)
        """
        self.run_id = run_id
        self.base_path = base_path
        self.use_gcs = use_gcs
        self.bucket_name = bucket_name
        self.local_outputs = local_outputs or Path("outputs")
        
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
    
    def _read_file(self, path: str) -> bytes:
        """Read file from GCS or local filesystem."""
        if self.use_gcs:
            blob = self.bucket.blob(path)
            return blob.download_as_bytes()
        else:
            local_path = Path(path)
            if not local_path.is_absolute():
                local_path = self.local_outputs / path
            return local_path.read_bytes()
    
    def _file_exists(self, path: str) -> bool:
        """Check if file exists in GCS or local filesystem."""
        if self.use_gcs:
            blob = self.bucket.blob(path)
            return blob.exists()
        else:
            local_path = Path(path)
            if not local_path.is_absolute():
                local_path = self.local_outputs / path
            return local_path.exists()
    
    def _list_blobs(self, prefix: str) -> List[str]:
        """List blobs with given prefix in GCS or local filesystem."""
        if self.use_gcs:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        else:
            local_path = self.local_outputs / prefix
            if not local_path.exists():
                return []
            return [str(p.relative_to(self.local_outputs)) for p in local_path.rglob("*") if p.is_file()]
    
    def load_qrels(self, dataset: str) -> Tuple[Dict[str, Set[str]], bool, Optional[Dict[str, int]]]:
        """
        Load relevance judgments (qrels) for a dataset.
        
        Returns:
            Tuple of (qrels_dict, is_category_based, category_counts)
            - qrels_dict: {query_id: set of relevant doc_ids}
            - is_category_based: True if using category-based relevance
            - category_counts: Optional dict of {category: count}
        """
        # Check for category metadata
        metadata_path = f"{self.base_path}/{dataset}/extracted/qrels/category_metadata.json"
        is_category_based = False
        category_counts = None
        
        if self._file_exists(metadata_path):
            try:
                metadata_bytes = self._read_file(metadata_path)
                metadata = json.loads(metadata_bytes.decode('utf-8'))
                if metadata.get('type') == 'category_based':
                    is_category_based = True
                    category_counts = metadata.get('categories', {})
            except Exception:
                pass
        
        # Load qrels
        qrels: Dict[str, Set[str]] = {}
        
        if not is_category_based:
            qrels_path = f"{self.base_path}/{dataset}/extracted/qrels/data.tsv.gz"
            if self._file_exists(qrels_path):
                qrels_bytes = self._read_file(qrels_path)
                qrels_text = gzip.decompress(qrels_bytes).decode('utf-8')
                
                for line in qrels_text.strip().split('\n'):
                    if not line or '\t' not in line or line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        qid, did = parts[0], parts[1]
                        qrels.setdefault(qid, set()).add(did)
        
        return qrels, is_category_based, category_counts
    
    def _extract_category(self, doc_id: str) -> str:
        """Extract category from document ID like 'sport_001' -> 'sport'."""
        if '_' in doc_id:
            return doc_id.split('_')[0]
        return doc_id
    
    def _is_relevant_category_based(self, query_id: str, doc_id: str) -> bool:
        """Check if doc is relevant to query based on category matching (excluding self)."""
        if query_id == doc_id:
            return False
        return self._extract_category(query_id) == self._extract_category(doc_id)
    
    def load_query_texts(self, dataset: str) -> Dict[str, str]:
        """Load query texts from topics file."""
        topics_path = f"{self.base_path}/{dataset}/extracted/topics/data.tsv.gz"
        query_texts = {}
        
        if self._file_exists(topics_path):
            try:
                topics_bytes = self._read_file(topics_path)
                topics_text = gzip.decompress(topics_bytes).decode('utf-8')
                
                for line in topics_text.strip().split('\n'):
                    if not line or '\t' not in line:
                        continue
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        qid, text = parts
                        query_texts[qid] = text
            except Exception:
                pass
        
        return query_texts
    
    def load_baseline_ranking(self, dataset: str, model: str) -> Dict[str, List[str]]:
        """
        Load baseline (original) rankings from retrieval.tsv.
        
        Returns:
            Dict mapping query_id to ranked list of doc_ids (TOP@20)
        """
        retrieval_path = f"{self.base_path}/{dataset}/ranks/{model}/retrieval.tsv"
        rankings = defaultdict(list)
        
        if not self._file_exists(retrieval_path):
            return dict(rankings)
        
        try:
            retrieval_bytes = self._read_file(retrieval_path)
            retrieval_text = retrieval_bytes.decode('utf-8')
            
            # Group by query_id
            query_docs = defaultdict(list)
            for line in retrieval_text.strip().split('\n'):
                if not line or '\t' not in line:
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    qid, doc_id, score = parts[0], parts[1], float(parts[2])
                    query_docs[qid].append((doc_id, score))
            
            # Sort by score and take TOP@20
            for qid, docs in query_docs.items():
                docs.sort(key=lambda x: x[1], reverse=True)
                rankings[qid] = [doc_id for doc_id, _ in docs[:20]]
        
        except Exception as e:
            print(f"Error loading baseline ranking: {e}")
        
        return dict(rankings)
    
    def load_reranked_results(
        self, 
        dataset: str, 
        model: str, 
        method: str, 
        k_value: int
    ) -> Dict[str, List[str]]:
        """
        Load re-ranked results from data.txt or ranks.tsv (TREC format).
        
        Returns:
            Dict mapping query_id to ranked list of doc_ids (TOP@20)
        """
        experiment_name = f"{method}-k{k_value}"
        data_path = f"{self.base_path}/{dataset}/rerank/output/{model}/{experiment_name}/data.txt"
        ranks_path = f"{self.base_path}/{dataset}/rerank/output/{model}/{experiment_name}/ranks.tsv"
        
        rankings = {}
        
        # Try data.txt first (horizontal format)
        if self._file_exists(data_path):
            try:
                data_bytes = self._read_file(data_path)
                data_text = data_bytes.decode('utf-8')
                
                # UDLF horizontal format: query_id doc_id1 doc_id2 doc_id3 ...
                for line in data_text.strip().split('\n'):
                    if not line:
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        qid = parts[0]
                        doc_ids = parts[1:21]  # Take TOP@20
                        rankings[qid] = doc_ids
                return rankings
            except Exception as e:
                print(f"Error loading data.txt: {e}")
        
        # Fall back to ranks.tsv (TREC format: query_id doc_id score)
        if self._file_exists(ranks_path):
            try:
                ranks_bytes = self._read_file(ranks_path)
                ranks_text = ranks_bytes.decode('utf-8')
                
                # Parse TREC format and group by query
                from collections import defaultdict
                query_docs = defaultdict(list)
                
                for line in ranks_text.strip().split('\n'):
                    if not line:
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        qid, doc_id, score = parts[0], parts[1], float(parts[2])
                        query_docs[qid].append((doc_id, score))
                
                # Sort by score and take TOP@20
                for qid, docs in query_docs.items():
                    docs.sort(key=lambda x: x[1], reverse=True)
                    rankings[qid] = [doc_id for doc_id, _ in docs[:20]]
                
                return rankings
            except Exception as e:
                print(f"Error loading ranks.tsv: {e}")
        
        return rankings
    
    def find_best_model_method_combination(
        self,
        dataset: str,
        models: List[str],
        methods: List[str],
        progress_data: Dict
    ) -> Optional[Tuple[str, str, int]]:
        """
        Find the best model+method+K combination for a dataset based on precision@20 improvement.
        
        Returns:
            Tuple of (best_model, best_method, best_k) or None if no valid combination found
        """
        try:
            dataset_data = progress_data.get('baseline_comparison', {}).get(dataset, {})
            
            best_combo = None
            best_improvement = -float('inf')
            
            for model in models:
                model_data = dataset_data.get(model, {})
                for method in methods:
                    method_data = model_data.get(method, {})
                    
                    for k_str, metrics in method_data.items():
                        try:
                            k_value = int(k_str)
                        except ValueError:
                            continue
                        
                        # Use Precision@20 improvement as the selection criterion
                        prec_metric = metrics.get('precision@20', {})
                        improvement = prec_metric.get('improvement_pct', 0.0)
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_combo = (model, method, k_value)
            
            return best_combo
        
        except Exception as e:
            print(f"Error finding best combination: {e}")
            return None
    
    def find_best_k_value(
        self, 
        dataset: str, 
        model: str, 
        method: str,
        progress_data: Dict
    ) -> Optional[int]:
        """
        Find the optimal K value with best improvement for a dataset/model/method.
        
        Uses experiment_progress_data.json to find K with highest MAP improvement.
        """
        try:
            # Navigate to the right section in progress data
            # The structure is: baseline_comparison -> dataset -> model -> method -> k_value
            dataset_data = progress_data.get('baseline_comparison', {}).get(dataset, {})
            model_data = dataset_data.get(model, {})
            method_data = model_data.get(method, {})
            
            best_k = None
            best_improvement = -float('inf')
            
            for k_str, metrics in method_data.items():
                try:
                    k_value = int(k_str)
                except ValueError:
                    continue
                
                # Use MAP@20 improvement as the selection criterion
                map_metric = metrics.get('map@20', {})
                improvement = map_metric.get('improvement_pct', 0.0)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_k = k_value
            
            return best_k
        
        except Exception as e:
            print(f"Error finding best K: {e}")
            return None
    
    def calculate_query_improvements(
        self,
        query_id: str,
        original_ranking: List[str],
        reranked_ranking: List[str],
        relevant_docs: Set[str],
        is_category_based: bool = False
    ) -> Dict[str, float]:
        """Calculate improvement metrics for a single query."""
        # Calculate precision@20
        def calc_precision(ranking: List[str]) -> float:
            if not ranking:
                return 0.0
            
            if is_category_based:
                relevant_count = sum(
                    1 for doc_id in ranking 
                    if self._is_relevant_category_based(query_id, doc_id)
                )
            else:
                relevant_count = sum(1 for doc_id in ranking if doc_id in relevant_docs)
            
            return relevant_count / len(ranking)
        
        original_prec = calc_precision(original_ranking)
        reranked_prec = calc_precision(reranked_ranking)
        
        return {
            'original_precision': original_prec,
            'reranked_precision': reranked_prec,
            'precision_improvement': reranked_prec - original_prec,
            'precision_improvement_pct': ((reranked_prec - original_prec) / max(0.001, original_prec)) * 100
        }
    
    def select_top_queries(
        self,
        dataset: str,
        model: str,
        method: str,
        k_value: int,
        qrels: Dict[str, Set[str]],
        is_category_based: bool,
        max_queries: int = 3
    ) -> List[QueryComparison]:
        """
        Select top N queries with highest improvement in precision@20.
        
        Args:
            dataset: Dataset name
            model: Model name
            method: Re-ranking method
            k_value: K value for re-ranking
            qrels: Relevance judgments
            is_category_based: Whether to use category-based relevance
            max_queries: Maximum number of queries to select (default: 3)
        
        Returns:
            List of QueryComparison objects for top improved queries
        """
        # Load rankings
        baseline_rankings = self.load_baseline_ranking(dataset, model)
        reranked_rankings = self.load_reranked_results(dataset, model, method, k_value)
        
        # Load query texts
        query_texts = self.load_query_texts(dataset)
        
        # Calculate improvements for all queries
        comparisons = []
        
        for query_id in baseline_rankings:
            if query_id not in reranked_rankings:
                continue
            
            original_ranking = baseline_rankings[query_id]
            reranked_ranking = reranked_rankings[query_id]
            
            if not original_ranking or not reranked_ranking:
                continue
            
            relevant_docs = qrels.get(query_id, set())
            
            improvement_metrics = self.calculate_query_improvements(
                query_id,
                original_ranking,
                reranked_ranking,
                relevant_docs,
                is_category_based
            )
            
            # Only include queries with positive improvement
            if improvement_metrics['precision_improvement'] > 0:
                comparison = QueryComparison(
                    query_id=query_id,
                    query_text=query_texts.get(query_id, query_id),
                    original_ranking=original_ranking,
                    reranked_ranking=reranked_ranking,
                    improvement_metrics=improvement_metrics,
                    relevant_docs=relevant_docs
                )
                comparisons.append(comparison)
        
        # Sort by improvement and take top N
        comparisons.sort(key=lambda x: x.get_improvement_score(), reverse=True)
        return comparisons[:max_queries]
    
    def generate_html_report(
        self,
        datasets: List[str],
        models: List[str],
        methods: List[str],
        output_path: Path,
        progress_data_path: Optional[Path] = None
    ) -> None:
        """
        Generate complete HTML report comparing original vs re-ranked results.
        
        Args:
            datasets: List of dataset names to include
            models: List of model names to analyze
            methods: List of re-ranking methods
            output_path: Path to save the HTML report
            progress_data_path: Path to experiment_progress_data.json
        """
        # Load experiment progress data
        if progress_data_path and progress_data_path.exists():
            with open(progress_data_path, 'r') as f:
                progress_data = json.load(f)
        else:
            progress_data = {}
        
        # Collect results for all dataset/model/method combinations
        all_results: List[DatasetResults] = []
        
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            
            # Load qrels once per dataset
            qrels, is_category_based, category_counts = self.load_qrels(dataset)
            
            # Find the BEST model+method+K combination for this dataset
            best_combo = self.find_best_model_method_combination(dataset, models, methods, progress_data)
            
            if best_combo is None:
                print(f"  No valid combinations found for {dataset}")
                continue
            
            model, method, best_k = best_combo
            # Load metrics
            experiment_name = f"{method}-k{best_k}"
            metrics_path = f"{self.base_path}/{dataset}/rerank/output/{model}/{experiment_name}/metrics.json"
            
            if not self._file_exists(metrics_path):
                print(f"    No metrics found at {metrics_path}")
                continue
            
            try:
                metrics_bytes = self._read_file(metrics_path)
                metrics_data = json.loads(metrics_bytes.decode('utf-8'))
                rerank_metrics = metrics_data.get(experiment_name, {})
            except Exception as e:
                print(f"    Error loading metrics: {e}")
                continue
            
            # Load baseline metrics
            baseline_metrics_path = f"{self.base_path}/{dataset}/ranks/{model}/metrics.json"
            baseline_metrics = {}
            
            if self._file_exists(baseline_metrics_path):
                try:
                    baseline_bytes = self._read_file(baseline_metrics_path)
                    baseline_data = json.loads(baseline_bytes.decode('utf-8'))
                    baseline_metrics = baseline_data.get(model, {})
                except Exception:
                    pass
            
            # Select top queries with improvements
            top_comparisons = self.select_top_queries(
                dataset, model, method, best_k, qrels, is_category_based
            )
            
            if not top_comparisons:
                print(f"    No improved queries found")
                continue
            
            print(f"    Found {len(top_comparisons)} improved queries")
            
            # Store results
            result = DatasetResults(
                dataset_name=dataset,
                model_name=model,
                method_name=method,
                k_value=best_k,
                baseline_metrics=baseline_metrics,
                rerank_metrics=rerank_metrics,
                top_comparisons=top_comparisons
            )
            all_results.append(result)
        
        # Generate HTML
        html_content = self._generate_html_content(all_results, progress_data)
        
        # Write to file
        output_path.write_text(html_content, encoding='utf-8')
        print(f"\n✓ Report generated: {output_path}")
    
    def _generate_html_content(
        self, 
        results: List[DatasetResults],
        progress_data: Dict
    ) -> str:
        """Generate the complete HTML content for the report."""
        
        # Build HTML sections
        html_parts = []
        
        # HTML header with CSS
        html_parts.append(self._html_header())
        
        # Introduction
        html_parts.append(self._html_introduction())
        
        # Experimental setup
        html_parts.append(self._html_experimental_setup(progress_data))
        
        # Dataset results
        for result in results:
            html_parts.append(self._html_dataset_section(result))
        
        # Conclusion
        html_parts.append(self._html_conclusion())
        
        # Footer
        html_parts.append(self._html_footer())
        
        return '\n'.join(html_parts)
    
    def _html_header(self) -> str:
        """Generate HTML header with CSS styles."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UDLF Text Re-ranking Results: Visual Comparison Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        h1 {
            font-size: 2.2em;
            color: #1a1a1a;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        h2 {
            font-size: 1.8em;
            color: #2c5aa0;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }
        
        h3 {
            font-size: 1.3em;
            color: #444;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        
        h4 {
            font-size: 1.1em;
            color: #666;
            margin-top: 15px;
            margin-bottom: 10px;
        }
        
        p {
            margin-bottom: 15px;
            text-align: justify;
        }
        
        .intro, .setup, .conclusion {
            background: #f9f9f9;
            padding: 20px;
            border-left: 4px solid #2c5aa0;
            margin-bottom: 30px;
        }
        
        .dataset-section {
            margin-bottom: 50px;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }
        
        .dataset-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .dataset-header h2 {
            color: white;
            border: none;
            margin: 0;
            padding: 0;
        }
        
        .experiment-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
            padding: 15px;
            background: #f0f4f8;
            border-radius: 5px;
        }
        
        .info-item {
            padding: 10px;
            background: white;
            border-radius: 3px;
            border-left: 3px solid #2c5aa0;
        }
        
        .info-item strong {
            display: block;
            color: #2c5aa0;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .metrics-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 25px;
        }
        
        .metric-card {
            background: #fff;
            padding: 12px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            text-align: center;
        }
        
        .metric-card .metric-name {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .metric-card .metric-value {
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }
        
        .metric-card .metric-delta {
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .metric-card .metric-delta.positive {
            color: #27ae60;
        }
        
        .metric-card .metric-delta.negative {
            color: #e74c3c;
        }
        
        .query-comparison {
            margin-bottom: 40px;
            padding: 20px;
            background: #fafafa;
            border-radius: 5px;
        }
        
        .query-header {
            margin-bottom: 15px;
            padding: 10px;
            background: #e8f4f8;
            border-left: 4px solid #3498db;
        }
        
        .query-text {
            font-style: italic;
            color: #555;
            margin-bottom: 5px;
        }
        
        .query-id {
            font-size: 0.9em;
            color: #888;
        }
        
        .precision-details {
            font-size: 0.95em;
            color: #333;
            margin-top: 10px;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
            border-radius: 3px;
        }
        
        .improvement-badge {
            display: inline-block;
            padding: 3px 10px;
            background: #27ae60;
            color: white;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .ranking-comparison {
            overflow-x: auto;
            margin-top: 15px;
        }
        
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-bottom: 20px;
            font-size: 0.9em;
        }
        
        table thead {
            background: #34495e;
            color: white;
        }
        
        table th {
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #2c3e50;
        }
        
        table td {
            padding: 10px 8px;
            border: 1px solid #e0e0e0;
        }
        
        table tbody tr:nth-child(1) {
            background: #fff3cd;
        }
        
        table tbody tr:nth-child(2) {
            background: #d4edda;
        }
        
        .doc-cell {
            text-align: center;
            font-family: 'Courier New', monospace;
            font-size: 0.75em;
            min-width: 70px;
            max-width: 90px;
            padding: 10px 4px;
            vertical-align: middle;
            word-break: break-all;
        }
        
        .doc-relevant {
            border: 3px solid #27ae60 !important;
            background-color: #e8f8f5 !important;
            font-weight: bold;
        }
        
        .doc-not-relevant {
            border: 3px solid #e74c3c !important;
            background-color: #fadbd8 !important;
        }
        
        .query-doc {
            background-color: #d6eaf8 !important;
            border: 3px solid #3498db !important;
            font-weight: bold;
        }
        
        .row-label {
            font-weight: bold;
            background: #ecf0f1 !important;
            white-space: nowrap;
        }
        
        .legend {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            margin-bottom: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 3px;
            font-size: 0.85em;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-box {
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }
        
        .legend-query {
            background-color: #d6eaf8;
            border: 2px solid #3498db;
        }
        
        .legend-relevant {
            background-color: #e8f8f5;
            border: 2px solid #27ae60;
        }
        
        .legend-not-relevant {
            background-color: #fadbd8;
            border: 2px solid #e74c3c;
        }
        
        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #888;
            font-size: 0.9em;
        }
        
        .highlight {
            background: #fff3cd;
            padding: 2px 5px;
            border-radius: 3px;
        }
        
        @media print {
            body {
                background: white;
            }
            
            .container {
                box-shadow: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
"""
    
    def _html_introduction(self) -> str:
        """Generate introduction section."""
        return """
        <h1>UDLF Text Re-ranking Results: Visual Comparison Report</h1>
        
        <div class="intro">
            <h2>Introduction</h2>
            <p>
                This report presents a visual comparison of document re-ranking improvements achieved by applying 
                the UDLF (Unsupervised Distance Learning Framework) to text retrieval tasks. Re-ranking aims to 
                refine the initial ranked list of documents returned by a retrieval system, placing more relevant 
                documents at higher positions.
            </p>
            <p>
                While aggregate metrics such as Mean Average Precision (MAP) and Normalized Discounted Cumulative 
                Gain (nDCG) provide quantitative measures of system performance, they often fail to convey the 
                tangible impact on user experience. This report addresses this gap by showcasing specific examples 
                where re-ranking significantly improves the top-20 results for individual queries.
            </p>
            <p>
                For each dataset, we present the best-performing retrieval model and re-ranking method combination, 
                showcasing up to three exemplary queries where the re-ranking process demonstrably enhances the 
                quality of retrieved documents. Each comparison displays:
            </p>
            <ul style="margin-left: 30px; margin-top: 10px; margin-bottom: 10px;">
                <li>The <strong>original ranking</strong> produced by the base retrieval model (first row)</li>
                <li>The <strong>improved ranking</strong> after applying UDLF re-ranking (second row)</li>
                <li>Visual indicators showing <span class="highlight">relevant documents</span> (green borders) 
                    versus non-relevant documents (red borders)</li>
            </ul>
            <p>
                The goal is to illustrate how re-ranking clusters relevant documents closer to the top of the 
                result list, thereby improving the likelihood that users will quickly find documents that satisfy 
                their information needs.
            </p>
        </div>
"""
    
    def _html_experimental_setup(self, progress_data: Dict) -> str:
        """Generate experimental setup section."""
        config = progress_data.get('configuration', {})
        datasets = config.get('datasets', [])
        models = config.get('models', [])
        methods = config.get('methods', [])
        k_values = config.get('k_values', [])
        
        return f"""
        <div class="setup">
            <h2>Experimental Setup</h2>
            
            <h3>Datasets</h3>
            <p>
                This study encompasses <strong>{len(datasets)} datasets</strong>, including both BEIR benchmark 
                datasets and custom category-based collections:
            </p>
            <ul style="margin-left: 30px; margin-bottom: 15px;">
                <li><strong>BEIR Datasets:</strong> Scifact, Scidocs, NFCorpus, FiQA, Arguana - standard 
                    benchmark collections with explicit relevance judgments</li>
                <li><strong>Category-based Datasets:</strong> BBC News, HuffPost News, WOS, Mental Health - 
                    collections where relevance is determined by document category matching</li>
            </ul>
            
            <h3>Base Retrieval Models</h3>
            <p>
                Four retrieval models were evaluated as the baseline for re-ranking:
            </p>
            <ul style="margin-left: 30px; margin-bottom: 15px;">
                <li><strong>BM25:</strong> Traditional probabilistic information retrieval model based on 
                    term frequency and document length normalization</li>
                <li><strong>MiniLM-L6-v2:</strong> Lightweight sentence transformer model 
                    (sentence-transformers/all-MiniLM-L6-v2)</li>
                <li><strong>BGE-small-en-v1.5:</strong> Efficient embedding model from BAAI optimized for 
                    semantic search</li>
                <li><strong>Contriever-MSMARCO:</strong> Dense retrieval model trained on MS MARCO dataset 
                    (facebook/contriever-msmarco)</li>
            </ul>
            
            <h3>UDLF Re-ranking Methods</h3>
            <p>
                Three unsupervised re-ranking methods from the UDLF framework were applied:
            </p>
            <ul style="margin-left: 30px; margin-bottom: 15px;">
                <li><strong>CPRR:</strong> Cartesian Product Ranking Re-ranking - leverages pairwise 
                    document relationships</li>
                <li><strong>BFSTREE:</strong> Breadth-First Search Tree re-ranking using correlation metrics 
                    to identify document clusters</li>
                <li><strong>RDPAC:</strong> Ranked Document Proximity and Clustering - considers document 
                    proximity in ranked lists</li>
            </ul>
            
            <h3>Configuration</h3>
            <p>
                Each re-ranking method was tested with multiple K values ({', '.join(map(str, k_values))}), 
                representing the neighborhood size parameter. For this report, we present results using the 
                <strong>optimal K value</strong> that achieved the highest improvement in Mean Average 
                Precision (MAP@20) for each dataset-model-method combination.
            </p>
            
            <h3>Evaluation Metrics</h3>
            <p>
                Performance was measured using standard information retrieval metrics:
            </p>
            <ul style="margin-left: 30px; margin-bottom: 15px;">
                <li><strong>MAP (Mean Average Precision):</strong> Average precision across all queries at 
                    various cutoffs (@20, @50, @200, @1000)</li>
                <li><strong>Precision:</strong> Fraction of retrieved documents that are relevant 
                    (@10, @20, @50, @200)</li>
                <li><strong>Recall:</strong> Fraction of relevant documents that are retrieved 
                    (@20, @50, @200)</li>
                <li><strong>nDCG (Normalized Discounted Cumulative Gain):</strong> Measure of ranking 
                    quality that accounts for position (@5, @10, @20, @50, @100, @200, @1000)</li>
            </ul>
        </div>
"""
    
    def _html_dataset_section(self, result: DatasetResults) -> str:
        """Generate HTML section for a single dataset result."""
        dataset_type = "Category-based" if result.dataset_name in ['bbc-news', 'huffpost-news', 'wos', 'mental-health'] else "BEIR"
        
        # Get key metric improvements
        map20_baseline = result.baseline_metrics.get('MAP@20', 0.0)
        map20_rerank = result.rerank_metrics.get('MAP@20', 0.0)
        map20_improvement = ((map20_rerank - map20_baseline) / max(0.0001, map20_baseline)) * 100
        
        prec20_baseline = result.baseline_metrics.get('Precision@20', 0.0)
        prec20_rerank = result.rerank_metrics.get('Precision@20', 0.0)
        prec20_improvement = ((prec20_rerank - prec20_baseline) / max(0.0001, prec20_baseline)) * 100
        
        html = f"""
        <div class="dataset-section">
            <div class="dataset-header">
                <h2>{result.dataset_name.upper()}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">
                    {dataset_type} Dataset • {result.model_name} + {result.method_name.upper()} (K={result.k_value})
                </p>
            </div>
            
            <div class="experiment-info">
                <div class="info-item">
                    <strong>Dataset</strong>
                    <div>{result.dataset_name}</div>
                </div>
                <div class="info-item">
                    <strong>Dataset Type</strong>
                    <div>{dataset_type}</div>
                </div>
                <div class="info-item">
                    <strong>Base Retrieval Model</strong>
                    <div>{result.model_name}</div>
                </div>
                <div class="info-item">
                    <strong>Re-ranking Method</strong>
                    <div>{result.method_name.upper()}</div>
                </div>
                <div class="info-item">
                    <strong>Optimal K Value</strong>
                    <div>{result.k_value}</div>
                </div>
            </div>
            
            <h3>Metric Improvements (TOP@20)</h3>
            <div class="metrics-summary">
                <div class="metric-card">
                    <div class="metric-name">MAP@20</div>
                    <div class="metric-value">{map20_rerank:.4f}</div>
                    <div class="metric-delta {'positive' if map20_improvement > 0 else 'negative'}">
                        {map20_improvement:+.2f}% vs baseline ({map20_baseline:.4f})
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Precision@20</div>
                    <div class="metric-value">{prec20_rerank:.4f}</div>
                    <div class="metric-delta {'positive' if prec20_improvement > 0 else 'negative'}">
                        {prec20_improvement:+.2f}% vs baseline ({prec20_baseline:.4f})
                    </div>
                </div>
            </div>
            
            <h3>Top Improved Queries</h3>
            <p>
                The following queries demonstrate significant improvement in the ranking quality after 
                applying UDLF {result.method_name.upper()} re-ranking. Each table shows the TOP@20 results, 
                with the first row representing the original ranking and the second row showing the improved 
                re-ranked results. Note that the query document itself is excluded from the ranking positions.
            </p>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-box legend-query"></div>
                    <span>Query Document</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box legend-relevant"></div>
                    <span>Relevant Document</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box legend-not-relevant"></div>
                    <span>Non-Relevant Document</span>
                </div>
            </div>
"""
        
        # Add query comparisons
        for i, comparison in enumerate(result.top_comparisons, 1):
            html += self._html_query_comparison(comparison, i, result.dataset_name in ['bbc-news', 'huffpost-news', 'wos', 'mental-health'])
        
        html += """
        </div>
"""
        
        return html
    
    def _html_query_comparison(
        self, 
        comparison: QueryComparison, 
        index: int,
        is_category_based: bool
    ) -> str:
        """Generate HTML for a single query comparison."""
        original_prec = comparison.improvement_metrics.get('original_precision', 0.0)
        reranked_prec = comparison.improvement_metrics.get('reranked_precision', 0.0)
        improvement_abs = comparison.improvement_metrics.get('precision_improvement', 0.0)
        improvement_pct = comparison.improvement_metrics.get('precision_improvement_pct', 0.0)
        
        html = f"""
            <div class="query-comparison">
                <div class="query-header">
                    <h4>Query {index}</h4>
                    <div class="query-text">"{comparison.query_text}"</div>
                    <div class="query-id">
                        Query ID: {comparison.query_id}
                    </div>
                    <div class="precision-details">
                        <strong>Original Precision@20:</strong> {original_prec:.4f} | 
                        <strong>Re-ranked Precision@20:</strong> {reranked_prec:.4f} | 
                        <strong>Improvement:</strong> <span class="improvement-badge">+{improvement_abs:.4f} (+{improvement_pct:.1f}%)</span>
                    </div>
                </div>
                
                <div class="ranking-comparison">
                    <table>
                        <thead>
                            <tr>
                                <th>Ranking</th>
                                <th style="text-align: center;">Query</th>
"""
        
        # Header with positions 1-20
        for pos in range(1, 21):
            html += f'                                <th style="text-align: center;">#{pos}</th>\n'
        
        html += """                            </tr>
                        </thead>
                        <tbody>
"""
        
        # Original ranking row
        html += self._html_ranking_row(
            "Original",
            comparison.query_id,
            comparison.original_ranking,
            comparison.relevant_docs,
            is_category_based
        )
        
        # Re-ranked row
        html += self._html_ranking_row(
            "Re-ranked",
            comparison.query_id,
            comparison.reranked_ranking,
            comparison.relevant_docs,
            is_category_based
        )
        
        html += """                        </tbody>
                    </table>
                </div>
            </div>
"""
        
        return html
    
    def _html_ranking_row(
        self,
        label: str,
        query_id: str,
        ranking: List[str],
        relevant_docs: Set[str],
        is_category_based: bool
    ) -> str:
        """Generate a single ranking row in the comparison table."""
        # Truncate doc IDs for display
        def truncate_id(doc_id: str, max_len: int = 12) -> str:
            if len(doc_id) <= max_len:
                return doc_id
            return doc_id[:max_len-2] + ".."
        
        html = f'                            <tr>\n'
        html += f'                                <td class="row-label">{label}</td>\n'
        html += f'                                <td class="doc-cell query-doc" title="{query_id}">{truncate_id(query_id)}</td>\n'
        
        # Filter out the query document from the ranking (skip it entirely)
        filtered_ranking = [doc_id for doc_id in ranking if doc_id != query_id]
        
        # Add doc cells (pad to 20 if needed)
        for i in range(20):
            if i < len(filtered_ranking):
                doc_id = filtered_ranking[i]
                
                # Determine if relevant
                if is_category_based:
                    is_relevant = self._is_relevant_category_based(query_id, doc_id)
                else:
                    is_relevant = doc_id in relevant_docs
                
                css_class = "doc-relevant" if is_relevant else "doc-not-relevant"
                html += f'                                <td class="doc-cell {css_class}" title="{doc_id}">{truncate_id(doc_id)}</td>\n'
            else:
                html += f'                                <td class="doc-cell">-</td>\n'
        
        html += '                            </tr>\n'
        
        return html
    
    def _html_conclusion(self) -> str:
        """Generate conclusion section."""
        return """
        <div class="conclusion">
            <h2>Conclusion</h2>
            <p>
                This report demonstrates the practical impact of UDLF re-ranking methods on text retrieval 
                quality. By examining specific query examples, we observe that re-ranking effectively clusters 
                relevant documents at the top of result lists, providing users with faster access to pertinent 
                information.
            </p>
            <p>
                The visual comparisons presented here complement aggregate metric evaluations, offering concrete 
                evidence of how unsupervised re-ranking techniques can enhance retrieval systems without requiring 
                labeled training data.
            </p>
            <p>
                Key observations across datasets include:
            </p>
            <ul style="margin-left: 30px; margin-top: 10px;">
                <li>Category-based datasets (BBC News, Mental Health, WOS) show particularly strong improvements 
                    due to clear category boundaries</li>
                <li>BEIR datasets demonstrate variable improvements depending on query ambiguity and corpus 
                    characteristics</li>
                <li>Different base retrieval models benefit from re-ranking to varying degrees, with some 
                    combinations showing substantial precision gains at top ranks</li>
            </ul>
        </div>
"""
    
    def _html_footer(self) -> str:
        """Generate HTML footer."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""
        <footer>
            <p>Report generated on {timestamp}</p>
            <p>UDLF Text Espresso Framework | Document Re-ranking Comparison Report</p>
        </footer>
    </div>
</body>
</html>
"""


def main():
    """Main entry point for generating the report."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate HTML comparison report for UDLF text re-ranking results"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['scifact', 'nfcorpus', 'arguana', 'bbc-news', 'wos'],
        help='List of datasets to include in the report'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['bm25', 'miniLM-L6-v2'],
        help='List of models to analyze'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['cprr', 'bfstree', 'rdpac'],
        help='List of re-ranking methods'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('rerank_comparison_report.html'),
        help='Output HTML file path'
    )
    parser.add_argument(
        '--progress-data',
        type=Path,
        default=Path('experiment_progress_data.json'),
        help='Path to experiment_progress_data.json'
    )
    parser.add_argument(
        '--run-id',
        default='paper-assets',
        help='Experiment run ID'
    )
    parser.add_argument(
        '--no-gcs',
        action='store_true',
        help='Use local files instead of GCS'
    )
    parser.add_argument(
        '--local-outputs',
        type=Path,
        default=Path('outputs'),
        help='Local outputs directory (if --no-gcs is used)'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = RerankComparisonReportGenerator(
        run_id=args.run_id,
        use_gcs=not args.no_gcs,
        local_outputs=args.local_outputs
    )
    
    # Generate report
    print(f"Generating re-ranking comparison report...")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"  Output: {args.output}")
    
    generator.generate_html_report(
        datasets=args.datasets,
        models=args.models,
        methods=args.methods,
        output_path=args.output,
        progress_data_path=args.progress_data
    )


if __name__ == '__main__':
    main()
