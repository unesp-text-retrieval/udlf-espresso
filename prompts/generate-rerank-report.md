# Generate Re-ranking Comparison Report - Complete Specification

## Context

This prompt describes how to generate an HTML report that visualizes and compares re-ranking improvements in the UDLF Text Expresso project. The report shows before/after ranked lists to demonstrate how unsupervised re-ranking methods improve document retrieval quality.

## Objective

Create a Python module (`src/udlf_text_espresso/reporting/rerank_comparison_report.py`) and CLI script (`generate_rerank_report.py`) that:

1. Reads experiment results from `experiment_progress_data.json`
2. Selects the **single best model+method combination per dataset** based on precision@20 improvement
3. Finds the **top 3 queries** with highest precision improvements for each dataset
4. Generates a self-contained HTML report showing TOP@20 rankings before and after re-ranking
5. Color-codes document relevance (green=relevant, red=non-relevant, blue=query document)
6. Excludes the query document entirely from displayed ranking positions

## Data Structure

### Input: experiment_progress_data.json

```json
{
  "baseline_comparison": {
    "DATASET_NAME": {
      "MODEL_NAME": {
        "METHOD_NAME": {
          "k_VALUE": {
            "metrics": {
              "map@20": 0.1234,
              "precision@20": 0.5678,
              "improvement_pct": 12.34,
              "total_queries": 100
            },
            "query_improvements": [
              {
                "query_id": "business_244",
                "baseline_precision@20": 0.10,
                "reranked_precision@20": 0.80,
                "improvement_pct": 700.0,
                "baseline_ranking": ["doc1", "doc2", "doc3", ...],
                "reranked_ranking": ["doc5", "doc6", "doc1", ...]
              }
            ]
          }
        }
      }
    }
  }
}
```

### Qrels Data Sources

**Location**: `gs://text-udlf-espresso/outputs/paper-assets/dataset/{dataset}/qrels.txt`

**Two formats exist:**

1. **BEIR format** (scifact, nfcorpus, arguana, scidocs, fiqa):
   ```
   query_id doc_id relevance_score
   123 456 1
   123 789 0
   ```

2. **Category-based format** (bbc-news, wos, mental-health, huffpost-news):
   ```
   doc_id\tcategory
   business_244\tbusiness
   politics_380\tpolitics
   ```
   - Documents are relevant if they share the query's category
   - **The query document itself is NOT considered relevant** (excluded from relevance calculation)

### Rankings Data

**Original rankings**: `gs://text-udlf-espresso/outputs/paper-assets/dataset/{dataset}/rerank/input/{model}/ranked-list/data.txt`

**Re-ranked**: 
- Primary: `gs://text-udlf-espresso/outputs/paper-assets/dataset/{dataset}/rerank/output/{model}/{method}-k{k}/data.txt`
- Fallback: `gs://text-udlf-espresso/outputs/paper-assets/dataset/{dataset}/rerank/output/{model}/{method}-k{k}/ranks.tsv`

**Formats**:

1. **Horizontal format (data.txt)** - Space-separated, one line per query:
   ```
   query_id.txt doc1.txt doc2.txt doc3.txt ...
   ```

2. **TREC format (ranks.tsv)** - Tab-separated, one line per query-document pair:
   ```
   query_id\tdoc_id\tscore
   business_244\tbusiness_409\t0.9543
   business_244\tbusiness_111\t0.8921
   ```

**Important**: The report generator should try `data.txt` first, then fall back to `ranks.tsv` if `data.txt` doesn't exist. Both formats contain the same information but in different layouts. The TREC format needs to be grouped by query and sorted by score to reconstruct the ranking.

**Note**: Document IDs have `.txt` extensions in the data files but should be displayed without extensions in the HTML report for cleaner presentation.

## Core Requirements

### 1. Best Combination Selection

For each dataset, select the **single best** model+method+K combination:

```python
def find_best_model_method_combination(data: dict, dataset: str) -> tuple:
    """
    Returns: (best_model, best_method, best_k, best_improvement_pct)
    
    Selection criteria:
    - Use precision@20 improvement_pct as the metric
    - Find the maximum improvement across all model/method/k combinations
    - If tied, prefer lower K values for efficiency
    """
    best_improvement = -float('inf')
    best_combo = None
    
    for model, methods in data['baseline_comparison'][dataset].items():
        for method, k_values in methods.items():
            for k, metrics_data in k_values.items():
                improvement = metrics_data['metrics']['improvement_pct']
                k_num = int(k.replace('k', ''))
                
                if improvement > best_improvement or \
                   (improvement == best_improvement and best_combo and k_num < int(best_combo[2].replace('k', ''))):
                    best_improvement = improvement
                    best_combo = (model, method, k, improvement)
    
    return best_combo
```

### 2. Query Selection

For the selected best combination, pick the **top 3 queries** with highest precision@20 improvements:

```python
def select_top_queries(query_improvements: list, max_queries: int = 3) -> list:
    """
    Returns: List of top improved query data, sorted by improvement_pct descending
    """
    sorted_queries = sorted(
        query_improvements,
        key=lambda x: x['improvement_pct'],
        reverse=True
    )
    return sorted_queries[:max_queries]
```

### 3. Query Document Filtering

**Critical requirement**: The query document must be **completely excluded** from displayed ranking positions.

```python
def _html_ranking_row(ranking: list, query_id: str, qrels: dict, is_category_based: bool, query_category: str = None):
    """
    Filter out the query document before displaying positions #1-#20
    """
    # Remove query document from ranking
    filtered_ranking = [doc_id for doc_id in ranking if doc_id != query_id]
    
    # Display first 20 positions (excluding query)
    for pos, doc_id in enumerate(filtered_ranking[:20], 1):
        # Determine relevance
        if is_category_based:
            doc_category = doc_id.split('_')[0]
            is_relevant = (doc_category == query_category)
        else:
            is_relevant = qrels.get(query_id, {}).get(doc_id, 0) > 0
        
        # CSS class based on relevance
        css_class = "doc-relevant" if is_relevant else "doc-not-relevant"
        
        # Generate HTML cell
        # ...
```

### 4. HTML Structure

**Self-contained HTML with embedded CSS**:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Re-ranking Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; }
        .dataset-section { margin-bottom: 50px; }
        .query-comparison { margin: 20px 0; border: 1px solid #ddd; padding: 15px; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        .position { font-size: 0.75em; font-weight: bold; }
        .doc-id { font-size: 0.75em; }
        .doc-relevant { border: 2px solid #4CAF50; background-color: #e8f5e9; }
        .doc-not-relevant { border: 2px solid #f44336; background-color: #ffebee; }
        .query-doc { border: 2px solid #2196F3; background-color: #e3f2fd; }
        .metrics { background-color: #fff9c4; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Re-ranking Comparison Report</h1>
    <p>This report compares the Top@20 ranked lists before and after applying unsupervised re-ranking methods.</p>
    
    <!-- For each dataset -->
    <div class="dataset-section">
        <h2>Dataset: BBC-NEWS</h2>
        <div class="metrics">
            <strong>Best Configuration:</strong> bm25 + cprr (K=50)<br>
            <strong>Overall Improvement:</strong> +7.68% precision@20
        </div>
        
        <!-- For each of the top 3 queries -->
        <div class="query-comparison">
            <h3>Query ID: business_244</h3>
            <div class="precision-details">
                <strong>Original Precision@20:</strong> 0.1000 | 
                <strong>Re-ranked Precision@20:</strong> 0.8000 | 
                <strong>Improvement:</strong> +0.7000 (+700.0%)
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Position</th>
                        <th>Original Ranking</th>
                        <th>Re-ranked</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="position">#1</td>
                        <td class="doc-id doc-not-relevant">politics_380</td>
                        <td class="doc-id doc-relevant">business_409</td>
                    </tr>
                    <!-- Positions #2-#20 -->
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
```

### 5. Precision Display Format

For each query comparison, display precision metrics clearly:

```
Original Precision@20: 0.1000 | Re-ranked Precision@20: 0.8000 | Improvement: +0.7000 (+700.0%)
```

- Show 4 decimal places for precision values (0.0000 format)
- Display absolute improvement (e.g., +0.7000)
- Display percentage improvement (e.g., +700.0%)
- Format: `Original → Re-ranked | Absolute (Percentage)`

This provides context for large percentage improvements - a +800% improvement is more credible when readers see it's from 0.10 to 0.80.

### 6. Color Coding Rules

- **Green border** (`doc-relevant`): Document is relevant to the query
- **Red border** (`doc-not-relevant`): Document is not relevant to the query
- **Blue border** (`query-doc`): The query document itself (should only appear if not filtered correctly)

For category-based datasets:
- Document is relevant if `doc_category == query_category`
- Query document itself is **never** marked as relevant

## Implementation Structure

### Module: rerank_comparison_report.py

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import json
import gzip

# Optional GCS support
try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

@dataclass
class QueryComparison:
    query_id: str
    baseline_precision: float
    reranked_precision: float
    improvement_pct: float
    baseline_ranking: List[str]
    reranked_ranking: List[str]

class RerankComparisonReportGenerator:
    def __init__(self, data_file: str, use_gcs: bool = True):
        self.data_file = data_file
        self.use_gcs = use_gcs and HAS_GCS
        self.gcs_client = storage.Client() if self.use_gcs else None
        
    def load_qrels(self, dataset: str) -> Tuple[Dict, bool, Optional[Dict]]:
        """
        Load qrels and detect format.
        
        Returns:
            - qrels_dict: {query_id: {doc_id: relevance_score}} for BEIR
                         OR {doc_id: category} for category-based
            - is_category_based: boolean flag
            - category_counts: {category: count} for category-based, None otherwise
        """
        # Implementation details...
        
    def find_best_model_method_combination(self, data: dict, dataset: str) -> Tuple:
        """Find single best model+method+k combination per dataset."""
        # Implementation details...
        
    def select_top_queries(self, query_improvements: list, max_queries: int = 3) -> List:
        """Select top N queries by improvement."""
        # Implementation details...
        
    def generate_report(self, output_path: str, datasets: Optional[List[str]] = None):
        """Generate complete HTML report."""
        # Implementation details...
```

### CLI Script: generate_rerank_report.py

```python
#!/usr/bin/env python3
"""
Generate Re-ranking Comparison Report

Usage:
    python generate_rerank_report.py
    python generate_rerank_report.py --output custom_report.html
    python generate_rerank_report.py --datasets BBC-NEWS WOS
"""

import argparse
from udlf_text_espresso.reporting.rerank_comparison_report import RerankComparisonReportGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate re-ranking comparison report')
    parser.add_argument('--data-file', default='experiment_progress_data.json',
                       help='Path to experiment progress data JSON')
    parser.add_argument('--output', default='rerank_comparison_report.html',
                       help='Output HTML file path')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to include (default: all)')
    parser.add_argument('--use-local', action='store_true',
                       help='Use local data instead of GCS')
    
    args = parser.parse_args()
    
    generator = RerankComparisonReportGenerator(
        data_file=args.data_file,
        use_gcs=not args.use_local
    )
    
    generator.generate_report(
        output_path=args.output,
        datasets=args.datasets
    )
    
    print(f"✅ Report generated: {args.output}")

if __name__ == '__main__':
    main()
```

## Edge Cases and Refinements

### 1. Dataset Name Normalization

The JSON uses uppercase dataset names, but GCS paths use lowercase:

```python
dataset_path = dataset.lower()
gcs_path = f"gs://text-udlf-espresso/outputs/paper-assets/dataset/{dataset_path}/..."
```

### 2. File Extension Handling

Rankings in GCS have `.txt` extensions, but display should not:

```python
# When reading from GCS
query_id_with_ext = "business_244.txt"
query_id = query_id_with_ext.replace('.txt', '')

# When displaying
display_id = doc_id.replace('.txt', '')
```

### 3. Category Extraction for Category-based Datasets

```python
def extract_category(doc_id: str) -> str:
    """Extract category from doc_id like 'business_244' -> 'business'"""
    return doc_id.split('_')[0]
```

### 4. Mental Health Dataset Special Case

Mental Health dataset IDs may not have `.txt` extensions in GCS data:

```python
# Handle both formats
if dataset.lower() == 'mental-health':
    # Check if line starts with or without .txt
    query_match = query_id if query_id in line else f"{query_id}.txt"
```

### 5. Fallback to ranks.tsv When data.txt Missing

Some model/method combinations may only have `ranks.tsv` (TREC format) instead of `data.txt` (horizontal format). The report generator must handle both:

```python
def load_reranked_results(dataset, model, method, k_value):
    data_path = f"{base_path}/{dataset}/rerank/output/{model}/{method}-k{k_value}/data.txt"
    ranks_path = f"{base_path}/{dataset}/rerank/output/{model}/{method}-k{k_value}/ranks.tsv"
    
    # Try data.txt first
    if file_exists(data_path):
        # Parse horizontal format: query_id doc1 doc2 doc3 ...
        return parse_horizontal_format(data_path)
    
    # Fall back to ranks.tsv
    if file_exists(ranks_path):
        # Parse TREC format: query_id\tdoc_id\tscore
        # Group by query, sort by score descending, take TOP@20
        query_docs = defaultdict(list)
        for line in read_file(ranks_path):
            qid, doc_id, score = line.split('\t')
            query_docs[qid].append((doc_id, float(score)))
        
        rankings = {}
        for qid, docs in query_docs.items():
            docs.sort(key=lambda x: x[1], reverse=True)
            rankings[qid] = [doc_id for doc_id, _ in docs[:20]]
        return rankings
    
    return {}  # No data available
```

**Why this matters**: Some experiments (especially with newer models like contriever-msmarco) may only generate `ranks.tsv` without `data.txt`. Without this fallback, those experiments would be missing from the report even though they have valid results.

## Validation

After generating the report, validate rankings with this script:

```bash
#!/bin/bash
# Quick validation: check if displayed rankings match GCS data

# Extract query ID from HTML
QUERY_ID="business_244"
DATASET="bbc-news"
MODEL="bm25"
METHOD="cprr"
K="50"

# Get original ranking
echo "Original TOP@10:"
gsutil cat gs://text-udlf-espresso/outputs/paper-assets/dataset/$DATASET/rerank/input/$MODEL/ranked-list/data.txt | \
  awk -v qid="${QUERY_ID}.txt" '$1 == qid {for(i=2; i<=11; i++) print "  #"i-1": "$i}'

# Get re-ranked
echo "Re-ranked TOP@10:"
gsutil cat gs://text-udlf-espresso/outputs/paper-assets/dataset/$DATASET/rerank/output/$MODEL/$METHOD-k$K/data.txt | \
  awk -v qid="${QUERY_ID}.txt" '$1 == qid {for(i=2; i<=11; i++) print "  #"i-1": "$i}'
```

## Expected Output Example

**Final Report Characteristics:**
- File size: ~100KB for 3 datasets × 3 queries
- Shows dramatic improvements with actual values: Precision 0.10 → 0.80 (+0.70, +700.0%)
- Displays both absolute and percentage improvements for credibility
- Clear visual distinction between relevant (green) and non-relevant (red) documents
- Self-contained: no external CSS/JS dependencies
- Professional formatting suitable for academic papers

**Sample Dataset Results:**
- BBC-NEWS: +7.68% improvement (bm25 + cprr, K=50)
- WOS: +3.56% improvement (miniLM-L6-v2 + cprr, K=75)
- MENTAL-HEALTH: +12.83% improvement (bm25 + cprr, K=75)

## Common Pitfalls to Avoid

1. ❌ **Don't show multiple configurations** - only the single best per dataset
2. ❌ **Don't include query document in positions** - filter it completely
3. ❌ **Don't mark query as relevant** in category-based datasets
4. ❌ **Don't forget .txt extension** when querying GCS data
5. ❌ **Don't use MAP@20** - use precision@20 for selection
6. ❌ **Don't show more than 3 queries** per dataset - keeps report focused
7. ❌ **Don't use diagonal text** - normal horizontal text is more readable

## Success Criteria

✅ Report shows exactly 3 query examples per dataset  
✅ Each dataset shows only one model+method+K combination (the best)  
✅ Query document does not appear in ranking positions #1-#20  
✅ Green borders on relevant documents, red on non-relevant  
✅ Rankings match actual GCS data when validated  
✅ HTML is self-contained with embedded CSS  
✅ File size is reasonable (~100KB for typical report)  
✅ Precision improvements are clearly visible in the data  

## Additional Notes

- The report is designed for academic research presentation
- Focuses on **unsupervised** re-ranking methods (CPRR, BFSTREE, RDPAC)
- Part of UDLF Text Expresso framework for document list re-ranking
- Original UDLF framework was designed for image re-ranking at UNESP
- This extends UDLF to text document retrieval research
