#!/usr/bin/env python3
"""
Monitor progress of UDLF Text Espresso experiments in Google Cloud Storage.

This script checks for the existence of metrics.json files in GCS and generates
an HTML progress report showing completed runs and percentage remaining.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from google.cloud import storage
from collections import defaultdict


# Experiment configuration
DATASETS = ["arguana", "scifact", "nfcorpus", "scidocs", "fiqa", "wos", "bbc-news"]
MODELS = ["bm25", "miniLM-L6-v2", "bge-small-en-v1.5", "contriever-msmarco"]
METHODS = ["cprr", "bfstree", "rdpac"]
K_VALUES = [1, 3, 5, 10, 20, 30, 40, 50, 75]

# GCS configuration
BUCKET_NAME = "text-udlf-espresso"
BASE_PATH = "outputs/paper-assets/dataset"

# Expected runs per dataset and total
RUNS_PER_DATASET = len(MODELS) * len(METHODS) * len(K_VALUES)  # 4 * 3 * 9 = 108
TOTAL_EXPECTED_RUNS = len(DATASETS) * RUNS_PER_DATASET  # 7 * 108 = 756


def check_file_exists(bucket, blob_path: str) -> bool:
    """Check if a file exists in GCS bucket or locally."""
    # First check local file
    local_path = blob_path
    if os.path.exists(local_path):
        return True
    
    # Then check GCS
    blob = bucket.blob(blob_path)
    return blob.exists()


def load_metrics_from_gcs(bucket, blob_path: str) -> Dict:
    """Load metrics.json from GCS or local filesystem and return the data."""
    # First try local file
    local_path = blob_path
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load local metrics at {local_path}: {e}")
    
    # Then try GCS
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None
    
    try:
        content = blob.download_as_text()
        return json.loads(content)
    except Exception as e:
        print(f"  Warning: Could not load metrics at {blob_path}: {e}")
        return None


def check_metrics_valid(bucket, blob_path: str) -> bool:
    """Check if metrics.json exists and contains non-zero values (local or GCS)."""
    # First try local file
    local_path = blob_path
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r') as f:
                metrics_data = json.load(f)
            
            # Validate structure
            if not metrics_data or not isinstance(metrics_data, dict):
                return False
            
            # Get the first experiment's metrics (usually only one)
            experiment_metrics = list(metrics_data.values())[0]
            
            # Validate metrics dictionary
            if not experiment_metrics or not isinstance(experiment_metrics, dict):
                return False
            
            # Check if all numeric metrics are zero
            numeric_values = [v for v in experiment_metrics.values() if isinstance(v, (int, float))]
            
            if not numeric_values:
                return False
            
            all_zero = all(abs(value) < 1e-10 for value in numeric_values)
            return not all_zero
        except Exception as e:
            print(f"  Warning: Could not validate local metrics at {local_path}: {e}")
            # Fall through to try GCS
    
    # Then try GCS
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return False
    
    try:
        # Download and parse the metrics file
        content = blob.download_as_text()
        metrics_data = json.loads(content)
        
        # Validate structure
        if not metrics_data or not isinstance(metrics_data, dict):
            return False
        
        # Get the first experiment's metrics (usually only one)
        experiment_metrics = list(metrics_data.values())[0]
        
        # Validate metrics dictionary
        if not experiment_metrics or not isinstance(experiment_metrics, dict):
            return False
        
        # Check if all numeric metrics are zero
        # Filter to only numeric values to avoid issues with non-numeric fields
        numeric_values = [v for v in experiment_metrics.values() if isinstance(v, (int, float))]
        
        if not numeric_values:
            return False
        
        # Check if all metrics are zero (using small epsilon for floating point comparison)
        all_zero = all(abs(value) < 1e-10 for value in numeric_values)
        
        return not all_zero  # Valid if NOT all zeros
    except Exception as e:
        print(f"  Warning: Could not validate metrics at {blob_path}: {e}")
        return False  # Treat as invalid if we can't read it


def generate_all_paths() -> List[Dict[str, str]]:
    """Generate all expected file paths for the experiments."""
    paths = []
    for dataset in DATASETS:
        for model in MODELS:
            for method in METHODS:
                for k_value in K_VALUES:
                    # Format: <dataset>/rerank/output/<model>/<method>-k<k_value>/metrics.json
                    path = f"{BASE_PATH}/{dataset}/rerank/output/{model}/{method}-k{k_value}/metrics.json"
                    paths.append({
                        "path": path,
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "k_value": k_value
                    })
    return paths


def check_progress(bucket) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Check which experiments have been completed with valid metrics."""
    all_paths = generate_all_paths()
    completed = []
    missing = []
    invalid_metrics = []
    
    print(f"\nChecking {len(all_paths)} experiment runs (validating metrics)...")
    for i, path_info in enumerate(all_paths, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(all_paths)} checked...")
        
        # Check if metrics are valid (file exists AND non-zero metrics)
        if check_metrics_valid(bucket, path_info["path"]):
            completed.append(path_info)
        elif check_file_exists(bucket, path_info["path"]):
            # File exists but has all zero metrics or couldn't be validated
            path_info_copy = path_info.copy()
            path_info_copy["issue"] = "all_zero_metrics"
            invalid_metrics.append(path_info_copy)
        else:
            # File doesn't exist yet
            missing.append(path_info)
    
    print(f"✓ Check complete: {len(completed)} valid, {len(invalid_metrics)} invalid (zero metrics), {len(missing)} missing")
    return completed, missing, invalid_metrics
    
    print(f"✓ Check complete: {len(completed)} valid, {len(invalid_metrics)} invalid (zero metrics), {len(missing)} missing")
    return completed, missing, invalid_metrics


def load_baseline_metrics(bucket) -> Dict:
    """Load baseline metrics from ranks/{model}/metrics.json for each dataset/model."""
    print("\nLoading baseline metrics from retrieval phase...")
    baseline_metrics = {}
    
    for dataset in DATASETS:
        baseline_metrics[dataset] = {}
        for model in MODELS:
            baseline_path = f"{BASE_PATH}/{dataset}/ranks/{model}/metrics.json"
            metrics = load_metrics_from_gcs(bucket, baseline_path)
            
            if metrics:
                # Extract first experiment's metrics (usually only one)
                exp_metrics = list(metrics.values())[0] if metrics else {}
                baseline_metrics[dataset][model] = exp_metrics
            else:
                baseline_metrics[dataset][model] = None
    
    print("✓ Baseline metrics loaded")
    return baseline_metrics


def check_input_files(bucket) -> Dict:
    """Check for input files (ranked-list/data.txt and lists/data.txt) for each dataset/model."""
    print("\nChecking input files (ranked-list and lists for retrieval phase)...")
    input_status = {}
    
    for dataset in DATASETS:
        input_status[dataset] = {}
        for model in MODELS:
            # Correct path structure: dataset/rerank/input/model/ranked-list/data.txt
            ranked_list_path = f"{BASE_PATH}/{dataset}/rerank/input/{model}/ranked-list/data.txt"
            lists_path = f"{BASE_PATH}/{dataset}/rerank/input/{model}/lists/data.txt"
            
            has_ranked_list = check_file_exists(bucket, ranked_list_path)
            has_lists = check_file_exists(bucket, lists_path)
            
            input_status[dataset][model] = {
                "ranked_list": has_ranked_list,
                "lists": has_lists,
                "ready_for_rerank": has_ranked_list and has_lists
            }
    
    print("✓ Input files check complete")
    return input_status


def compute_baseline_comparison(bucket, completed: List[Dict], baseline_metrics: Dict) -> Dict:
    """Compare rerank metrics against baseline for each completed experiment."""
    print("\nComputing baseline comparisons...")
    comparisons = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    # Define key metrics with their possible name variations
    key_metrics = {
        "map@20": ["MAP@20", "map@20"],
        "map@50": ["MAP@50", "map@50"],
        "map@200": ["MAP@200", "map@200"],
        "precision@20": ["Precision@20", "precision@20"],
        "precision@50": ["Precision@50", "precision@50"],
        "precision@200": ["Precision@200", "precision@200"],
        "recall@20": ["Recall@20", "recall@20"],
        "recall@50": ["Recall@50", "recall@50"],
        "recall@200": ["Recall@200", "recall@200"]
    }
    
    def find_metric_value(metrics_dict, metric_variations):
        """Find metric value from dict using any of the possible key variations."""
        for key in metric_variations:
            if key in metrics_dict:
                return metrics_dict[key]
        return None
    
    for exp in completed:
        dataset = exp["dataset"]
        model = exp["model"]
        method = exp["method"]
        k_value = exp["k_value"]
        
        # Load rerank metrics
        rerank_path = f"{BASE_PATH}/{dataset}/rerank/output/{model}/{method}-k{k_value}/metrics.json"
        rerank_metrics = load_metrics_from_gcs(bucket, rerank_path)
        
        if not rerank_metrics or not baseline_metrics.get(dataset, {}).get(model):
            continue
        
        baseline = baseline_metrics[dataset][model]
        rerank_exp = list(rerank_metrics.values())[0] if rerank_metrics else {}
        
        # Compute deltas for key metrics
        deltas = {}
        for metric_name, variations in key_metrics.items():
            baseline_val = find_metric_value(baseline, variations)
            rerank_val = find_metric_value(rerank_exp, variations)
            
            if baseline_val is not None and rerank_val is not None:
                if isinstance(baseline_val, (int, float)) and isinstance(rerank_val, (int, float)):
                    deltas[metric_name] = {
                        "baseline": baseline_val,
                        "rerank": rerank_val,
                        "delta": rerank_val - baseline_val,
                        "improvement_pct": ((rerank_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                    }
        
        comparisons[dataset][model][method][k_value] = deltas
    
    print("✓ Baseline comparison complete")
    return dict(comparisons)


def calculate_statistics(completed: List[Dict], missing: List[Dict], invalid_metrics: List[Dict], input_status: Dict) -> Dict:
    """Calculate progress statistics overall and per dataset."""
    total_incomplete = len(missing) + len(invalid_metrics)
    
    stats = {
        "overall": {
            "completed": len(completed),
            "missing": len(missing),
            "invalid_metrics": len(invalid_metrics),
            "total_incomplete": total_incomplete,
            "total": TOTAL_EXPECTED_RUNS,
            "percentage_complete": (len(completed) / TOTAL_EXPECTED_RUNS) * 100,
            "percentage_remaining": (total_incomplete / TOTAL_EXPECTED_RUNS) * 100
        },
        "by_dataset": {},
        "input_files": input_status
    }
    
    # Calculate per-dataset statistics
    for dataset in DATASETS:
        dataset_completed = [p for p in completed if p["dataset"] == dataset]
        dataset_missing = [p for p in missing if p["dataset"] == dataset]
        dataset_invalid = [p for p in invalid_metrics if p["dataset"] == dataset]
        dataset_incomplete = len(dataset_missing) + len(dataset_invalid)
        
        # Organize by model/method/k for granular view
        k_details = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"status": "missing", "valid": False})))
        
        for exp in dataset_completed:
            k_details[exp["model"]][exp["method"]][exp["k_value"]] = {"status": "completed", "valid": True}
        
        for exp in dataset_invalid:
            k_details[exp["model"]][exp["method"]][exp["k_value"]] = {"status": "invalid", "valid": False}
        
        for exp in dataset_missing:
            k_details[exp["model"]][exp["method"]][exp["k_value"]] = {"status": "missing", "valid": False}
        
        stats["by_dataset"][dataset] = {
            "completed": len(dataset_completed),
            "missing": len(dataset_missing),
            "invalid_metrics": len(dataset_invalid),
            "total_incomplete": dataset_incomplete,
            "total": RUNS_PER_DATASET,
            "percentage_complete": (len(dataset_completed) / RUNS_PER_DATASET) * 100,
            "percentage_remaining": (dataset_incomplete / RUNS_PER_DATASET) * 100,
            "input_ready": input_status[dataset],
            "k_value_details": dict(k_details)
        }
    
    return stats


def generate_html_report(stats: Dict, completed: List[Dict], missing: List[Dict], invalid_metrics: List[Dict], baseline_comparison: Dict = None, output_file: str = "experiment_progress_report.html"):
    """Generate an HTML progress report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UDLF Text Espresso - Experiment Progress Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header .timestamp {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background-color: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}
        .stat-card.completed .value {{ color: #4CAF50; }}
        .stat-card.missing .value {{ color: #f44336; }}
        .stat-card.invalid .value {{ color: #ff9800; }}
        table {{
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge.ready {{ background-color: #e8f5e9; color: #2e7d32; }}
        .badge.not-ready {{ background-color: #ffebee; color: #c62828; }}
        .badge.complete {{ background-color: #e8f5e9; color: #2e7d32; }}
        .badge.incomplete {{ background-color: #fff3e0; color: #ef6c00; }}
        .warning-box {{
            background-color: #fff3cd;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .warning-box h3 {{
            margin-top: 0;
            color: #f57c00;
        }}
        .k-status {{
            display: inline-block;
            width: 20px;
            height: 20px;
            text-align: center;
            line-height: 20px;
            border-radius: 3px;
            font-size: 0.75em;
            font-weight: bold;
            margin: 2px;
        }}
        .k-status.completed {{ background-color: #4caf50; color: white; }}
        .k-status.invalid {{ background-color: #ff9800; color: white; }}
        .k-status.missing {{ background-color: #e0e0e0; color: #666; }}
        .improvement {{ color: #4caf50; font-weight: bold; }}
        .degradation {{ color: #f44336; font-weight: bold; }}
        .neutral {{ color: #666; }}
        .dataset-section {{
            margin: 30px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .dataset-header {{
            background: #667eea;
            color: white;
            padding: 15px;
            font-size: 1.3em;
            font-weight: bold;
        }}
        .dataset-body {{
            padding: 20px;
            background: white;
        }}
        .model-table {{
            width: 100%;
            margin: 10px 0;
            border-collapse: collapse;
        }}
        .model-table th {{
            background: #f5f5f5;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e0e0e0;
        }}
        .model-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}
        .collapsible:hover {{
            opacity: 0.8;
        }}
        .collapsible-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        .collapsible-content.active {{
            max-height: 2000px;
        }}
    </style>
    <script>
        function toggleCollapsible(id) {{
            var content = document.getElementById(id);
            content.classList.toggle('active');
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>UDLF Text Espresso - Experiment Progress Report</h1>
        <div class="timestamp">Generated: {timestamp}</div>
    </div>

    <div class="stats-grid">
        <div class="stat-card completed">
            <div class="label">Completed (Valid)</div>
            <div class="value">{stats['overall']['completed']}</div>
        </div>
        <div class="stat-card invalid">
            <div class="label">Invalid (Zero Metrics)</div>
            <div class="value">{stats['overall']['invalid_metrics']}</div>
        </div>
        <div class="stat-card missing">
            <div class="label">Missing</div>
            <div class="value">{stats['overall']['missing']}</div>
        </div>
        <div class="stat-card">
            <div class="label">Total Expected</div>
            <div class="value">{stats['overall']['total']}</div>
        </div>
    </div>

    <div class="progress-bar">
        <div class="progress-fill" style="width: {stats['overall']['percentage_complete']:.1f}%">
            {stats['overall']['percentage_complete']:.1f}% Complete
        </div>
    </div>

    {generate_invalid_metrics_section(invalid_metrics) if invalid_metrics else ''}
    
    {generate_baseline_comparison_section(baseline_comparison) if baseline_comparison else ''}

    <div class="section">
        <h2>Progress by Dataset</h2>
        <table>
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Valid</th>
                    <th>Invalid</th>
                    <th>Missing</th>
                    <th>Total</th>
                    <th>% Complete</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for dataset in DATASETS:
        dataset_stats = stats["by_dataset"][dataset]
        
        html_content += f"""
                <tr>
                    <td><strong>{dataset}</strong></td>
                    <td>{dataset_stats['completed']}</td>
                    <td>{dataset_stats['invalid_metrics']}</td>
                    <td>{dataset_stats['missing']}</td>
                    <td>{dataset_stats['total']}</td>
                    <td>{dataset_stats['percentage_complete']:.1f}%</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
    
    <!-- Detailed K-value breakdown by dataset -->
    <div class="section">
        <h2>Detailed K-Value Status by Dataset</h2>
        <p>Click on each dataset to expand and see granular K-value completion status for each model and method.</p>
"""
    
    # Generate detailed K-value sections for each dataset
    for dataset in DATASETS:
        dataset_stats = stats["by_dataset"][dataset]
        k_details = dataset_stats.get("k_value_details", {})
        
        html_content += f"""
        <div class="dataset-section">
            <div class="dataset-header collapsible" onclick="toggleCollapsible('dataset-{dataset}')">
                {dataset.upper()} - {dataset_stats['percentage_complete']:.1f}% Complete ({dataset_stats['completed']}/{dataset_stats['total']})
            </div>
            <div id="dataset-{dataset}" class="collapsible-content">
                <div class="dataset-body">
"""
        
        for model in MODELS:
            if model not in k_details:
                continue
                
            html_content += f"""
                    <h4>{model}</h4>
                    <table class="model-table">
                        <thead>
                            <tr>
                                <th>Method</th>
"""
            
            for k in K_VALUES:
                html_content += f"                                <th>K={k}</th>\n"
            
            html_content += """
                            </tr>
                        </thead>
                        <tbody>
"""
            
            for method in METHODS:
                html_content += f"""
                            <tr>
                                <td><strong>{method.upper()}</strong></td>
"""
                
                for k in K_VALUES:
                    status = k_details.get(model, {}).get(method, {}).get(k, {}).get("status", "missing")
                    html_content += f"""                                <td><span class="k-status {status}">{k}</span></td>\n"""
                
                html_content += """
                            </tr>
"""
            
            html_content += """
                        </tbody>
                    </table>
"""
        
        html_content += """
                </div>
            </div>
        </div>
"""
    
    html_content += """
    </div>

</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✓ HTML report generated: {output_file}")


def generate_invalid_metrics_section(invalid_metrics: List[Dict]) -> str:
    """Generate HTML section for invalid (zero) metrics."""
    if not invalid_metrics:
        return ""
    
    # Group by dataset and model
    by_dataset_model = defaultdict(lambda: defaultdict(list))
    for item in invalid_metrics:
        by_dataset_model[item['dataset']][item['model']].append(item)
    
    html = """
    <div class="warning-box">
        <h3>⚠️ Invalid Metrics (All Zeros)</h3>
        <p>These experiments have metrics.json files but all values are 0.0. They need to be re-run:</p>
        <table>
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Model</th>
                    <th>Method</th>
                    <th>K Values</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for dataset in sorted(by_dataset_model.keys()):
        for model in sorted(by_dataset_model[dataset].keys()):
            items = by_dataset_model[dataset][model]
            # Group by method
            by_method = defaultdict(list)
            for item in items:
                by_method[item['method']].append(item['k_value'])
            
            for method, k_values in sorted(by_method.items()):
                k_values_str = ", ".join(map(str, sorted(k_values)))
                html += f"""
                <tr>
                    <td><strong>{dataset}</strong></td>
                    <td>{model}</td>
                    <td>{method}</td>
                    <td>{k_values_str}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
"""
    return html


def generate_baseline_comparison_section(baseline_comparison: Dict) -> str:
    """Generate HTML section showing improvement vs baseline."""
    if not baseline_comparison:
        return ""
    
    html = """
    <div class="section">
        <h2>📊 Baseline Comparison - Improvement Analysis</h2>
        <p>Comparing re-ranking results against baseline (retrieval-only) metrics. Each cell shows the K value with the highest improvement for that specific metric.</p>
"""
    
    metric_groups = [
        ('MAP', ['map@20', 'map@50', 'map@200']),
        ('Precision', ['precision@20', 'precision@50', 'precision@200']),
        ('Recall', ['recall@20', 'recall@50', 'recall@200'])
    ]
    
    for dataset in DATASETS:
        if dataset not in baseline_comparison:
            continue
        
        html += f"""
        <h3>{dataset.upper()}</h3>
        <table>
            <thead>
                <tr>
                    <th rowspan="2">Model</th>
                    <th rowspan="2">Method</th>
"""
        
        for group_name, _ in metric_groups:
            html += f"                    <th colspan=\"3\">{group_name}</th>\n"
        
        html += """
                </tr>
                <tr>
"""
        
        for _, metrics in metric_groups:
            for metric in metrics:
                cutoff = metric.split('@')[1]
                html += f"                    <th>@{cutoff}</th>\n"
        
        html += """
                </tr>
            </thead>
            <tbody>
"""
        
        for model in MODELS:
            if model not in baseline_comparison[dataset]:
                continue
            
            for method in METHODS:
                if method not in baseline_comparison[dataset][model]:
                    continue
                
                method_data = baseline_comparison[dataset][model][method]
                
                html += f"""
                <tr>
                    <td>{model}</td>
                    <td><strong>{method.upper()}</strong></td>
"""
                
                # For each metric, find K with highest improvement (delta)
                for _, metrics in metric_groups:
                    for metric in metrics:
                        best_k = None
                        best_delta = -float('inf')
                        
                        # Find K with highest improvement for this specific metric
                        for k, k_metrics in method_data.items():
                            if metric in k_metrics:
                                delta = k_metrics[metric].get('delta', -float('inf'))
                                if delta > best_delta:
                                    best_delta = delta
                                    best_k = k
                        
                        if best_k is not None and metric in method_data[best_k]:
                            m = method_data[best_k][metric]
                            delta = m.get('delta', 0)
                            improvement_pct = m.get('improvement_pct', 0)
                            rerank_val = m.get('rerank', 0)
                            
                            delta_class = 'improvement' if delta > 0 else ('degradation' if delta < 0 else 'neutral')
                            sign = '+' if delta > 0 else ''
                            
                            html += f"""
                    <td>
                        <div style="font-weight: bold;">{rerank_val:.4f}</div>
                        <div class="{delta_class}" style="font-size: 0.8em;">{sign}{delta:.4f} ({sign}{improvement_pct:.1f}%)</div>
                        <div style="font-size: 0.7em; color: #666;">K={best_k}</div>
                    </td>
"""
                        else:
                            html += """
                    <td>-</td>
"""
                
                html += """
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
    
    html += """
    </div>
"""
    return html


def export_json_report(stats: Dict, completed: List[Dict], missing: List[Dict], invalid_metrics: List[Dict], baseline_comparison: Dict = None, output_file: str = "experiment_progress_data.json"):
    """Export detailed progress data as JSON for programmatic use."""
    
    # Organize completed runs by dataset/model/method
    completed_by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in completed:
        completed_by_dataset[run["dataset"]][run["model"]][run["method"]].append(run["k_value"])
    
    # Organize missing runs by dataset/model/method
    missing_by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in missing:
        missing_by_dataset[run["dataset"]][run["model"]][run["method"]].append(run["k_value"])
    
    # Organize invalid runs by dataset/model/method
    invalid_by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in invalid_metrics:
        invalid_by_dataset[run["dataset"]][run["model"]][run["method"]].append(run["k_value"])
    
    # Build comprehensive JSON structure
    data = {
        "generated_at": datetime.now().isoformat(),
        "configuration": {
            "datasets": DATASETS,
            "models": MODELS,
            "methods": METHODS,
            "k_values": K_VALUES,
            "bucket": BUCKET_NAME,
            "base_path": BASE_PATH
        },
        "summary": {
            "total_expected_runs": TOTAL_EXPECTED_RUNS,
            "completed_runs": len(completed),
            "missing_runs": len(missing),
            "invalid_metrics_runs": len(invalid_metrics),
            "percentage_complete": stats["overall"]["percentage_complete"],
            "percentage_remaining": stats["overall"]["percentage_remaining"]
        },
        "input_files_status": stats["input_files"],
        "baseline_comparison": baseline_comparison or {},
        "datasets": {}
    }
    
    # Per-dataset detailed information
    for dataset in DATASETS:
        dataset_info = {
            "completed": stats["by_dataset"][dataset]["completed"],
            "missing": stats["by_dataset"][dataset]["missing"],
            "invalid_metrics": stats["by_dataset"][dataset]["invalid_metrics"],
            "total": stats["by_dataset"][dataset]["total"],
            "percentage_complete": stats["by_dataset"][dataset]["percentage_complete"],
            "percentage_remaining": stats["by_dataset"][dataset]["percentage_remaining"],
            "input_files": stats["by_dataset"][dataset]["input_ready"],
            "needs_retrieval_step": []
        }
        
        # Identify models that need retrieval step (missing input files)
        for model in MODELS:
            model_input = stats["input_files"][dataset][model]
            if not model_input["ready_for_rerank"]:
                dataset_info["needs_retrieval_step"].append({
                    "model": model,
                    "missing_ranked_list": not model_input["ranked_list"],
                    "missing_lists": not model_input["lists"]
                })
        
        data["datasets"][dataset] = dataset_info
    
    # Add detailed breakdown by dataset/model/method
    data["missing_runs_details"] = dict(missing_by_dataset)
    data["completed_runs_details"] = dict(completed_by_dataset)
    data["invalid_metrics_details"] = dict(invalid_by_dataset)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ JSON report generated: {output_file}")


def print_console_summary(stats: Dict, invalid_metrics: List[Dict]):
    """Print a summary to the console."""
    print("\n" + "="*70)
    print("EXPERIMENT PROGRESS SUMMARY")
    print("="*70)
    print(f"\nOverall Progress:")
    print(f"  ✓ Completed (Valid):      {stats['overall']['completed']}/{stats['overall']['total']} ({stats['overall']['percentage_complete']:.1f}%)")
    print(f"  ⚠ Invalid (Zero Metrics): {stats['overall']['invalid_metrics']}/{stats['overall']['total']}")
    print(f"  ✗ Missing:                {stats['overall']['missing']}/{stats['overall']['total']}")
    print(f"  📊 Remaining Work:        {stats['overall']['total_incomplete']}/{stats['overall']['total']} ({stats['overall']['percentage_remaining']:.1f}%)")
    
    if invalid_metrics:
        print(f"\n⚠️  WARNING: {len(invalid_metrics)} experiments have all-zero metrics and need re-run")
        print(f"   Check the HTML report or JSON file for details")
    
    print(f"\nProgress by Dataset:")
    for dataset in DATASETS:
        ds_stats = stats["by_dataset"][dataset]
        print(f"  {dataset:12s}: {ds_stats['completed']:3d} valid, {ds_stats['invalid_metrics']:3d} invalid, {ds_stats['missing']:3d} missing ({ds_stats['percentage_complete']:5.1f}% complete)")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    print("UDLF Text Espresso - Experiment Progress Monitor")
    print("=" * 70)
    
    # Initialize GCS client
    print(f"\nConnecting to GCS bucket: {BUCKET_NAME}")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        print("✓ Connected to GCS")
    except Exception as e:
        print(f"✗ Error connecting to GCS: {e}")
        return
    
    # Check progress and validate metrics
    completed, missing, invalid_metrics = check_progress(bucket)
    
    # Check input files (retrieval phase completion)
    input_status = check_input_files(bucket)
    
    # Load baseline metrics
    baseline_metrics = load_baseline_metrics(bucket)
    
    # Compute baseline comparison
    baseline_comparison = compute_baseline_comparison(bucket, completed, baseline_metrics)
    
    # Calculate statistics
    stats = calculate_statistics(completed, missing, invalid_metrics, input_status)
    
    # Print console summary
    print_console_summary(stats, invalid_metrics)
    
    # Generate reports
    generate_html_report(stats, completed, missing, invalid_metrics, baseline_comparison)
    export_json_report(stats, completed, missing, invalid_metrics, baseline_comparison)
    
    print("\n✓ All reports generated successfully!")


if __name__ == "__main__":
    main()
