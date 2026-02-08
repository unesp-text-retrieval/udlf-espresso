# UDLF Text Espresso - Experiment Workflow Guide

## Overview

This document describes the iterative workflow for monitoring and executing the 756 UDLF text re-ranking experiments across multiple datasets, models, methods, and K values.

## Experiment Configuration

### Datasets (7 total)
- arguana
- scifact
- nfcorpus
- scidocs
- fiqa
- wos
- bbc-news

### Models (4 total)
- bm25
- miniLM-L6-v2
- bge-small-en-v1.5
- contriever-msmarco

### Re-ranking Methods (3 total)
- **CPRR** (CPR-based Re-ranking)
- **BFSTREE** (BFS Tree Re-ranking with correlation metric)
- **RDPAC** (RDP-based Re-ranking with additional parameters)

### K Values (9 total)
[1, 3, 5, 10, 20, 30, 40, 50, 75]

### Total Experiments
- Per dataset: 4 models × 3 methods × 9 K values = **108 experiments**
- Overall: 7 datasets × 108 = **756 experiments**

## Prerequisites

### 1. Google Cloud Storage Access
- Bucket: `gs://text-udlf-espresso`
- Base path: `outputs/paper-assets/dataset/`
- Authenticated with `gcloud` CLI

### 2. Python Environment
```bash
conda activate udlf-espresso  # or your environment name
```

### 3. Required Dependencies
- `google-cloud-storage`
- Python 3.11+
- Hydra configuration system

## File Structure

### Input Files (from Retrieval Phase)
```
{dataset}/rerank/input/{model}/ranked-list/data.txt
{dataset}/rerank/input/{model}/lists/data.txt
```

### Baseline Metrics
```
{dataset}/ranks/{model}/metrics.json
```

### Output Files (from Re-ranking Phase)
```
{dataset}/rerank/output/{model}/{method}-k{k_value}/metrics.json
```

## Workflow Process

### Step 1: Monitor Current Progress

Run the monitoring script to check experiment status:

```bash
python monitor_experiment_progress.py
```

**Outputs:**
- `experiment_progress_report.html` - Visual progress report with charts
- `experiment_progress_data.json` - Structured data for automation
- Console summary with statistics

**Three Status Categories:**
1. **Valid** - Experiment completed with non-zero metrics
2. **Invalid** - Experiment has metrics.json but all metrics are 0.0 (needs re-run)
3. **Missing** - No metrics.json file exists yet

### Step 2: Analyze Progress Data

Review the JSON report to identify:

1. **Input File Readiness**
   - Check `input_files_status` section
   - Identifies which dataset/model combinations are ready for re-ranking
   - Lists models needing retrieval phase completion

2. **Missing Experiments**
   - Check `missing_runs_details` section
   - Shows which K values are missing per dataset/model/method

3. **Invalid Experiments**
   - Check `invalid_metrics_details` section
   - Shows experiments with all-zero metrics that need re-run

4. **Baseline Comparison**
   - Check `baseline_comparison` section (if available)
   - Shows improvement delta vs baseline metrics
   - Identifies best performing K value per method

### Step 3: Run Missing/Invalid Experiments

Use the selective experiment runner:

```bash
python run_selective_experiments.py
```

**What it does:**
- Reads `experiment_progress_data.json`
- Filters to dataset/model combinations with input files ready
- Identifies missing and invalid experiments
- Runs only those that need execution
- Can be filtered by method (e.g., only RDPAC)

**Configuration Notes:**
- **RDPAC** requires: `l_mult=2` (plus k_start, k_end, k_inc, p, pl parameters)
- **BFSTREE** requires: `correlation_metric=RBO`
- **CPRR** requires: only `k` parameter

### Step 4: Handle Input File Dependencies

If experiments can't run due to missing input files:

**Identify missing input files:**
```json
"needs_retrieval_step": [
  {
    "model": "bm25",
    "missing_ranked_list": true,
    "missing_lists": true
  }
]
```

**Run retrieval phase for affected models:**
```bash
python -m udlf_text_espresso.runner \
  run_id=paper-assets \
  dataset.name={dataset} \
  index.name={model} \
  pipeline.steps=[retrieve]
```

### Step 5: Iterate

Repeat Steps 1-4 until all experiments are complete:
1. Monitor progress
2. Analyze what's missing/invalid
3. Run necessary experiments
4. Check for input file dependencies
5. Re-monitor to verify

## Progress Tracking

### Current Status Snapshot
As of the last monitoring run, track:
- **Valid experiments**: Completed with valid metrics
- **Invalid experiments**: Need re-run (all-zero metrics)
- **Missing experiments**: Not yet executed
- **Input file gaps**: Models needing retrieval phase

### Expected Completion
- Target: 756/756 experiments (100%)
- Critical path: Input file availability for fiqa and wos datasets

## Metrics Validation

### Valid Metrics Criteria
- `metrics.json` file exists in GCS
- File is valid JSON
- Contains experiment metrics dictionary
- At least one numeric metric > 0.0 (not all zeros)

### Invalid Metrics Detection
- All numeric values are exactly 0.0 or within epsilon (< 1e-10)
- Indicates experiment failure or configuration error
- Must be re-run

### Baseline Comparison Metrics
Key metrics compared against baseline (retrieval-only) metrics. For each metric, the K value showing the **highest improvement (delta)** is selected and displayed:

**MAP (Mean Average Precision):**
- **MAP@20** - Precision averaged over top 20 documents
- **MAP@50** - Precision averaged over top 50 documents
- **MAP@200** - Precision averaged over top 200 documents

**Precision:**
- **Precision@20** - Fraction of relevant documents in top 20
- **Precision@50** - Fraction of relevant documents in top 50
- **Precision@200** - Fraction of relevant documents in top 200

**Recall:**
- **Recall@20** - Fraction of all relevant documents found in top 20
- **Recall@50** - Fraction of all relevant documents found in top 50
- **Recall@200** - Fraction of all relevant documents found in top 200

**Important Notes:**
- Each metric independently shows the K value with the highest improvement
- Different metrics may show different "best K" values
- Improvement delta calculated as: `rerank_metric - baseline_metric`
- Positive delta = improvement, negative delta = degradation

## Common Issues and Solutions

### Issue: "ValueError: UDLF configuration missing required key 'l_mult' for method 'RDPAC'"
**Solution:** Ensure RDPAC experiments include all required parameters:
```python
config_str = f"{{k:{k},l_mult:2,k_start:1,k_end:{k},k_inc:1,p:0.6,pl:0.99}}"
```

### Issue: Experiments running wrong method/K values
**Solution:** Hydra's experiments list in config overrides command-line parameters. Use the Python runner script which overrides the entire experiments list:
```bash
python run_selective_experiments.py
```

### Issue: Input files not found for model
**Solution:** Run retrieval phase first:
```bash
python -m udlf_text_espresso.runner \
  run_id=paper-assets \
  dataset.name={dataset} \
  index.name={model} \
  pipeline.steps=[retrieve]
```

### Issue: All metrics are 0.0 after experiment completes
**Possible causes:**
1. Incorrect UDLF configuration parameters
2. Input file formatting issues
3. Method-specific parameter errors
4. Resource/memory constraints during execution

**Solution:** Check experiment logs, validate input files, and re-run with correct configuration.

## Automation Tips

### Filter by Method
Edit `run_selective_experiments.py` to run only specific methods:
```python
# Filter to only RDPAC experiments
experiments = [(d, m, method, k) for d, m, method, k in experiments if method.lower() == 'rdpac']
```

### Filter by Dataset
```python
# Run only specific datasets
experiments = [(d, m, method, k) for d, m, method, k in experiments if d in ['wos', 'bbc-news']]
```

### Run in Background
For long-running experiments, use screen or tmux:
```bash
screen -S experiments
python run_selective_experiments.py
# Ctrl+A, D to detach
```

## Monitoring Best Practices

1. **Regular Checks**: Run monitor every few hours during active experimentation
2. **Verify Metrics**: Always check that completed experiments have valid (non-zero) metrics
3. **Track Input Files**: Before running experiments, ensure input files are available
4. **Baseline Comparison**: Review improvement deltas to validate re-ranking is beneficial
5. **Log Review**: If metrics are invalid, review experiment logs for errors

## Final Validation

Before considering experiments complete:
1. ✓ All 756 experiments show as "valid" in monitor
2. ✓ No "invalid" or "missing" experiments remain
3. ✓ Baseline comparison shows meaningful improvement deltas
4. ✓ All input files present for all dataset/model combinations
5. ✓ HTML report shows 100% completion

## File Outputs

### Monitor Script Outputs
- `experiment_progress_report.html` - Interactive visual report
- `experiment_progress_data.json` - Structured data for automation

### Experiment Outputs
Each experiment produces:
- `metrics.json` - Evaluation metrics (MAP, NDCG, Recall, MRR, etc.)
- `ranked-list.txt` - Re-ranked document list
- `log.txt` - Execution logs

## Contact and Support

For issues with:
- **Hydra configuration**: Check `conf/rerank/udlf.yaml`
- **Pipeline steps**: Check `conf/pipeline/steps.yaml`
- **Method parameters**: Review method-specific documentation
- **GCS access**: Verify authentication with `gcloud auth list`

---

**Last Updated**: December 25, 2025
