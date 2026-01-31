# UDLF Reranking Pipeline: Format Conversion Flow

## Overview
The UDLF reranking pipeline converts between TREC format (used for retrieval results) and UDLF's ranked-list format, then converts the reranked output back to TREC format for metrics calculation.

## Data Flow

### 1. Input: TREC Format (retrieval.tsv)
Let's suppose our system retrieves only 3 documents per query, in the TREC format (vertical), the first column of the file is the identification of the text used as query (it can be a query or other document used as query), the second column of the tab separated value file is the retrieved document identification followed by the third column with the score of it, which determines the position on the ranking, we can assume they are alredy ordered from highest prioriest appearing first.

Each document is used as well as query, therefore, in a system with 4 documents to retrieve there will be 12 lines that might be produced, 3 for each of them.
```
PLAIN-2    MED-10   0.688
PLAIN-2    MED-14   0.6859
PLAIN-2    MED-2429 0.6856
MED-10     MED-10   0.99
MED-10     MED-2429 0.22
MED-10     PLAIN-2  0.01
MED-2429    ... ...
MED-2429    ... ...
MED-2429    ... ...
MED-14  ... ...
MED-14  ... ...
MED-14  ... ...
```

### 2. Transform to UDLF Format

#### A. ranked-list.txt (query followed by docs sorted by score)
```
PLAIN-2 MED-10 MED-14 MED-2429
MED-10 PLAIN-2 MED-14 MED-2429
MED-14 ...
MED-2429 ...
```

#### B. lists.txt (all unique document or query IDs)
This is always the first column of ranked-list.txt
```
PLAIN-2
MED-10
MED-14
MED-2429
```

### 3. UDLF Processing
The pyUDLF binary processes the ranked-list and lists files using unsupervised learning methods (CPRR, BFSTREE, RDPAC) and produces a reranked output in ranked-list format.

#### UDLF Output (reranked-list.txt)
```
PLAIN-2 MED-14 MED-10 MED-2429
```
*(Notice MED-14 moved to first position)*

### 4. Transform Back to TREC Format
The reranked output is converted back to TREC format with **linear scores** based on rank position:
- Total documents = N
- Rank 1 → Score = N
- Rank 2 → Score = N-1
- Rank 3 → Score = N-2
- ...

#### Final Output (reranked TREC format)
```
PLAIN-2    MED-14      200    (rank 1, score = 200)
PLAIN-2    MED-10      199    (rank 2, score = 199)
PLAIN-2    MED-2429    198    (rank 3, score = 198)
...
```

## Implementation

### Core Function: `prepare_udlf_input_files()`
Located in: `src/udlf_text_espresso/rerank/udlf_runner.py`

**Purpose**: Converts retrieval rows (TREC format) to UDLF input files

**Returns**: `UDLFInputArtifacts` with:
- `ranked_list_path`: Path to ranked-list.txt
- `lists_path`: Path to lists.txt
- `num_queries`: Count of queries
- `num_docs`: Count of unique documents

### Output Parsing Logic
After UDLF execution, the output is parsed and converted back:

```python
total_docs = len(doc_list)
for rank, did in enumerate(doc_list, start=1):
    rerank_score = float(total_docs - rank + 1)  # Linear scoring
```

## Testing
Comprehensive tests in:
- `tests/udlf_text_espresso/rerank/test_udlf_input_prep.py`
- `tests/udlf_text_espresso/rerank/test_udlf_trec_conversion.py`

Tests verify:
✅ TREC → UDLF format conversion (ranked-list and lists files)
✅ UDLF output → TREC format conversion with linear scores
✅ Multi-query handling
✅ Score sorting and rank assignment
