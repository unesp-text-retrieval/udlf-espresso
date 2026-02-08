#!/usr/bin/env python3
"""
Script to print the actual text content for a WOS dataset query ranking.

This script extracts document text for the query construction-management_a853f6cdb7 
from the WOS dataset, showing the first 20 results before and after CPRR re-ranking.

Usage:
    python scripts/print_ranking_with_text_wos.py
"""

import json
from pathlib import Path
from textwrap import wrap

# ==============================================================================
# Configuration
# ==============================================================================

# Path to the WOS corpus TSV file
CORPUS_PATH = Path("/Users/luisvenezian/Documents/masters/udlf-text-retrieval/collections-non-beir/WOS/trec-format/corpus-and-topics.tsv")

# Query ID we're interested in
QUERY_ID = "construction-management_a853f6cdb7"

# Document IDs extracted from the report for construction-management_a853f6cdb7
# Category-based Dataset • contriever-msmarco + CPRR (K=75)

ORIGINAL_RANKING = [
    "water-pollution_b0b5bb1ed0",              # #1 - not relevant
    "green-building_12f7e1cf1f",               # #2 - not relevant
    "state-space-representation_423d248425",  # #3 - not relevant
    "water-pollution_46231c5d67",              # #4 - not relevant
    "hydraulics_dbcf636d0d",                   # #5 - not relevant
    "water-pollution_fd69dcbe6f",              # #6 - not relevant
    "control-engineering_b3a12848d6",          # #7 - not relevant
    "water-pollution_5e56ee8d60",              # #8 - not relevant
    "computer-programming_1f4d7690ec",         # #9 - not relevant
    "rainwater-harvesting_1a616c2a59",         # #10 - not relevant
    "state-space-representation_171a48dfd7",  # #11 - not relevant
    "rainwater-harvesting_72ca17dd74",         # #12 - not relevant
    "water-pollution_e77ad746e4",              # #13 - not relevant
    "enzymology_33bbf9c978",                   # #14 - not relevant
    "water-pollution_3858b24435",              # #15 - not relevant
    "construction-management_eed2253a76",      # #16 - RELEVANT
    "water-pollution_35ab751e95",              # #17 - not relevant
    "remote-sensing_9e4d5665f5",               # #18 - not relevant
    "rainwater-harvesting_09c0d0c039",         # #19 - not relevant
]

RERANKED_RANKING = [
    "construction-management_5473cbe0c0",      # #1 - relevant
    "construction-management_89ac5cf1a1",      # #2 - relevant
    "construction-management_aac4ab4116",      # #3 - relevant
    "construction-management_5db7558628",      # #4 - relevant
    "construction-management_c6161e524c",      # #5 - relevant
    "construction-management_5ba93fa7b7",      # #6 - relevant
    "construction-management_448e7cc972",      # #7 - relevant
    "control-engineering_b3a12848d6",          # #8 - not relevant
    "construction-management_1610c31eba",      # #9 - relevant
    "construction-management_5c21f4abb1",      # #10 - relevant
    "construction-management_3e0536fc6a",      # #11 - relevant
    "construction-management_7b623553fc",      # #12 - relevant
    "construction-management_5e32dabaa4",      # #13 - relevant
    "construction-management_c2a08c41fe",      # #14 - relevant
    "construction-management_bee2226e34",      # #15 - relevant
    "construction-management_ed4a056c14",      # #16 - relevant
    "construction-management_84c50dc228",      # #17 - relevant
    "computer-aided-design_6eb8bfef55",        # #18 - not relevant
    "green-building_096825f52d",               # #19 - not relevant
]

# Relevance (for marking in output)
ORIGINAL_RELEVANCE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
RERANKED_RELEVANCE = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

# Maximum characters to show for each document
MAX_TEXT_LENGTH = 300


def load_corpus(corpus_path: Path) -> dict[str, str]:
    """
    Load the corpus TSV file into a dictionary mapping doc_id -> text.
    
    Args:
        corpus_path: Path to the TSV file (id<TAB>text format)
        
    Returns:
        Dictionary of doc_id -> document text
    """
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                doc_id, text = parts
                corpus[doc_id] = text
    return corpus


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if needed.
    
    Args:
        text: The text to truncate
        max_length: Maximum number of characters
        
    Returns:
        Truncated text with ellipsis if it was shortened
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def get_category_from_id(doc_id: str) -> str:
    """
    Extract the category from a document ID.
    
    Args:
        doc_id: Document ID in format category_hash
        
    Returns:
        Category name (e.g., 'construction-management', 'water-pollution')
    """
    return doc_id.rsplit('_', 1)[0]


def print_ranking_with_text(
    corpus: dict[str, str],
    title: str,
    ranking: list[str],
    relevance: list[int],
    query_id: str
) -> None:
    """
    Print a ranking with document text.
    
    Args:
        corpus: Dictionary mapping doc_id -> text
        title: Title for this ranking section
        ranking: List of document IDs in ranking order
        relevance: List of relevance indicators (1=relevant, 0=not relevant)
        query_id: The query document ID
    """
    print("=" * 100)
    print(f" {title}")
    print("=" * 100)
    print()
    
    # Print query first
    query_text = corpus.get(query_id, "[TEXT NOT FOUND]")
    query_category = get_category_from_id(query_id)
    print(f"[QUERY] {query_id}")
    print(f"        Category: {query_category}")
    print(f"        Text: {truncate_text(query_text)}")
    print()
    print("-" * 100)
    print()
    
    # Print each result
    for i, (doc_id, is_relevant) in enumerate(zip(ranking, relevance), start=1):
        doc_text = corpus.get(doc_id, "[TEXT NOT FOUND]")
        doc_category = get_category_from_id(doc_id)
        relevance_marker = "✓ RELEVANT" if is_relevant else "✗ not relevant"
        
        print(f"[{i:2d}] {relevance_marker}")
        print(f"     ID: {doc_id}")
        print(f"     Category: {doc_category}")
        print(f"     Text: {truncate_text(doc_text)}")
        print()
    
    # Summary
    relevant_count = sum(relevance)
    print("-" * 100)
    print(f"Summary: {relevant_count}/{len(ranking)} relevant documents in top {len(ranking)} (Precision@{len(ranking)} = {relevant_count/len(ranking):.2%})")
    print()


def main():
    """Main function to print rankings with text."""
    
    print("\n" + "=" * 100)
    print(" RANKING TEXT EXTRACTION")
    print(" Dataset: WOS (Category-based)")
    print(" Model: contriever-msmarco + CPRR (K=75)")
    print(" Query: construction-management_a853f6cdb7")
    print("=" * 100 + "\n")
    
    # Load corpus
    print(f"Loading corpus from: {CORPUS_PATH}")
    corpus = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(corpus)} documents\n")
    
    # Check if query exists
    if QUERY_ID not in corpus:
        print(f"WARNING: Query ID '{QUERY_ID}' not found in corpus!")
        return
    
    # Print original ranking
    print_ranking_with_text(
        corpus=corpus,
        title="ORIGINAL RANKING (Before CPRR Re-ranking)",
        ranking=ORIGINAL_RANKING,
        relevance=ORIGINAL_RELEVANCE,
        query_id=QUERY_ID
    )
    
    # Print re-ranked results
    print_ranking_with_text(
        corpus=corpus,
        title="RE-RANKED RESULTS (After CPRR K=75)",
        ranking=RERANKED_RANKING,
        relevance=RERANKED_RELEVANCE,
        query_id=QUERY_ID
    )
    
    # Comparison summary
    print("=" * 100)
    print(" COMPARISON SUMMARY")
    print("=" * 100)
    original_precision = sum(ORIGINAL_RELEVANCE) / len(ORIGINAL_RELEVANCE)
    reranked_precision = sum(RERANKED_RELEVANCE) / len(RERANKED_RELEVANCE)
    improvement = reranked_precision - original_precision
    improvement_pct = (improvement / original_precision) * 100 if original_precision > 0 else 0
    
    print(f"Original Precision@{len(ORIGINAL_RANKING)}:  {original_precision:.2%} ({sum(ORIGINAL_RELEVANCE)}/{len(ORIGINAL_RANKING)} relevant)")
    print(f"Re-ranked Precision@{len(RERANKED_RANKING)}: {reranked_precision:.2%} ({sum(RERANKED_RELEVANCE)}/{len(RERANKED_RANKING)} relevant)")
    print(f"Improvement: +{improvement:.2%} (+{improvement_pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()
