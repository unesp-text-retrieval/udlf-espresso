#!/usr/bin/env python3
"""
Script to print the actual text content for a specific query's ranking.

This script extracts document text for the query science_8f895bca22 from the
HuffPost News dataset, showing the first 20 results before and after CPRR re-ranking.

Usage:
    python scripts/print_ranking_with_text.py
"""

import json
from pathlib import Path
from textwrap import wrap

# ==============================================================================
# Configuration
# ==============================================================================

# Path to the HuffPost corpus TSV file
CORPUS_PATH = Path("/Users/luisvenezian/Documents/masters/udlf-text-retrieval/collections-non-beir/huffpost-news-category/raw-data/corpus_and_topics.tsv")

# Query ID we're interested in
QUERY_ID = "science_8f895bca22"

# Document IDs extracted from the report for science_8f895bca22
# Category-based Dataset • bge-small-en-v1.5 + CPRR (K=75)

ORIGINAL_RANKING = [
    "wellness_46d37a1b2b",    # #1 - not relevant
    "science_e91a2a7179",     # #2 - relevant
    "wellness_1dc642b561",    # #3 - not relevant
    "science_bab0ec9859",     # #4 - relevant
    "travel_fee3c93785",      # #5 - not relevant
    "wellness_00b9a1ce74",    # #6 - not relevant
    "women_7c7fa0f380",       # #7 - not relevant
    "comedy_6dc22385a5",      # #8 - not relevant
    "comedy_79c7caf6e6",      # #9 - not relevant
    "science_29f5f09a25",     # #10 - relevant
    "politics_5bc6cda3da",    # #11 - not relevant
    "travel_102ac9af32",      # #12 - not relevant
    "science_fd9b9621ab",     # #13 - relevant
    "u-s-news_2dc1ae4221",    # #14 - not relevant
    "wellness_18ef701b1d",    # #15 - not relevant
    "science_461415a202",     # #16 - relevant
    "parents_10bd59b5a0",     # #17 - not relevant
    "style-beauty_a7950f3747",# #18 - not relevant
    "parenting_3509b04d18",   # #19 - not relevant
]

RERANKED_RANKING = [
    "science_f01205bbc2",     # #1 - relevant
    "science_e91a2a7179",     # #2 - relevant
    "science_3e7046bc76",     # #3 - relevant
    "science_d8298425a6",     # #4 - relevant
    "science_8e146c78ae",     # #5 - relevant
    "science_0b272ec928",     # #6 - relevant
    "science_d0a767e1ff",     # #7 - relevant
    "science_01a702267d",     # #8 - relevant
    "science_cfef5a2bdb",     # #9 - relevant
    "science_dd507314e7",     # #10 - relevant
    "science_98196af162",     # #11 - relevant
    "science_29f5f09a25",     # #12 - relevant
    "science_dcb9451969",     # #13 - relevant
    "science_4cfcbf91ea",     # #14 - relevant
    "science_6e749e008d",     # #15 - relevant
    "science_37cfe2d8df",     # #16 - relevant
    "wellness_01b7bb3d6f",    # #17 - not relevant
    "science_736a87dee3",     # #18 - relevant
    "wellness_6488e4c311",    # #19 - not relevant
]

# Relevance (for marking in output)
ORIGINAL_RELEVANCE = [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
RERANKED_RELEVANCE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]

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
        Category name (e.g., 'science', 'wellness')
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
    print(" Dataset: HuffPost News (Category-based)")
    print(" Model: bge-small-en-v1.5 + CPRR (K=75)")
    print(" Query: science_8f895bca22")
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


def print_side_by_side_comparison(
    corpus: dict[str, str],
    query_id: str,
    original_ranking: list[str],
    original_relevance: list[int],
    reranked_ranking: list[str],
    reranked_relevance: list[int],
    max_text_length: int = 80
) -> None:
    """
    Print a compact side-by-side comparison suitable for images.
    """
    print("\n" + "=" * 120)
    print(" COMPACT COMPARISON VIEW (for paper/image)")
    print("=" * 120)
    
    # Query info
    query_text = corpus.get(query_id, "[TEXT NOT FOUND]")
    print(f"\n📍 QUERY [{query_id}]:")
    print(f"   \"{truncate_text(query_text, max_text_length)}\"")
    print()
    
    print("-" * 120)
    print(f"{'ORIGINAL RANKING':<60} | {'RE-RANKED (CPRR K=75)':<60}")
    print("-" * 120)
    
    for i, (orig_id, orig_rel, rerank_id, rerank_rel) in enumerate(
        zip(original_ranking, original_relevance, reranked_ranking, reranked_relevance), 
        start=1
    ):
        # Original side
        orig_text = corpus.get(orig_id, "[NOT FOUND]")
        orig_marker = "✓" if orig_rel else "✗"
        orig_cat = get_category_from_id(orig_id).upper()[:8]
        orig_display = f"{orig_marker} [{orig_cat:<8}] {truncate_text(orig_text, 45)}"
        
        # Reranked side
        rerank_text = corpus.get(rerank_id, "[NOT FOUND]")
        rerank_marker = "✓" if rerank_rel else "✗"
        rerank_cat = get_category_from_id(rerank_id).upper()[:8]
        rerank_display = f"{rerank_marker} [{rerank_cat:<8}] {truncate_text(rerank_text, 45)}"
        
        print(f"#{i:2d} {orig_display:<55} | #{i:2d} {rerank_display:<55}")
    
    print("-" * 120)
    orig_prec = sum(original_relevance) / len(original_relevance)
    rerank_prec = sum(reranked_relevance) / len(reranked_relevance)
    print(f"    Precision@{len(original_ranking)}: {orig_prec:.2%} ({sum(original_relevance)}/{len(original_relevance)} relevant)" + 
          " " * 20 + 
          f"| Precision@{len(reranked_ranking)}: {rerank_prec:.2%} ({sum(reranked_relevance)}/{len(reranked_ranking)} relevant)")
    print()


if __name__ == "__main__":
    main()
    
    # Also print compact comparison
    print("\n\n")
    corpus = load_corpus(CORPUS_PATH)
    print_side_by_side_comparison(
        corpus=corpus,
        query_id=QUERY_ID,
        original_ranking=ORIGINAL_RANKING,
        original_relevance=ORIGINAL_RELEVANCE,
        reranked_ranking=RERANKED_RANKING,
        reranked_relevance=RERANKED_RELEVANCE
    )
