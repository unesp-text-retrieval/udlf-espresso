#!/usr/bin/env python3
"""
Generate HTML re-ranking comparison report - convenient CLI script.

Usage:
    python generate_rerank_report.py
    
    # Or with custom settings:
    python generate_rerank_report.py --datasets scifact bbc-news --models bm25 miniLM-L6-v2
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from udlf_text_espresso.reporting.rerank_comparison_report import main

if __name__ == '__main__':
    main()
