#!/usr/bin/env python3
"""
Generate a professional SVG visualization for UDLF re-ranking results.

This script creates a research-quality figure showing the before/after
comparison of document rankings for the WOS dataset.

Dataset: WOS (Category-based)
Model: contriever-msmarco + CPRR (K=75)
Query: relational-databases_6b0fe87fb1

Usage:
    python scripts/generate_rerank_svg_wos.py
"""

import textwrap
from pathlib import Path
from typing import List, Tuple

# ==============================================================================
# Configuration
# ==============================================================================

CORPUS_PATH = Path("/Users/luisvenezian/Documents/masters/udlf-text-retrieval/collections-non-beir/WOS/trec-format/corpus-and-topics.tsv")
OUTPUT_PATH = Path("/Users/luisvenezian/Documents/masters/udlf-text-espresso/outputs/paper-assets/wos_rerank_visualization.svg")

QUERY_ID = "relational-databases_6b0fe87fb1"

# Document IDs from the report
ORIGINAL_RANKING = [
    "system-identification_3df656d6d7",       # #1 - not relevant
    "pid-controller_3f6fafd8c0",              # #2 - not relevant
    "relational-databases_1fa3bf9f1c",        # #3 - relevant
    "system-identification_8e9d158d30",       # #4 - not relevant
    "parallel-computing_95cc78a7d9",          # #5 - not relevant
    "algorithm-design_8eefd8673c",            # #6 - not relevant
    "system-identification_ef34e94931",       # #7 - not relevant
    "attention_047e39341d",                   # #8 - not relevant
    "software-engineering_bc4bef8253",        # #9 - not relevant
    "pid-controller_24b4057164",              # #10 - not relevant
]

RERANKED_RANKING = [
    "relational-databases_b3afac7ae4",        # #1 - relevant
    "relational-databases_02b95de38e",        # #2 - relevant
    "relational-databases_dfd7a4010e",        # #3 - relevant
    "relational-databases_13002b5906",        # #4 - relevant
    "relational-databases_2c40413278",        # #5 - relevant
    "relational-databases_28dffc100a",        # #6 - relevant
    "relational-databases_9dd4e24ab4",        # #7 - relevant
    "relational-databases_c0ec3321d4",        # #8 - relevant
    "relational-databases_1fa3bf9f1c",        # #9 - relevant
    "relational-databases_d10d56f2b9",        # #10 - relevant
]

ORIGINAL_RELEVANCE = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
RERANKED_RELEVANCE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Visual Configuration
NUM_RESULTS_TO_SHOW = 5  # Show first 5 results with text
MAX_TEXT_LENGTH = 120


def load_corpus(corpus_path: Path) -> dict:
    """Load corpus TSV into dictionary."""
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


def get_category(doc_id: str) -> str:
    """Extract category from document ID."""
    return doc_id.rsplit('_', 1)[0].replace('.txt', '')


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to max length."""
    # Remove quotes at start if present
    text = text.strip('"').strip()
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))


def wrap_text_svg(text: str, max_width: int = 45) -> List[str]:
    """Wrap text for SVG display."""
    return textwrap.wrap(text, width=max_width)


def generate_svg(corpus: dict) -> str:
    """Generate the SVG visualization."""
    
    # Colors
    RELEVANT_COLOR = "#c8e6c9"      # Light green
    RELEVANT_BORDER = "#4caf50"     # Green border
    NOT_RELEVANT_COLOR = "#ffcdd2"  # Light red
    NOT_RELEVANT_BORDER = "#e57373" # Red border
    QUERY_COLOR = "#bbdefb"         # Light blue
    QUERY_BORDER = "#2196f3"        # Blue border
    
    # Category colors for WOS dataset
    CATEGORY_COLORS = {
        "relational-databases": "#4caf50",    # Green
        "system-identification": "#9c27b0",   # Purple
        "pid-controller": "#ff9800",          # Orange
        "parallel-computing": "#f44336",      # Red
        "algorithm-design": "#2196f3",        # Blue
        "attention": "#795548",               # Brown
        "software-engineering": "#607d8b",    # Blue Grey
    }
    
    # Short category labels for compact display
    CATEGORY_SHORT = {
        "relational-databases": "REL-DB",
        "system-identification": "SYS-ID",
        "pid-controller": "PID-CTRL",
        "parallel-computing": "PARALLEL",
        "algorithm-design": "ALGO",
        "attention": "ATTENTION",
        "software-engineering": "SW-ENG",
    }
    
    # Get query text
    query_text = corpus.get(QUERY_ID, "[Query not found]")
    query_text_truncated = truncate_text(query_text, 500)
    
    # SVG dimensions
    width = 1200
    height = 1300  # Same height as BBC version
    
    svg_parts = []
    
    # SVG header with improved font sizes
    svg_parts.append(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <style>
      .title {{ font-family: 'Georgia', serif; font-size: 26px; font-weight: bold; fill: #1a1a1a; }}
      .subtitle {{ font-family: 'Arial', sans-serif; font-size: 16px; fill: #666; }}
      .label {{ font-family: 'Arial', sans-serif; font-size: 16px; font-weight: bold; fill: #333; }}
      .category {{ font-family: 'Arial', sans-serif; font-size: 12px; font-weight: bold; fill: white; }}
      .category-small {{ font-family: 'Arial', sans-serif; font-size: 10px; font-weight: bold; fill: white; }}
      .doc-text {{ font-family: 'Georgia', serif; font-size: 14px; fill: #333; }}
      .rank-num {{ font-family: 'Arial', sans-serif; font-size: 14px; font-weight: bold; fill: #555; }}
      .section-title {{ font-family: 'Arial', sans-serif; font-size: 18px; font-weight: bold; fill: #333; }}
      .query-text {{ font-family: 'Georgia', serif; font-size: 15px; fill: #333; font-style: italic; }}
      .metric {{ font-family: 'Arial', sans-serif; font-size: 15px; fill: #333; }}
      .metric-value {{ font-family: 'Arial', sans-serif; font-size: 16px; font-weight: bold; }}
      .legend-text {{ font-family: 'Arial', sans-serif; font-size: 14px; fill: #444; }}
    </style>
  </defs>
  
  <!-- Background -->
  <rect width="{width}" height="{height}" fill="white"/>
''')

    # Title
    svg_parts.append(f'''
  <!-- Title -->
  <text x="{width//2}" y="40" text-anchor="middle" class="title">UDLF Re-ranking Effect: Category-based Retrieval Improvement</text>
  <text x="{width//2}" y="65" text-anchor="middle" class="subtitle">WOS Dataset • contriever-msmarco + CPRR (K=75) • Query: relational-databases_6b0fe87fb1</text>
''')

    # ========================================================================
    # TOP SECTION: Horizontal ranking overview (positions #1-#10)
    # ========================================================================
    
    top_section_y = 95
    box_width = 95
    box_height = 40
    box_spacing = 10
    start_x = 140
    
    # Original ranking row
    svg_parts.append(f'''
  <!-- Top Section: Ranking Overview -->
  <text x="35" y="{top_section_y + 20}" class="section-title">Original</text>
  <text x="35" y="{top_section_y + 36}" class="subtitle">Ranking</text>
''')
    
    for i, (doc_id, rel) in enumerate(zip(ORIGINAL_RANKING, ORIGINAL_RELEVANCE)):
        x = start_x + i * (box_width + box_spacing)
        y = top_section_y
        cat = get_category(doc_id)
        cat_short = CATEGORY_SHORT.get(cat, cat.upper()[:8])
        bg_color = RELEVANT_COLOR if rel else NOT_RELEVANT_COLOR
        border_color = RELEVANT_BORDER if rel else NOT_RELEVANT_BORDER
        cat_color = CATEGORY_COLORS.get(cat, "#757575")
        
        # Calculate category box width based on text length
        cat_box_width = len(cat_short) * 7 + 12
        
        svg_parts.append(f'''
  <rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" fill="{bg_color}" stroke="{border_color}" stroke-width="2" rx="4"/>
  <text x="{x + 8}" y="{y + 16}" class="rank-num">#{i+1}</text>
  <rect x="{x + 8}" y="{y + 22}" width="{cat_box_width}" height="14" fill="{cat_color}" rx="2"/>
  <text x="{x + 8 + cat_box_width//2}" y="{y + 32}" text-anchor="middle" class="category-small">{cat_short}</text>
''')
    
    # Re-ranked row
    rerank_row_y = top_section_y + box_height + 20
    svg_parts.append(f'''
  <text x="35" y="{rerank_row_y + 20}" class="section-title">Re-ranked</text>
  <text x="35" y="{rerank_row_y + 36}" class="subtitle">Results</text>
''')
    
    for i, (doc_id, rel) in enumerate(zip(RERANKED_RANKING, RERANKED_RELEVANCE)):
        x = start_x + i * (box_width + box_spacing)
        y = rerank_row_y
        cat = get_category(doc_id)
        cat_short = CATEGORY_SHORT.get(cat, cat.upper()[:8])
        bg_color = RELEVANT_COLOR if rel else NOT_RELEVANT_COLOR
        border_color = RELEVANT_BORDER if rel else NOT_RELEVANT_BORDER
        cat_color = CATEGORY_COLORS.get(cat, "#757575")
        
        # Calculate category box width based on text length
        cat_box_width = len(cat_short) * 7 + 12
        
        svg_parts.append(f'''
  <rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" fill="{bg_color}" stroke="{border_color}" stroke-width="2" rx="4"/>
  <text x="{x + 8}" y="{y + 16}" class="rank-num">#{i+1}</text>
  <rect x="{x + 8}" y="{y + 22}" width="{cat_box_width}" height="14" fill="{cat_color}" rx="2"/>
  <text x="{x + 8 + cat_box_width//2}" y="{y + 32}" text-anchor="middle" class="category-small">{cat_short}</text>
''')

    # ========================================================================
    # MIDDLE SECTION: Query box
    # ========================================================================
    
    query_section_y = rerank_row_y + box_height + 45
    query_box_width = 900
    query_box_height = 160
    query_box_x = (width - query_box_width) // 2
    query_cat_color = CATEGORY_COLORS.get("relational-databases", "#4caf50")
    
    svg_parts.append(f'''
  <!-- Query Section -->
  <rect x="{query_box_x}" y="{query_section_y}" width="{query_box_width}" height="{query_box_height}" 
        fill="{QUERY_COLOR}" stroke="{QUERY_BORDER}" stroke-width="2" rx="6"/>
  <text x="{query_box_x + 25}" y="{query_section_y + 30}" class="label">QUERY: {escape_xml(QUERY_ID)}</text>
  <rect x="{query_box_x + 25}" y="{query_section_y + 40}" width="160" height="22" fill="{query_cat_color}" rx="3"/>
  <text x="{query_box_x + 105}" y="{query_section_y + 56}" text-anchor="middle" class="category">RELATIONAL-DATABASES</text>
''')
    
    # Wrap query text - wider wrapping to fill the box
    query_lines = wrap_text_svg(query_text_truncated, 100)
    for j, line in enumerate(query_lines[:5]):  # Max 5 lines
        svg_parts.append(f'''  <text x="{query_box_x + 25}" y="{query_section_y + 82 + j * 18}" class="query-text">{escape_xml(line)}</text>
''')

    # ========================================================================
    # BOTTOM SECTION: Side-by-side detailed rankings with text
    # ========================================================================
    
    detail_section_y = query_section_y + query_box_height + 45
    column_width = 550
    left_column_x = 40
    right_column_x = width - column_width - 40
    doc_box_height = 130
    doc_spacing = 18
    
    # Section headers
    svg_parts.append(f'''
  <!-- Detailed Ranking Section -->
  <text x="{left_column_x + column_width//2}" y="{detail_section_y}" text-anchor="middle" class="section-title">Original Retrieval (contriever-msmarco)</text>
  <text x="{left_column_x + column_width//2}" y="{detail_section_y + 18}" text-anchor="middle" class="subtitle">Precision@{NUM_RESULTS_TO_SHOW}: {sum(ORIGINAL_RELEVANCE[:NUM_RESULTS_TO_SHOW])/NUM_RESULTS_TO_SHOW:.0%} ({sum(ORIGINAL_RELEVANCE[:NUM_RESULTS_TO_SHOW])}/{NUM_RESULTS_TO_SHOW} relevant)</text>
  
  <text x="{right_column_x + column_width//2}" y="{detail_section_y}" text-anchor="middle" class="section-title">Re-ranked Results (CPRR K=75)</text>
  <text x="{right_column_x + column_width//2}" y="{detail_section_y + 18}" text-anchor="middle" class="subtitle">Precision@{NUM_RESULTS_TO_SHOW}: {sum(RERANKED_RELEVANCE[:NUM_RESULTS_TO_SHOW])/NUM_RESULTS_TO_SHOW:.0%} ({sum(RERANKED_RELEVANCE[:NUM_RESULTS_TO_SHOW])}/{NUM_RESULTS_TO_SHOW} relevant)</text>
''')
    
    # Draw detailed document boxes
    for i in range(NUM_RESULTS_TO_SHOW):
        y = detail_section_y + 35 + i * (doc_box_height + doc_spacing)
        
        # Left column (Original)
        doc_id = ORIGINAL_RANKING[i]
        rel = ORIGINAL_RELEVANCE[i]
        doc_text = corpus.get(doc_id, "[Document not found]")
        doc_text_trunc = truncate_text(doc_text, 220)
        cat = get_category(doc_id)
        cat_short = CATEGORY_SHORT.get(cat, cat.upper()[:8])
        bg_color = RELEVANT_COLOR if rel else NOT_RELEVANT_COLOR
        border_color = RELEVANT_BORDER if rel else NOT_RELEVANT_BORDER
        cat_color = CATEGORY_COLORS.get(cat, "#757575")
        rel_symbol = "✓" if rel else "✗"
        cat_box_width = len(cat_short) * 8 + 14
        
        svg_parts.append(f'''
  <rect x="{left_column_x}" y="{y}" width="{column_width}" height="{doc_box_height}" 
        fill="{bg_color}" stroke="{border_color}" stroke-width="2" rx="5"/>
  <text x="{left_column_x + 12}" y="{y + 22}" class="label">#{i+1} {rel_symbol}</text>
  <rect x="{left_column_x + 55}" y="{y + 8}" width="{cat_box_width}" height="18" fill="{cat_color}" rx="3"/>
  <text x="{left_column_x + 55 + cat_box_width//2}" y="{y + 21}" text-anchor="middle" class="category">{cat_short}</text>
''')
        
        doc_lines = wrap_text_svg(doc_text_trunc, 60)
        for j, line in enumerate(doc_lines[:5]):
            svg_parts.append(f'''  <text x="{left_column_x + 12}" y="{y + 48 + j * 16}" class="doc-text">{escape_xml(line)}</text>
''')
        
        # Right column (Re-ranked)
        doc_id = RERANKED_RANKING[i]
        rel = RERANKED_RELEVANCE[i]
        doc_text = corpus.get(doc_id, "[Document not found]")
        doc_text_trunc = truncate_text(doc_text, 220)
        cat = get_category(doc_id)
        cat_short = CATEGORY_SHORT.get(cat, cat.upper()[:8])
        bg_color = RELEVANT_COLOR if rel else NOT_RELEVANT_COLOR
        border_color = RELEVANT_BORDER if rel else NOT_RELEVANT_BORDER
        cat_color = CATEGORY_COLORS.get(cat, "#757575")
        rel_symbol = "✓" if rel else "✗"
        cat_box_width = len(cat_short) * 8 + 14
        
        svg_parts.append(f'''
  <rect x="{right_column_x}" y="{y}" width="{column_width}" height="{doc_box_height}" 
        fill="{bg_color}" stroke="{border_color}" stroke-width="2" rx="5"/>
  <text x="{right_column_x + 12}" y="{y + 22}" class="label">#{i+1} {rel_symbol}</text>
  <rect x="{right_column_x + 55}" y="{y + 8}" width="{cat_box_width}" height="18" fill="{cat_color}" rx="3"/>
  <text x="{right_column_x + 55 + cat_box_width//2}" y="{y + 21}" text-anchor="middle" class="category">{cat_short}</text>
''')
        
        doc_lines = wrap_text_svg(doc_text_trunc, 60)
        for j, line in enumerate(doc_lines[:5]):
            svg_parts.append(f'''  <text x="{right_column_x + 12}" y="{y + 48 + j * 16}" class="doc-text">{escape_xml(line)}</text>
''')

    # ========================================================================
    # Legend (positioned at the very bottom with proper spacing)
    # ========================================================================
    
    legend_y = height - 60
    svg_parts.append(f'''
  <!-- Legend -->
  <rect x="50" y="{legend_y}" width="18" height="18" fill="{RELEVANT_COLOR}" stroke="{RELEVANT_BORDER}" stroke-width="2" rx="3"/>
  <text x="75" y="{legend_y + 14}" class="legend-text">Relevant (same category)</text>
  
  <rect x="280" y="{legend_y}" width="18" height="18" fill="{NOT_RELEVANT_COLOR}" stroke="{NOT_RELEVANT_BORDER}" stroke-width="2" rx="3"/>
  <text x="305" y="{legend_y + 14}" class="legend-text">Not Relevant (different category)</text>
  
  <rect x="550" y="{legend_y}" width="18" height="18" fill="{QUERY_COLOR}" stroke="{QUERY_BORDER}" stroke-width="2" rx="3"/>
  <text x="575" y="{legend_y + 14}" class="legend-text">Query Document</text>
  
  <text x="{width - 50}" y="{legend_y + 14}" text-anchor="end" class="metric">Precision@20: <tspan font-weight="bold">15% → 95%</tspan> (+533% improvement)</text>
''')

    # Close SVG
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def main():
    """Main function."""
    print("Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(corpus)} documents")
    
    print("\nGenerating SVG visualization...")
    svg_content = generate_svg(corpus)
    
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Write SVG
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"SVG saved to: {OUTPUT_PATH}")
    
    # Also print text summary
    print("\n" + "=" * 80)
    print(" TEXT SUMMARY FOR REFERENCE")
    print("=" * 80)
    
    query_text = corpus.get(QUERY_ID, "[Not found]")
    print(f"\nQUERY [{QUERY_ID}]:")
    print(f"  {truncate_text(query_text, 200)}")
    
    print("\n--- Original Ranking (contriever-msmarco) ---")
    for i, (doc_id, rel) in enumerate(zip(ORIGINAL_RANKING[:5], ORIGINAL_RELEVANCE[:5])):
        cat = get_category(doc_id)
        marker = "✓" if rel else "✗"
        text = truncate_text(corpus.get(doc_id, "[Not found]"), 100)
        print(f"#{i+1} {marker} [{cat}] {text}")
    
    print("\n--- Re-ranked Results (CPRR K=75) ---")
    for i, (doc_id, rel) in enumerate(zip(RERANKED_RANKING[:5], RERANKED_RELEVANCE[:5])):
        cat = get_category(doc_id)
        marker = "✓" if rel else "✗"
        text = truncate_text(corpus.get(doc_id, "[Not found]"), 100)
        print(f"#{i+1} {marker} [{cat}] {text}")


if __name__ == "__main__":
    main()
