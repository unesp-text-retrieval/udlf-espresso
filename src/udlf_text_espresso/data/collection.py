from __future__ import annotations
from typing import Dict, Any, Tuple, List
from pathlib import Path
import json
import shutil
import gzip
from ..core import paths as pathbuilder
from ..core.text import normalize_query_text, DEFAULT_QUERY_CONTENT
try:
    from beir.datasets.data_loader import GenericDataLoader  # type: ignore
    from beir.util import download_and_unzip  # type: ignore
except ImportError:  # pragma: no cover
    GenericDataLoader = None
    download_and_unzip = None

class DatasetCollector:
    """
    Implement dataset collection here.

    Example responsibilities:
    - Download files from URLs or S3
    - Unpack archives (gz, zip)
    - Normalize to canonical formats (topics.tsv, corpus.tsv, qrels.tsv)
    - Return local or remote URIs to produced artifacts
    """

    def __init__(self, source_cfg: Dict[str, Any]):
        self.cfg = source_cfg

    def collect_topics(self) -> str:
        """Collect topics and return URI to topics file (e.g., s3://.../topics/data.tsv.gz)."""
        raise NotImplementedError

    def collect_corpus(self) -> str:
        """Collect corpus and return URI to corpus file (e.g., s3://.../documents/data.tsv.gz)."""
        raise NotImplementedError

    def collect_qrels(self) -> str:
        """Collect qrels and return URI to qrels file (e.g., s3://.../qrels/data.tsv)."""
        raise NotImplementedError


class BEIRDatasetCollector(DatasetCollector):
    """Collector for BEIR datasets (Scifact, Scidocs, etc.) using BEIR utilities."""

    def __init__(self, source_cfg: Dict[str, Any], dataset_id: str):
        super().__init__(source_cfg)
        if GenericDataLoader is None or download_and_unzip is None:
            raise RuntimeError("'beir' is not installed. Please add beir to requirements and install it.")
        if not dataset_id:
            raise ValueError("dataset_id must be provided for BEIRDatasetCollector")
        self.dataset_id = dataset_id

    def _download_dataset(self, target_dir: str) -> str:
        dataset = self.dataset_id
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = download_and_unzip(url, target_dir)
        return data_path

    def collect_all(self, base_out: str, dataset_name: str) -> Tuple[str, str, str]:
        tmp_dir = Path(self.cfg.get("work_dir", ".beir_cache")).resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        data_path = self._download_dataset(str(tmp_dir))
        split = self.cfg.get("split", "test")
        # BEIR GenericDataLoader expects the dataset folder path as positional arg
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

        # Canonical paths
        topics_path = Path(base_out) / pathbuilder.extracted_path(dataset_name, "topics")
        corpus_path = Path(base_out) / pathbuilder.extracted_path(dataset_name, "documents")
        qrels_path = Path(base_out) / pathbuilder.extracted_path(dataset_name, "qrels")
        jsonl_path = Path(base_out) / pathbuilder.transformed_documents_jsonl(dataset_name)
        merged_path = Path(base_out) / pathbuilder.transformed_documents_and_topics_tsv(dataset_name)

        topics_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        qrels_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        merged_entries: List[Tuple[str, str]] = []

        with gzip.open(topics_path, "wt", encoding="utf-8") as f:
            for qid, q in queries.items():
                topic_text = str(q)
                f.write(f"{qid}\t{topic_text}\n")
                merged_entries.append((str(qid), normalize_query_text(topic_text)))

        with gzip.open(corpus_path, "wt", encoding="utf-8") as f, open(jsonl_path, "w", encoding="utf-8") as jf:
            for did, doc in corpus.items():
                title = str(doc.get("title", ""))
                text = str(doc.get("text", ""))
                raw_combined = (title + "\n" + text).strip()
                if not raw_combined:
                    raw_combined = DEFAULT_QUERY_CONTENT
                clean_text = raw_combined.replace("\t", " ").replace("\n", " ").strip()
                f.write(f"{did}\t{clean_text}\n")
                jf.write(json.dumps({"id": did, "contents": clean_text}) + "\n")
                merged_entries.append((str(did), normalize_query_text(clean_text)))

        merged_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merged_path, "w", encoding="utf-8") as mf:
            for mid, mtext in merged_entries:
                mf.write(f"{mid}\t{mtext}\n")
        with gzip.open(qrels_path, "wt", encoding="utf-8") as f:
            for qid, doc_map in qrels.items():
                for did, relevance in doc_map.items():
                    try:
                        rel_val = float(relevance)
                    except (TypeError, ValueError):
                        rel_val = 0.0
                    if rel_val > 0.0:
                        f.write(f"{qid}\t{did}\n")  # match BEIR by skipping non-relevant grades

        if self.cfg.get("cleanup_cache", False):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return str(topics_path), str(corpus_path), str(qrels_path)


class BEIRScifactCollector(BEIRDatasetCollector):  # backward compatibility alias
    def __init__(self, source_cfg: Dict[str, Any]):
        super().__init__(source_cfg, dataset_id="scifact")


class BEIRNFCorpusCollector(BEIRDatasetCollector):
    def __init__(self, source_cfg: Dict[str, Any]):
        super().__init__(source_cfg, dataset_id="nfcorpus")


class BEIRArguanaCollector(BEIRDatasetCollector):
    def __init__(self, source_cfg: Dict[str, Any]):
        super().__init__(source_cfg, dataset_id="arguana")


class CategoryBasedDatasetCollector(DatasetCollector):
    """
    Collector for non-BEIR datasets where:
    - Single corpus file acts as both documents and queries
    - Document IDs have category prefix (e.g., sport_001, business_005)
    - Relevance is category-based: query category must match result category
    - Expected GCS path format: gs://.../dataset/{dataset_name}/extracted/documents/corpus-and-topics.tsv
    """

    def __init__(self, source_cfg: Dict[str, Any], dataset_id: str):
        super().__init__(source_cfg)
        if not dataset_id:
            raise ValueError("dataset_id must be provided for CategoryBasedDatasetCollector")
        self.dataset_id = dataset_id

    def _extract_category(self, doc_id: str) -> str:
        """Extract category from document ID like 'sport_001' -> 'sport'"""
        if "_" in doc_id:
            return doc_id.split("_", 1)[0]
        return "unknown"

    def collect_all(self, base_out: str, dataset_name: str) -> Tuple[str, str, str]:
        """
        For category-based datasets:
        - Read corpus-and-topics.tsv from GCS
        - Create topics.tsv.gz (same as corpus since all docs are queries)
        - Create corpus.tsv.gz (all documents)
        - Create qrels.tsv.gz (category-based relevance: all docs in same category are relevant)
        - Create documents_and_topics.tsv (merged for dense indexing)
        - Create corpus.jsonl (for Pyserini indexing)
        """
        import gzip
        import json
        from ..core.text import normalize_query_text

        # Expected GCS path
        gcs_path = self.cfg.get("gcs_corpus_path")
        if not gcs_path:
            raise ValueError("gcs_corpus_path must be specified in config for category-based datasets")

        # Download from GCS
        tmp_corpus = Path(base_out) / "tmp_corpus.tsv"
        tmp_corpus.parent.mkdir(parents=True, exist_ok=True)
        
        import subprocess
        subprocess.run(["gsutil", "cp", gcs_path, str(tmp_corpus)], check=True)

        # Read corpus
        documents = {}
        with open(tmp_corpus, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    doc_id, text = parts
                    documents[doc_id] = text

        # Build category map
        category_map = {}  # category -> [doc_ids]
        for doc_id in documents.keys():
            category = self._extract_category(doc_id)
            if category not in category_map:
                category_map[category] = []
            category_map[category].append(doc_id)

        # Canonical paths
        topics_path = Path(base_out) / pathbuilder.extracted_path(dataset_name, "topics")
        corpus_path = Path(base_out) / pathbuilder.extracted_path(dataset_name, "documents")
        qrels_path = Path(base_out) / pathbuilder.extracted_path(dataset_name, "qrels")
        jsonl_path = Path(base_out) / pathbuilder.transformed_documents_jsonl(dataset_name)
        merged_path = Path(base_out) / pathbuilder.transformed_documents_and_topics_tsv(dataset_name)

        topics_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        qrels_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.parent.mkdir(parents=True, exist_ok=True)

        merged_entries: List[Tuple[str, str]] = []

        # Write empty topics file (category-based datasets don't have separate topics)
        # All documents will be used as queries via document_queries in retrieval
        with gzip.open(topics_path, "wt", encoding="utf-8") as f:
            f.write("# Category-based dataset - no separate topics file\n")
            f.write("# All documents are used as queries in retrieval step\n")

        # Write corpus and jsonl
        with gzip.open(corpus_path, "wt", encoding="utf-8") as f, open(jsonl_path, "w", encoding="utf-8") as jf:
            for doc_id, text in documents.items():
                clean_text = text.replace("\t", " ").replace("\n", " ").strip()
                f.write(f"{doc_id}\t{clean_text}\n")
                jf.write(json.dumps({"id": doc_id, "contents": clean_text}) + "\n")
                # Add to merged entries for indexing
                merged_entries.append((doc_id, normalize_query_text(text)))

        # Write merged (for dense indexing)
        with open(merged_path, "w", encoding="utf-8") as mf:
            for mid, mtext in merged_entries:
                mf.write(f"{mid}\t{mtext}\n")

        # For category-based datasets, create a metadata file instead of massive qrels
        # This file indicates it's a category-based dataset and stores category counts
        metadata_path = qrels_path.parent / "category_metadata.json"
        metadata = {
            "type": "category_based",
            "categories": {cat: len(docs) for cat, docs in category_map.items()},
            "total_documents": len(documents),
            "extract_category_pattern": "split_by_underscore"  # doc_id.split('_')[0]
        }
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2)

        # Create empty qrels file as placeholder (metrics will use category-based logic)
        with gzip.open(qrels_path, "wt", encoding="utf-8") as f:
            f.write("# Category-based dataset - relevance computed on-the-fly\n")

        # Cleanup tmp file
        tmp_corpus.unlink()

        return str(topics_path), str(corpus_path), str(qrels_path)
