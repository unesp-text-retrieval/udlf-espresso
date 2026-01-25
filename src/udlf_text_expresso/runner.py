from __future__ import annotations
import hydra
from omegaconf import DictConfig, OmegaConf
from .core import paths as pathbuilder
from .core.pipeline import PipelineContext, LinearPipeline, PipelineStep
from .core.storage import LocalStorage, S3Storage, GCSStorage
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseIndexer, load_jsonl_documents, encode_texts_subprocess, get_faiss
from .transform.rerank_transform import to_udlf_format
from .rerank.udlf_runner import UDLFReranker
from .metrics.rerank_eval import evaluate_rows as evaluate_rerank_rows
from .metrics.rerank_eval import load_qrels as load_rerank_qrels
from .metrics.rerank_eval import write_rerank_outputs
from .core.artifacts import Artifact
from typing import List, Dict, Any, Tuple, Set, Optional
import json
from .data.steps import DataCollectionStep
import gzip
from pathlib import Path
import logging
from collections import OrderedDict
import io


def _prepare_rerank_inputs(
    base_out: str | Path,
    dataset_name: str,
    index_name: str,
    rows: List[Dict[str, Any]],
    limit: int | None,
) -> tuple[Path, Path] | None:
    if not rows:
        return None
    base_path = Path(base_out)
    
    # Track source statistics
    source_query_ids: Set[str] = set()
    source_doc_ids: Set[str] = set()
    
    ordered_queries: "OrderedDict[str, List[Tuple[float, str]]]" = OrderedDict()
    for row in rows:
        qid = str(row.get("query_id") or row.get("qid"))
        did = str(row.get("doc_id") or row.get("did"))
        score_val = row.get("score") or row.get("rerank_score")
        try:
            score = float(score_val)
        except Exception:
            score = 0.0
        if qid not in ordered_queries:
            ordered_queries[qid] = []
        ordered_queries[qid].append((score, did))
        source_query_ids.add(qid)
        source_doc_ids.add(did)

    ranked_path = pathbuilder.rerank_ranked_list_path(base_path, dataset_name, index_name)
    ranked_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    doc_pool: Set[str] = set()
    for qid, hits in ordered_queries.items():
        hits_sorted = sorted(hits, key=lambda item: item[0], reverse=True)
        if limit is not None and limit > 0:
            hits_sorted = hits_sorted[:limit]
        doc_ids = [doc_id for _, doc_id in hits_sorted]
        doc_pool.update(doc_ids)
        if doc_ids:
            lines.append(" ".join([qid] + doc_ids))
        else:
            lines.append(qid)
    ranked_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lists_path = pathbuilder.rerank_lists_path(base_path, dataset_name, index_name)
    lists_path.parent.mkdir(parents=True, exist_ok=True)
    seen: Set[str] = set()
    entries: List[str] = []
    for qid in ordered_queries.keys():
        if qid not in seen:
            entries.append(qid)
            seen.add(qid)
    for doc_id in sorted(doc_pool):
        if doc_id not in seen:
            entries.append(doc_id)
            seen.add(doc_id)
    lists_content = "\n".join(entries) + "\n"
    lists_path.write_text(lists_content, encoding="utf-8")
    
    # Calculate final statistics and log
    final_queries = len(ordered_queries)
    final_docs = len(seen)
    overlap_count = len(source_query_ids & source_doc_ids)
    
    logger = logging.getLogger("udlf_text_expresso")
    logger.info(
        "Rerank input prepared: dataset=%s, index=%s | "
        "Source: %d queries, %d docs | "
        "Output: %d queries, %d unique IDs in lists.txt | "
        "ID overlap: %d",
        dataset_name,
        index_name,
        len(source_query_ids),
        len(source_doc_ids),
        final_queries,
        final_docs,
        overlap_count,
    )
    
    if overlap_count > 0:
        logger.warning(
            "Found %d IDs appearing as both queries and documents. "
            "These IDs will create duplicate retrievals in ranked-list.txt if not handled properly.",
            overlap_count,
        )
    
    return ranked_path, lists_path


def _merge_unique_queries(
    primary: List[Dict[str, str]],
    extras: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    seen: Set[str] = set()
    logger = logging.getLogger("udlf_text_expresso")

    for source_label, entries in (("primary", primary), ("extras", extras)):
        for entry in entries:
            qid = str(entry.get("id", "")).strip()
            if not qid:
                continue
            if qid in seen:
                message = (
                    f"Duplicate query identifier '{qid}' encountered while merging {source_label} queries. "
                    "Ensure topic and document query IDs are unique."
                )
                logger.error(message)
                raise ValueError(message)
            text_value = entry.get("text", "")
            merged.append({"id": qid, "text": str(text_value)})
            seen.add(qid)

    return merged

class IndexBM25Step(PipelineStep):
    name = "index_bm25"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        # build if cache doesn't exist or force
        if ctx.force:
            return True
        cp = self.cfg.cache_path
        return not (cp and ctx.storage.exists(cp))

    def run(self, ctx: PipelineContext):
        # load corpus from artifact produced by data_collection
        corpus_art = ctx.artifacts.get("corpus") if "corpus" in ctx.artifacts else None
        if corpus_art is None:
            raise ValueError("Corpus artifact not found; ensure data_collection runs before index_bm25.")
        # Support gzipped corpus
        if str(corpus_art.uri).endswith('.gz'):
            with gzip.open(Path(corpus_art.uri), 'rt', encoding='utf-8') as gf:
                lines = gf.read().splitlines()
        else:
            lines = ctx.storage.read_bytes(corpus_art.uri).decode().splitlines()
        corpus: List[Dict[str, str]] = []
        for line in lines:
            if not line or "\t" not in line:
                continue
            did, text = line.split("\t", 1)
            corpus.append({"id": did, "text": text})
        cache_path = self.cfg.cache_path
        # Build and persist index via retriever constructor
        _ = BM25Retriever(corpus=corpus, k=1, cache_path=cache_path)  # k irrelevant for building
        art = Artifact(name="bm25_index", uri=cache_path, kind="index", config_snapshot=dict(self.cfg), input_artifact_ids={"corpus": corpus_art.compute_hash()})
        return [art]


class IndexDenseStep(PipelineStep):
    name = "index_dense"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        if ctx.force:
            return True
        base_out = getattr(self.cfg, "output_dir", "outputs")
        dataset_name = getattr(self.cfg, "dataset_name", "dataset")
        model_alias = self.cfg.model_alias if hasattr(self.cfg, "model_alias") else self.cfg.model_name.split("/")[-1]
        index_dir = pathbuilder.index_root_dir(base_out, dataset_name, model_alias)
        index_path = index_dir / "faiss.index"
        docids_path = index_dir / "docids.json"
        return not (index_path.exists() and docids_path.exists())

    def run(self, ctx: PipelineContext):
        # Locate transformed documents JSONL
        base_out = self.cfg.output_dir
        dataset_name = self.cfg.dataset_name
        jsonl_path = Path(base_out) / pathbuilder.transformed_documents_jsonl(dataset_name)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Transformed documents JSONL not found: {jsonl_path}")

        ids, texts = load_jsonl_documents(str(jsonl_path))
        batch_size_cfg = getattr(self.cfg, "batch_size", None)
        batch_size = None
        if batch_size_cfg not in (None, ""):
            batch_size = int(batch_size_cfg)
        omp_threads_cfg = getattr(self.cfg, "omp_threads", None)
        omp_threads = int(omp_threads_cfg) if omp_threads_cfg not in (None, "") else None
        indexer = DenseIndexer(
            model_name=self.cfg.model_name,
            normalize=bool(getattr(self.cfg, "normalize", True)),
            device=getattr(self.cfg, "device", None),
            batch_size=batch_size,
            use_subprocess=bool(getattr(self.cfg, "use_subprocess", False)),
            omp_threads=omp_threads,
        )
        embs = indexer.encode_texts(texts)
        index = indexer.build_index(embs)

        # Paths: outputs/<run>/dataset/<dataset>/indexes/<model_alias>/
        model_alias = self.cfg.model_alias if hasattr(self.cfg, "model_alias") else self.cfg.model_name.split("/")[-1]
        index_dir = pathbuilder.index_root_dir(base_out, dataset_name, model_alias)
        index_path = index_dir / "faiss.index"
        docids_path = index_dir / "docids.json"
        indexer.save(index, index_path, docids_path, ids)

        art = Artifact(name="dense_index", uri=str(index_path), kind="index", config_snapshot=dict(self.cfg))
        return [art]


class DenseRetrievalStep(PipelineStep):
    name = "dense_retrieval"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        return True

    def run(self, ctx: PipelineContext):
        base_out = self.cfg.output_dir
        dataset_name = self.cfg.dataset_name
        model_name = self.cfg.model_name
        normalize = bool(getattr(self.cfg, "normalize", True))
        model_alias = self.cfg.model_alias if hasattr(self.cfg, "model_alias") else model_name.split("/")[-1]

        index_dir = pathbuilder.index_root_dir(base_out, dataset_name, model_alias)
        index_path = Path(index_dir) / "faiss.index"
        docids_path = Path(index_dir) / "docids.json"
        if not index_path.exists() or not docids_path.exists():
            raise FileNotFoundError(f"Dense index not found at {index_path} or docids {docids_path}")
        ranks_dir = pathbuilder.ranks_root_dir(base_out, dataset_name, model_alias)
        Path(ranks_dir).mkdir(parents=True, exist_ok=True)

        docids_data = json.loads(Path(docids_path).read_text())
        if isinstance(docids_data, dict) and "doc_ids" in docids_data:
            docids_list = docids_data["doc_ids"]
        elif isinstance(docids_data, list):
            docids_list = docids_data
        else:
            raise ValueError(f"Unexpected docids format in {docids_path}")

        jsonl_path = Path(base_out) / pathbuilder.transformed_documents_jsonl(dataset_name)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Transformed documents JSONL missing for retrieval: {jsonl_path}")
        doc_ids_for_queries, doc_texts_for_queries = load_jsonl_documents(str(jsonl_path))

        topics_lines: List[str] = []
        if "topics" in ctx.artifacts:
            topics_art = ctx.artifacts["topics"]
            try:
                if str(topics_art.uri).endswith('.gz'):
                    with gzip.open(Path(topics_art.uri), 'rt', encoding='utf-8') as gf:
                        topics_lines = gf.read().splitlines()
                else:
                    topics_lines = ctx.storage.read_bytes(topics_art.uri).decode().splitlines()
            except Exception:
                topics_lines = []
        if not topics_lines:
            extracted_topics = pathbuilder.dataset_extracted_dir(base_out, dataset_name) / "topics" / "data.tsv.gz"
            if extracted_topics.exists():
                with gzip.open(extracted_topics, 'rt', encoding='utf-8') as gf:
                    topics_lines = gf.read().splitlines()
        if not topics_lines:
            raise ValueError("Topics artifact not available and fallback file missing; run data_collection first.")
        
        # Build document ID set for overlap detection
        doc_id_set = set(doc_ids_for_queries)
        
        # Parse topics and detect overlaps
        topic_queries: List[Dict[str, str]] = []
        topic_ids: Set[str] = set()
        overlap_ids: Set[str] = set()
        for line in topics_lines:
            if not line or "\t" not in line:
                continue
            qid, text = line.split("\t", 1)
            topic_ids.add(qid)
            # If this topic ID also exists in documents, skip the topic version
            # We'll use the document version instead for corpus similarity
            if qid not in doc_id_set:
                topic_queries.append({"id": qid, "text": text})
            else:
                overlap_ids.add(qid)

        # Use all documents as queries (including overlapping IDs)
        # For overlapping IDs, we prefer document content over topic content
        doc_queries: List[Dict[str, str]] = [
            {"id": doc_id, "text": text}
            for doc_id, text in zip(doc_ids_for_queries, doc_texts_for_queries)
        ]

        queries = _merge_unique_queries(topic_queries, doc_queries)

        logger = logging.getLogger("udlf_text_expresso")
        logger.info(
            "DenseRetrieval: queries prepared (topics_only=%d, documents=%d, overlap_replaced=%d, total=%d)",
            len(topic_queries),
            len(doc_queries),
            len(overlap_ids),
            len(queries),
        )
        if overlap_ids:
            logger.warning(
                "DenseRetrieval: replaced %d overlapping IDs with document version instead of topic version. "
                "Using document content for corpus-to-corpus similarity in UDLF.",
                len(overlap_ids),
            )

        qtexts = [q["text"] for q in queries]
        batch_size_cfg = getattr(self.cfg, "batch_size", None)
        batch_size = int(batch_size_cfg) if batch_size_cfg not in (None, "") else 64
        device = getattr(self.cfg, "device", None)
        omp_threads_cfg = getattr(self.cfg, "omp_threads", None)
        omp_threads = int(omp_threads_cfg) if omp_threads_cfg not in (None, "") else None
        use_subprocess = bool(getattr(self.cfg, "use_subprocess", True))
        if use_subprocess:
            qembs = encode_texts_subprocess(
                model_name,
                qtexts,
                device=device,
                batch_size=batch_size,
            )
        else:
            query_indexer = DenseIndexer(
                model_name=model_name,
                normalize=False,
                device=device,
                batch_size=batch_size,
                use_subprocess=False,
                omp_threads=omp_threads,
            )
            qembs = query_indexer.encode_texts(qtexts)
            query_indexer.cleanup()

        import numpy as np  # type: ignore
        import gc

        emb_path = Path(ranks_dir) / "query_embeddings.npy"
        np.save(emb_path, qembs.astype(np.float32))
        del qembs
        gc.collect()
        qembs = np.load(emb_path)

        qembs = qembs.astype(np.float32)
        faiss_mod = get_faiss()
        if omp_threads is not None and hasattr(faiss_mod, "omp_set_num_threads"):
            try:
                faiss_mod.omp_set_num_threads(int(omp_threads))
            except Exception:
                pass
        logger.info("DenseRetrieval: normalizing %d query embeddings (normalize=%s)", qembs.shape[0], normalize)
        if normalize:
            faiss_mod.normalize_L2(qembs)
        logger.info("DenseRetrieval: loading FAISS index from %s", index_path)
        index = faiss_mod.read_index(str(index_path))
        top_k = getattr(self.cfg, "k", 200)
        logger.info(
            "DenseRetrieval: index dimension=%d, query embedding shape=%s",
            getattr(index, "d", -1),
            qembs.shape,
        )
        logger.info("DenseRetrieval: running search for %d queries k=%d", len(queries), top_k)
        D, I = index.search(qembs, top_k)
        logger.info("DenseRetrieval: search completed; first score=%s", float(D[0][0]) if D.size else None)

        out_path = Path(ranks_dir) / "retrieval.tsv"
        lines_out: List[str] = []
        rows_for_rerank: List[Dict[str, Any]] = []
        for qi, q in enumerate(queries):
            for di, score in zip(I[qi], D[qi]):
                did = docids_list[di] if 0 <= di < len(docids_list) else None
                if did is None:
                    continue
                float_score = float(score)
                lines_out.append(f"{q['id']}\t{did}\t{float_score}")
                rows_for_rerank.append({"query_id": q["id"], "doc_id": did, "score": float_score})
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + "\n")
        logger.info(
            "DenseRetrieval: wrote %d rows to %s",
            len(rows_for_rerank),
            out_path,
        )

        limit_cfg = getattr(self.cfg, "rerank_list_limit", None)
        if limit_cfg in (None, ""):
            list_limit = top_k
        else:
            try:
                list_limit = int(limit_cfg)
            except (TypeError, ValueError):
                list_limit = top_k
        rerank_files = _prepare_rerank_inputs(
            base_out,
            dataset_name,
            model_alias,
            rows_for_rerank,
            list_limit,
        )

        cfg_snapshot = dict(self.cfg)
        cfg_snapshot.setdefault("k", top_k)
        metadata = {
            "dataset_name": str(dataset_name),
            "index_name": str(model_alias),
            "base_output_dir": str(base_out),
        }
        artifacts: List[Artifact] = [
            Artifact(
                name="dense_retrieval",
                uri=str(out_path),
                kind="retrieval",
                config_snapshot=cfg_snapshot,
                metadata=metadata,
            )
        ]
        if rerank_files is not None:
            ranked_path, lists_path = rerank_files
            artifacts.append(
                Artifact(
                    name="dense_rerank_ranked_list",
                    uri=str(ranked_path),
                    kind="rerank_input",
                    config_snapshot={"limit": list_limit, **cfg_snapshot},
                    metadata={"input_type": "ranked_list"},
                )
            )
            artifacts.append(
                Artifact(
                    name="dense_rerank_query_list",
                    uri=str(lists_path),
                    kind="rerank_input",
                    config_snapshot={"limit": list_limit, **cfg_snapshot},
                    metadata={"input_type": "query_list"},
                )
            )
        return artifacts

class RetrievalStep(PipelineStep):
    name = "retrieval"
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        return True

    def run(self, ctx: PipelineContext):
        # prefer pre-built index cache if available
        cache_path = getattr(self.cfg, "cache_path", None)
        corpus: List[Dict[str, str]] = []
        logger = logging.getLogger("udlf_text_expresso")
        if cache_path and ctx.storage.exists(cache_path):
            # need corpus doc ids & texts to feed retrieval if cache was built earlier
            corpus_art = ctx.artifacts.get("corpus") if "corpus" in ctx.artifacts else None
            if corpus_art is None:
                raise ValueError("Corpus artifact required to accompany cached index.")
            if str(corpus_art.uri).endswith('.gz'):
                with gzip.open(Path(corpus_art.uri), 'rt', encoding='utf-8') as gf:
                    raw_lines = gf.read().splitlines()
            else:
                raw_lines = ctx.storage.read_bytes(corpus_art.uri).decode().splitlines()
            for line in raw_lines:
                if not line or "\t" not in line:
                    continue
                did, text = line.split("\t", 1)
                corpus.append({"id": did, "text": text})
            retriever = BM25Retriever(corpus=corpus, k=self.cfg.k, cache_path=cache_path)
        else:
            # fallback: build on the fly (will also persist if cache_path provided)
            if hasattr(self.cfg, "corpus") and len(self.cfg.corpus) > 0:
                corpus = [{"id": d.id, "text": d.text} for d in self.cfg.corpus]
            else:
                corpus_art = ctx.artifacts.get("corpus") if "corpus" in ctx.artifacts else None
                if corpus_art is None:
                    raise ValueError("Corpus not provided and artifact missing.")
                if str(corpus_art.uri).endswith('.gz'):
                    with gzip.open(Path(corpus_art.uri), 'rt', encoding='utf-8') as gf:
                        raw_lines = gf.read().splitlines()
                else:
                    raw_lines = ctx.storage.read_bytes(corpus_art.uri).decode().splitlines()
                for line in raw_lines:
                    if not line or "\t" not in line:
                        continue
                    did, text = line.split("\t", 1)
                    corpus.append({"id": did, "text": text})
            retriever = BM25Retriever(corpus=corpus, k=self.cfg.k, cache_path=cache_path)
        
        # Load queries: if config empty, fallback to topics artifact
        if hasattr(self.cfg, "queries") and len(self.cfg.queries) > 0:
            queries = [{"id": q.id, "text": q.text} for q in self.cfg.queries]
        else:
            topics_art = ctx.artifacts.get("topics") if "topics" in ctx.artifacts else None
            if topics_art is None:
                raise ValueError("Queries not provided and topics artifact missing.")
            if str(topics_art.uri).endswith('.gz'):
                with gzip.open(Path(topics_art.uri), 'rt', encoding='utf-8') as gf:
                    lines = gf.read().splitlines()
            else:
                lines = ctx.storage.read_bytes(topics_art.uri).decode().splitlines()
            queries = []
            for line in lines:
                if not line or "\t" not in line:
                    continue
                qid, text = line.split("\t", 1)
                queries.append({"id": qid, "text": text})
        
        # Build document ID set for overlap detection (consistent with dense retrieval)
        doc_id_set = set(item.get("id") for item in corpus)
        
        # Parse topics and detect overlaps
        topic_queries: List[Dict[str, str]] = []
        topic_ids: Set[str] = set()
        overlap_ids: Set[str] = set()
        for q in queries:
            qid = q["id"]
            topic_ids.add(qid)
            # If this topic ID also exists in documents, skip the topic version
            # We'll use the document version instead for corpus similarity (consistent with dense)
            if qid not in doc_id_set:
                topic_queries.append(q)
            else:
                overlap_ids.add(qid)
        
        # Use all documents as queries (including overlapping IDs)
        # For overlapping IDs, we prefer document content over topic content
        # NO doc:: prefix - this creates consistent query sets with dense retrieval
        doc_queries: List[Dict[str, str]] = []
        for item in corpus:
            doc_id = str(item.get("id"))
            text = str(item.get("text", ""))
            doc_queries.append({"id": doc_id, "text": text})
        
        queries = _merge_unique_queries(topic_queries, doc_queries)
        logger.info(
            "Retrieval: queries prepared (topics_only=%d, documents=%d, overlap_replaced=%d, total=%d)",
            len(topic_queries),
            len(doc_queries),
            len(overlap_ids),
            len(queries),
        )
        if overlap_ids:
            logger.warning(
                "Retrieval: replaced %d overlapping IDs with document version instead of topic version. "
                "Using document content for corpus-to-corpus similarity in UDLF.",
                len(overlap_ids),
            )
        if not queries:
            raise ValueError("No queries found for retrieval.")
        results = retriever.retrieve(queries)
        logger.info("Retrieval: received results for %d queries", len(results))
        rows = to_udlf_format(results)
        # Write TSV: query_id\tdoc_id\tscore
        data_uri = f"{self.cfg.output_dir}/retrieval.tsv"
        storage = ctx.storage
        tsv_lines = []
        for r in rows:
            tsv_lines.append(f"{r['query_id']}\t{r['doc_id']}\t{r['score']}")
        storage.write_bytes(data_uri, ("\n".join(tsv_lines) + "\n").encode())
        logger.info("Retrieval: wrote %d rows to %s", len(rows), data_uri)
        default_limit_val = getattr(self.cfg, "k", None)
        try:
            default_limit = int(default_limit_val) if default_limit_val is not None else None
        except (TypeError, ValueError):
            default_limit = None
        limit_cfg = getattr(self.cfg, "rerank_list_limit", None)
        if limit_cfg in (None, ""):
            list_limit = default_limit
        else:
            try:
                list_limit = int(limit_cfg)
            except (TypeError, ValueError):
                list_limit = default_limit
        base_run_dir = getattr(self.cfg, "base_output_dir", None)
        if not base_run_dir:
            try:
                base_run_dir = str(pathbuilder.run_root_dir(ctx.run_id))
            except Exception:
                out_dir_path = Path(self.cfg.output_dir)
                base_run_dir = str(out_dir_path.parents[4] if len(out_dir_path.parents) >= 5 else out_dir_path.parent)
        dataset_name = getattr(self.cfg, "dataset_name", getattr(self.cfg, "dataset", "dataset"))
        index_name = getattr(self.cfg, "index_name", getattr(self.cfg, "index", "index"))
        # Convert retrieval TSV into rerank input artifacts.
        rerank_files = _prepare_rerank_inputs(
            base_run_dir,
            str(dataset_name),
            str(index_name),
            rows,
            list_limit,
        )
        cfg_snapshot = dict(self.cfg)
        cfg_snapshot["base_output_dir"] = base_run_dir
        metadata = {
            "dataset_name": str(dataset_name),
            "index_name": str(index_name),
            "base_output_dir": str(base_run_dir),
        }
        artifacts: List[Artifact] = [
            Artifact(
                name="retrieval",
                uri=data_uri,
                kind="retrieval",
                config_snapshot=cfg_snapshot,
                metadata=metadata,
            )
        ]
        if rerank_files is not None:
            ranked_path, lists_path = rerank_files
            artifacts.append(
                Artifact(
                    name="retrieval_rerank_ranked_list",
                    uri=str(ranked_path),
                    kind="rerank_input",
                    config_snapshot={"limit": list_limit, **cfg_snapshot},
                    metadata={"input_type": "ranked_list"},
                )
            )
            artifacts.append(
                Artifact(
                    name="retrieval_rerank_query_list",
                    uri=str(lists_path),
                    kind="rerank_input",
                    config_snapshot={"limit": list_limit, **cfg_snapshot},
                    metadata={"input_type": "query_list"},
                )
            )
        return artifacts

class ReRankStep(PipelineStep):
    name = "rerank"
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        return True

    def run(self, ctx: PipelineContext):
        storage = ctx.storage
        logger = logging.getLogger("udlf_text_expresso")

        dataset_name = str(getattr(self.cfg, "dataset_name", getattr(self.cfg, "dataset", "dataset")))
        index_name = str(getattr(self.cfg, "index_name", getattr(self.cfg, "index", "index")))
        method_root_name = str(getattr(self.cfg, "method", getattr(self.cfg, "method_name", "method")))
        base_out = str(getattr(self.cfg, "base_output_dir", self.cfg.get("output_dir", "outputs")))

        # First, try to get pre-generated rerank input files (ranked-list and lists)
        # These are much smaller than retrieval.tsv and should be preferred
        ranked_list_path = pathbuilder.rerank_ranked_list_path(Path(base_out), dataset_name, index_name)
        lists_path = pathbuilder.rerank_lists_path(Path(base_out), dataset_name, index_name)
        
        ranked_list_exists = ranked_list_path.exists()
        lists_exists = lists_path.exists()
        
        # Try to download the input files from GCS if they don't exist locally
        if not ranked_list_exists:
            ranked_list_uri = str(ranked_list_path)
            if storage.exists(ranked_list_uri):
                logger.info(f"Downloading ranked-list file from GCS: {ranked_list_uri}")
                ranked_list_data = storage.read_bytes(ranked_list_uri).decode()
                ranked_list_path.parent.mkdir(parents=True, exist_ok=True)
                ranked_list_path.write_text(ranked_list_data, encoding="utf-8")
                ranked_list_exists = True
                logger.info(f"Successfully downloaded ranked-list file")
        
        if not lists_exists:
            lists_uri = str(lists_path)
            if storage.exists(lists_uri):
                logger.info(f"Downloading lists file from GCS: {lists_uri}")
                lists_data = storage.read_bytes(lists_uri).decode()
                lists_path.parent.mkdir(parents=True, exist_ok=True)
                lists_path.write_text(lists_data, encoding="utf-8")
                lists_exists = True
                logger.info(f"Successfully downloaded lists file")
        
        # Only load retrieval.tsv if we need to create the input files
        rows = []
        retrieval_path = None
        if not ranked_list_exists or not lists_exists:
            logger.warning(
                "Rerank input files not found (ranked_list=%s, lists=%s). "
                "Downloading retrieval.tsv to create temporary files.",
                ranked_list_path,
                lists_path,
            )
            
            # Determine retrieval rows. Prefer artifact produced earlier, otherwise load from disk.
            try:
                ret_art = ctx.get("retrieval")
                retrieval_uri = ret_art.uri
                retrieval_path = Path(retrieval_uri)
                logger.info(f"Downloading retrieval data from: {retrieval_uri}")
                raw = storage.read_bytes(retrieval_uri).decode().strip().split("\n")
                logger.info(f"Successfully downloaded retrieval data ({len(raw)} lines)")
            except KeyError:
                ranks_dir = pathbuilder.ranks_root_dir(base_out, dataset_name, index_name)
                fallback = ranks_dir / "retrieval.tsv"
                retrieval_uri = str(fallback)
                if fallback.exists():
                    retrieval_path = fallback
                    raw = fallback.read_text(encoding="utf-8").strip().split("\n")
                elif storage.exists(retrieval_uri):
                    retrieval_path = fallback
                    logger.info(f"Downloading retrieval data from GCS: {retrieval_uri}")
                    raw = storage.read_bytes(retrieval_uri).decode().strip().split("\n")
                    logger.info(f"Successfully downloaded retrieval data ({len(raw)} lines)")
                else:
                    raise FileNotFoundError(
                        f"retrieval.tsv not found for dataset='{dataset_name}', index='{index_name}' at {fallback}"
                    )

            for line in raw:
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    qid, did, score = parts[0], parts[1], parts[2]
                    try:
                        rows.append({"query_id": qid, "doc_id": did, "score": float(score)})
                    except ValueError:
                        continue

            if not rows:
                raise ValueError("No retrieval rows available for reranking")
            
            # Input files will be created from rows as temporary files
            ranked_list_path = None
            lists_path = None
        else:
            # We have the input files, no need to load rows
            # Set retrieval_path to the ranked_list_path since we used pre-generated files
            retrieval_path = ranked_list_path
            logger.info(
                "Using pre-generated rerank input files (ranked_list=%s, lists=%s). "
                "Skipping retrieval.tsv download.",
                ranked_list_path,
                lists_path,
            )

        # Check if this is a category-based dataset
        metadata_path = pathbuilder.dataset_extracted_dir(base_out, dataset_name) / "qrels" / "category_metadata.json"
        is_category_based = False
        category_counts: Optional[Dict[str, int]] = None
        
        # Try to load metadata file (local or from storage)
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as mf:
                    metadata = json.load(mf)
                    if metadata.get("type") == "category_based":
                        is_category_based = True
                        category_counts = metadata.get("categories", {})
                        logger.info("Detected category-based dataset for reranking: %s", dataset_name)
            except Exception as exc:
                logger.warning("Failed to read category metadata: %s", exc)
        elif storage.exists(str(metadata_path)):
            # Download metadata from storage if not local
            try:
                metadata_bytes = storage.read_bytes(str(metadata_path))
                metadata = json.loads(metadata_bytes.decode("utf-8"))
                if metadata.get("type") == "category_based":
                    is_category_based = True
                    category_counts = metadata.get("categories", {})
                    logger.info("Detected category-based dataset from storage for reranking: %s", dataset_name)
                    # Save locally for future use
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(metadata_path, "w", encoding="utf-8") as mf:
                        json.dump(metadata, mf)
            except Exception as exc:
                logger.warning("Failed to read category metadata from storage: %s", exc)
        
        qrels_path = pathbuilder.dataset_extracted_dir(base_out, dataset_name) / "qrels" / "data.tsv.gz"
        qrels: Dict[str, set] = {}
        
        if not is_category_based:
            if qrels_path.exists():
                qrels = load_rerank_qrels(qrels_path)
            elif storage.exists(str(qrels_path)):
                qrels_bytes = storage.read_bytes(str(qrels_path))
                with gzip.open(io.BytesIO(qrels_bytes), "rt", encoding="utf-8") as gf:
                    qrels_lines = gf.read().strip().split("\n")
                for line in qrels_lines:
                    if not line or "\t" not in line:
                        continue
                    qid, did = line.split("\t", 1)
                    qrels.setdefault(qid, set()).add(did)
            else:
                raise FileNotFoundError(
                    f"Qrels not found locally at {qrels_path} or in configured storage"
                )

        def _parse_limit(value: Any) -> int | None:
            if value in (None, ""):
                return None
            try:
                return int(value)
            except Exception:
                logger.warning("Ignoring non-numeric rerank limit value: %s", value)
                return None

        default_limit = _parse_limit(getattr(self.cfg, "limit", None))
        experiments_cfg = getattr(self.cfg, "experiments", None)
        if not experiments_cfg:
            experiments_cfg = [
                {
                    "name": method_root_name,
                    "method": method_root_name,
                    "config": dict(getattr(self.cfg, "config", {})),
                    "limit": default_limit,
                }
            ]

        cutoffs_cfg: Dict[str, Any] = {}
        if hasattr(self.cfg, "cutoffs") and self.cfg.cutoffs is not None:
            try:
                cutoffs_cfg = OmegaConf.to_container(self.cfg.cutoffs, resolve=True)  # type: ignore[arg-type]
            except Exception:
                cutoffs_cfg = dict(self.cfg.cutoffs)

        def _normalize_cutoffs(value: Any, *, default: int, fallback: Optional[List[int]] = None) -> List[int]:
            raw: List[int] = []
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    try:
                        raw.append(int(item))
                    except Exception:
                        continue
            elif value not in (None, ""):
                try:
                    raw.append(int(value))
                except Exception:
                    raw = []
            if not raw and fallback:
                raw = list(fallback)
            if not raw:
                raw = [default]
            return sorted({k for k in raw if k and k > 0})

        base_k_cfg = getattr(self.cfg, "k", None)
        default_map_cutoffs = _normalize_cutoffs(base_k_cfg, default=1000, fallback=[1000])
        default_recall_cutoffs = list(default_map_cutoffs)
        default_precision_cutoffs = _normalize_cutoffs(None, default=10, fallback=[10])
        default_ndcg_cutoffs = list(default_precision_cutoffs)

        map_cutoffs = _normalize_cutoffs(cutoffs_cfg.get("map"), default=1000, fallback=default_map_cutoffs)
        recall_cutoffs = _normalize_cutoffs(cutoffs_cfg.get("recall"), default=1000, fallback=default_recall_cutoffs)
        precision_cutoffs = _normalize_cutoffs(cutoffs_cfg.get("precision"), default=10, fallback=default_precision_cutoffs)
        ndcg_cutoffs = _normalize_cutoffs(cutoffs_cfg.get("ndcg"), default=10, fallback=default_ndcg_cutoffs)

        artifacts: List[Artifact] = []
        # Build base rerank configuration (resolve and drop experiments)
        if isinstance(self.cfg, DictConfig):
            base_rerank_cfg = OmegaConf.to_container(self.cfg, resolve=True)  # type: ignore[arg-type]
        else:
            base_rerank_cfg = dict(self.cfg)  # type: ignore[type-var]
        if not isinstance(base_rerank_cfg, dict):
            base_rerank_cfg = {}
        base_rerank_cfg.pop("experiments", None)

        if not experiments_cfg:
            logger.warning("No rerank experiments configured; skipping rerank step.")
            return []

        for experiment in experiments_cfg:
            if isinstance(experiment, DictConfig):
                experiment_dict = OmegaConf.to_container(experiment, resolve=True)  # type: ignore[arg-type]
            else:
                experiment_dict = experiment

            if not isinstance(experiment_dict, dict):
                logger.warning("Skipping rerank experiment with unsupported config type: %r", type(experiment_dict))
                continue

            raw_experiment_name = experiment_dict.get("name", method_root_name)
            experiment_name = str(raw_experiment_name) if raw_experiment_name is not None else method_root_name

            raw_method = experiment_dict.get("method", method_root_name)
            method_name = str(raw_method) if raw_method is not None else method_root_name

            limit = _parse_limit(experiment_dict.get("limit", default_limit))

            method_cfg = experiment_dict.get("config", {})
            if isinstance(method_cfg, DictConfig):
                method_cfg = OmegaConf.to_container(method_cfg, resolve=True)  # type: ignore[arg-type]
            if method_cfg is None:
                method_cfg = {}
            if not isinstance(method_cfg, dict):
                logger.warning("Experiment '%s' has non-dict 'config'; skipping", experiment_name)
                continue

            experiment_overrides = {
                k: v for k, v in experiment_dict.items() if k not in {"name", "method", "limit", "config"}
            }

            experiment_config = dict(base_rerank_cfg)
            experiment_config.update(experiment_overrides)
            experiment_config.update(method_cfg)
            experiment_config["method"] = method_name
            experiment_config["name"] = experiment_name
            if limit is not None:
                experiment_config["limit"] = limit

            if not method_name:
                logger.warning("Skipping experiment '%s' because it has no 'method' defined.", experiment_name)
                continue

            # Check if experiment already completed (has data.txt output)
            experiment_output_dir = pathbuilder.rerank_experiment_dir(
                base_out,
                dataset_name,
                index_name,
                method_name,
                experiment_name,
            )
            output_data_file = experiment_output_dir / "data.txt"
            if output_data_file.exists():
                logger.info(
                    "Skipping UDLF rerank experiment '%s' - output already exists at %s",
                    experiment_name,
                    output_data_file,
                )
                continue

            logger.info(
                "Running UDLF rerank experiment '%s' (method='%s', limit=%s)",
                experiment_name,
                method_name,
                limit,
            )

            # The config snapshot is now the fully merged config
            snapshot = experiment_config
            
            # The output path for this specific experiment is now passed in the config
            snapshot["output_path"] = str(experiment_output_dir)

            # Save the config snapshot
            config_path = pathbuilder.rerank_experiment_config_path(
                base_out,
                dataset_name,
                index_name,
                method_name,
                experiment_name,
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)

            reranker = UDLFReranker(method_name=method_name, config=snapshot)
            adjusted_rows = reranker.rerank(rows, ranked_list_path, lists_path)

            horizontal_path = pathbuilder.rerank_experiment_horizontal_path(
                base_out,
                dataset_name,
                index_name,
                method_name,
                experiment_name,
            )
            trec_path = pathbuilder.rerank_experiment_trec_path(
                base_out,
                dataset_name,
                index_name,
                method_name,
                experiment_name,
            )
            ranks_path = pathbuilder.rerank_experiment_ranks_path(
                base_out,
                dataset_name,
                index_name,
                method_name,
                experiment_name,
            )
            limited_rows = write_rerank_outputs(
                adjusted_rows,
                method_label=method_name,
                horizontal_path=horizontal_path,
                trec_path=trec_path,
                ranks_path=ranks_path,
                limit=limit,
            )

            config_path = pathbuilder.rerank_experiment_config_path(
                base_out,
                dataset_name,
                index_name,
                method_name,
                experiment_name,
            )
            config_payload = {
                "method": method_name,
                "experiment": experiment_name,
                "limit": limit,
                "params": snapshot,
                "dataset_name": dataset_name,
                "index_name": index_name,
                "retrieval_path": str(retrieval_path),
                "query_count": len({row["query_id"] for row in limited_rows}),
            }
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

            # Validate evaluation prerequisites before computing metrics
            if is_category_based:
                if not category_counts:
                    logger.error(
                        "Category-based evaluation requested for dataset '%s' but category_counts is empty. "
                        "Ensure category_metadata.json exists and was loaded correctly. Skipping metrics computation.",
                        dataset_name,
                    )
                    continue
            else:
                if not qrels or len(qrels) == 0:
                    logger.error(
                        "Standard evaluation requested for dataset '%s' but qrels is empty. "
                        "Ensure qrels file exists in extracted/qrels/ directory. Skipping metrics computation.",
                        dataset_name,
                    )
                    continue

            metrics = evaluate_rerank_rows(
                limited_rows,
                qrels,
                map_k=map_cutoffs,
                recall_k=recall_cutoffs,
                precision_k=precision_cutoffs,
                ndcg_k=ndcg_cutoffs,
                category_based=is_category_based,
                category_counts=category_counts,
            )

            # Validate computed metrics - check if all values are zero (indicates evaluation failure)
            all_zero = all(value == 0.0 for value in metrics.values())
            if all_zero:
                logger.error(
                    "All metrics are zero for experiment '%s' (dataset='%s', index='%s', method='%s'). "
                    "This likely indicates missing or incorrect qrels/category_metadata. "
                    "Skipping metrics file creation.",
                    experiment_name,
                    dataset_name,
                    index_name,
                    method_name,
                )
                continue

            metrics_path = pathbuilder.rerank_experiment_metrics_path(
                base_out,
                dataset_name,
                index_name,
                method_name,
                experiment_name,
            )
            metrics_payload = {experiment_name: metrics}
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

            artifacts.append(
                Artifact(
                    name=f"rerank:{method_name}:{experiment_name}",
                    uri=str(ranks_path),
                    kind="rerank",
                    config_snapshot={
                        "dataset_name": dataset_name,
                        "index_name": index_name,
                        "method": method_name,
                        "experiment": experiment_name,
                        "limit": limit,
                        "params": snapshot,
                    },
                    metadata={
                        "dataset_name": dataset_name,
                        "index_name": index_name,
                        "base_output_dir": base_out,
                        "method_name": method_name,
                        "experiment_name": experiment_name,
                        "retrieval_path": str(retrieval_path),
                    },
                )
            )
            artifacts.append(
                Artifact(
                    name=f"rerank-metrics:{method_name}:{experiment_name}",
                    uri=str(metrics_path),
                    kind="metrics",
                    config_snapshot={
                        "dataset_name": dataset_name,
                        "index_name": index_name,
                        "method": method_name,
                        "experiment": experiment_name,
                        "cutoffs": {
                            "map": map_cutoffs,
                            "recall": recall_cutoffs,
                            "precision": precision_cutoffs,
                            "ndcg": ndcg_cutoffs,
                        },
                    },
                    metadata={
                        "dataset_name": dataset_name,
                        "index_name": index_name,
                        "base_output_dir": base_out,
                        "method_name": method_name,
                        "experiment_name": experiment_name,
                    },
                )
            )

        return artifacts

class MetricsStep(PipelineStep):
    name = "metrics"
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def should_run(self, ctx: PipelineContext) -> bool:
        return True

    def run(self, ctx: PipelineContext):
        storage = ctx.storage
        logger = logging.getLogger("udlf_text_expresso")

        default_dataset = str(getattr(self.cfg, "dataset_name", getattr(self.cfg, "dataset", "dataset")))
        default_index = str(getattr(self.cfg, "index_name", getattr(self.cfg, "index", "bm25")))
        run_id_val = getattr(self.cfg, "run_id", "")
        default_base = str(getattr(self.cfg, "base_output_dir", None) or (Path("outputs") / run_id_val if run_id_val else Path("outputs")))

        base_k_cfg = getattr(self.cfg, "k", None)
        cutoffs_cfg: Dict[str, Any] = {}
        if hasattr(self.cfg, "cutoffs") and self.cfg.cutoffs is not None:
            try:
                cutoffs_cfg = OmegaConf.to_container(self.cfg.cutoffs, resolve=True)  # type: ignore[arg-type]
            except Exception:
                cutoffs_cfg = dict(self.cfg.cutoffs)

        def _normalize_cutoffs(value: Any, *, default: int, fallback: Optional[List[int]] = None) -> List[int]:
            raw: List[int] = []
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    try:
                        raw.append(int(item))
                    except Exception:
                        continue
            elif value not in (None, ""):
                try:
                    raw.append(int(value))
                except Exception:
                    raw = []
            if not raw and fallback:
                raw = list(fallback)
            if not raw:
                raw = [default]
            return sorted({k for k in raw if k and k > 0})

        default_map_cutoffs = _normalize_cutoffs(base_k_cfg, default=1000, fallback=[1000])
        default_recall_cutoffs = list(default_map_cutoffs)
        default_precision_cutoffs = _normalize_cutoffs(None, default=10, fallback=[10])
        default_ndcg_cutoffs = list(default_precision_cutoffs)

        map_cutoffs = _normalize_cutoffs(cutoffs_cfg.get("map"), default=1000, fallback=default_map_cutoffs)
        recall_cutoffs = _normalize_cutoffs(cutoffs_cfg.get("recall"), default=1000, fallback=default_recall_cutoffs)
        precision_cutoffs = _normalize_cutoffs(cutoffs_cfg.get("precision"), default=10, fallback=default_precision_cutoffs)
        ndcg_cutoffs = _normalize_cutoffs(cutoffs_cfg.get("ndcg"), default=10, fallback=default_ndcg_cutoffs)

        method_name = getattr(self.cfg, "method", "rerank")

        retrieval_targets: Dict[Tuple[str, str, str], tuple[Artifact | None, str]] = {}
        for art in ctx.artifacts.values():
            if getattr(art, "kind", "") != "retrieval":
                continue
            meta = art.metadata or {}
            dataset_name = str(meta.get("dataset_name") or default_dataset)
            index_name = str(meta.get("index_name") or default_index)
            base_out = str(meta.get("base_output_dir") or default_base)
            retrieval_targets[(base_out, dataset_name, index_name)] = (art, art.uri)

        if not retrieval_targets:
            fallback_retrieval = pathbuilder.ranks_root_dir(default_base, default_dataset, default_index) / "retrieval.tsv"
            if fallback_retrieval.exists():
                retrieval_targets[(default_base, default_dataset, default_index)] = (None, str(fallback_retrieval))
            else:
                logger.warning(
                    "metrics skipping: no retrieval artifacts found and fallback file missing at %s",
                    fallback_retrieval,
                )
                return []

        qrels_cache: Dict[Tuple[str, str], tuple[Dict[str, Set[str]], bool, Optional[Dict[str, int]]]] = {}
        produced: List[Artifact] = []

        def load_qrels(dataset_name: str, base_out: str) -> tuple[Dict[str, Set[str]], bool, Optional[Dict[str, int]]]:
            """Load qrels and detect if category-based. Returns (qrels, is_category_based, category_counts)"""
            qrels_lines: List[str] = []
            metadata_path = pathbuilder.dataset_extracted_dir(base_out, dataset_name) / "qrels" / "category_metadata.json"
            
            # Check if this is a category-based dataset
            is_category_based = False
            category_counts: Optional[Dict[str, int]] = None
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as mf:
                        metadata = json.load(mf)
                        if metadata.get("type") == "category_based":
                            is_category_based = True
                            category_counts = metadata.get("categories", {})
                            logger.info("Detected category-based dataset: %s", dataset_name)
                except Exception as exc:
                    logger.warning("Failed to read category metadata: %s", exc)
            
            if "qrels" in ctx.artifacts:
                qrels_art = ctx.artifacts["qrels"]
                try:
                    if str(qrels_art.uri).endswith(".gz"):
                        with gzip.open(Path(qrels_art.uri), "rt", encoding="utf-8") as gf:
                            qrels_lines = gf.read().strip().split("\n")
                    else:
                        qrels_lines = storage.read_bytes(qrels_art.uri).decode().strip().split("\n")
                except Exception as exc:
                    logger.warning("metrics unable to read qrels artifact at %s: %s", qrels_art.uri, exc)
            if not qrels_lines:
                qrels_path = pathbuilder.dataset_extracted_dir(base_out, dataset_name) / "qrels" / "data.tsv.gz"
                if qrels_path.exists():
                    with gzip.open(qrels_path, "rt", encoding="utf-8") as gf:
                        qrels_lines = gf.read().strip().split("\n")
                elif not is_category_based:
                    raise FileNotFoundError(f"Qrels not found as artifact or at {qrels_path}")
            
            qrels: Dict[str, Set[str]] = {}
            if not is_category_based:
                for line in qrels_lines:
                    if not line or "\t" not in line or line.startswith("#"):
                        continue
                    qid, did = line.split("\t", 1)
                    qrels.setdefault(qid, set()).add(did)
            
            return qrels, is_category_based, category_counts

        for (base_out, dataset_name, index_name), (art_obj, retrieval_uri) in sorted(retrieval_targets.items()):
            retrieval_lines: List[str] = []
            try:
                retrieval_bytes = storage.read_bytes(retrieval_uri)
                retrieval_lines = retrieval_bytes.decode().strip().split("\n")
            except Exception:
                if not str(retrieval_uri).startswith("s3://"):
                    local_path = Path(retrieval_uri)
                    if local_path.exists():
                        retrieval_lines = local_path.read_text(encoding="utf-8").strip().split("\n")
            if not retrieval_lines:
                logger.warning(
                    "metrics skipping index '%s' (dataset='%s', run_id='%s'): retrieval file not found at %s",
                    index_name,
                    dataset_name,
                    Path(base_out).name,
                    retrieval_uri,
                )
                continue

            ret_rows: List[Dict[str, Any]] = []
            for line in retrieval_lines:
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    qid, did, score = parts[0], parts[1], parts[2]
                    try:
                        ret_rows.append({"query_id": qid, "doc_id": did, "score": float(score)})
                    except (TypeError, ValueError):
                        continue
            if not ret_rows:
                logger.warning(
                    "metrics skipping index '%s' (dataset='%s', run_id='%s'): no valid retrieval rows parsed",
                    index_name,
                    dataset_name,
                    Path(base_out).name,
                )
                continue

            qrels_key = (base_out, dataset_name)
            if qrels_key not in qrels_cache:
                qrels_result = load_qrels(dataset_name, base_out)
                qrels_cache[qrels_key] = qrels_result
            qrels, is_category_based, category_counts = qrels_cache[qrels_key]

            logger.info(
                "metrics evaluating index '%s' (dataset='%s', run_id='%s'%s)",
                index_name,
                dataset_name,
                Path(base_out).name,
                ", category-based" if is_category_based else "",
            )

            # Validate evaluation prerequisites before computing metrics
            if is_category_based:
                if not category_counts:
                    logger.error(
                        "Category-based evaluation requested for dataset '%s' index '%s' but category_counts is empty. "
                        "Ensure category_metadata.json exists and was loaded correctly. Skipping metrics computation.",
                        dataset_name,
                        index_name,
                    )
                    continue
            else:
                if not qrels or len(qrels) == 0:
                    logger.error(
                        "Standard evaluation requested for dataset '%s' index '%s' but qrels is empty. "
                        "Ensure qrels file exists in extracted/qrels/ directory. Skipping metrics computation.",
                        dataset_name,
                        index_name,
                    )
                    continue

            retrieval_metrics = evaluate_rerank_rows(
                ret_rows,
                qrels,
                map_k=map_cutoffs,
                recall_k=recall_cutoffs,
                precision_k=precision_cutoffs,
                ndcg_k=ndcg_cutoffs,
                category_based=is_category_based,
                category_counts=category_counts,
            )

            # Validate computed metrics - check if all values are zero (indicates evaluation failure)
            all_zero = all(value == 0.0 for value in retrieval_metrics.values())
            if all_zero:
                logger.error(
                    "All metrics are zero for index '%s' (dataset='%s'). "
                    "This likely indicates missing or incorrect qrels/category_metadata. "
                    "Skipping metrics file creation.",
                    index_name,
                    dataset_name,
                )
                continue

            metrics_payload: Dict[str, Dict[str, float]] = {index_name: retrieval_metrics}

            metrics_path = pathbuilder.metrics_file_path(base_out, dataset_name, index_name)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            storage.write_json(str(metrics_path), metrics_payload)

            input_ids: Dict[str, str] = {}
            if art_obj is not None:
                input_ids[index_name] = art_obj.compute_hash()
            snapshot = {
                "dataset_name": dataset_name,
                "index_name": index_name,
                "base_output_dir": base_out,
                "cutoffs": {
                    "map": map_cutoffs,
                    "recall": recall_cutoffs,
                    "precision": precision_cutoffs,
                    "ndcg": ndcg_cutoffs,
                },
            }
            produced.append(
                Artifact(
                    name=f"metrics:{index_name}",
                    uri=str(metrics_path),
                    kind="metrics",
                    config_snapshot=snapshot,
                    input_artifact_ids=input_ids or None,
                )
            )

        return produced

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # storage selection
    if cfg.storage.kind == "local":
        storage = LocalStorage()
    elif cfg.storage.kind == "s3":
        storage = S3Storage(bucket=cfg.storage.bucket, prefix=cfg.storage.prefix)
    elif cfg.storage.kind == "gcs":
        storage = GCSStorage(bucket=cfg.storage.bucket, prefix=cfg.storage.prefix)
    else:
        raise ValueError(f"Unknown storage kind {cfg.storage.kind}")

    ctx = PipelineContext(run_id=cfg.run_id, storage=storage, force=cfg.force)
    # Allow adding derived keys into cfg (disable struct mode)
    OmegaConf.set_struct(cfg, False)
    cfg.run_id = ctx.run_id

    # Compute canonical directories based on config names
    # Support both dataset=name and dataset.name=name
    dataset_cfg = getattr(cfg, "dataset", "dataset")
    if isinstance(dataset_cfg, str):
        dataset_name = dataset_cfg
    else:
        dataset_name = getattr(dataset_cfg, "name", "dataset")
    
    index_name = getattr(cfg, "index").name if hasattr(cfg, "index") else "index"
    method_name = getattr(cfg, "method").name if hasattr(cfg, "method") else "method"
    
    # Handle unified model parameter
    model = getattr(cfg, "model", "bm25")
    if model and model.lower() != "bm25":
        # It's a dense model - set up dense_indexing config if not already present
        model_configs = {
            "miniLM-L6-v2": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "normalize": True,
                "k": 200,
            },
            "contriever-msmarco": {
                "model_name": "facebook/contriever-msmarco",
                "normalize": False,
                "k": 200,
            },
            "bge-small-en-v1.5": {
                "model_name": "BAAI/bge-small-en-v1.5",
                "normalize": True,
                "k": 200,
            },
        }
        
        if model in model_configs:
            model_cfg = model_configs[model]
            if not hasattr(cfg, "dense_indexing"):
                cfg.dense_indexing = OmegaConf.create({
                    "model_name": model_cfg["model_name"],
                    "model_alias": model,
                    "normalize": model_cfg["normalize"],
                    "k": model_cfg["k"],
                    "device": None,
                    "batch_size": 64,
                    "use_subprocess": False,
                    "omp_threads": 1,
                    "rerank_list_limit": 200,
                })
            else:
                cfg.dense_indexing.model_name = model_cfg["model_name"]
                cfg.dense_indexing.model_alias = model
                cfg.dense_indexing.normalize = model_cfg["normalize"]
                cfg.dense_indexing.k = model_cfg["k"]
        
        # Update index_name to match the model
        index_name = model

    # Base output dir
    base_out_path = pathbuilder.run_root_dir(ctx.run_id)
    base_out = str(base_out_path)

    # Detect dataset type and configure data collection
    data_source = cfg.data_collection.source
    
    # Check if this is a category-based dataset (non-BEIR)
    # We identify them by checking if data_collection.source starts with "category_"
    # OR if it doesn't start with "beir_", we try to construct the GCS path
    if not data_source.startswith("beir_"):
        # Non-BEIR dataset - assume category-based
        # Construct GCS path: gs://bucket/outputs/run_id/dataset/{dataset_name}/extracted/documents/corpus-and-topics.tsv
        storage_cfg = cfg.storage
        bucket = storage_cfg.get("bucket", "text-udlf-expresso")
        prefix = storage_cfg.get("prefix", "")
        
        if prefix:
            gcs_corpus_path = f"gs://{bucket}/{prefix}/outputs/{ctx.run_id}/dataset/{dataset_name}/extracted/documents/corpus-and-topics.tsv"
        else:
            gcs_corpus_path = f"gs://{bucket}/outputs/{ctx.run_id}/dataset/{dataset_name}/extracted/documents/corpus-and-topics.tsv"
        
        cfg.data_collection.gcs_corpus_path = gcs_corpus_path
        
        # Update source to use category_ prefix if not already set
        if not data_source.startswith("category_"):
            cfg.data_collection.source = f"category_{dataset_name}"

    # Update step output dirs to canonical schema
    # Pass base output root and dataset name to data collection
    cfg.data_collection.output_dir = str(base_out)
    cfg.data_collection.dataset_name = dataset_name
    cfg.indexing.cache_path = str(pathbuilder.index_bm25_cache_file(pathbuilder.index_root_dir(base_out, dataset_name, index_name)))
    cfg.retrieval.output_dir = str(pathbuilder.ranks_root_dir(base_out, dataset_name, index_name))
    cfg.retrieval.base_output_dir = str(base_out)
    cfg.retrieval.index_name = index_name
    cfg.retrieval.dataset_name = dataset_name
    cfg.rerank.output_dir = str(pathbuilder.rerank_root_dir(base_out, dataset_name, method_name))
    cfg.rerank.base_output_dir = str(base_out)
    cfg.rerank.dataset_name = dataset_name
    cfg.rerank.index_name = index_name
    # Ensure method name is available to rerank step
    cfg.rerank.method = method_name
    cfg.metrics.output_dir = str(pathbuilder.metrics_file_path(base_out, dataset_name, index_name).parent)
    cfg.metrics.dataset_name = dataset_name
    cfg.metrics.index_name = index_name
    cfg.metrics.base_output_dir = str(base_out)
    cfg.metrics.run_id = cfg.run_id
    # If dense indexing present, propagate dataset/base and derive model_alias
    if hasattr(cfg, "dense_indexing"):
        cfg.dense_indexing.output_dir = str(base_out)
        cfg.dense_indexing.dataset_name = dataset_name
        if not hasattr(cfg.dense_indexing, "model_alias"):
            cfg.dense_indexing.model_alias = cfg.dense_indexing.model_name.split("/")[-1]

    # Determine if we're using dense or sparse retrieval
    is_dense = model and model.lower() != "bm25"
    
    steps_map = {
        "data_collection": DataCollectionStep(cfg.data_collection),
        "index_bm25": IndexBM25Step(cfg.indexing),
        "index_dense": IndexDenseStep(cfg.dense_indexing) if hasattr(cfg, "dense_indexing") else None,
        "retrieval": RetrievalStep(cfg.retrieval),
        "dense_retrieval": DenseRetrievalStep(cfg.dense_indexing) if hasattr(cfg, "dense_indexing") else None,
        "rerank": ReRankStep(cfg.rerank),
        "metrics": MetricsStep(cfg.metrics),
        # Unified aliases that work with any model
        "index": IndexDenseStep(cfg.dense_indexing) if is_dense and hasattr(cfg, "dense_indexing") else IndexBM25Step(cfg.indexing),
        "retrieve": DenseRetrievalStep(cfg.dense_indexing) if is_dense and hasattr(cfg, "dense_indexing") else RetrievalStep(cfg.retrieval),
    }
    selected = [steps_map[name] for name in cfg.pipeline.steps if steps_map.get(name) is not None]
    pipeline = LinearPipeline(selected)
    pipeline.run(ctx)

    print("Artifacts produced:")
    for name, art in ctx.artifacts.items():
        print(name, art.uri)

if __name__ == "__main__":  # pragma: no cover
    main()
