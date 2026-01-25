from __future__ import annotations
from typing import List, Tuple, Any
from pathlib import Path
import json
import numpy as np
from types import ModuleType
import multiprocessing as mp
import gc

_faiss_module: ModuleType | None = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # pragma: no cover


def get_faiss() -> ModuleType:
    global _faiss_module
    if _faiss_module is None:
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover - handled at runtime
            raise RuntimeError("faiss not installed.") from exc
        _faiss_module = faiss
    return _faiss_module


def load_jsonl_documents(jsonl_path: str | Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            did = obj.get("id")
            contents = obj.get("contents")
            if did is None or contents is None:
                continue
            ids.append(str(did))
            texts.append(str(contents))
    return ids, texts


class DenseIndexer:
    """
    Build a FAISS index from text documents using sentence-transformers models.

    - Supports cosine (IndexFlatIP with normalized vectors) or L2 (IndexFlatL2).
    - Persists index and docid mapping.
    """

    def __init__(
        self,
        model_name: str,
        normalize: bool = True,
        index_type: str = "flat",
        device: str | None = None,
        batch_size: int | None = None,
        use_subprocess: bool = False,
        omp_threads: int | None = None,
    ):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed.")
        self.model_name = model_name
        self.normalize = normalize
        self.index_type = index_type
        self.device = device
        self.batch_size = batch_size or 64
        self.use_subprocess = use_subprocess
        self.omp_threads = omp_threads
        self.model: SentenceTransformer | None = None  # type: ignore[name-defined]
        if not self.use_subprocess:
            init_kwargs: dict[str, Any] = {}
            if device:
                init_kwargs["device"] = device
            self.model = SentenceTransformer(model_name, **init_kwargs)
        self.faiss = get_faiss()
        if self.omp_threads is not None and hasattr(self.faiss, "omp_set_num_threads"):
            try:
                self.faiss.omp_set_num_threads(int(self.omp_threads))
            except Exception:
                pass

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        # batched encoding; returns float32 array (n, d)
        if self.use_subprocess:
            embs = encode_texts_subprocess(
                self.model_name,
                texts,
                device=self.device,
                batch_size=self.batch_size,
            )
        else:
            assert self.model is not None
            encode_kwargs: dict[str, Any] = {
                "batch_size": self.batch_size,
                "show_progress_bar": True,
                "convert_to_numpy": True,
                "normalize_embeddings": False,
            }
            if self.device:
                encode_kwargs["device"] = self.device
            embs = self.model.encode(texts, **encode_kwargs)
        embs = embs.astype(np.float32)
        if self.normalize:
            self.faiss.normalize_L2(embs)
        return embs

    def build_index(self, embeddings: np.ndarray) -> Any:
        d = embeddings.shape[1]
        # Always use inner-product search; when embeddings are normalized this
        # recovers cosine similarity and avoids mixing distance conventions.
        index = self.faiss.IndexFlatIP(d)
        index.add(embeddings)
        return index

    def save(self, index: Any, index_path: str | Path, docids_path: str | Path, doc_ids: List[str]) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(docids_path).parent.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(index, str(index_path))
        with open(docids_path, "w", encoding="utf-8") as f:
            json.dump({"doc_ids": doc_ids}, f)

    def cleanup(self) -> None:
        if self.model is not None:
            try:
                del self.model
            except AttributeError:
                pass
            self.model = None
        _clear_torch_cache(self.device)
        gc.collect()


def _encode_texts_worker(
    conn: mp.connection.Connection,  # type: ignore[attr-defined]
    model_name: str,
    texts: List[str],
    device: str | None,
    batch_size: int,
) -> None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model_kwargs: dict[str, Any] = {}
        if device:
            model_kwargs["device"] = device
        model = SentenceTransformer(model_name, **model_kwargs)
        encode_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "show_progress_bar": True,
            "convert_to_numpy": True,
            "normalize_embeddings": False,
        }
        if device:
            encode_kwargs["device"] = device
        embs = model.encode(texts, **encode_kwargs)
        arr = np.asarray(embs, dtype=np.float32)
        conn.send(("ok", arr))
    except Exception as exc:
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def encode_texts_subprocess(
    model_name: str,
    texts: List[str],
    device: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_encode_texts_worker, args=(child_conn, model_name, texts, device, batch_size))
    proc.start()
    status, payload = parent_conn.recv()
    parent_conn.close()
    proc.join()
    if status != "ok":
        raise RuntimeError(f"SentenceTransformer encoding failed: {payload}")
    return payload  # type: ignore[return-value]


def _clear_torch_cache(device: str | None) -> None:
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - torch not available
        return
    if device and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device in {"mps", "auto", None} and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
